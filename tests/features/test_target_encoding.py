"""Tests for out-of-fold target encoding (leakage contract) and the fitted mapping."""

from __future__ import annotations

import dataclasses
import pickle

import numpy as np
import pandas as pd
import pytest

from pmlab.features.target_encoding import FittedTargetEncoder, TargetEncoder

CVS = ["time", "kfold", "group"]


def make_df(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "cat": rng.choice(np.array(["A", "B", "C", "D", None], dtype=object), size=n),
            "grp": rng.integers(0, 8, size=n),
            "t": np.arange(n) // 2,  # pairs of rows share a timestamp (ties)
            "y": rng.integers(0, 2, size=n).astype(float),
        }
    )


def make_encoder(cv: str, **kw) -> TargetEncoder:
    params = dict(col="cat", target="y", cv=cv, n_folds=4, smoothing=3.0, seed=11)
    if cv == "time":
        params["time_col"] = "t"
    if cv == "group":
        params["group_col"] = "grp"
    params.update(kw)
    return TargetEncoder(**params)  # type: ignore[arg-type]


def same(a: float, b: float) -> bool:
    return bool((np.isnan(a) and np.isnan(b)) or a == b)


def reference_oof(df: pd.DataFrame, enc: TargetEncoder) -> np.ndarray:
    """Slow, independent re-implementation of the contract, one row at a time."""
    folds = enc.assign_folds(df)
    m = enc.smoothing
    out = np.full(len(df), np.nan)
    for i in range(len(df)):
        train = (folds < folds[i]) if enc.cv == "time" else (folds != folds[i])
        sub = df[train & df["y"].notna().to_numpy()]
        if sub.empty:
            continue
        prior = sub["y"].mean()
        cat = df["cat"].iloc[i]
        same_cat = sub["cat"].isna() if pd.isna(cat) else sub["cat"] == cat
        n_c = int(same_cat.sum())
        s_c = float(sub.loc[same_cat, "y"].sum())
        out[i] = prior if n_c == 0 else (s_c + m * prior) / (n_c + m)
    return out


class TestFoldAssignment:
    def test_time_folds_are_monotone_and_ties_share_a_fold(self):
        df = make_df()
        folds = make_encoder("time").assign_folds(df)
        order = np.argsort(df["t"].to_numpy(), kind="stable")
        assert (np.diff(folds[order]) >= 0).all()
        for _, idx in df.groupby("t").indices.items():
            assert len(set(folds[idx])) == 1
        assert set(folds) == {0, 1, 2, 3}

    def test_time_folds_capped_by_distinct_timestamps(self):
        df = pd.DataFrame({"cat": list("ab"), "t": [1, 1], "y": [0.0, 1.0]})
        enc = TargetEncoder("cat", "y", time_col="t", n_folds=5)
        assert set(enc.assign_folds(df)) == {0}

    def test_kfold_is_balanced_and_seeded(self):
        df = make_df(n=41)
        enc = make_encoder("kfold", n_folds=5)
        folds = enc.assign_folds(df)
        counts = np.bincount(folds)
        assert counts.max() - counts.min() <= 1
        assert (folds == make_encoder("kfold", n_folds=5).assign_folds(df)).all()
        other = make_encoder("kfold", n_folds=5, seed=12).assign_folds(df)
        assert (folds != other).any()

    def test_group_folds_keep_a_group_together(self):
        df = make_df()
        folds = make_encoder("group").assign_folds(df)
        for _, idx in df.groupby("grp").indices.items():
            assert len(set(folds[idx])) == 1

    def test_group_nan_is_its_own_group(self):
        df = pd.DataFrame(
            {"cat": list("abcd"), "g": [np.nan, 1.0, np.nan, 2.0], "y": [0.0, 1.0, 1.0, 0.0]}
        )
        folds = TargetEncoder("cat", "y", cv="group", group_col="g", n_folds=2).assign_folds(df)
        assert folds[0] == folds[2]

    def test_empty_frame(self):
        df = make_df().iloc[:0]
        for cv in CVS:
            assert len(make_encoder(cv).assign_folds(df)) == 0


class TestOutOfFoldContract:
    """No row's encoding may see its own target."""

    @pytest.mark.parametrize("cv", CVS)
    @pytest.mark.parametrize("garbage", [1e6, -7.0, np.nan])
    def test_changing_a_rows_own_target_never_changes_its_encoding(self, cv, garbage):
        df = make_df()
        enc = make_encoder(cv)
        base = enc.fit_transform(df)["cat_te"].to_numpy()
        for i in range(len(df)):
            mutated = df.copy()
            mutated.loc[i, "y"] = garbage
            got = enc.fit_transform(mutated)["cat_te"].to_numpy()
            assert same(got[i], base[i]), f"row {i} sees its own target ({cv})"

    @pytest.mark.parametrize("cv", CVS)
    def test_permuting_all_targets_inside_a_fold_never_changes_that_folds_encodings(self, cv):
        df = make_df()
        enc = make_encoder(cv)
        folds = enc.assign_folds(df)
        base = enc.fit_transform(df)["cat_te"].to_numpy()
        rng = np.random.default_rng(5)
        for k in np.unique(folds):
            mutated = df.copy()
            idx = np.flatnonzero(folds == k)
            mutated.loc[idx, "y"] = rng.permutation(df["y"].to_numpy()[idx]) * 5.0 + 1.0
            got = enc.fit_transform(mutated)["cat_te"].to_numpy()
            for i in idx:
                assert same(got[i], base[i])

    @pytest.mark.parametrize("cv", CVS)
    def test_targets_do_influence_other_folds(self, cv):
        # Guards against the OOF tests passing vacuously (an encoder that ignores y).
        df = make_df()
        enc = make_encoder(cv)
        base = enc.fit_transform(df)["cat_te"].to_numpy()
        mutated = df.copy()
        mutated["y"] = 1.0 - mutated["y"]
        flipped = enc.fit_transform(mutated)["cat_te"].to_numpy()
        assert not np.allclose(base, flipped, equal_nan=True)

    @pytest.mark.parametrize("cv", CVS)
    def test_matches_brute_force_reference(self, cv):
        df = make_df()
        enc = make_encoder(cv)
        got = enc.fit_transform(df)["cat_te"].to_numpy()
        np.testing.assert_allclose(got, reference_oof(df, enc), equal_nan=True)

    @pytest.mark.parametrize("cv", CVS)
    def test_nan_targets_are_ignored_in_the_statistics(self, cv):
        df = make_df()
        df.loc[[0, 5, 9, 20], "y"] = np.nan
        enc = make_encoder(cv)
        got = enc.fit_transform(df)["cat_te"].to_numpy()
        np.testing.assert_allclose(got, reference_oof(df, enc), equal_nan=True)

    def test_group_mode_rows_of_one_group_do_not_see_each_other(self):
        df = make_df()
        enc = make_encoder("group")
        base = enc.fit_transform(df)["cat_te"].to_numpy()
        for g, idx in df.groupby("grp").indices.items():
            mutated = df.copy()
            mutated.loc[mutated["grp"] == g, "y"] = 123.0
            got = enc.fit_transform(mutated)["cat_te"].to_numpy()
            for i in idx:
                assert same(got[i], base[i])


class TestTimeRespectingFolds:
    """With cv="time" a row's encoding is a function of strictly earlier targets only."""

    def test_future_and_same_time_targets_never_change_a_rows_encoding(self):
        df = make_df()
        enc = make_encoder("time")
        base = enc.fit_transform(df)["cat_te"].to_numpy()
        for i in range(len(df)):
            mutated = df.copy()
            not_earlier = mutated["t"] >= df["t"].iloc[i]
            mutated.loc[not_earlier, "y"] = 1e6
            got = enc.fit_transform(mutated)["cat_te"].to_numpy()
            assert same(got[i], base[i]), f"row {i} depends on a target at t >= its own"

    def test_earlier_targets_do_matter(self):
        df = make_df()
        enc = make_encoder("time")
        base = enc.fit_transform(df)["cat_te"].to_numpy()
        mutated = df.copy()
        mutated.loc[mutated["t"] < 3, "y"] = 1e6
        got = enc.fit_transform(mutated)["cat_te"].to_numpy()
        later = (df["t"] >= 10).to_numpy()
        assert not np.allclose(base[later], got[later], equal_nan=True)

    def test_first_fold_has_no_history_and_is_nan(self):
        df = make_df()
        enc = make_encoder("time")
        folds = enc.assign_folds(df)
        out = enc.fit_transform(df)["cat_te"].to_numpy()
        assert np.isnan(out[folds == 0]).all()
        assert not np.isnan(out[folds >= 1]).any()

    def test_row_order_of_the_frame_is_irrelevant(self):
        df = make_df()
        enc = make_encoder("time")
        base = enc.fit_transform(df)["cat_te"]
        shuffled = df.sample(frac=1.0, random_state=3)
        got = enc.fit_transform(shuffled)["cat_te"]
        pd.testing.assert_series_equal(got.loc[df.index], base, check_names=False)

    def test_works_with_datetime_time_column(self):
        df = make_df()
        df["t"] = pd.Timestamp("2025-01-01") + pd.to_timedelta(df["t"], unit="D")
        enc = make_encoder("time")
        base = make_encoder("time").fit_transform(make_df())["cat_te"]
        pd.testing.assert_series_equal(enc.fit_transform(df)["cat_te"], base)

    def test_nan_in_time_column_raises(self):
        df = make_df()
        df["t"] = df["t"].astype(float)
        df.loc[3, "t"] = np.nan
        with pytest.raises(ValueError, match="time"):
            make_encoder("time").fit_transform(df)


class TestSmoothingPriorUnseenNan:
    @pytest.fixture
    def train(self):
        return pd.DataFrame(
            {
                "cat": ["A", "A", "A", "A", "B", "B", None, None],
                "y": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            }
        )

    def test_m_estimate_formula(self, train):
        fitted = TargetEncoder("cat", "y", time_col="t", smoothing=2.0).fit(
            train.assign(t=range(len(train)))
        )
        prior = 5 / 8
        assert fitted.prior == pytest.approx(prior)
        assert fitted.mapping["A"] == pytest.approx((3 + 2 * prior) / (4 + 2))
        assert fitted.mapping["B"] == pytest.approx((0 + 2 * prior) / (2 + 2))

    def test_zero_smoothing_is_the_raw_mean(self, train):
        fitted = TargetEncoder("cat", "y", cv="kfold", smoothing=0.0).fit(train)
        assert fitted.mapping["A"] == pytest.approx(0.75)
        assert fitted.mapping["B"] == pytest.approx(0.0)

    def test_huge_smoothing_collapses_to_the_prior(self, train):
        fitted = TargetEncoder("cat", "y", cv="kfold", smoothing=1e9).fit(train)
        assert fitted.mapping["A"] == pytest.approx(fitted.prior, abs=1e-6)
        assert fitted.mapping["B"] == pytest.approx(fitted.prior, abs=1e-6)

    def test_unseen_category_gets_the_prior(self, train):
        fitted = TargetEncoder("cat", "y", cv="kfold").fit(train)
        out = fitted.transform(pd.DataFrame({"cat": ["A", "ZZZ"]}))
        assert out["cat_te"].iloc[1] == pytest.approx(fitted.prior)
        assert out["cat_te"].iloc[0] == pytest.approx(fitted.mapping["A"])

    def test_nan_category_is_its_own_category_when_seen(self, train):
        fitted = TargetEncoder("cat", "y", cv="kfold", smoothing=2.0).fit(train)
        prior = 5 / 8
        assert fitted.nan_value == pytest.approx((2 + 2 * prior) / (2 + 2))
        assert fitted.nan_value != pytest.approx(prior)
        out = fitted.transform(pd.DataFrame({"cat": [None, np.nan, "A"]}))
        assert out["cat_te"].iloc[0] == pytest.approx(fitted.nan_value)
        assert out["cat_te"].iloc[1] == pytest.approx(fitted.nan_value)

    def test_nan_category_unseen_in_training_gets_the_prior(self, train):
        fitted = TargetEncoder("cat", "y", cv="kfold").fit(train.dropna())
        out = fitted.transform(pd.DataFrame({"cat": [np.nan]}))
        assert out["cat_te"].iloc[0] == pytest.approx(fitted.prior)
        assert fitted.nan_value == pytest.approx(fitted.prior)

    def test_oof_category_missing_from_the_training_folds_gets_the_fold_prior(self):
        # One row carries a category nobody else has: its OOF value is the prior of the
        # other folds' targets, never its own target.
        df = make_df()
        df.loc[7, "cat"] = "RARE"
        for y in (0.0, 1.0):
            df.loc[7, "y"] = y
            enc = make_encoder("kfold")
            folds = enc.assign_folds(df)
            others = df.loc[folds != folds[7], "y"].mean()
            assert enc.fit_transform(df)["cat_te"].iloc[7] == pytest.approx(others)

    def test_nan_category_rows_are_encoded_oof(self):
        df = make_df()
        assert df["cat"].isna().any()
        enc = make_encoder("kfold")
        out = enc.fit_transform(df)["cat_te"]
        assert not out.isna().any()
        np.testing.assert_allclose(out.to_numpy(), reference_oof(df, enc))

    def test_categorical_and_integer_categories(self):
        df = make_df()
        as_cat = df.assign(cat=df["cat"].astype("category"))
        as_int = df.assign(cat=(df["grp"] % 3).astype(int))
        for frame in (as_cat, as_int):
            enc = make_encoder("kfold")
            got = enc.fit_transform(frame)["cat_te"].to_numpy()
            assert np.isfinite(got).all()
            fitted = enc.fit(frame)
            assert np.isfinite(fitted.transform(frame)["cat_te"].to_numpy()).all()


class TestFittedMapping:
    def test_fit_equals_full_train_smoothed_stats(self):
        df = make_df()
        m = 3.0
        fitted = make_encoder("kfold", smoothing=m).fit(df)
        prior = df["y"].mean()
        stats = df.dropna(subset=["cat"]).groupby("cat")["y"].agg(["sum", "count"])
        assert fitted.prior == pytest.approx(prior)
        assert set(fitted.mapping) == set(stats.index)
        for cat, row in stats.iterrows():
            assert fitted.mapping[cat] == pytest.approx(
                (row["sum"] + m * prior) / (row["count"] + m)
            )
        nan_rows = df[df["cat"].isna()]
        assert fitted.nan_value == pytest.approx(
            (nan_rows["y"].sum() + m * prior) / (len(nan_rows) + m)
        )

    def test_transform_applies_the_mapping_to_new_rows_without_a_target(self):
        df = make_df()
        fitted = make_encoder("time").fit(df)
        live = pd.DataFrame({"cat": ["A", "B", "ZZZ", None], "other": [1, 2, 3, 4]})
        out = fitted.transform(live)
        assert out["cat_te"].tolist() == pytest.approx(
            [fitted.mapping["A"], fitted.mapping["B"], fitted.prior, fitted.nan_value]
        )
        assert list(out.columns) == ["cat", "other", "cat_te"]

    def test_fitted_transform_is_not_the_oof_path(self):
        df = make_df()
        enc = make_encoder("kfold")
        oof = enc.fit_transform(df)["cat_te"].to_numpy()
        full = enc.fit(df).transform(df)["cat_te"].to_numpy()
        assert not np.allclose(oof, full)

    def test_fitted_encoder_ignores_nan_targets(self):
        df = make_df()
        df.loc[[0, 1, 2], "y"] = np.nan
        fitted = make_encoder("kfold").fit(df)
        assert fitted.prior == pytest.approx(df["y"].mean())

    def test_fit_without_any_usable_target_raises(self):
        df = make_df().assign(y=np.nan)
        with pytest.raises(ValueError, match="target"):
            make_encoder("kfold").fit(df)

    def test_fitted_encoder_is_frozen_and_picklable(self):
        fitted = make_encoder("time").fit(make_df())
        with pytest.raises(dataclasses.FrozenInstanceError):
            fitted.prior = 0.0  # type: ignore[misc]
        clone = pickle.loads(pickle.dumps(fitted))
        assert clone == fitted
        live = pd.DataFrame({"cat": ["A", "ZZZ", None]})
        pd.testing.assert_frame_equal(clone.transform(live), fitted.transform(live))

    def test_encoder_config_is_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            make_encoder("time").smoothing = 1.0  # type: ignore[misc]

    def test_fit_returns_fitted_encoder_type(self):
        assert isinstance(make_encoder("time").fit(make_df()), FittedTargetEncoder)


class TestDeterminism:
    @pytest.mark.parametrize("cv", CVS)
    def test_same_seed_same_output(self, cv):
        df = make_df()
        a = make_encoder(cv).fit_transform(df)
        b = make_encoder(cv).fit_transform(df)
        pd.testing.assert_frame_equal(a, b)

    def test_seed_changes_kfold_and_group_assignments_but_not_time(self):
        df = make_df()
        for cv, should_differ in (("kfold", True), ("group", True), ("time", False)):
            f1 = make_encoder(cv, seed=1).assign_folds(df)
            f2 = make_encoder(cv, seed=2).assign_folds(df)
            assert bool((f1 != f2).any()) is should_differ


class TestValidationAndCopyOnWrite:
    def test_time_cv_requires_time_col(self):
        with pytest.raises(ValueError, match="time_col"):
            TargetEncoder("cat", "y")  # default cv is "time": no silent fallback to KFold

    def test_group_cv_requires_group_col(self):
        with pytest.raises(ValueError, match="group_col"):
            TargetEncoder("cat", "y", cv="group")

    def test_kfold_must_be_requested_explicitly(self):
        enc = TargetEncoder("cat", "y", cv="kfold")
        assert enc.cv == "kfold"

    def test_bad_parameters(self):
        with pytest.raises(ValueError, match="n_folds"):
            TargetEncoder("cat", "y", cv="kfold", n_folds=1)
        with pytest.raises(ValueError, match="smoothing"):
            TargetEncoder("cat", "y", cv="kfold", smoothing=-1.0)
        with pytest.raises(ValueError, match="cv"):
            TargetEncoder("cat", "y", cv="loo")  # type: ignore[arg-type]

    @pytest.mark.parametrize("missing", ["cat", "y", "t", "grp"])
    def test_missing_columns_raise_key_error(self, missing):
        cv = "group" if missing == "grp" else "time"
        df = make_df().drop(columns=[missing])
        enc = make_encoder(cv)
        with pytest.raises(KeyError, match=missing):
            enc.fit_transform(df)
        if missing in ("cat", "y"):
            with pytest.raises(KeyError, match=missing):
                enc.fit(df)

    def test_transform_missing_column_raises(self):
        fitted = make_encoder("time").fit(make_df())
        with pytest.raises(KeyError, match="cat"):
            fitted.transform(pd.DataFrame({"x": [1]}))

    @pytest.mark.parametrize("cv", CVS)
    def test_no_mutation_and_new_column_only(self, cv):
        df = make_df()
        before = df.copy()
        enc = make_encoder(cv, suffix="_enc")
        out = enc.fit_transform(df)
        pd.testing.assert_frame_equal(df, before)
        assert out is not df
        assert list(out.columns) == [*df.columns, "cat_enc"]
        assert out.index.equals(df.index)
        assert out["cat_enc"].dtype == np.dtype("float64")
        live = pd.DataFrame({"cat": ["A"]})
        live_before = live.copy()
        enc.fit(df).transform(live)
        pd.testing.assert_frame_equal(live, live_before)

    def test_duplicate_index_labels_are_handled_positionally(self):
        df = make_df()
        df.index = [0, 1] * (len(df) // 2)
        enc = make_encoder("kfold")
        got = enc.fit_transform(df)["cat_te"].to_numpy()
        ref = enc.fit_transform(df.reset_index(drop=True))["cat_te"].to_numpy()
        np.testing.assert_allclose(got, ref, equal_nan=True)

    def test_empty_frame_fit_transform(self):
        out = make_encoder("kfold").fit_transform(make_df().iloc[:0])
        assert len(out) == 0 and "cat_te" in out.columns


class TestExports:
    NAMES = (
        "add_missing_indicators",
        "add_interactions",
        "TargetEncoder",
        "FittedTargetEncoder",
    )

    def test_exported_from_features_and_top_level(self):
        import pmlab
        import pmlab.features

        for name in self.NAMES:
            assert name in pmlab.features.__all__
            assert name in pmlab.__all__
            assert getattr(pmlab, name) is getattr(pmlab.features, name)
