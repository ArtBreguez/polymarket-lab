"""TDD (RED first): leakage guards.

The real-data dogfood reported AUC 1.000 / Brier 0.000 without a single warning,
because a feature column was derived from the label. These guards make that
class of mistake loud instead of silent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pmlab.data import LeakageError, check_no_leakage


def _frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    price = rng.uniform(0.05, 0.95, n)
    return pd.DataFrame(
        {
            "market_id": [f"m{i}" for i in range(n)],
            "decision_date": ["2025-01-01"] * n,
            "outcome_label": ["Yes"] * n,
            "winning_label": np.where(price > 0.5, "Yes", "No"),
            "market_price": price,
            "feature_price": price,
            "feature_noise": rng.normal(0, 1, n),
        }
    )


class TestLeakageDetection:
    def test_feature_perfectly_predicting_label_raises(self):
        # feature_price → winning_label is deterministic (price>0.5). That is the
        # exact trap from the dogfood; it must be flagged.
        df = _frame()
        with pytest.raises(LeakageError) as exc:
            check_no_leakage(df)
        assert "feature_price" in str(exc.value)

    def test_clean_features_pass(self):
        df = _frame()
        clean = df.drop(columns=["feature_price"])
        # feature_noise is independent of the label → no leakage.
        check_no_leakage(clean)  # must not raise

    def test_returns_report_when_not_raising(self):
        df = _frame().drop(columns=["feature_price"])
        report = check_no_leakage(df, raise_on_leak=False)
        assert "feature_noise" in report.scores
        assert report.leaked == []

    def test_report_lists_leaked_columns(self):
        df = _frame()
        report = check_no_leakage(df, raise_on_leak=False)
        assert "feature_price" in report.leaked

    def test_threshold_is_configurable(self):
        # feature_noise separates the label only weakly. A strict default might
        # theoretically flag borderline cases; a lax threshold never does.
        df = _frame().drop(columns=["feature_price"])
        report = check_no_leakage(df, raise_on_leak=False, max_auc=0.999)
        assert report.leaked == []
        # and the perfect separator is always caught, even at the lax threshold
        leaky = _frame()
        report2 = check_no_leakage(leaky, raise_on_leak=False, max_auc=0.999)
        assert "feature_price" in report2.leaked


class TestLeakageEdgeCases:
    def test_no_feature_columns_is_noop(self):
        df = pd.DataFrame({"winning_label": ["Yes", "No"], "market_price": [0.4, 0.6]})
        check_no_leakage(df)  # nothing to check → passes

    def test_single_class_label_is_noop(self):
        # AUC undefined with one class; must not crash.
        df = _frame()
        df["winning_label"] = "Yes"
        check_no_leakage(df.drop(columns=["feature_price"]))

    def test_missing_label_column_raises_valueerror(self):
        df = pd.DataFrame({"feature_x": [1.0, 2.0]})
        with pytest.raises(ValueError):
            check_no_leakage(df)

    def test_constant_feature_is_skipped(self):
        # a feature with <2 distinct usable values can't be AUC-scored; skip it.
        df = _frame().drop(columns=["feature_price"])
        df["feature_const"] = 1.0
        report = check_no_leakage(df, raise_on_leak=False)
        assert "feature_const" not in report.scores


# ---------------------------------------------------------------------------
# check_price_timing: a price must be observed strictly before the decision (issue #28)
# ---------------------------------------------------------------------------

from pmlab.data import PriceTimingReport, check_price_timing  # noqa: E402


def _timed(n: int = 6, **over: object) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "market_id": [f"m{i}" for i in range(n)],
            "market_price": [0.3] * n,
            "price_ts": ["2025-01-05T07:30:00+00:00"] * n,
            "decision_ts": ["2025-01-05T08:00:00+00:00"] * n,
        }
    )
    for col, value in over.items():
        df[col] = value
    return df


class TestCheckPriceTiming:
    def test_prices_observed_before_the_decision_pass(self):
        report = check_price_timing(_timed())
        assert isinstance(report, PriceTimingReport)
        assert (report.n_rows, report.n_checked, report.late, report.untimed) == (6, 6, 0, 0)

    def test_a_price_after_the_decision_raises(self):
        df = _timed()
        df.loc[2, "price_ts"] = "2025-01-05T09:00:00+00:00"
        with pytest.raises(LeakageError, match=r"1 row.*after.*m2"):
            check_price_timing(df)

    def test_a_price_at_the_decision_instant_is_late(self):
        # Strictly before, as FeatureSnapshotStore.as_of reads it.
        df = _timed(price_ts="2025-01-05T08:00:00+00:00")
        with pytest.raises(LeakageError):
            check_price_timing(df)

    def test_a_price_one_microsecond_early_is_fine(self):
        df = _timed(price_ts="2025-01-05T07:59:59.999999+00:00")
        assert check_price_timing(df).late == 0

    def test_offsets_are_compared_in_utc(self):
        # 09:30+02:00 is 07:30Z (in time); 08:30+02:00 is 06:30Z; 10:30+02:00 is 08:30Z (late).
        ok = _timed(price_ts="2025-01-05T09:30:00+02:00")
        assert check_price_timing(ok).late == 0
        late = _timed(price_ts="2025-01-05T10:30:00+02:00")
        with pytest.raises(LeakageError):
            check_price_timing(late)

    def test_a_row_with_a_cutoff_but_no_price_time_cannot_be_verified(self):
        df = _timed()
        df.loc[1, "price_ts"] = None
        with pytest.raises(LeakageError, match=r"1 row.*no price time"):
            check_price_timing(df)

    def test_an_unparseable_price_time_is_untimed(self):
        with pytest.raises(LeakageError):
            check_price_timing(_timed(price_ts="whenever"))

    def test_the_report_is_returned_when_raising_is_off(self):
        df = _timed()
        df.loc[0, "price_ts"] = "2025-01-05T09:00:00+00:00"
        df.loc[3, "price_ts"] = None
        report = check_price_timing(df, raise_on_leak=False)
        assert (report.late, report.untimed, report.n_checked) == (1, 1, 6)
        assert report.late_index == [0] and report.untimed_index == [3]

    def test_a_panel_without_the_columns_is_not_checked(self):
        df = _timed().drop(columns=["price_ts", "decision_ts"])
        report = check_price_timing(df)
        assert (report.n_rows, report.n_checked, report.late) == (6, 0, 0)

    def test_a_panel_with_only_one_of_the_columns_is_not_checked(self):
        assert check_price_timing(_timed().drop(columns=["decision_ts"])).n_checked == 0
        assert check_price_timing(_timed().drop(columns=["price_ts"])).n_checked == 0

    def test_column_names_are_configurable(self):
        df = _timed().rename(columns={"price_ts": "quote_at", "decision_ts": "cutoff"})
        df.loc[0, "quote_at"] = "2025-01-05T09:00:00+00:00"
        with pytest.raises(LeakageError):
            check_price_timing(df, price_col="quote_at", cutoff_col="cutoff")

    def test_datetime_columns_are_accepted(self):
        df = _timed()
        df["price_ts"] = pd.to_datetime(df["price_ts"], utc=True)
        df["decision_ts"] = pd.to_datetime(df["decision_ts"], utc=True)
        assert check_price_timing(df).late == 0

    def test_an_empty_frame_has_nothing_to_check(self):
        assert check_price_timing(_timed(0)).n_rows == 0
