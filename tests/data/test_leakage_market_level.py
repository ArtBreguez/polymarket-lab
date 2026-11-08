"""check_no_leakage and a market-level copy of the realised value (issue #39).

On a per-bin panel the target is the per-row outcome ``outcome_label == winning_label``. A
feature that is the same for every bin of a market scores exactly 0.5 against it, so a
market-level copy of the realised value (``feature_realised_tmax`` repeated on every bin row)
was invisible to the check (the limit 0.11.2 documented). The guard now also scores such a
feature by how close it sits to each bin: a copy of the realised value is inside the winning
bin, every other bin is further away, an honest forecast is only sometimes closest to it.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from pmlab.data import LeakageError, check_no_leakage
from pmlab.data.panel_builder import build_panel

INF = float("inf")
# (label, lower, upper): both bounds inclusive, open ends are +-inf. The label is what the
# panel carries; the bounds say which bin a realised value lands in (and so the winner).
Bin = tuple[str, float, float]


# ---- bin layouts ("dialects"): how a market's outcome labels are written ------------------


def _singles_with_tails(centre: int) -> list[Bin]:
    bins: list[Bin] = [(f"<={centre - 5}°C", -INF, centre - 5)]
    bins += [(f"{v}°C", v, v) for v in range(centre - 4, centre + 5)]
    return [*bins, (f">={centre + 5}°C", centre + 5, INF)]


def _worded_tails(centre: int) -> list[Bin]:
    bins: list[Bin] = [(f"{centre - 5}°C or below", -INF, centre - 5)]
    bins += [(f"{v}°C", v, v) for v in range(centre - 4, centre + 5)]
    return [*bins, (f"{centre + 5}°C or above", centre + 5, INF)]


def _ranges_of_two(centre: int) -> list[Bin]:
    lo = centre - 5
    bins: list[Bin] = [(f"below {lo}°C", -INF, lo - 1)]
    bins += [(f"{v}-{v + 1}°C", v, v + 1) for v in range(lo, lo + 10, 2)]
    return [*bins, (f"{lo + 10}°C or higher", lo + 10, INF)]


def _mixed_widths(centre: int) -> list[Bin]:
    # a wide range between narrow bins and a tail: the bin's centre is far from its edges
    return [
        (f"<={centre - 3}°C", -INF, centre - 3),
        (f"{centre - 2}°C", centre - 2, centre - 2),
        (f"{centre - 1}°C", centre - 1, centre - 1),
        (f"{centre}-{centre + 7}°C", centre, centre + 7),
        (f"{centre + 8}°C", centre + 8, centre + 8),
        (f">={centre + 9}°C", centre + 9, INF),
    ]


def _below_zero(centre: int) -> list[Bin]:  # a cold climate: "-3°C", "<=-9°C" (see _BASE)
    bins: list[Bin] = [(f"<={centre - 5}°C", -INF, centre - 5)]
    bins += [(f"{v}°C", v, v) for v in range(centre - 4, centre + 5)]
    return [*bins, (f">={centre + 5}°C", centre + 5, INF)]


DIALECTS: dict[str, Callable[[int], list[Bin]]] = {
    "singles_with_tails": _singles_with_tails,
    "worded_tails": _worded_tails,
    "ranges_of_two": _ranges_of_two,
    "mixed_widths": _mixed_widths,
    "below_zero": _below_zero,
}


_BASE = {"below_zero": -3}  # where a dialect's markets sit; the others sit around 20


def _market_level_panel(
    dialect: str = "singles_with_tails",
    n_markets: int = 90,
    sigma: float = 1.5,
    seed: int = 0,
    dates_per_market: int = 1,
) -> pd.DataFrame:
    """A per-bin panel (one row per bin) with an honest forecast and a copy of the realised value.

    ``feature_fcst`` is the realised value plus noise (``sigma``), ``feature_realised`` is the
    realised value itself; both are repeated on every bin row of the market. The realised value
    deviates from where the market's bins sit enough that some markets resolve to a tail bin.
    """
    rng = np.random.default_rng(seed)
    make_bins = DIALECTS[dialect]
    rows: list[dict[str, object]] = []
    for i in range(n_markets):
        centre = _BASE.get(dialect, 20) + int(rng.integers(-5, 6))  # where the bins sit
        real = centre + int(np.rint(rng.normal(0, 3)))  # lands in a tail about one time in eight
        for d in range(dates_per_market):
            fcst = real + float(rng.normal(0, sigma))
            bins = make_bins(centre)
            winner = next(label for label, lo, hi in bins if lo <= real <= hi)
            for label, _lo, _hi in bins:
                rows.append(
                    {
                        "market_id": f"m{i:03d}",
                        "decision_date": f"2025-02-{1 + d:02d}",
                        "outcome_label": label,
                        "winning_label": winner,
                        "market_price": 0.1,
                        "feature_fcst": fcst
                        + 0.2 * d,  # the same for every bin of the market, per date
                        "feature_realised": float(real) + 0.2 * d,  # the copy of the realised value
                        "_realised": float(real),
                    }
                )
    return pd.DataFrame(rows)


def _won(df: pd.DataFrame) -> pd.Series:
    return (df["outcome_label"] == df["winning_label"]).astype(int)


# ---- the copy is caught --------------------------------------------------------------------


@pytest.mark.parametrize("dialect", sorted(DIALECTS))
class TestAMarketLevelCopyIsCaught:
    def test_the_fixture_is_a_per_bin_panel_with_some_tail_winners(self, dialect: str) -> None:
        df = _market_level_panel(dialect)
        assert df.groupby("market_id")["outcome_label"].nunique().gt(1).all()
        assert _won(df).groupby(df["market_id"]).sum().eq(1).all()
        tails = df.loc[_won(df) == 1, "outcome_label"].str.contains(r"<|>|or|below|and", regex=True)
        assert 0 < tails.sum() < tails.size  # tails win sometimes, not always

    def test_a_copy_of_the_realised_value_on_every_bin_row_raises(self, dialect: str) -> None:
        df = _market_level_panel(dialect)
        with pytest.raises(LeakageError, match="feature_realised"):
            check_no_leakage(df)

    def test_the_copy_scores_one_and_the_honest_forecast_does_not(self, dialect: str) -> None:
        report = check_no_leakage(_market_level_panel(dialect), raise_on_leak=False)
        assert report.scores["feature_realised"] > 0.99  # "below 15" and "15-16" share the 15
        assert report.leaked == ["feature_realised"]
        assert (
            0.5 < report.scores["feature_fcst"] < 0.97
        )  # near a wide bin, a good forecast is close

    def test_the_honest_forecast_is_not_flagged(self, dialect: str) -> None:
        df = _market_level_panel(dialect).drop(columns=["feature_realised"])
        assert check_no_leakage(df).leaked == []

    def test_build_panel_refuses_the_copy(self, dialect: str) -> None:
        df = _market_level_panel(dialect)
        markets = [g.to_dict("records") for _, g in df.groupby("market_id", sort=False)]
        with pytest.raises(LeakageError, match="feature_realised"):
            build_panel(markets, check_leakage=True)  # type: ignore[arg-type]


class TestTheReport:
    def test_it_lists_the_features_scored_bin_relatively(self) -> None:
        report = check_no_leakage(_market_level_panel(), raise_on_leak=False)
        assert sorted(report.bin_relative) == ["feature_fcst", "feature_realised"]

    def test_the_error_names_the_feature_and_how_it_was_scored(self) -> None:
        with pytest.raises(LeakageError) as err:
            check_no_leakage(_market_level_panel())
        message = str(err.value)
        assert "feature_realised (score=1.000, by its distance to each bin)" in message
        assert "feature_fcst" not in message

    def test_the_target_is_still_the_per_row_outcome(self) -> None:
        report = check_no_leakage(_market_level_panel(), raise_on_leak=False)
        assert report.target == "outcome_label == winning_label"

    def test_a_feature_that_varies_across_bins_is_scored_as_before_not_bin_relatively(self) -> None:
        df = _market_level_panel()
        df["feature_bin_dist"] = np.arange(len(df)) % 7
        report = check_no_leakage(df, raise_on_leak=False)
        assert "feature_bin_dist" in report.scores
        assert "feature_bin_dist" not in report.bin_relative

    def test_a_feature_that_differs_in_a_single_bin_is_not_market_level(self) -> None:
        df = _market_level_panel()
        df["feature_almost"] = df["feature_realised"]
        df.loc[df.index[0], "feature_almost"] += 1.0  # one bin of one market disagrees
        report = check_no_leakage(df, raise_on_leak=False)
        assert "feature_almost" not in report.bin_relative
        assert report.scores["feature_almost"] < 0.97  # not caught: it is not market-level


# ---- the transform is more than "is it the realised value" -----------------------------------


class TestWhatTheTransformDoesNotMistake:
    def test_a_copy_with_noise_a_third_of_a_bin_wide_is_still_caught(self) -> None:
        df = _market_level_panel("singles_with_tails", n_markets=150)
        rng = np.random.default_rng(3)
        noise = pd.Series(
            rng.normal(0, 0.25, df["market_id"].nunique()), index=df["market_id"].unique()
        )
        df["feature_realised"] = df["_realised"] + df["market_id"].map(noise)
        assert check_no_leakage(df, raise_on_leak=False).leaked == ["feature_realised"]

    def test_a_forecast_off_by_a_degree_or_more_is_an_honest_forecast(self) -> None:
        df = _market_level_panel("singles_with_tails", n_markets=150, sigma=1.0)
        report = check_no_leakage(df.drop(columns=["feature_realised"]))
        assert 0.5 < report.scores["feature_fcst"] < 0.97

    def test_features_in_other_units_are_not_mistaken_for_leaks(self) -> None:
        df = _market_level_panel()
        rng = np.random.default_rng(5)
        per_market = df["market_id"].map(
            {m: rng.uniform(0, 1) for m in df["market_id"].unique()}  # type: ignore[arg-type]
        )
        df["feature_humidity"] = 30 + 60 * per_market  # a percentage, far from the bins
        df["feature_liquidity"] = 1e4 + 5e3 * per_market  # four orders of magnitude away
        df["feature_neg"] = -50 - 20 * per_market
        report = check_no_leakage(df.drop(columns=["feature_realised"]), raise_on_leak=False)
        assert report.leaked == []
        for name in ("feature_humidity", "feature_liquidity", "feature_neg"):
            assert report.scores[name] < 0.7

    def test_market_level_features_with_missing_values_are_scored_on_the_rest(self) -> None:
        df = _market_level_panel("singles_with_tails", n_markets=120)
        holes = df["market_id"].isin(df["market_id"].unique()[::4])
        df["feature_realised"] = df["feature_realised"].mask(holes)
        report = check_no_leakage(df, raise_on_leak=False)
        assert report.leaked == ["feature_realised"]

    def test_a_market_seen_on_several_decision_dates_is_market_level_per_date(self) -> None:
        # the value changes between a market's dates (a forecast refreshed) but is the same
        # for all bins at one date: that is what the backtest chooses within
        df = _market_level_panel("singles_with_tails", dates_per_market=2)
        assert df.groupby("market_id")["feature_realised"].nunique().eq(2).all()  # differs by date
        report = check_no_leakage(df, raise_on_leak=False)
        assert "feature_realised" in report.bin_relative
        assert report.leaked == ["feature_realised"]

    def test_labels_that_are_not_numbers_leave_the_check_as_it_was(self) -> None:
        df = _market_level_panel()
        alias = {label: f"bin-{k}" for k, label in enumerate(df["outcome_label"].unique())}
        df["outcome_label"] = df["outcome_label"].map(alias)
        df["winning_label"] = df["winning_label"].map(alias)
        report = check_no_leakage(df, raise_on_leak=False)
        assert report.target == "outcome_label == winning_label"
        assert report.bin_relative == []
        assert report.scores["feature_realised"] == pytest.approx(0.5)  # nothing to measure against

    def test_a_few_unreadable_labels_are_left_out_not_fatal(self) -> None:
        df = _market_level_panel()
        df.loc[df.index[:: len(df) // 20], "outcome_label"] = "n/a"
        report = check_no_leakage(df, raise_on_leak=False)
        assert report.leaked == ["feature_realised"]


# ---- no false positive on honest, bundled data ------------------------------------------------


@pytest.fixture(scope="module")
def sample_rows() -> list[list[dict[str, object]]]:
    from pmlab.datasets import load_sample

    sample = load_sample()
    resolver = sample.plugin()
    return [
        resolver.build_training_rows(
            spec, "previous_evening", decision_time=f"{day}T12:00:00+00:00"
        )
        for day in sample.days[:60]
        for spec in sample.plugin(day).discover_markets()
    ]


class TestTheBundledSampleStillPasses:
    def test_check_no_leakage_accepts_the_honest_sample(self, sample_rows) -> None:
        panel = build_panel(sample_rows)
        report = check_no_leakage(panel)
        assert report.leaked == []
        assert report.target == "outcome_label == winning_label"

    def test_build_panel_with_check_leakage_accepts_the_honest_sample(self, sample_rows) -> None:
        assert len(build_panel(sample_rows, check_leakage=True)) == 60 * 5 * 9

    def test_every_forecast_feature_is_scored_bin_relatively_and_stays_below_the_threshold(
        self, sample_rows
    ) -> None:
        report = check_no_leakage(build_panel(sample_rows))
        features = [c for c in report.scores]
        assert len(features) == 8 and sorted(report.bin_relative) == sorted(features)
        assert max(report.scores.values()) < 0.9  # fcst_tmax, the best forecast, is ~0.7

    def test_a_market_level_copy_of_the_realised_value_is_caught_on_the_sample(
        self, sample_rows
    ) -> None:
        panel = build_panel(sample_rows)
        realised = panel["winning_label"].str.extract(r"(-?\d+)", expand=False).astype(float)
        with pytest.raises(LeakageError, match="feature_realised_tmax"):
            check_no_leakage(panel.assign(feature_realised_tmax=realised))


# ---- other shapes are untouched ----------------------------------------------------------------


def _one_row_per_market(n: int = 90, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    real = rng.integers(10, 30, n)
    return pd.DataFrame(
        {
            "market_id": [f"m{i}" for i in range(n)],
            "decision_date": ["2025-01-01"] * n,
            "outcome_label": [f"{r}°C" for r in real],
            "winning_label": [f"{r}°C" if i % 3 else "heat" for i, r in enumerate(real)],
            "market_price": 0.5,
            "feature_fcst": real + rng.normal(0, 1.5, n),
        }
    )


class TestOtherShapesAreUnchanged:
    def test_one_row_per_market_scores_the_winning_label_and_applies_no_transform(self) -> None:
        report = check_no_leakage(_one_row_per_market(), raise_on_leak=False)
        assert report.target == "winning_label"
        assert report.bin_relative == []

    def test_an_explicit_label_is_scored_as_given_without_the_transform(self) -> None:
        df = _market_level_panel()
        df["won"] = _won(df)
        report = check_no_leakage(df, label_col="won", raise_on_leak=False)
        assert report.target == "won"
        assert report.bin_relative == []
        assert report.scores["feature_realised"] == pytest.approx(0.5)

    def test_a_per_bin_panel_without_market_level_features_has_nothing_bin_relative(self) -> None:
        df = _market_level_panel().drop(columns=["feature_fcst", "feature_realised"])
        df["feature_pos"] = np.arange(len(df)) % 9
        report = check_no_leakage(df, raise_on_leak=False)
        assert report.bin_relative == []
