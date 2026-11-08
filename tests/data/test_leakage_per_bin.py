"""check_no_leakage on a ranged-market panel, one row per bin (issue #35).

``rolling_origin_eval`` trains on ``outcome_label == winning_label``, a binary outcome per
row. On a per-bin panel ``winning_label`` is the same for every bin of a market, so scoring a
feature against it asks "does this feature tell *which market* resolved to the rarest label",
and an honest forecast answers yes. The check must score what the model learns instead, and
must still flag a feature that is a copy of the outcome.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pmlab.data import LeakageError, check_no_leakage
from pmlab.data.panel_builder import build_panel

N_BINS = 9
HALF = N_BINS // 2
# every winning label but the last market's is shared by many markets; the last one is a
# heat wave that no other market resolves to, so it is the rarest label by a wide margin
REALISED_CYCLE = (18, 19, 20, 21, 22)
HEAT_WAVE = 30


def _per_bin_rows(n_markets: int = 60, seed: int = 0) -> list[list[dict[str, object]]]:
    """A ranged-market panel in the shape ``build_training_rows`` returns: a list per market."""
    rng = np.random.default_rng(seed)
    realised = [REALISED_CYCLE[i % len(REALISED_CYCLE)] for i in range(n_markets - 1)]
    realised.append(HEAT_WAVE)
    markets = []
    for i, real in enumerate(realised):
        # an honest forecast: informative (within a degree of the truth), never the truth
        fcst = real + (-1, 0, 1)[i % 3] + 0.1 * float(rng.uniform(-1, 1))
        centre = round(fcst)
        rows = []
        for k in range(-HALF, HALF + 1):
            b = centre + k
            dist = b - fcst
            rows.append(
                {
                    "market_id": f"m{i:03d}",
                    "decision_date": f"2025-01-{1 + i % 28:02d}",
                    "outcome_label": f"{b}°C",
                    "winning_label": f"{real}°C",
                    "market_price": float(np.clip(0.4 * np.exp(-0.5 * (dist / 2) ** 2), 0.02, 0.9)),
                    "feature_fcst": fcst,  # the same for every bin of the market
                    "feature_bin_dist": dist,  # relative to the bin
                    "_realised": float(real),  # kept to build leaks; not a feature
                    "_bin": float(b),
                }
            )
        markets.append(rows)
    return markets


def _panel(**kwargs: object) -> pd.DataFrame:
    rows = [r for market in _per_bin_rows(**kwargs) for r in market]  # type: ignore[arg-type]
    return pd.DataFrame(rows)


def _won(df: pd.DataFrame) -> pd.Series:
    return (df["outcome_label"] == df["winning_label"]).astype(int)


class TestTheFalsePositive:
    def test_the_fixture_is_a_real_per_bin_panel(self):
        df = _panel()
        assert df.groupby("market_id")["outcome_label"].nunique().eq(N_BINS).all()
        assert _won(df).groupby(df["market_id"]).sum().eq(1).all()  # one winning bin per market
        assert df["winning_label"].value_counts().index[-1] == f"{HEAT_WAVE}°C"

    def test_honest_features_of_a_per_bin_panel_do_not_raise(self):
        # RED before the fix: feature_fcst scores ~1.0 against the multiclass winning_label
        # (the heat-wave market's forecast ranks first), which is a forecast doing its job.
        report = check_no_leakage(_panel())
        assert report.leaked == []

    def test_build_panel_check_leakage_accepts_honest_per_bin_rows(self):
        panel = build_panel(_per_bin_rows(), check_leakage=True)
        assert len(panel) == 60 * N_BINS

    def test_the_report_says_what_it_scored(self):
        report = check_no_leakage(_panel())
        assert report.target == "outcome_label == winning_label"

    def test_a_feature_the_same_for_every_bin_scores_one_half(self):
        # The model learns per row; a feature identical across a market's bins cannot tell
        # its winning bin from its losing ones. That is by construction, not an accident.
        report = check_no_leakage(_panel())
        assert report.scores["feature_fcst"] == pytest.approx(0.5)

    def test_a_bin_relative_feature_is_scored_per_row(self):
        report = check_no_leakage(_panel())
        assert "feature_bin_dist" in report.scores
        assert report.scores["feature_bin_dist"] < 0.97


class TestARealLeakIsStillCaught:
    def test_a_per_bin_feature_copied_from_the_outcome_is_caught(self):
        df = _panel()
        df["feature_copy"] = _won(df) + np.random.default_rng(1).normal(0, 0.05, len(df))
        with pytest.raises(LeakageError, match="feature_copy"):
            check_no_leakage(df)

    def test_a_feature_derived_from_the_realised_value_is_caught(self):
        # distance of the bin to what actually happened, instead of to the forecast
        df = _panel()
        df["feature_dist_to_realised"] = (df["_bin"] - df["_realised"]).abs()
        report = check_no_leakage(df, raise_on_leak=False)
        assert report.leaked == ["feature_dist_to_realised"]
        assert report.scores["feature_dist_to_realised"] > 0.99

    def test_build_panel_refuses_a_per_bin_leak(self):
        rows = _per_bin_rows()
        for market in rows:
            for r in market:
                r["feature_dist_to_realised"] = abs(float(r["_bin"]) - float(r["_realised"]))  # type: ignore[arg-type]
        with pytest.raises(LeakageError, match="feature_dist_to_realised"):
            build_panel(rows, check_leakage=True)

    def test_the_error_names_the_target_it_scored(self):
        df = _panel()
        df["feature_copy"] = _won(df).astype(float)
        with pytest.raises(LeakageError, match=r"outcome_label == winning_label"):
            check_no_leakage(df)


def _one_row_per_market(seed: int = 0) -> pd.DataFrame:
    """One row per market: the model's pick (``outcome_label``) against a 3-way winner."""
    rng = np.random.default_rng(seed)
    winners = np.array(["A"] * 40 + ["B"] * 30 + ["C"] * 20)  # C is the rarest
    rng.shuffle(winners)
    n = len(winners)
    return pd.DataFrame(
        {
            "market_id": [f"m{i}" for i in range(n)],
            "decision_date": ["2025-01-01"] * n,
            "outcome_label": rng.choice(["A", "B", "C"], n),
            "winning_label": winners,
            "market_price": rng.uniform(0.1, 0.9, n),
            "feature_noise": rng.normal(0, 1, n),
        }
    )


class TestOtherPanelShapesAreUnchanged:
    def test_one_row_per_market_still_scores_the_winning_label(self):
        report = check_no_leakage(_one_row_per_market(), raise_on_leak=False)
        assert report.target == "winning_label"

    def test_a_copy_of_the_winner_on_a_one_row_per_market_panel_is_still_caught(self):
        # The pick is random, so a feature that encodes the winner says nothing about
        # `outcome_label == winning_label`: only the winning_label score can see it.
        df = _one_row_per_market()
        df["feature_winner"] = df["winning_label"].map({"A": 0.0, "B": 1.0, "C": 2.0})
        with pytest.raises(LeakageError, match="feature_winner"):
            check_no_leakage(df)

    def test_a_market_seen_on_two_dates_with_one_bin_each_is_not_per_bin(self):
        # per-bin means several bins of one market at one decision; the backtest chooses
        # within (market_id, decision_date), so one row there leaves nothing to choose from
        df = _one_row_per_market()
        later = df.assign(decision_date="2025-01-02", outcome_label="B")
        both = pd.concat([df.assign(outcome_label="A"), later], ignore_index=True)
        report = check_no_leakage(both, raise_on_leak=False)
        assert report.target == "winning_label"

    def test_a_frame_without_market_ids_scores_the_winning_label(self):
        df = _one_row_per_market().drop(columns=["market_id"])
        assert check_no_leakage(df, raise_on_leak=False).target == "winning_label"


class TestAnExplicitLabelIsUsedAsGiven:
    def test_label_col_overrides_the_automatic_target(self):
        # the tutorial's old workaround keeps working: an explicit column is scored as is
        df = _panel()
        df["won"] = _won(df)
        report = check_no_leakage(df, label_col="won", raise_on_leak=False)
        assert report.target == "won"
        assert report.leaked == []

    def test_asking_for_winning_label_gets_the_multiclass_score(self):
        # an explicit request is honoured, which is how to reproduce the old behaviour
        report = check_no_leakage(_panel(), label_col="winning_label", raise_on_leak=False)
        assert report.target == "winning_label"
        assert "feature_fcst" in report.leaked

    def test_a_missing_explicit_label_still_raises(self):
        with pytest.raises(ValueError, match="nope"):
            check_no_leakage(_panel(), label_col="nope")

    def test_a_per_bin_panel_without_winning_label_raises_as_before(self):
        with pytest.raises(ValueError, match="winning_label"):
            check_no_leakage(_panel().drop(columns=["winning_label"]))
