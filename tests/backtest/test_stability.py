"""Tests for backtest.stability — bootstrap confidence intervals.

A single point estimate (total_pnl / hit_rate) hides sampling risk: a "GO" built
on 60 lucky trades is not the same as one built on 600 consistent ones. The
stability report resamples the trade log with replacement to put a confidence
interval — and a probability the edge is real — around each metric.
"""

import numpy as np
import pandas as pd
import pytest

from pmlab.backtest.stability import StabilityReport, stability_report


def _winning_trades(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Positive-EV book: mean +0.05/trade with noise → clearly profitable.
    pnl = rng.normal(0.05, 0.2, n)
    return pd.DataFrame({"realized_pnl": pnl, "outcome": ["won" if p > 0 else "lost" for p in pnl]})


def _coinflip_trades(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Zero-EV book: mean 0 → PnL CI should straddle zero.
    pnl = rng.normal(0.0, 0.2, n)
    return pd.DataFrame({"realized_pnl": pnl, "outcome": ["won" if p > 0 else "lost" for p in pnl]})


def test_returns_stability_report(tmp_path):
    rep = stability_report(_winning_trades(), n_boot=500, seed=1)
    assert isinstance(rep, StabilityReport)
    assert rep.num_trades == 200
    assert rep.n_boot == 500


def test_ci_ordering_and_point_estimate():
    rep = stability_report(_winning_trades(), n_boot=500, seed=1)
    # Each interval is lower <= point <= upper.
    for lo, pt, hi in (
        rep.total_pnl_ci,
        rep.avg_pnl_per_trade_ci,
        rep.hit_rate_ci,
    ):
        assert lo <= pt <= hi
    # The point estimate matches the observed sample mean (not a bootstrap mean).
    obs = _winning_trades()
    assert rep.total_pnl_ci[1] == pytest.approx(float(obs["realized_pnl"].sum()))
    assert rep.avg_pnl_per_trade_ci[1] == pytest.approx(float(obs["realized_pnl"].mean()))


def test_profitable_book_ci_above_zero():
    """A clearly +EV book: the lower bound of avg PnL should exceed zero, and
    prob_positive_pnl should be very high."""
    rep = stability_report(_winning_trades(n=400, seed=2), n_boot=1000, seed=7)
    assert rep.avg_pnl_per_trade_ci[0] > 0
    assert rep.prob_positive_pnl > 0.95


def test_coinflip_book_ci_straddles_zero():
    """A zero-EV book: the avg-PnL interval should contain zero and
    prob_positive_pnl should be near 0.5, so a naive 'GO' is exposed as noise."""
    rep = stability_report(_coinflip_trades(n=400, seed=3), n_boot=1000, seed=7)
    lo, _, hi = rep.avg_pnl_per_trade_ci
    assert lo < 0 < hi
    assert 0.25 < rep.prob_positive_pnl < 0.75


def test_deterministic_with_seed():
    a = stability_report(_winning_trades(), n_boot=500, seed=42)
    b = stability_report(_winning_trades(), n_boot=500, seed=42)
    assert a.total_pnl_ci == b.total_pnl_ci
    assert a.hit_rate_ci == b.hit_rate_ci
    assert a.prob_positive_pnl == b.prob_positive_pnl


def test_different_seed_changes_intervals():
    a = stability_report(_winning_trades(), n_boot=500, seed=1)
    b = stability_report(_winning_trades(), n_boot=500, seed=2)
    # Point estimates are identical (same sample); bootstrap bounds differ.
    assert a.total_pnl_ci[1] == pytest.approx(b.total_pnl_ci[1])
    assert a.total_pnl_ci != b.total_pnl_ci


def test_custom_ci_level_widens_interval():
    narrow = stability_report(_winning_trades(), n_boot=1000, seed=5, ci=0.80)
    wide = stability_report(_winning_trades(), n_boot=1000, seed=5, ci=0.99)
    # A higher confidence level must not produce a narrower interval.
    assert wide.total_pnl_ci[0] <= narrow.total_pnl_ci[0]
    assert wide.total_pnl_ci[2] >= narrow.total_pnl_ci[2]


def test_empty_trades_is_degenerate():
    rep = stability_report(pd.DataFrame(columns=["realized_pnl"]), n_boot=100, seed=1)
    assert rep.num_trades == 0
    assert rep.total_pnl_ci == (0.0, 0.0, 0.0)
    assert rep.hit_rate_ci == (0.0, 0.0, 0.0)
    assert rep.prob_positive_pnl == 0.0


def test_invalid_ci_raises():
    with pytest.raises(ValueError, match="ci"):
        stability_report(_winning_trades(), ci=1.5)


def test_invalid_n_boot_raises():
    with pytest.raises(ValueError, match="n_boot"):
        stability_report(_winning_trades(), n_boot=0)


def test_to_dict_roundtrips_fields():
    rep = stability_report(_winning_trades(), n_boot=200, seed=1)
    d = rep.to_dict()
    assert d["num_trades"] == 200
    assert d["n_boot"] == 200
    assert tuple(d["total_pnl_ci"]) == rep.total_pnl_ci
    assert d["prob_positive_pnl"] == rep.prob_positive_pnl
    assert d["ci_level"] == rep.ci_level


def test_hit_rate_derived_from_pnl_sign_when_no_outcome_column():
    """Without an ``outcome`` column, wins are derived from the sign of
    realized_pnl — matching compute_metrics, so the two never disagree. This is
    the shape rolling_origin_eval emits directly."""
    rng = np.random.default_rng(11)
    pnl = rng.normal(0.05, 0.2, 300)
    df = pd.DataFrame({"realized_pnl": pnl})  # no 'outcome' column
    rep = stability_report(df, n_boot=500, seed=1)
    expected_hit = float((pnl > 0).mean())
    assert rep.hit_rate_ci[1] == pytest.approx(expected_hit)
    assert rep.hit_rate_ci[0] <= rep.hit_rate_ci[1] <= rep.hit_rate_ci[2]
