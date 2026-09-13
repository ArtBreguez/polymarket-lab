"""Bootstrap stability report for backtest results.

A single point estimate (total PnL, hit rate) can't tell a real edge from a lucky
streak. `stability_report` resamples the trade log with replacement to put a
confidence interval — and a probability the edge is positive — around each
metric, so a "GO" is judged on a distribution rather than one number.

Pure function over the same trades DataFrame that `compute_metrics` consumes; it
does not touch `rolling_origin_eval` or change any existing result. Deterministic
given `seed`, honoring the "reproducible over clever" principle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# (lower_bound, point_estimate, upper_bound)
CI = tuple[float, float, float]


@dataclass(frozen=True)
class StabilityReport:
    """Bootstrap confidence intervals around the headline backtest metrics.

    Each ``*_ci`` is ``(lower, point, upper)`` where ``point`` is the observed
    sample statistic (not a bootstrap mean) and ``lower``/``upper`` are the
    bootstrap percentile bounds at ``ci_level``. ``prob_positive_pnl`` is the
    fraction of bootstrap replicas whose total PnL is > 0 — an intuitive
    "how sure are we the edge is real" number.
    """

    num_trades: int
    n_boot: int
    ci_level: float
    total_pnl_ci: CI
    avg_pnl_per_trade_ci: CI
    hit_rate_ci: CI
    prob_positive_pnl: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_trades": self.num_trades,
            "n_boot": self.n_boot,
            "ci_level": self.ci_level,
            "total_pnl_ci": list(self.total_pnl_ci),
            "avg_pnl_per_trade_ci": list(self.avg_pnl_per_trade_ci),
            "hit_rate_ci": list(self.hit_rate_ci),
            "prob_positive_pnl": self.prob_positive_pnl,
        }


def _won_mask(trades: pd.DataFrame) -> np.ndarray:
    """Boolean win mask: use ``outcome`` when present, else sign of PnL — the
    same convention as ``compute_metrics`` so the two never disagree."""
    if "outcome" in trades.columns:
        return (trades["outcome"] == "won").to_numpy()
    return (trades["realized_pnl"] > 0).to_numpy()


def stability_report(
    trades: pd.DataFrame,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    ci: float = 0.95,
) -> StabilityReport:
    """Bootstrap confidence intervals for a trade log.

    Args:
        trades: DataFrame with a ``realized_pnl`` column (and optional
            ``outcome``), exactly as produced by ``rolling_origin_eval`` /
            consumed by ``compute_metrics``.
        n_boot: number of bootstrap resamples (with replacement). Must be > 0.
        seed: RNG seed for reproducibility.
        ci: confidence level in (0, 1), e.g. 0.95 for a 95% interval.

    Returns:
        StabilityReport. For an empty trade log every interval is (0, 0, 0) and
        ``prob_positive_pnl`` is 0.0.

    Raises:
        ValueError: if ``ci`` is not in (0, 1), ``n_boot`` <= 0, or
            ``realized_pnl`` contains any non-finite value (NaN/Inf).
    """
    if not (0.0 < ci < 1.0):
        raise ValueError(f"ci must be in (0, 1), got {ci}")
    if n_boot <= 0:
        raise ValueError(f"n_boot must be > 0, got {n_boot}")

    n = len(trades)
    if n == 0:
        zero: CI = (0.0, 0.0, 0.0)
        return StabilityReport(
            num_trades=0,
            n_boot=n_boot,
            ci_level=ci,
            total_pnl_ci=zero,
            avg_pnl_per_trade_ci=zero,
            hit_rate_ci=zero,
            prob_positive_pnl=0.0,
        )

    pnl = trades["realized_pnl"].to_numpy(dtype=float)
    # Fail loud on non-finite PnL. A NaN (e.g. an unsettled trade) or Inf would
    # silently poison every statistic — numpy's sum/percentile propagate NaN,
    # producing a (nan, nan, nan) interval and an understated prob_positive_pnl,
    # and to_dict() would then emit invalid JSON (bare NaN/Infinity) into the
    # ExperimentTracker. A stability report over non-finite data is meaningless,
    # so surface it instead of hiding it.
    if not np.isfinite(pnl).all():
        n_bad = int((~np.isfinite(pnl)).sum())
        raise ValueError(
            f"realized_pnl contains {n_bad} non-finite value(s) (NaN/Inf); "
            "clean or drop unsettled trades before computing a stability report"
        )
    won = _won_mask(trades).astype(float)

    rng = np.random.default_rng(seed)
    # Draw all bootstrap indices at once: (n_boot, n). Each row is one resample.
    idx = rng.integers(0, n, size=(n_boot, n))
    sample_pnl = pnl[idx]  # (n_boot, n)
    sample_won = won[idx]  # (n_boot, n)

    boot_total_pnl = sample_pnl.sum(axis=1)
    boot_avg_pnl = sample_pnl.mean(axis=1)
    boot_hit_rate = sample_won.mean(axis=1)

    lo_q = (1.0 - ci) / 2.0 * 100.0
    hi_q = (1.0 + ci) / 2.0 * 100.0

    def _ci(boot: np.ndarray, point: float) -> CI:
        lo, hi = np.percentile(boot, [lo_q, hi_q])
        return (float(lo), float(point), float(hi))

    total_pnl_point = float(pnl.sum())
    avg_pnl_point = float(pnl.mean())
    hit_rate_point = float(won.mean())

    return StabilityReport(
        num_trades=n,
        n_boot=n_boot,
        ci_level=ci,
        total_pnl_ci=_ci(boot_total_pnl, total_pnl_point),
        avg_pnl_per_trade_ci=_ci(boot_avg_pnl, avg_pnl_point),
        hit_rate_ci=_ci(boot_hit_rate, hit_rate_point),
        prob_positive_pnl=float((boot_total_pnl > 0).mean()),
    )
