"""Backtest performance metrics."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class BacktestMetrics:
    num_trades: int
    total_pnl: float
    hit_rate: float
    avg_pnl_per_trade: float
    avg_edge: float


def compute_metrics(trades: pd.DataFrame) -> BacktestMetrics:
    """Compute backtest metrics from a trades DataFrame.

    Args:
        trades: DataFrame with column ``realized_pnl`` (float). Optional columns:
            - ``outcome`` ("won" | "lost") — if absent, derived from the sign of
              ``realized_pnl`` so the output of ``rolling_origin_eval`` composes
              here directly.
            - ``edge`` (float) — averaged when present, otherwise reported as 0.

    Returns:
        BacktestMetrics with aggregated stats. All zeros if trades is empty.
    """
    if trades.empty:
        return BacktestMetrics(
            num_trades=0,
            total_pnl=0.0,
            hit_rate=0.0,
            avg_pnl_per_trade=0.0,
            avg_edge=0.0,
        )

    n = len(trades)
    total_pnl = float(trades["realized_pnl"].sum())
    if "outcome" in trades.columns:
        won = (trades["outcome"] == "won").sum()
    else:
        won = (trades["realized_pnl"] > 0).sum()
    hit_rate = float(won / n)
    avg_pnl = total_pnl / n
    avg_edge = float(trades["edge"].mean()) if "edge" in trades.columns else 0.0

    return BacktestMetrics(
        num_trades=n,
        total_pnl=total_pnl,
        hit_rate=hit_rate,
        avg_pnl_per_trade=avg_pnl,
        avg_edge=avg_edge,
    )
