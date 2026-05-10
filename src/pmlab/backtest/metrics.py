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
        trades: DataFrame with columns:
            - realized_pnl (float)
            - outcome ("won" | "lost")
            - edge (float)

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
    won = (trades["outcome"] == "won").sum()
    hit_rate = float(won / n)
    avg_pnl = total_pnl / n
    avg_edge = float(trades["edge"].mean())

    return BacktestMetrics(
        num_trades=n,
        total_pnl=total_pnl,
        hit_rate=hit_rate,
        avg_pnl_per_trade=avg_pnl,
        avg_edge=avg_edge,
    )
