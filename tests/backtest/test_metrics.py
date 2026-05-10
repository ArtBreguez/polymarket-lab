"""Tests for backtest.metrics."""

import pandas as pd
import pytest

from pmlab.backtest.metrics import compute_metrics


def test_empty_trades():
    df = pd.DataFrame(columns=["realized_pnl", "outcome", "edge"])
    m = compute_metrics(df)
    assert m.num_trades == 0
    assert m.total_pnl == 0.0
    assert m.hit_rate == 0.0
    assert m.avg_pnl_per_trade == 0.0
    assert m.avg_edge == 0.0


def test_pnl_sum():
    df = pd.DataFrame(
        {
            "realized_pnl": [1.0, -0.5, 2.0],
            "outcome": ["won", "lost", "won"],
            "edge": [0.1, -0.05, 0.2],
        }
    )
    m = compute_metrics(df)
    assert m.num_trades == 3
    assert m.total_pnl == pytest.approx(2.5)
    assert m.avg_pnl_per_trade == pytest.approx(2.5 / 3)


def test_hit_rate():
    df = pd.DataFrame(
        {
            "realized_pnl": [1.0, -0.5, -0.3, 0.8],
            "outcome": ["won", "lost", "lost", "won"],
            "edge": [0.1, -0.05, -0.03, 0.08],
        }
    )
    m = compute_metrics(df)
    assert m.hit_rate == pytest.approx(0.5)


def test_avg_edge():
    df = pd.DataFrame(
        {
            "realized_pnl": [1.0, -0.5],
            "outcome": ["won", "lost"],
            "edge": [0.2, 0.1],
        }
    )
    m = compute_metrics(df)
    assert m.avg_edge == pytest.approx(0.15)
