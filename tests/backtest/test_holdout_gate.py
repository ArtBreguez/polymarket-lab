"""Tests for backtest.holdout_gate."""

import pandas as pd
import pytest

from pmlab.backtest.holdout_gate import HoldoutGateResult


def _make_trades(segments_counts: dict[str, int], pnl_per_trade: float = 0.1) -> pd.DataFrame:
    """Helper: create trades DataFrame for given segment trade counts."""
    rows = []
    for seg, n in segments_counts.items():
        for _ in range(n):
            rows.append({"realized_pnl": pnl_per_trade, "outcome": "won", "segment": seg})
    return (
        pd.DataFrame(rows) if rows else pd.DataFrame(columns=["realized_pnl", "outcome", "segment"])
    )


def test_all_pass_go():
    trades = _make_trades({"A": 50, "B": 60})
    result = HoldoutGateResult.evaluate(
        trades, required_segments=["A", "B"], min_trades_per_segment=40
    )
    assert result.decision == "GO"
    assert all(r.passes for r in result.segment_results)


def test_one_fail_pnl_nogo():
    rows = []
    for _ in range(50):
        rows.append({"realized_pnl": -0.1, "outcome": "lost", "segment": "A"})
    for _ in range(50):
        rows.append({"realized_pnl": 0.1, "outcome": "won", "segment": "B"})
    trades = pd.DataFrame(rows)

    result = HoldoutGateResult.evaluate(
        trades,
        required_segments=["A", "B"],
        min_trades_per_segment=40,
        min_pnl_per_segment=0.0,
    )
    assert result.decision == "NO_GO"
    seg_a = next(r for r in result.segment_results if r.segment == "A")
    assert not seg_a.passes
    assert seg_a.reason == "negative_pnl"


def test_trade_count_fail():
    trades = _make_trades({"A": 10, "B": 50})
    result = HoldoutGateResult.evaluate(
        trades, required_segments=["A", "B"], min_trades_per_segment=40
    )
    assert result.decision == "NO_GO"
    seg_a = next(r for r in result.segment_results if r.segment == "A")
    assert seg_a.reason == "insufficient_trades"


def test_missing_segment_nogo():
    trades = _make_trades({"A": 50})
    result = HoldoutGateResult.evaluate(
        trades, required_segments=["A", "B"], min_trades_per_segment=40
    )
    assert result.decision == "NO_GO"
    seg_b = next(r for r in result.segment_results if r.segment == "B")
    assert seg_b.reason == "missing"
    assert not seg_b.passes


def test_reasons_correct():
    rows = []
    # A: ok (50 trades, positive pnl)
    for _ in range(50):
        rows.append({"realized_pnl": 0.1, "outcome": "won", "segment": "A"})
    # B: insufficient_trades (5 trades)
    for _ in range(5):
        rows.append({"realized_pnl": 0.1, "outcome": "won", "segment": "B"})
    # C: negative_pnl
    for _ in range(50):
        rows.append({"realized_pnl": -0.1, "outcome": "lost", "segment": "C"})
    trades = pd.DataFrame(rows)

    result = HoldoutGateResult.evaluate(
        trades,
        required_segments=["A", "B", "C", "D"],
        min_trades_per_segment=40,
        min_pnl_per_segment=0.0,
    )
    reasons = {r.segment: r.reason for r in result.segment_results}
    assert reasons["A"] == "ok"
    assert reasons["B"] == "insufficient_trades"
    assert reasons["C"] == "negative_pnl"
    assert reasons["D"] == "missing"
    assert result.decision == "NO_GO"


def test_to_dict_from_dict_roundtrip():
    trades = _make_trades({"A": 50, "B": 60})
    result = HoldoutGateResult.evaluate(
        trades, required_segments=["A", "B"], min_trades_per_segment=40
    )
    d = result.to_dict()
    restored = HoldoutGateResult.from_dict(d)
    assert restored.decision == result.decision
    assert restored.aggregate_pnl == pytest.approx(result.aggregate_pnl)
    assert len(restored.segment_results) == len(result.segment_results)
