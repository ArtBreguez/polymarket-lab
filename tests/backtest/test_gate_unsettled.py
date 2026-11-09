"""A trade with no realized PnL is not evidence: the gate must not count it as a pass (#51).

``realized_pnl`` is NaN/None for a trade that has not settled and for one whose cost was NaN.
``DataFrame.sum()`` skips it, so before 1.1.1 such trades counted towards
``min_trades_per_segment`` and added 0 to the PnL: sixty open trades were a GO.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pmlab import HoldoutGateResult, compute_metrics


def _trades(pnls: list[float | None], segment: str = "A") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "realized_pnl": p,
                "outcome": None if p is None or p != p else ("won" if p > 0 else "lost"),
                "segment": segment,
            }
            for p in pnls
        ]
    )


@pytest.mark.parametrize("required", [None, [], ["A"]])
def test_open_trades_are_not_a_go(required: list[str] | None) -> None:
    gate = HoldoutGateResult.evaluate(_trades([None] * 60), required_segments=required)
    assert gate.decision == "NO_GO"
    assert [r.reason for r in gate.segment_results] == ["unsettled_trades"]
    assert not gate.segment_results[0].passes


def test_one_open_trade_among_winners_is_not_a_go() -> None:
    gate = HoldoutGateResult.evaluate(_trades([0.1] * 60 + [None]), required_segments=["A"])
    assert gate.decision == "NO_GO"
    assert gate.segment_results[0].reason == "unsettled_trades"


def test_nan_pnl_is_not_a_go() -> None:
    gate = HoldoutGateResult.evaluate(_trades([math.nan] * 60), required_segments=["A"])
    assert gate.decision == "NO_GO"


def test_infinite_pnl_is_not_a_go() -> None:
    gate = HoldoutGateResult.evaluate(_trades([0.1] * 60 + [math.inf]), required_segments=["A"])
    assert gate.decision == "NO_GO"
    assert gate.segment_results[0].reason == "non_finite_pnl"


def test_only_the_segment_with_open_trades_fails() -> None:
    trades = pd.concat([_trades([0.1] * 50, "A"), _trades([None] * 50, "B")], ignore_index=True)
    gate = HoldoutGateResult.evaluate(trades, required_segments=["A", "B"])
    by_seg = {r.segment: r for r in gate.segment_results}
    assert by_seg["A"].passes and by_seg["A"].reason == "ok"
    assert not by_seg["B"].passes and by_seg["B"].reason == "unsettled_trades"
    assert gate.decision == "NO_GO"


def test_settled_trades_still_go() -> None:
    gate = HoldoutGateResult.evaluate(_trades([0.1] * 60), required_segments=["A"])
    assert gate.decision == "GO"
    assert gate.segment_results[0].reason == "ok"


def test_compute_metrics_counts_only_settled_trades() -> None:
    m = compute_metrics(_trades([1.0, -0.5, None, np.nan]))
    assert m.num_trades == 2
    assert m.total_pnl == pytest.approx(0.5)
    assert m.avg_pnl_per_trade == pytest.approx(0.25)
    assert m.hit_rate == pytest.approx(0.5)


def test_compute_metrics_with_nothing_settled_is_all_zero() -> None:
    m = compute_metrics(_trades([None] * 5))
    assert (m.num_trades, m.total_pnl, m.hit_rate, m.avg_pnl_per_trade) == (0, 0.0, 0.0, 0.0)
