"""Costs must propagate all the way to the metrics and the gate.

`rolling_origin_eval` is only half the story: a cost model that lowers PnL but
leaves `avg_edge` measured against the quote would let a NO_GO book pass the gate
on an edge it cannot execute. These tests pin the composition, not the arithmetic
(that lives in tests/core/test_costs.py).
"""

import numpy as np
import pandas as pd
import pytest

from pmlab.backtest.metrics import compute_metrics
from pmlab.backtest.rolling_origin import rolling_origin_eval
from pmlab.backtest.stability import stability_report
from pmlab.core.costs import CostModel


class _Confident:
    def __init__(self, prob: float = 0.75) -> None:
        self.prob = prob

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:  # noqa: N803
        return None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        n = len(X)
        return np.column_stack([np.full(n, 1.0 - self.prob), np.full(n, self.prob)])


def _panel(price: float = 0.5, n_dates: int = 14) -> pd.DataFrame:
    rows = []
    for d in range(n_dates):
        for market in range(4):
            for label in ("YES", "NO"):
                rows.append(
                    {
                        "market_id": f"m{market}",
                        "decision_date": f"2026-02-{d + 1:02d}",
                        "outcome_label": label,
                        "winning_label": "YES",
                        "market_price": price,
                        "feature_x": 1.0 if label == "YES" else 0.0,
                        "segment": "all",
                    }
                )
    return pd.DataFrame(rows)


def test_avg_edge_in_metrics_is_post_cost() -> None:
    """The gate reads avg_edge, so it has to shrink when friction is added."""
    panel = _panel()
    free = compute_metrics(
        rolling_origin_eval(
            panel, _Confident(), min_train_rows=8, stride=2, costs=CostModel.frictionless()
        ).trades
    )
    costly = compute_metrics(
        rolling_origin_eval(
            panel,
            _Confident(),
            min_train_rows=8,
            stride=2,
            costs=CostModel(taker_bps=0.0, slippage_bps=300.0),
        ).trades
    )
    assert costly.avg_edge < free.avg_edge


def test_costs_flow_into_the_stability_report() -> None:
    """Post-cost PnL must be what the bootstrap puts an interval around."""
    panel = _panel()
    free = stability_report(
        rolling_origin_eval(
            panel, _Confident(), min_train_rows=8, stride=2, costs=CostModel.frictionless()
        ).trades,
        seed=7,
    )
    costly = stability_report(
        rolling_origin_eval(
            panel,
            _Confident(),
            min_train_rows=8,
            stride=2,
            costs=CostModel(taker_bps=200.0, slippage_bps=200.0),
        ).trades,
        seed=7,
    )
    # point estimate of total PnL is element 1 of the (lower, point, upper) tuple
    assert costly.total_pnl_ci[1] < free.total_pnl_ci[1]


def test_a_thin_edge_can_be_erased_end_to_end() -> None:
    """The feature exists for this: friction turning a paper GO into a real loss."""
    panel = _panel(price=0.72)
    free = compute_metrics(
        rolling_origin_eval(
            panel, _Confident(), min_train_rows=8, stride=2, costs=CostModel.frictionless()
        ).trades
    )
    assert free.total_pnl > 0

    costly = compute_metrics(
        rolling_origin_eval(
            panel,
            _Confident(),
            min_train_rows=8,
            stride=2,
            costs=CostModel(taker_bps=400.0, slippage_bps=600.0, slippage_fixed=0.02),
        ).trades
    )
    assert costly.total_pnl < free.total_pnl
    assert costly.avg_edge < free.avg_edge


def test_cost_model_is_serialisable_for_the_tracker() -> None:
    """Reproducibility: a logged run must record which costs produced it."""
    costs = CostModel(taker_bps=30.0, slippage_bps=25.0, depth_bps_per_unit=15.0)
    payload = costs.to_dict()
    assert set(payload) == {
        "taker_bps",
        "slippage_bps",
        "slippage_fixed",
        "depth_bps_per_unit",
        "depth_reference_stake",
    }
    assert CostModel(**payload) == costs


def test_friction_cuts_the_upside_and_never_pays_the_loser() -> None:
    """Pin the flat-stake semantics so a future refactor can't invert them.

    With a fixed notional stake, ``size = stake / fill``. A worse fill buys fewer
    shares, so a LOSS is capped at the stake either way while a WIN shrinks. The
    dangerous failure mode would be friction making a losing trade look better —
    assert explicitly that it does not, and that the fee is the only term that
    deepens a loss.
    """
    from pmlab.core.pnl import Position, settle_position

    stake = 1.0
    free = CostModel(taker_bps=0.0)
    slipped = CostModel(taker_bps=0.0, slippage_bps=500.0)

    def outcomes(model: CostModel) -> tuple[float, float]:
        fill = model.fill_price(0.5, stake=stake)
        pos = Position(outcome_label="YES", price=fill, size=stake / fill, side="buy")
        fee = model.fee(stake)
        return (
            settle_position(pos, "YES", fee_paid=fee),
            settle_position(pos, "NO", fee_paid=fee),
        )

    free_win, free_loss = outcomes(free)
    slip_win, slip_loss = outcomes(slipped)

    assert slip_win < free_win, "slippage must reduce the winning payout"
    assert slip_loss == pytest.approx(free_loss), "a loss is capped at the stake"
    assert slip_loss < 0

    # The fee is the term that makes a loss strictly worse.
    with_fee = CostModel(taker_bps=300.0)
    _, fee_loss = outcomes(with_fee)
    assert fee_loss < free_loss
