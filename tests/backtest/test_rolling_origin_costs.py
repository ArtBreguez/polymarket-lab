"""Tests for costs= wired into rolling_origin_eval.

Two things must hold at once: an opt-in cost model has to change fills and PnL in
the honest direction, and omitting it has to reproduce the pre-existing numbers
bit for bit. The second is the one that protects already-published results.
"""

import numpy as np
import pandas as pd
import pytest

from pmlab.backtest.rolling_origin import rolling_origin_eval
from pmlab.core.costs import CostModel


class _AlwaysConfident:
    """Deterministic stub: predicts a fixed probability, no fitting involved.

    Using a real learner here would make the assertions depend on LightGBM's
    behaviour; the subject under test is cost arithmetic, not the model.
    """

    def __init__(self, prob: float = 0.9) -> None:
        self.prob = prob

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:  # noqa: N803
        return None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        n = len(X)
        return np.column_stack([np.full(n, 1.0 - self.prob), np.full(n, self.prob)])


def _panel(n_dates: int = 12, price: float = 0.5) -> pd.DataFrame:
    """A panel where the favoured bin always wins, so PnL is positive pre-cost."""
    rows = []
    for d in range(n_dates):
        date = f"2026-01-{d + 1:02d}"
        for market in range(4):
            for label in ("YES", "NO"):
                rows.append(
                    {
                        "market_id": f"m{market}",
                        "decision_date": date,
                        "outcome_label": label,
                        "winning_label": "YES",
                        "market_price": price,
                        "feature_x": 1.0 if label == "YES" else 0.0,
                        "segment": "all",
                    }
                )
    return pd.DataFrame(rows)


class TestBackwardCompatibility:
    """No costs= argument means byte-identical results to before this feature."""

    def test_omitting_costs_matches_explicit_legacy_taker_bps(self) -> None:
        panel = _panel()
        legacy = rolling_origin_eval(panel, _AlwaysConfident(), min_train_rows=8, stride=2)
        with_default = rolling_origin_eval(
            panel, _AlwaysConfident(), min_train_rows=8, stride=2, costs=CostModel()
        )
        pd.testing.assert_frame_equal(legacy.trades, with_default.trades)

    def test_taker_bps_argument_still_honoured(self) -> None:
        """The pre-existing taker_bps parameter must keep working on its own."""
        panel = _panel()
        cheap = rolling_origin_eval(
            panel, _AlwaysConfident(), min_train_rows=8, stride=2, taker_bps=0.0
        )
        dear = rolling_origin_eval(
            panel, _AlwaysConfident(), min_train_rows=8, stride=2, taker_bps=100.0
        )
        assert cheap.trades["realized_pnl"].sum() > dear.trades["realized_pnl"].sum()

    def test_costs_and_taker_bps_together_is_rejected(self) -> None:
        """Two sources of truth for the fee would silently double-charge."""
        with pytest.raises(ValueError, match="taker_bps"):
            rolling_origin_eval(
                _panel(),
                _AlwaysConfident(),
                min_train_rows=8,
                stride=2,
                taker_bps=50.0,
                costs=CostModel(taker_bps=10.0),
            )


class TestCostsReducePnL:
    """Friction can only ever make a winning book worse."""

    def test_slippage_lowers_total_pnl(self) -> None:
        panel = _panel()
        free = rolling_origin_eval(
            panel, _AlwaysConfident(), min_train_rows=8, stride=2, costs=CostModel.frictionless()
        )
        slipped = rolling_origin_eval(
            panel,
            _AlwaysConfident(),
            min_train_rows=8,
            stride=2,
            costs=CostModel(taker_bps=0.0, slippage_bps=200.0),
        )
        assert slipped.trades["realized_pnl"].sum() < free.trades["realized_pnl"].sum()

    def test_depth_penalty_scales_with_stake(self) -> None:
        panel = _panel()
        costs = CostModel(taker_bps=0.0, depth_bps_per_unit=100.0, depth_reference_stake=1.0)
        small = rolling_origin_eval(
            panel, _AlwaysConfident(), min_train_rows=8, stride=2, flat_stake=1.0, costs=costs
        )
        big = rolling_origin_eval(
            panel, _AlwaysConfident(), min_train_rows=8, stride=2, flat_stake=5.0, costs=costs
        )
        # Per-unit-staked PnL must be worse for the larger order.
        assert big.trades["realized_pnl"].sum() / 5.0 < small.trades["realized_pnl"].sum()

    def test_enough_friction_turns_a_winner_into_a_loser(self) -> None:
        """The whole point: a real cost model can erase a paper edge."""
        panel = _panel(price=0.98)  # thin edge, expensive outcome
        free = rolling_origin_eval(
            panel, _AlwaysConfident(), min_train_rows=8, stride=2, costs=CostModel.frictionless()
        )
        assert free.trades["realized_pnl"].sum() > 0
        costly = rolling_origin_eval(
            panel,
            _AlwaysConfident(),
            min_train_rows=8,
            stride=2,
            costs=CostModel(taker_bps=300.0, slippage_bps=500.0),
        )
        assert costly.trades["realized_pnl"].sum() < 0


class TestTradeLogRecordsTheFill:
    """An auditable backtest has to show the price it actually paid."""

    def test_fill_price_column_is_present_and_worse_than_market(self) -> None:
        result = rolling_origin_eval(
            _panel(),
            _AlwaysConfident(),
            min_train_rows=8,
            stride=2,
            costs=CostModel(slippage_bps=200.0),
        )
        assert "fill_price" in result.trades.columns
        assert (result.trades["fill_price"] > result.trades["market_price"]).all()

    def test_edge_is_computed_against_the_fill_not_the_quote(self) -> None:
        """Edge measured on market_price would overstate a costed trade."""
        costs = CostModel(slippage_bps=400.0)
        result = rolling_origin_eval(
            _panel(), _AlwaysConfident(prob=0.6), min_train_rows=8, stride=2, costs=costs
        )
        row = result.trades.iloc[0]
        assert row["edge"] == pytest.approx(row["predicted_prob"] - row["fill_price"])

    def test_frictionless_fill_equals_market_price(self) -> None:
        result = rolling_origin_eval(
            _panel(), _AlwaysConfident(), min_train_rows=8, stride=2, costs=CostModel.frictionless()
        )
        assert (result.trades["fill_price"] == result.trades["market_price"]).all()
