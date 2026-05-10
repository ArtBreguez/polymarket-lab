"""Tests for PnL accounting primitives."""

from __future__ import annotations

from pmlab.core.pnl import Position, settle_position


class TestSettlePosition:
    def test_buy_win(self) -> None:
        pos = Position(outcome_label="YES", price=0.3, size=3.33, side="buy")
        pnl = settle_position(pos, winning_label="YES", fee_paid=0.003)
        expected = (1.0 - 0.3) * 3.33 - 0.003
        assert abs(pnl - expected) < 1e-9

    def test_buy_lose(self) -> None:
        pos = Position(outcome_label="YES", price=0.3, size=3.33, side="buy")
        pnl = settle_position(pos, winning_label="NO", fee_paid=0.003)
        expected = (0.0 - 0.3) * 3.33 - 0.003
        assert abs(pnl - expected) < 1e-9

    def test_sell_win(self) -> None:
        """Sell side: we sold YES (NO position). We win when outcome != our label."""
        pos = Position(outcome_label="NO", price=0.7, size=1.43, side="sell")
        pnl = settle_position(pos, winning_label="NO", fee_paid=0.003)
        # sell: (price - payout) * size - fee; payout=1.0 when label==winning
        expected = (0.7 - 1.0) * 1.43 - 0.003
        assert abs(pnl - expected) < 1e-9

    def test_sell_lose(self) -> None:
        pos = Position(outcome_label="NO", price=0.7, size=1.43, side="sell")
        pnl = settle_position(pos, winning_label="YES", fee_paid=0.003)
        # payout=0.0
        expected = (0.7 - 0.0) * 1.43 - 0.003
        assert abs(pnl - expected) < 1e-9

    def test_zero_fee(self) -> None:
        pos = Position(outcome_label="A", price=0.5, size=2.0, side="buy")
        pnl = settle_position(pos, winning_label="A", fee_paid=0.0)
        assert abs(pnl - 1.0) < 1e-9  # (1-0.5)*2 = 1.0

    def test_flat_stake_round_trip(self) -> None:
        """Buying at price p with size=1/p should produce gross_pnl≈(1-p)/p."""
        price = 0.25
        size = 1.0 / price  # 4.0 shares
        pos = Position(outcome_label="X", price=price, size=size, side="buy")
        pnl = settle_position(pos, winning_label="X", fee_paid=0.003)
        assert pnl > 0
        assert abs(pnl - ((1.0 - price) * size - 0.003)) < 1e-9
