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


class TestBuyNoSide:
    """`buy_no` is a long position in the NO token: `price` is the NO price."""

    def test_the_label_occurring_loses_the_whole_stake(self) -> None:
        # stake 1 at a NO price of 0.70 buys 1/0.70 shares that now pay nothing
        pos = Position(outcome_label="YES", price=0.7, size=1.0 / 0.7, side="buy_no")
        assert abs(settle_position(pos, winning_label="YES") - (-1.0)) < 1e-9

    def test_the_label_not_occurring_pays_one_per_share(self) -> None:
        pos = Position(outcome_label="YES", price=0.7, size=1.0 / 0.7, side="buy_no")
        assert abs(settle_position(pos, winning_label="NO") - (1.0 / 0.7 - 1.0)) < 1e-9

    def test_the_fee_comes_off_once(self) -> None:
        pos = Position(outcome_label="YES", price=0.7, size=10.0 / 0.7, side="buy_no")
        assert abs(settle_position(pos, "NO", fee_paid=0.03) - (10.0 / 0.7 - 10.0 - 0.03)) < 1e-9
        assert abs(settle_position(pos, "YES", fee_paid=0.03) - (-10.03)) < 1e-9

    def test_buy_no_is_the_mirror_of_buying_yes_at_the_complement(self) -> None:
        """Buying NO at 0.7 and buying a YES token at 0.7 on the opposite event agree."""
        no = Position(outcome_label="YES", price=0.7, size=2.0, side="buy_no")
        yes_on_the_opposite = Position(outcome_label="NO", price=0.7, size=2.0, side="buy")
        for winning in ("YES", "NO"):
            assert (
                abs(settle_position(no, winning) - settle_position(yes_on_the_opposite, winning))
                < 1e-12
            )


class TestUnknownSide:
    def test_a_side_that_is_not_buy_buy_no_or_sell_is_an_error(self) -> None:
        """A typo must not silently settle as the short-YES formula."""
        pos = Position(outcome_label="YES", price=0.7, size=1.0, side="buy-no")
        try:
            settle_position(pos, winning_label="YES")
        except ValueError as exc:
            assert "buy-no" in str(exc)
        else:
            raise AssertionError("an unknown side was accepted")
