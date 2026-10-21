"""PnL accounting for binary outcome token positions.

Matches the settlement logic used in polymarket-tmax-lab (pmtmax.backtest.pnl).
Works for any Polymarket market family — weather, sports, crypto.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Position:
    """An open binary outcome token position.

    Attributes:
        outcome_label: The outcome bin label this position is on (e.g. "YES", "30°C", "Verstappen").
        price: Entry price per share (0.0–1.0), of the token actually held: the
            label's price for "buy", the NO token's price for "buy_no", and the
            label's (YES) price for "sell".
        size: Number of shares held (flat_stake / entry_price).
        side: "buy" (long the label's YES token), "buy_no" (long its NO token, which
            pays 1 when the label does NOT win) or "sell" (short the YES token; the
            same payoff as "buy_no" only when `price` is the YES price).
    """

    outcome_label: str
    price: float
    size: float
    side: str  # "buy" | "buy_no" | "sell"


def settle_position(position: Position, winning_label: str, fee_paid: float = 0.0) -> float:
    """Compute realized PnL when market resolves.

    Formula (mirrors Polymarket binary settlement):
        payout = 1.0 if position.outcome_label == winning_label else 0.0
        buy:    pnl = (payout - price) * size - fee_paid
        buy_no: pnl = ((1 - payout) - price) * size - fee_paid
        sell:   pnl = (price - payout) * size - fee_paid

    Args:
        position: The open position to settle.
        winning_label: The resolved outcome label.
        fee_paid: Total fee already paid at entry (not charged again here).

    Returns:
        Signed PnL in USDC (positive = profit, negative = loss).

    Raises:
        ValueError: If ``position.side`` is not "buy", "buy_no" or "sell". An
            unrecognised side used to fall through to the short-YES formula.
    """
    payout = 1.0 if position.outcome_label == winning_label else 0.0
    if position.side == "buy":
        return (payout - position.price) * position.size - fee_paid
    if position.side == "buy_no":
        return ((1.0 - payout) - position.price) * position.size - fee_paid
    if position.side == "sell":
        return (position.price - payout) * position.size - fee_paid
    raise ValueError(f"unknown side {position.side!r}; expected 'buy', 'buy_no' or 'sell'")
