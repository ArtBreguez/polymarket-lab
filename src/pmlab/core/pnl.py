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
        price: Entry price per share (0.0–1.0).
        size: Number of shares held (flat_stake / entry_price).
        side: "buy" (long YES) or "sell" (short YES / long NO).
    """

    outcome_label: str
    price: float
    size: float
    side: str  # "buy" | "sell"


def settle_position(position: Position, winning_label: str, fee_paid: float = 0.0) -> float:
    """Compute realized PnL when market resolves.

    Formula (mirrors Polymarket binary settlement):
        payout = 1.0 if position.outcome_label == winning_label else 0.0
        buy:  pnl = (payout - price) * size - fee_paid
        sell: pnl = (price - payout) * size - fee_paid

    Args:
        position: The open position to settle.
        winning_label: The resolved outcome label.
        fee_paid: Total fee already paid at entry (not charged again here).

    Returns:
        Signed PnL in USDC (positive = profit, negative = loss).
    """
    payout = 1.0 if position.outcome_label == winning_label else 0.0
    if position.side == "buy":
        return (payout - position.price) * position.size - fee_paid
    return (position.price - payout) * position.size - fee_paid
