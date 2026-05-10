"""Position sizing utilities."""

from __future__ import annotations


def flat_stake_size(flat_stake: float, entry_price: float) -> float:
    """Compute share count for a flat-stake bet.

    Buys flat_stake / entry_price shares so the maximum loss is flat_stake
    (when the trade goes to 0) and the gross profit at win is
    flat_stake * (1 - entry_price) / entry_price.

    Args:
        flat_stake: Fixed USDC amount to risk per trade.
        entry_price: Price per share at execution (0–1).

    Returns:
        Number of shares to buy.
    """
    return flat_stake / max(entry_price, 1e-9)
