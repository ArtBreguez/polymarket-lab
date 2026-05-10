"""Position sizing utilities: flat stake and fractional Kelly."""
from __future__ import annotations

__all__ = ["flat_stake_size", "kelly_fraction", "kelly_stake_size"]


def flat_stake_size(flat_stake: float, entry_price: float) -> float:
    """Compute share count for a flat-stake bet.

    Buys flat_stake / entry_price shares so the maximum loss is flat_stake
    (when the trade goes to 0) and the gross profit at win is
    flat_stake * (1 - entry_price) / entry_price.

    Args:
        flat_stake: Fixed USDC amount to risk per trade.
        entry_price: Price per share at execution (0-1).

    Returns:
        Number of shares to buy.
    """
    return flat_stake / max(entry_price, 1e-9)


def kelly_fraction(
    win_prob: float,
    entry_price: float,
    fraction: float = 0.25,
) -> float:
    """Compute the (fractional) Kelly fraction of bankroll to wager.

    Uses the binary-outcome Kelly formula:
        f* = (b * p - q) / b
    where b = net_odds = (1 - entry_price) / entry_price, p = win_prob, q = 1 - win_prob.

    Args:
        win_prob: Model-estimated probability the bet wins (0-1).
        entry_price: Cost per share (0-1), i.e. market-implied probability.
        fraction: Fraction of full-Kelly to use (default 0.25 = quarter-Kelly).

    Returns:
        Fraction of bankroll to wager (0-1, floored at 0).
    """
    if entry_price <= 0.0 or entry_price >= 1.0:
        return 0.0
    b = (1.0 - entry_price) / entry_price  # net odds
    q = 1.0 - win_prob
    full_kelly = (b * win_prob - q) / b
    return max(0.0, fraction * full_kelly)


def kelly_stake_size(
    win_prob: float,
    entry_price: float,
    bankroll: float,
    fraction: float = 0.25,
    max_exposure: float = 0.05,
) -> float:
    """Compute Kelly-sized USDC stake.

    Args:
        win_prob: Model probability the bet wins.
        entry_price: Cost per share (0-1).
        bankroll: Total USDC bankroll available.
        fraction: Fractional Kelly multiplier (default 0.25).
        max_exposure: Max fraction of bankroll per trade (default 5%).

    Returns:
        USDC stake amount (>= 0).
    """
    kf = kelly_fraction(win_prob, entry_price, fraction)
    capped = min(kf, max_exposure)
    return capped * bankroll
