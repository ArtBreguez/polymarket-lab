"""Position sizing utilities: flat stake and fractional Kelly."""

from __future__ import annotations

import math

from pmlab.core.costs import CostModel

__all__ = ["flat_stake_size", "kelly_fraction", "kelly_stake_size"]


def _require_range(name: str, value: float, lo: float, hi: float) -> None:
    """``ValueError`` unless ``value`` is finite and ``lo <= value <= hi``."""
    if not (math.isfinite(value) and lo <= value <= hi):
        raise ValueError(f"{name} must be a finite number in [{lo}, {hi}], got {value!r}")


def _require_costs(costs: object) -> CostModel | None:
    if costs is not None and not isinstance(costs, CostModel):
        raise TypeError(f"costs must be a CostModel or None, got {type(costs).__name__}")
    return costs


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
    *,
    costs: CostModel | None = None,
) -> float:
    """Compute the (fractional) Kelly fraction of bankroll to wager.

    Uses the binary-outcome Kelly formula:
        f* = (b * p - q) / b
    where b = net_odds = (1 - cost) / cost, p = win_prob, q = 1 - win_prob, and ``cost`` is what
    one share-worth of payoff costs: ``entry_price``, or with ``costs`` the entry price **plus the
    taker fee**, ``entry_price x (1 + fee per unit of stake)`` (for the Polymarket model
    ``entry_price x (1 + rate x (1 - entry_price))``). The fraction is then of the bankroll that
    leaves it, the stake and its fee together, which is what the payoff above is net of.

    Args:
        win_prob: Model-estimated probability the bet wins (0-1).
        entry_price: Cost per share (0-1), i.e. market-implied probability.
        fraction: Fraction of full-Kelly to use (default 0.25 = quarter-Kelly).
        costs: Keyword-only. A `CostModel` whose fee the payoff is net of, taken at
            ``entry_price`` on a unit stake. Omit it for the gross payoff, as before 1.5.0 (that
            ignores the fee, which on Polymarket is 2.5% of the stake at 0.50, so it oversizes).
            Slippage is not applied: pass the price you expect to pay as ``entry_price``.

    Returns:
        Fraction of bankroll to wager (0-1, floored at 0). 0 when the fee leaves no edge.

    Raises:
        ValueError: If ``win_prob`` or ``fraction`` is out of range.
        TypeError: If ``costs`` is not a `CostModel` (or ``None``).
    """
    _require_range("win_prob", win_prob, 0.0, 1.0)
    _require_range("fraction", fraction, 0.0, math.inf)
    costs = _require_costs(costs)
    if entry_price <= 0.0 or entry_price >= 1.0:
        return 0.0
    cost = entry_price if costs is None else entry_price * (1.0 + costs.fee(1.0, entry_price))
    if cost >= 1.0:  # a fee that takes the cost of a share to its payoff leaves no profit to size
        return 0.0
    b = (1.0 - cost) / cost  # net odds
    q = 1.0 - win_prob
    full_kelly = (b * win_prob - q) / b
    return max(0.0, fraction * full_kelly)


def kelly_stake_size(
    win_prob: float,
    entry_price: float,
    bankroll: float,
    fraction: float = 0.25,
    max_exposure: float = 0.05,
    *,
    costs: CostModel | None = None,
) -> float:
    """Compute Kelly-sized USDC stake.

    Args:
        win_prob: Model probability the bet wins.
        entry_price: Cost per share (0-1).
        bankroll: Total USDC bankroll available.
        fraction: Fractional Kelly multiplier (default 0.25).
        max_exposure: Max fraction of bankroll per trade (default 5%).
        costs: Keyword-only. A `CostModel` whose taker fee the sizing is net of (see
            `kelly_fraction`). The exposure cap is on what leaves the bankroll, so the stake
            returned is the one whose stake **plus fee** is the capped fraction: ``stake + fee(stake)
            <= max_exposure x bankroll``. Omit it and the stake is the capped fraction itself, with
            no room for a fee.

    Returns:
        USDC stake amount (>= 0). The fee is paid on top of it when ``costs`` is given.

    Raises:
        ValueError: If ``bankroll``, ``max_exposure``, ``win_prob`` or ``fraction`` is out of range.
        TypeError: If ``costs`` is not a `CostModel` (or ``None``).
    """
    _require_range("bankroll", bankroll, 0.0, math.inf)
    _require_range("max_exposure", max_exposure, 0.0, 1.0)
    kf = kelly_fraction(win_prob, entry_price, fraction, costs=costs)
    capped = min(kf, max_exposure)
    outlay = capped * bankroll
    if costs is None or outlay == 0.0:
        return outlay
    # What leaves the bankroll is the stake and its fee: `stake x (1 + fee per unit of stake)`.
    return outlay / (1.0 + costs.fee(1.0, entry_price))
