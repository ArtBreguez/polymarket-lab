"""Edge calculation — domain-agnostic.

Edge is the after-cost expected value of a trade: how much better
our fair probability is than the market price, minus costs.
"""

from __future__ import annotations


def compute_edge(
    fair_probability: float,
    executable_price: float,
    fee_estimate: float = 0.0,
    slippage_estimate: float = 0.0,
) -> float:
    """Compute after-cost probability edge for a binary outcome bet.

    Args:
        fair_probability: Our model's estimated probability for this outcome (0–1).
        executable_price: Market price we actually pay per share (0–1).
        fee_estimate: Expected taker fee on this trade.
        slippage_estimate: Expected price impact / slippage.

    Returns:
        Edge value. Positive = expected profit. Negative = expected loss.
    """
    return fair_probability - executable_price - fee_estimate - slippage_estimate
