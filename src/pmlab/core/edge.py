"""Edge calculation — domain-agnostic.

Edge is the after-cost expected value of a trade: how much better
our fair probability is than the market price, minus costs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pmlab.core.costs import CostModel

__all__ = ["DEFAULT_MIN_EDGE", "QuoteEdges", "compute_edge", "edge_clears", "quote_edges"]

# `pmlab record-trades --min-edge` default: the smallest best_edge worth a trade. The
# backtest's live policy reads the same constant, so the two cannot drift apart.
DEFAULT_MIN_EDGE = 0.20

# Slack for float round-off when an edge is held against `--min-edge`: 0.6 - 0.5 is
# 0.09999999999999998, and an edge of exactly 0.10 must clear a threshold of 0.10.
_EDGE_TOLERANCE = 1e-9


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


@dataclass(frozen=True)
class QuoteEdges:
    """The edge of each side of one outcome label, and the better of the two.

    An edge is ``-inf`` for a side that cannot be bought (see `quote_edges`).
    `direction` is ``"yes"`` or ``"no"``; a tie goes to YES.
    """

    yes_fill: float
    no_fill: float
    yes_edge: float
    no_edge: float
    best_edge: float
    direction: str


def _side_edge(prob: float, quote: float, costs: CostModel, stake: float) -> tuple[float, float]:
    """(fill, after-fee edge) of buying a token worth `prob` quoted at `quote`."""
    fill = costs.fill_price(quote, stake=stake)
    # A token at 0 is stake / 0 shares, and one at certainty pays at most what it costs:
    # neither is a trade, and a free token must not read as an infinite edge.
    if not 0.0 < fill < 1.0:
        return fill, -math.inf
    return fill, compute_edge(prob, fill, fee_estimate=costs.fee(stake) / stake)


def quote_edges(
    model_prob: float,
    yes_price: float,
    costs: CostModel,
    stake: float,
    directions: tuple[str, ...] = ("yes", "no"),
) -> QuoteEdges:
    """After-fee edge of buying YES and of buying NO on one outcome label.

    This is the one place the edge is computed: the backtest's trade policy and a live
    `EdgeSignal` (`EdgeSignal.from_quote`) both call it, so a champion is promoted on
    the same decision rule that trades.

        yes_edge = p       - fill(yes_price)     - fee_rate
        no_edge  = (1 - p) - fill(1 - yes_price) - fee_rate

    `fill` is the cost model's executable price (slippage and depth included) and
    `fee_rate` is the taker fee per unit of stake.

    Args:
        model_prob: The model's probability that the *label occurs*, whichever side is
            bought (the NO token is worth ``1 - model_prob``).
        yes_price: The label's YES price at decision time.
        costs: The cost model of the backtest or the broker.
        stake: Notional risked, which the cost model's depth term depends on.
        directions: The sides the strategy may take; `best_edge` and `direction` pick
            among these. Both edges are always reported.

    Raises:
        ValueError: `model_prob` or `yes_price` is not a number in [0, 1], or
            `directions` is empty or names a side other than yes/no.
    """
    if not directions or set(directions) - {"yes", "no"}:
        raise ValueError(
            f"directions must be a non-empty subset of ('yes', 'no'), got {directions!r}"
        )
    if not _is_unit_interval(model_prob):
        raise ValueError(f"model_prob must be a number in [0, 1], got {model_prob!r}")
    if not _is_unit_interval(yes_price):
        raise ValueError(f"yes_price must be a number in [0, 1], got {yes_price!r}")
    yes_fill, yes_edge = _side_edge(model_prob, yes_price, costs, stake)
    no_fill, no_edge = _side_edge(1.0 - model_prob, 1.0 - yes_price, costs, stake)
    candidates = {"yes": yes_edge, "no": no_edge}
    # max() keeps the first of equal values, and "yes" is listed first: a tie is YES.
    direction = max((d for d in ("yes", "no") if d in directions), key=lambda d: candidates[d])
    return QuoteEdges(
        yes_fill=yes_fill,
        no_fill=no_fill,
        yes_edge=yes_edge,
        no_edge=no_edge,
        best_edge=candidates[direction],
        direction=direction,
    )


def edge_clears(best_edge: float, min_edge: float) -> bool:
    """Is `best_edge` at least `min_edge`? The one comparison `record-trades` and the backtest share."""
    return best_edge >= min_edge - _EDGE_TOLERANCE


def _is_unit_interval(value: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )
