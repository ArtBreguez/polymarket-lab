"""EdgeSignal — structured signal from edge detection layer."""

from __future__ import annotations

import math
from dataclasses import dataclass

from pmlab.core.costs import CostModel
from pmlab.core.edge import quote_edges


@dataclass
class EdgeSignal:
    """A trading signal with computed edge metrics ready for paper-broker ingestion."""

    market_id: str
    city_or_segment: str
    target_date: str  # YYYY-MM-DD
    horizon: str  # "market_open" | "previous_evening" | "morning_of"
    outcome_label: str
    direction: str  # "yes" | "no"
    gamma_price: float
    model_prob: float
    best_edge: float
    yes_edge: float
    no_edge: float
    question: str = ""

    @classmethod
    def from_quote(
        cls,
        *,
        market_id: str,
        city_or_segment: str,
        target_date: str,
        horizon: str,
        outcome_label: str,
        yes_price: float,
        model_prob: float,
        costs: CostModel,
        stake: float,
        directions: tuple[str, ...] = ("yes", "no"),
        question: str = "",
    ) -> EdgeSignal | None:
        """The signal for one outcome label, edges computed as the backtest computes them.

        `pmlab.core.edge.quote_edges` is the one definition of the after-fee edge of each
        side; the backtest's live policy calls it too, so a plugin's scan-edge step that
        builds its signals here trades the strategy the champion was promoted on.

        Args:
            yes_price: The label's YES price (recorded as `gamma_price`, whichever side
                is bought).
            model_prob: The probability the *label occurs* (recorded as `model_prob`,
                whichever side is bought; calibration scores it against the label).
            costs: The champion's cost model.
            stake: The flat stake the broker will trade.
            directions: The sides the strategy may take.

        Returns:
            The signal, or None if no allowed side can be bought (a price of 0, or a
            fill pushed to certainty). `best_edge` can still be negative; deciding
            whether it is worth a trade is the caller's `min_edge`.
        """
        quote = quote_edges(model_prob, yes_price, costs, stake, directions)
        if quote.best_edge == -math.inf:
            return None
        # A side that cannot be bought is -inf inside the policy; on a signal that would be
        # written as `-Infinity`, which strict JSON refuses. It cannot do worse than losing
        # the whole stake, so that is what it reads as.
        floor = -1.0
        return cls(
            market_id=market_id,
            city_or_segment=city_or_segment,
            target_date=target_date,
            horizon=horizon,
            outcome_label=outcome_label,
            direction=quote.direction,
            gamma_price=float(yes_price),
            model_prob=float(model_prob),
            best_edge=quote.best_edge,
            yes_edge=max(quote.yes_edge, floor),
            no_edge=max(quote.no_edge, floor),
            question=question,
        )
