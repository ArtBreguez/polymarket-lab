"""EdgeSignal — structured signal from edge detection layer."""

from __future__ import annotations

from dataclasses import dataclass


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
