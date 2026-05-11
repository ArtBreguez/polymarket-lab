"""Generic market domain models — domain-agnostic core primitives.

These models describe any Polymarket market family without embedding
domain-specific logic (no weather, no sports, no crypto here).
Domain plugins extend these via MarketPlugin in pmlab.plugins.base.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any


class OutcomeBin(BaseModel):
    """A single outcome bin for a prediction market.

    Supports:
    - Unbounded bins (binary YES/NO)
    - Half-open numeric ranges (temperature, price)
    - Categorical labels (driver name, country)

    For categorical outcomes, leave lower/upper as None and match by label externally.
    """

    label: str
    lower: float | None = None
    upper: float | None = None
    lower_inclusive: bool = True
    upper_inclusive: bool = True

    def contains(self, value: float) -> bool:
        """Return True if *value* falls inside this bin's numeric range."""
        if self.lower is not None:
            if self.lower_inclusive and value < self.lower:
                return False
            if not self.lower_inclusive and value <= self.lower:
                return False
        if self.upper is not None:
            if self.upper_inclusive and value > self.upper:
                return False
            if not self.upper_inclusive and value >= self.upper:
                return False
        return True


class MarketSpec(BaseModel):
    """Domain-agnostic descriptor for a single Polymarket market.

    Carries the minimum information needed by the core framework —
    market identity, outcome structure, and timing.
    Domain-specific fields (e.g. city, weather station) live in ``metadata``.
    """

    market_id: str
    slug: str
    question: str
    outcome_bins: list[OutcomeBin]
    close_time: str
    market_family: str  # "binary" | "range" | "categorical" | "numeric"

    # Optional enrichment
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)  # domain-specific extras

    def resolve_winning_bin(self, realized_value: float) -> str | None:
        """Return the label of the first bin whose range contains *realized_value*.

        Returns None if no bin matches (gap in coverage or uncovered value).
        For categorical markets, resolve externally by label — do not call this.
        """
        for bin_ in self.outcome_bins:
            if bin_.contains(realized_value):
                return bin_.label
        return None
