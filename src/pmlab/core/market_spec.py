"""Generic market domain models — domain-agnostic core primitives.

These models describe any Polymarket market family without embedding
domain-specific logic (no weather, no sports, no crypto here).
Domain plugins extend these via MarketPlugin in pmlab.plugins.base.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class OutcomeBin(BaseModel):
    """A single outcome bin for a prediction market.

    Supports:
    - Unbounded bins (binary YES/NO)
    - Half-open numeric ranges (temperature, price)
    - Categorical labels (driver name, country)

    For categorical outcomes, leave lower/upper as None and match by label externally.

    Raises:
        ValueError: (a pydantic ``ValidationError``) if a bound is NaN, which compares False
            with everything and so accepted every value, or ``lower`` is above ``upper``, a bin
            that can never contain anything.
    """

    label: str
    lower: float | None = None
    upper: float | None = None
    lower_inclusive: bool = True
    upper_inclusive: bool = True

    @model_validator(mode="after")
    def _check_bounds(self) -> OutcomeBin:
        for name in ("lower", "upper"):
            bound = getattr(self, name)
            if bound is not None and bound != bound:
                raise ValueError(f"{name} must be a number, not NaN")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError(
                f"lower ({self.lower}) must not be above upper ({self.upper}): "
                f"bin {self.label!r} could never contain a value"
            )
        return self

    def contains(self, value: float) -> bool:
        """Return True if *value* falls inside this bin's numeric range.

        NaN is in no bin: every comparison with it is False, so without this check no bound
        would reject it and it would land in the first bin (#52).
        """
        if value != value:
            return False
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

    Raises:
        ValueError: (a pydantic ``ValidationError``) if two bins share a label, or
            ``close_time`` is neither blank nor an ISO-8601 date or timestamp: the backtest reads
            it to refuse a decision taken after the close, and skips a time it cannot read.

    An empty ``outcome_bins`` and a blank ``close_time`` are accepted on purpose: a sparse
    Gamma payload (no outcomes, no ``endDate``) is not discarded during discovery, and a spec
    without bins resolves to no winner. Plugins that need more check it themselves.
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

    @field_validator("outcome_bins")
    @classmethod
    def _labels_are_unique(cls, bins: list[OutcomeBin]) -> list[OutcomeBin]:
        labels = [b.label for b in bins]
        repeated = sorted({label for label in labels if labels.count(label) > 1})
        if repeated:
            raise ValueError(
                f"duplicate outcome bin label(s) {repeated}: a label must name one bin, or a "
                "winner and a price cannot be matched to it"
            )
        return bins

    @field_validator("close_time")
    @classmethod
    def _close_time_is_readable(cls, value: str) -> str:
        if value.strip():
            try:
                datetime.fromisoformat(value.strip())
            except ValueError:
                raise ValueError(
                    "close_time must be an ISO-8601 date or timestamp "
                    f"(or blank when unknown), got {value!r}"
                ) from None
        return value

    def resolve_winning_bin(self, realized_value: float) -> str | None:
        """Return the label of the first bin whose range contains *realized_value*.

        Returns None if no bin matches (gap in coverage or uncovered value).
        For categorical markets, resolve externally by label — do not call this.
        """
        for bin_ in self.outcome_bins:
            if bin_.contains(realized_value):
                return bin_.label
        return None
