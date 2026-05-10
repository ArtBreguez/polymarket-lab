"""MarketPlugin — abstract interface every domain plugin must implement.

A plugin encapsulates all domain-specific knowledge for one Polymarket market family:
how to discover markets, how to generate features, and how to fetch the realized truth.

The core framework (backtest, paper trading, settlement) is plugin-agnostic —
it calls only the four abstract methods defined here.

To add a new market family:
1. Subclass MarketPlugin
2. Set family = "my_family"
3. Implement all four abstract methods
4. Register with PluginRegistry

See docs/plugin-authoring.md for the full guide.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pmlab.core.market_spec import MarketSpec


class MarketPlugin(ABC):
    """Abstract base class for all pmlab market family plugins."""

    family: str  # unique identifier, e.g. "weather_tmax", "sports_f1"

    @abstractmethod
    def discover_markets(self, **kwargs: Any) -> list[MarketSpec]:
        """Fetch currently open markets for this family from Polymarket.

        Returns:
            List of MarketSpec objects for tradeable markets.
        """
        ...

    @abstractmethod
    def fetch_features(self, spec: MarketSpec, horizon: str, **kwargs: Any) -> dict[str, float]:
        """Return a flat feature dict for a single (market, decision horizon) pair.

        Features must be numeric and named consistently across calls —
        the modeling layer will assemble a DataFrame from many such dicts.

        Args:
            spec: The market to generate features for.
            horizon: Decision point identifier, e.g. "market_open", "previous_evening".

        Returns:
            Dict mapping feature name → float value.
        """
        ...

    @abstractmethod
    def fetch_truth(self, spec: MarketSpec, **kwargs: Any) -> float | str | None:
        """Return the realized outcome value for a resolved market.

        For range/numeric markets: return a float (e.g. 31.2 for temperature).
        For categorical markets: return the winning label string (e.g. "Verstappen").
        For unresolved markets: return None.

        Args:
            spec: The market whose outcome to fetch.

        Returns:
            Realized outcome or None if not yet resolved.
        """
        ...

    @abstractmethod
    def build_training_row(
        self, spec: MarketSpec, horizon: str, **kwargs: Any
    ) -> dict[str, Any] | None:
        """Assemble a single labeled training row for model training.

        Combines fetch_features() output with target labels and market prices.
        Return None if required data is unavailable for this (market, horizon).

        The returned dict should include at minimum:
            - All feature columns (numeric)
            - "winning_label": str  (resolved outcome label)
            - "market_price": float  (historical CLOB mid at decision time)
            - "market_id": str
            - "decision_horizon": str
            - "outcome_label": str  (the bin this row represents)

        Args:
            spec: The market.
            horizon: Decision point identifier.

        Returns:
            Dict for one training row, or None if data unavailable.
        """
        ...

    def is_truth_final(self, spec: MarketSpec, **kwargs: Any) -> bool:
        """Return True if the truth for *spec* is final and safe to settle.

        Default: True whenever fetch_truth() returns non-None.
        Override if your data source has a lag before finalization
        (e.g. weather stations may have preliminary vs confirmed readings).
        """
        return self.fetch_truth(spec, **kwargs) is not None
