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
            - "market_price": float  (the YES price of ``outcome_label`` at decision time)
            - "market_id": str
            - "decision_horizon": str
            - "outcome_label": str  (the bin this row represents)

        A market with several outcome bins is **several rows**: see
        :meth:`build_training_rows`, which is what a panel should be built from. This
        method describes one (market, bin, decision) and is the only one a plugin has to
        write.

        Args:
            spec: The market.
            horizon: Decision point identifier.

        Returns:
            Dict for one training row, or None if data unavailable.
        """
        ...

    def build_training_rows(
        self, spec: MarketSpec, horizon: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Assemble every labeled training row of one (market, decision): one per outcome bin.

        The backtest chooses between a market's bins and between buying and selling each,
        so it needs to see all of them, each at the price *that bin* had when the decision
        was taken, labelled with whether *that bin* won (``outcome_label`` against
        ``winning_label``). A panel holding only the bin that won makes every YES a
        winner and every NO a loser, and its backtest reports an edge that is not there.

        The default wraps :meth:`build_training_row`, so a plugin written for the
        singular form keeps working: one row, or none when that returns None. Override
        this for a market with several bins. Then every row must carry:

            - ``market_id``, ``decision_date`` (YYYY-MM-DD, the UTC date of the decision),
              ``decision_horizon``
            - ``outcome_label`` (the bin this row is) and ``winning_label``
            - ``market_price`` that bin's YES price, observed strictly before the decision
            - feature columns (``feature_*``)

        and should say where the price came from and when it was observed
        (``price_source``, ``price_ts``) against the decision (``decision_ts``), which
        :func:`pmlab.data.check_price_timing` then enforces. A bin with no usable price
        gets no row; never fill one in.

        Args:
            spec: The market.
            horizon: Decision point identifier.

        Returns:
            The rows, possibly empty.
        """
        row = self.build_training_row(spec, horizon, **kwargs)
        return [] if row is None else [row]

    def is_truth_final(self, spec: MarketSpec, **kwargs: Any) -> bool:
        """Return True if the truth for *spec* is final and safe to settle.

        Default: True whenever fetch_truth() returns non-None.
        Override if your data source has a lag before finalization
        (e.g. weather stations may have preliminary vs confirmed readings).
        """
        return self.fetch_truth(spec, **kwargs) is not None
