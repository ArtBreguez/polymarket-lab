"""SportsF1Plugin — Formula 1 race outcome markets on Polymarket.

Handles markets like "Who wins the [GP] race?", "Will [Driver] score points?",
"Which team wins the Constructors Championship?".

Market family: "sports_f1"

Key differences from WeatherTmaxPlugin:
- Truth is categorical (driver/team name string), not numeric
- Outcome bins have no numeric bounds — matched by label only
- Features come from qualifying telemetry, lap time gaps, historical performance
- Markets can have 2–20+ outcome bins
"""

from __future__ import annotations

from typing import Any

from pmlab.core.market_spec import MarketSpec, OutcomeBin
from pmlab.plugins.base import MarketPlugin


class SportsF1Plugin(MarketPlugin):
    """Plugin for Polymarket Formula 1 race and championship markets."""

    family = "sports_f1"

    def __init__(
        self,
        gamma_client: Any | None = None,
        telemetry_client: Any | None = None,
        results_client: Any | None = None,
    ) -> None:
        self._gamma = gamma_client
        self._telemetry = telemetry_client
        self._results = results_client

    def discover_markets(self, **kwargs: Any) -> list[MarketSpec]:
        if self._gamma is None:
            raise RuntimeError("SportsF1Plugin requires a gamma_client.")
        raw_markets = self._gamma.fetch_markets(tag="f1", **kwargs)
        return [self._build_spec(m) for m in raw_markets if self._is_f1_market(m)]

    def fetch_features(
        self, spec: MarketSpec, horizon: str, **kwargs: Any
    ) -> dict[str, float]:
        """Return session-based features for this F1 market at *horizon*."""
        if self._telemetry is None:
            return {"quali_gap_s": 0.0, "historical_win_rate": 0.1}
        gp: str = spec.metadata.get("gp", "")
        return self._telemetry.get_features(gp=gp, horizon=horizon)

    def fetch_truth(self, spec: MarketSpec, **kwargs: Any) -> str | None:
        """Return the winning driver/team label, or None if race not completed."""
        if self._results is None:
            return None
        gp: str = spec.metadata.get("gp", "")
        market_type: str = spec.metadata.get("market_type", "race_winner")
        return self._results.get_winner(gp=gp, market_type=market_type)

    def build_training_row(
        self, spec: MarketSpec, horizon: str, **kwargs: Any
    ) -> dict[str, Any] | None:
        features = self.fetch_features(spec, horizon)
        truth = self.fetch_truth(spec)
        if truth is None:
            return None
        # For categorical: winning_label is the string truth directly
        winning_label = truth
        return {
            "market_id": spec.market_id,
            "decision_horizon": horizon,
            "winning_label": winning_label,
            "market_price": spec.metadata.get("market_price", 0.5),
            **features,
        }

    # ------------------------------------------------------------------

    @staticmethod
    def _is_f1_market(raw: dict[str, Any]) -> bool:
        q = raw.get("question", "").lower()
        return any(kw in q for kw in ("f1", "formula 1", "grand prix", "gp winner"))

    def _build_spec(self, raw: dict[str, Any]) -> MarketSpec:
        tokens: list[dict] = raw.get("tokens", []) or raw.get("outcomes", [])
        bins = [
            OutcomeBin(label=str(t.get("outcome", t.get("label", t.get("name", "?")))))
            for t in tokens
        ]
        return MarketSpec(
            market_id=raw.get("id", ""),
            slug=raw.get("slug", ""),
            question=raw.get("question", ""),
            outcome_bins=bins,
            close_time=raw.get("endDate", ""),
            market_family="categorical",
            tags=["sports", "f1"],
            metadata={
                "gp": raw.get("gp", ""),
                "market_type": raw.get("market_type", "race_winner"),
            },
        )
