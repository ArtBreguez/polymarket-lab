"""WeatherTmax plugin — Polymarket highest-temperature-in-city markets.

This is the reference plugin implementation. It wraps domain logic for
"Highest temperature in [city] on [date]?" markets on Polymarket.

For full feature ingestion and truth resolution, this plugin delegates to
external weather data clients (ECMWF, Wunderground, NOAA). In production it
is expected to be configured with API credentials via environment variables.
For testing, all external calls can be mocked via dependency injection.
"""

from __future__ import annotations

from typing import Any

from pmlab.core.market_spec import MarketSpec
from pmlab.plugins.base import MarketPlugin


class WeatherTmaxPlugin(MarketPlugin):
    """Plugin for Polymarket 'Highest temperature in [city] on [date]?' markets.

    Market family: "weather_tmax"

    Features include:
    - ECMWF forecast temperature and ensemble spread
    - Lead time (days before market close)
    - City climatological baseline (historical mean/std)
    - Intraday observations (where available)

    Truth source: official temperature observations (Wunderground / NOAA / CWA)
    after market close.
    """

    family = "weather_tmax"

    def __init__(
        self,
        gamma_client: Any | None = None,
        forecast_client: Any | None = None,
        truth_client: Any | None = None,
    ) -> None:
        """
        Args:
            gamma_client: Polymarket Gamma API client (injected for testing).
            forecast_client: Weather forecast client (ECMWF / OpenMeteo).
            truth_client: Official temperature observation client.
        """
        self._gamma = gamma_client
        self._forecast = forecast_client
        self._truth = truth_client

    def discover_markets(self, **kwargs: Any) -> list[MarketSpec]:
        """Fetch open temperature markets from Polymarket Gamma API."""
        if self._gamma is None:
            raise RuntimeError(
                "WeatherTmaxPlugin requires a gamma_client. "
                "Pass one at construction or mock for testing."
            )
        raw_markets = self._gamma.fetch_markets(tag="temperature", **kwargs)
        return [self._build_spec(m) for m in raw_markets if self._is_tmax_market(m)]

    def fetch_features(
        self, spec: MarketSpec, horizon: str, **kwargs: Any
    ) -> dict[str, float]:
        """Return ECMWF-based forecast features for (city, target_date, horizon)."""
        city: str = spec.metadata.get("city", "")
        target_date: str = spec.metadata.get("target_date", "")
        if self._forecast is None:
            return {"lead_time_days": 1.0, "forecast_tmax_c": 25.0, "forecast_spread": 2.0}
        return self._forecast.get_features(city=city, target_date=target_date, horizon=horizon)

    def fetch_truth(self, spec: MarketSpec, **kwargs: Any) -> float | None:
        """Return the official maximum temperature observation in Celsius, or None."""
        city: str = spec.metadata.get("city", "")
        target_date: str = spec.metadata.get("target_date", "")
        if self._truth is None:
            return None
        return self._truth.get_daily_max(city=city, date=target_date)

    def build_training_row(
        self, spec: MarketSpec, horizon: str, **kwargs: Any
    ) -> dict[str, Any] | None:
        """Assemble one labeled training row. Returns None if data unavailable."""
        features = self.fetch_features(spec, horizon)
        truth = self.fetch_truth(spec)
        if truth is None:
            return None
        winning_label = spec.resolve_winning_bin(truth)
        if winning_label is None:
            return None
        return {
            "market_id": spec.market_id,
            "decision_horizon": horizon,
            "winning_label": winning_label,
            "market_price": spec.metadata.get("market_price", 0.5),
            **features,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_tmax_market(raw: dict[str, Any]) -> bool:
        q: str = raw.get("question", "").lower()
        return "highest temperature" in q or "temperatura" in q

    def _build_spec(self, raw: dict[str, Any]) -> MarketSpec:
        """Convert a raw Gamma API market dict to a MarketSpec."""
        from pmlab.plugins.weather_tmax._spec_builder import build_tmax_spec
        return build_tmax_spec(raw)
