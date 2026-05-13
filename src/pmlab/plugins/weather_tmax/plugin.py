"""WeatherTmax plugin — Polymarket highest-temperature-in-city markets.

This is the reference plugin implementation. It wraps domain logic for
"Highest temperature in [city] on [date]?" markets on Polymarket.

For full feature ingestion and truth resolution, this plugin delegates to
external weather data clients (ECMWF, Wunderground, NOAA). In production it
is expected to be configured with API credentials via environment variables.
For testing, all external calls can be mocked via dependency injection.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pmlab.core.market_spec import MarketSpec, OutcomeBin
from pmlab.plugins.base import MarketPlugin

# Primary filter: question must match "Highest temperature in <city> on <date>?"
_TMAX_RE = re.compile(
    r"highest temperature in .+ on .+?",
    re.IGNORECASE,
)

# Secondary filter: description should reference an authoritative source
_RESOLUTION_HINTS = (
    "wunderground",
    "central weather administration",
    "hong kong observatory",
    "highest temperature recorded",
)

# Regex to parse city and date from question string
_QUESTION_RE = re.compile(
    r"highest temperature in (?P<city>[\w\s,]+?) on (?P<date>.+?)\??$",
    re.IGNORECASE,
)

# Stub feature keys returned when forecast_client is unavailable
_STUB_FEATURES: dict[str, float] = {
    "forecast_temperature_2m_max": 0.0,
    "forecast_temperature_2m_mean": 0.0,
    "forecast_temperature_2m_min": 0.0,
    "forecast_dew_point_2m_mean": 0.0,
    "forecast_relative_humidity_2m_mean": 0.0,
    "forecast_wind_speed_10m_mean": 0.0,
    "forecast_cloud_cover_mean": 0.0,
    "lead_hours": 0.0,
}


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

    # ------------------------------------------------------------------
    # MarketPlugin interface
    # ------------------------------------------------------------------

    def discover_markets(self, **kwargs: Any) -> list[MarketSpec]:
        """Fetch open temperature markets from the Polymarket Gamma API.

        Paginates through the Gamma API until all markets are retrieved,
        then filters by:
          1. Question matches ``highest temperature in ... on ...?`` (regex).
          2. Description references an authoritative source (soft check).
          3. Outcomes contain temperature bins with °C or °F (soft check).
        """
        if self._gamma is None:
            raise RuntimeError(
                "WeatherTmaxPlugin requires a gamma_client. "
                "Pass one at construction or mock for testing."
            )

        limit: int = kwargs.pop("limit", 100)
        offset: int = 0
        all_raw: list[dict[str, Any]] = []

        while True:
            page: list[dict[str, Any]] = self._gamma.fetch_markets(
                offset=offset, limit=limit, **kwargs
            )
            if not page:
                break
            all_raw.extend(page)
            if len(page) < limit:
                break  # reached the last page
            offset += limit

        return [self._build_spec(m) for m in all_raw if self._is_tmax_market(m)]

    def fetch_features(self, spec: MarketSpec, horizon: str, **kwargs: Any) -> dict[str, float]:
        """Return forecast features for (city, target_date, horizon).

        When *forecast_client* is None (e.g. during testing), returns a stub
        dict with the canonical feature keys expected by the model, all set to
        zero.  In production the forecast client populates real values.
        """
        city: str = spec.metadata.get("city", "")
        target_date: str = spec.metadata.get("target_date", "")

        if self._forecast is None:
            return dict(_STUB_FEATURES)

        result: dict[str, float] = dict(
            self._forecast.get_features(city=city, target_date=target_date, horizon=horizon)
        )
        return result

    def fetch_truth(self, spec: MarketSpec, **kwargs: Any) -> float | None:
        """Return the official maximum temperature observation in Celsius, or None."""
        city: str = spec.metadata.get("city", "")
        target_date: str = spec.metadata.get("target_date", "")
        if self._truth is None:
            return None
        result: float | None = self._truth.get_daily_max(city=city, date=target_date)
        return result

    def build_training_row(
        self, spec: MarketSpec, horizon: str, **kwargs: Any
    ) -> dict[str, Any] | None:
        """Assemble one labeled training row.  Returns None if data unavailable.

        The returned dict contains:
          - ``market_id``
          - ``decision_horizon``
          - ``winning_label`` — label of the bin that contains the realized truth
          - ``outcome_label``  — same as winning_label (the resolved outcome bin)
          - ``market_price``   — best market price from metadata
          - All feature columns (prefixed ``forecast_*`` / ``lead_hours``)
        """
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
            "outcome_label": winning_label,
            "market_price": spec.metadata.get("market_price", 0.5),
            **features,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_tmax_market(raw: dict[str, Any]) -> bool:
        """Return True when *raw* looks like a highest-temperature market.

        Filtering strategy (three signals, soft-AND):
          1. Question *must* match the regex (hard requirement).
          2. Description should reference a known resolution source.
          3. Outcomes should contain degree symbols (°C / °F).

        Signals 2 and 3 are only checked when the corresponding field is
        non-empty.  A market that has neither description nor outcome data
        still passes on regex alone, to avoid incorrectly discarding sparse
        API responses during discovery.
        """
        question: str = raw.get("question", "")
        if not _TMAX_RE.search(question):
            return False

        # Collect secondary signals only when data is available
        description: str = (raw.get("description") or "").lower()
        tokens: list[Any] = raw.get("tokens") or []
        outcomes_raw: list[Any] = raw.get("outcomes") or []

        # Build outcomes string from whichever field is populated
        if tokens:
            outcomes_str = " ".join(
                str(t.get("outcome", t.get("label", t.get("name", ""))))
                if isinstance(t, dict)
                else str(t)
                for t in tokens
            )
        else:
            outcomes_str = " ".join(str(o) for o in outcomes_raw)

        has_desc_hint: bool | None = (
            any(hint in description for hint in _RESOLUTION_HINTS) if description else None
        )
        has_temp_bins: bool | None = (
            ("°C" in outcomes_str or "°F" in outcomes_str) if outcomes_str.strip() else None
        )

        # Reject only when both secondary signals are explicitly negative
        return not (has_desc_hint is False and has_temp_bins is False)

    def _build_spec(self, raw: dict[str, Any]) -> MarketSpec:
        """Convert a raw Gamma API market dict to a MarketSpec.

        Handles both the legacy ``tokens`` format and the canonical Gamma API
        format with ``outcomes`` / ``outcomePrices`` / ``clobTokenIds``.
        """
        market_id: str = str(raw.get("id", raw.get("market_id", "")))
        slug: str = raw.get("slug", market_id)
        question: str = raw.get("question", "")
        close_time: str = raw.get("endDate", raw.get("close_time", ""))

        # --- Parse city and date from question ---
        q_match = _QUESTION_RE.match(question)
        city: str = q_match.group("city").strip() if q_match else raw.get("city", "")
        target_date: str = q_match.group("date").strip() if q_match else raw.get("target_date", "")

        # --- Parse token IDs ---
        clob_raw = raw.get("clobTokenIds", [])
        if isinstance(clob_raw, str):
            try:
                token_ids: list[str] = json.loads(clob_raw)
            except (json.JSONDecodeError, ValueError):
                token_ids = []
        else:
            token_ids = list(clob_raw) if clob_raw else []

        # --- Parse outcomes and prices ---
        # Prefer the canonical Gamma API fields; fall back to legacy ``tokens``
        outcome_labels: list[str] = []
        outcome_prices: list[float] = []

        outcomes_field = raw.get("outcomes")
        prices_field = raw.get("outcomePrices")

        if outcomes_field:
            # outcomes may be a JSON string or a Python list
            if isinstance(outcomes_field, str):
                try:
                    outcome_labels = json.loads(outcomes_field)
                except (json.JSONDecodeError, ValueError):
                    outcome_labels = []
            else:
                outcome_labels = [str(o) for o in outcomes_field]

            if isinstance(prices_field, str):
                try:
                    raw_prices = json.loads(prices_field)
                    outcome_prices = [float(p) for p in raw_prices]
                except (json.JSONDecodeError, ValueError, TypeError):
                    outcome_prices = [0.5] * len(outcome_labels)
            elif prices_field:
                try:
                    outcome_prices = [float(p) for p in prices_field]
                except (TypeError, ValueError):
                    outcome_prices = [0.5] * len(outcome_labels)
            else:
                outcome_prices = [0.5] * len(outcome_labels)
        else:
            # Fall back to legacy ``tokens`` list
            tokens: list[dict[str, Any]] = raw.get("tokens") or []
            for t in tokens:
                if isinstance(t, dict):
                    lbl = str(t.get("outcome", t.get("label", t.get("name", ""))))
                    raw_price = t.get("price", t.get("outcomePrices", 0.5))
                    price = float(raw_price) if raw_price is not None else 0.5
                else:
                    lbl = str(t)
                    price = 0.5
                outcome_labels.append(lbl)
                outcome_prices.append(price)

        # Pad prices if lengths differ
        if len(outcome_prices) < len(outcome_labels):
            outcome_prices += [0.5] * (len(outcome_labels) - len(outcome_prices))

        # --- Determine unit ---
        all_outcomes_str = " ".join(outcome_labels)
        unit: str = "C" if "°C" in all_outcomes_str else "F"

        # --- Build outcome schema and OutcomeBin objects ---
        outcome_schema: list[dict[str, Any]] = []
        outcome_bins: list[OutcomeBin] = []

        for label, price in zip(outcome_labels, outcome_prices, strict=False):
            outcome_schema.append({"label": label, "price": price})
            lower, upper = _parse_temp_bounds(label)
            outcome_bins.append(OutcomeBin(label=label, lower=lower, upper=upper))

        # Best market price: highest single-outcome price
        market_price: float = max(outcome_prices) if outcome_prices else 0.5

        return MarketSpec(
            market_id=market_id,
            slug=slug,
            question=question,
            outcome_bins=outcome_bins,
            close_time=close_time,
            market_family="range",
            tags=["weather", "temperature"],
            metadata={
                "city": city,
                "target_date": target_date,
                "token_ids": token_ids,
                "outcome_schema": outcome_schema,
                "market_price": market_price,
                "unit": unit,
                "question": question,
            },
        )


# ---------------------------------------------------------------------------
# Module-level helpers (not part of the plugin class)
# ---------------------------------------------------------------------------


def _parse_temp_bounds(label: str) -> tuple[float | None, float | None]:
    """Parse numeric temperature bounds from a bin label string.  Best-effort."""
    clean = label.replace("°C", "").replace("°F", "").strip()

    # Range: "28-30" or "28–30"
    m = re.match(r"([\d.]+)\s*[-\u2013to]+\s*([\d.]+)", clean)
    if m:
        return float(m.group(1)), float(m.group(2))

    # Greater-than: ">35" or ">=35"
    m = re.match(r">=?\s*([\d.]+)", clean)
    if m:
        return float(m.group(1)), None

    # Less-than: "<20" or "<=20"
    m = re.match(r"<=?\s*([\d.]+)", clean)
    if m:
        return None, float(m.group(1))

    # Single value: "30"
    m = re.match(r"^([\d.]+)$", clean)
    if m:
        v = float(m.group(1))
        return v - 0.5, v + 0.5

    return None, None
