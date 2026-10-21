"""WeatherTmax plugin — Polymarket highest-temperature-in-city markets.

This is the reference plugin implementation. It wraps domain logic for
"Highest temperature in [city] on [date]?" markets on Polymarket.

For full feature ingestion and truth resolution, this plugin delegates to
external weather data clients (ECMWF, Wunderground, NOAA). In production it
is expected to be configured with API credentials via environment variables.
For testing, all external calls can be mocked via dependency injection.

Training rows are **one per temperature bin** (:meth:`WeatherTmaxPlugin.build_training_rows`),
each at that bin's own price observed before the decision. See the method for the contract.
"""

from __future__ import annotations

import json
import logging
import math
import re
import warnings
from collections import Counter
from collections.abc import Callable, MutableMapping
from datetime import UTC, datetime
from typing import Any

from pmlab.core.market_spec import MarketSpec, OutcomeBin
from pmlab.plugins.base import MarketPlugin

_logger = logging.getLogger(__name__)

# Why a bin, or a whole market, contributed no training row. Counted in the ``exclusions``
# mapping a caller passes to ``build_training_rows`` (and logged when it passes none).
PRICE_MISSING = "price_missing"  # the spec holds no price for the bin
PRICE_INVALID = "price_invalid"  # not a number, not finite, or outside [0, 1]
PRICE_DEGENERATE = "price_degenerate"  # exactly 0 or 1: a settlement, not a quote to trade at
PRICE_UNTIMED = "price_untimed"  # no readable observation time: cannot be shown to be in time
PRICE_AFTER_CUTOFF = "price_after_cutoff"  # observed at or after the decision
MARKET_UNRESOLVED = "market_unresolved"  # no truth yet
MARKET_TRUTH_UNBINNED = "market_truth_unbinned"  # the truth falls in none of the bins
MARKET_NO_USABLE_PRICE = "market_no_usable_price"  # resolved, but every bin was excluded
EXCLUSION_REASONS = (
    PRICE_MISSING,
    PRICE_INVALID,
    PRICE_DEGENERATE,
    PRICE_UNTIMED,
    PRICE_AFTER_CUTOFF,
    MARKET_UNRESOLVED,
    MARKET_TRUTH_UNBINNED,
    MARKET_NO_USABLE_PRICE,
)

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
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """
        Args:
            gamma_client: Polymarket Gamma API client (injected for testing).
            forecast_client: Weather forecast client (ECMWF / OpenMeteo).
            truth_client: Official temperature observation client.
            clock: Returns "now"; stamps when discovery observed the prices (naive is
                taken as UTC). Defaults to the real UTC clock; inject it in tests.
        """
        self._gamma = gamma_client
        self._forecast = forecast_client
        self._truth = truth_client
        self._clock = clock or (lambda: datetime.now(UTC))

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

        Every spec records when its prices were seen (``metadata["price_observed_at"]``,
        one clock reading per call) and that they came from discovery
        (``metadata["price_source"]``): a training row may only use a price observed
        before its decision, and cannot tell that without a time.
        """
        if self._gamma is None:
            raise RuntimeError(
                "WeatherTmaxPlugin requires a gamma_client. "
                "Pass one at construction or mock for testing."
            )

        page_size: int = kwargs.pop("limit", 100)
        max_results: int = kwargs.pop("max_results", 1000)

        # Use the client's offset-based pagination helper. (Calling
        # fetch_markets(offset=...) directly is wrong — the single-page
        # fetch_markets() has no offset parameter and raises TypeError.)
        all_raw: list[dict[str, Any]] = self._gamma.fetch_markets_all(
            page_size=page_size, max_results=max_results, **kwargs
        )

        observed_at = _iso_utc(self._clock())
        return [
            self._build_spec(m, observed_at=observed_at) for m in all_raw if self._is_tmax_market(m)
        ]

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

    def build_training_rows(
        self, spec: MarketSpec, horizon: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Assemble the labeled training rows of one market: **one per temperature bin**.

        Each row describes one bin at the moment of the decision:

          - ``market_id``, ``decision_horizon``, ``decision_date`` (UTC, YYYY-MM-DD)
          - ``outcome_label`` — *this* bin; ``winning_label`` — the bin the truth fell in
          - ``market_price`` — *this* bin's YES price
          - ``price_source`` and ``price_ts`` — where that price came from and when it
            was observed; ``decision_ts`` — the decision it informs
          - the forecast features, identical on every row of the market

        A bin's price comes from ``spec.metadata["outcome_schema"]`` (``{"label",
        "price"}``, optionally ``"observed_at"`` and ``"source"`` per bin), with
        ``price_observed_at`` / ``price_source`` on the metadata as the default for bins
        that do not say. Discovery fills these in from the Gamma payload, so the only
        price this plugin has is the one discovery saw: **not** a CLOB mid at the horizon.
        To train on better prices, put them in the spec before calling this.

        A bin gets no row, and is counted in *exclusions* under the reason, when its price
        is missing (``price_missing``), not a probability (``price_invalid``), exactly 0 or
        1 (``price_degenerate``), has no readable time (``price_untimed``), or was observed
        at or after the decision (``price_after_cutoff``). It is never replaced by a
        default. A market that is not resolved, whose truth falls in no bin, or that has
        no usable bin gives no rows (``market_unresolved``, ``market_truth_unbinned``,
        ``market_no_usable_price``).

        Args:
            spec: The market.
            horizon: Decision point identifier.
            decision_time: **Required.** When the decision is taken (ISO-8601 or a
                ``datetime``; naive is UTC). No price observed at or after it is used.
            exclusions: Optional mapping the exclusion counts are added to, so a caller
                can total them over many markets. Without one, a market's exclusions are
                logged at WARNING instead of lost.

        Raises:
            ValueError: ``decision_time`` is missing or not a timestamp.
        """
        raw_time = kwargs.get("decision_time")
        if raw_time is None:
            raise ValueError(
                "build_training_rows needs decision_time (ISO-8601 or datetime): without a "
                "decision time there is no way to tell which prices were known by then"
            )
        decision = _as_utc(raw_time)
        if decision is None:
            raise ValueError(f"decision_time is not an ISO-8601 timestamp: {raw_time!r}")
        return self._bin_rows(spec, horizon, decision, kwargs.get("exclusions"))

    def build_training_row(
        self, spec: MarketSpec, horizon: str, **kwargs: Any
    ) -> dict[str, Any] | None:
        """Deprecated: the one row of the bin that won. Use :meth:`build_training_rows`.

        A panel of one row per market, labelled with the winner, shows the backtest only
        winners (issue #28): every YES it takes wins, so its PnL is a property of the
        panel and not of the model. The row is now at least self-consistent: the winning
        bin's *own* price, not the market's highest. ``decision_time`` is optional here, as
        it always was, and without it nothing checks the price against a decision.
        """
        warnings.warn(
            "WeatherTmaxPlugin.build_training_row emits only the winning bin, which biases "
            "any backtest built from it; use build_training_rows (one row per bin) instead",
            DeprecationWarning,
            stacklevel=2,
        )
        raw_time = kwargs.get("decision_time")
        decision = None if raw_time is None else _as_utc(raw_time)
        if raw_time is not None and decision is None:
            raise ValueError(f"decision_time is not an ISO-8601 timestamp: {raw_time!r}")
        rows = self._bin_rows(spec, horizon, decision, kwargs.get("exclusions"))
        return next((r for r in rows if r["outcome_label"] == r["winning_label"]), None)

    def _bin_rows(
        self,
        spec: MarketSpec,
        horizon: str,
        decision: datetime | None,
        exclusions: MutableMapping[str, int] | None,
    ) -> list[dict[str, Any]]:
        """The rows of :meth:`build_training_rows`; with no *decision* nothing is time-checked."""
        counts: Counter[str] = Counter()
        rows = self._collect_bin_rows(spec, horizon, decision, counts)
        if exclusions is not None:
            for reason, n in counts.items():
                exclusions[reason] = exclusions.get(reason, 0) + n
        elif counts:
            _logger.warning(
                "weather_tmax market %s: excluded %s",
                spec.market_id,
                ", ".join(f"{reason}={n}" for reason, n in sorted(counts.items())),
            )
        return rows

    def _collect_bin_rows(
        self, spec: MarketSpec, horizon: str, decision: datetime | None, counts: Counter[str]
    ) -> list[dict[str, Any]]:
        truth = self.fetch_truth(spec)
        if truth is None:
            counts[MARKET_UNRESOLVED] += 1
            return []
        winning_label = spec.resolve_winning_bin(truth)
        if winning_label is None:
            counts[MARKET_TRUTH_UNBINNED] += 1
            return []

        features = self.fetch_features(spec, horizon)
        schema = {
            str(entry.get("label")): entry
            for entry in spec.metadata.get("outcome_schema") or []
            if isinstance(entry, dict)
        }
        default_observed = spec.metadata.get("price_observed_at")
        default_source = spec.metadata.get("price_source")

        rows: list[dict[str, Any]] = []
        for bin_ in spec.outcome_bins:
            entry = schema.get(bin_.label, {})
            price, reason = _usable_price(entry.get("price"))
            observed = _as_utc(entry.get("observed_at") or default_observed)
            if reason is None and decision is not None:
                if observed is None:
                    reason = PRICE_UNTIMED
                elif observed >= decision:
                    reason = PRICE_AFTER_CUTOFF
            if reason is not None:
                counts[reason] += 1
                continue

            row: dict[str, Any] = {
                **features,
                "market_id": spec.market_id,
                "decision_horizon": horizon,
                "outcome_label": bin_.label,
                "winning_label": winning_label,
                "market_price": price,
                "price_source": str(entry.get("source") or default_source or "unknown"),
                "price_ts": None if observed is None else observed.isoformat(),
            }
            if decision is not None:
                row["decision_date"] = decision.date().isoformat()
                row["decision_ts"] = decision.isoformat()
            rows.append(row)

        if not rows:
            counts[MARKET_NO_USABLE_PRICE] += 1
        return rows

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

    def _build_spec(self, raw: dict[str, Any], observed_at: str | None = None) -> MarketSpec:
        """Convert a raw Gamma API market dict to a MarketSpec.

        Handles both the legacy ``tokens`` format and the canonical Gamma API
        format with ``outcomes`` / ``outcomePrices`` / ``clobTokenIds``.

        A price that is absent or unreadable is recorded as ``None``: a stand-in such as
        0.5 would reach the backtest as if the market had quoted it.

        Args:
            raw: The Gamma payload.
            observed_at: ISO-8601 time the payload's prices were seen, recorded as
                ``metadata["price_observed_at"]``; omit it when unknown.
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
        outcome_prices: list[float | None] = []

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

            raw_prices: Any = prices_field
            if isinstance(prices_field, str):
                try:
                    raw_prices = json.loads(prices_field)
                except (json.JSONDecodeError, ValueError):
                    raw_prices = []
            if isinstance(raw_prices, list):
                outcome_prices = [_price_or_none(p) for p in raw_prices]
        else:
            # Fall back to legacy ``tokens`` list
            tokens: list[dict[str, Any]] = raw.get("tokens") or []
            for t in tokens:
                if isinstance(t, dict):
                    lbl = str(t.get("outcome", t.get("label", t.get("name", ""))))
                    price = _price_or_none(t.get("price", t.get("outcomePrices")))
                else:
                    lbl = str(t)
                    price = None
                outcome_labels.append(lbl)
                outcome_prices.append(price)

        # A short price list leaves the remaining bins unpriced
        if len(outcome_prices) < len(outcome_labels):
            outcome_prices += [None] * (len(outcome_labels) - len(outcome_prices))

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

        # The highest known bin price. For display only: it is one bin's price, so it is
        # never a row's `market_price` (that was issue #28).
        known = [p for p in outcome_prices if p is not None and math.isfinite(p)]
        market_price: float | None = max(known) if known else None

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
                "price_source": "discovery",
                **({"price_observed_at": observed_at} if observed_at else {}),
                "unit": unit,
                "question": question,
            },
        )


# ---------------------------------------------------------------------------
# Module-level helpers (not part of the plugin class)
# ---------------------------------------------------------------------------


def _price_or_none(value: Any) -> float | None:
    """A price as a float, or None when it is absent or unreadable (never a stand-in)."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _usable_price(value: Any) -> tuple[float | None, str | None]:
    """(price, None) when *value* can be a bin's decision-time quote, else (None, reason)."""
    if value is None:
        return None, PRICE_MISSING
    if isinstance(value, bool):
        return None, PRICE_INVALID
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None, PRICE_INVALID
    if not math.isfinite(price) or not 0.0 <= price <= 1.0:
        return None, PRICE_INVALID
    if price in (0.0, 1.0):
        return None, PRICE_DEGENERATE
    return price, None


def _as_utc(value: Any) -> datetime | None:
    """*value* (a datetime, or an ISO-8601 string; naive means UTC) in UTC, or None."""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.strip())
        except ValueError:
            return None
    else:
        return None
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def _iso_utc(value: datetime) -> str:
    converted = _as_utc(value)
    assert converted is not None  # a datetime always converts
    return converted.isoformat()


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
