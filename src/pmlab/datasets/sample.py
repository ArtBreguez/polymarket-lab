"""The bundled weather-temperature sample and the offline clients that serve it.

The file holds, per market, what a forward-collection run would have stored: the book as it
stood when it was discovered (nine bin prices), the forecast features of that moment, and
the day's official maximum, which is only *known* once the day has ended.

``SampleDataset.plugin`` wires the **real** ``WeatherTmaxPlugin`` to three small clients
that read that file instead of the Polymarket and weather APIs. The plugin itself is not
touched or subclassed: its constructor already takes its clients by injection, which is
exactly how its own tests drive it. So ``discover_markets``, ``build_training_rows`` and
``is_truth_final`` run their production code on the sample.

Time is replayed, not faked: a plugin built for a ``day`` stamps its discovery at 07:30 UTC
that morning and can only see observations that were final by then, so the leakage guards
and the settlement lag behave on the sample exactly as they do live.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from importlib import resources
from typing import Any

from pmlab.plugins.weather_tmax.plugin import WeatherTmaxPlugin

SAMPLE_FAMILY = "weather_tmax"
SAMPLE_HORIZON = "previous_evening"
_DATA = "weather_tmax_sample.json.gz"
_SUM = "weather_tmax_sample.sha256"
_SCHEMA_VERSION = 1
_N_BINS = 9
_DESCRIPTION = (
    "This market resolves to the highest temperature recorded in the city on the date, "
    "as published by Wunderground, in whole degrees Celsius."
)


class SampleDataError(RuntimeError):
    """The bundled sample is missing, corrupt, or not the shape this version expects."""


def _read(name: str) -> bytes:
    return resources.files("pmlab.datasets").joinpath("data", name).read_bytes()


def _labels(center: int, half_width: int) -> list[str]:
    """Nine bin labels: a tail, a single degree each, a tail (the plugin parses these)."""
    singles = [f"{center + k}°C" for k in range(-half_width, half_width + 1)]
    return [f"<={center - half_width - 1}°C", *singles, f">={center + half_width + 1}°C"]


class SampleDataset:
    """The parsed sample. Build it with `load_sample`."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.payload: dict[str, Any] = dict(payload)
        self.meta: dict[str, Any] = dict(self.payload["meta"])
        self.markets: list[dict[str, Any]] = list(self.payload["markets"])
        self._half_width = int(self.meta["half_width"])
        self._by_target: dict[str, list[dict[str, Any]]] = {}
        self._by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for m in self.markets:
            self._by_target.setdefault(m["target_date"], []).append(m)
            self._by_key[(m["city"], m["target_date"])] = m
        targets = sorted(self._by_target)
        # The day the market is listed and decided on: the day before its target day.
        self.days: list[str] = [
            (date.fromisoformat(t) - timedelta(days=1)).isoformat() for t in targets
        ]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> SampleDataset:
        """Validate a decoded payload and build the dataset; ``SampleDataError`` if it is off."""
        if payload.get("schema_version") != _SCHEMA_VERSION:
            raise SampleDataError(
                f"sample schema_version {payload.get('schema_version')!r} is not "
                f"{_SCHEMA_VERSION}: this pmlab and its data file are out of step"
            )
        meta, markets = payload.get("meta"), payload.get("markets")
        if not isinstance(meta, dict) or not isinstance(markets, list) or not markets:
            raise SampleDataError("sample has no meta or no markets")
        cities = list(meta.get("cities", {}))
        keys = {(m["city"], m["target_date"]) for m in markets}
        days = {m["target_date"] for m in markets}
        if len(keys) != len(markets) or len(markets) != len(cities) * len(days):
            raise SampleDataError("sample must hold one market per city per day, none missing")
        for m in markets:
            if len(m["prices"]) != _N_BINS:
                raise SampleDataError(f"market {m.get('id')} does not have {_N_BINS} bins")
        return cls(payload)

    # ---- shape ----------------------------------------------------------------------

    @property
    def n_markets(self) -> int:
        return len(self.markets)

    @property
    def cities(self) -> list[str]:
        return list(self.meta["cities"])

    @property
    def city_timezones(self) -> dict[str, str]:
        """city -> IANA zone, for ``PaperBroker.record(city_timezones=...)``."""
        return dict(self.meta["cities"])

    @property
    def first_target_date(self) -> str:
        return min(self._by_target)

    @property
    def last_target_date(self) -> str:
        return max(self._by_target)

    def labels(self, market: Mapping[str, Any]) -> list[str]:
        """The market's nine bin labels, lowest first."""
        return _labels(int(market["center"]), self._half_width)

    def winning_index(self, market: Mapping[str, Any]) -> int:
        """Index (0-8) of the bin the day's maximum fell in."""
        offset = int(market["truth_c"]) - int(market["center"]) + self._half_width + 1
        return max(0, min(_N_BINS - 1, offset))

    # ---- the offline plugin ------------------------------------------------------------

    def plugin(self, day: str | None = None) -> WeatherTmaxPlugin:
        """The real ``WeatherTmaxPlugin``, reading this file instead of the APIs.

        Args:
            day: ``YYYY-MM-DD``. The plugin as it stood at 07:30 UTC that morning: it
                discovers the markets open for the next day, stamps their prices with
                that time, and sees only observations that were final by then (days
                before ``day``). ``None``: the plugin after the last market resolved. It
                knows every truth, to label a panel and to settle, but has nothing "open"
                to discover and says so, like the real plugin without a Gamma client.
        """
        if day is None:
            gamma: SampleGammaClient | None = None
            clock = datetime(2100, 1, 1, tzinfo=UTC)  # unused: no discovery without a day
        else:
            gamma = SampleGammaClient(self, day)
            clock = datetime.fromisoformat(f"{day}T07:30:00+00:00")
        return WeatherTmaxPlugin(
            gamma_client=gamma,
            forecast_client=SampleForecastClient(self),
            truth_client=SampleTruthClient(self, day),
            clock=lambda: clock,
        )

    def __repr__(self) -> str:
        return (
            f"SampleDataset({self.n_markets} markets, {len(self.cities)} cities, "
            f"{self.first_target_date}..{self.last_target_date}, synthetic)"
        )


def load_sample() -> SampleDataset:
    """Load the bundled sample, verifying it against its checksum first.

    Raises:
        SampleDataError: the file is missing, does not match its checksum, or is not the
            schema this version reads.
    """
    try:
        blob, expected = _read(_DATA), _read(_SUM).decode().strip()
    except FileNotFoundError as exc:
        raise SampleDataError(
            f"bundled sample file not found ({exc}): reinstall pmlab, the data ships in the wheel"
        ) from exc
    digest = hashlib.sha256(blob).hexdigest()
    if digest != expected:
        raise SampleDataError(
            f"bundled sample failed its checksum (sha256 {digest[:12]}, expected {expected[:12]}): "
            "the file is corrupt, reinstall pmlab"
        )
    try:
        payload = json.loads(gzip.decompress(blob))
    except (OSError, EOFError, ValueError) as exc:  # pragma: no cover - checksum catches it
        raise SampleDataError(f"bundled sample is not valid gzip JSON: {exc}") from exc
    return SampleDataset.from_payload(payload)


class SampleGammaClient:
    """Stands in for the Gamma client: serves the markets listed on one morning."""

    def __init__(self, dataset: SampleDataset, day: str) -> None:
        self._dataset = dataset
        self._target = (date.fromisoformat(day) + timedelta(days=1)).isoformat()

    def fetch_markets_all(
        self, page_size: int = 100, max_results: int = 1000, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Gamma-shaped payloads of the markets open for tomorrow (``page_size`` is moot)."""
        out: list[dict[str, Any]] = []
        for m in self._dataset._by_target.get(self._target, [])[:max_results]:
            out.append(
                {
                    "id": m["id"],
                    "slug": m["id"],
                    "question": f"Highest temperature in {m['city']} on {m['target_date']}?",
                    "description": _DESCRIPTION,
                    "endDate": f"{m['target_date']}T23:00:00Z",
                    "outcomes": json.dumps(self._dataset.labels(m), ensure_ascii=False),
                    "outcomePrices": json.dumps([str(p) for p in m["prices"]]),
                }
            )
        return out


class SampleForecastClient:
    """Stands in for the forecast client: the features of a market at ``previous_evening``."""

    def __init__(self, dataset: SampleDataset) -> None:
        self._dataset = dataset

    def get_features(self, city: str, target_date: str, horizon: str) -> dict[str, float]:
        if horizon != SAMPLE_HORIZON:
            raise SampleDataError(
                f"the sample only has the {SAMPLE_HORIZON!r} horizon, not {horizon!r}"
            )
        market = self._dataset._by_key.get((city, target_date))
        if market is None:
            raise SampleDataError(f"the sample has no market for {city} on {target_date}")
        return {k: float(v) for k, v in market["features"].items()}


class SampleTruthClient:
    """Stands in for the observation client: the day's maximum, once the day has ended."""

    def __init__(self, dataset: SampleDataset, day: str | None = None) -> None:
        self._dataset = dataset
        self._day = day

    def get_daily_max(self, city: str, date: str) -> float | None:
        """The official maximum (C), or ``None`` if unknown or not final yet.

        With a morning ``day``, an observation is only known for days *before* it.
        """
        if self._day is not None and date >= self._day:
            return None
        market = self._dataset._by_key.get((city, date))
        return None if market is None else float(market["truth_c"])
