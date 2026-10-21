"""A deterministic synthetic weather world whose prices are *fair* (issue #28).

Every market has five temperature bins. The price of a bin is exactly its true
probability and the winner is drawn from those probabilities, so a backtest that
sees every bin at its decision-time price has an expected PnL of exactly zero for
*any* trade policy: a bet at a fair price is worth nothing, whoever places it.

That is the property the regression tests lean on. A panel that shows the policy only
the winner's row cannot reproduce it, so its backtest "finds" an edge that is not there.

Nothing here touches the network. The stubs implement just the client methods
``WeatherTmaxPlugin`` calls.
"""

from __future__ import annotations

import json
import math
import warnings
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

LABELS = ("<20°C", "20-22°C", "22-24°C", "24-26°C", ">26°C")
# One temperature strictly inside each bin; the bins share their edges, so a value on an
# edge would resolve to the first of two bins.
MIDPOINTS = (19.0, 21.0, 23.0, 25.0, 27.0)
CITIES = ("Paris", "Berlin", "Madrid", "Rome", "Vienna", "Oslo", "Lisbon", "Athens")

_PRICE_UNITS = 10_000  # prices are whole units of 1e-4 and sum to exactly 1
_FLOOR_UNITS = 300  # no bin is priced below 0.03, which keeps a YES bet's variance bounded


@dataclass(frozen=True)
class SyntheticMarket:
    market_id: str
    city: str
    decision_date: str
    target_date: str  # as the question prints it, e.g. "March 2"
    prices: tuple[float, ...]  # per bin, == its true probability
    winner: int
    forecast_max: float

    @property
    def target_date_iso(self) -> str:
        """The day the temperature is read, as an ISO date (the question prints it as text)."""
        return (date.fromisoformat(self.decision_date) + timedelta(days=1)).isoformat()

    @property
    def winning_label(self) -> str:
        return LABELS[self.winner]

    @property
    def truth(self) -> float:
        return MIDPOINTS[self.winner]

    def raw(self) -> dict[str, Any]:
        """The market as the Gamma API returns it: JSON-encoded lists, string prices."""
        return {
            "id": self.market_id,
            "question": f"Highest temperature in {self.city} on {self.target_date}?",
            "endDate": f"{self.target_date_iso}T23:59:00Z",
            "description": "Resolves to the highest temperature recorded by Wunderground.",
            "outcomes": json.dumps(list(LABELS)),
            "outcomePrices": json.dumps([f"{p:.4f}" for p in self.prices]),
            "clobTokenIds": json.dumps([f"{self.market_id}-{i}" for i in range(len(LABELS))]),
        }


@dataclass(frozen=True)
class World:
    markets: tuple[SyntheticMarket, ...]
    by_day: dict[str, list[SyntheticMarket]] = field(default_factory=dict)

    @property
    def decision_dates(self) -> list[str]:
        return sorted(self.by_day)


def make_world(n_days: int = 60, markets_per_day: int = 8, seed: int = 7) -> World:
    """Draw ``n_days * markets_per_day`` markets with fair prices, from ``seed``."""
    rng = np.random.default_rng(seed)
    start = date(2026, 3, 1)
    markets: list[SyntheticMarket] = []
    by_day: dict[str, list[SyntheticMarket]] = {}
    for d in range(n_days):
        decision = start + timedelta(days=d)
        target = decision + timedelta(days=1)
        for c in range(markets_per_day):
            shares = rng.dirichlet(np.full(len(LABELS), 1.5))
            units = _FLOOR_UNITS + np.floor(
                shares * (_PRICE_UNITS - _FLOOR_UNITS * len(LABELS))
            ).astype(int)
            units[int(np.argmax(units))] += _PRICE_UNITS - int(units.sum())
            prices = tuple(float(u) / _PRICE_UNITS for u in units)
            winner = int(rng.choice(len(LABELS), p=np.asarray(units) / _PRICE_UNITS))
            expected = sum(p * m for p, m in zip(prices, MIDPOINTS, strict=True))
            market = SyntheticMarket(
                market_id=f"m{d:03d}-{c}",
                city=CITIES[c % len(CITIES)],
                decision_date=decision.isoformat(),
                target_date=f"{target:%B} {target.day}",
                prices=prices,
                winner=winner,
                forecast_max=float(expected + rng.normal(0.0, 0.5)),
            )
            markets.append(market)
            by_day.setdefault(market.decision_date, []).append(market)
    return World(markets=tuple(markets), by_day=by_day)


class StubGamma:
    """Serves one day's markets, as Gamma would the morning they are decided."""

    def __init__(self, markets: list[SyntheticMarket]) -> None:
        self._raw = [m.raw() for m in markets]

    def fetch_markets_all(self, **kwargs: Any) -> list[dict[str, Any]]:
        return list(self._raw)


class StubForecast:
    """Market-level features only, keyed the way the plugin asks: (city, target_date)."""

    def __init__(self, world: World) -> None:
        self._by_key = {(m.city, m.target_date): m for m in world.markets}

    def get_features(self, city: str, target_date: str, horizon: str) -> dict[str, float]:
        m = self._by_key[(city, target_date)]
        return {"feature_fcst_max": m.forecast_max, "feature_lead_days": 1.0}


class StubTruth:
    def __init__(self, world: World) -> None:
        self._by_key = {(m.city, m.target_date): m for m in world.markets}

    def get_daily_max(self, city: str, date: str) -> float:
        return self._by_key[(city, date)].truth


def reference_panel_rows(world: World) -> list[dict[str, Any]]:
    """The panel a *correct* plugin must reproduce: every bin, at its own fair price."""
    rows: list[dict[str, Any]] = []
    for m in world.markets:
        for label, price in zip(LABELS, m.prices, strict=True):
            rows.append(
                {
                    "market_id": m.market_id,
                    "decision_date": m.decision_date,
                    "outcome_label": label,
                    "winning_label": m.winning_label,
                    "market_price": price,
                    "feature_fcst_max": m.forecast_max,
                    "feature_lead_days": 1.0,
                }
            )
    return rows


def _plugin_for_day(world: World, day: str) -> Any:
    """A plugin that discovers ``day``'s markets at 07:30Z, half an hour before the decision."""
    from pmlab.plugins.weather_tmax.plugin import WeatherTmaxPlugin

    observed = datetime.fromisoformat(f"{day}T07:30:00+00:00")
    return WeatherTmaxPlugin(
        gamma_client=StubGamma(world.by_day[day]),
        forecast_client=StubForecast(world),
        truth_client=StubTruth(world),
        clock=lambda: observed,
    )


def decision_time(day: str) -> str:
    return f"{day}T08:00:00Z"


def plugin_panel(world: World, exclusions: Counter[str] | None = None) -> pd.DataFrame:
    """The panel ``WeatherTmaxPlugin.build_training_rows`` builds for the whole world."""
    from pmlab.data.panel_builder import build_panel

    rows: list[dict[str, Any]] = []
    for day in world.decision_dates:
        plugin = _plugin_for_day(world, day)
        for spec in plugin.discover_markets():
            rows.extend(
                plugin.build_training_rows(
                    spec, "morning_of", decision_time=decision_time(day), exclusions=exclusions
                )
            )
    return build_panel(rows)


def legacy_panel(world: World) -> pd.DataFrame:
    """The panel the one-row-per-market ``build_training_row`` builds: winners only."""
    from pmlab.data.panel_builder import build_panel

    rows: list[dict[str, Any]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for day in world.decision_dates:
            plugin = _plugin_for_day(world, day)
            for spec in plugin.discover_markets():
                row = plugin.build_training_row(spec, "morning_of")
                assert row is not None
                row["decision_date"] = day  # the legacy row carries no decision date
                rows.append(row)
    return build_panel(rows)


def pnl_z_score(trades: Any) -> tuple[float, float, float]:
    """(total PnL, its standard deviation under fair prices, z) of a frictionless stake-1 log.

    A YES bet at price ``p`` pays ``(1[win] - p) / p``, variance ``(1 - p) / p``; a NO bet at
    YES-price ``p`` is a bet at ``q = 1 - p`` on the complement, variance ``p / (1 - p)``.
    The decisions are made before the outcome is known, so the per-trade PnLs are
    independent mean-zero terms and their sum has the standard deviation computed here.
    """
    if trades.empty:
        return 0.0, 0.0, 0.0
    price = trades["market_price"].to_numpy(dtype=float)
    is_yes = (trades["direction"] == "yes").to_numpy()
    variance = np.where(is_yes, (1.0 - price) / price, price / (1.0 - price))
    total = float(trades["realized_pnl"].sum())
    sd = math.sqrt(float(variance.sum()))
    return total, sd, total / sd
