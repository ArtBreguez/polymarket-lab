"""``WeatherTmaxPlugin.discover_markets`` on live Gamma's per-bin Yes/No temperature markets (#86, #87).

The markets are the 32 recorded in ``tests/markets/fixtures/per_bin_markets.json`` (see
``tests/markets/test_tmax_bins.py``). The plugin reads the ``Daily Temperature`` tag, not "the
first 1000 markets by id", and turns the bins of one city and day into one `MarketSpec` that the
rest of the pipeline (truth lookup, training rows) treats like a multi-outcome market.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
import respx

from pmlab.markets.gamma_client import GammaClient
from pmlab.plugins.weather_tmax.plugin import WeatherTmaxPlugin

FIXTURE = Path(__file__).parents[2] / "markets" / "fixtures" / "per_bin_markets.json"
OBSERVED = datetime(2026, 11, 25, 7, 0, tzinfo=UTC)
BASE = "https://mock-gamma.test"


def recorded() -> list[dict[str, Any]]:
    return json.loads(FIXTURE.read_text())["markets"]


def plugin_over(markets: list[dict[str, Any]], truth: Any = None) -> tuple[WeatherTmaxPlugin, Any]:
    gamma = MagicMock()
    gamma.fetch_markets_all.return_value = markets
    return WeatherTmaxPlugin(gamma_client=gamma, truth_client=truth, clock=lambda: OBSERVED), gamma


def spec_of(market_id: str, **kwargs: Any):
    plugin, _ = plugin_over(recorded())
    return next(s for s in plugin.discover_markets(**kwargs) if s.market_id == market_id)


class Truth:
    def __init__(self, values: dict[tuple[str, str], float]) -> None:
        self.values = values

    def get_daily_max(self, city: str, date: str) -> float | None:
        return self.values.get((city, date))


class TestTheSpecOfACityAndDay:
    def test_one_spec_per_city_and_day(self) -> None:
        plugin, _ = plugin_over(recorded())
        specs = plugin.discover_markets()
        assert [s.market_id for s in specs] == ["479718", "1436284", "1451212", "1451693"]
        assert all(
            s.market_family == "range" and s.tags == ["weather", "temperature"] for s in specs
        )

    def test_the_shape_is_the_one_of_a_multi_outcome_market(self) -> None:
        spec = spec_of("1451693")
        assert spec.question == "Highest temperature in Atlanta on November 26?"
        assert spec.slug == "highest-temperature-in-atlanta-on-november-26-2026"
        assert spec.close_time == "2026-11-26T12:00:00Z"
        bins = [m for m in recorded() if m["events"][0]["id"] == "1451693"]
        assert [b.label for b in spec.outcome_bins] == [m["groupItemTitle"] for m in bins]
        meta = spec.metadata
        assert (meta["city"], meta["target_date"], meta["unit"]) == ("Atlanta", "2026-11-26", "F")
        assert meta["question"] == spec.question
        assert meta["token_ids"] == [json.loads(m["clobTokenIds"])[0] for m in bins]  # Yes tokens
        assert meta["outcome_schema"] == [
            {"label": m["groupItemTitle"], "price": float(json.loads(m["outcomePrices"])[0])}
            for m in bins
        ]
        assert meta["market_price"] == 0.565  # the highest bin price: display only
        assert meta["price_source"] == "discovery"
        assert meta["price_observed_at"] == OBSERVED.isoformat()

    def test_what_only_a_per_bin_spec_knows(self) -> None:
        spec = spec_of("1451693")
        bins = [m for m in recorded() if m["events"][0]["id"] == "1451693"]
        assert spec.metadata["event_id"] == "1451693"
        assert spec.metadata["bin_market_ids"] == [m["id"] for m in bins]
        assert spec.metadata["bins_complete"] is True
        assert spec_of("1436284").metadata["bins_complete"] is False  # 3 of 11 bins are still open

    def test_the_bins_are_half_open_and_tile_the_line(self) -> None:
        spec = spec_of("1451693")
        assert all(b.lower_inclusive and not b.upper_inclusive for b in spec.outcome_bins)
        for tenths in range(300, 900):
            t = tenths / 10
            assert sum(b.contains(t) for b in spec.outcome_bins) == 1, t

    @pytest.mark.parametrize(
        ("market_id", "truth", "winner"),
        [
            ("1451693", 52.0, "52-53°F"),
            ("1451693", 53.0, "52-53°F"),  # a reading of 53 is in the 52-53 bin, not the next
            ("1451693", 51.6, "52-53°F"),  # 51.6 rounds to 52
            ("1451693", 51.4, "50-51°F"),
            ("1451693", 30.0, "43°F or below"),
            ("1451693", 90.0, "62°F or higher"),
            ("1451212", -5.0, "-5°C"),
            ("1451212", -9.0, "-6°C or below"),
            ("1451212", 4.0, "4°C or higher"),
        ],
    )
    def test_the_winning_bin_of_a_reading(self, market_id, truth, winner) -> None:
        assert spec_of(market_id).resolve_winning_bin(truth) == winner

    def test_a_reading_in_a_bin_that_is_gone_resolves_to_nothing(self) -> None:
        panama = spec_of("1436284")  # 32, 34 and 35 only
        assert panama.resolve_winning_bin(33.0) is None
        assert panama.resolve_winning_bin(34.0) == "34°C"


class TestTheRestOfThePipelineSeesIt:
    def test_training_rows_are_one_per_bin_at_that_bins_own_yes_price(self) -> None:
        truth = Truth({("Atlanta", "2026-11-26"): 52.0})
        plugin, _ = plugin_over(recorded(), truth)
        spec = next(s for s in plugin.discover_markets() if s.market_id == "1451693")
        exclusions: dict[str, int] = {}
        rows = plugin.build_training_rows(
            spec, "morning_of", decision_time="2026-11-25T08:00:00Z", exclusions=exclusions
        )
        prices = {
            m["groupItemTitle"]: float(json.loads(m["outcomePrices"])[0])
            for m in recorded()
            if m["events"][0]["id"] == "1451693"
        }
        assert exclusions == {}
        assert {r["outcome_label"]: r["market_price"] for r in rows} == prices
        assert {r["winning_label"] for r in rows} == {"52-53°F"}
        assert {r["market_id"] for r in rows} == {"1451693"}
        assert {r["price_source"] for r in rows} == {"discovery"}

    def test_a_partial_event_gives_rows_for_its_bins_and_counts_a_truth_it_cannot_place(
        self,
    ) -> None:
        plugin, _ = plugin_over(recorded(), Truth({("Panama City", "2026-11-23"): 33.0}))
        spec = next(s for s in plugin.discover_markets() if s.market_id == "1436284")
        exclusions: dict[str, int] = {}
        rows = plugin.build_training_rows(
            spec, "morning_of", decision_time="2026-11-22T08:00:00Z", exclusions=exclusions
        )
        assert rows == [] and exclusions == {"market_truth_unbinned": 1}  # 33 is the closed bin

    def test_the_target_day_is_a_real_date_so_a_late_decision_is_refused(self) -> None:
        """The legacy form keeps the question's "November 26" (no year): never read as a day. This
        one is ISO, so a decision after the day is over is caught as one."""
        plugin, _ = plugin_over(recorded(), Truth({("Atlanta", "2026-11-26"): 52.0}))
        spec = next(s for s in plugin.discover_markets() if s.market_id == "1451693")
        exclusions: dict[str, int] = {}
        rows = plugin.build_training_rows(
            spec, "late", decision_time="2026-11-27T01:00:00Z", exclusions=exclusions
        )
        assert rows == [] and exclusions == {"decision_after_close": 1}

    def test_features_and_truth_are_asked_for_by_city_and_iso_day(self) -> None:
        forecast = MagicMock()
        forecast.get_features.return_value = {"f": 1.0}
        truth = MagicMock()
        truth.get_daily_max.return_value = 52.0
        gamma = MagicMock()
        gamma.fetch_markets_all.return_value = recorded()
        plugin = WeatherTmaxPlugin(gamma, forecast, truth, clock=lambda: OBSERVED)
        spec = next(s for s in plugin.discover_markets() if s.market_id == "1451693")
        assert plugin.fetch_features(spec, "h") == {"f": 1.0}
        forecast.get_features.assert_called_once_with(
            city="Atlanta", target_date="2026-11-26", horizon="h"
        )
        assert plugin.fetch_truth(spec) == 52.0
        truth.get_daily_max.assert_called_once_with(city="Atlanta", date="2026-11-26")


class TestWhatIsLeftOut:
    def test_the_multi_outcome_form_still_comes_first_and_unchanged(self) -> None:
        legacy = {
            "id": "L1",
            "question": "Highest temperature in Paris on May 13?",
            "slug": "legacy",
            "description": "Resolves with the highest temperature recorded on Wunderground.",
            "outcomes": '["30°C", "31°C"]',
            "outcomePrices": '["0.4", "0.6"]',
            "clobTokenIds": '["a", "b"]',
            "endDate": "2026-05-13T12:00:00Z",
        }
        plugin, _ = plugin_over([*recorded()[:3], legacy])
        specs = plugin.discover_markets()
        assert [s.market_id for s in specs] == ["L1", "479718"]
        assert specs[0].metadata["token_ids"] == ["a", "b"] and "event_id" not in specs[0].metadata

    def test_a_per_bin_market_never_becomes_a_yes_no_spec(self) -> None:
        plugin, _ = plugin_over(recorded())
        for spec in plugin.discover_markets():
            assert [b.label for b in spec.outcome_bins if b.label in ("Yes", "No")] == []

    def test_what_cannot_be_read_is_one_warning(self, caplog) -> None:
        plugin, _ = plugin_over(recorded())
        with caplog.at_level(logging.WARNING, logger="pmlab"):
            plugin.discover_markets()
        said = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert len(said) == 1 and said[0].startswith("discover_markets: 4 per-bin temperature")
        assert "no Yes token id" in said[0]

    def test_lowest_temperature_markets_are_not_specs(self) -> None:
        lowest = [m for m in recorded() if "lowest temperature" in m["question"]]
        plugin, _ = plugin_over(lowest)
        assert plugin.discover_markets() == []


class TestScopedByTag:
    """It reads the weather tag, all of it, by default (#86)."""

    def test_the_default_reads_the_daily_temperature_tag_without_a_market_cut(self) -> None:
        plugin, gamma = plugin_over([])
        plugin.discover_markets()
        (call,) = gamma.fetch_markets_all.call_args_list
        assert call.kwargs["tag_id"] == 103040
        assert call.kwargs["page_size"] == 100
        # 2,633 markets sit under the tag: stopping at 1000 would cut it at an event's middle
        assert call.kwargs["max_results"] >= 100_000

    def test_unscoped_is_what_it_was_the_first_1000_markets(self) -> None:
        plugin, gamma = plugin_over([])
        plugin.discover_markets(tag_id=None)
        (call,) = gamma.fetch_markets_all.call_args_list
        assert call.kwargs == {"page_size": 100, "max_results": 1000}  # no tag_id at all

    def test_the_arguments_that_were_there_still_go_through(self) -> None:
        plugin, gamma = plugin_over([])
        plugin.discover_markets(limit=50, max_results=300, tag="weather", closed=False)
        (call,) = gamma.fetch_markets_all.call_args_list
        assert call.kwargs == {
            "page_size": 50,
            "max_results": 300,
            "tag": "weather",
            "closed": False,
            "tag_id": 103040,
        }

    def test_another_tag_can_be_asked_for(self) -> None:
        plugin, gamma = plugin_over([])
        plugin.discover_markets(tag_id=84)
        assert gamma.fetch_markets_all.call_args.kwargs["tag_id"] == 84
        assert gamma.fetch_markets_all.call_args.kwargs["max_results"] >= 100_000

    def test_the_old_default_of_1000_is_what_an_unscoped_call_keeps(self) -> None:
        """Round 2 of #85 (TEXT-3): the documented default of 1000 had no test."""
        plugin, gamma = plugin_over([])
        plugin.discover_markets(tag_id=None)
        assert gamma.fetch_markets_all.call_args.kwargs["max_results"] == 1000

    def test_through_the_real_client_the_whole_tag_is_read_past_a_thousand_markets(self) -> None:
        filler = [{"id": f"f{i}", "question": "Will it rain?"} for i in range(1100)]
        universe = [*filler, *recorded()]
        seen: list[dict[str, str]] = []

        def keyset(request: httpx.Request) -> httpx.Response:
            params = dict(request.url.params)
            seen.append(params)
            start = int(params["after_cursor"].removeprefix("c")) if "after_cursor" in params else 0
            body: dict[str, Any] = {"markets": universe[start : start + 100]}
            if start + 100 < len(universe):
                body["next_cursor"] = f"c{start + 100}"
            return httpx.Response(200, json=body)

        with respx.mock:
            respx.get(f"{BASE}/markets/keyset").mock(side_effect=keyset)
            with GammaClient(base_url=BASE) as client:
                plugin = WeatherTmaxPlugin(gamma_client=client, clock=lambda: OBSERVED)
                scoped = plugin.discover_markets()
                unscoped = plugin.discover_markets(tag_id=None)
        assert [s.market_id for s in scoped] == ["479718", "1436284", "1451212", "1451693"]
        assert unscoped == []  # the first 1000 markets hold none of them: today's behaviour
        assert [p.get("tag_id") for p in seen][0] == "103040"
        assert sum(1 for p in seen if p.get("tag_id") == "103040") == 12
        assert sum(1 for p in seen if "tag_id" not in p) == 10  # ten pages of 100, then it stops
