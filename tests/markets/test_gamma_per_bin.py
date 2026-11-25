"""``discover_tmax_markets`` returns the per-bin Yes/No temperature markets, one entry per city and
day (#87), read from the markets recorded in ``fixtures/per_bin_markets.json`` (see
``test_tmax_bins.py`` for what is in it), served page by page the way ``/markets/keyset`` does.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from pmlab.markets.async_gamma_client import AsyncGammaClient
from pmlab.markets.gamma_client import GammaClient, TmaxMarketInfo

BASE = "https://mock-gamma.test"
FIXTURE = Path(__file__).parent / "fixtures" / "per_bin_markets.json"


def recorded() -> list[dict[str, Any]]:
    return json.loads(FIXTURE.read_text())["markets"]


def legacy(i: int) -> dict[str, Any]:
    """The multi-outcome form: one market whose outcomes are the bins."""
    return {
        "id": f"legacy{i}",
        "question": f"Highest temperature in City{i} on May 13?",
        "slug": f"legacy-{i}",
        "description": "Resolves with the highest temperature recorded on Wunderground.",
        "outcomes": '["30°C", "31°C"]',
        "outcomePrices": '["0.4", "0.6"]',
        "clobTokenIds": '["a", "b"]',
        "endDate": "2026-05-13T12:00:00Z",
        "active": True,
    }


def serve(markets: list[dict[str, Any]], size: int, seen: list[dict[str, str]] | None = None):
    """A ``/markets/keyset`` over *markets*: *size* per page, the cursor is ``cur<offset>``."""

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        if seen is not None:
            seen.append(params)
        start = int(params["after_cursor"].removeprefix("cur")) if "after_cursor" in params else 0
        body: dict[str, Any] = {"markets": markets[start : start + size]}
        if start + size < len(markets):
            body["next_cursor"] = f"cur{start + size}"
        return httpx.Response(200, json=body)

    return handler


def discover_sync(markets, size=5, **kwargs) -> list[TmaxMarketInfo]:
    with respx.mock:
        respx.get(f"{BASE}/markets/keyset").mock(side_effect=serve(markets, size))
        with GammaClient(base_url=BASE) as client:
            return client.discover_tmax_markets(page_size=size, **kwargs)


def discover_async(markets, size=5, **kwargs) -> list[TmaxMarketInfo]:
    async def run() -> list[TmaxMarketInfo]:
        with respx.mock:
            respx.get(f"{BASE}/markets/keyset").mock(side_effect=serve(markets, size))
            async with AsyncGammaClient(base_url=BASE) as client:
                return await client.discover_tmax_markets(page_size=size, **kwargs)

    return asyncio.run(run())


@pytest.fixture(params=["sync", "async"])
def discover(request):
    return discover_sync if request.param == "sync" else discover_async


def warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]


class TestDiscoveryOfPerBinMarkets:
    def test_one_entry_per_city_and_day_in_the_order_they_first_appear(self, discover) -> None:
        found = discover(recorded())
        assert [m.market_id for m in found] == ["479718", "1436284", "1451212", "1451693"]
        assert [m.city for m in found] == ["Atlanta", "Panama City", "Warsaw", "Atlanta"]
        assert [m.target_date for m in found] == [
            "2026-05-15",
            "2026-11-23",
            "2026-11-26",
            "2026-11-26",
        ]
        assert [m.unit for m in found] == ["F", "C", "C", "F"]
        assert [len(m.outcome_labels) for m in found] == [1, 3, 11, 11]
        assert all(isinstance(m, TmaxMarketInfo) for m in found)

    def test_an_entry_has_the_shape_of_a_multi_outcome_market(self, discover) -> None:
        atlanta = discover(recorded())[-1]
        raw = {m["id"]: m for m in recorded()}
        bins = [m for m in recorded() if m["events"][0]["id"] == "1451693"]
        assert atlanta.question == "Highest temperature in Atlanta on November 26?"
        assert atlanta.slug == "highest-temperature-in-atlanta-on-november-26-2026"
        assert atlanta.end_date == "2026-11-26T12:00:00Z" and atlanta.active is True
        assert atlanta.outcome_labels == [m["groupItemTitle"] for m in bins]
        assert atlanta.token_ids == [
            json.loads(m["clobTokenIds"])[0] for m in bins
        ]  # the Yes tokens
        assert atlanta.outcome_prices == [float(json.loads(m["outcomePrices"])[0]) for m in bins]
        assert len(atlanta.token_ids) == len(atlanta.outcome_labels) == len(atlanta.outcome_prices)
        assert raw[bins[0]["id"]]["outcomes"] == '["Yes", "No"]'

    def test_the_bins_of_an_event_that_sit_on_two_pages_are_one_entry(self, discover) -> None:
        for size in (1, 3, 4, 7, 100):
            found = discover(recorded(), size=size)
            assert [len(m.outcome_labels) for m in found] == [1, 3, 11, 11], size

    def test_a_default_call_is_scoped_to_the_tag_and_finishes(self) -> None:
        seen: list[dict[str, str]] = []
        with respx.mock:
            respx.get(f"{BASE}/markets/keyset").mock(side_effect=serve(recorded(), 100, seen))
            with GammaClient(base_url=BASE) as client:
                assert len(client.discover_tmax_markets()) == 4
        assert {p["tag_id"] for p in seen} == {"103040"} and len(seen) == 1

    def test_the_stale_event_without_tokens_is_refused_with_one_warning(
        self, discover, caplog
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="pmlab"):
            found = discover(recorded())
        assert "497034" not in [m.market_id for m in found]
        (said,) = warnings(caplog)
        assert said.startswith("discover_tmax_markets: 4 per-bin temperature markets were not read")
        assert "no Yes token id" in said

    def test_lowest_temperature_markets_are_not_returned(self, discover) -> None:
        lowest = [m for m in recorded() if "lowest temperature" in m["question"]]
        assert len(lowest) == 2 and discover(lowest) == []

    def test_a_price_gamma_did_not_give_is_nan_not_zero(self, discover) -> None:
        market = recorded()[-1] | {"outcomePrices": None}
        (entry,) = discover([market])
        assert len(entry.outcome_prices) == 1 and math.isnan(entry.outcome_prices[0])

    def test_the_multi_outcome_form_still_works_beside_it(self, discover) -> None:
        mixed = [legacy(1), *recorded(), legacy(2)]
        found = discover(mixed)
        assert [m.market_id for m in found] == [
            "legacy1",
            "legacy2",
            "479718",
            "1436284",
            "1451212",
            "1451693",
        ]
        assert found[0].outcome_labels == ["30°C", "31°C"] and found[0].city == "City1"

    def test_a_question_of_this_family_that_cannot_be_read_is_refused_not_guessed(
        self, discover, caplog
    ) -> None:
        odd = {
            **recorded()[-1],
            "id": "1",
            "question": "Will the highest temperature in X exceed 5°C on May 1?",
        }
        with caplog.at_level(logging.WARNING, logger="pmlab"):
            assert discover([odd]) == []
        (said,) = warnings(caplog)
        assert "1 per-bin temperature market was not read" in said and "(1: 1)" in said

    def test_nothing_refused_is_nothing_said(self, discover, caplog) -> None:
        good = [m for m in recorded() if m["events"][0]["id"] == "1451693"]
        with caplog.at_level(logging.WARNING, logger="pmlab"):
            assert len(discover(good)) == 1
        assert warnings(caplog) == []

    def test_a_market_on_two_pages_is_one_bin(self, discover) -> None:
        good = [m for m in recorded() if m["events"][0]["id"] == "1451693"]
        (entry,) = discover([*good, *good[:3]], size=4)  # three bins come round again
        assert len(entry.outcome_labels) == 11

    def test_the_per_bin_wording_is_never_passed_to_the_multi_outcome_reader(
        self, discover, caplog
    ) -> None:
        """Its regex would take ``X be 30°C`` for a city: the wording decides, not the outcomes."""
        odd = legacy(1) | {"question": "Will the highest temperature in Rome be 30°C on May 13?"}
        with caplog.at_level(logging.WARNING, logger="pmlab"):
            assert discover([odd]) == []
        (said,) = warnings(caplog)
        assert "outcomes are not Yes/No" in said

    def test_an_entry_is_active_only_while_every_bin_is(self, discover) -> None:
        good = [dict(m) for m in recorded() if m["events"][0]["id"] == "1451693"]
        assert discover(good)[0].active is True
        good[4]["active"] = False
        assert discover(good)[0].active is False
