"""The keyset paging against replies recorded from live Gamma (#83).

``fixtures/`` holds what ``https://gamma-api.polymarket.com/markets/keyset`` answered on 2026-11-25,
with each market trimmed to the fields discovery reads (the cursors, the page shape and the end of
the data are as received):

* ``keyset_weather_page_1.json`` / ``_2.json``: ``?limit=2&active=true&closed=false&tag_id=103040``
  and the same with ``after_cursor`` set to the first reply's ``next_cursor``;
* ``keyset_last_page.json``: a last page, which has no ``next_cursor`` at all;
* ``keyset_empty.json``: ``{"markets": []}``, what a filter that matches nothing answers;
* ``keyset_invalid_cursor.json``: the HTTP 422 body of a cursor Gamma cannot read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from pmlab.markets.gamma_client import GammaClient

FIXTURES = Path(__file__).parent / "fixtures"
BASE = "https://mock-gamma.test"
KEYSET_URL = f"{BASE}/markets/keyset"


def recorded(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text())


def test_the_recorded_pages_have_the_shape_the_client_relies_on() -> None:
    first, second = recorded("keyset_weather_page_1.json"), recorded("keyset_weather_page_2.json")
    last, empty = recorded("keyset_last_page.json"), recorded("keyset_empty.json")
    for page in (first, second):
        assert isinstance(page["markets"], list) and isinstance(page["next_cursor"], str)
        assert page["next_cursor"]
    assert first["next_cursor"] != second["next_cursor"]
    assert "next_cursor" not in last and last["markets"]  # the end: the key is simply absent
    assert empty == {"markets": []}


class TestARecordedWalk:
    def walk(self, replies: list[httpx.Response]) -> tuple[list[dict[str, Any]], list[str | None]]:
        seen: list[str | None] = []

        def serve(request: httpx.Request) -> httpx.Response:
            seen.append(request.url.params.get("after_cursor"))
            return replies[len(seen) - 1]

        with respx.mock:
            respx.get(KEYSET_URL).mock(side_effect=serve)
            with GammaClient(base_url=BASE) as client:
                return client.fetch_markets_all(page_size=2, max_results=1000), seen

    def test_the_cursor_goes_back_byte_for_byte_and_the_missing_one_ends_it(self) -> None:
        first = recorded("keyset_weather_page_1.json")
        second = recorded("keyset_weather_page_2.json")
        last = recorded("keyset_last_page.json")
        markets, cursors = self.walk(
            [httpx.Response(200, json=page) for page in (first, second, last)]
        )
        assert cursors == [None, first["next_cursor"], second["next_cursor"]]
        expected = [m["id"] for page in (first, second, last) for m in page["markets"]]
        assert [m["id"] for m in markets] == expected
        assert len(set(expected)) == len(expected) == 5

    def test_an_empty_universe(self) -> None:
        markets, cursors = self.walk([httpx.Response(200, json=recorded("keyset_empty.json"))])
        assert markets == [] and cursors == [None]

    def test_an_invalid_cursor_is_an_error_not_the_end_of_the_data(self) -> None:
        first = recorded("keyset_weather_page_1.json")
        bad = httpx.Response(422, json=recorded("keyset_invalid_cursor.json"))
        with pytest.raises(httpx.HTTPStatusError) as err:
            self.walk([httpx.Response(200, json=first), bad])
        assert err.value.response.status_code == 422
        assert "invalid cursor" in err.value.response.text

    def test_discovery_walks_the_same_pages_without_error(self) -> None:
        """What it keeps of them is the filter's business (these are per-bin Yes/No markets), not
        the paging's: all three pages are read, each cursor sent back, and nothing raises."""
        pages = [
            recorded("keyset_weather_page_1.json"),
            recorded("keyset_weather_page_2.json"),
            recorded("keyset_last_page.json"),
        ]
        calls: list[str | None] = []

        def serve(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.params.get("after_cursor"))
            return httpx.Response(200, json=pages[len(calls) - 1])

        with respx.mock:
            respx.get(KEYSET_URL).mock(side_effect=serve)
            with GammaClient(base_url=BASE) as client:
                client.discover_tmax_markets(page_size=2)
        assert calls == [None, pages[0]["next_cursor"], pages[1]["next_cursor"]]
