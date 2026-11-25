"""Market discovery pages through ``/markets/keyset`` by cursor (#83).

Live Gamma serves ``/markets`` by ``offset`` only up to about 2000 and answers HTTP 422 "offset too
large, use /markets/keyset" from 2100 on, so a discovery of every active market stopped at the first
2100 (#79 made that a warning, #83 removes it). ``/markets/keyset`` answers ``{"markets": [...],
"next_cursor": "..."}`` and takes the cursor back as ``after_cursor``; the last page has no
``next_cursor`` (see ``tests/markets/fixtures`` for recorded replies). Sleeping is patched out.
"""

from __future__ import annotations

import asyncio
import logging

import httpx
import pytest
import respx

from pmlab.markets import async_gamma_client, gamma_client
from pmlab.markets.async_gamma_client import AsyncGammaClient
from pmlab.markets.gamma_client import GammaClient, TmaxMarketInfo

BASE = "https://mock-gamma.test"
KEYSET_URL = f"{BASE}/markets/keyset"
OFFSET_URL = f"{BASE}/markets"
PAGE = 2  # page_size used throughout, so a "full page" is two markets


def tmax(i: int) -> dict:
    return {
        "id": str(i),
        "question": f"Highest temperature in City{i} on May 13?",
        "slug": f"tmax-{i}",
        "description": "Resolves with the highest temperature recorded on Wunderground.",
        "outcomes": '["30°C", "31°C"]',
        "outcomePrices": '["0.4", "0.6"]',
        "clobTokenIds": '["a", "b"]',
        "endDate": "2026-05-13T12:00:00Z",
        "active": True,
    }


def other(i: int) -> dict:
    return {"id": f"x{i}", "question": f"Will event {i} happen?"}


class KeysetServer:
    """A fake Gamma ``/markets/keyset``.

    ``total`` tmax markets in pages of ``PAGE`` (or explicit ``pages``); the cursor for page ``k`` is
    ``cur<k>``, and the last page has no ``next_cursor``, as live Gamma's. ``failures`` scripts the
    replies of a page (by index) before it is served: a response, or an exception to raise.
    """

    def __init__(
        self,
        total: int | None = None,
        *,
        pages: list[list[dict]] | None = None,
        failures: dict[int, list] | None = None,
        cursor_on_full_page: bool = False,
    ) -> None:
        assert (total is None) != (pages is None)
        self.total = total
        self.explicit = pages
        self.cursor_on_full_page = cursor_on_full_page
        self.failures = {k: list(v) for k, v in (failures or {}).items()}
        self.requests: list[dict[str, str]] = []

    def page(self, index: int) -> list[dict]:
        if self.explicit is not None:
            return self.explicit[index] if index < len(self.explicit) else []
        assert self.total is not None
        return [tmax(i) for i in range(index * PAGE, min((index + 1) * PAGE, self.total))]

    def has_next(self, index: int) -> bool:
        page = self.page(index)
        if self.cursor_on_full_page:  # a last page that is full still points at one more (empty)
            return len(page) == PAGE
        if self.explicit is not None:
            return index + 1 < len(self.explicit)
        assert self.total is not None
        return (index + 1) * PAGE < self.total

    @property
    def cursors(self) -> list[str | None]:
        """The ``after_cursor`` of each request, in order (None on the first)."""
        return [r.get("after_cursor") for r in self.requests]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        self.requests.append(params)
        cursor = params.get("after_cursor")
        index = 0 if cursor is None else int(cursor.removeprefix("cur"))
        scripted = self.failures.get(index)
        if scripted:
            outcome = scripted.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        body: dict = {"markets": self.page(index)}
        if self.has_next(index):
            body["next_cursor"] = f"cur{index + 1}"
        return httpx.Response(200, json=body)


class OffsetServer:
    """A fake ``/markets`` by offset (the fallback for a Gamma that does not know the keyset).

    ``overlap`` makes every page after the first start that many markets early, so the last
    market(s) of a page come again at the top of the next one (offset paging under writes does).
    """

    def __init__(
        self, total: int, failures: dict[int, list] | None = None, *, overlap: int = 0
    ) -> None:
        self.total = total
        self.overlap = overlap
        self.failures = {k: list(v) for k, v in (failures or {}).items()}
        self.calls: list[int] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        offset = int(request.url.params["offset"])
        self.calls.append(offset)
        scripted = self.failures.get(offset)
        if scripted:
            return scripted.pop(0)
        start = max(offset - self.overlap, 0)
        return httpx.Response(
            200, json=[tmax(i) for i in range(start, min(offset + PAGE, self.total))]
        )


def status(code: int, **headers: str) -> httpx.Response:
    return httpx.Response(code, headers=headers, json={"error": code})


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    slept: list[float] = []

    async def fake_async(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(gamma_client, "_sleep", slept.append)
    monkeypatch.setattr(async_gamma_client, "_sleep", fake_async)
    return slept


def _mount(keyset: KeysetServer, offset: OffsetServer | None) -> None:
    respx.get(KEYSET_URL).mock(side_effect=keyset)
    if offset is not None:  # only when the test expects the fallback: an unexpected call fails
        respx.get(OFFSET_URL).mock(side_effect=offset)


def run_sync(
    keyset: KeysetServer, offset: OffsetServer | None = None, **kwargs
) -> list[TmaxMarketInfo]:
    with respx.mock:
        _mount(keyset, offset)
        with GammaClient(base_url=BASE) as client:
            return client.discover_tmax_markets(page_size=PAGE, **kwargs)


async def run_async(
    keyset: KeysetServer, offset: OffsetServer | None = None, **kwargs
) -> list[TmaxMarketInfo]:
    with respx.mock:
        _mount(keyset, offset)
        async with AsyncGammaClient(base_url=BASE) as client:
            return await client.discover_tmax_markets(page_size=PAGE, **kwargs)


def run_fetch_all(keyset: KeysetServer, offset: OffsetServer | None = None, **kwargs) -> list[dict]:
    with respx.mock:
        _mount(keyset, offset)
        with GammaClient(base_url=BASE) as client:
            return client.fetch_markets_all(page_size=PAGE, **kwargs)


@pytest.fixture(params=["sync", "async"])
def discover(request):
    """``discover(keyset, offset=None)`` for the sync client, or the async one driven to the end."""
    if request.param == "sync":
        return run_sync
    return lambda keyset, offset=None, **kw: asyncio.run(run_async(keyset, offset, **kw))


def pmlab_warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]


# ---------------------------------------------------------------------------------------------
# The cursor loop
# ---------------------------------------------------------------------------------------------
class TestTheCursorLoop:
    def test_every_cursor_is_followed_and_only_tmax_markets_come_back(
        self, discover, sleeps
    ) -> None:
        keyset = KeysetServer(total=5)
        found = discover(keyset)
        assert [m.market_id for m in found] == ["0", "1", "2", "3", "4"]
        assert isinstance(found[0], TmaxMarketInfo) and found[0].city == "City0"
        assert keyset.cursors == [None, "cur1", "cur2"]
        assert sleeps == []

    def test_what_is_asked_for_is_the_keyset_query_and_not_an_offset(
        self, discover, sleeps
    ) -> None:
        keyset = KeysetServer(total=3)
        discover(keyset)
        assert keyset.requests[0] == {"limit": "2", "active": "true", "closed": "false"}
        assert keyset.requests[1] == {
            "limit": "2",
            "active": "true",
            "closed": "false",
            "after_cursor": "cur1",
        }
        assert all("offset" not in r for r in keyset.requests)

    def test_active_false_is_passed_on(self, discover, sleeps) -> None:
        keyset = KeysetServer(total=1)
        discover(keyset, active=False)
        assert keyset.requests[0]["active"] == "false" and keyset.requests[0]["closed"] == "false"

    def test_non_tmax_markets_are_dropped_and_do_not_end_the_paging(self, discover, sleeps) -> None:
        keyset = KeysetServer(pages=[[other(0), other(1)], [tmax(7), other(2)], [tmax(8)]])
        assert [m.market_id for m in discover(keyset)] == ["7", "8"]
        assert keyset.cursors == [None, "cur1", "cur2"]

    def test_nothing_to_discover(self, discover, sleeps) -> None:
        keyset = KeysetServer(pages=[[]])
        assert discover(keyset) == []
        assert keyset.cursors == [None]


class TestTheEndOfTheData:
    def test_a_missing_next_cursor_ends_it_even_on_a_full_page(self, discover, sleeps) -> None:
        keyset = KeysetServer(total=4)  # the second page is full and has no next_cursor
        assert len(discover(keyset)) == 4
        assert keyset.cursors == [None, "cur1"]  # no third request

    def test_a_full_last_page_that_points_at_an_empty_one(self, discover, sleeps) -> None:
        keyset = KeysetServer(total=4, cursor_on_full_page=True)
        assert len(discover(keyset)) == 4
        assert keyset.cursors == [None, "cur1", "cur2"]

    @pytest.mark.parametrize("empty", [None, ""])
    def test_a_null_or_empty_next_cursor_is_the_end_too(self, discover, sleeps, empty) -> None:
        reply = httpx.Response(200, json={"markets": [tmax(0)], "next_cursor": empty})
        keyset = KeysetServer(total=99, failures={0: [reply]})
        assert [m.market_id for m in discover(keyset)] == ["0"]
        assert keyset.cursors == [None]

    def test_a_short_page_does_not_end_it_while_a_cursor_comes_with_it(
        self, discover, sleeps
    ) -> None:
        keyset = KeysetServer(pages=[[tmax(0)], [tmax(1)], [tmax(2)]])  # one market per page
        assert [m.market_id for m in discover(keyset)] == ["0", "1", "2"]
        assert keyset.cursors == [None, "cur1", "cur2"]


class TestEachMarketOnce:
    def test_a_market_served_on_two_pages_is_returned_once(self, discover, sleeps) -> None:
        keyset = KeysetServer(pages=[[tmax(0), tmax(1)], [tmax(1), tmax(2)], [tmax(2), tmax(3)]])
        assert [m.market_id for m in discover(keyset)] == ["0", "1", "2", "3"]

    def test_a_page_of_only_repeats_changes_nothing(self, discover, sleeps) -> None:
        keyset = KeysetServer(pages=[[tmax(0), tmax(1)], [tmax(0), tmax(1)], [tmax(2)]])
        assert [m.market_id for m in discover(keyset)] == ["0", "1", "2"]

    def test_the_first_copy_wins(self, discover, sleeps) -> None:
        stale, fresh = tmax(1), tmax(1)
        stale["outcomePrices"], fresh["outcomePrices"] = '["0.1", "0.9"]', '["0.7", "0.3"]'
        keyset = KeysetServer(pages=[[tmax(0), stale], [fresh]])
        found = {m.market_id: m for m in discover(keyset)}
        assert found["1"].outcome_prices == [0.1, 0.9]

    def test_markets_with_no_id_are_not_merged(self, discover, sleeps) -> None:
        nameless = [{**tmax(i), "id": None} for i in (1, 2)]  # no id to tell them apart by
        keyset = KeysetServer(pages=[[tmax(0), nameless[0]], [nameless[1]]])
        assert len(discover(keyset)) == 3

    def test_a_market_with_a_condition_id_and_no_id_is_told_by_it(self, discover, sleeps) -> None:
        """``id`` is what Gamma sends; a payload that only has ``conditionId`` is still one market."""
        by_condition = [{**tmax(i), "id": None, "conditionId": f"0xc{i}"} for i in (0, 1)]
        keyset = KeysetServer(
            pages=[[by_condition[0], tmax(2)], [by_condition[0], by_condition[1]]]
        )
        found = discover(keyset)
        assert [m.market_id for m in found] == ["0xc0", "2", "0xc1"]

    def test_the_id_wins_over_the_condition_id(self, discover, sleeps) -> None:
        """Two markets with one ``conditionId`` and two ids are two markets (an id is the key)."""
        twins = [{**tmax(i), "conditionId": "0xsame"} for i in (1, 2)]
        keyset = KeysetServer(pages=[[twins[0]], [twins[1]]])
        assert [m.market_id for m in discover(keyset)] == ["1", "2"]


# ---------------------------------------------------------------------------------------------
# The cap, and a cursor that does not move
# ---------------------------------------------------------------------------------------------
class TestARunawayLoopStops:
    @pytest.fixture(autouse=True)
    def _cap(self, monkeypatch):
        for module in (gamma_client, async_gamma_client):
            monkeypatch.setattr(module, "_MAX_DISCOVERY_PAGES", 3)

    def test_cursors_for_ever_raise_after_the_cap(self, discover, sleeps) -> None:
        keyset = KeysetServer(total=10**9)  # a universe that never runs out
        with pytest.raises(RuntimeError, match="3 pages") as err:
            discover(keyset)
        assert "next_cursor" in str(err.value)
        assert keyset.cursors == [None, "cur1", "cur2"]

    def test_a_universe_that_ends_on_the_last_allowed_page_is_fine(self, discover, sleeps) -> None:
        keyset = KeysetServer(total=5)  # pages 0, 1, 2; the third has no cursor
        assert len(discover(keyset)) == 5
        assert len(keyset.requests) == 3

    def test_a_retried_page_counts_once(self, discover, sleeps) -> None:
        keyset = KeysetServer(
            total=5, failures={0: [status(500)], 1: [status(500)], 2: [status(500)]}
        )
        assert len(discover(keyset)) == 5

    def test_the_cap_counts_pages_not_markets(self, discover, sleeps) -> None:
        keyset = KeysetServer(total=10**9)
        with pytest.raises(RuntimeError, match="3 pages") as err:
            discover(keyset)
        assert "6 markets" in str(err.value)  # three pages of PAGE

    def test_the_message_blames_the_size_of_the_universe_not_a_loop(self, discover, sleeps) -> None:
        """On live Gamma the cap fires because the universe is larger than 100,000 markets, with
        every cursor new; a loop has its own error ("the same next_cursor twice"), so the cap's
        must not send the reader looking for one."""
        keyset = KeysetServer(total=10**9)
        with pytest.raises(RuntimeError) as err:
            discover(keyset)
        message = str(err.value)
        assert "the cap counts pages, not markets" in message
        assert "larger than" in message and "6 markets" in message
        assert "looping" not in message


def test_the_default_cap_is_a_guard_of_at_least_1000_pages() -> None:
    assert gamma_client._MAX_DISCOVERY_PAGES >= 1000
    assert async_gamma_client._MAX_DISCOVERY_PAGES >= 1000


class TestACursorThatDoesNotMove:
    def test_the_same_next_cursor_twice_is_a_loop_and_raises_at_once(
        self, discover, sleeps
    ) -> None:
        # a server that ignores after_cursor: every reply is page 0 and points at cur1
        stuck = httpx.Response(200, json={"markets": [tmax(0), tmax(1)], "next_cursor": "cur1"})
        keyset = KeysetServer(total=10**9, failures={0: [stuck], 1: [stuck]})
        with pytest.raises(RuntimeError, match="same next_cursor"):
            discover(keyset)
        assert keyset.cursors == [None, "cur1"]  # not a thousand requests

    def test_a_cursor_seen_earlier_is_a_loop_too(self, discover, sleeps) -> None:
        back = httpx.Response(200, json={"markets": [tmax(5), tmax(6)], "next_cursor": "cur1"})
        keyset = KeysetServer(total=10**9, failures={2: [back]})
        with pytest.raises(RuntimeError, match="same next_cursor"):
            discover(keyset)
        assert keyset.cursors == [None, "cur1", "cur2"]


# ---------------------------------------------------------------------------------------------
# Retry, backoff, Retry-After (#79) apply to every keyset page
# ---------------------------------------------------------------------------------------------
class TestATransientFailureCostsOnlyThatPage:
    @pytest.mark.parametrize("code", [429, 500, 502, 503, 504, 408])
    def test_a_transient_status_on_a_late_page_is_retried_and_nothing_is_lost(
        self, discover, sleeps, code
    ) -> None:
        keyset = KeysetServer(total=10, failures={3: [status(code)]})
        found = discover(keyset)
        assert [m.market_id for m in found] == [str(i) for i in range(10)]
        assert keyset.cursors == [None, "cur1", "cur2", "cur3", "cur3", "cur4"]
        assert sleeps == [1.0]

    def test_the_first_page_is_retried_too_and_does_not_fall_back(self, discover, sleeps) -> None:
        keyset = KeysetServer(total=3, failures={0: [status(503)]})
        assert len(discover(keyset)) == 3  # no offset route mounted: a fallback would fail
        assert keyset.cursors == [None, None, "cur1"]

    def test_the_wait_doubles_between_attempts(self, discover, sleeps) -> None:
        keyset = KeysetServer(total=4, failures={1: [status(503)] * 4})
        assert len(discover(keyset)) == 4
        assert sleeps == [1.0, 2.0, 4.0, 8.0]

    def test_retry_after_is_honoured_for_a_429(self, discover, sleeps) -> None:
        keyset = KeysetServer(total=4, failures={1: [status(429, **{"Retry-After": "7"})]})
        assert len(discover(keyset)) == 4
        assert sleeps == [7.0]

    def test_retry_after_is_capped_and_an_unusable_one_falls_back_to_the_backoff(
        self, discover, sleeps
    ) -> None:
        for header, expected in (("3600", 60.0), ("Wed, 21 Oct 2026 07:28:00 GMT", 1.0)):
            sleeps.clear()
            keyset = KeysetServer(total=4, failures={1: [status(429, **{"Retry-After": header})]})
            assert len(discover(keyset)) == 4
            assert sleeps == [expected]

    @pytest.mark.parametrize(
        "make",
        [
            lambda: httpx.ReadTimeout("slow"),
            lambda: httpx.ConnectTimeout("slow"),
            lambda: httpx.ReadError("reset"),
            lambda: httpx.RemoteProtocolError("cut"),
        ],
    )
    def test_a_timeout_or_a_connection_cut_mid_request_is_retried(
        self, discover, sleeps, make
    ) -> None:
        keyset = KeysetServer(total=6, failures={2: [make()]})
        assert len(discover(keyset)) == 6
        assert sleeps == [1.0]

    def test_a_transient_failure_that_persists_raises_after_five_tries(
        self, discover, sleeps
    ) -> None:
        keyset = KeysetServer(total=10, failures={2: [status(503)] * 9})
        with pytest.raises(httpx.HTTPStatusError) as err:
            discover(keyset)
        assert err.value.response.status_code == 503
        assert keyset.cursors.count("cur2") == 5 and sleeps == [1.0, 2.0, 4.0, 8.0]

    def test_a_persistent_failure_on_the_first_page_raises_and_does_not_fall_back(
        self, discover, sleeps
    ) -> None:
        keyset = KeysetServer(total=10, failures={0: [status(503)] * 9})
        with pytest.raises(httpx.HTTPStatusError):
            discover(keyset)  # no offset route mounted
        assert len(keyset.requests) == 5


class TestWhatIsNotTransientIsNotRetried:
    def test_a_refused_connection_raises_at_once(self, discover, sleeps) -> None:
        keyset = KeysetServer(total=10, failures={2: [httpx.ConnectError("refused")]})
        with pytest.raises(httpx.ConnectError):
            discover(keyset)
        assert sleeps == [] and len(keyset.requests) == 3

    def test_a_client_error_on_a_later_page_raises_at_once_and_is_not_a_fallback(
        self, discover, sleeps
    ) -> None:
        keyset = KeysetServer(total=10, failures={2: [status(404)]})
        with pytest.raises(httpx.HTTPStatusError) as err:
            discover(keyset)  # no offset route mounted: a fallback would fail differently
        assert err.value.response.status_code == 404 and sleeps == []

    def test_an_invalid_cursor_mid_stream_raises(self, discover, sleeps) -> None:
        bad = httpx.Response(422, json={"type": "validation error", "error": "invalid cursor"})
        keyset = KeysetServer(total=10, failures={2: [bad]})
        with pytest.raises(httpx.HTTPStatusError) as err:
            discover(keyset)
        assert err.value.response.status_code == 422 and len(keyset.requests) == 3

    @pytest.mark.parametrize(
        "body",
        [
            [tmax(0)],  # the offset shape
            "not json at all",
            {"data": []},  # no "markets"
            {"markets": None},
            {"markets": {"0": tmax(0)}},
            {"markets": [tmax(0)], "next_cursor": 12},
            {"markets": [tmax(0)], "next_cursor": ["a"]},
        ],
        ids=["list", "text", "no-markets", "null", "dict", "int-cursor", "list-cursor"],
    )
    def test_a_reply_that_is_not_a_keyset_page_is_a_value_error(
        self, discover, sleeps, body
    ) -> None:
        reply = (
            httpx.Response(200, text=body)
            if isinstance(body, str)
            else httpx.Response(200, json=body)
        )
        keyset = KeysetServer(total=10, failures={1: [reply]})
        with pytest.raises(ValueError, match="/markets/keyset"):
            discover(keyset)


# ---------------------------------------------------------------------------------------------
# A Gamma that does not know the keyset endpoint: the offset path, with its warning
# ---------------------------------------------------------------------------------------------
REFUSALS = [400, 404, 405, 410, 422, 501]


class TestAKeysetThatIsRefused:
    @pytest.mark.parametrize("code", REFUSALS)
    def test_a_refused_first_page_falls_back_to_offset_paging_and_warns(
        self, discover, sleeps, caplog, code
    ) -> None:
        keyset = KeysetServer(total=10, failures={0: [status(code)]})
        offset = OffsetServer(total=5)
        with caplog.at_level("WARNING", logger="pmlab"):
            found = discover(keyset, offset)
        assert [m.market_id for m in found] == ["0", "1", "2", "3", "4"]
        assert offset.calls == [0, 2, 4] and len(keyset.requests) == 1  # keyset asked once
        (warning,) = pmlab_warnings(caplog)
        for needle in ("/markets/keyset", str(code), "offset", "incomplete"):
            assert needle in warning, (needle, warning)

    @pytest.mark.parametrize("code", [401, 403])
    def test_credentials_and_blocks_are_not_a_missing_endpoint(
        self, discover, sleeps, code
    ) -> None:
        keyset = KeysetServer(total=10, failures={0: [status(code)]})
        with pytest.raises(httpx.HTTPStatusError) as err:
            discover(keyset)  # no offset route mounted: the offset path would meet the same wall
        assert err.value.response.status_code == code

    def test_the_fallback_keeps_the_end_of_the_offset_space(self, discover, sleeps, caplog) -> None:
        too_large = httpx.Response(
            422, json={"error": "offset too large, use /markets/keyset for deeper pagination"}
        )
        keyset = KeysetServer(total=10, failures={0: [status(404)]})
        offset = OffsetServer(total=10**6, failures={6: [too_large]})
        with caplog.at_level("WARNING", logger="pmlab"):
            found = discover(keyset, offset)
        assert [m.market_id for m in found] == [str(i) for i in range(6)]
        said = pmlab_warnings(caplog)
        assert len(said) == 2 and "falling back" in said[0] and "offset=6" in said[1]

    def test_the_fallback_retries_a_transient_page_and_has_the_page_cap(
        self, discover, sleeps, monkeypatch
    ) -> None:
        keyset = KeysetServer(total=10, failures={0: [status(404)]})
        offset = OffsetServer(total=5, failures={2: [status(503)]})
        assert len(discover(keyset, offset)) == 5 and sleeps == [1.0]
        for module in (gamma_client, async_gamma_client):
            monkeypatch.setattr(module, "_MAX_DISCOVERY_PAGES", 3)
        keyset = KeysetServer(total=10, failures={0: [status(404)]})
        with pytest.raises(RuntimeError, match="3 pages"):
            discover(keyset, OffsetServer(total=10**9))

    def test_the_fallback_returns_a_market_served_on_two_offset_pages_once(
        self, discover, sleeps
    ) -> None:
        """Offset paging under writes repeats the edge of a page; keyset paging never did, and the
        fallback holds the same promise (a market is returned once, the first copy winning)."""
        keyset = KeysetServer(total=10, failures={0: [status(404)]})
        found = discover(keyset, OffsetServer(total=7, overlap=1))
        assert [m.market_id for m in found] == [str(i) for i in range(7)]

    def test_the_fallback_is_for_the_first_request_only(self, discover, sleeps, caplog) -> None:
        keyset = KeysetServer(total=10)
        with caplog.at_level("WARNING", logger="pmlab"):
            discover(keyset)
        assert pmlab_warnings(caplog) == []


# ---------------------------------------------------------------------------------------------
# fetch_markets_all
# ---------------------------------------------------------------------------------------------
class TestFetchMarketsAll:
    def test_every_page_is_followed_by_its_cursor(self, sleeps) -> None:
        keyset = KeysetServer(total=5)
        found = run_fetch_all(keyset, max_results=1000)
        assert [m["id"] for m in found] == ["0", "1", "2", "3", "4"]
        assert keyset.cursors == [None, "cur1", "cur2"]
        assert all("offset" not in r for r in keyset.requests)

    def test_the_query_is_the_one_it_always_sent_plus_the_cursor(self, sleeps) -> None:
        keyset = KeysetServer(total=3)
        run_fetch_all(keyset, tag="weather", active=False, closed=True)
        assert keyset.requests[0] == {
            "limit": "2",
            "active": "false",
            "closed": "true",
            "tag": "weather",
        }
        assert (
            keyset.requests[1]["after_cursor"] == "cur1" and keyset.requests[1]["tag"] == "weather"
        )

    def test_the_cap_stops_asking_once_it_has_enough(self, sleeps) -> None:
        keyset = KeysetServer(total=10**9)
        assert len(run_fetch_all(keyset, max_results=3)) == 3
        assert keyset.cursors == [None, "cur1"]  # two pages hold three

    def test_a_cap_on_a_page_boundary_does_not_ask_for_one_more(self, sleeps) -> None:
        keyset = KeysetServer(total=10**9)
        assert len(run_fetch_all(keyset, max_results=4)) == 4
        assert keyset.cursors == [None, "cur1"]

    def test_the_keyword_filter_applies_after_paging_and_the_cap_to_the_hits(self, sleeps) -> None:
        keyset = KeysetServer(total=6)
        assert [m["id"] for m in run_fetch_all(keyset, keyword="city3")] == ["3"]

    def test_a_market_on_two_pages_is_returned_once_and_counts_once(self, sleeps) -> None:
        keyset = KeysetServer(pages=[[tmax(0), tmax(1)], [tmax(1), tmax(2)], [tmax(3), tmax(4)]])
        found = run_fetch_all(keyset, max_results=4)
        assert [m["id"] for m in found] == ["0", "1", "2", "3"]

    def test_nothing_to_fetch(self, sleeps) -> None:
        assert run_fetch_all(KeysetServer(pages=[[]])) == []

    def test_a_cap_of_zero_asks_for_nothing(self, sleeps) -> None:
        keyset = KeysetServer(total=10)
        assert run_fetch_all(keyset, max_results=0) == [] and keyset.requests == []

    def test_an_entry_that_is_not_a_market_is_passed_through(self, sleeps) -> None:
        """It was returned as it came before the keyset (no id to tell it by), and still is."""
        keyset = KeysetServer(pages=[[tmax(0), "odd", "odd"]])
        assert run_fetch_all(keyset, max_results=1000) == [tmax(0), "odd", "odd"]

    def test_a_transient_page_is_retried(self, sleeps) -> None:
        keyset = KeysetServer(total=6, failures={1: [status(429, **{"Retry-After": "2"})]})
        assert len(run_fetch_all(keyset, max_results=1000)) == 6 and sleeps == [2.0]

    def test_a_cursor_that_does_not_move_raises(self, sleeps) -> None:
        stuck = httpx.Response(200, json={"markets": [tmax(0), tmax(1)], "next_cursor": "cur1"})
        keyset = KeysetServer(total=10**9, failures={0: [stuck], 1: [stuck]})
        with pytest.raises(RuntimeError, match="same next_cursor"):
            run_fetch_all(keyset, max_results=1000)

    def test_a_refused_keyset_falls_back_to_offsets_and_warns(self, sleeps, caplog) -> None:
        keyset = KeysetServer(total=10, failures={0: [status(404)]})
        offset = OffsetServer(total=5)
        with caplog.at_level("WARNING", logger="pmlab"):
            found = run_fetch_all(keyset, offset, max_results=1000)
        assert [m["id"] for m in found] == ["0", "1", "2", "3", "4"]
        assert offset.calls == [0, 2, 4]
        assert "falling back" in " ".join(pmlab_warnings(caplog))

    def test_the_offset_fallback_returns_a_repeated_market_once(self, sleeps) -> None:
        """``fetch_markets_all`` used to return what offset paging served, repeats included."""
        keyset = KeysetServer(total=10, failures={0: [status(404)]})
        found = run_fetch_all(keyset, OffsetServer(total=7, overlap=1), max_results=1000)
        assert [m["id"] for m in found] == [str(i) for i in range(7)]

    def test_a_market_with_only_a_condition_id_is_returned_once(self, sleeps) -> None:
        market = {"conditionId": "0xc0", "question": "Will it?"}
        keyset = KeysetServer(pages=[[market, tmax(1)], [market]])
        assert run_fetch_all(keyset, max_results=1000) == [market, tmax(1)]

    def test_a_later_error_raises(self, sleeps) -> None:
        keyset = KeysetServer(total=10, failures={1: [status(404)]})
        with pytest.raises(httpx.HTTPStatusError):
            run_fetch_all(keyset, max_results=1000)

    def test_a_reply_that_is_not_a_keyset_page_is_a_value_error(self, sleeps) -> None:
        keyset = KeysetServer(total=10, failures={0: [httpx.Response(200, json=[tmax(0)])]})
        with pytest.raises(ValueError, match="/markets/keyset"):
            run_fetch_all(keyset)


def test_page_size_is_still_validated_before_any_request() -> None:
    with respx.mock, GammaClient(base_url=BASE) as client:  # nothing mounted: any call fails
        for bad in (0, 101, True, "5"):
            with pytest.raises(ValueError, match="page_size"):
                client.discover_tmax_markets(page_size=bad)  # type: ignore[arg-type]
            with pytest.raises(ValueError, match="page_size"):
                client.fetch_markets_all(page_size=bad)  # type: ignore[arg-type]


def test_the_log_is_quiet_when_nothing_goes_wrong(sleeps, caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="pmlab"):
        run_sync(KeysetServer(total=7))
    assert pmlab_warnings(caplog) == []
