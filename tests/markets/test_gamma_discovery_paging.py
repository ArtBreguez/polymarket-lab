"""The offset fallback of `discover_tmax_markets` and `fetch_markets_all` (#68, #83).

Discovery pages by cursor on ``/markets/keyset`` (``test_gamma_keyset.py``). These tests run the
older offset paging of #68 and #79 the way it still runs: against a Gamma that does not know the
keyset endpoint (it answers 404), so every test below goes through the fallback, with its retry,
cap and "offset too large" handling intact.

`discover_tmax_markets` pages through every active market (#68).

It looped ``while True``: one 429 or 500 on page 40 lost the 39 pages before it, and an API that
ignored ``offset`` and kept returning full pages never let it finish. A transient failure of one
page is now retried with backoff (honouring ``Retry-After``), a failure that is not transient is
not, and the loop stops, loudly, after ``_MAX_DISCOVERY_PAGES`` pages. Sleeping is patched out.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from pmlab.markets import async_gamma_client, gamma_client
from pmlab.markets.async_gamma_client import AsyncGammaClient
from pmlab.markets.gamma_client import GammaClient, TmaxMarketInfo

BASE = "https://mock-gamma.test"
URL = f"{BASE}/markets"
KEYSET_URL = f"{BASE}/markets/keyset"
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


def page_of(offset: int, total: int) -> list[dict]:
    """The markets at [offset, offset + PAGE) of a universe of `total`, tmax markets throughout."""
    return [tmax(i) for i in range(offset, min(offset + PAGE, total))]


class Server:
    """A fake Gamma `/markets`: `total` markets, plus scripted failures per offset."""

    def __init__(self, total: int, failures: dict[int, list] | None = None) -> None:
        self.total = total
        self.failures = {k: list(v) for k, v in (failures or {}).items()}
        self.calls: list[int] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        offset = int(request.url.params["offset"])
        self.calls.append(offset)
        scripted = self.failures.get(offset)
        if scripted:
            outcome = scripted.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return httpx.Response(200, json=page_of(offset, self.total))


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    slept: list[float] = []

    async def fake_async(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(gamma_client, "_sleep", slept.append)
    monkeypatch.setattr(async_gamma_client, "_sleep", fake_async)
    return slept


def status(code: int, **headers: str) -> httpx.Response:
    return httpx.Response(code, headers=headers, json={"error": code})


def no_keyset() -> None:
    """Mount the keyset endpoint of an old Gamma: not found, so paging falls back to offsets."""
    respx.get(KEYSET_URL).mock(return_value=httpx.Response(404, json={"error": "not found"}))


def run_sync(server: Server) -> list[TmaxMarketInfo]:
    with respx.mock:
        no_keyset()
        respx.get(URL).mock(side_effect=server)
        with GammaClient(base_url=BASE) as client:
            return client.discover_tmax_markets(page_size=PAGE)


async def run_async(server: Server) -> list[TmaxMarketInfo]:
    with respx.mock:
        no_keyset()
        respx.get(URL).mock(side_effect=server)
        async with AsyncGammaClient(base_url=BASE) as client:
            return await client.discover_tmax_markets(page_size=PAGE)


@pytest.fixture(params=["sync", "async"])
def discover(request):
    """`discover(server)` for the sync client, or the async one driven to completion."""
    if request.param == "sync":
        return run_sync

    import asyncio

    return lambda server: asyncio.run(run_async(server))


class TestTheHappyPathIsUnchanged:
    def test_every_page_is_read_and_only_tmax_markets_come_back(self, discover, sleeps) -> None:
        server = Server(total=5)
        found = discover(server)
        assert [m.market_id for m in found] == ["0", "1", "2", "3", "4"]
        assert server.calls == [0, 2, 4]  # the short third page ends it
        assert sleeps == []

    def test_a_universe_that_ends_on_a_full_page_gets_one_empty_page(
        self, discover, sleeps
    ) -> None:
        server = Server(total=4)
        assert len(discover(server)) == 4
        assert server.calls == [0, 2, 4]

    def test_nothing_to_discover(self, discover, sleeps) -> None:
        assert discover(Server(total=0)) == []


class TestATransientFailureCostsOnlyThatPage:
    @pytest.mark.parametrize("code", [429, 500, 502, 503, 504, 408])
    def test_a_transient_status_on_a_late_page_is_retried_and_nothing_is_lost(
        self, discover, sleeps, code
    ) -> None:
        server = Server(total=7, failures={4: [status(code)]})
        found = discover(server)
        assert [m.market_id for m in found] == [str(i) for i in range(7)]
        assert server.calls == [0, 2, 4, 4, 6]  # only offset 4 was asked twice
        assert sleeps == [1.0]

    def test_the_wait_doubles_between_attempts(self, discover, sleeps) -> None:
        server = Server(total=3, failures={2: [status(500), status(503), status(502)]})
        assert len(discover(server)) == 3
        assert sleeps == [1.0, 2.0, 4.0]

    def test_retry_after_is_honoured_for_a_429(self, discover, sleeps) -> None:
        server = Server(total=3, failures={0: [status(429, **{"Retry-After": "7"})]})
        assert len(discover(server)) == 3
        assert sleeps == [7.0]

    def test_retry_after_is_capped_and_an_unusable_one_falls_back_to_the_backoff(
        self, discover, sleeps
    ) -> None:
        server = Server(
            total=3,
            failures={
                0: [status(429, **{"Retry-After": "3600"}), status(429, **{"Retry-After": "soon"})]
            },
        )
        assert len(discover(server)) == 3
        assert sleeps == [60.0, 2.0]

    @pytest.mark.parametrize(
        "error",
        [
            httpx.ReadError("reset"),
            httpx.ReadTimeout("slow"),
            httpx.ConnectTimeout("no answer"),
            httpx.RemoteProtocolError("cut"),
        ],
    )
    def test_a_timeout_or_a_connection_cut_mid_request_is_retried(
        self, discover, sleeps, error
    ) -> None:
        server = Server(total=3, failures={2: [error]})
        assert len(discover(server)) == 3
        assert server.calls == [0, 2, 2]
        assert sleeps == [1.0]

    def test_the_backoff_is_capped(self, discover, sleeps, monkeypatch) -> None:
        for module in (gamma_client, async_gamma_client):
            monkeypatch.setattr(module, "_RETRY_ATTEMPTS", 9)
        server = Server(total=1, failures={0: [status(500)] * 8})
        assert len(discover(server)) == 1
        assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0, 30.0]


class TestWhatIsNotTransientIsNotRetried:
    @pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
    def test_a_client_error_raises_at_once(self, discover, sleeps, code) -> None:
        server = Server(total=7, failures={2: [status(code)]})
        with pytest.raises(httpx.HTTPStatusError) as err:
            discover(server)
        assert err.value.response.status_code == code
        assert server.calls == [0, 2] and sleeps == []

    def test_a_transient_failure_that_persists_raises_after_the_attempts(
        self, discover, sleeps
    ) -> None:
        server = Server(total=7, failures={2: [status(503)] * 10})
        with pytest.raises(httpx.HTTPStatusError) as err:
            discover(server)
        assert err.value.response.status_code == 503
        assert server.calls == [0, 2, 2, 2, 2, 2]  # five attempts at offset 2, page 0 not redone
        assert sleeps == [1.0, 2.0, 4.0, 8.0]

    def test_a_connection_error_that_persists_is_raised_as_it_is(self, discover, sleeps) -> None:
        server = Server(total=3, failures={0: [httpx.ReadError("reset")] * 10})
        with pytest.raises(httpx.ReadError):
            discover(server)
        assert len(server.calls) == 5

    def test_a_refused_connection_raises_at_once(self, discover, sleeps) -> None:
        """No network, a wrong base_url: nothing is listening, and waiting 1 + 2 + 4 + 8 s for it
        (as 1.1.3 never did) only delays the same error."""
        server = Server(total=7, failures={2: [httpx.ConnectError("refused")] * 10})
        with pytest.raises(httpx.ConnectError, match="refused"):
            discover(server)
        assert server.calls == [0, 2] and sleeps == []

    def test_a_refused_connection_on_the_first_page_raises_at_once_too(
        self, discover, sleeps
    ) -> None:
        server = Server(total=7, failures={0: [httpx.ConnectError("refused")] * 10})
        with pytest.raises(httpx.ConnectError):
            discover(server)
        assert server.calls == [0] and sleeps == []

    def test_a_reply_that_is_not_a_list_is_a_value_error(self, discover, sleeps) -> None:
        server = Server(total=3, failures={0: [httpx.Response(200, json={"error": "nope"})]})
        with pytest.raises(ValueError, match="list"):
            discover(server)


class TestARunawayLoopStops:
    @pytest.fixture(autouse=True)
    def _cap(self, monkeypatch):
        for module in (gamma_client, async_gamma_client):
            monkeypatch.setattr(module, "_MAX_DISCOVERY_PAGES", 3)

    def test_full_pages_for_ever_raise_after_the_cap(self, discover, sleeps) -> None:
        server = Server(total=10**9)  # an API that never runs out (or ignores `offset`)
        with pytest.raises(RuntimeError, match="3 pages") as err:
            discover(server)
        assert "offset" in str(err.value)
        assert server.calls == [0, 2, 4]

    def test_a_universe_that_ends_on_the_last_allowed_page_is_fine(self, discover, sleeps) -> None:
        server = Server(total=5)  # pages at 0, 2 and 4: the third is short
        assert len(discover(server)) == 5
        assert server.calls == [0, 2, 4]

    def test_a_retried_page_counts_once(self, discover, sleeps) -> None:
        server = Server(total=5, failures={0: [status(500)], 2: [status(500)], 4: [status(500)]})
        assert len(discover(server)) == 5


def test_the_default_cap_is_a_guard_of_at_least_1000_pages() -> None:
    assert gamma_client._MAX_DISCOVERY_PAGES >= 1000


# ---------------------------------------------------------------------------------------------
# The end of the offset space. Live Gamma answers HTTP 422 {"error": "offset too large, use
# /markets/keyset for deeper pagination"} from offset 2100 on (200 up to 2000), so a discovery of
# every active market failed at page 22 and threw away the 21 it had read (#68, review of #79).
# ---------------------------------------------------------------------------------------------
OFFSET_TOO_LARGE = {"error": "offset too large, use /markets/keyset for deeper pagination"}


def offset_too_large() -> httpx.Response:
    return httpx.Response(422, json=OFFSET_TOO_LARGE)


# A 422 whose body merely mentions the offset is a bad request, not the end of the offset space:
# only the sentence Gamma uses ("offset too large", or the "/markets/keyset" it points to) is.
NOT_THE_END = [
    {"error": "offset must be an integer"},
    {"detail": [{"loc": ["query", "offset"], "msg": "value is not a valid integer"}]},
    {"detail": "limit invalid for query limit=100&offset=200"},
    "<html><body>Unprocessable Entity: offset</body></html>",
]
THE_END = [
    OFFSET_TOO_LARGE,
    {"error": "OFFSET TOO LARGE, USE /MARKETS/KEYSET FOR DEEPER PAGINATION"},
    {"error": "offset too large"},
    {"error": "use /markets/keyset"},
]


def reply_422(body: object) -> httpx.Response:
    if isinstance(body, str):
        return httpx.Response(422, text=body)
    return httpx.Response(422, json=body)


def run_fetch_all(server: Server, **kwargs) -> list[dict]:
    with respx.mock:
        no_keyset()
        respx.get(URL).mock(side_effect=server)
        with GammaClient(base_url=BASE) as client:
            return client.fetch_markets_all(page_size=PAGE, **kwargs)


class TestTheEndOfTheOffsetSpace:
    def test_discovery_returns_what_it_read_and_warns(self, discover, sleeps, caplog) -> None:
        server = Server(total=10**6, failures={6: [offset_too_large()]})
        with caplog.at_level("WARNING", logger="pmlab"):
            found = discover(server)
        assert [m.market_id for m in found] == [str(i) for i in range(6)]
        assert server.calls == [0, 2, 4, 6] and sleeps == []  # not retried: nothing to wait for
        warnings_ = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings_) == 2, warnings_  # the keyset refusal, then the end of the offsets
        assert "falling back" in warnings_[0], warnings_[0]
        for needle in ("offset=6", "incomplete", "/markets/keyset", "6 markets"):
            assert needle in warnings_[1], (needle, warnings_[1])

    def test_it_is_the_end_whatever_the_page_it_comes_on(self, discover, sleeps) -> None:
        server = Server(total=10**6, failures={2: [offset_too_large()]})
        assert [m.market_id for m in discover(server)] == ["0", "1"]

    def test_a_422_on_the_first_page_is_an_error_not_an_end(self, discover, sleeps) -> None:
        server = Server(total=10**6, failures={0: [offset_too_large()]})
        with pytest.raises(httpx.HTTPStatusError) as err:
            discover(server)
        assert err.value.response.status_code == 422 and server.calls == [0]

    def test_a_422_that_is_not_about_the_offset_still_raises(self, discover, sleeps) -> None:
        other_422 = httpx.Response(422, json={"error": "limit must be at most 100"})
        server = Server(total=10**6, failures={2: [other_422]})
        with pytest.raises(httpx.HTTPStatusError):
            discover(server)

    @pytest.mark.parametrize("body", NOT_THE_END)
    def test_a_422_that_only_mentions_the_offset_still_raises(
        self, discover, sleeps, body: object
    ) -> None:
        server = Server(total=10**6, failures={2: [reply_422(body)]})
        with pytest.raises(httpx.HTTPStatusError) as err:
            discover(server)
        assert err.value.response.status_code == 422
        assert server.calls == [0, 2] and sleeps == []

    @pytest.mark.parametrize("body", THE_END)
    def test_either_phrase_of_gamma_s_answer_is_the_end(
        self, discover, sleeps, body: object
    ) -> None:
        server = Server(total=10**6, failures={2: [reply_422(body)]})
        assert [m.market_id for m in discover(server)] == ["0", "1"]

    def test_a_422_after_a_transient_failure_is_still_the_end(self, discover, sleeps) -> None:
        server = Server(total=10**6, failures={2: [status(503), offset_too_large()]})
        assert [m.market_id for m in discover(server)] == ["0", "1"]
        assert server.calls == [0, 2, 2] and sleeps == [1.0]

    def test_the_default_run_that_reaches_the_end_of_a_real_sized_universe(
        self, discover, sleeps
    ) -> None:
        """21 full pages, then the 422 Gamma gives at offset 2100 (page_size 100 in real life)."""
        server = Server(total=10**6, failures={42: [offset_too_large()]})
        found = discover(server)
        assert len(found) == 42 and server.calls[-1] == 42


class TestFetchMarketsAllAtTheEndOfTheOffsetSpace:
    def test_it_returns_what_it_read_and_warns(self, sleeps, caplog) -> None:
        server = Server(total=10**6, failures={6: [offset_too_large()]})
        with caplog.at_level("WARNING", logger="pmlab"):
            found = run_fetch_all(server, max_results=1000)
        assert [m["id"] for m in found] == [str(i) for i in range(6)]
        assert server.calls == [0, 2, 4, 6]
        text = " ".join(r.getMessage() for r in caplog.records if r.levelname == "WARNING")
        assert "offset=6" in text and "incomplete" in text and "/markets/keyset" in text

    def test_the_keyword_filter_and_the_cap_still_apply_to_what_was_read(self, sleeps) -> None:
        server = Server(total=10**6, failures={8: [offset_too_large()]})
        found = run_fetch_all(server, max_results=1000, keyword="city1")
        assert [m["id"] for m in found] == ["1"]
        server = Server(total=10**6, failures={8: [offset_too_large()]})
        assert len(run_fetch_all(server, max_results=3)) == 3

    def test_a_422_on_the_first_page_raises(self, sleeps) -> None:
        server = Server(total=10**6, failures={0: [offset_too_large()]})
        with pytest.raises(httpx.HTTPStatusError):
            run_fetch_all(server)

    def test_a_422_that_is_not_about_the_offset_raises(self, sleeps) -> None:
        server = Server(total=10**6, failures={2: [status(422)]})
        with pytest.raises(httpx.HTTPStatusError):
            run_fetch_all(server)

    @pytest.mark.parametrize("body", NOT_THE_END)
    def test_a_422_that_only_mentions_the_offset_raises(self, sleeps, body: object) -> None:
        server = Server(total=10**6, failures={2: [reply_422(body)]})
        with pytest.raises(httpx.HTTPStatusError) as err:
            run_fetch_all(server)
        assert err.value.response.status_code == 422 and server.calls == [0, 2]

    @pytest.mark.parametrize("body", THE_END)
    def test_either_phrase_of_gamma_s_answer_is_the_end(self, sleeps, body: object) -> None:
        server = Server(total=10**6, failures={2: [reply_422(body)]})
        assert [m["id"] for m in run_fetch_all(server, max_results=1000)] == ["0", "1"]

    def test_a_cap_below_the_end_never_meets_it(self, sleeps, caplog) -> None:
        server = Server(total=10**6, failures={6: [offset_too_large()]})
        with caplog.at_level("WARNING", logger="pmlab"):
            assert len(run_fetch_all(server, max_results=4)) == 4
        assert server.calls == [0, 2] and "refused offset" not in caplog.text  # only the fallback's
