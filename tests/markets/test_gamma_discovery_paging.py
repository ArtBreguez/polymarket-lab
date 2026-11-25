"""`discover_tmax_markets` pages through every active market (#68).

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


def run_sync(server: Server) -> list[TmaxMarketInfo]:
    with respx.mock:
        respx.get(URL).mock(side_effect=server)
        with GammaClient(base_url=BASE) as client:
            return client.discover_tmax_markets(page_size=PAGE)


async def run_async(server: Server) -> list[TmaxMarketInfo]:
    with respx.mock:
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


def test_the_default_cap_is_far_above_any_real_universe() -> None:
    assert gamma_client._MAX_DISCOVERY_PAGES >= 1000
