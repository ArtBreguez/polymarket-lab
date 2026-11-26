"""The four clients read fee fields (#71): CLOB ``/fee-rate`` and the Gamma market by slug, cached.

Each test is written once and runs twice, for the blocking clients and for their async twins (the
issue asks that the two be identical), through ``Clients`` and the ``clients`` fixture. Replies are the
ones recorded on 2026-11-26 (see ``test_fees.py``); no test here touches the network.

Gamma serves one market by slug on ``/markets/slug/<slug>``, whatever its state (HTTP 404 and
``{"type": "not found error", ...}`` for a slug that is no market). ``/markets?slug=<slug>`` lists
open markets only and answers ``[]`` (HTTP 200) for a closed one, so a client built on it reports a
market that exists as unknown: ``serve_gamma`` serves both routes the way live Gamma does, and
``gamma_closed_crypto.json`` is a market that had closed that day.
"""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import httpx
import pytest
import respx

from pmlab.core.fee_schedule import FeeSchedule, category_fee_schedule
from pmlab.markets.async_clob_client import AsyncClobClient
from pmlab.markets.async_gamma_client import AsyncGammaClient
from pmlab.markets.cache import DiskCache
from pmlab.markets.clob_client import ClobClient
from pmlab.markets.fees import FeeSnapshot
from pmlab.markets.gamma_client import GammaClient

FIXTURES = Path(__file__).parent / "fixtures" / "fees"
GAMMA = "https://mock-gamma.test"
CLOB = "https://mock-clob.test"
SLUG_PATH = "/markets/slug/"
SPORTS_SLUG = "cs2-aim1-kamui-2026-11-26-map-handicap-home-1pt5"


def recorded(name: str) -> Any:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def market(name: str) -> dict[str, Any]:
    data: dict[str, Any] = recorded(f"gamma_{name}")
    return data


def yes_token(name: str) -> str:
    return str(json.loads(market(name)["clobTokenIds"])[0])


def without_fee_fields(name: str) -> dict[str, Any]:
    return {k: v for k, v in market(name).items() if not k.startswith("fee")}


def warnings_of(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def serve_gamma(*markets: dict[str, Any]) -> respx.Route:
    """Gamma as it answers live; returns the route of ``/markets/slug/<slug>`` (the one the clients
    use), so ``route.call_count`` counts the reads of a market.

    That route serves the market with the slug, closed or open, and 404s any other slug. The other,
    ``/markets?slug=<slug>``, lists open markets only (``[]`` for a closed one, unless it is sent
    ``closed=true``), as live Gamma does.
    """

    def by_path(request: httpx.Request) -> httpx.Response:
        slug = unquote(request.url.path.removeprefix(SLUG_PATH))
        found = [m for m in markets if m["slug"] == slug]
        if not found:
            return httpx.Response(404, json=recorded("gamma_slug_not_found"))
        return httpx.Response(200, json=found[0])

    def by_query(request: httpx.Request) -> httpx.Response:
        slug = request.url.params["slug"]
        closed = request.url.params.get("closed") == "true"
        listed = [m for m in markets if m["slug"] == slug and bool(m.get("closed")) == closed]
        return httpx.Response(200, json=listed or recorded("gamma_query_slug_of_a_closed_market"))

    respx.get(f"{GAMMA}/markets").mock(side_effect=by_query)
    return respx.get(host="mock-gamma.test", path__startswith=SLUG_PATH).mock(side_effect=by_path)


def serve_fee_rate(**by_token: Any) -> respx.Route:
    def reply(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=by_token[request.url.params["token_id"]])

    return respx.get(f"{CLOB}/fee-rate").mock(side_effect=reply)


class Clients:
    """A Gamma and a CLOB client of one kind (blocking or async) behind one awaitable interface."""

    def __init__(
        self,
        kind: str,
        gamma_cache: DiskCache | None = None,
        clob_cache: DiskCache | None = None,
    ) -> None:
        if kind == "sync":
            self.gamma: Any = GammaClient(base_url=GAMMA, cache=gamma_cache)
            self.clob: Any = ClobClient(base_url=CLOB, cache=clob_cache)
        else:
            self.gamma = AsyncGammaClient(base_url=GAMMA, cache=gamma_cache)
            self.clob = AsyncClobClient(base_url=CLOB, cache=clob_cache)

    async def rate(self, token: Any) -> int | None:
        result = self.clob.fetch_fee_rate(token)
        return await result if inspect.isawaitable(result) else result  # type: ignore[no-any-return]

    async def snapshot(
        self, slug: Any, *, category: str | None = None, with_clob: bool = False
    ) -> FeeSnapshot:
        kwargs: dict[str, Any] = {"category": category}
        if with_clob:
            kwargs["clob"] = self.clob
        result = self.gamma.fetch_fee_snapshot(slug, **kwargs)
        return await result if inspect.isawaitable(result) else result  # type: ignore[no-any-return]

    async def aclose(self) -> None:
        for client in (self.gamma, self.clob):
            closed = client.aclose() if hasattr(client, "aclose") else client.close()
            if inspect.isawaitable(closed):
                await closed


@pytest.fixture(params=["sync", "async"])
async def clients(request: pytest.FixtureRequest) -> AsyncIterator[Callable[..., Clients]]:
    made: list[Clients] = []

    def build(gamma_cache: DiskCache | None = None, clob_cache: DiskCache | None = None) -> Clients:
        made.append(Clients(request.param, gamma_cache, clob_cache))
        return made[-1]

    yield build
    for each in made:
        await each.aclose()


class TestFetchFeeRate:
    @respx.mock
    async def test_reads_base_fee_of_the_token(self, clients: Callable[..., Clients]) -> None:
        route = respx.get(f"{CLOB}/fee-rate").mock(
            return_value=httpx.Response(200, json=recorded("clob_fee_rate_weather"))
        )
        assert await clients().rate("tok-1") == 1000
        assert route.calls.last.request.url.params["token_id"] == "tok-1"

    @respx.mock
    async def test_a_fee_free_market_is_zero_not_unknown(
        self, clients: Callable[..., Clients]
    ) -> None:
        respx.get(f"{CLOB}/fee-rate").mock(
            return_value=httpx.Response(200, json=recorded("clob_fee_rate_geopolitics_fee_free"))
        )
        assert await clients().rate("tok-1") == 0

    @respx.mock
    @pytest.mark.parametrize(
        "reply",
        [
            httpx.Response(404, json={"error": "not found"}),
            httpx.Response(500),
            httpx.Response(200, text="<html>"),
            httpx.Response(200, json={}),
            httpx.Response(200, json={"base_fee": None}),
        ],
    )
    async def test_a_reply_that_says_nothing_is_unknown_and_not_cached(
        self, reply: httpx.Response, clients: Callable[..., Clients], tmp_path: Path
    ) -> None:
        route = respx.get(f"{CLOB}/fee-rate").mock(return_value=reply)
        c = clients(clob_cache=DiskCache(tmp_path, ttl_seconds=3600))
        assert await c.rate("tok-1") is None
        assert await c.rate("tok-1") is None
        assert route.call_count == 2  # a failure is asked again, never remembered

    @respx.mock
    async def test_a_failed_request_is_logged_with_the_token(
        self, clients: Callable[..., Clients], caplog: pytest.LogCaptureFixture
    ) -> None:
        respx.get(f"{CLOB}/fee-rate").mock(return_value=httpx.Response(500))
        with caplog.at_level("WARNING", logger="pmlab"):
            await clients().rate("tok-77")
        assert any("tok-77" in w for w in warnings_of(caplog))

    @respx.mock
    @pytest.mark.parametrize("token", ["", None, 5])
    async def test_a_token_that_is_not_an_id_is_refused_without_a_request(
        self, token: object, clients: Callable[..., Clients]
    ) -> None:
        route = respx.get(f"{CLOB}/fee-rate").mock(return_value=httpx.Response(200, json={}))
        with pytest.raises(ValueError, match="token"):
            await clients().rate(token)
        assert route.call_count == 0

    @respx.mock
    async def test_inside_the_ttl_it_is_not_asked_again_and_a_zero_is_remembered(
        self, clients: Callable[..., Clients], tmp_path: Path
    ) -> None:
        route = serve_fee_rate(a={"base_fee": 1000}, b={"base_fee": 0})
        c = clients(clob_cache=DiskCache(tmp_path, ttl_seconds=3600))
        assert [await c.rate(t) for t in ("a", "b", "a", "b")] == [1000, 0, 1000, 0]
        assert route.call_count == 2

    @respx.mock
    async def test_after_the_ttl_it_is_asked_again(
        self, clients: Callable[..., Clients], tmp_path: Path
    ) -> None:
        route = serve_fee_rate(a={"base_fee": 1000})
        c = clients(clob_cache=DiskCache(tmp_path, ttl_seconds=0))
        await c.rate("a")
        await c.rate("a")
        assert route.call_count == 2


class TestGammaFetchFeeSnapshot:
    @respx.mock
    async def test_reads_the_schedule_of_the_market_with_that_slug(
        self, clients: Callable[..., Clients]
    ) -> None:
        route = serve_gamma(market("sports_v3"), market("weather"))
        snap = await clients().snapshot(SPORTS_SLUG)
        assert route.calls.last.request.url.path == f"{SLUG_PATH}{SPORTS_SLUG}"
        assert snap.slug == SPORTS_SLUG and snap.market_id == market("sports_v3")["id"]
        assert snap.schedule == FeeSchedule(rate=0.05, rebate_rate=0.15)
        assert snap.source == "gamma" and snap.fee_type == "sports_fees_v3"
        assert snap.base_fee is None and snap.seen_at.tzinfo is not None
        assert abs((datetime.now(UTC) - snap.seen_at).total_seconds()) < 60

    @respx.mock
    async def test_a_closed_market_is_read_though_the_open_only_list_has_not_got_it(
        self, clients: Callable[..., Clients]
    ) -> None:
        closed = market("closed_crypto")
        assert closed["closed"] is True and closed["feesEnabled"] is True
        serve_gamma(closed, market("weather"))
        snap = await clients().snapshot(closed["slug"])
        assert snap.market_id == "7379374" and snap.slug == closed["slug"]
        assert snap.schedule == FeeSchedule(rate=0.01, rebate_rate=0.2)
        assert snap.source == "gamma" and not snap.fallback and snap.fee_type == "crypto_fees_v2"

    @respx.mock
    async def test_a_closed_market_checks_the_clob_flag_too(
        self, clients: Callable[..., Clients], caplog: pytest.LogCaptureFixture
    ) -> None:
        closed = market("closed_crypto")
        serve_gamma(closed)
        serve_fee_rate(**{yes_token("closed_crypto"): {"base_fee": 1000}})
        with caplog.at_level("WARNING", logger="pmlab"):
            snap = await clients().snapshot(closed["slug"], with_clob=True)
        assert snap.base_fee == 1000 and snap.schedule.rate == 0.01
        assert warnings_of(caplog) == []

    @respx.mock
    async def test_the_slug_is_sent_as_one_path_segment(
        self, clients: Callable[..., Clients]
    ) -> None:
        route = serve_gamma(market("weather"))
        with pytest.raises(LookupError):
            await clients().snapshot("odd slug/with?chars#x")
        assert (
            route.calls.last.request.url.raw_path == b"/markets/slug/odd%20slug%2Fwith%3Fchars%23x"
        )

    @respx.mock
    async def test_an_unknown_slug_is_a_lookup_error_naming_it(
        self, clients: Callable[..., Clients]
    ) -> None:
        serve_gamma(market("sports_v3"))
        with pytest.raises(LookupError, match="no-such-market"):
            await clients().snapshot("no-such-market")

    @respx.mock
    async def test_a_reply_that_is_not_a_market_object_is_refused(
        self, clients: Callable[..., Clients]
    ) -> None:
        respx.get(host="mock-gamma.test", path__startswith=SLUG_PATH).mock(
            return_value=httpx.Response(200, json=[market("weather")])
        )
        with pytest.raises(ValueError, match="market object"):
            await clients().snapshot("a-slug")

    @respx.mock
    async def test_a_reply_for_another_slug_is_not_that_market(
        self, clients: Callable[..., Clients]
    ) -> None:
        respx.get(host="mock-gamma.test", path__startswith=SLUG_PATH).mock(
            return_value=httpx.Response(200, json=market("weather"))
        )
        with pytest.raises(LookupError, match="a-slug"):
            await clients().snapshot("a-slug")

    @respx.mock
    async def test_an_http_error_is_raised(self, clients: Callable[..., Clients]) -> None:
        # 422 is what Gamma answers for a slug that cannot be one; it is not retried
        respx.get(host="mock-gamma.test", path__startswith=SLUG_PATH).mock(
            return_value=httpx.Response(422, json={"type": "validation error"})
        )
        with pytest.raises(httpx.HTTPStatusError):
            await clients().snapshot("a-slug")

    @respx.mock
    @pytest.mark.parametrize("slug", ["", None, 7])
    async def test_a_slug_that_is_not_one_is_refused_without_a_request(
        self, slug: object, clients: Callable[..., Clients]
    ) -> None:
        route = serve_gamma(market("weather"))
        with pytest.raises(ValueError, match="slug"):
            await clients().snapshot(slug)
        assert route.call_count == 0

    @respx.mock
    async def test_a_market_with_no_fee_fields_falls_back_to_the_category_and_says_so(
        self, clients: Callable[..., Clients]
    ) -> None:
        bare = without_fee_fields("weather")
        serve_gamma(bare)
        snap = await clients().snapshot(bare["slug"], category="weather")
        assert snap.fallback and snap.category == "weather"
        assert snap.schedule == category_fee_schedule("weather")

    @respx.mock
    async def test_the_clob_flag_is_checked_against_gamma_through_the_first_token(
        self, clients: Callable[..., Clients], caplog: pytest.LogCaptureFixture
    ) -> None:
        serve_gamma(market("sports_v3"))
        fee = serve_fee_rate(**{yes_token("sports_v3"): {"base_fee": 1000}})
        with caplog.at_level("WARNING", logger="pmlab"):
            snap = await clients().snapshot(SPORTS_SLUG, with_clob=True)
        assert fee.calls.last.request.url.params["token_id"] == yes_token("sports_v3")
        assert snap.base_fee == 1000 and snap.schedule.rate == 0.05  # 0.05, not 1000 / 10_000
        assert warnings_of(caplog) == []

    @respx.mock
    async def test_a_clob_that_disagrees_warns_naming_the_market_and_gamma_wins(
        self, clients: Callable[..., Clients], caplog: pytest.LogCaptureFixture
    ) -> None:
        serve_gamma(market("sports_v3"))
        serve_fee_rate(**{yes_token("sports_v3"): {"base_fee": 0}})
        with caplog.at_level("WARNING", logger="pmlab"):
            snap = await clients().snapshot(SPORTS_SLUG, with_clob=True)
        assert snap.schedule.rate == 0.05 and snap.source == "gamma" and snap.base_fee == 0
        (warning,) = warnings_of(caplog)
        assert SPORTS_SLUG in warning

    @respx.mock
    async def test_a_market_without_tokens_is_not_sent_to_the_clob(
        self, clients: Callable[..., Clients]
    ) -> None:
        serve_gamma(market("missing_fields"))
        fee = serve_fee_rate()
        snap = await clients().snapshot(market("missing_fields")["slug"], with_clob=True)
        assert fee.call_count == 0 and snap.base_fee is None
        assert snap.schedule == FeeSchedule(rate=0.01, rebate_rate=0.25)

    @respx.mock
    async def test_inside_the_ttl_the_market_is_not_fetched_again_and_the_sighting_is_the_first(
        self, clients: Callable[..., Clients], tmp_path: Path
    ) -> None:
        route = serve_gamma(market("weather"))
        slug = market("weather")["slug"]
        c = clients(gamma_cache=DiskCache(tmp_path, ttl_seconds=3600))
        first = await c.snapshot(slug)
        second = await c.snapshot(slug)
        assert route.call_count == 1
        assert second == first  # the same seen_at too: it was seen once, and when

    @respx.mock
    async def test_a_closed_market_is_cached_like_any_other(
        self, clients: Callable[..., Clients], tmp_path: Path
    ) -> None:
        route = serve_gamma(market("closed_crypto"))
        slug = market("closed_crypto")["slug"]
        c = clients(gamma_cache=DiskCache(tmp_path, ttl_seconds=3600))
        assert await c.snapshot(slug) == await c.snapshot(slug)
        assert route.call_count == 1

    @respx.mock
    async def test_after_the_ttl_it_is_fetched_again_and_a_fee_change_is_seen(
        self, clients: Callable[..., Clients], tmp_path: Path
    ) -> None:
        old = market("sports_v3")
        new = {**old, "feeSchedule": {**old["feeSchedule"], "rate": 0.04}}
        replies = iter([old, new])
        route = respx.get(host="mock-gamma.test", path__startswith=SLUG_PATH).mock(
            side_effect=lambda request: httpx.Response(200, json=next(replies))
        )
        c = clients(gamma_cache=DiskCache(tmp_path, ttl_seconds=0))
        first = await c.snapshot(SPORTS_SLUG)
        second = await c.snapshot(SPORTS_SLUG)
        assert route.call_count == 2
        assert (first.schedule.rate, second.schedule.rate) == (0.05, 0.04)
        assert second.seen_at > first.seen_at

    @respx.mock
    async def test_the_cache_keeps_only_what_a_fee_is_read_from(
        self, clients: Callable[..., Clients], tmp_path: Path
    ) -> None:
        noisy = {**market("weather"), "description": "x" * 5000, "volume": "123"}
        serve_gamma(noisy)
        c = clients(gamma_cache=DiskCache(tmp_path, ttl_seconds=3600))
        await c.snapshot(noisy["slug"])
        (entry,) = list(tmp_path.glob("*.json"))
        assert "xxxx" not in entry.read_text() and "volume" not in entry.read_text()
        assert "weather_fees" in entry.read_text()

    @respx.mock
    async def test_the_category_is_not_part_of_what_is_cached(
        self, clients: Callable[..., Clients], tmp_path: Path
    ) -> None:
        bare = without_fee_fields("weather")
        route = serve_gamma(bare)
        c = clients(gamma_cache=DiskCache(tmp_path, ttl_seconds=3600))
        weather = await c.snapshot(bare["slug"], category="weather")
        sports = await c.snapshot(bare["slug"], category="sports")
        assert route.call_count == 1
        assert (weather.schedule.rate, sports.schedule.rate) == (0.01, 0.05)

    @respx.mock
    async def test_a_lookup_error_is_not_cached(
        self, clients: Callable[..., Clients], tmp_path: Path
    ) -> None:
        route = serve_gamma()
        c = clients(gamma_cache=DiskCache(tmp_path, ttl_seconds=3600))
        for _ in range(2):
            with pytest.raises(LookupError):
                await c.snapshot("no-such-market")
        assert route.call_count == 2 and list(tmp_path.glob("*.json")) == []
