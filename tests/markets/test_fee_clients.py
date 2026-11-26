"""The four clients read fee fields (#71): CLOB ``/fee-rate`` and the Gamma market by slug, cached.

Replies are the ones recorded on 2026-11-26 (see ``test_fees.py``); no test here touches the network.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from pmlab.core.fee_schedule import FeeSchedule, category_fee_schedule
from pmlab.markets.async_clob_client import AsyncClobClient
from pmlab.markets.async_gamma_client import AsyncGammaClient
from pmlab.markets.cache import DiskCache
from pmlab.markets.clob_client import ClobClient
from pmlab.markets.gamma_client import GammaClient

FIXTURES = Path(__file__).parent / "fixtures" / "fees"
GAMMA = "https://mock-gamma.test"
CLOB = "https://mock-clob.test"
SPORTS_SLUG = "cs2-aim1-kamui-2026-11-26-map-handicap-home-1pt5"


def recorded(name: str) -> Any:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def market(name: str) -> dict[str, Any]:
    data: dict[str, Any] = recorded(f"gamma_{name}")
    return data


def yes_token(name: str) -> str:
    return str(json.loads(market(name)["clobTokenIds"])[0])


def warnings_of(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def serve_gamma(*markets: dict[str, Any]) -> respx.Route:
    """``/markets?slug=<slug>`` answers the market with that slug, ``[]`` for any other."""

    def reply(request: httpx.Request) -> httpx.Response:
        slug = request.url.params["slug"]
        return httpx.Response(200, json=[m for m in markets if m["slug"] == slug])

    return respx.get(f"{GAMMA}/markets").mock(side_effect=reply)


def serve_fee_rate(**by_token: Any) -> respx.Route:
    def reply(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=by_token[request.url.params["token_id"]])

    return respx.get(f"{CLOB}/fee-rate").mock(side_effect=reply)


class TestClobFetchFeeRate:
    @respx.mock
    def test_reads_base_fee_of_the_token(self) -> None:
        route = respx.get(f"{CLOB}/fee-rate").mock(
            return_value=httpx.Response(200, json=recorded("clob_fee_rate_weather"))
        )
        with ClobClient(base_url=CLOB) as client:
            assert client.fetch_fee_rate("tok-1") == 1000
        assert route.calls.last.request.url.params["token_id"] == "tok-1"

    @respx.mock
    def test_a_fee_free_market_is_zero_not_unknown(self) -> None:
        respx.get(f"{CLOB}/fee-rate").mock(
            return_value=httpx.Response(200, json=recorded("clob_fee_rate_geopolitics_fee_free"))
        )
        with ClobClient(base_url=CLOB) as client:
            assert client.fetch_fee_rate("tok-1") == 0

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
    def test_a_reply_that_says_nothing_is_unknown_and_not_cached(
        self, reply: httpx.Response, tmp_path: Path
    ) -> None:
        route = respx.get(f"{CLOB}/fee-rate").mock(return_value=reply)
        cache = DiskCache(tmp_path, ttl_seconds=3600)
        with ClobClient(base_url=CLOB, cache=cache) as client:
            assert client.fetch_fee_rate("tok-1") is None
            assert client.fetch_fee_rate("tok-1") is None
        assert route.call_count == 2  # a failure is asked again, never remembered

    @respx.mock
    def test_a_failed_request_is_logged_with_the_token(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        respx.get(f"{CLOB}/fee-rate").mock(return_value=httpx.Response(500))
        with caplog.at_level("WARNING", logger="pmlab"), ClobClient(base_url=CLOB) as client:
            client.fetch_fee_rate("tok-77")
        assert any("tok-77" in w for w in warnings_of(caplog))

    @respx.mock
    @pytest.mark.parametrize("token", ["", None, 5])
    def test_a_token_that_is_not_an_id_is_refused_without_a_request(self, token: object) -> None:
        route = respx.get(f"{CLOB}/fee-rate").mock(return_value=httpx.Response(200, json={}))
        with ClobClient(base_url=CLOB) as client, pytest.raises(ValueError, match="token"):
            client.fetch_fee_rate(token)  # type: ignore[arg-type]
        assert route.call_count == 0

    @respx.mock
    def test_inside_the_ttl_it_is_not_asked_again_and_a_zero_is_remembered(
        self, tmp_path: Path
    ) -> None:
        route = serve_fee_rate(a={"base_fee": 1000}, b={"base_fee": 0})
        cache = DiskCache(tmp_path, ttl_seconds=3600)
        with ClobClient(base_url=CLOB, cache=cache) as client:
            assert [client.fetch_fee_rate(t) for t in ("a", "b", "a", "b")] == [1000, 0, 1000, 0]
        assert route.call_count == 2

    @respx.mock
    def test_after_the_ttl_it_is_asked_again(self, tmp_path: Path) -> None:
        route = serve_fee_rate(a={"base_fee": 1000})
        with ClobClient(base_url=CLOB, cache=DiskCache(tmp_path, ttl_seconds=0)) as client:
            client.fetch_fee_rate("a")
            client.fetch_fee_rate("a")
        assert route.call_count == 2


class TestAsyncClobFetchFeeRate:
    @respx.mock
    async def test_reads_base_fee_and_caches_it(self, tmp_path: Path) -> None:
        route = serve_fee_rate(a={"base_fee": 1000}, b={"base_fee": 0})
        cache = DiskCache(tmp_path, ttl_seconds=3600)
        async with AsyncClobClient(base_url=CLOB, cache=cache) as client:
            assert await client.fetch_fee_rate("a") == 1000
            assert await client.fetch_fee_rate("b") == 0
            assert await client.fetch_fee_rate("a") == 1000
        assert route.call_count == 2

    @respx.mock
    async def test_unknown_when_the_reply_says_nothing(self) -> None:
        respx.get(f"{CLOB}/fee-rate").mock(return_value=httpx.Response(404))
        async with AsyncClobClient(base_url=CLOB) as client:
            assert await client.fetch_fee_rate("a") is None

    @respx.mock
    async def test_after_the_ttl_it_is_asked_again(self, tmp_path: Path) -> None:
        route = serve_fee_rate(a={"base_fee": 1000})
        cache = DiskCache(tmp_path, ttl_seconds=0)
        async with AsyncClobClient(base_url=CLOB, cache=cache) as client:
            await client.fetch_fee_rate("a")
            await client.fetch_fee_rate("a")
        assert route.call_count == 2

    async def test_a_token_that_is_not_an_id_is_refused(self) -> None:
        async with AsyncClobClient(base_url=CLOB) as client:
            with pytest.raises(ValueError, match="token"):
                await client.fetch_fee_rate("")


class TestGammaFetchFeeSnapshot:
    @respx.mock
    def test_reads_the_schedule_of_the_market_with_that_slug(self) -> None:
        route = serve_gamma(market("sports_v3"), market("weather"))
        with GammaClient(base_url=GAMMA) as client:
            snap = client.fetch_fee_snapshot(SPORTS_SLUG)
        assert route.calls.last.request.url.params["slug"] == SPORTS_SLUG
        assert snap.slug == SPORTS_SLUG and snap.market_id == market("sports_v3")["id"]
        assert snap.schedule == FeeSchedule(rate=0.05, rebate_rate=0.15)
        assert snap.source == "gamma" and snap.fee_type == "sports_fees_v3"
        assert snap.base_fee is None and snap.seen_at.tzinfo is not None
        assert abs((datetime.now(UTC) - snap.seen_at).total_seconds()) < 60

    @respx.mock
    def test_an_unknown_slug_is_a_lookup_error_naming_it(self) -> None:
        serve_gamma(market("sports_v3"))
        with GammaClient(base_url=GAMMA) as client, pytest.raises(LookupError, match="no-such"):
            client.fetch_fee_snapshot("no-such-market")

    @respx.mock
    def test_the_recorded_empty_reply_for_an_unknown_slug(self) -> None:
        respx.get(f"{GAMMA}/markets").mock(
            return_value=httpx.Response(200, json=recorded("gamma_by_slug_unknown"))
        )
        with GammaClient(base_url=GAMMA) as client, pytest.raises(LookupError):
            client.fetch_fee_snapshot("whatever")

    @respx.mock
    def test_a_reply_that_is_not_a_list_of_markets_is_refused(self) -> None:
        respx.get(f"{GAMMA}/markets").mock(return_value=httpx.Response(200, json={"error": "x"}))
        with GammaClient(base_url=GAMMA) as client, pytest.raises(ValueError, match="markets"):
            client.fetch_fee_snapshot("a-slug")

    @respx.mock
    def test_an_http_error_is_raised(self) -> None:
        respx.get(f"{GAMMA}/markets").mock(return_value=httpx.Response(404))
        with GammaClient(base_url=GAMMA) as client, pytest.raises(httpx.HTTPStatusError):
            client.fetch_fee_snapshot("a-slug")

    @respx.mock
    @pytest.mark.parametrize("slug", ["", None, 7])
    def test_a_slug_that_is_not_one_is_refused_without_a_request(self, slug: object) -> None:
        route = serve_gamma(market("weather"))
        with GammaClient(base_url=GAMMA) as client, pytest.raises(ValueError, match="slug"):
            client.fetch_fee_snapshot(slug)  # type: ignore[arg-type]
        assert route.call_count == 0

    @respx.mock
    def test_a_market_with_no_fee_fields_falls_back_to_the_category_and_says_so(self) -> None:
        bare = {k: v for k, v in market("weather").items() if not k.startswith("fee")}
        serve_gamma(bare)
        with GammaClient(base_url=GAMMA) as client:
            snap = client.fetch_fee_snapshot(bare["slug"], category="weather")
        assert snap.fallback and snap.category == "weather"
        assert snap.schedule == category_fee_schedule("weather")

    @respx.mock
    def test_the_clob_flag_is_checked_against_gamma_through_the_first_token(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        serve_gamma(market("sports_v3"))
        fee = serve_fee_rate(**{yes_token("sports_v3"): {"base_fee": 1000}})
        with (
            caplog.at_level("WARNING", logger="pmlab"),
            GammaClient(base_url=GAMMA) as gamma,
            ClobClient(base_url=CLOB) as clob,
        ):
            snap = gamma.fetch_fee_snapshot(SPORTS_SLUG, clob=clob)
        assert fee.calls.last.request.url.params["token_id"] == yes_token("sports_v3")
        assert snap.base_fee == 1000 and snap.schedule.rate == 0.05  # 0.05, not 1000 / 10_000
        assert warnings_of(caplog) == []

    @respx.mock
    def test_a_clob_that_disagrees_warns_naming_the_market_and_gamma_wins(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        serve_gamma(market("sports_v3"))
        serve_fee_rate(**{yes_token("sports_v3"): {"base_fee": 0}})
        with (
            caplog.at_level("WARNING", logger="pmlab"),
            GammaClient(base_url=GAMMA) as gamma,
            ClobClient(base_url=CLOB) as clob,
        ):
            snap = gamma.fetch_fee_snapshot(SPORTS_SLUG, clob=clob)
        assert snap.schedule.rate == 0.05 and snap.source == "gamma" and snap.base_fee == 0
        (warning,) = warnings_of(caplog)
        assert SPORTS_SLUG in warning

    @respx.mock
    def test_a_market_without_tokens_is_not_sent_to_the_clob(self) -> None:
        serve_gamma(market("missing_fields"))
        fee = serve_fee_rate()
        slug = market("missing_fields")["slug"]
        with GammaClient(base_url=GAMMA) as gamma, ClobClient(base_url=CLOB) as clob:
            snap = gamma.fetch_fee_snapshot(slug, clob=clob)
        assert fee.call_count == 0 and snap.base_fee is None
        assert snap.schedule == FeeSchedule(rate=0.01, rebate_rate=0.25)

    @respx.mock
    def test_inside_the_ttl_the_market_is_not_fetched_again_and_the_sighting_is_the_first(
        self, tmp_path: Path
    ) -> None:
        route = serve_gamma(market("weather"))
        slug = market("weather")["slug"]
        cache = DiskCache(tmp_path, ttl_seconds=3600)
        with GammaClient(base_url=GAMMA, cache=cache) as client:
            first = client.fetch_fee_snapshot(slug)
            second = client.fetch_fee_snapshot(slug)
        assert route.call_count == 1
        assert second == first  # the same seen_at too: it was seen once, and when

    @respx.mock
    def test_after_the_ttl_it_is_fetched_again_and_a_fee_change_is_seen(
        self, tmp_path: Path
    ) -> None:
        old = market("sports_v3")
        new = {**old, "feeSchedule": {**old["feeSchedule"], "rate": 0.04}}
        replies = iter([[old], [new]])
        route = respx.get(f"{GAMMA}/markets").mock(
            side_effect=lambda request: httpx.Response(200, json=next(replies))
        )
        with GammaClient(base_url=GAMMA, cache=DiskCache(tmp_path, ttl_seconds=0)) as client:
            first = client.fetch_fee_snapshot(SPORTS_SLUG)
            second = client.fetch_fee_snapshot(SPORTS_SLUG)
        assert route.call_count == 2
        assert (first.schedule.rate, second.schedule.rate) == (0.05, 0.04)
        assert second.seen_at > first.seen_at

    @respx.mock
    def test_the_cache_keeps_only_what_a_fee_is_read_from(self, tmp_path: Path) -> None:
        noisy = {**market("weather"), "description": "x" * 5000, "volume": "123"}
        serve_gamma(noisy)
        cache = DiskCache(tmp_path, ttl_seconds=3600)
        with GammaClient(base_url=GAMMA, cache=cache) as client:
            client.fetch_fee_snapshot(noisy["slug"])
        (entry,) = list(tmp_path.glob("*.json"))
        assert "xxxx" not in entry.read_text() and "volume" not in entry.read_text()

    @respx.mock
    def test_the_category_is_not_part_of_what_is_cached(self, tmp_path: Path) -> None:
        bare = {k: v for k, v in market("weather").items() if not k.startswith("fee")}
        route = serve_gamma(bare)
        cache = DiskCache(tmp_path, ttl_seconds=3600)
        with GammaClient(base_url=GAMMA, cache=cache) as client:
            weather = client.fetch_fee_snapshot(bare["slug"], category="weather")
            sports = client.fetch_fee_snapshot(bare["slug"], category="sports")
        assert route.call_count == 1
        assert (weather.schedule.rate, sports.schedule.rate) == (0.01, 0.05)


class TestAsyncGammaFetchFeeSnapshot:
    @respx.mock
    async def test_reads_the_schedule_and_checks_the_clob_flag(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        serve_gamma(market("sports_v3"))
        serve_fee_rate(**{yes_token("sports_v3"): {"base_fee": 0}})
        async with AsyncGammaClient(base_url=GAMMA) as gamma, AsyncClobClient(base_url=CLOB) as c:
            with caplog.at_level("WARNING", logger="pmlab"):
                snap = await gamma.fetch_fee_snapshot(SPORTS_SLUG, clob=c)
        assert snap.schedule == FeeSchedule(rate=0.05, rebate_rate=0.15) and snap.base_fee == 0
        (warning,) = warnings_of(caplog)
        assert SPORTS_SLUG in warning

    @respx.mock
    async def test_an_unknown_slug_is_a_lookup_error_naming_it(self) -> None:
        serve_gamma(market("sports_v3"))
        async with AsyncGammaClient(base_url=GAMMA) as client:
            with pytest.raises(LookupError, match="no-such"):
                await client.fetch_fee_snapshot("no-such-market")

    @respx.mock
    async def test_inside_the_ttl_the_market_is_not_fetched_again(self, tmp_path: Path) -> None:
        route = serve_gamma(market("weather"))
        slug = market("weather")["slug"]
        cache = DiskCache(tmp_path, ttl_seconds=3600)
        async with AsyncGammaClient(base_url=GAMMA, cache=cache) as client:
            first = await client.fetch_fee_snapshot(slug)
            second = await client.fetch_fee_snapshot(slug)
        assert route.call_count == 1 and second == first

    @respx.mock
    async def test_after_the_ttl_it_is_fetched_again(self, tmp_path: Path) -> None:
        route = serve_gamma(market("weather"))
        slug = market("weather")["slug"]
        async with AsyncGammaClient(base_url=GAMMA, cache=DiskCache(tmp_path, 0)) as client:
            first = await client.fetch_fee_snapshot(slug)
            second = await client.fetch_fee_snapshot(slug)
        assert route.call_count == 2 and second.seen_at > first.seen_at

    @respx.mock
    async def test_a_fallback_is_recorded(self) -> None:
        bare = {k: v for k, v in market("weather").items() if not k.startswith("fee")}
        serve_gamma(bare)
        async with AsyncGammaClient(base_url=GAMMA) as client:
            snap = await client.fetch_fee_snapshot(bare["slug"], category="weather")
        assert snap.fallback and snap.schedule == category_fee_schedule("weather")
