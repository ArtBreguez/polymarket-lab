"""The CLOB and Gamma clients refuse arguments and replies that are not what they claim (#63)."""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

from pmlab.markets.async_clob_client import AsyncClobClient
from pmlab.markets.clob_client import CLOB_API_BASE, ClobClient
from pmlab.markets.gamma_client import GammaClient

BASE = "https://mock-clob.test"


def _server(router: respx.MockRouter) -> None:
    def reply(request: httpx.Request) -> httpx.Response:
        token = request.url.params["token_id"]
        bodies = {
            "ok": {"mid": "0.42"},
            "nan": {"mid": "nan"},
            "big": {"mid": "1.5"},
            "neg": {"mid": "-0.2"},
            "inf": {"mid": "inf"},
            "null": {"mid": None},
            "zero": {"mid": "0"},
            "one": {"mid": "1"},
        }
        if token == "list":
            return httpx.Response(200, json=[1, 2])
        if token == "text":
            return httpx.Response(200, content=b"<html>", headers={"content-type": "text/html"})
        return httpx.Response(200, json=bodies[token])

    router.get(f"{BASE}/midpoint").mock(side_effect=reply)


TOKENS = ["ok", "nan", "big", "neg", "inf", "null", "list", "text", "zero", "one"]
EXPECTED = {"ok": 0.42, "zero": 0.0, "one": 1.0}


@respx.mock
def test_sync_client_keeps_only_midpoints_that_are_probabilities():
    _server(respx.mock)
    with ClobClient(base_url=BASE) as client:
        assert client.fetch_prices(TOKENS) == EXPECTED


@pytest.mark.asyncio
@respx.mock
async def test_async_client_keeps_only_midpoints_that_are_probabilities():
    _server(respx.mock)
    async with AsyncClobClient(base_url=BASE) as client:
        assert await client.fetch_prices(TOKENS) == EXPECTED


@pytest.mark.parametrize("bad", [0, -1])
def test_async_concurrency_must_be_positive(bad):
    with pytest.raises(ValueError, match="concurrency"):
        AsyncClobClient(base_url=BASE, concurrency=bad)


@pytest.mark.parametrize("bad", [None, 1.5, True, "10"])
def test_async_concurrency_must_be_an_int(bad):
    with pytest.raises(TypeError, match="concurrency"):
        AsyncClobClient(base_url=BASE, concurrency=bad)


@pytest.mark.asyncio
@respx.mock
async def test_async_concurrency_one_does_not_hang():
    _server(respx.mock)
    async with AsyncClobClient(base_url=BASE, concurrency=1) as client:
        got = await asyncio.wait_for(client.fetch_prices(["ok", "zero"]), 5)
    assert got == {"ok": 0.42, "zero": 0.0}


@pytest.mark.parametrize("page_size", [0, -1, 101, 500])
def test_gamma_page_size_is_between_1_and_100(page_size):
    """The API caps a page at 100; a larger page_size stopped after the first page, silently."""
    client = GammaClient(base_url="https://mock-gamma.test")
    with pytest.raises(ValueError, match="page_size"):
        client.fetch_markets_all(page_size=page_size)
    with pytest.raises(ValueError, match="page_size"):
        client.discover_tmax_markets(page_size=page_size)


def test_default_base_url_is_unchanged():
    assert ClobClient().base_url == CLOB_API_BASE
