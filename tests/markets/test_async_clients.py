"""Tests for async Gamma and CLOB clients."""

from __future__ import annotations

import httpx
import pytest
import respx

from pmlab.markets.async_clob_client import AsyncClobClient
from pmlab.markets.async_gamma_client import AsyncGammaClient


class TestAsyncGammaClient:
    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_markets(self):
        respx.get("https://mock-gamma.test/markets").mock(
            return_value=httpx.Response(200, json=[{"question": "Will it rain?"}])
        )
        async with AsyncGammaClient(base_url="https://mock-gamma.test") as client:
            markets = await client.fetch_markets()
        assert len(markets) == 1

    @pytest.mark.asyncio
    @respx.mock
    async def test_keyword_filter(self):
        respx.get("https://mock-gamma.test/markets").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {"question": "Will it rain in London?"},
                    {"question": "Will it snow in Paris?"},
                ],
            )
        )
        async with AsyncGammaClient(base_url="https://mock-gamma.test") as client:
            markets = await client.fetch_markets(keyword="rain")
        assert len(markets) == 1

    @pytest.mark.asyncio
    async def test_context_manager(self):
        async with AsyncGammaClient() as client:
            assert client._client is not None


class TestAsyncClobClient:
    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_prices(self):
        respx.get("https://mock-clob.test/midpoint").mock(
            return_value=httpx.Response(200, json={"mid": "0.65"})
        )
        async with AsyncClobClient(base_url="https://mock-clob.test") as client:
            prices = await client.fetch_prices(["tok1", "tok2"])
        assert "tok1" in prices
        assert prices["tok1"] == pytest.approx(0.65)

    @pytest.mark.asyncio
    async def test_empty_token_ids(self):
        async with AsyncClobClient() as client:
            prices = await client.fetch_prices([])
        assert prices == {}

    @pytest.mark.asyncio
    @respx.mock
    async def test_failed_token_skipped(self):
        respx.get("https://mock-clob.test/midpoint").mock(return_value=httpx.Response(500))
        async with AsyncClobClient(base_url="https://mock-clob.test") as client:
            prices = await client.fetch_prices(["bad-tok"])
        assert prices == {}
