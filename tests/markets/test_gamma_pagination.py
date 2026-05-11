"""Tests for GammaClient pagination."""
from __future__ import annotations

import pytest
import respx
import httpx

from pmlab.markets.gamma_client import GammaClient


class TestFetchMarketsAll:
    @respx.mock
    def test_single_page(self):
        # Only one page returned (less than page_size)
        respx.get("https://mock-gamma.test/markets").mock(
            return_value=httpx.Response(200, json=[{"question": "Q1"}, {"question": "Q2"}])
        )
        client = GammaClient(base_url="https://mock-gamma.test")
        markets = client.fetch_markets_all(page_size=100)
        assert len(markets) == 2
        client.close()

    @respx.mock
    def test_two_pages(self):
        # First page full (3 items = page_size), second page partial
        call_count = 0
        def side_effect(request):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(200, json=[{"question": f"Q{i}"} for i in range(3)])
            return httpx.Response(200, json=[{"question": "Q_last"}])

        respx.get("https://mock-gamma.test/markets").mock(side_effect=side_effect)
        client = GammaClient(base_url="https://mock-gamma.test")
        markets = client.fetch_markets_all(page_size=3)
        assert len(markets) == 4
        client.close()

    @respx.mock
    def test_empty_response(self):
        respx.get("https://mock-gamma.test/markets").mock(
            return_value=httpx.Response(200, json=[])
        )
        client = GammaClient(base_url="https://mock-gamma.test")
        markets = client.fetch_markets_all()
        assert markets == []
        client.close()

    @respx.mock
    def test_keyword_filter(self):
        respx.get("https://mock-gamma.test/markets").mock(
            return_value=httpx.Response(200, json=[
                {"question": "Will it rain?"},
                {"question": "Will it snow?"},
            ])
        )
        client = GammaClient(base_url="https://mock-gamma.test")
        markets = client.fetch_markets_all(keyword="rain")
        assert len(markets) == 1
        client.close()

    @respx.mock
    def test_max_results_cap(self):
        respx.get("https://mock-gamma.test/markets").mock(
            return_value=httpx.Response(200, json=[{"question": f"Q{i}"} for i in range(5)])
        )
        client = GammaClient(base_url="https://mock-gamma.test")
        markets = client.fetch_markets_all(page_size=5, max_results=3)
        assert len(markets) == 3
        client.close()
