import httpx
import pytest
import respx

from pmlab.markets.gamma_client import GAMMA_API_BASE, GammaClient, fetch_gamma_markets


@respx.mock
def test_fetch_markets_returns_list():
    respx.get(f"{GAMMA_API_BASE}/markets").mock(
        return_value=httpx.Response(
            200,
            json=[
                {"id": "m1", "question": "Highest temperature in NYC on May 10?", "active": True},
                {"id": "m2", "question": "Will it rain?", "active": True},
            ],
        )
    )
    markets = fetch_gamma_markets()
    assert len(markets) == 2
    assert markets[0]["id"] == "m1"


@respx.mock
def test_fetch_markets_keyword_filter():
    respx.get(f"{GAMMA_API_BASE}/markets").mock(
        return_value=httpx.Response(
            200,
            json=[
                {"id": "m1", "question": "Highest temperature in NYC?"},
                {"id": "m2", "question": "Who wins the race?"},
            ],
        )
    )
    markets = fetch_gamma_markets(keyword="temperature")
    assert len(markets) == 1
    assert markets[0]["id"] == "m1"


@respx.mock
def test_fetch_markets_raises_on_error():
    respx.get(f"{GAMMA_API_BASE}/markets").mock(return_value=httpx.Response(500))
    with pytest.raises(httpx.HTTPStatusError):
        fetch_gamma_markets()


@respx.mock
def test_gamma_client_context_manager():
    respx.get(f"{GAMMA_API_BASE}/markets").mock(return_value=httpx.Response(200, json=[]))
    with GammaClient() as client:
        result = client.fetch_markets()
    assert result == []


@respx.mock
def test_empty_result():
    respx.get(f"{GAMMA_API_BASE}/markets").mock(return_value=httpx.Response(200, json=[]))
    markets = fetch_gamma_markets(tag="temperature")
    assert markets == []
