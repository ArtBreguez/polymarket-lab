import httpx
import respx

from pmlab.markets.clob_client import CLOB_API_BASE, ClobClient, fetch_token_prices


@respx.mock
def test_fetch_prices_returns_dict():
    respx.get(f"{CLOB_API_BASE}/midpoint").mock(
        return_value=httpx.Response(200, json={"mid": "0.35"})
    )
    prices = fetch_token_prices(["token_abc"])
    assert "token_abc" in prices
    assert abs(prices["token_abc"] - 0.35) < 1e-9


def test_fetch_prices_empty_list():
    # No HTTP call should be made
    prices = fetch_token_prices([])
    assert prices == {}


@respx.mock
def test_fetch_prices_skips_failed_tokens():
    # First token fails, second succeeds — use side_effect for reliable ordering
    respx.get(f"{CLOB_API_BASE}/midpoint").mock(
        side_effect=[
            httpx.Response(404),
            httpx.Response(200, json={"mid": "0.60"}),
        ]
    )
    prices = fetch_token_prices(["bad", "good"])
    assert "bad" not in prices
    assert "good" in prices
    assert abs(prices["good"] - 0.60) < 1e-9


@respx.mock
def test_clob_client_context_manager():
    respx.get(f"{CLOB_API_BASE}/midpoint").mock(
        return_value=httpx.Response(200, json={"mid": "0.50"})
    )
    with ClobClient() as client:
        prices = client.fetch_prices(["tok1"])
    assert "tok1" in prices
