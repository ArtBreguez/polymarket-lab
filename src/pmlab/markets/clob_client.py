# src/pmlab/markets/clob_client.py
"""Polymarket CLOB read client — fetch current market prices."""

from __future__ import annotations

from typing import Any

import httpx

CLOB_API_BASE = "https://clob.polymarket.com"


def parse_midpoint(data: Any) -> float | None:
    """The midpoint of a ``/midpoint`` reply, or ``None`` if it is not a probability.

    A reply that is not an object, has no ``mid``, or whose ``mid`` is NaN, infinite or outside
    [0, 1] is a failed request, not a price: stored, it failed later in the middle of a backtest
    or was traded on (#63).
    """
    if not isinstance(data, dict):
        return None
    mid = data.get("mid")
    if mid is None:
        return None
    try:
        value = float(mid)
    except (TypeError, ValueError):
        return None
    return value if 0.0 <= value <= 1.0 else None


def fetch_token_prices(
    token_ids: list[str],
    base_url: str = CLOB_API_BASE,
    client: httpx.Client | None = None,
) -> dict[str, float]:
    """Fetch midpoint prices for a list of token IDs."""
    if not token_ids:
        return {}

    own_client = client is None
    _client: httpx.Client = client if client is not None else httpx.Client(timeout=15.0)

    prices: dict[str, float] = {}
    try:
        for token_id in token_ids:
            try:
                resp = _client.get(f"{base_url}/midpoint", params={"token_id": token_id})
                resp.raise_for_status()
                mid = parse_midpoint(resp.json())
                if mid is not None:
                    prices[token_id] = mid
            except (httpx.HTTPError, ValueError, KeyError):
                continue
    finally:
        if own_client:
            _client.close()

    return prices


class ClobClient:
    """Stateful CLOB read client with connection reuse."""

    def __init__(self, base_url: str = CLOB_API_BASE, timeout: float = 15.0) -> None:
        self.base_url = base_url
        self._client = httpx.Client(timeout=timeout)

    def fetch_prices(self, token_ids: list[str]) -> dict[str, float]:
        return fetch_token_prices(token_ids, base_url=self.base_url, client=self._client)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ClobClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
