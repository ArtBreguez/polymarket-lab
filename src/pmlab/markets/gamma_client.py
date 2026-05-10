# src/pmlab/markets/gamma_client.py
"""Polymarket Gamma API client — fetch open markets by tag/keyword."""
from __future__ import annotations

from typing import Any

import httpx

GAMMA_API_BASE = "https://gamma-api.polymarket.com"


def fetch_gamma_markets(
    tag: str | None = None,
    keyword: str | None = None,
    limit: int = 100,
    active: bool = True,
    closed: bool = False,
    base_url: str = GAMMA_API_BASE,
    client: httpx.Client | None = None,
) -> list[dict[str, Any]]:
    """Fetch markets from Polymarket Gamma API.

    Args:
        tag: Filter by tag (e.g. 'temperature', 'sports').
        keyword: Filter by keyword in question text (client-side filter).
        limit: Max results per page.
        active: Include active markets.
        closed: Include closed markets.
        base_url: Override API base URL (for testing).
        client: Optional httpx.Client (for testing / connection reuse).

    Returns:
        List of raw market dicts from API.
    """
    params: dict[str, Any] = {"limit": limit, "active": active, "closed": closed}
    if tag:
        params["tag"] = tag

    url = f"{base_url}/markets"
    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=30.0)

    try:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        markets: list[dict] = resp.json()
    finally:
        if own_client:
            client.close()

    if keyword:
        kw_lower = keyword.lower()
        markets = [m for m in markets if kw_lower in m.get("question", "").lower()]

    return markets


class GammaClient:
    """Stateful Gamma API client with optional TTL disk cache."""

    def __init__(
        self,
        base_url: str = GAMMA_API_BASE,
        timeout: float = 30.0,
        cache: "DiskCache | None" = None,
    ) -> None:
        from pmlab.markets.cache import DiskCache as _DiskCache  # noqa: F401 (type guard)
        self.base_url = base_url
        self._client = httpx.Client(timeout=timeout)
        self._cache = cache

    def fetch_markets(
        self,
        tag: str | None = None,
        keyword: str | None = None,
        limit: int = 100,
        active: bool = True,
        closed: bool = False,
    ) -> list[dict[str, Any]]:
        """Fetch markets, using cache if configured."""
        if self._cache is not None:
            cache_key = f"gamma:markets:{tag}:{keyword}:{limit}:{active}:{closed}"
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached  # type: ignore[return-value]

        result = fetch_gamma_markets(
            tag=tag, keyword=keyword, limit=limit,
            active=active, closed=closed,
            base_url=self.base_url, client=self._client,
        )

        if self._cache is not None:
            self._cache.set(cache_key, result)

        return result

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "GammaClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
