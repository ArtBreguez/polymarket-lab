# src/pmlab/markets/gamma_client.py
"""Polymarket Gamma API client — fetch open markets by tag/keyword."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from pmlab.markets.cache import DiskCache

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
    _client: httpx.Client = client if client is not None else httpx.Client(timeout=30.0)

    try:
        resp = _client.get(url, params=params)
        resp.raise_for_status()
        markets: list[dict[str, Any]] = resp.json()
    finally:
        if own_client:
            _client.close()

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
        cache: DiskCache | None = None,
    ) -> None:
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
            cached: list[dict[str, Any]] | None = self._cache.get(cache_key)
            if cached is not None:
                return cached

        result = fetch_gamma_markets(
            tag=tag, keyword=keyword, limit=limit,
            active=active, closed=closed,
            base_url=self.base_url, client=self._client,
        )

        if self._cache is not None:
            cache_key = f"gamma:markets:{tag}:{keyword}:{limit}:{active}:{closed}"
            self._cache.set(cache_key, result)

        return result

    def fetch_markets_all(
        self,
        tag: str | None = None,
        keyword: str | None = None,
        page_size: int = 100,
        max_results: int = 1000,
        active: bool = True,
        closed: bool = False,
    ) -> list[dict[str, Any]]:
        """Fetch all markets using offset-based pagination.

        Iterates pages until fewer than page_size results are returned
        or max_results is reached.

        Args:
            tag: Filter by tag.
            keyword: Client-side keyword filter applied after pagination.
            page_size: Results per page (max 100).
            max_results: Hard cap on total results returned.
            active: Include active markets.
            closed: Include closed markets.

        Returns:
            All matching markets up to max_results.
        """
        from pmlab.logging import get_logger
        _log = get_logger(__name__)

        all_markets: list[dict[str, Any]] = []
        offset = 0

        while len(all_markets) < max_results:
            params: dict[str, Any] = {
                "limit": page_size,
                "offset": offset,
                "active": active,
                "closed": closed,
            }
            if tag:
                params["tag"] = tag

            resp = self._client.get(f"{self.base_url}/markets", params=params)
            resp.raise_for_status()
            page: list[dict[str, Any]] = resp.json()

            if not page:
                break

            all_markets.extend(page)
            _log.debug("Paginated: fetched %d markets (offset=%d)", len(page), offset)

            if len(page) < page_size:
                break
            offset += page_size

        if keyword:
            kw_lower = keyword.lower()
            all_markets = [m for m in all_markets if kw_lower in m.get("question", "").lower()]

        return all_markets[:max_results]

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> GammaClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
