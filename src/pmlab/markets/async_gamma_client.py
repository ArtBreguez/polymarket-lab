"""Async Polymarket Gamma API client."""

from __future__ import annotations

from typing import Any

import httpx

from pmlab.markets.gamma_client import (
    TmaxMarketInfo,
    _build_tmax_market_info,
    _is_tmax_market,
)

GAMMA_API_BASE = "https://gamma-api.polymarket.com"
__all__ = ["AsyncGammaClient"]


class AsyncGammaClient:
    def __init__(self, base_url: str = GAMMA_API_BASE, timeout: float = 30.0) -> None:
        self.base_url = base_url
        self._client = httpx.AsyncClient(timeout=timeout)

    async def fetch_markets(
        self,
        tag: str | None = None,
        keyword: str | None = None,
        limit: int = 100,
        active: bool = True,
        closed: bool = False,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit, "active": active, "closed": closed}
        if tag:
            params["tag"] = tag
        resp = await self._client.get(f"{self.base_url}/markets", params=params)
        resp.raise_for_status()
        markets: list[dict[str, Any]] = resp.json()
        if keyword:
            kw_lower = keyword.lower()
            markets = [m for m in markets if kw_lower in m.get("question", "").lower()]
        return markets

    async def discover_tmax_markets(
        self,
        page_size: int = 100,
        active: bool = True,
    ) -> list[TmaxMarketInfo]:
        """Paginate through all active markets and return temperature-max markets.

        Paginates using offset-based iteration, stopping when a page has fewer
        than *page_size* results.  Each page is filtered with the temperature-max
        heuristic and parsed into :class:`TmaxMarketInfo`.

        Args:
            page_size: Results per API page (max 100).
            active: If True, only fetch active markets.

        Returns:
            List of parsed :class:`TmaxMarketInfo` for every matching market.
        """
        results: list[TmaxMarketInfo] = []
        offset = 0

        while True:
            params: dict[str, Any] = {
                "limit": page_size,
                "offset": offset,
                "active": active,
                "closed": False,
            }
            resp = await self._client.get(f"{self.base_url}/markets", params=params)
            resp.raise_for_status()
            page: list[dict[str, Any]] = resp.json()

            for market in page:
                if _is_tmax_market(market):
                    info = _build_tmax_market_info(market)
                    if info is not None:
                        results.append(info)

            if len(page) < page_size:
                break
            offset += page_size

        return results

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> AsyncGammaClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()
