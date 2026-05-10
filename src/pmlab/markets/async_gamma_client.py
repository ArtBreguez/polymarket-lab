"""Async Polymarket Gamma API client."""
from __future__ import annotations
from typing import Any
import httpx

GAMMA_API_BASE = "https://gamma-api.polymarket.com"
__all__ = ["AsyncGammaClient"]

class AsyncGammaClient:
    def __init__(self, base_url: str = GAMMA_API_BASE, timeout: float = 30.0) -> None:
        self.base_url = base_url
        self._client = httpx.AsyncClient(timeout=timeout)

    async def fetch_markets(self, tag: str | None = None, keyword: str | None = None, limit: int = 100, active: bool = True, closed: bool = False) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit, "active": active, "closed": closed}
        if tag:
            params["tag"] = tag
        resp = await self._client.get(f"{self.base_url}/markets", params=params)
        resp.raise_for_status()
        markets: list[dict] = resp.json()
        if keyword:
            kw_lower = keyword.lower()
            markets = [m for m in markets if kw_lower in m.get("question", "").lower()]
        return markets

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "AsyncGammaClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()
