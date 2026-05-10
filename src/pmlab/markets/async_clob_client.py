"""Async Polymarket CLOB read client."""
from __future__ import annotations
import asyncio
from typing import Any
import httpx

CLOB_API_BASE = "https://clob.polymarket.com"
__all__ = ["AsyncClobClient"]

class AsyncClobClient:
    def __init__(self, base_url: str = CLOB_API_BASE, timeout: float = 15.0, concurrency: int = 10) -> None:
        self.base_url = base_url
        self._client = httpx.AsyncClient(timeout=timeout)
        self._sem = asyncio.Semaphore(concurrency)

    async def fetch_prices(self, token_ids: list[str]) -> dict[str, float]:
        if not token_ids:
            return {}
        results: dict[str, float] = {}
        tasks = [self._fetch_one(tid, results) for tid in token_ids]
        await asyncio.gather(*tasks, return_exceptions=True)
        return results

    async def _fetch_one(self, token_id: str, results: dict[str, float]) -> None:
        async with self._sem:
            try:
                resp = await self._client.get(f"{self.base_url}/midpoint", params={"token_id": token_id})
                resp.raise_for_status()
                data = resp.json()
                mid = data.get("mid")
                if mid is not None:
                    results[token_id] = float(mid)
            except (httpx.HTTPError, ValueError, KeyError):
                pass

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "AsyncClobClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()
