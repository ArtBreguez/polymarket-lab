"""Async Polymarket CLOB read client."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from pmlab.markets.clob_client import parse_midpoint

CLOB_API_BASE = "https://clob.polymarket.com"
__all__ = ["AsyncClobClient"]


class AsyncClobClient:
    """Async client for CLOB midpoint prices, on ``httpx.AsyncClient``.

    Use it as an ``async with`` block, or call :meth:`aclose` when done.

    Args:
        base_url: The CLOB API root.
        timeout: Per-request timeout, in seconds.
        concurrency: The most requests in flight at once.
    """

    def __init__(
        self, base_url: str = CLOB_API_BASE, timeout: float = 15.0, concurrency: int = 10
    ) -> None:
        if isinstance(concurrency, bool) or not isinstance(concurrency, int):
            raise TypeError(f"concurrency must be an int >= 1, got {concurrency!r}")
        if concurrency < 1:
            # asyncio.Semaphore(0) would make every request wait for ever.
            raise ValueError(f"concurrency must be >= 1, got {concurrency}")
        self.base_url = base_url
        self._client = httpx.AsyncClient(timeout=timeout)
        self._sem = asyncio.Semaphore(concurrency)

    async def fetch_prices(self, token_ids: list[str]) -> dict[str, float]:
        """Fetch the midpoint price of each token concurrently.

        Args:
            token_ids: CLOB token ids.

        Returns:
            ``{token_id: midpoint}``. A token whose request failed, or that has no midpoint, is
            absent from the result rather than raising.
        """
        if not token_ids:
            return {}
        results: dict[str, float] = {}
        tasks = [self._fetch_one(tid, results) for tid in token_ids]
        await asyncio.gather(*tasks, return_exceptions=True)
        return results

    async def _fetch_one(self, token_id: str, results: dict[str, float]) -> None:
        async with self._sem:
            try:
                resp = await self._client.get(
                    f"{self.base_url}/midpoint", params={"token_id": token_id}
                )
                resp.raise_for_status()
                mid = parse_midpoint(resp.json())
                if mid is not None:
                    results[token_id] = mid
            except (httpx.HTTPError, ValueError, KeyError):
                pass

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> AsyncClobClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()
