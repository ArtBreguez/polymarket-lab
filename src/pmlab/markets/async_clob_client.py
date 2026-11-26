"""Async Polymarket CLOB read client."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import httpx

from pmlab.markets.clob_client import _check_token_id, parse_midpoint
from pmlab.markets.fees import parse_fee_rate

if TYPE_CHECKING:
    from pmlab.markets.cache import DiskCache

CLOB_API_BASE = "https://clob.polymarket.com"
__all__ = ["AsyncClobClient"]


class AsyncClobClient:
    """Async client for CLOB midpoint prices, on ``httpx.AsyncClient``.

    Use it as an ``async with`` block, or call :meth:`aclose` when done.

    Args:
        base_url: The CLOB API root.
        timeout: Per-request timeout, in seconds.
        concurrency: The most requests in flight at once.
        cache: Optional `DiskCache` for ``fetch_fee_rate``; an entry lives for the cache's TTL.
    """

    def __init__(
        self,
        base_url: str = CLOB_API_BASE,
        timeout: float = 15.0,
        concurrency: int = 10,
        cache: DiskCache | None = None,
    ) -> None:
        if isinstance(concurrency, bool) or not isinstance(concurrency, int):
            raise TypeError(f"concurrency must be an int >= 1, got {concurrency!r}")
        if concurrency < 1:
            # asyncio.Semaphore(0) would make every request wait for ever.
            raise ValueError(f"concurrency must be >= 1, got {concurrency}")
        self.base_url = base_url
        self._client = httpx.AsyncClient(timeout=timeout)
        self._sem = asyncio.Semaphore(concurrency)
        self._cache = cache

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

    async def fetch_fee_rate(self, token_id: str) -> int | None:
        """CLOB's ``base_fee`` for a token (``GET /fee-rate?token_id=``), in basis points.

        The same as `ClobClient.fetch_fee_rate`: a flag, not the fee rate; ``None`` when the request
        fails or the reply has no usable ``base_fee`` (not cached); cached for the cache's TTL
        otherwise, a 0 included.

        Args:
            token_id: A CLOB token id (an element of a Gamma market's ``clobTokenIds``).

        Returns:
            ``base_fee``, or ``None`` if it could not be read.

        Raises:
            ValueError: If ``token_id`` is not a non-empty string.
        """
        _check_token_id(token_id)
        key = f"clob:fee-rate:{token_id}"
        if self._cache is not None:
            cached = self._cache.get(key)
            if isinstance(cached, int) and not isinstance(cached, bool):
                return cached
        async with self._sem:
            try:
                resp = await self._client.get(
                    f"{self.base_url}/fee-rate", params={"token_id": token_id}
                )
                resp.raise_for_status()
                base_fee = parse_fee_rate(resp.json())
            except (httpx.HTTPError, ValueError) as exc:
                from pmlab.logging import get_logger

                get_logger(__name__).warning(
                    "CLOB /fee-rate failed for token %s: %s", token_id, exc
                )
                return None
        if base_fee is not None and self._cache is not None:
            self._cache.set(key, base_fee)
        return base_fee

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> AsyncClobClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()
