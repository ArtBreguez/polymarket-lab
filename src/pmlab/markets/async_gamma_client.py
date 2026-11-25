"""Async Polymarket Gamma API client."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from pmlab.markets.gamma_client import (
    _MAX_DISCOVERY_PAGES,
    _RETRY_ATTEMPTS,
    _TRANSIENT_ERRORS,
    _TRANSIENT_STATUS,
    TmaxMarketInfo,
    _as_page,
    _build_tmax_market_info,
    _check_page_size,
    _is_tmax_market,
    _retry_delay,
)

GAMMA_API_BASE = "https://gamma-api.polymarket.com"
__all__ = ["AsyncGammaClient"]

_sleep = asyncio.sleep  # a name of its own so a test can replace it


class AsyncGammaClient:
    """Async client for the Polymarket Gamma API (market discovery), on ``httpx.AsyncClient``.

    Use it as an ``async with`` block, or call :meth:`aclose` when done.

    Args:
        base_url: The Gamma API root.
        timeout: Per-request timeout, in seconds.
    """

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
        """Fetch one page of markets, optionally filtered by tag and by a keyword in the question.

        Args:
            tag: Gamma tag to filter by.
            keyword: Keep only markets whose question contains this (case-insensitive).
            limit: Page size requested from the API.
            active: Only active markets.
            closed: Include closed markets.

        Returns:
            The raw market dicts from the API.

        Raises:
            httpx.HTTPStatusError: The API answered with an error status.
        """
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

    async def _get_page(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """One page of ``/markets``; a timeout, a connection cut mid-request, a 408, 429
        or 5xx is retried with backoff (up to five tries), any other failure is raised."""
        from pmlab.logging import get_logger

        url = f"{self.base_url}/markets"
        for attempt in range(_RETRY_ATTEMPTS):
            final = attempt == _RETRY_ATTEMPTS - 1
            try:
                resp = await self._client.get(url, params=params)
            except _TRANSIENT_ERRORS as exc:
                if final:
                    raise
                delay = _retry_delay(attempt, None)
                reason = type(exc).__name__
            else:
                if resp.status_code not in _TRANSIENT_STATUS or final:
                    resp.raise_for_status()
                    return _as_page(resp.json())
                delay = _retry_delay(attempt, resp)
                reason = f"HTTP {resp.status_code}"
            get_logger(__name__).warning(
                "Gamma /markets offset=%s: %s, retrying in %.1fs (attempt %d of %d)",
                params.get("offset"),
                reason,
                delay,
                attempt + 1,
                _RETRY_ATTEMPTS,
            )
            await _sleep(delay)
        raise AssertionError("unreachable: the last attempt returns or raises")

    async def discover_tmax_markets(
        self,
        page_size: int = 100,
        active: bool = True,
    ) -> list[TmaxMarketInfo]:
        """Paginate through all active markets and return temperature-max markets.

        Paginates using offset-based iteration, stopping when a page has fewer
        than *page_size* results.  Each page is filtered with the temperature-max
        heuristic and parsed into `TmaxMarketInfo`.

        A page that fails for a transient reason (a timeout, a connection cut mid-request, a 408,
        429 or 5xx; a numeric ``Retry-After`` is honoured) is retried with backoff, so one failure
        does not lose the pages before it; after five tries, or on any other error (a refused
        connection included: nothing is listening, waiting does not help), the failure is raised.

        Args:
            page_size: Results per API page (max 100).
            active: If True, only fetch active markets.

        Returns:
            List of parsed `TmaxMarketInfo` for every matching market.

        Raises:
            httpx.HTTPStatusError: If a page keeps failing, or fails with an error that is not
                transient (a 4xx other than 408 and 429).
            httpx.TransportError: If the connection keeps failing.
            ValueError: If Gamma answers with something other than a list of markets, or
                ``page_size`` is not an int between 1 and 100.
            RuntimeError: If it is still receiving full pages after 1000 of them (an API that
                ignores ``offset`` would otherwise never let it finish).
        """
        _check_page_size(page_size)
        results: list[TmaxMarketInfo] = []
        offset = 0

        for _ in range(_MAX_DISCOVERY_PAGES):
            params: dict[str, Any] = {
                "limit": page_size,
                "offset": offset,
                "active": active,
                "closed": False,
            }
            page = await self._get_page(params)

            for market in page:
                if _is_tmax_market(market):
                    info = _build_tmax_market_info(market)
                    if info is not None:
                        results.append(info)

            if len(page) < page_size:
                return results
            offset += page_size

        raise RuntimeError(
            f"discover_tmax_markets: still receiving full pages after {_MAX_DISCOVERY_PAGES} "
            f"pages ({offset} markets); stopping (is the API ignoring `offset`?)"
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> AsyncGammaClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()
