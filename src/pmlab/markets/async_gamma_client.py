"""Async Polymarket Gamma API client."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import Any

import httpx

from pmlab.markets.gamma_client import (
    _KEYSET_PATH,
    _MAX_DISCOVERY_PAGES,
    _OFFSET_PATH,
    _RETRY_ATTEMPTS,
    _TRANSIENT_ERRORS,
    _TRANSIENT_STATUS,
    DAILY_TEMPERATURE_TAG_ID,
    TmaxMarketInfo,
    _as_page,
    _check_page_size,
    _check_tag_id,
    _keyset_refused,
    _KeysetWalk,
    _offset_capped,
    _offset_exhausted,
    _retry_delay,
    _TmaxCollector,
    _unseen,
    _warn_keyset_refused,
    _warn_offset_exhausted,
    _warn_tag_matched_nothing,
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
        """Fetch one page of markets, optionally filtered by a keyword in the question.

        Args:
            tag: Sent as the ``tag`` query parameter. Live Gamma ignores it (it filters by
                ``tag_id``), so it narrows nothing there; use *keyword* for a client-side filter.
                `discover_tmax_markets` takes ``tag_id``.
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

    async def _send(self, path: str, params: dict[str, Any], label: str) -> httpx.Response:
        """GET *path*; a timeout, a connection cut mid-request, a 408, 429 or 5xx is retried with
        backoff (up to five tries), and the reply that ends it is returned for the caller to read
        (after the last try, the failing one). Transport errors that are not transient are raised
        as they come."""
        from pmlab.logging import get_logger

        url = f"{self.base_url}{path}"
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
                    return resp
                delay = _retry_delay(attempt, resp)
                reason = f"HTTP {resp.status_code}"
            get_logger(__name__).warning(
                "Gamma %s %s: %s, retrying in %.1fs (attempt %d of %d)",
                path,
                label,
                reason,
                delay,
                attempt + 1,
                _RETRY_ATTEMPTS,
            )
            await _sleep(delay)
        raise AssertionError("unreachable: the last attempt returns or raises")

    async def _pages(
        self, where: str, params: dict[str, Any], page_size: int
    ) -> AsyncGenerator[list[dict[str, Any]], None]:
        """The pages of ``/markets/keyset``, each without the markets of the pages before it.

        Falls back to `_offset_pages` when the first reply says the endpoint is not served here.
        """
        seen: set[str] = set()
        walk = _KeysetWalk(where, params)
        while walk.pages < _MAX_DISCOVERY_PAGES:
            resp = await self._send(_KEYSET_PATH, walk.params, walk.label)
            if walk.pages == 0 and _keyset_refused(resp):
                _warn_keyset_refused(where, resp)
                async for page in self._offset_pages(where, params, page_size):
                    yield page
                return
            resp.raise_for_status()
            yield _unseen(walk.digest(resp), seen)
            if walk.finished:
                if walk.read == 0:
                    _warn_tag_matched_nothing(where, params)
                return
        raise walk.capped()

    async def _offset_pages(
        self, where: str, params: dict[str, Any], page_size: int
    ) -> AsyncGenerator[list[dict[str, Any]], None]:
        """The pages of ``/markets`` by offset: the fallback, which Gamma ends at about offset 2000
        with an HTTP 422 (taken as the end: warned about, and what was read is kept)."""
        seen: set[str] = set()
        offset = 0
        for _ in range(_MAX_DISCOVERY_PAGES):
            resp = await self._send(_OFFSET_PATH, {**params, "offset": offset}, f"offset={offset}")
            if _offset_exhausted(resp, offset):
                _warn_offset_exhausted(where, offset, resp)
                return
            resp.raise_for_status()
            page = _as_page(resp.json())
            yield _unseen(page, seen)
            if len(page) < page_size:
                if offset == 0 and not page:
                    _warn_tag_matched_nothing(where, params)
                return
            offset += page_size
        raise _offset_capped(where, _MAX_DISCOVERY_PAGES, offset)

    async def discover_tmax_markets(
        self,
        page_size: int = 100,
        active: bool = True,
        *,
        tag_id: int | None = DAILY_TEMPERATURE_TAG_ID,
    ) -> list[TmaxMarketInfo]:
        """Page through the active markets of one tag and return the temperature-max ones.

        Scoped to Gamma's ``Daily Temperature`` tag (103040) unless *tag_id* says otherwise, which
        is 27 pages on live Gamma (2,633 markets on 2026-11-25), not the universe of more than
        100,000. Pages by cursor on ``/markets/keyset``: each reply's ``next_cursor`` is sent back as
        ``after_cursor`` until a reply has none, so the whole tag is read. A market served on two
        pages is read once.

        Two kinds of market are returned, one `TmaxMarketInfo` each. A **per-bin** market (what
        live Gamma lists: one Yes/No market per bin, ``Will the highest temperature in Warsaw be
        -4°C or below on November 26?``) is grouped with the others of its city and day into one
        entry: ``outcome_labels`` are the bins from the lowest to the highest (``"-6°C or below"``,
        ``"-5°C"``, ... ``"3°C or higher"``; ``"88-89°F"``), ``token_ids`` their Yes tokens and
        ``outcome_prices`` their Yes prices (NaN where Gamma gave none), ``market_id`` is the event's
        id on Gamma, ``target_date`` is ISO, and ``end_date`` is the earliest close of its bins. An
        event with some bins closed already has only the rest. A **multi-outcome** market (one
        market whose outcomes are ``30°C``, ``31°C``, ...) is parsed as before; those are listed
        first. *Lowest* temperature markets are not read. What a bin means (a bin holds the whole
        degrees that round into it), and what is refused, is `pmlab.markets.tmax_bins`; refused
        markets are counted in one warning.

        The cap is a loop guard, not a budget: it counts pages (1000, so 100,000 markets at
        ``page_size=100``), and a walk still receiving a ``next_cursor`` after that raises. Scoped
        by a tag that is nowhere near it; ``tag_id=None`` reads every market, and the live universe
        is larger than the cap, so that call raises after 1000 requests (minutes: 209 s in the
        fastest of our walks, about 27 minutes in the slowest).

        A page that fails for a transient reason (a timeout, a connection cut mid-request, a 408,
        429 or 5xx; a numeric ``Retry-After`` is honoured) is retried with backoff, so one failure
        does not lose the pages before it; after five tries, or on any other error (a refused
        connection included: nothing is listening, waiting does not help), the failure is raised.

        A Gamma that does not serve ``/markets/keyset`` (it answers the first page with a 4xx other
        than 401, 403, 408 and 429) is read by ``offset`` on ``/markets`` instead, with a warning:
        that stops at about 2000 markets (HTTP 422 "offset too large", taken as the end, and warned
        about), so the result is incomplete and an empty one with that warning means "not read".

        Args:
            page_size: Results per API page (1 to 100; Gamma serves at most 100).
            active: If True, only fetch active markets.
            tag_id: Keyword-only. The Gamma tag to read, sent as ``tag_id`` on every page; the
                default is ``Daily Temperature`` (103040, ``DAILY_TEMPERATURE_TAG_ID``). ``None``
                reads every market (see the cap above). A tag Gamma does not know reads no market
                and logs a warning saying so.

        Returns:
            One `TmaxMarketInfo` per multi-outcome market read and per city and day of per-bin
            markets read (the second kind has the bins of the event as its outcomes).

        Raises:
            httpx.HTTPStatusError: If a page keeps failing, or fails with an error that is not
                transient (a 4xx other than 408 and 429; on the first page, one that says the
                endpoint is not served falls back to offsets instead).
            httpx.TransportError: If the connection keeps failing.
            ValueError: If Gamma answers with something that is not a page of markets,
                ``page_size`` is not an int between 1 and 100, or ``tag_id`` is not a positive int
                (or None).
            RuntimeError: If the cursor loops (a ``next_cursor`` it has sent before), or pages are
                still coming after 1000 of them. The cap counts pages, not markets: ``page_size=100``
                allows 100,000 markets and ``page_size=10`` allows 10,000.
        """
        _check_page_size(page_size)
        _check_tag_id(tag_id)
        params: dict[str, Any] = {"limit": page_size, "active": active, "closed": False}
        if tag_id is not None:
            params["tag_id"] = tag_id
        collector = _TmaxCollector()
        async with aclosing(self._pages("discover_tmax_markets", params, page_size)) as pages:
            async for page in pages:
                collector.add_page(page)
        return collector.result("discover_tmax_markets")

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> AsyncGammaClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()
