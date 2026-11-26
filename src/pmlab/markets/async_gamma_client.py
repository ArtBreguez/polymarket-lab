"""Async Polymarket Gamma API client."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import aclosing
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx

from pmlab.markets.fees import FeeSnapshot, fee_fields, first_token_id, resolve_fee_snapshot
from pmlab.markets.gamma_client import (
    _KEYSET_PATH,
    _MAX_DISCOVERY_PAGES,
    _OFFSET_PATH,
    _RETRY_ATTEMPTS,
    _TRANSIENT_ERRORS,
    _TRANSIENT_STATUS,
    TmaxMarketInfo,
    _as_page,
    _check_page_size,
    _check_slug,
    _check_tag_id,
    _keyset_refused,
    _KeysetWalk,
    _market_of_slug,
    _offset_capped,
    _offset_exhausted,
    _retry_delay,
    _slug_path,
    _TmaxCollector,
    _unseen,
    _warn_keyset_refused,
    _warn_offset_exhausted,
    _warn_tag_matched_nothing,
)
from pmlab.markets.tmax_bins import DAILY_TEMPERATURE_TAG_ID

if TYPE_CHECKING:
    from pmlab.markets.async_clob_client import AsyncClobClient
    from pmlab.markets.cache import DiskCache

GAMMA_API_BASE = "https://gamma-api.polymarket.com"
__all__ = ["AsyncGammaClient"]

_sleep = asyncio.sleep  # a name of its own so a test can replace it


class AsyncGammaClient:
    """Async client for the Polymarket Gamma API (market discovery), on ``httpx.AsyncClient``.

    Use it as an ``async with`` block, or call :meth:`aclose` when done.

    Args:
        base_url: The Gamma API root.
        timeout: Per-request timeout, in seconds.
        cache: Optional `DiskCache` for ``fetch_fee_snapshot``; an entry lives for the cache's own
            TTL (`DiskCache` defaults to one hour; ``PMLAB_CACHE_TTL`` is not read here). Keys
            carry the slug, not ``base_url``: give a staging and a production client each its own
            cache directory.
    """

    def __init__(
        self,
        base_url: str = GAMMA_API_BASE,
        timeout: float = 30.0,
        cache: DiskCache | None = None,
    ) -> None:
        self.base_url = base_url
        self._client = httpx.AsyncClient(timeout=timeout)
        self._cache = cache

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

    async def fetch_fee_snapshot(
        self,
        slug: str,
        *,
        category: str | None = None,
        clob: AsyncClobClient | None = None,
    ) -> FeeSnapshot:
        """The fee schedule of the market with this slug, as it is on Gamma now.

        The same as `GammaClient.fetch_fee_snapshot` (``GET /markets/slug/{slug}``, so a closed
        market is read too), with an `AsyncClobClient` for the optional
        ``base_fee`` check: Gamma's ``feeSchedule`` is the schedule, a ``base_fee`` that disagrees
        on whether a fee is charged only warns (naming the market), and the category default
        stands in only when Gamma gives no schedule (see `resolve_fee_snapshot`).

        Args:
            slug: The market's slug.
            category: The category whose default is the fallback if Gamma gives no schedule
                (``"weather"``, ``"sports"``).
            clob: An `AsyncClobClient` to read ``base_fee`` with, or ``None`` to skip that check.

        Returns:
            The snapshot, seen now.

        Raises:
            LookupError: If Gamma has no market with that slug (it answers 404).
            ValueError: If ``slug`` is not a non-empty string, Gamma's reply is not a market object,
                or a fallback is needed and ``category`` is missing or unknown.
            httpx.HTTPStatusError: If Gamma answers with another error status (422 for a slug that
                cannot be one).
        """
        _check_slug(slug)
        entry = await self._fee_entry(slug)
        market = entry["market"]
        token = first_token_id(market)
        base_fee = await clob.fetch_fee_rate(token) if clob is not None and token else None
        return resolve_fee_snapshot(
            market, base_fee=base_fee, category=category, seen_at=entry["seen_at"]
        )

    async def _fee_entry(self, slug: str) -> dict[str, Any]:
        """``{"seen_at": ISO, "market": the fee keys of the market}``, from the cache or Gamma."""
        key = f"gamma:fee-market:{slug}"
        if self._cache is not None:
            cached = self._cache.get(key)
            if isinstance(cached, dict) and "seen_at" in cached and "market" in cached:
                return cached
        resp = await self._send(_slug_path(slug), {}, slug)
        entry = {
            "seen_at": datetime.now(UTC).isoformat(),
            "market": fee_fields(_market_of_slug(resp, slug)),
        }
        if self._cache is not None:
            self._cache.set(key, entry)
        return entry

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
