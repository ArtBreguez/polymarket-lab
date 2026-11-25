# src/pmlab/markets/gamma_client.py
"""Polymarket Gamma API client — fetch open markets by tag/keyword."""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from pmlab.markets.cache import DiskCache

# ---------------------------------------------------------------------------
# Tmax market filtering constants
# ---------------------------------------------------------------------------
_TMAX_RE = re.compile(r"highest temperature in (.+) on (.+?)\?", re.I)
_TMAX_SOURCE_HINTS = (
    "wunderground",
    "hong kong observatory",
    "central weather administration",
    "highest temperature recorded",
)


@dataclass
class TmaxMarketInfo:
    """Parsed representation of a Polymarket temperature-max market."""

    market_id: str
    question: str
    slug: str
    city: str
    target_date: str  # YYYY-MM-DD
    unit: str  # 'C' or 'F'
    token_ids: list[str] = field(default_factory=list)
    outcome_labels: list[str] = field(default_factory=list)
    outcome_prices: list[float] = field(default_factory=list)
    end_date: str = ""
    active: bool = True


def _parse_tmax_date(date_str: str, reference_year: int | None = None) -> str:
    """Parse a human-readable date like 'May 13' into 'YYYY-MM-DD'.

    Falls back to the raw string if parsing fails.
    """
    year = reference_year if reference_year is not None else date.today().year
    date_str = date_str.strip()
    for fmt in ("%B %d", "%b %d", "%B %d, %Y", "%b %d, %Y"):
        try:
            parsed = datetime.strptime(date_str, fmt)
            return parsed.replace(year=year).strftime("%Y-%m-%d")
        except ValueError:
            continue
    # Already YYYY-MM-DD or unrecognised — return as-is
    return date_str


# ---------------------------------------------------------------------------
# Discovery paging: how many pages, and what a failed page costs
# ---------------------------------------------------------------------------
# A runaway guard, not a budget: 1000 pages of 100 is far beyond Gamma's active markets. An API
# that ignores `offset` returns full pages for ever; the loop stops, loudly, instead.
_MAX_DISCOVERY_PAGES = 1000
# One page is asked for up to this many times (the first try included) before its failure is
# raised: a 429 or a 5xx on page 40 should not cost the 39 pages before it.
_RETRY_ATTEMPTS = 5
_RETRY_BASE_DELAY = 1.0  # seconds before the first retry; doubles each time
_RETRY_MAX_DELAY = 30.0
_RETRY_AFTER_MAX = 60.0  # a Retry-After is honoured up to this
_TRANSIENT_STATUS = frozenset({408, 429, 500, 502, 503, 504})
# Timeouts, and a connection that was cut mid-request (a reset, a half-written reply). Not
# ConnectError: nothing is listening (offline, a wrong base_url) and four waits only delay the same
# error; nor UnsupportedProtocol or a bad proxy, which a retry cannot fix either.
_TRANSIENT_ERRORS = (httpx.TimeoutException, httpx.ReadError, httpx.RemoteProtocolError)
# What Gamma's 422 says when the offset space is over; a 422 that merely names `offset` is not it.
_OFFSET_EXHAUSTED_PHRASES = ("offset too large", "/markets/keyset")
_sleep = time.sleep  # a name of its own so a test can replace it


def _retry_delay(attempt: int, response: httpx.Response | None) -> float:
    """Seconds to wait after attempt number *attempt* (0-based) failed.

    A numeric ``Retry-After`` on the response is honoured, up to a minute; otherwise the wait
    doubles from one second and stops growing at thirty.
    """
    if response is not None:
        raw = response.headers.get("Retry-After")
        if raw is not None:
            try:
                seconds = float(raw)
            except ValueError:
                seconds = math.nan  # an HTTP-date, or junk: fall back to the backoff
            if math.isfinite(seconds) and seconds >= 0:
                return min(seconds, _RETRY_AFTER_MAX)
    return min(_RETRY_BASE_DELAY * 2.0**attempt, _RETRY_MAX_DELAY)


def _offset_exhausted(resp: httpx.Response, offset: Any) -> bool:
    """True when *resp* is Gamma saying the offset space is over, not a failure of the page.

    Live Gamma serves ``/markets`` up to about offset 2000 and answers HTTP 422
    ``{"error": "offset too large, use /markets/keyset for deeper pagination"}`` from there on.
    That is the end of what offset paging can read. It is only recognised past the first page
    (a 422 at offset 0 is a bad request) and when the body says it in Gamma's own words
    ("offset too large", or the ``/markets/keyset`` it points to), so any other 422, one that
    merely mentions the offset included, is still an error.
    """
    if resp.status_code != 422 or not isinstance(offset, int) or offset <= 0:
        return False
    body = resp.text.lower()
    return any(phrase in body for phrase in _OFFSET_EXHAUSTED_PHRASES)


def _warn_offset_exhausted(where: str, offset: int, resp: httpx.Response) -> None:
    """Say that the result stops at *offset* because Gamma refused to page any deeper."""
    from pmlab.logging import get_logger

    get_logger(__name__).warning(
        "%s: Gamma refused offset=%d (HTTP 422: %s); offset paging ends there, so the result is "
        "incomplete (the %d markets before it were read). Deeper pages need /markets/keyset.",
        where,
        offset,
        resp.text.strip()[:200],
        offset,
    )


def _as_page(payload: Any) -> list[dict[str, Any]]:
    """The markets of one page; ``ValueError`` if Gamma answered with something else."""
    if not isinstance(payload, list):
        raise ValueError(
            f"Gamma /markets returned a {type(payload).__name__}, expected a list of markets"
        )
    return payload


def _check_page_size(page_size: int) -> None:
    """The API serves at most 100 per page; asking for more stopped the paging after one page."""
    if isinstance(page_size, bool) or not isinstance(page_size, int) or not 1 <= page_size <= 100:
        raise ValueError(f"page_size must be an int between 1 and 100, got {page_size!r}")


def _parse_json_list(raw: Any) -> list[Any]:
    """Return a Python list from a JSON string or existing list."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return []


def _is_tmax_market(market: dict[str, Any]) -> bool:
    """Return True if *market* matches the temperature-max family filter."""
    question = (market.get("question") or "").strip()
    description = (market.get("description") or "").strip().lower()
    if not _TMAX_RE.search(question):
        return False
    if not any(hint in description for hint in _TMAX_SOURCE_HINTS):
        return False
    outcome_labels: list[str] = _parse_json_list(market.get("outcomes"))
    has_temp_bins = any("°c" in lbl.lower() or "°f" in lbl.lower() for lbl in outcome_labels)
    return has_temp_bins


def _build_tmax_market_info(market: dict[str, Any]) -> TmaxMarketInfo | None:
    """Parse a raw Gamma market dict into a TmaxMarketInfo.

    Returns None if the question doesn't match the expected pattern.
    """
    question = (market.get("question") or "").strip()
    m = _TMAX_RE.search(question)
    if m is None:
        return None

    city = m.group(1).strip()
    date_raw = m.group(2).strip()
    target_date = _parse_tmax_date(date_raw)

    outcome_labels: list[str] = _parse_json_list(market.get("outcomes"))
    unit = "F" if any("°f" in lbl.lower() for lbl in outcome_labels) else "C"

    raw_prices = _parse_json_list(market.get("outcomePrices"))
    outcome_prices: list[float] = []
    for p in raw_prices:
        try:
            outcome_prices.append(float(p))
        except (TypeError, ValueError):
            outcome_prices.append(0.0)

    token_ids: list[str] = _parse_json_list(market.get("clobTokenIds"))

    return TmaxMarketInfo(
        market_id=str(market.get("id") or market.get("conditionId") or ""),
        question=question,
        slug=str(market.get("slug") or ""),
        city=city,
        target_date=target_date,
        unit=unit,
        token_ids=[str(t) for t in token_ids],
        outcome_labels=outcome_labels,
        outcome_prices=outcome_prices,
        end_date=str(market.get("endDate") or ""),
        active=bool(market.get("active", True)),
    )


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
            tag=tag,
            keyword=keyword,
            limit=limit,
            active=active,
            closed=closed,
            base_url=self.base_url,
            client=self._client,
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
        or max_results is reached. Gamma stops serving offsets at about 2000 (HTTP 422 "offset
        too large"); that is taken as the end: what was read is returned and a warning says the
        result is incomplete.

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
        _check_page_size(page_size)

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
            if _offset_exhausted(resp, offset):
                _warn_offset_exhausted("fetch_markets_all", offset, resp)
                break
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

    def _get_page(self, params: dict[str, Any]) -> list[dict[str, Any]] | None:
        """One page of ``/markets``; a timeout, a connection cut mid-request, a 408, 429 or 5xx
        is retried with backoff (up to `_RETRY_ATTEMPTS` tries), any other failure is raised.
        ``None`` means Gamma refused the offset as too large: the end of the offset space."""
        from pmlab.logging import get_logger

        url = f"{self.base_url}/markets"
        for attempt in range(_RETRY_ATTEMPTS):
            final = attempt == _RETRY_ATTEMPTS - 1
            try:
                resp = self._client.get(url, params=params)
            except _TRANSIENT_ERRORS as exc:
                if final:
                    raise
                delay = _retry_delay(attempt, None)
                reason = type(exc).__name__
            else:
                if _offset_exhausted(resp, params.get("offset")):
                    _warn_offset_exhausted("discover_tmax_markets", params["offset"], resp)
                    return None
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
            _sleep(delay)
        raise AssertionError("unreachable: the last attempt returns or raises")

    def discover_tmax_markets(
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
        Gamma stops serving offsets at about 2000 (HTTP 422 "offset too large"): that is the end
        of what offset paging can read, so the markets read so far are returned and a warning
        says the result is incomplete (deeper pages need ``/markets/keyset``).

        Against the live API, as of 2026-11-25, that is the first 2100 active markets, and none
        of them is a temperature market: ``discover_tmax_markets()`` returns ``[]`` with that
        warning, where it used to raise at page 22, until discovery moves to ``/markets/keyset``
        (#83). An empty result with the warning means "not read", not "no markets today".

        Args:
            page_size: Results per API page (max 100).
            active: If True, only fetch active markets.

        Returns:
            List of parsed `TmaxMarketInfo` for every matching market read.

        Raises:
            httpx.HTTPStatusError: If a page keeps failing, or fails with an error that is not
                transient (a 4xx other than 408 and 429).
            httpx.TransportError: If the connection keeps failing.
            ValueError: If Gamma answers with something other than a list of markets, or
                ``page_size`` is not an int between 1 and 100.
            RuntimeError: If it is still receiving full pages after 1000 of them (an API that
                ignores ``offset`` would otherwise never let it finish).
        """
        from pmlab.logging import get_logger

        _log = get_logger(__name__)
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
            page = self._get_page(params)
            if page is None:  # Gamma's offset paging is over; keep what was read (warned)
                return results

            _log.debug(
                "discover_tmax_markets: fetched %d markets (offset=%d)",
                len(page),
                offset,
            )

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

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> GammaClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
