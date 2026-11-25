# src/pmlab/markets/gamma_client.py
"""Polymarket Gamma API client — fetch open markets by tag/keyword."""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Generator
from contextlib import closing
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
# A runaway guard, not a budget: it counts pages (requests), not markets, so `page_size=10` allows
# 10,000 markets and the default 100,000. An API that ignores `after_cursor` (or `offset`) serves
# pages for ever; the loop stops, loudly, instead.
_MAX_DISCOVERY_PAGES = 1000
# Discovery pages by cursor on /markets/keyset; /markets by offset is only the fallback for a Gamma
# that does not know the keyset endpoint (and live Gamma refuses it from offset 2100).
_KEYSET_PATH = "/markets/keyset"
_OFFSET_PATH = "/markets"
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


# Statuses of the first keyset request that say "wrong credentials, or slow down", not "this Gamma
# has no /markets/keyset": the offset path would meet the same wall (or is retried anyway).
_NOT_A_MISSING_KEYSET = frozenset({401, 403, 408, 429})


def _keyset_refused(resp: httpx.Response) -> bool:
    """True when the first ``/markets/keyset`` reply says this server does not serve the endpoint.

    A 404 or 405 from a deployment that predates it, a 400 or 422 from one that rejects the path,
    a 501 from a proxy: every 4xx but a credentials failure (401, 403) or a transient one (408, 429,
    retried before this is asked), and 501. Only the first page: a 4xx after a cursor was served is
    an error, not an old server.
    """
    code = resp.status_code
    return code == 501 or (400 <= code < 500 and code not in _NOT_A_MISSING_KEYSET)


def _warn_keyset_refused(where: str, resp: httpx.Response) -> None:
    """Say that paging falls back to offsets, which cannot read past about 2000 markets."""
    from pmlab.logging import get_logger

    get_logger(__name__).warning(
        "%s: Gamma refused %s (HTTP %d: %s); falling back to offset paging on %s, which Gamma "
        "stops serving at about offset 2000, so the result may be incomplete.",
        where,
        _KEYSET_PATH,
        resp.status_code,
        resp.text.strip()[:200],
        _OFFSET_PATH,
    )


def _keyset_json(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except ValueError as exc:
        raise ValueError(f"Gamma {_KEYSET_PATH} returned a reply that is not JSON: {exc}") from exc


def _keyset_page(payload: Any) -> tuple[list[dict[str, Any]], str | None]:
    """The markets and the ``next_cursor`` of one ``/markets/keyset`` reply.

    Gamma answers ``{"markets": [...], "next_cursor": "..."}`` and leaves ``next_cursor`` out of
    the last page (an empty ``{"markets": []}`` when nothing matches); ``None`` and ``""`` are read
    as absent. ``ValueError`` for any other shape.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("markets"), list):
        raise ValueError(
            f"Gamma {_KEYSET_PATH} returned a {type(payload).__name__}"
            ", expected an object with a 'markets' list"
        )
    cursor = payload.get("next_cursor")
    if cursor is None or cursor == "":
        return payload["markets"], None
    if not isinstance(cursor, str):
        raise ValueError(
            f"Gamma {_KEYSET_PATH} returned a next_cursor that is a {type(cursor).__name__}"
            ", expected a string"
        )
    return payload["markets"], cursor


class _KeysetWalk:
    """The bookkeeping of one walk down ``/markets/keyset``, shared by the sync and async clients.

    It builds each request (``after_cursor`` from the second on), digests each reply and refuses a
    cursor it has been given before: a Gamma that ignores ``after_cursor`` would answer the first
    page for ever, and the cursor it points at is the same every time.
    """

    def __init__(self, where: str, params: dict[str, Any]) -> None:
        self.where = where
        self._params = params
        self._cursors: set[str] = set()
        self.cursor: str | None = None
        self.pages = 0  # replies digested (a retried request counts once)
        self.read = 0  # markets in them

    @property
    def params(self) -> dict[str, Any]:
        return (
            self._params if self.cursor is None else {**self._params, "after_cursor": self.cursor}
        )

    @property
    def label(self) -> str:
        return f"page {self.pages + 1}"

    @property
    def finished(self) -> bool:
        """True once a reply came without a ``next_cursor``: that is the end of the data."""
        return self.pages > 0 and self.cursor is None

    def digest(self, resp: httpx.Response) -> list[dict[str, Any]]:
        page, cursor = _keyset_page(_keyset_json(resp))
        self.pages += 1
        self.read += len(page)
        if cursor is not None:
            if cursor in self._cursors:
                raise RuntimeError(
                    f"{self.where}: Gamma {_KEYSET_PATH} sent the same next_cursor twice (at page "
                    f"{self.pages}); stopping (is it ignoring `after_cursor`?)"
                )
            self._cursors.add(cursor)
        self.cursor = cursor
        return page

    def capped(self) -> RuntimeError:
        return RuntimeError(
            f"{self.where}: still receiving a next_cursor after {_MAX_DISCOVERY_PAGES} pages "
            f"({self.read} markets read); stopping (is {_KEYSET_PATH} looping?)"
        )


def _offset_capped(where: str, pages: int, offset: int) -> RuntimeError:
    return RuntimeError(
        f"{where}: still receiving full pages after {pages} pages ({offset} markets); "
        "stopping (is the API ignoring `offset`?)"
    )


def _market_key(market: Any) -> str | None:
    """What tells one market from another across pages; ``None`` when it carries nothing."""
    if not isinstance(market, dict):
        return None
    key = market.get("id") or market.get("conditionId")
    return str(key) if key else None


def _unseen(page: list[dict[str, Any]], seen: set[str]) -> list[dict[str, Any]]:
    """The markets of *page* not met on an earlier page (the first copy wins); records them.

    A market with no id is kept as it comes: there is nothing to say it is a copy.
    """
    fresh: list[dict[str, Any]] = []
    for market in page:
        key = _market_key(market)
        if key is not None:
            if key in seen:
                continue
            seen.add(key)
        fresh.append(market)
    return fresh


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
        """Fetch all markets, paging by cursor on ``/markets/keyset``.

        Follows each page's ``next_cursor`` until the last page (which has none) or until
        *max_results* markets are in hand; a market served on two pages is returned once. Pages come
        in Gamma's order (ascending market id), so a *max_results* below the number of active markets
        is the first of them, not the newest. A page that fails for a transient reason (a timeout, a
        connection cut mid-request, a 408, 429 or 5xx; a numeric ``Retry-After`` is honoured) is
        retried with backoff, as in `discover_tmax_markets`.

        A Gamma that does not serve ``/markets/keyset`` (it answers its first page with a 4xx other
        than 401, 403, 408 and 429) is paged by ``offset`` on ``/markets`` instead, with a warning:
        that reads at most about 2000 markets (HTTP 422 "offset too large" is taken as the end, and
        warned about too).

        Args:
            tag: Filter by tag.
            keyword: Client-side keyword filter applied after pagination.
            page_size: Results per page (1 to 100).
            max_results: Hard cap on total results returned.
            active: Include active markets.
            closed: Include closed markets.

        Returns:
            All matching markets up to max_results.

        Raises:
            httpx.HTTPStatusError: If a page keeps failing, or fails with an error that is not
                transient.
            ValueError: If Gamma answers with something that is not a page of markets, or
                ``page_size`` is not an int between 1 and 100.
            RuntimeError: If the cursor loops, or pages are still coming after 1000 of them.
        """
        from pmlab.logging import get_logger

        _log = get_logger(__name__)
        _check_page_size(page_size)

        params: dict[str, Any] = {"limit": page_size, "active": active, "closed": closed}
        if tag:
            params["tag"] = tag

        all_markets: list[dict[str, Any]] = []
        if max_results > 0:
            with closing(self._pages("fetch_markets_all", params, page_size)) as pages:
                for page in pages:
                    all_markets.extend(page)
                    _log.debug("Paginated: fetched %d markets", len(page))
                    if len(all_markets) >= max_results:
                        break

        if keyword:
            kw_lower = keyword.lower()
            all_markets = [m for m in all_markets if kw_lower in m.get("question", "").lower()]

        return all_markets[:max_results]

    def _send(self, path: str, params: dict[str, Any], label: str) -> httpx.Response:
        """GET *path*; a timeout, a connection cut mid-request, a 408, 429 or 5xx is retried with
        backoff (up to `_RETRY_ATTEMPTS` tries), and the reply that ends it is returned for the
        caller to read (after the last try, the failing one). Transport errors that are not
        transient are raised as they come."""
        from pmlab.logging import get_logger

        url = f"{self.base_url}{path}"
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
            _sleep(delay)
        raise AssertionError("unreachable: the last attempt returns or raises")

    def _pages(
        self, where: str, params: dict[str, Any], page_size: int
    ) -> Generator[list[dict[str, Any]], None, None]:
        """The pages of ``/markets/keyset``, each without the markets of the pages before it.

        Falls back to `_offset_pages` when the first reply says the endpoint is not served here.
        """
        seen: set[str] = set()
        walk = _KeysetWalk(where, params)
        while walk.pages < _MAX_DISCOVERY_PAGES:
            resp = self._send(_KEYSET_PATH, walk.params, walk.label)
            if walk.pages == 0 and _keyset_refused(resp):
                _warn_keyset_refused(where, resp)
                yield from self._offset_pages(where, params, page_size)
                return
            resp.raise_for_status()
            yield _unseen(walk.digest(resp), seen)
            if walk.finished:
                return
        raise walk.capped()

    def _offset_pages(
        self, where: str, params: dict[str, Any], page_size: int
    ) -> Generator[list[dict[str, Any]], None, None]:
        """The pages of ``/markets`` by offset: the fallback, which Gamma ends at about offset 2000
        with an HTTP 422 (taken as the end: warned about, and what was read is kept)."""
        seen: set[str] = set()
        offset = 0
        for _ in range(_MAX_DISCOVERY_PAGES):
            resp = self._send(_OFFSET_PATH, {**params, "offset": offset}, f"offset={offset}")
            if _offset_exhausted(resp, offset):
                _warn_offset_exhausted(where, offset, resp)
                return
            resp.raise_for_status()
            page = _as_page(resp.json())
            yield _unseen(page, seen)
            if len(page) < page_size:
                return
            offset += page_size
        raise _offset_capped(where, _MAX_DISCOVERY_PAGES, offset)

    def discover_tmax_markets(
        self,
        page_size: int = 100,
        active: bool = True,
    ) -> list[TmaxMarketInfo]:
        """Page through all active markets and return the temperature-max ones.

        Pages by cursor on ``/markets/keyset``: each reply's ``next_cursor`` is sent back as
        ``after_cursor`` until a reply has none, so the whole universe is read however large it is
        (``/markets`` by offset ends at about 2000 on live Gamma). A market served on two pages is
        read once. Each page is filtered with the temperature-max heuristic and parsed into
        `TmaxMarketInfo`.

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

        Returns:
            List of parsed `TmaxMarketInfo` for every matching market read.

        Raises:
            httpx.HTTPStatusError: If a page keeps failing, or fails with an error that is not
                transient (a 4xx other than 408 and 429; on the first page, one that says the
                endpoint is not served falls back to offsets instead).
            httpx.TransportError: If the connection keeps failing.
            ValueError: If Gamma answers with something that is not a page of markets, or
                ``page_size`` is not an int between 1 and 100.
            RuntimeError: If the cursor loops (a ``next_cursor`` it has sent before), or pages are
                still coming after 1000 of them. The cap counts pages, not markets: ``page_size=100``
                allows 100,000 markets and ``page_size=10`` allows 10,000.
        """
        from pmlab.logging import get_logger

        _log = get_logger(__name__)
        _check_page_size(page_size)

        params: dict[str, Any] = {"limit": page_size, "active": active, "closed": False}
        results: list[TmaxMarketInfo] = []
        for page in self._pages("discover_tmax_markets", params, page_size):
            _log.debug("discover_tmax_markets: fetched %d new markets", len(page))
            for market in page:
                if _is_tmax_market(market):
                    info = _build_tmax_market_info(market)
                    if info is not None:
                        results.append(info)
        return results

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> GammaClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
