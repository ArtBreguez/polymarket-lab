# src/pmlab/markets/gamma_client.py
"""Polymarket Gamma API client — fetch open markets by tag/keyword."""

from __future__ import annotations

import json
import re
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
        or max_results is reached.

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

    def discover_tmax_markets(
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
        from pmlab.logging import get_logger

        _log = get_logger(__name__)

        results: list[TmaxMarketInfo] = []
        offset = 0

        while True:
            params: dict[str, Any] = {
                "limit": page_size,
                "offset": offset,
                "active": active,
                "closed": False,
            }
            resp = self._client.get(f"{self.base_url}/markets", params=params)
            resp.raise_for_status()
            page: list[dict[str, Any]] = resp.json()

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
                break
            offset += page_size

        return results

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> GammaClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
