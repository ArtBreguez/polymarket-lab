# src/pmlab/markets/clob_client.py
"""Polymarket CLOB read client — fetch current market prices."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx

from pmlab.markets.fees import parse_fee_rate

if TYPE_CHECKING:
    from pmlab.markets.cache import DiskCache

CLOB_API_BASE = "https://clob.polymarket.com"


def parse_midpoint(data: Any) -> float | None:
    """The midpoint of a ``/midpoint`` reply, or ``None`` if it is not a probability.

    A reply that is not an object, has no ``mid``, or whose ``mid`` is NaN, infinite or outside
    [0, 1] is a failed request, not a price: stored, it failed later in the middle of a backtest
    or was traded on (#63).
    """
    if not isinstance(data, dict):
        return None
    mid = data.get("mid")
    if mid is None:
        return None
    try:
        value = float(mid)
    except (TypeError, ValueError):
        return None
    return value if 0.0 <= value <= 1.0 else None


def _check_token_id(token_id: object) -> None:
    if not isinstance(token_id, str) or not token_id.strip():
        raise ValueError(f"token_id must be a non-empty string, got {token_id!r}")


def fetch_token_prices(
    token_ids: list[str],
    base_url: str = CLOB_API_BASE,
    client: httpx.Client | None = None,
) -> dict[str, float]:
    """Fetch midpoint prices for a list of token IDs."""
    if not token_ids:
        return {}

    own_client = client is None
    _client: httpx.Client = client if client is not None else httpx.Client(timeout=15.0)

    prices: dict[str, float] = {}
    try:
        for token_id in token_ids:
            try:
                resp = _client.get(f"{base_url}/midpoint", params={"token_id": token_id})
                resp.raise_for_status()
                mid = parse_midpoint(resp.json())
                if mid is not None:
                    prices[token_id] = mid
            except (httpx.HTTPError, ValueError, KeyError):
                continue
    finally:
        if own_client:
            _client.close()

    return prices


class ClobClient:
    """Stateful CLOB read client with connection reuse.

    Args:
        base_url: The CLOB API root.
        timeout: Per-request timeout, in seconds.
        cache: Optional `DiskCache` for ``fetch_fee_rate``; an entry lives for the cache's own TTL
            (`DiskCache` defaults to one hour; ``PMLAB_CACHE_TTL`` is not read here). Keys carry
            the token id, not ``base_url``: give a staging and a production client each its own
            cache directory.
    """

    def __init__(
        self,
        base_url: str = CLOB_API_BASE,
        timeout: float = 15.0,
        cache: DiskCache | None = None,
    ) -> None:
        self.base_url = base_url
        self._client = httpx.Client(timeout=timeout)
        self._cache = cache

    def fetch_prices(self, token_ids: list[str]) -> dict[str, float]:
        return fetch_token_prices(token_ids, base_url=self.base_url, client=self._client)

    def fetch_fee_rate(self, token_id: str) -> int | None:
        """CLOB's ``base_fee`` for a token (``GET /fee-rate?token_id=``), in basis points.

        It is a flag (1000 on every market that charges a fee, 0 on one that does not), not the fee
        rate: see `resolve_fee_snapshot`. A request that fails, or a reply with no usable
        ``base_fee``, gives ``None`` (unknown, with a warning when the request failed) and is not
        cached; a 0 is cached like any other answer.

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
        try:
            resp = self._client.get(f"{self.base_url}/fee-rate", params={"token_id": token_id})
            resp.raise_for_status()
            base_fee = parse_fee_rate(resp.json())
        except (httpx.HTTPError, ValueError) as exc:
            from pmlab.logging import get_logger

            get_logger(__name__).warning("CLOB /fee-rate failed for token %s: %s", token_id, exc)
            return None
        if base_fee is not None and self._cache is not None:
            self._cache.set(key, base_fee)
        return base_fee

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ClobClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
