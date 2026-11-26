"""Fee schedules read from the live Gamma and CLOB endpoints (1.4.0, #71), used from the wheel.

Standalone (public names only, no repo imports; needs the network):  python audit_fee_schedule_live.py
Prints ``audit fee schedule live: OK`` and exits 0, or lists every check that failed and exits 1.

Read-only GETs, about ten: one page of the newest active markets to find a market that charges a fee,
Gamma's ``geopolitics`` tag to find one that does not, then ``/markets?slug=`` and CLOB's
``/fee-rate`` for each, sync and async. It is the check that what ``base_fee`` means (a flag: 1000
where a fee is charged, 0 where not, never the rate) is still what Polymarket serves: if CLOB and
Gamma stop agreeing on whether a fee is charged, ``fetch_fee_snapshot`` warns and this audit fails.
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import traceback
from collections.abc import Callable
from datetime import timedelta
from typing import Any

import httpx

import pmlab
from pmlab import (
    AsyncClobClient,
    AsyncGammaClient,
    ClobClient,
    DiskCache,
    FeeScheduleStore,
    GammaClient,
)

GAMMA = "https://gamma-api.polymarket.com"
CHECKS: list[tuple[str, Callable[[], None]]] = []


def check(fn: Callable[[], None]) -> Callable[[], None]:
    CHECKS.append((fn.__name__, fn))
    return fn


class _Warnings(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _page(params: dict[str, Any], path: str = "/markets/keyset") -> Any:
    reply = httpx.get(f"{GAMMA}{path}", params=params, timeout=30.0)
    reply.raise_for_status()
    return reply.json()


def _has_tokens(market: dict[str, Any]) -> bool:
    raw = market.get("clobTokenIds")
    tokens = json.loads(raw) if isinstance(raw, str) else raw
    return bool(tokens) and bool(market.get("slug"))


def _find() -> tuple[str, str | None]:
    """The slug of a live fee-charging market and of a live fee-free one (``None`` if none is
    among Gamma's first 30 active geopolitics markets today)."""
    newest = _page(
        {"limit": 100, "active": "true", "closed": "false", "order": "id", "ascending": "false"}
    )
    charging = next(
        m["slug"]
        for m in newest["markets"]
        if m.get("feesEnabled") is True
        and isinstance(m.get("feeSchedule"), dict)
        and _has_tokens(m)
    )
    tag = _page({}, "/tags/slug/geopolitics")
    geo = _page({"limit": 30, "active": "true", "closed": "false", "tag_id": int(tag["id"])})
    free = next(
        (m["slug"] for m in geo["markets"] if m.get("feesEnabled") is False and _has_tokens(m)),
        None,
    )
    return charging, free


SLUGS = _find()
WARNINGS = _Warnings()
logging.getLogger("pmlab").addHandler(WARNINGS)


def _no_new_warnings(before: int) -> None:
    assert WARNINGS.messages[before:] == [], WARNINGS.messages[before:]


@check
def a_live_fee_charging_market_has_its_schedule_from_gamma_and_base_fee_agrees() -> None:
    before = len(WARNINGS.messages)
    with GammaClient() as gamma, ClobClient() as clob:
        snap = gamma.fetch_fee_snapshot(SLUGS[0], clob=clob)
    assert snap.source == "gamma" and not snap.fallback and snap.slug == SLUGS[0], snap
    assert snap.schedule.enabled and 0.0 < snap.schedule.rate < 1.0, snap
    assert snap.schedule.exponent > 0 and snap.fee_type, snap
    assert isinstance(snap.base_fee, int) and snap.base_fee > 0, snap  # a flag: a fee is charged
    _no_new_warnings(before)
    print(f"     {snap.slug}: {snap.fee_type} rate {snap.schedule.rate} base_fee {snap.base_fee}")


@check
def a_live_fee_free_market_has_no_fee_and_base_fee_is_zero() -> None:
    if SLUGS[1] is None:
        print("     no fee-free market among the first 30 geopolitics markets today: skipped")
        return
    before = len(WARNINGS.messages)
    with GammaClient() as gamma, ClobClient() as clob:
        snap = gamma.fetch_fee_snapshot(SLUGS[1], clob=clob)
    assert snap.source == "gamma" and not snap.fallback, snap
    assert snap.schedule.enabled is False and snap.schedule.rate == 0.0, snap
    assert snap.base_fee == 0, snap
    _no_new_warnings(before)


@check
def the_async_clients_read_the_same_schedule() -> None:
    async def both() -> tuple[pmlab.FeeSnapshot, pmlab.FeeSnapshot]:
        async with AsyncGammaClient() as gamma, AsyncClobClient() as clob:
            return await gamma.fetch_fee_snapshot(
                SLUGS[0], clob=clob
            ), await gamma.fetch_fee_snapshot(SLUGS[0])

    with GammaClient() as gamma, ClobClient() as clob:
        sync = gamma.fetch_fee_snapshot(SLUGS[0], clob=clob)
    with_clob, without_clob = asyncio.run(both())
    assert with_clob.schedule == without_clob.schedule == sync.schedule, (with_clob, sync)
    assert with_clob.base_fee == sync.base_fee and without_clob.base_fee is None


@check
def the_cache_and_the_store_keep_what_was_seen_and_when() -> None:
    with tempfile.TemporaryDirectory() as root:
        cache = DiskCache(f"{root}/cache", ttl_seconds=3600)
        with GammaClient(cache=cache) as gamma, ClobClient(cache=cache) as clob:
            first = gamma.fetch_fee_snapshot(SLUGS[0], clob=clob)
            again = gamma.fetch_fee_snapshot(SLUGS[0], clob=clob)
        assert again == first, "inside the TTL the sighting is the first one, not a new one"
        store = FeeScheduleStore(f"{root}/store")
        assert store.record([first, again]) == 1
        assert store.as_of(first.seen_at) == {}  # strictly before: not known yet at that instant
        known = store.as_of(first.seen_at + timedelta(seconds=1))
        assert known[first.market_id] == first and store.history(first.market_id) == [first]


@check
def an_unknown_slug_is_a_lookup_error() -> None:
    with GammaClient() as gamma:
        try:
            gamma.fetch_fee_snapshot("pmlab-no-such-market-slug-71")
        except LookupError as error:
            assert "pmlab-no-such-market-slug-71" in str(error)
        else:
            raise AssertionError("an unknown slug was accepted")


def main() -> int:
    failures: list[str] = []
    for name, fn in CHECKS:
        try:
            fn()
        except Exception:
            failures.append(name)
            print(f"FAIL {name}\n{traceback.format_exc(limit=3)}")
        else:
            print(f"ok   {name}")
    if failures:
        print(
            f"audit fee schedule live: {len(failures)} of {len(CHECKS)} checks FAILED: {failures}"
        )
        return 1
    print(f"audit fee schedule live: OK ({len(CHECKS)} checks, pmlab {pmlab.__version__})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
