"""Market discovery pages by cursor on ``/markets/keyset`` (1.2.2, #83), used from the installed wheel.

Standalone (public names only, no repo imports):  python audit_gamma_keyset.py
Prints ``audit gamma keyset: OK`` and exits 0, or lists every check that failed and exits 1.

Live Gamma serves ``/markets`` by ``offset`` up to about 2000 and answers HTTP 422 "offset too large,
use /markets/keyset" after that, so 1.2.1 read the first 2100 active markets and warned. The stub
here plays both endpoints the way live Gamma does (the cursor is opaque and full of characters a
query string must escape, the last page has no ``next_cursor``), on 127.0.0.1 with a free port and
the stdlib ``http.server``: no network, no sleeping (the scripted failures say ``Retry-After: 0``).

Every check is independent and reports on its own, so a run against an older wheel says which part
it lacks.
"""

from __future__ import annotations

import asyncio
import contextlib
import http.server
import json
import logging
import os
import threading
import traceback
import urllib.parse
from collections.abc import Callable, Iterator
from typing import Any

import httpx

import pmlab
from pmlab import AsyncGammaClient, GammaClient

CHECKS: list[tuple[str, Callable[[], None]]] = []
OFFSET_TOO_LARGE = '{"error": "offset too large, use /markets/keyset for deeper pagination"}'
INVALID_CURSOR = '{"type": "validation error", "error": "invalid cursor"}'
# What the cursor of a position looks like: opaque, and every character that breaks a query string
# unless it is escaped. A client that sends it back unescaped gets "invalid cursor" from the stub.
CURSOR_PREFIX = "v1+/=&%20?#"
Reply = tuple[int, dict[str, str], str]  # status, headers, body


def check(fn: Callable[[], None]) -> Callable[[], None]:
    CHECKS.append((fn.__name__, fn))
    return fn


def refused(
    exc_type: type[Exception], fn: Callable[..., object], *args: object, **kw: object
) -> str:
    try:
        fn(*args, **kw)
    except exc_type as error:
        return str(error)
    raise AssertionError(f"{getattr(fn, '__name__', fn)}{args}{kw} was accepted")


def tmax_market(i: int) -> dict[str, Any]:
    return {
        "id": str(i),
        "question": f"Highest temperature in City{i} on May 13?",
        "slug": f"tmax-{i}",
        "description": "Resolves with the highest temperature recorded on Wunderground.",
        "outcomes": '["30°C", "31°C"]',
        "outcomePrices": '["0.4", "0.6"]',
        "clobTokenIds": '["a", "b"]',
        "endDate": "2026-05-13T12:00:00Z",
        "active": True,
    }


class _Warnings(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@contextlib.contextmanager
def logged_warnings() -> Iterator[list[str]]:
    """The WARNING messages ``pmlab`` logs inside the block."""
    handler, logger = _Warnings(), logging.getLogger("pmlab")
    level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        yield handler.messages
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)


class GammaStub:
    """Live Gamma's market endpoints on 127.0.0.1.

    ``/markets/keyset``: ``total`` markets by ``limit`` and ``after_cursor`` (``endless``: for ever),
    ``{"markets": [...], "next_cursor": "..."}`` and no ``next_cursor`` on the last page. ``/markets``:
    the same markets by ``offset``, up to offset ``offset_limit``, then HTTP 422 "offset too large".
    ``script`` replaces the reply to the request for the market at a position (a failure to retry, a
    422); ``keyset_status`` answers every keyset request with that status (a Gamma without the
    endpoint, or one that refuses us); ``overlap`` repeats the previous page's last market at the
    top of the page; ``ignore_cursor`` is an API that answers the first page whatever the cursor;
    ``trailing_empty`` gives the last full page a cursor to one more page, which is empty.
    """

    def __init__(
        self,
        total: int = 0,
        *,
        script: dict[int, list[Reply]] | None = None,
        offset_limit: int = 2000,
        keyset_status: int | None = None,
        overlap: bool = False,
        ignore_cursor: bool = False,
        endless: bool = False,
        trailing_empty: bool = False,
    ) -> None:
        self.total = total
        self.script = {pos: list(replies) for pos, replies in (script or {}).items()}
        self.offset_limit = offset_limit
        self.keyset_status = keyset_status
        self.overlap = overlap
        self.ignore_cursor = ignore_cursor
        self.endless = endless
        self.trailing_empty = trailing_empty
        self.keyset_requests: list[dict[str, str]] = []
        self.offset_requests: list[int] = []
        stub = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 (the name http.server looks for)
                url = urllib.parse.urlsplit(self.path)
                query = {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}
                if url.path == "/markets/keyset":
                    stub.keyset(self, query)
                elif url.path == "/markets" and "offset" in query:
                    stub.offsets(self, query)
                else:
                    self.reply(400, {}, '{"error": "unexpected request"}')

            def reply(self, status: int, headers: dict[str, str], body: str) -> None:
                data = body.encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                for name, value in headers.items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: object) -> None:  # keep the audit's output readable
                pass

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"

    def ids(self, start: int, limit: int) -> list[int]:
        stop = start + limit if self.endless else min(start + limit, self.total)
        return list(range(start, max(start, stop)))

    def keyset(self, handler: Any, query: dict[str, str]) -> None:
        self.keyset_requests.append(query)
        if self.keyset_status is not None:
            handler.reply(self.keyset_status, {}, json.dumps({"error": self.keyset_status}))
            return
        limit = int(query["limit"])
        cursor = query.get("after_cursor")
        if cursor is None or self.ignore_cursor:
            start = 0
        elif cursor.startswith(CURSOR_PREFIX) and cursor.removeprefix(CURSOR_PREFIX).isdigit():
            start = int(cursor.removeprefix(CURSOR_PREFIX))
        else:  # what a cursor that was mangled on the way back looks like
            handler.reply(422, {}, INVALID_CURSOR)
            return
        scripted = self.script.get(start)
        if scripted:
            handler.reply(*scripted.pop(0))
            return
        first = max(0, start - 1) if self.overlap and start else start
        body: dict[str, Any] = {
            "markets": [tmax_market(i) for i in self.ids(first, limit + (start - first))]
        }
        more = self.endless or start + limit < self.total
        if self.trailing_empty and start < self.total and len(body["markets"]) == limit:
            more = True
        if more:
            body["next_cursor"] = f"{CURSOR_PREFIX}{start + limit}"
        handler.reply(200, {}, json.dumps(body))

    def offsets(self, handler: Any, query: dict[str, str]) -> None:
        offset, limit = int(query["offset"]), int(query["limit"])
        self.offset_requests.append(offset)
        if offset > self.offset_limit:
            handler.reply(422, {}, OFFSET_TOO_LARGE)
            return
        page = [tmax_market(i) for i in self.ids(offset, limit)]
        handler.reply(200, {}, json.dumps(page))

    def __enter__(self) -> GammaStub:
        # a proxy set in the environment must not stand between the client and 127.0.0.1
        os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()


def discover(stub: GammaStub, **kw: Any) -> list[Any]:
    with GammaClient(base_url=stub.url) as client:
        return client.discover_tmax_markets(**kw)


def discover_async(stub: GammaStub, **kw: Any) -> list[Any]:
    async def run() -> list[Any]:
        async with AsyncGammaClient(base_url=stub.url) as client:
            return await client.discover_tmax_markets(**kw)

    return asyncio.run(run())


def fetch_all(stub: GammaStub, **kw: Any) -> list[dict[str, Any]]:
    with GammaClient(base_url=stub.url) as client:
        return client.fetch_markets_all(**kw)


def ids_of(found: list[Any]) -> list[str]:
    return [m.market_id if hasattr(m, "market_id") else m["id"] for m in found]


@check
def this_is_the_installed_wheel_not_a_checkout() -> None:
    assert "site-packages" in pmlab.__file__, pmlab.__file__


# ── 1. discovery follows the cursor past the offset limit ───────────────────────────────────────
def assert_read_everything_by_cursor(found: list[Any], stub: GammaStub, said: list[str]) -> None:
    assert ids_of(found) == [str(i) for i in range(2500)], (len(found), said)
    assert stub.offset_requests == [], stub.offset_requests  # nothing paged by offset
    # 25 full pages, the last without a next_cursor: no 26th request for an empty page
    assert len(stub.keyset_requests) == 25, len(stub.keyset_requests)
    assert stub.keyset_requests[0] == {"limit": "100", "active": "true", "closed": "false"}
    assert [r["after_cursor"] for r in stub.keyset_requests[1:3]] == [
        f"{CURSOR_PREFIX}100",
        f"{CURSOR_PREFIX}200",
    ]  # the cursor came back exactly as it was sent, special characters and all
    assert all("offset" not in r for r in stub.keyset_requests)
    assert said == [], said  # nothing was incomplete


@check
def discover_tmax_markets_reads_past_the_2000_offset_limit() -> None:
    with GammaStub(total=2500) as stub, logged_warnings() as said:
        found = discover(stub)
    assert_read_everything_by_cursor(found, stub, said)


@check
def async_discover_tmax_markets_reads_past_the_2000_offset_limit() -> None:
    with GammaStub(total=2500) as stub, logged_warnings() as said:
        found = discover_async(stub)
    assert_read_everything_by_cursor(found, stub, said)


@check
def the_query_is_the_one_asked_for() -> None:
    with GammaStub(total=5) as stub:
        discover(stub, page_size=2, active=False)
    assert stub.keyset_requests[0] == {"limit": "2", "active": "false", "closed": "false"}, (
        stub.keyset_requests
    )
    for bad in (0, 101):
        with GammaStub(total=5) as stub:
            assert "page_size" in refused(ValueError, discover, stub, page_size=bad)
        assert stub.keyset_requests == []  # refused before any request


# ── 2. the end of the data ──────────────────────────────────────────────────────────────────────
@check
def the_last_page_has_no_cursor_and_nothing_more_is_asked() -> None:
    with GammaStub(total=300) as stub:  # an exact multiple of the page size
        assert len(discover(stub)) == 300
    assert len(stub.keyset_requests) == 3, stub.keyset_requests
    with GammaStub(total=0) as stub:  # `{"markets": []}`, no next_cursor
        assert discover(stub) == []
    assert len(stub.keyset_requests) == 1
    with GammaStub(total=300, trailing_empty=True) as stub:  # a full last page pointing at none
        assert len(discover(stub)) == 300
    assert len(stub.keyset_requests) == 4, stub.keyset_requests


# ── 3. fetch_markets_all ────────────────────────────────────────────────────────────────────────
@check
def fetch_markets_all_pages_by_cursor_and_stops_at_max_results() -> None:
    with GammaStub(total=2500) as stub:
        found = fetch_all(stub, page_size=100, max_results=3000)
    assert ids_of(found) == [str(i) for i in range(2500)], len(found)
    assert stub.offset_requests == [] and len(stub.keyset_requests) == 25
    with GammaStub(total=2500) as stub:
        found = fetch_all(stub, page_size=100, max_results=250)
    assert ids_of(found) == [str(i) for i in range(250)], len(found)
    assert len(stub.keyset_requests) == 3, stub.keyset_requests  # not one more than it needed
    with GammaStub(total=2500) as stub:
        assert len(fetch_all(stub)) == 1000  # the default max_results
        assert len(stub.keyset_requests) == 10


# ── 4. each market once ─────────────────────────────────────────────────────────────────────────
@check
def a_market_served_on_two_pages_is_returned_once() -> None:
    expected = [str(i) for i in range(250)]
    with GammaStub(total=250, overlap=True) as stub:
        found = discover(stub)
    assert ids_of(found) == expected, ids_of(found)
    assert stub.offset_requests == [] and len(stub.keyset_requests) == 3, stub.keyset_requests
    with GammaStub(total=250, overlap=True) as stub:
        found = fetch_all(stub, max_results=1000)
    assert ids_of(found) == expected, ids_of(found)
    with GammaStub(total=250, overlap=True) as stub:
        assert ids_of(discover_async(stub)) == expected


# ── 5. retry and backoff of #79 reach every cursor page ─────────────────────────────────────────
@check
def a_429_and_a_500_on_a_cursor_page_are_retried_in_place() -> None:
    script: dict[int, list[Reply]] = {
        200: [(429, {"Retry-After": "0"}, '{"error": "slow down"}')],
        400: [(500, {"Retry-After": "0"}, '{"error": "boom"}')],
    }
    with GammaStub(total=600, script=script) as stub, logged_warnings() as said:
        found = discover(stub)
    assert ids_of(found) == [str(i) for i in range(600)], len(found)
    cursors = [r.get("after_cursor") for r in stub.keyset_requests]
    assert cursors.count(f"{CURSOR_PREFIX}200") == 2 and cursors.count(f"{CURSOR_PREFIX}400") == 2
    assert sum("retrying" in m for m in said) == 2, said
    with GammaStub(total=600, script={0: [(503, {"Retry-After": "0"}, "{}")]}) as stub:
        assert len(discover_async(stub)) == 600  # the first page too, and no fallback
    assert stub.offset_requests == [], stub.offset_requests


# ── 6. a Gamma without the keyset endpoint ──────────────────────────────────────────────────────
def assert_fell_back(found: list[Any], stub: GammaStub, said: list[str]) -> None:
    assert ids_of(found) == [str(i) for i in range(450)], len(found)
    assert stub.offset_requests == [0, 100, 200, 300, 400], stub.offset_requests
    assert len(stub.keyset_requests) == 1, stub.keyset_requests  # asked once, not per page
    assert len(said) == 1, said
    for needle in ("/markets/keyset", "404", "falling back", "offset"):
        assert needle in said[0], (needle, said[0])


@check
def a_gamma_without_the_keyset_falls_back_to_offsets_with_a_warning() -> None:
    with GammaStub(total=450, keyset_status=404) as stub, logged_warnings() as said:
        found = discover(stub)
    assert_fell_back(found, stub, said)
    with GammaStub(total=450, keyset_status=404) as stub, logged_warnings() as said:
        found = discover_async(stub)
    assert_fell_back(found, stub, said)
    with GammaStub(total=450, keyset_status=404) as stub, logged_warnings() as said:
        found = fetch_all(stub, max_results=1000)
    assert_fell_back(found, stub, said)


@check
def the_fallback_still_ends_at_gamma_s_offset_limit_with_a_warning() -> None:
    with (
        GammaStub(total=10**6, keyset_status=405, offset_limit=300) as stub,
        logged_warnings() as said,
    ):
        found = discover(stub)
    assert ids_of(found) == [str(i) for i in range(400)], len(found)  # offsets 0..300
    assert len(said) == 2 and "falling back" in said[0], said
    assert "offset=400" in said[1] and "incomplete" in said[1], said


@check
def a_credentials_failure_is_not_a_missing_endpoint() -> None:
    for status in (401, 403):
        with GammaStub(total=450, keyset_status=status) as stub:
            error = refused(httpx.HTTPStatusError, discover, stub)
        assert str(status) in error, error
        assert stub.offset_requests == [], stub.offset_requests  # it did not try the other path


# ── 7. what is not a page, and what does not move ───────────────────────────────────────────────
@check
def an_invalid_cursor_midway_raises_and_is_not_a_fallback() -> None:
    script: dict[int, list[Reply]] = {300: [(422, {}, INVALID_CURSOR)]}
    with GammaStub(total=600, script=script) as stub:
        error = refused(httpx.HTTPStatusError, discover, stub)
    assert "422" in error, error
    assert stub.offset_requests == [] and len(stub.keyset_requests) == 4, stub.keyset_requests


@check
def a_reply_that_is_not_a_keyset_page_is_a_value_error() -> None:
    for body in ("[]", '{"markets": 3}', '{"data": []}', '{"markets": [], "next_cursor": 7}'):
        with GammaStub(total=600, script={100: [(200, {}, body)]}) as stub:
            error = refused(ValueError, discover, stub)
        assert "/markets/keyset" in error, (body, error)


@check
def a_cursor_that_does_not_move_raises_at_once() -> None:
    with GammaStub(total=1000, ignore_cursor=True) as stub:
        error = refused(RuntimeError, discover, stub)
    assert "same next_cursor" in error, error
    assert len(stub.keyset_requests) == 2, len(stub.keyset_requests)  # not a thousand
    with GammaStub(total=1000, ignore_cursor=True) as stub:
        assert "same next_cursor" in refused(RuntimeError, fetch_all, stub, max_results=5000)
    with GammaStub(total=1000, ignore_cursor=True) as stub:
        assert "same next_cursor" in refused(RuntimeError, discover_async, stub)


@check
def the_cap_of_1000_pages_stops_a_cursor_that_never_ends() -> None:
    with GammaStub(endless=True) as stub:
        error = refused(RuntimeError, discover, stub, page_size=1)
    assert "1000 pages" in error, error
    assert len(stub.keyset_requests) == 1000, len(stub.keyset_requests)


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
        print(f"audit gamma keyset: {len(failures)} of {len(CHECKS)} checks FAILED: {failures}")
        return 1
    print(f"audit gamma keyset: OK ({len(CHECKS)} checks, pmlab {pmlab.__version__})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
