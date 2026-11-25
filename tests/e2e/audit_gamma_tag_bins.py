"""Discovery by tag and the per-bin temperature markets (1.3.0, #86 and #87), used from the wheel.

Standalone (public names only, no repo imports):  python audit_gamma_tag_bins.py
Prints ``audit gamma tag bins: OK`` and exits 0, or lists every check that failed and exits 1.

Live Gamma lists no temperature market with bins as its outcomes: it lists one Yes/No market per bin
(``Will the highest temperature in Warsaw be -4°C or below on November 26?``) and ties a city's and
day's eleven together as an event. They sit under the ``Daily Temperature`` tag (103040), which the
keyset endpoint filters by ``tag_id``; the unfiltered universe is over 100,000 markets.

The stub here plays that on 127.0.0.1 (free port, stdlib ``http.server``, no network, no sleeping):
1,500 markets that are not weather, then the 32 recorded from live Gamma on 2026-11-25
(``tests/markets/fixtures/per_bin_markets.json``: Atlanta and Warsaw on November 26 in full, Panama
City with 3 of its 11 bins, Atlanta on May 15 with 1, a stale Jinan event with no token ids, and two
lowest-temperature markets). Only the 32 carry the tag. A walk of the whole stub is 307 pages of 5;
one scoped to the tag is 7. The plugin check puts 1,100 more markets under the tag, before the
32, so that 1,132 are behind it: a walk that stopped at 1000 markets would find none of them.

Every check is independent and reports on its own, so a run against an older wheel says which part
it lacks.

The plugin and the ``pmlab scan-markets`` command read a whole tag in pages of 100 whatever
``limit`` they are given, so the 32 are one request; they are run from the installed wheel, as is
an injected client written for 1.2.1 (its ``fetch_markets_all`` takes no ``tag_id``).
"""

from __future__ import annotations

import asyncio
import contextlib
import http.server
import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import traceback
import urllib.parse
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pmlab
from pmlab import AsyncGammaClient, GammaClient, WeatherTmaxPlugin

CHECKS: list[tuple[str, Callable[[], None]]] = []
FIXTURE = Path(__file__).resolve().parents[1] / "markets" / "fixtures" / "per_bin_markets.json"
CURSOR_PREFIX = "v1+/=&%20?#"  # opaque, and every character that breaks an unescaped query string
TAG = 103040
FILLER = 1500
PAGE = 5


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


def recorded() -> list[dict[str, Any]]:
    return json.loads(FIXTURE.read_text())["markets"]


def filler(n: int = FILLER) -> list[dict[str, Any]]:
    return [{"id": f"f{i}", "question": f"Will event {i} happen?"} for i in range(n)]


def multi_outcome() -> dict[str, Any]:
    """The other form: one market whose outcomes are the bins."""
    return {
        "id": "multi1",
        "question": "Highest temperature in Paris on May 13?",
        "slug": "multi-1",
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


class TagStub:
    """``/markets/keyset`` and ``/markets`` of a Gamma that knows tags.

    ``universe`` is served in order. ``tagged`` are the ids that carry the tag ``TAG``: a request
    with ``tag_id=TAG`` gets those only, another ``tag_id`` gets none (Gamma answers an unknown tag
    with an empty page), none gets everything. ``keyset_status`` answers every keyset request with
    that status (an old Gamma), which sends discovery to ``/markets`` by ``offset``. The cursor is
    the position in what the query matches, behind a prefix a query string cannot carry unescaped.
    """

    def __init__(
        self,
        universe: list[dict[str, Any]],
        tagged: set[str],
        *,
        keyset_status: int | None = None,
    ) -> None:
        self.universe = universe
        self.tagged = tagged
        self.keyset_status = keyset_status
        self.keyset_requests: list[dict[str, str]] = []
        self.offset_requests: list[dict[str, str]] = []
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
                    self.reply(400, '{"error": "unexpected request"}')

            def reply(self, status: int, body: str) -> None:
                data = body.encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: object) -> None:  # keep the audit's output readable
                pass

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"

    def matching(self, query: dict[str, str]) -> list[dict[str, Any]]:
        tag = query.get("tag_id")
        if tag is None:
            return self.universe
        if tag != str(TAG):
            return []
        return [m for m in self.universe if m["id"] in self.tagged]

    def keyset(self, handler: Any, query: dict[str, str]) -> None:
        self.keyset_requests.append(query)
        if self.keyset_status is not None:
            handler.reply(self.keyset_status, json.dumps({"error": self.keyset_status}))
            return
        limit, cursor = int(query["limit"]), query.get("after_cursor")
        start = 0
        if cursor is not None:
            if not (
                cursor.startswith(CURSOR_PREFIX) and cursor.removeprefix(CURSOR_PREFIX).isdigit()
            ):
                handler.reply(422, '{"type": "validation error", "error": "invalid cursor"}')
                return
            start = int(cursor.removeprefix(CURSOR_PREFIX))
        found = self.matching(query)
        body: dict[str, Any] = {"markets": found[start : start + limit]}
        if start + limit < len(found):
            body["next_cursor"] = f"{CURSOR_PREFIX}{start + limit}"
        handler.reply(200, json.dumps(body))

    def offsets(self, handler: Any, query: dict[str, str]) -> None:
        self.offset_requests.append(query)
        offset, limit = int(query["offset"]), int(query["limit"])
        handler.reply(200, json.dumps(self.matching(query)[offset : offset + limit]))

    def __enter__(self) -> TagStub:
        # a proxy set in the environment must not stand between the client and 127.0.0.1
        os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()


def lowest_clones(n: int) -> list[dict[str, Any]]:
    """``n`` lowest-temperature markets (a family that is not read) with ids of their own."""
    template = next(m for m in recorded() if "lowest temperature" in m["question"])
    return [{**template, "id": f"low{i}"} for i in range(n)]


def world(*, tagged_extra: int = 0, **kw: Any) -> TagStub:
    """1,500 markets that are not weather, then the recorded ones, which carry the tag.

    ``tagged_extra`` puts that many more markets under the tag, before the recorded ones: a walk
    that stops at 1000 markets then never reaches the recorded ones.
    """
    weather = recorded()
    extra = lowest_clones(tagged_extra)
    return TagStub([*filler(), *extra, *weather], {m["id"] for m in [*extra, *weather]}, **kw)


def discover(stub: TagStub, **kw: Any) -> list[Any]:
    kw.setdefault("page_size", PAGE)
    with GammaClient(base_url=stub.url) as client:
        return client.discover_tmax_markets(**kw)


def discover_async(stub: TagStub, **kw: Any) -> list[Any]:
    kw.setdefault("page_size", PAGE)

    async def run() -> list[Any]:
        async with AsyncGammaClient(base_url=stub.url) as client:
            return await client.discover_tmax_markets(**kw)

    return asyncio.run(run())


def by_id(found: list[Any]) -> dict[str, Any]:
    return {m.market_id: m for m in found}


def raw_by_event(event: str) -> list[dict[str, Any]]:
    return [m for m in recorded() if m["events"][0]["id"] == event]


@check
def this_is_the_installed_wheel_not_a_checkout() -> None:
    assert "site-packages" in pmlab.__file__, pmlab.__file__


# ── 1. discovery is scoped by tag ───────────────────────────────────────────────────────────────
@check
def a_default_discover_tmax_markets_reads_the_tag_and_not_the_universe() -> None:
    with world() as stub, logged_warnings():
        found = discover(stub)
    assert len(stub.keyset_requests) == 7, len(stub.keyset_requests)  # 32 markets, 5 a page
    assert all(r["tag_id"] == str(TAG) for r in stub.keyset_requests), stub.keyset_requests
    assert stub.keyset_requests[0] == {
        "limit": str(PAGE),
        "active": "true",
        "closed": "false",
        "tag_id": str(TAG),
    }, stub.keyset_requests[0]
    assert sorted(by_id(found)) == ["1436284", "1451212", "1451693", "479718"], sorted(by_id(found))


@check
def tag_id_none_reads_the_whole_universe_as_1_2_2_did() -> None:
    with world() as stub, logged_warnings():
        found = discover(stub, tag_id=None)
    assert len(stub.keyset_requests) == 307, len(stub.keyset_requests)  # (1,500 + 32) / 5
    assert all("tag_id" not in r for r in stub.keyset_requests)
    assert sorted(by_id(found)) == ["1436284", "1451212", "1451693", "479718"]


@check
def async_discover_tmax_markets_is_scoped_and_groups_too() -> None:
    with world() as stub, logged_warnings():
        found = discover_async(stub)
    assert len(stub.keyset_requests) == 7
    assert all(r["tag_id"] == str(TAG) for r in stub.keyset_requests)
    assert [len(m.outcome_labels) for m in found] == [1, 3, 11, 11], [m.market_id for m in found]
    with world() as stub, logged_warnings():
        assert len(discover_async(stub, tag_id=None)) == 4
    assert len(stub.keyset_requests) == 307 and all("tag_id" not in r for r in stub.keyset_requests)


@check
def fetch_markets_all_sends_tag_id_on_every_page_and_only_when_asked() -> None:
    with world() as stub, GammaClient(base_url=stub.url) as client:
        scoped = client.fetch_markets_all(page_size=PAGE, tag_id=TAG, max_results=10**6)
    assert len(scoped) == 32 and len(stub.keyset_requests) == 7
    assert all(r["tag_id"] == str(TAG) for r in stub.keyset_requests)
    with world() as stub, GammaClient(base_url=stub.url) as client:
        everything = client.fetch_markets_all(page_size=100, max_results=10**6)
    assert len(everything) == FILLER + 32  # no tag_id: not scoped, as before
    assert all("tag_id" not in r for r in stub.keyset_requests)


@check
def the_offset_fallback_is_scoped_too() -> None:
    with world(keyset_status=404) as stub, logged_warnings() as said:
        found = discover(stub)
    assert len(found) == 4 and stub.offset_requests, (len(found), stub.offset_requests)
    assert all(r["tag_id"] == str(TAG) for r in stub.offset_requests), stub.offset_requests[:2]
    assert len(stub.offset_requests) == 7
    assert any("falling back to offset paging" in m for m in said), said


@check
def a_bad_tag_id_is_refused_before_any_request() -> None:
    for bad in (True, 0, -1, 1.5, "103040"):
        with world() as stub:
            assert "tag_id" in refused(ValueError, discover, stub, tag_id=bad), bad
            assert "tag_id" in refused(ValueError, discover_async, stub, tag_id=bad), bad
            with GammaClient(base_url=stub.url) as client:
                assert "tag_id" in refused(ValueError, client.fetch_markets_all, tag_id=bad), bad
        assert stub.keyset_requests == [], (bad, stub.keyset_requests)


@check
def a_scoped_walk_that_reads_nothing_says_the_tag_may_be_gone() -> None:
    with world() as stub, logged_warnings() as said:
        assert discover(stub, tag_id=999999) == []
    assert len(said) == 1 and "999999" in said[0] and "no markets" in said[0], said
    with TagStub([], set()) as empty, logged_warnings() as said:  # unscoped, empty: no tag to blame
        assert discover(empty, tag_id=None) == []
    assert said == [], said


# ── 2. the per-bin markets ──────────────────────────────────────────────────────────────────────
@check
def the_per_bin_markets_of_a_city_and_day_are_one_entry_with_yes_tokens_and_prices() -> None:
    with world() as stub, logged_warnings():
        found = discover(stub)
    atlanta = by_id(found)["1451693"]
    bins = raw_by_event("1451693")
    assert atlanta.question == "Highest temperature in Atlanta on November 26?"
    assert (atlanta.city, atlanta.target_date, atlanta.unit) == ("Atlanta", "2026-11-26", "F")
    assert atlanta.slug == "highest-temperature-in-atlanta-on-november-26-2026"
    assert atlanta.outcome_labels == [m["groupItemTitle"] for m in bins], atlanta.outcome_labels
    assert atlanta.token_ids == [json.loads(m["clobTokenIds"])[0] for m in bins]  # the Yes tokens
    assert atlanta.outcome_prices == [float(json.loads(m["outcomePrices"])[0]) for m in bins]
    warsaw = by_id(found)["1451212"]  # negative degrees, an open bottom and an open top
    assert (warsaw.unit, warsaw.target_date) == ("C", "2026-11-26")
    assert warsaw.outcome_labels[0] == "-6°C or below" and warsaw.outcome_labels[1] == "-5°C"
    assert warsaw.outcome_labels[-1].endswith("°C or higher")
    assert [len(m.outcome_labels) for m in found] == [1, 3, 11, 11]  # the partial events are kept


@check
def the_stale_event_without_tokens_is_refused_with_one_warning() -> None:
    with world() as stub, logged_warnings() as said:
        found = discover(stub)
    assert "497034" not in by_id(found)
    assert len(said) == 1, said
    assert "4 per-bin temperature markets were not read" in said[0] and "no Yes token id" in said[0]
    assert "2290078" in said[0], said[0]


@check
def lowest_temperature_markets_and_unreadable_questions_are_not_returned() -> None:
    lowest = [m for m in recorded() if "lowest temperature" in m["question"]]
    odd = {**raw_by_event("1451693")[0], "id": "odd1"}
    odd["question"] = "Will the highest temperature in Atlanta exceed 50°F on November 26?"
    with TagStub([*lowest], {m["id"] for m in lowest}) as stub, logged_warnings() as said:
        assert discover(stub) == [] and said == [], (
            said
        )  # another family: not read, not warned about
    with TagStub([odd], {"odd1"}) as stub, logged_warnings() as said:
        assert discover(stub) == []
    assert len(said) == 1 and "1 per-bin temperature market was not read" in said[0], said


@check
def the_multi_outcome_form_still_works_beside_the_per_bin_one() -> None:
    weather = [multi_outcome(), *recorded()]
    with TagStub(weather, {m["id"] for m in weather}) as stub, logged_warnings():
        found = discover(stub)
    assert [m.market_id for m in found] == ["multi1", "479718", "1436284", "1451212", "1451693"]
    assert found[0].outcome_labels == ["30°C", "31°C"] and found[0].city == "Paris"


@check
def a_price_gamma_did_not_give_is_nan_and_never_zero() -> None:
    market = {**raw_by_event("1451693")[0], "outcomePrices": None}
    with TagStub([market], {market["id"]}) as stub, logged_warnings():
        (entry,) = discover(stub)
    assert len(entry.outcome_prices) == 1 and math.isnan(entry.outcome_prices[0])


# ── 3. the plugin ───────────────────────────────────────────────────────────────────────────────
class Truth:
    def __init__(self, values: dict[tuple[str, str], float]) -> None:
        self.values = values

    def get_daily_max(self, city: str, date: str) -> float | None:
        return self.values.get((city, date))


def plugin(stub: TagStub, client: GammaClient, truth: Any = None) -> WeatherTmaxPlugin:
    seen = datetime(2026, 11, 25, 7, 0, tzinfo=UTC)
    return WeatherTmaxPlugin(gamma_client=client, truth_client=truth, clock=lambda: seen)


@check
def the_plugin_reads_the_whole_tag_past_a_thousand_markets_and_builds_one_spec_per_day() -> None:
    # 1,132 markets under the tag, the recorded 32 the last of them: past the 1000th
    with (
        world(tagged_extra=1100) as stub,
        GammaClient(base_url=stub.url) as client,
        logged_warnings() as said,
    ):
        specs = plugin(stub, client).discover_markets(limit=PAGE)
    assert [s.market_id for s in specs] == ["479718", "1436284", "1451212", "1451693"]
    assert all(r["tag_id"] == str(TAG) for r in stub.keyset_requests)
    # full pages of 100 however small the limit named (5 here): 11 of them, then 32 markets
    assert len(stub.keyset_requests) == 12, len(stub.keyset_requests)
    assert {r["limit"] for r in stub.keyset_requests} == {"100"}, stub.keyset_requests
    atlanta = specs[-1]
    assert [b.label for b in atlanta.outcome_bins] == [
        m["groupItemTitle"] for m in raw_by_event("1451693")
    ]
    meta = atlanta.metadata
    assert (meta["city"], meta["target_date"], meta["unit"]) == ("Atlanta", "2026-11-26", "F")
    assert meta["token_ids"] == [json.loads(m["clobTokenIds"])[0] for m in raw_by_event("1451693")]
    assert [e["price"] for e in meta["outcome_schema"]] == [
        float(json.loads(m["outcomePrices"])[0]) for m in raw_by_event("1451693")
    ]
    assert meta["bins_complete"] is True and specs[1].metadata["bins_complete"] is False
    assert meta["price_source"] == "discovery" and meta["price_observed_at"].startswith(
        "2026-11-25T07"
    )
    assert [b.label for s in specs for b in s.outcome_bins if b.label in ("Yes", "No")] == []
    assert len(said) == 1 and "no Yes token id" in said[0], said


@check
def the_plugin_unscoped_is_what_it_was_the_first_1000_markets() -> None:
    with world() as stub, GammaClient(base_url=stub.url) as client, logged_warnings():
        specs = plugin(stub, client).discover_markets(tag_id=None, limit=100)
    assert specs == []  # the first 1000 of the stub hold none of the weather markets
    assert len(stub.keyset_requests) == 10 and all("tag_id" not in r for r in stub.keyset_requests)


@check
def the_bins_of_a_full_event_tile_the_line() -> None:
    with world() as stub, GammaClient(base_url=stub.url) as client, logged_warnings():
        specs = {s.market_id: s for s in plugin(stub, client).discover_markets(limit=PAGE)}
    for event in ("1451693", "1451212"):
        spec = specs[event]
        for tenths in range(-300, 1300):
            t = tenths / 10
            n = sum(b.contains(t) for b in spec.outcome_bins)
            assert n == 1, (event, t, n)
    atlanta = specs["1451693"]
    for reading, winner in (
        (52.0, "52-53°F"),
        (53.0, "52-53°F"),
        (51.4, "50-51°F"),
        (90.0, "62°F or higher"),
    ):
        assert atlanta.resolve_winning_bin(reading) == winner, (
            reading,
            atlanta.resolve_winning_bin(reading),
        )
    assert specs["1451212"].resolve_winning_bin(-5.0) == "-5°C"
    assert specs["1436284"].resolve_winning_bin(33.0) is None  # the 33°C bin has closed


@check
def training_rows_are_one_per_bin_at_that_bins_own_yes_price() -> None:
    truth = Truth({("Atlanta", "2026-11-26"): 52.0})
    with world() as stub, GammaClient(base_url=stub.url) as client, logged_warnings():
        p = plugin(stub, client, truth)
        spec = next(s for s in p.discover_markets(limit=PAGE) if s.market_id == "1451693")
    exclusions: dict[str, int] = {}
    rows = p.build_training_rows(
        spec, "morning_of", decision_time="2026-11-25T08:00:00Z", exclusions=exclusions
    )
    prices = {
        m["groupItemTitle"]: float(json.loads(m["outcomePrices"])[0])
        for m in raw_by_event("1451693")
    }
    assert exclusions == {}, exclusions
    assert {r["outcome_label"]: r["market_price"] for r in rows} == prices
    assert {r["winning_label"] for r in rows} == {"52-53°F"} and len(rows) == 11
    late: dict[str, int] = {}  # the target day is a real date: a decision after it is refused
    assert p.build_training_rows(
        spec, "late", decision_time="2026-11-27T01:00:00Z", exclusions=late
    ) == [] and late == {"decision_after_close": 1}, late


# ── 4. the command, an injected client, a small limit ───────────────────────────────────────────
def pmlab_command(stub: TagStub, *args: str) -> subprocess.CompletedProcess[str]:
    """The installed ``pmlab`` console script (the wheel's entry point) against the stub.

    Run in an empty directory, so a ``.env`` of the checkout cannot be read as the settings, and
    wide, so the table is not wrapped.
    """
    exe = Path(sys.executable).parent / "pmlab"
    env = {**os.environ, "GAMMA_API_BASE": stub.url, "COLUMNS": "200"}
    with tempfile.TemporaryDirectory() as empty:
        return subprocess.run(
            [str(exe), *args], capture_output=True, text=True, env=env, cwd=empty, timeout=120
        )


@check
def the_scan_markets_command_reads_the_tag_and_lists_the_per_bin_markets() -> None:
    with world() as stub:
        done = pmlab_command(stub, "scan-markets", "--plugin", "weather_tmax")
    assert done.returncode == 0, (done.returncode, done.stdout, done.stderr)
    assert "4 'weather_tmax' markets found" in done.stdout, done.stdout
    assert "Highest temperature in Atlanta on November 26?" in done.stdout, done.stdout
    assert {q["tag_id"] for q in stub.keyset_requests} == {str(TAG)}, stub.keyset_requests
    assert len(stub.keyset_requests) == 1, len(stub.keyset_requests)  # 32 markets, a page of 100


class Gamma121:
    """An injected client written for 1.2.1: ``fetch_markets_all`` takes no ``tag_id``, no
    ``**kwargs``. It reads the stub through the real client, so what it returns is real."""

    def __init__(self, client: GammaClient) -> None:
        self.client = client

    def fetch_markets_all(
        self,
        tag: str | None = None,
        keyword: str | None = None,
        page_size: int = 100,
        max_results: int = 1000,
        active: bool = True,
        closed: bool = False,
    ) -> list[dict[str, Any]]:
        return self.client.fetch_markets_all(
            tag=tag,
            keyword=keyword,
            page_size=page_size,
            max_results=max_results,
            active=active,
            closed=closed,
        )


@check
def an_injected_client_without_tag_id_is_called_as_in_1_2_1_and_warns_once() -> None:
    weather = recorded()
    with (
        TagStub(weather, set()) as stub,
        GammaClient(base_url=stub.url) as client,
        logged_warnings() as said,
    ):
        specs = WeatherTmaxPlugin(gamma_client=Gamma121(client)).discover_markets()
        assert [s.market_id for s in specs] == ["479718", "1436284", "1451212", "1451693"]
        assert all("tag_id" not in r for r in stub.keyset_requests), stub.keyset_requests
        about_tag = [m for m in said if "tag_id" in m]
        assert len(about_tag) == 1 and "not scoped" in about_tag[0], said
        said.clear()
        WeatherTmaxPlugin(gamma_client=Gamma121(client)).discover_markets(tag_id=None)
        assert [m for m in said if "tag_id" in m] == [], said  # asked for unscoped: not warned
        # the caller named a scope the client cannot give: that is theirs to see, as a TypeError
        assert "tag_id" in refused(
            TypeError, WeatherTmaxPlugin(gamma_client=Gamma121(client)).discover_markets, tag_id=TAG
        )


@check
def a_small_limit_does_not_multiply_the_requests_of_a_whole_tag_read() -> None:
    with world() as stub, GammaClient(base_url=stub.url) as client, logged_warnings():
        specs = plugin(stub, client).discover_markets(limit=1)
    assert len(specs) == 4
    assert len(stub.keyset_requests) == 1, len(stub.keyset_requests)  # it was 32 of 1
    assert {r["limit"] for r in stub.keyset_requests} == {"100"}, stub.keyset_requests
    with world() as stub:
        done = pmlab_command(stub, "scan-markets", "--plugin", "weather_tmax", "--limit", "1")
    assert done.returncode == 0, (done.returncode, done.stdout, done.stderr)
    assert "4 'weather_tmax' markets found" in done.stdout, done.stdout
    assert len(stub.keyset_requests) == 1, stub.keyset_requests
    # raising a small page must not make an invalid one valid: --limit 0 is still refused, no request
    with world() as stub:
        refused_limit = pmlab_command(
            stub, "scan-markets", "--plugin", "weather_tmax", "--limit", "0"
        )
    said = " ".join((refused_limit.stdout + refused_limit.stderr).split())
    assert refused_limit.returncode == 1, (refused_limit.returncode, said)
    assert "page_size must be an int between 1 and 100" in said, said
    assert stub.keyset_requests == [], stub.keyset_requests
    with world() as stub:
        helped = pmlab_command(stub, "scan-markets", "--help")
    text = " ".join(re.sub(r"[│╭╮╰╯─]", " ", helped.stdout).split())
    assert helped.returncode == 0 and "Max markets to fetch" not in text, text
    assert "page size" in text.lower() and "not a number of markets" in text, text


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
        print(f"audit gamma tag bins: {len(failures)} of {len(CHECKS)} checks FAILED: {failures}")
        return 1
    print(f"audit gamma tag bins: OK ({len(CHECKS)} checks, pmlab {pmlab.__version__})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
