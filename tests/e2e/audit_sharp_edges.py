"""The sharp edges the 1.1.0 audit left (1.2.1, #68), used from the installed wheel.

Standalone (public names only, no repo imports):  python audit_sharp_edges.py
Prints ``audit sharp edges: OK`` and exits 0, or lists every check that failed and exits 1.

Every check is independent and reports on its own, so a run against an older wheel says which
fixes it lacks. ``pmlab`` is run as a user runs it: the console script beside this interpreter.
"""

from __future__ import annotations

import asyncio
import contextlib
import http.server
import json
import logging
import math
import os
import pathlib
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.parse
import warnings
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

import pmlab
from pmlab import (
    AsyncGammaClient,
    CalibrationTracker,
    ChampionManifest,
    EdgeSignal,
    GammaClient,
    HoldoutGateResult,
    LGBMForecaster,
    MarketPlugin,
    MarketSpec,
    OutcomeBin,
    PanelSchemaError,
    PaperBroker,
    SegmentGateResult,
    SettlementEngine,
    WeatherTmaxPlugin,
    build_panel,
    segment_allowed,
)
from pmlab.datasets import load_sample
from pmlab.plugins.sports_f1.plugin import SportsF1Plugin
from pmlab.reproducibility import compare_metrics

PMLAB = Path(sys.executable).with_name("pmlab")
CHECKS: list[tuple[str, Callable[[], None]]] = []


def check(fn: Callable[[], None]) -> Callable[[], None]:
    CHECKS.append((fn.__name__, fn))
    return fn


@contextlib.contextmanager
def workdir() -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="pmlab-sharp-") as name:
        yield Path(name)


def refused(
    exc_type: type[Exception], fn: Callable[..., object], *args: object, **kw: object
) -> str:
    try:
        fn(*args, **kw)
    except exc_type as error:
        return str(error)
    raise AssertionError(f"{getattr(fn, '__name__', fn)}{args}{kw} was accepted")


def cli(*args: str | Path, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(PMLAB), *map(str, args)],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
        env={**os.environ, "NO_COLOR": "1", "TERM": "dumb", "COLUMNS": "200"},
    )


@check
def this_is_the_installed_wheel_not_a_checkout() -> None:
    assert "site-packages" in pmlab.__file__, pmlab.__file__


# ── 1. build_panel: one row per bin of a market and decision ────────────────────────────────────
def bin_rows(market: str, date: str = "2025-01-05") -> list[dict[str, Any]]:
    prices = {"a": 0.2, "b": 0.5, "c": 0.3}
    return [
        {
            "market_id": market,
            "decision_date": date,
            "outcome_label": label,
            "winning_label": "b",
            "market_price": price,
            "feature_x": 1.0,
        }
        for label, price in prices.items()
    ]


@check
def build_panel_refuses_duplicated_rows() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore"
        )  # these toy rows carry no price_ts/decision_ts: not the point
        _build_panel_refuses_duplicated_rows()


def _build_panel_refuses_duplicated_rows() -> None:
    other_bins_dates_markets = [bin_rows("m1"), bin_rows("m1", date="2025-01-06"), bin_rows("m2")]
    assert len(build_panel(other_bins_dates_markets)) == 9  # not duplicates: still builds
    rows = bin_rows("m1") + bin_rows("m2")
    rows.append(dict(rows[0]))
    message = refused(PanelSchemaError, build_panel, [rows])
    assert "duplicate" in message.lower() and "market_id='m1'" in message, message
    assert "outcome_label='a'" in message and "decision_date='2025-01-05'" in message, message
    refetched = bin_rows("m1")
    for row in refetched:
        row["feature_x"] = 2.0  # same market and day, another snapshot: which is right is not ours
    refused(PanelSchemaError, build_panel, [bin_rows("m1"), refetched])
    # a horizon column under another name makes rows look repeated; the message says what to name it
    by_days = [{**r, "horizon_days": d} for d in (1, 2) for r in bin_rows("m1")]
    hint = refused(PanelSchemaError, build_panel, [by_days])
    assert "decision_horizon" in hint and "'horizon'" in hint, hint
    named = [{**r, "decision_horizon": f"h{d}"} for d in (1, 2) for r in bin_rows("m1")]
    assert len(build_panel([named])) == 6  # naming it that way is the way out


# ── 2. MarketSpec and OutcomeBin ────────────────────────────────────────────────────────────────
def spec(bins: list[OutcomeBin] | None = None, close_time: str = "2026-05-10") -> MarketSpec:
    return MarketSpec(
        market_id="m1",
        slug="s",
        question="q",
        outcome_bins=[OutcomeBin(label="YES"), OutcomeBin(label="NO")] if bins is None else bins,
        close_time=close_time,
        market_family="binary",
    )


@check
def market_spec_and_outcome_bin_refuse_impossible_input() -> None:
    refused(ValueError, OutcomeBin, label="nan", lower=math.nan)
    refused(ValueError, OutcomeBin, label="nan", upper=math.nan)
    message = refused(ValueError, OutcomeBin, label="backwards", lower=30.0, upper=20.0)
    assert "backwards" in message, message
    assert "duplicate" in refused(
        ValueError, spec, [OutcomeBin(label="YES"), OutcomeBin(label="YES")]
    )
    assert "ISO-8601" in refused(ValueError, spec, close_time="tomorrow")
    refused(ValueError, spec, close_time="2026-02-30")  # no such day


@check
def market_spec_still_accepts_what_sparse_payloads_need() -> None:
    OutcomeBin(label="ok", lower=20.0, upper=20.0)  # a point bin is a bin
    OutcomeBin(label="open", lower=20.0)
    for close_time in (
        "",
        "   ",
        "2026-05-10",
        "2024-01-15T20:00:00Z",
        "2024-01-15T20:00:00+02:00",
    ):
        assert spec(close_time=close_time).close_time == close_time
    assert spec(bins=[]).outcome_bins == []  # discovery keeps sparse Gamma payloads


# ── 3. a gate file that says GO must allow a segment ────────────────────────────────────────────
def segment(name: str, passes: object) -> dict[str, Any]:
    ok = passes is True
    return {
        "segment": name,
        "num_trades": 100,
        "total_pnl": 10.0 if ok else -3.0,
        "passes": passes,
        "reason": "ok" if ok else "pnl below the minimum",
    }


def gate_dict(decision: object, *segments: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision": decision,
        "segment_results": list(segments),
        "aggregate_pnl": 7.0,
        "aggregate_trades": 200,
    }


class StubModel:
    def save(self, path: Path) -> None:
        Path(path).write_bytes(b"stub")


@check
def a_forged_go_gate_is_refused() -> None:
    nothing_passed = gate_dict("GO", segment("nyc", False), segment("london", False))
    assert "no segment passed" in refused(ValueError, HoldoutGateResult.from_dict, nothing_passed)
    for decision in ("go", "MAYBE", "", None, True):
        message = refused(
            ValueError, HoldoutGateResult.from_dict, gate_dict(decision, segment("a", True))
        )
        assert "GO" in message and "NO_GO" in message, (decision, message)
    for passes in ("false", "true", 1, 0, None):  # the string "false" is truthy
        message = refused(
            ValueError, HoldoutGateResult.from_dict, gate_dict("GO", segment("a", passes))
        )
        assert "passes" in message and "'a'" in message, (passes, message)
    with workdir() as tmp:
        built = HoldoutGateResult(  # built in Python, not loaded: publish checks it as well
            decision="GO",
            segment_results=[SegmentGateResult("nyc", 100, -3.0, False, "pnl below the minimum")],
            aggregate_pnl=-3.0,
            aggregate_trades=100,
        )
        message = refused(
            ValueError, ChampionManifest.publish, StubModel(), built, tmp / "out", "demo"
        )
        assert "no segment passed" in message, message
        assert not (tmp / "out").exists()  # refused before anything was written


@check
def a_documented_go_gate_with_a_failed_segment_still_loads_and_publishes() -> None:
    mixed = gate_dict("GO", segment("politics", True), segment("sports", False))
    loaded = HoldoutGateResult.from_dict(mixed)
    assert loaded.to_dict() == mixed
    no_go = gate_dict("NO_GO", segment("a", False))
    assert HoldoutGateResult.from_dict(no_go).to_dict() == no_go
    with workdir() as tmp:
        manifest = ChampionManifest.publish(StubModel(), loaded, tmp / "out", "demo")
        assert manifest.get_allowed_segments() == {"politics"}
        assert (tmp / "out" / "champion.json").is_file()


def small_model(path: Path) -> Path:
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"feature_x": rng.normal(size=80)})
    model = LGBMForecaster(n_estimators=5, num_leaves=4, min_child_samples=5)
    model.fit(X, pd.Series((X["feature_x"] > 0).astype(int)))
    model.save(path)
    return path


@check
def promote_champion_refuses_a_forged_gate_file_and_warns_about_an_unknown_family() -> None:
    with workdir() as tmp:
        model = small_model(tmp / "model.pkl")
        forged, genuine = tmp / "forged.json", tmp / "gate.json"
        forged.write_text(
            json.dumps(gate_dict("GO", segment("nyc", False), segment("london", False)))
        )
        genuine.write_text(json.dumps(gate_dict("GO", segment("A", True))))

        def promote(gate: Path, family: str, out: str) -> subprocess.CompletedProcess[str]:
            return cli(
                "promote-champion", model, "--gate-path", gate, "--plugin", family,
                "--output-dir", tmp / out, cwd=tmp,
            )  # fmt: skip

        refusal = promote(forged, "weather_tmax", "refused")
        both = refusal.stdout + refusal.stderr
        assert refusal.returncode == 1, (refusal.returncode, both)
        assert "Not a valid gate file" in both and "no segment passed" in both, both
        assert "Traceback" not in both and not (tmp / "refused" / "champion.json").exists()

        typo = promote(genuine, "weather_tmx", "typo")  # flagged, not refused
        assert typo.returncode == 0, typo.stdout + typo.stderr
        assert "warning:" in typo.stderr and "'weather_tmx'" in typo.stderr, typo.stderr
        assert "weather_tmax" in typo.stderr and "sports_f1" in typo.stderr, typo.stderr
        assert "warning" not in typo.stdout  # stdout stays what it was
        recorded = json.loads((tmp / "typo" / "champion.json").read_text())
        assert recorded["plugin_family"] == "weather_tmx"  # recorded as given

        known = promote(genuine, "weather_tmax", "known")
        assert known.returncode == 0 and "warning" not in (known.stdout + known.stderr).lower()


# ── 5. a str is not a collection of segments ────────────────────────────────────────────────────
@check
def a_str_for_allowed_segments_is_refused() -> None:
    assert segment_allowed("Buenos Aires", {"Buenos Aires"}) is True
    assert segment_allowed("Aires", {"Buenos Aires"}) is False
    assert "str" in refused(TypeError, segment_allowed, "Aires", "Buenos Aires")
    with workdir() as tmp:
        log = tmp / "trades.json"
        message = refused(TypeError, PaperBroker, trades_path=log, allowed_segments="Buenos Aires")
        assert "allowed_segments" in message, message
        assert not log.exists()  # refused in the constructor, before any trade
        PaperBroker(trades_path=log, allowed_segments={"Buenos Aires"})
    message = refused(
        TypeError, HoldoutGateResult.evaluate, pd.DataFrame(), required_segments="nyc"
    )
    assert "required_segments" in message, message


# ── 6. CalibrationTracker ───────────────────────────────────────────────────────────────────────
def settled(prob: object, occurred: bool, segment_name: str = "politics") -> dict[str, Any]:
    return {
        "recorded_at": "2026-06-15T05:00:00+00:00",
        "city_or_segment": segment_name,
        "target_date": "2026-06-15",
        "outcome_label": "YES",
        "direction": "yes",
        "gamma_price": 0.5,
        "fill_price": 0.5,
        "model_prob": prob,
        "edge_after_fee": 0.1,
        "horizon": "morning_of",
        "flat_stake": 10.0,
        "size": 20.0,
        "fee_paid": 0.03,
        "outcome": "won" if occurred else "lost",
        "realized_pnl": 1.0,
    }


@check
def calibration_tracker_leaves_out_non_probabilities_and_checks_its_inputs() -> None:
    good = [settled(0.7, True), settled(0.3, False), settled(0.6, True)]
    for bad in (math.nan, math.inf, -math.inf, 1.5, -0.01):
        tracker = CalibrationTracker([*good, settled(bad, True)])
        assert (tracker.n_realized, tracker.n_skipped) == (3, 1), (bad, tracker.n_realized)
        window = tracker.overall()  # used to fail here, far from the cause
        assert window is not None and window.n_trades == 3 and math.isfinite(window.brier_score)
    edge = CalibrationTracker([settled(0.0, False), settled(1.0, True)])
    assert (edge.n_realized, edge.n_skipped) == (2, 0)  # the ends of [0, 1] are forecasts
    assert "n_bins" in refused(ValueError, CalibrationTracker, good, n_bins=0)
    for bad_bins in (2.5, True, None, "10"):
        assert "n_bins" in refused(TypeError, CalibrationTracker, good, n_bins=bad_bins)
    assert CalibrationTracker(good, n_bins=5).overall() is not None


@check
def calibration_tracker_says_what_is_wrong_with_a_file_that_is_not_a_trade_log() -> None:
    with workdir() as tmp:
        path = tmp / "trades.json"
        for payload in ([], {}, {"trades": "x"}, {"trades": [1, 2]}, {"other": 1}):
            path.write_text(json.dumps(payload))
            message = refused(ValueError, CalibrationTracker.from_path, path)
            assert "trades.json" in message and "trades" in message, (payload, message)
        path.write_text(json.dumps({"trades": [settled(0.7, True), settled(0.3, False)]}))
        assert CalibrationTracker.from_path(path).n_realized == 2
        refused(FileNotFoundError, CalibrationTracker.from_path, tmp / "missing.json")


# ── 8. the trade log is replaced whole or not at all; "today" is a UTC date ─────────────────────
def signal(city: str) -> EdgeSignal:
    return EdgeSignal(
        market_id="mkt-001",
        city_or_segment=city,
        target_date="2099-12-31",
        horizon="morning_of",
        outcome_label="YES",
        direction="yes",
        gamma_price=0.6,
        model_prob=0.72,
        best_edge=0.12,
        yes_edge=0.12,
        no_edge=-0.02,
    )


def disk_full(*args: object, **kwargs: object) -> None:
    raise OSError(28, "No space left on device")


@check
def a_failed_write_leaves_the_trade_log_it_had() -> None:
    with workdir() as tmp:
        log = tmp / "logs" / "trades.json"
        broker = PaperBroker(trades_path=log)
        assert len(broker.record([signal("Chicago")])) == 1
        assert sorted(os.listdir(log.parent)) == ["trades.json"]  # no temp file left behind
        before = log.read_text()
        real_replace, os.replace = os.replace, disk_full  # type: ignore[assignment]
        try:
            assert "No space" in refused(OSError, broker.record, [signal("Dallas")])
        finally:
            os.replace = real_replace
        assert log.read_text() == before and len(json.loads(before)["trades"]) == 1
        assert sorted(os.listdir(log.parent)) == ["trades.json"]  # and no temp file after a failure
        os.chmod(log, 0o640)
        broker.record([signal("Boston")])
        assert stat.S_IMODE(log.stat().st_mode) == 0o640  # the file keeps the mode it had
        assert len(json.loads(log.read_text())["trades"]) == 2
        assert log.read_text().startswith('{\n  "trades": [')  # the same shape as before


def refuse_rename(*args: object, **kwargs: object) -> None:
    raise PermissionError(13, "Access is denied (another process has the file open)")


@check
def a_refused_rename_falls_back_to_writing_the_trade_log_in_place_with_a_warning() -> None:
    with workdir() as tmp:
        log = tmp / "trades.json"
        broker = PaperBroker(trades_path=log)
        broker.record([signal("Chicago")])
        real_replace, os.replace = os.replace, refuse_rename  # type: ignore[assignment]
        try:
            with logged_warnings() as said:
                assert len(broker.record([signal("Dallas")])) == 1  # it did not raise
        finally:
            os.replace = real_replace
        assert len(json.loads(log.read_text())["trades"]) == 2
        assert sorted(os.listdir(tmp)) == ["trades.json"]  # no temp file left
        assert len(said) == 1 and "in place" in said[0], said


@check
def a_write_in_place_that_fails_too_keeps_the_new_log_in_a_temp_file_the_error_names() -> None:
    """The rename is refused and so is the write in place, after it truncated the log (a full
    disk): the complete new log is in the temporary file and the error says where."""
    with workdir() as tmp:
        log = tmp / "trades.json"
        broker = PaperBroker(trades_path=log)
        broker.record([signal("Chicago")])

        def full(self: pathlib.Path, *args: object, **kwargs: object) -> int:
            self.write_bytes(b"")  # what open(..., "w") does before the first byte
            raise OSError(28, "No space left on device")

        real_replace, real_write_text = os.replace, pathlib.Path.write_text
        os.replace = refuse_rename  # type: ignore[assignment]
        pathlib.Path.write_text = full  # type: ignore[method-assign,assignment]
        try:
            message = refused(OSError, broker.record, [signal("Dallas")])
        finally:
            os.replace, pathlib.Path.write_text = real_replace, real_write_text  # type: ignore[assignment]
        kept = [p for p in tmp.iterdir() if p.name != "trades.json"]
        assert len(kept) == 1 and kept[0].name in message, (kept, message)
        assert len(json.loads(kept[0].read_text())["trades"]) == 2  # the whole new log


class Truth(MarketPlugin):
    family = "fake"

    def discover_markets(self, **kw: Any) -> list[MarketSpec]:
        return []

    def fetch_features(self, spec: MarketSpec, horizon: str, **kw: Any) -> dict[str, float]:
        return {}

    def fetch_truth(self, spec: MarketSpec, **kw: Any) -> str | None:
        return "YES"

    def build_training_row(self, spec: MarketSpec, horizon: str, **kw: Any) -> dict | None:
        return None

    def is_truth_final(self, spec: MarketSpec, **kw: Any) -> bool:
        return True


def open_trade(target_date: str) -> dict[str, Any]:
    return {
        "recorded_at": "2024-01-10T10:00:00+00:00", "city_or_segment": "Chicago",
        "target_date": target_date, "outcome_label": "YES", "direction": "yes",
        "gamma_price": 0.6, "edge_after_fee": 0.12, "horizon": "morning_of",
        "flat_stake": 1.0, "size": 1.666667, "fee_paid": 0.003, "outcome": None,
        "realized_pnl": None,
    }  # fmt: skip


def market(target_date: str) -> MarketSpec:
    return MarketSpec(
        market_id=f"m-{target_date}", slug="s", question="q",
        outcome_bins=[OutcomeBin(label="YES"), OutcomeBin(label="NO")],
        close_time=f"{target_date}T20:00:00Z", market_family="binary",
        metadata={"city": "Chicago", "target_date": target_date},
    )  # fmt: skip


@check
def settle_all_takes_today_from_the_utc_date_of_now_utc() -> None:
    specs = [market("2024-01-16"), market("2024-01-17")]
    # 23:30 on the 15th in New York is 04:30 on the 16th in UTC: the 16th is due, the 17th is not.
    new_york = timezone(timedelta(hours=-5))
    now = datetime(2024, 1, 15, 23, 30, tzinfo=new_york)

    def settle(**kwargs: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        with workdir() as tmp:
            log = tmp / "trades.json"
            log.write_text(
                json.dumps({"trades": [open_trade("2024-01-16"), open_trade("2024-01-17")]})
            )
            summary = SettlementEngine(plugin=Truth(), trades_path=log).settle_all(specs, **kwargs)
            assert sorted(os.listdir(tmp)) == ["trades.json"]
            return summary, json.loads(log.read_text())["trades"]

    summary, trades = settle(now_utc=now)
    assert (summary["settled"], summary["pending"]) == (1, 1), (
        summary
    )  # the machine's date is not it
    assert trades[0]["outcome"] == "won" and trades[1]["outcome"] is None
    assert trades[0]["settled_at"] == "2024-01-16T04:30:00+00:00", trades[0].get("settled_at")
    summary, _ = settle(now_utc=now, today_str="2024-01-17")  # an explicit date still wins
    assert (summary["settled"], summary["pending"]) == (2, 0), summary
    message = refused(ValueError, settle, now_utc=datetime(2024, 1, 15, 23, 30))
    assert "timezone-aware" in message, message


# ── 9. compare_metrics: what is not a number is judged too ──────────────────────────────────────
def one(recorded: object, reproduced: object) -> Any:
    (comparison,) = compare_metrics({"m": recorded}, {"m": reproduced})
    return comparison


@check
def compare_metrics_judges_metrics_that_are_not_numbers() -> None:
    same = one("hist", "hist")
    assert (
        same.ok and same.abs_diff is None and (same.recorded, same.reproduced) == ("hist", "hist")
    )
    assert not one("lightgbm", "sklearn").ok  # a changed string used to read as reproduced
    assert not one("GO", "go").ok and not one("a", "a ").ok
    for recorded, reproduced in [(1.5, "1.5"), ("1.5", 1.5), (3, "3"), ("x", 3)]:
        assert not one(recorded, reproduced).ok, (recorded, reproduced)
    assert not one(True, False).ok and one(True, True).ok
    assert not one("text", None).ok and [r.ok for r in compare_metrics({"m": "text"}, {})] == [
        False
    ]
    assert one({"a": [1, 2], "b": "x"}, {"a": [1, 2], "b": "x"}).ok
    assert not one({"a": {"b": [1, 2, 3]}}, {"a": {"b": [1, 2, 4]}}).ok
    assert not one({"a": 1}, {"a": 1, "b": 2}).ok and not one([1, 2], [1, 2, 3]).ok
    assert one({"a": [1.0, 2.0]}, {"a": [1.0 + 1e-12, 2.0]}).ok  # numbers inside use the tolerance
    assert not one({"a": [1, 2]}, {"a": [1.5, 2]}).ok
    assert not one({"a": math.nan}, {"a": math.nan}).ok


# ── 10. a backtest that takes no trades exits 0 and writes nothing ──────────────────────────────
@check
def a_backtest_with_no_trades_exits_0_and_writes_no_output_file() -> None:
    sample = load_sample()
    resolver = sample.plugin()
    rows = [
        resolver.build_training_rows(sp, "previous_evening", decision_time=f"{day}T12:00:00+00:00")
        for day in sample.days[:60]
        for sp in sample.plugin(day).discover_markets()
    ]
    with workdir() as tmp:
        build_panel(rows).reset_index(drop=True).to_parquet(tmp / "panel.parquet")
        common = ["backtest", "--plugin", "weather_tmax", "--panel", tmp / "panel.parquet",
                  "--stride", "10", "--min-train-rows", "400"]  # fmt: skip
        none = cli(*common, "--min-edge", "0.99", "--output", tmp / "none.json", cwd=tmp)
        said = " ".join((none.stdout + none.stderr).split())
        assert none.returncode == 0, (none.returncode, said)
        assert "No trades produced" in said and "none.json was not written" in said, said
        assert not (tmp / "none.json").exists() and "Traceback" not in said
        trades = cli(*common, "--min-edge", "0.05", "--output", tmp / "some.json", cwd=tmp)
        assert trades.returncode == 0 and (tmp / "some.json").is_file(), (
            trades.stdout + trades.stderr
        )


# ── 7. GammaClient / AsyncGammaClient paging, and the plugin's skip of an invalid market ────────
OFFSET_TOO_LARGE = '{"error": "offset too large, use /markets/keyset for deeper pagination"}'
Reply = tuple[int, dict[str, str], str]  # status, headers, body


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


class GammaStub:
    """Gamma's ``/markets`` on 127.0.0.1 (stdlib ``http.server``, no network): ``total`` markets
    served by ``limit`` and ``offset`` (or the first page for ever, with ``ignore_offset``), and
    scripted replies (a 429, a 500, the 422) per offset. It does not know ``/markets/keyset`` (404),
    like a Gamma that predates it, so these checks run the offset fallback of 1.2.2 (#83)."""

    def __init__(
        self,
        total: int,
        script: dict[int, list[Reply]] | None = None,
        ignore_offset: bool = False,
    ) -> None:
        self.total = total
        self.ignore_offset = ignore_offset  # an API that serves the first page whatever the offset
        self.script = {offset: list(replies) for offset, replies in (script or {}).items()}
        self.requests: list[int] = []
        stub = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 (the name http.server looks for)
                url = urllib.parse.urlsplit(self.path)
                query = urllib.parse.parse_qs(url.query)
                if url.path == "/markets/keyset":  # an old Gamma: paging falls back to offsets
                    self._reply(404, {}, '{"error": "not found"}')
                    return
                if url.path != "/markets" or "offset" not in query or "limit" not in query:
                    self._reply(400, {}, '{"error": "unexpected request"}')
                    return
                offset, limit = int(query["offset"][0]), int(query["limit"][0])
                stub.requests.append(offset)
                scripted = stub.script.get(offset)
                if scripted:
                    self._reply(*scripted.pop(0))
                    return
                if stub.ignore_offset and len(stub.requests) > 1100:
                    # past the cap: a client that has none (main's `while True`) ends here with
                    # the short page and the check fails, instead of paging for ever
                    self._reply(200, {}, "[]")
                    return
                start = 0 if stub.ignore_offset else offset
                page = [tmax_market(i) for i in range(start, min(start + limit, stub.total))]
                self._reply(200, {}, json.dumps(page))

            def _reply(self, status: int, headers: dict[str, str], body: str) -> None:
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

    def __enter__(self) -> GammaStub:
        # a proxy set in the environment must not stand between the client and 127.0.0.1
        os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()


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


def paging_script() -> dict[int, list[Reply]]:
    """Page 2: a 429 and page 4: a 500, both saying retry now (so the audit does not sleep);
    page 6: Gamma's end of the offsets."""
    return {
        2: [(429, {"Retry-After": "0"}, '{"error": "slow down"}')],
        4: [(500, {"Retry-After": "0"}, '{"error": "boom"}')],
        6: [(422, {}, OFFSET_TOO_LARGE)],
    }


def assert_paged_through_the_failures(found: list[Any], stub: GammaStub, said: list[str]) -> None:
    assert [m.market_id for m in found] == [str(i) for i in range(6)], found
    # each failed page was asked again and not skipped, and the 422 was not retried
    assert stub.requests == [0, 2, 2, 4, 4, 6], stub.requests
    assert sum("retrying" in m for m in said) == 2, said  # the 429 and the 500
    text = " | ".join(said)
    assert "offset=6" in text and "incomplete" in text and "/markets/keyset" in text, said


@check
def gamma_client_retries_a_failed_page_and_keeps_what_it_read_at_the_offset_limit() -> None:
    with (
        GammaStub(total=10**6, script=paging_script()) as stub,
        logged_warnings() as said,
        GammaClient(base_url=stub.url) as client,
    ):
        found = client.discover_tmax_markets(page_size=2)
    assert_paged_through_the_failures(found, stub, said)


@check
def async_gamma_client_retries_a_failed_page_and_keeps_what_it_read_at_the_offset_limit() -> None:
    async def run(url: str) -> list[Any]:
        async with AsyncGammaClient(base_url=url) as client:
            return await client.discover_tmax_markets(page_size=2)

    with GammaStub(total=10**6, script=paging_script()) as stub, logged_warnings() as said:
        found = asyncio.run(run(stub.url))
    assert_paged_through_the_failures(found, stub, said)


@check
def gamma_client_fetch_markets_all_stops_at_the_offset_limit_and_no_other_422() -> None:
    script: dict[int, list[Reply]] = {6: [(422, {}, OFFSET_TOO_LARGE)]}
    with (
        GammaStub(total=10**6, script=script) as stub,
        logged_warnings() as said,
        GammaClient(base_url=stub.url) as client,
    ):
        markets = client.fetch_markets_all(page_size=2, max_results=1000)
    assert [m["id"] for m in markets] == [str(i) for i in range(6)], markets
    assert stub.requests == [0, 2, 4, 6] and "incomplete" in " ".join(said), (stub.requests, said)
    # a 422 on the first page, or one that does not say "offset too large" (an unrelated one, or
    # one that merely mentions the offset), is still an error
    for offset, body in (
        (0, OFFSET_TOO_LARGE),
        (2, '{"error": "limit must be at most 100"}'),
        (2, '{"error": "offset must be an integer"}'),
        (2, '{"detail": [{"loc": ["query", "offset"], "msg": "not a valid integer"}]}'),
    ):
        with (
            GammaStub(total=10**6, script={offset: [(422, {}, body)]}) as stub,
            GammaClient(base_url=stub.url) as client,
        ):
            try:
                client.fetch_markets_all(page_size=2)
            except httpx.HTTPStatusError as error:
                assert error.response.status_code == 422
            else:
                raise AssertionError(f"a 422 at offset {offset} was swallowed")


# the end of the offsets again, without the words "offset too large"
KEYSET_ONLY = '{"error": "use /markets/keyset"}'


@check
def gamma_client_discover_ends_the_paging_on_a_422_that_only_names_the_keyset_endpoint() -> None:
    with (
        GammaStub(total=10**6, script={2: [(422, {}, KEYSET_ONLY)]}) as stub,
        logged_warnings() as said,
        GammaClient(base_url=stub.url) as client,
    ):
        found = client.discover_tmax_markets(page_size=2)
    assert [m.market_id for m in found] == ["0", "1"], found  # the page before it is kept
    assert stub.requests == [0, 2], stub.requests  # and the 422 was not retried
    text = " | ".join(said)
    assert "offset=2" in text and "incomplete" in text and "/markets/keyset" in text, said


@check
def gamma_client_discover_still_raises_a_422_that_only_mentions_the_offset() -> None:
    body = '{"error": "offset must be an integer"}'
    with (
        GammaStub(total=10**6, script={2: [(422, {}, body)]}) as stub,
        GammaClient(base_url=stub.url) as client,
    ):
        message = refused(httpx.HTTPStatusError, client.discover_tmax_markets, page_size=2)
    assert "422" in message, message
    assert stub.requests == [0, 2], stub.requests  # page 2 was asked once and the error raised


def run_async(make: Callable[[str], Any], url: str) -> Any:
    """``asyncio.run`` of ``await make(client)`` for an ``AsyncGammaClient`` on *url*."""

    async def go() -> Any:
        async with AsyncGammaClient(base_url=url) as client:
            return await make(client)

    return asyncio.run(go())


@check
def gamma_client_stops_with_a_runtime_error_after_a_thousand_full_pages() -> None:
    """An API that ignores ``offset`` returns a full page for ever; the loop used to be
    ``while True``. One market per page (``page_size=1``) keeps each reply tiny."""
    with GammaStub(total=10**6, ignore_offset=True) as stub, GammaClient(base_url=stub.url) as c:
        message = refused(RuntimeError, c.discover_tmax_markets, page_size=1)
    assert "1000 pages" in message, message
    assert len(stub.requests) == 1000, len(stub.requests)


@check
def gamma_client_gives_up_after_five_tries_on_a_page_that_keeps_failing() -> None:
    bad_gateway: Reply = (502, {"Retry-After": "0"}, '{"error": "bad gateway"}')
    for use_async in (False, True):
        with (
            GammaStub(total=10**6, script={0: [bad_gateway] * 10}) as stub,
            logged_warnings() as said,
        ):
            if use_async:
                error = refused(
                    httpx.HTTPStatusError,
                    run_async,
                    lambda c: c.discover_tmax_markets(page_size=2),
                    stub.url,
                )
            else:
                with GammaClient(base_url=stub.url) as client:
                    error = refused(
                        httpx.HTTPStatusError, client.discover_tmax_markets, page_size=2
                    )
        assert "502" in error, error
        assert stub.requests == [0] * 5, (use_async, stub.requests)  # five tries, not six
        assert sum("retrying" in m for m in said) == 4, said


@check
def gamma_client_refuses_a_reply_that_is_not_a_page_and_raises_a_4xx_at_once() -> None:
    for use_async in (False, True):
        for reply, error_type, needle in (
            ((200, {}, '{"error": "not a list"}'), ValueError, "expected a list"),
            ((404, {}, '{"error": "not found"}'), httpx.HTTPStatusError, "404"),
        ):
            with GammaStub(total=10**6, script={0: [reply] * 10}) as stub:
                if use_async:
                    message = refused(
                        error_type,
                        run_async,
                        lambda c: c.discover_tmax_markets(page_size=2),
                        stub.url,
                    )
                else:
                    with GammaClient(base_url=stub.url) as client:
                        message = refused(error_type, client.discover_tmax_markets, page_size=2)
            assert needle in message, (use_async, reply, message)
            assert stub.requests == [0], (use_async, reply, stub.requests)  # not retried


@check
def gamma_client_raises_a_refused_connection_at_once_instead_of_retrying_it() -> None:
    """Pins what ``main`` did and a retry of every error would have changed: with nothing
    listening (offline, a wrong ``base_url``) the first ``ConnectError`` is the answer, not the
    last of four waits of 1 + 2 + 4 + 8 seconds."""
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"
    with socket.socket() as probe:  # a free port, released again: nothing listens there
        probe.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{probe.getsockname()[1]}"
    for use_async in (False, True):
        started = time.monotonic()
        with logged_warnings() as said:
            if use_async:
                refused(
                    httpx.ConnectError,
                    run_async,
                    lambda c: c.discover_tmax_markets(page_size=2),
                    url,
                )
            else:
                with GammaClient(base_url=url) as client:
                    refused(httpx.ConnectError, client.discover_tmax_markets, page_size=2)
        assert time.monotonic() - started < 5.0, "it waited: the retries would take 15 s"
        assert said == [], said


class StubGamma:
    """A client for ``WeatherTmaxPlugin``: ``fetch_markets_all`` returns what it was given."""

    def __init__(self, raw: list[dict[str, Any]]) -> None:
        self.raw = raw

    def fetch_markets_all(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self.raw

    def fetch_markets(self, **kwargs: Any) -> list[dict[str, Any]]:  # what the F1 plugin calls
        return self.raw


@check
def the_weather_plugin_skips_a_market_that_is_not_a_valid_spec_with_a_warning() -> None:
    good = {
        "id": "good",
        "question": "Highest temperature in Paris on May 12?",
        "slug": "tmax-paris",
        "endDate": "2026-05-12T20:00:00Z",
        "outcomes": ["30°C", "31°C"],
    }
    repeated = {**good, "id": "bad-bins", "outcomes": ["30°C", "30°C"]}
    no_date = {**good, "id": "bad-date", "endDate": "next tuesday"}
    sparse = {"id": "sparse", "question": "Highest temperature in Oslo on May 12?"}
    with logged_warnings() as said:
        specs = WeatherTmaxPlugin(
            gamma_client=StubGamma([repeated, good, no_date])
        ).discover_markets()
    assert [s.market_id for s in specs] == ["good"], specs  # the others were not lost
    text = " | ".join(said)
    assert "bad-bins" in text and "bad-date" in text and "skipped" in text, said
    assert len(said) == 2, said  # one warning per skipped market, nothing for the good one
    with logged_warnings() as said:  # a sparse payload is still a spec, and says nothing
        specs = WeatherTmaxPlugin(gamma_client=StubGamma([sparse, good])).discover_markets()
    assert [s.market_id for s in specs] == ["sparse", "good"] and said == [], (specs, said)


@check
def the_f1_plugin_skips_a_market_whose_tokens_have_no_names_with_a_warning() -> None:
    nameless = {
        "id": "nameless",
        "question": "F1 nameless GP winner?",
        "tokens": [{"price": 0.5}, {"price": 0.5}],  # both would be the bin "?"
    }
    good = {
        "id": "ok",
        "question": "F1 Monaco GP winner 2026?",
        "endDate": "2026-05-25T16:00:00Z",
        "tokens": [{"outcome": "Norris"}, {"outcome": "Hamilton"}],
    }
    with logged_warnings() as said:
        specs = SportsF1Plugin(gamma_client=StubGamma([nameless, good])).discover_markets()
    assert [s.market_id for s in specs] == ["ok"], specs  # the good one was not lost
    assert len(said) == 1 and "nameless" in said[0] and "skipped" in said[0], said


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
        print(f"audit sharp edges: {len(failures)} of {len(CHECKS)} checks FAILED: {failures}")
        return 1
    print(f"audit sharp edges: OK ({len(CHECKS)} checks, pmlab {pmlab.__version__})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
