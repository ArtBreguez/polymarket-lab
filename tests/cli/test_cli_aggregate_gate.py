"""A champion promoted from an aggregate gate trades, and is watched, in every segment.

`HoldoutGateResult.evaluate(trades)` with no `required_segments` grades the whole trade log
as one segment called "all". The champion it publishes allowed `{"all"}`, while the broker
records signals under their real segment names (a city, a race), so `record-trades` used to
record nothing and the watchdog counted every settled trade as outside the champion's
segments. This drives the real commands end to end.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from pmlab.cli.main import app
from pmlab.core.market_spec import MarketSpec, OutcomeBin
from pmlab.execution.settlement import SettlementEngine
from tests.cli.test_cli_costs import _model, _record, _trades_file, _Truths, _write
from tests.cli.test_cli_policy import _backtest

runner = CliRunner()


def _aggregate_gate(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "gate.json",
        {
            "decision": "GO",
            "segment_results": [
                {
                    "segment": "all",
                    "num_trades": 100,
                    "total_pnl": 10.0,
                    "passes": True,
                    "reason": "ok",
                }
            ],
            "aggregate_pnl": 10.0,
            "aggregate_trades": 100,
        },
    )


def _per_segment_gate(tmp_path: Path, *segments: str) -> Path:
    return _write(
        tmp_path / "gate.json",
        {
            "decision": "GO",
            "segment_results": [
                {
                    "segment": segment,
                    "num_trades": 100,
                    "total_pnl": 10.0,
                    "passes": True,
                    "reason": "ok",
                }
                for segment in segments
            ],
            "aggregate_pnl": 10.0 * len(segments),
            "aggregate_trades": 100 * len(segments),
        },
    )


def _publish(tmp_path: Path, gate: Path | None = None) -> Path:
    artifacts = tmp_path / "art"
    _backtest(tmp_path, "--min-edge", "0.1")
    result = runner.invoke(
        app,
        [
            "promote-champion",
            str(_model(tmp_path)),
            "--gate-path",
            str(gate or _aggregate_gate(tmp_path)),
            "--plugin",
            "politics",
            "--output-dir",
            str(artifacts / "public_models"),
            "--expectation",
            str(tmp_path / "metrics.json"),
        ],  # fmt: skip
    )
    assert result.exit_code == 0, result.output
    return artifacts


def _signals(artifacts: Path, segments: tuple[str, ...] = ("Chicago", "Dallas")) -> None:
    signals = [
        {
            "market_id": f"{segment}-{i}",
            "city_or_segment": segment,
            "target_date": f"2099-01-{i + 1:02d}",
            "horizon": "morning_of",
            "outcome_label": "YES",
            "direction": "yes",
            "gamma_price": 0.40,
            "model_prob": 0.70,
            "best_edge": 0.30,
            "yes_edge": 0.30,
            "no_edge": -0.30,
            "question": "q",
        }
        for segment in segments
        for i in range(6)
    ]
    _write(
        artifacts / "workspaces" / "ws" / "signals" / "v2" / "scan_edge_signals.json",
        {"signals": signals},
    )


def _settle(path: Path, segments: tuple[str, ...]) -> None:
    specs = [
        MarketSpec(
            market_id=f"{segment}-{i}",
            slug="s",
            question="q",
            outcome_bins=[OutcomeBin(label="YES"), OutcomeBin(label="NO")],
            close_time="2099-12-31T00:00:00Z",
            market_family="binary",
            metadata={"city": segment, "target_date": f"2099-01-{i + 1:02d}"},
        )
        for segment in segments
        for i in range(6)
    ]
    SettlementEngine(plugin=_Truths(), trades_path=path).settle_all(
        specs, today_str="2099-12-31", now_utc=datetime(2099, 12, 31, 12, tzinfo=UTC)
    )


def test_the_champion_records_that_it_was_graded_in_aggregate(tmp_path: Path) -> None:
    artifacts = _publish(tmp_path)
    data = json.loads((artifacts / "public_models" / "champion.json").read_text())
    assert [r["segment"] for r in data["publish_gate"]["segment_results"]] == ["all"]


def test_record_trades_records_every_segment_and_says_why(tmp_path: Path) -> None:
    artifacts = _publish(tmp_path)
    _signals(artifacts)
    result = _record(artifacts, "ws", "--min-edge", "0.1")
    assert result.exit_code == 0, result.output
    assert "Recorded 12 new trades" in result.output
    assert "every segment" in result.output
    trades = json.loads(_trades_file(artifacts, "ws").read_text())["trades"]
    assert {t["city_or_segment"] for t in trades} == {"Chicago", "Dallas"}


def test_the_trades_are_counted_by_the_watchdog(tmp_path: Path) -> None:
    artifacts = _publish(tmp_path)
    _signals(artifacts)
    assert _record(artifacts, "ws", "--min-edge", "0.1").exit_code == 0
    _settle(_trades_file(artifacts, "ws"), ("Chicago", "Dallas"))

    out = tmp_path / "health.json"
    result = runner.invoke(
        app,
        [
            "monitor",
            "--artifacts-dir",
            str(artifacts),
            "--workspace",
            "ws",
            "--trades",
            str(_trades_file(artifacts, "ws")),
            "--skip",
            "drift",
            "--min-trades",
            "5",
            "--as-of",
            "2100-01-01T00:00:00Z",
            "--json",
            str(out),
        ],  # fmt: skip
    )
    assert result.exit_code in (0, 10, 20, 30), result.output
    report: dict[str, Any] = json.loads(out.read_text())
    pnl = next(c for c in report["checks"] if c["name"] == "pnl")["metrics"]
    assert pnl["n_out_of_segment"] == 0
    assert pnl["n_trades"] == 12
    assert report["champion"]["allowed_segments"] == ["all"]


def test_a_per_segment_champion_is_unchanged(tmp_path: Path) -> None:
    from tests.cli.test_cli_costs import _published_champion
    from tests.cli.test_cli_costs import _signals as per_segment_signals

    artifacts = _published_champion(tmp_path / "art", costs=None)  # gated on segment "A" only
    per_segment_signals(artifacts, "ws")
    assert _record(artifacts, "ws").exit_code == 0
    assert "every segment" not in _record(artifacts, "ws").output


@pytest.mark.parametrize("flag", ["--expectation"])
def test_promote_help_still_documents_the_expectation(flag: str) -> None:
    assert flag in runner.invoke(app, ["promote-champion", "--help"]).output


def test_a_gate_with_a_segment_named_all_records_only_the_segments_it_graded(
    tmp_path: Path,
) -> None:
    artifacts = _publish(tmp_path, _per_segment_gate(tmp_path, "Chicago", "all"))
    _signals(artifacts, segments=("Chicago", "Dallas", "all"))
    result = _record(artifacts, "ws", "--min-edge", "0.1")
    assert result.exit_code == 0, result.output
    assert "every segment" not in result.output
    trades = json.loads(_trades_file(artifacts, "ws").read_text())["trades"]
    assert {t["city_or_segment"] for t in trades} == {"Chicago", "all"}  # not Dallas
