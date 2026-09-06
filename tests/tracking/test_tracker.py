"""Tests for the experiment tracking layer (v0.8.0).

Contract:
  * A tracker records params, metrics, and the gate decision for a backtest run.
  * The local-JSON default persists one append-only record per run — no heavy dep.
  * Runs are listable and loadable point-in-time (reproducibility principle).
  * BacktestMetrics and HoldoutGateResult log directly (no manual glue).
  * The MLflow backend is optional and never imported by the core path.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from pmlab.backtest.holdout_gate import HoldoutGateResult
from pmlab.backtest.metrics import BacktestMetrics
from pmlab.tracking import (
    ExperimentTracker,
    LocalJSONTracker,
    RunRecord,
)


def _gate_go() -> HoldoutGateResult:
    trades = pd.DataFrame(
        {
            "outcome": ["won", "won", "lost", "won"],
            "realized_pnl": [1.0, 1.0, -1.0, 1.0],
            "segment": ["all", "all", "all", "all"],
        }
    )
    return HoldoutGateResult.evaluate(trades, required_segments=["all"], min_trades_per_segment=1)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_local_tracker_is_experiment_tracker():
    assert isinstance(LocalJSONTracker(root="."), ExperimentTracker)


# ---------------------------------------------------------------------------
# Core logging round-trip
# ---------------------------------------------------------------------------


def test_run_logs_params_metrics_and_persists(tmp_path):
    tracker = LocalJSONTracker(root=tmp_path)
    with tracker.run(experiment="weather_tmax", run_name="baseline") as run:
        run.log_params({"model": "lgbm", "stride": 10, "taker_bps": 30.0})
        run.log_metrics({"total_pnl": 2.5, "hit_rate": 0.75})

    # One JSON file persisted under the experiment dir.
    files = list((tmp_path / "weather_tmax").glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    assert data["experiment"] == "weather_tmax"
    assert data["run_name"] == "baseline"
    assert data["params"]["model"] == "lgbm"
    assert data["params"]["stride"] == 10
    assert data["metrics"]["total_pnl"] == 2.5
    assert data["status"] == "finished"
    assert data["run_id"]  # non-empty
    assert data["started_at"] and data["ended_at"]


def test_log_metrics_accepts_backtest_metrics_object(tmp_path):
    tracker = LocalJSONTracker(root=tmp_path)
    m = BacktestMetrics(
        num_trades=4, total_pnl=2.0, hit_rate=0.75, avg_pnl_per_trade=0.5, avg_edge=0.1
    )
    with tracker.run(experiment="exp", run_name="r") as run:
        run.log_metrics(m)  # dataclass logs directly

    rec = tracker.list_runs("exp")[0]
    assert rec.metrics["num_trades"] == 4
    assert rec.metrics["hit_rate"] == 0.75
    assert rec.metrics["avg_edge"] == 0.1


def test_log_gate_records_decision_and_segments(tmp_path):
    tracker = LocalJSONTracker(root=tmp_path)
    gate = _gate_go()
    with tracker.run(experiment="exp", run_name="r") as run:
        run.log_gate(gate)

    rec = tracker.list_runs("exp")[0]
    assert rec.gate["decision"] == "GO"
    # The gate's full dict is preserved for auditing.
    assert "segment_results" in rec.gate


# ---------------------------------------------------------------------------
# Listing / loading (reproducibility)
# ---------------------------------------------------------------------------


def test_list_runs_returns_all_records_sorted_by_start(tmp_path):
    tracker = LocalJSONTracker(root=tmp_path)
    for i in range(3):
        with tracker.run(experiment="exp", run_name=f"r{i}") as run:
            run.log_metrics({"total_pnl": float(i)})

    runs = tracker.list_runs("exp")
    assert len(runs) == 3
    assert all(isinstance(r, RunRecord) for r in runs)
    # newest first
    pnls = [r.metrics["total_pnl"] for r in runs]
    assert pnls == [2.0, 1.0, 0.0]


def test_list_runs_empty_for_unknown_experiment(tmp_path):
    tracker = LocalJSONTracker(root=tmp_path)
    assert tracker.list_runs("does_not_exist") == []


def test_get_run_by_id_round_trips(tmp_path):
    tracker = LocalJSONTracker(root=tmp_path)
    with tracker.run(experiment="exp", run_name="r") as run:
        run.log_params({"seed": 42})
        rid = run.run_id
    loaded = tracker.get_run("exp", rid)
    assert loaded is not None
    assert loaded.params["seed"] == 42
    assert loaded.run_id == rid


# ---------------------------------------------------------------------------
# Failure semantics
# ---------------------------------------------------------------------------


def test_run_records_failed_status_on_exception(tmp_path):
    tracker = LocalJSONTracker(root=tmp_path)
    with pytest.raises(RuntimeError), tracker.run(experiment="exp", run_name="boom") as run:
        run.log_params({"x": 1})
        raise RuntimeError("kaboom")

    rec = tracker.list_runs("exp")[0]
    assert rec.status == "failed"
    assert "kaboom" in (rec.error or "")
    # params logged before the failure are still persisted
    assert rec.params["x"] == 1


def test_append_only_runs_never_overwrite(tmp_path):
    tracker = LocalJSONTracker(root=tmp_path)
    with tracker.run(experiment="exp", run_name="same") as run:
        run.log_metrics({"total_pnl": 1.0})
    with tracker.run(experiment="exp", run_name="same") as run:
        run.log_metrics({"total_pnl": 2.0})
    # Same run_name, but two distinct persisted records (unique run_id).
    files = list((tmp_path / "exp").glob("*.json"))
    assert len(files) == 2
