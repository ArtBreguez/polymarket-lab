"""The record composes with the tracker, the champion manifest and the registry.

Backward compatibility is the point of half of these: every file written before 0.10.0 has no
``reproducibility`` key and must keep loading, as "no record", never as an error.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pmlab.backtest.holdout_gate import HoldoutGateResult
from pmlab.modeling.champion import ChampionManifest
from pmlab.modeling.lgbm_baseline import LGBMForecaster
from pmlab.modeling.registry import ModelRegistry
from pmlab.reproducibility import ReproducibilityRecord, fingerprint_panel
from pmlab.tracking import LocalJSONTracker
from tests.reproducibility.helpers import make_panel


def _record() -> ReproducibilityRecord:
    return ReproducibilityRecord.capture(
        kind="backtest",
        seeds={"python_random": 4, "numpy": 4, "lightgbm": 4},
        config={"plugin": "demo", "stride": 10},
        data=fingerprint_panel(make_panel(n_dates=5), source="p.parquet"),
        metrics={"total_trades": 3, "total_pnl": 0.5},
    )


def _gate() -> HoldoutGateResult:
    trades = pd.DataFrame([{"realized_pnl": 0.1, "outcome": "won", "segment": "A"}] * 50)
    return HoldoutGateResult.evaluate(trades, required_segments=["A"], min_trades_per_segment=40)


def _model() -> LGBMForecaster:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f1": rng.random(50), "f2": rng.random(50)})
    m = LGBMForecaster(n_estimators=3, min_child_samples=2)
    m.fit(X, pd.Series((X["f1"] > 0.5).astype(int)))
    return m


# ── tracker ──────────────────────────────────────────────────────────────
def test_a_tracked_run_stores_the_record_next_to_params_and_metrics(tmp_path):
    tracker = LocalJSONTracker(tmp_path / "runs")
    rec = _record()
    with tracker.run("demo", "r1") as run:
        run.log_params({"stride": 10})
        run.log_metrics({"total_pnl": 0.5})
        run.log_reproducibility(rec)
    stored = tracker.get_run("demo", run.run_id)
    assert stored is not None
    assert ReproducibilityRecord.from_dict(stored.reproducibility) == rec
    assert stored.params == {"stride": 10} and stored.metrics == {"total_pnl": 0.5}
    on_disk = json.loads((tmp_path / "runs" / "demo" / f"{run.run_id}.json").read_text())
    assert on_disk["reproducibility"]["seeds"] == {"python_random": 4, "numpy": 4, "lightgbm": 4}


def test_a_run_file_written_before_the_record_existed_still_loads(tmp_path):
    exp = tmp_path / "runs" / "old"
    exp.mkdir(parents=True)
    (exp / "abc123.json").write_text(
        json.dumps(
            {
                "run_id": "abc123",
                "experiment": "old",
                "run_name": "legacy",
                "params": {"stride": 5},
                "metrics": {"total_pnl": 1.0},
                "gate": {},
                "status": "finished",
                "started_at": "2026-01-01T00:00:00+00:00",
                "ended_at": None,
                "error": None,
            }
        )
    )
    run = LocalJSONTracker(tmp_path / "runs").get_run("old", "abc123")
    assert run is not None and run.reproducibility == {}
    assert run.metrics == {"total_pnl": 1.0}


# ── champion manifest ────────────────────────────────────────────────────
def test_publish_writes_the_record_into_champion_json_and_load_returns_it(tmp_path):
    rec = _record()
    ChampionManifest.publish(
        _model(), _gate(), tmp_path / "out", plugin_family="demo", reproducibility=rec
    )
    raw = json.loads((tmp_path / "out" / "champion.json").read_text())
    assert raw["reproducibility"] == rec.to_dict()
    loaded = ChampionManifest.load(tmp_path / "out" / "champion.json")
    assert loaded.reproducibility == rec


def test_an_old_manifest_without_the_key_loads_with_no_record(tmp_path):
    out = tmp_path / "out"
    ChampionManifest.publish(_model(), _gate(), out, plugin_family="demo")
    path = out / "champion.json"
    legacy = json.loads(path.read_text())
    for key in ("reproducibility", "costs", "backtest_expectation"):  # a pre-0.7 manifest
        legacy.pop(key, None)
    path.write_text(json.dumps(legacy))
    loaded = ChampionManifest.load(path)
    assert loaded.reproducibility is None
    assert loaded.plugin_family == "demo"


def test_a_record_this_pmlab_cannot_read_never_takes_the_champion_down(tmp_path):
    out = tmp_path / "out"
    ChampionManifest.publish(
        _model(), _gate(), out, plugin_family="demo", reproducibility=_record()
    )
    path = out / "champion.json"
    raw = json.loads(path.read_text())
    raw["reproducibility"]["schema_version"] = 99  # written by some future pmlab
    path.write_text(json.dumps(raw))
    with pytest.warns(UserWarning, match="ignoring the reproducibility record"):
        loaded = ChampionManifest.load(path)
    assert loaded.reproducibility is None
    assert loaded.get_allowed_segments() == {"A"}  # the gate, the part trading relies on, is intact


def test_publish_without_a_record_writes_null_not_a_missing_key(tmp_path):
    ChampionManifest.publish(_model(), _gate(), tmp_path / "out", plugin_family="demo")
    raw = json.loads((tmp_path / "out" / "champion.json").read_text())
    assert raw["reproducibility"] is None


# ── registry ─────────────────────────────────────────────────────────────
def test_the_registry_archives_and_rolls_back_the_record(tmp_path):
    rec = _record()
    manifest = ChampionManifest.publish(
        _model(), _gate(), tmp_path / "live", plugin_family="demo", reproducibility=rec
    )
    registry = ModelRegistry(tmp_path / "registry")
    version = registry.record(manifest)
    assert registry.get(version).reproducibility == rec  # archived with the version

    restored = registry.rollback(version, tmp_path / "restored")
    assert restored.reproducibility == rec  # and restored with the champion
    assert ChampionManifest.load(tmp_path / "restored" / "champion.json").reproducibility == rec


def test_the_registry_keeps_a_version_without_a_record_loadable(tmp_path):
    manifest = ChampionManifest.publish(_model(), _gate(), tmp_path / "live", plugin_family="demo")
    registry = ModelRegistry(tmp_path / "registry")
    version = registry.record(manifest)
    assert registry.get(version).reproducibility is None
    assert registry.rollback(version, tmp_path / "r").reproducibility is None
