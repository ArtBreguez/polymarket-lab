"""Tests for modeling.registry — versioned champion registry.

The registry wraps ChampionManifest.publish so every promotion is archived
(list/diff/rollback) instead of silently overwriting champion.json. No-lookahead
is irrelevant here (pure artifact management), but the champion HARD GATE must
still hold: a NO_GO can never enter the registry.
"""

import numpy as np
import pandas as pd
import pytest

from pmlab.backtest.holdout_gate import HoldoutGateResult
from pmlab.modeling.champion import ChampionManifest
from pmlab.modeling.lgbm_baseline import LGBMForecaster
from pmlab.modeling.registry import ModelRegistry


def _make_go_gate() -> HoldoutGateResult:
    rows = []
    for _ in range(50):
        rows.append({"realized_pnl": 0.1, "outcome": "won", "segment": "A"})
    for _ in range(50):
        rows.append({"realized_pnl": 0.1, "outcome": "won", "segment": "B"})
    trades = pd.DataFrame(rows)
    return HoldoutGateResult.evaluate(
        trades, required_segments=["A", "B"], min_trades_per_segment=40
    )


def _make_nogo_gate() -> HoldoutGateResult:
    rows = [{"realized_pnl": -0.1, "outcome": "lost", "segment": "A"}]
    trades = pd.DataFrame(rows)
    return HoldoutGateResult.evaluate(trades, required_segments=["A"], min_trades_per_segment=40)


def _make_fitted_lgbm(seed: int = 42) -> LGBMForecaster:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"f1": rng.random(50), "f2": rng.random(50)})
    y = pd.Series((X["f1"] > 0.5).astype(int))
    model = LGBMForecaster(objective="binary")
    model.fit(X, y)
    return model


def _publish_and_record(registry: ModelRegistry, tmp_path, model_name: str, seed: int = 42) -> str:
    """Publish a champion then archive it in the registry; return version_id."""
    gate = _make_go_gate()
    model = _make_fitted_lgbm(seed)
    manifest = ChampionManifest.publish(
        model=model,
        gate=gate,
        output_dir=tmp_path / "out",
        plugin_family="test_family",
        model_name=model_name,
    )
    return registry.record(manifest)


def test_record_returns_version_and_archives(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    vid = _publish_and_record(registry, tmp_path, "champ_v1")
    assert isinstance(vid, str) and vid
    versions = registry.list()
    assert len(versions) == 1
    assert versions[0].version_id == vid
    assert versions[0].model_name == "champ_v1"
    assert versions[0].gate_decision == "GO"


def test_record_rejects_nogo(tmp_path):
    """The registry must never archive a NO_GO — the hard gate holds end to end.

    ChampionManifest.publish already refuses a NO_GO, so we build a manifest that
    claims GO but carries a NO_GO gate (a corrupted/bypassed manifest) and confirm
    the registry independently rejects it rather than trusting the caller.
    """
    registry = ModelRegistry(tmp_path / "registry")
    gate = _make_go_gate()
    model = _make_fitted_lgbm()
    manifest = ChampionManifest.publish(
        model=model,
        gate=gate,
        output_dir=tmp_path / "out",
        plugin_family="test",
    )
    # Tamper: swap in a NO_GO gate after a legitimate publish.
    manifest.gate = _make_nogo_gate()
    with pytest.raises(ValueError, match="NO_GO"):
        registry.record(manifest)


def test_list_is_newest_first(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    v1 = _publish_and_record(registry, tmp_path, "champ_v1", seed=1)
    v2 = _publish_and_record(registry, tmp_path, "champ_v2", seed=2)
    versions = registry.list()
    assert [v.version_id for v in versions] == [v2, v1]


def test_diff_reports_changes(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    v1 = _publish_and_record(registry, tmp_path, "champ_v1", seed=1)
    v2 = _publish_and_record(registry, tmp_path, "champ_v2", seed=2)
    diff = registry.diff(v1, v2)
    assert diff["model_name"] == ("champ_v1", "champ_v2")
    # Both are GO gates, so decision is unchanged (reported as a single value or equal pair).
    assert "gate_decision" in diff


def test_rollback_restores_previous_champion(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    out = tmp_path / "out"
    v1 = _publish_and_record(registry, tmp_path, "champ_v1", seed=1)
    _publish_and_record(registry, tmp_path, "champ_v2", seed=2)

    # champion.json currently reflects v2 (last publish). Roll back to v1.
    restored = registry.rollback(v1, output_dir=out)
    assert restored.model_name == "champ_v1"

    reloaded = ChampionManifest.load(out / "champion.json")
    assert reloaded.model_name == "champ_v1"
    # The restored model must be loadable and usable.
    model = reloaded.load_model()
    proba = model.predict_proba(pd.DataFrame({"f1": [1.0], "f2": [0.5]}))
    assert proba.shape[1] == 2


def test_rollback_unknown_version_raises(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    _publish_and_record(registry, tmp_path, "champ_v1")
    with pytest.raises(KeyError):
        registry.rollback("nonexistent-version", output_dir=tmp_path / "out")


def test_get_returns_manifest(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    vid = _publish_and_record(registry, tmp_path, "champ_v1")
    manifest = registry.get(vid)
    assert manifest.model_name == "champ_v1"
    assert manifest.gate.decision == "GO"


def test_registry_persists_across_instances(tmp_path):
    """A fresh ModelRegistry pointed at the same dir sees prior records."""
    reg1 = ModelRegistry(tmp_path / "registry")
    vid = _publish_and_record(reg1, tmp_path, "champ_v1")
    reg2 = ModelRegistry(tmp_path / "registry")
    assert [v.version_id for v in reg2.list()] == [vid]


def test_calibrator_archived_and_restored(tmp_path):
    """A champion WITH a calibrator round-trips through record + rollback.

    Covers the calibrator branch in both record() (archive calibrator.pkl) and
    rollback() (restore it as the live calibrator), so a rolled-back champion
    keeps its calibration instead of silently dropping it.
    """
    from pmlab.modeling.calibration import IsotonicCalibrator

    registry = ModelRegistry(tmp_path / "registry")
    out = tmp_path / "out"
    gate = _make_go_gate()
    model = _make_fitted_lgbm()
    calibrator = IsotonicCalibrator()
    calibrator.fit(np.array([0.3, 0.5, 0.7]), np.array([0, 1, 1]))

    manifest = ChampionManifest.publish(
        model=model,
        gate=gate,
        output_dir=out,
        plugin_family="test_family",
        model_name="calibrated_champ",
        calibrator=calibrator,
    )
    vid = registry.record(manifest)

    # The archived version carries its own calibrator.pkl.
    archived = registry.get(vid)
    assert archived.calibrator_path is not None
    assert archived.calibrator_path.exists()

    # Overwrite the live champion with an uncalibrated one, then roll back.
    _publish_and_record(registry, tmp_path, "plain_champ", seed=7)
    restored = registry.rollback(vid, output_dir=out)
    assert restored.calibrator_path is not None
    assert restored.calibrator_path.exists()
    assert restored.calibrator_path.name == "calibrator.pkl"
