"""Tests for modeling.champion."""

import json

import numpy as np
import pandas as pd
import pytest

from pmlab.backtest.holdout_gate import HoldoutGateResult
from pmlab.modeling.champion import ChampionManifest
from pmlab.modeling.lgbm_baseline import LGBMForecaster


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
    return HoldoutGateResult.evaluate(
        trades, required_segments=["A"], min_trades_per_segment=40
    )


def _make_fitted_lgbm() -> LGBMForecaster:
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"f1": rng.random(50), "f2": rng.random(50)})
    y = pd.Series((X["f1"] > 0.5).astype(int))
    model = LGBMForecaster(objective="binary")
    model.fit(X, y)
    return model


def test_cannot_publish_nogo(tmp_path):
    gate = _make_nogo_gate()
    model = _make_fitted_lgbm()
    with pytest.raises(ValueError, match="NO_GO"):
        ChampionManifest.publish(
            model=model,
            gate=gate,
            output_dir=tmp_path / "out",
            plugin_family="test",
        )


def test_publish_go_creates_json(tmp_path):
    gate = _make_go_gate()
    model = _make_fitted_lgbm()
    ChampionManifest.publish(
        model=model,
        gate=gate,
        output_dir=tmp_path / "out",
        plugin_family="test_family",
    )

    json_path = tmp_path / "out" / "champion.json"
    assert json_path.exists()
    with open(json_path) as f:
        data = json.load(f)
    assert data["model_name"] == "champion"
    assert data["plugin_family"] == "test_family"


def test_json_has_go(tmp_path):
    gate = _make_go_gate()
    model = _make_fitted_lgbm()
    ChampionManifest.publish(
        model=model,
        gate=gate,
        output_dir=tmp_path / "out",
        plugin_family="test",
    )
    json_path = tmp_path / "out" / "champion.json"
    with open(json_path) as f:
        data = json.load(f)
    assert data["publish_gate"]["decision"] == "GO"


def test_get_allowed_segments(tmp_path):
    gate = _make_go_gate()
    model = _make_fitted_lgbm()
    manifest = ChampionManifest.publish(
        model=model,
        gate=gate,
        output_dir=tmp_path / "out",
        plugin_family="test",
    )
    allowed = manifest.get_allowed_segments()
    assert "A" in allowed
    assert "B" in allowed


def test_load_roundtrip(tmp_path):
    gate = _make_go_gate()
    model = _make_fitted_lgbm()
    manifest = ChampionManifest.publish(
        model=model,
        gate=gate,
        output_dir=tmp_path / "out",
        plugin_family="test_family",
        model_name="my_model",
    )

    json_path = tmp_path / "out" / "champion.json"
    loaded = ChampionManifest.load(json_path)

    assert loaded.model_name == manifest.model_name
    assert loaded.plugin_family == manifest.plugin_family
    assert loaded.gate.decision == "GO"
    assert loaded.get_allowed_segments() == manifest.get_allowed_segments()
    assert loaded.published_at == manifest.published_at
