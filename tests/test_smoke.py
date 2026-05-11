"""Smoke tests — package importable and version correct."""

from __future__ import annotations

import json
from pathlib import Path


def test_import() -> None:
    import pmlab

    assert pmlab.__version__ == "0.3.0"


def test_cli_version() -> None:
    from typer.testing import CliRunner
    from pmlab.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0, result.output
    assert "0.3.0" in result.output


def test_cli_help() -> None:
    from typer.testing import CliRunner
    from pmlab.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_public_api_exports():
    import pmlab
    assert hasattr(pmlab, "MarketSpec")
    assert hasattr(pmlab, "MarketPlugin")
    assert hasattr(pmlab, "ChampionManifest")
    assert hasattr(pmlab, "PaperBroker")
    assert hasattr(pmlab, "SettlementEngine")
    assert hasattr(pmlab, "__all__")
    for name in pmlab.__all__:
        assert hasattr(pmlab, name), f"{name} in __all__ but not importable"


def test_cli_scan_markets():
    """scan-markets hits real API or errors — just ensure it doesn't crash with exit 2 (arg error)."""
    from typer.testing import CliRunner
    from pmlab.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["scan-markets", "--plugin", "weather_tmax"])
    # Exit 0 (success) or 1 (API error) are both acceptable — exit 2 is argument error
    assert result.exit_code != 2, f"Argument error: {result.output}"


def test_cli_record_trades_no_champion(tmp_path):
    """record-trades exits 1 when no champion is published yet."""
    from typer.testing import CliRunner
    from pmlab.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, [
        "record-trades", "--plugin", "weather_tmax",
        "--artifacts-dir", str(tmp_path),
    ])
    assert result.exit_code == 1
    assert "champion" in result.output.lower()


def test_cli_settle_trades():
    """settle-trades always exits 0 (it just prints a guide message)."""
    from typer.testing import CliRunner
    from pmlab.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["settle-trades", "--plugin", "weather_tmax"])
    assert result.exit_code == 0, result.output


def test_cli_backtest_stride_guard():
    from typer.testing import CliRunner
    from pmlab.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, [
        "backtest", "--plugin", "weather_tmax",
        "--panel", "fake.parquet", "--stride", "1",
    ])
    assert result.exit_code != 0


def test_cli_backtest_missing_panel(tmp_path):
    """backtest exits 1 when panel file does not exist."""
    from typer.testing import CliRunner
    from pmlab.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, [
        "backtest", "--plugin", "weather_tmax",
        "--panel", str(tmp_path / "missing.parquet"),
        "--stride", "30",
    ])
    assert result.exit_code == 1


def test_cli_promote_champion_missing_model(tmp_path):
    """promote-champion exits 1 when model file does not exist."""
    from typer.testing import CliRunner
    from pmlab.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, [
        "promote-champion", str(tmp_path / "model.pkl"),
        "--gate-path", str(tmp_path / "gate.json"),
        "--plugin", "weather_tmax",
    ])
    assert result.exit_code == 1


def test_cli_promote_champion_no_go(tmp_path):
    """promote-champion exits 1 when gate is NO_GO."""
    from typer.testing import CliRunner
    from pmlab.cli.main import app
    import pickle
    from pmlab.modeling.lgbm_baseline import LGBMForecaster

    # Create a dummy model pkl
    model = LGBMForecaster()
    model_path = tmp_path / "model.pkl"
    import pandas as pd
    import numpy as np
    X = pd.DataFrame({"feature_x": [1.0, 2.0, 3.0]})
    y = pd.Series([0, 1, 0])
    model.fit(X, y)
    model.save(model_path)

    # Create a NO_GO gate
    gate = {
        "decision": "NO_GO",
        "aggregate_pnl": -5.0,
        "aggregate_trades": 10,
        "segment_results": [
            {"segment": "test", "num_trades": 10, "total_pnl": -5.0, "passes": False, "reason": "negative_pnl"}
        ],
    }
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(gate))

    runner = CliRunner()
    result = runner.invoke(app, [
        "promote-champion", str(model_path),
        "--gate-path", str(gate_path),
        "--plugin", "weather_tmax",
        "--output-dir", str(tmp_path / "out"),
    ])
    assert result.exit_code == 1
    assert "NO_GO" in result.output
