"""Tests for CLI commands."""

from __future__ import annotations

from typer.testing import CliRunner

from pmlab.cli.main import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "pmlab" in result.output


def test_status_no_champion(tmp_path):
    result = runner.invoke(app, ["status", "--artifacts-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "champion" in result.output.lower() or "No champion" in result.output


def test_backtest_missing_panel(tmp_path):
    result = runner.invoke(
        app,
        [
            "backtest",
            "--plugin",
            "test",
            "--panel",
            str(tmp_path / "missing.parquet"),
        ],
    )
    assert result.exit_code == 1


def test_backtest_low_stride():
    result = runner.invoke(
        app,
        [
            "backtest",
            "--plugin",
            "test",
            "--panel",
            "panel.parquet",
            "--stride",
            "3",
        ],
    )
    assert result.exit_code == 1
    assert "10" in result.output


def test_promote_champion_missing_files(tmp_path):
    result = runner.invoke(
        app,
        [
            "promote-champion",
            str(tmp_path / "model.pkl"),
            "--gate-path",
            str(tmp_path / "gate.json"),
            "--plugin",
            "test",
        ],
    )
    assert result.exit_code == 1
