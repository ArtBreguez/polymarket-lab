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


def test_scan_markets_unknown_plugin_errors():
    """An unknown --plugin exits 1 and lists available plugins (no raw dump)."""
    result = runner.invoke(app, ["scan-markets", "--plugin", "banana"])
    assert result.exit_code == 1
    assert "banana" in result.output
    assert "weather_tmax" in result.output  # available list shown


def test_scan_markets_delegates_to_plugin(monkeypatch):
    """Regression: scan-markets must route through the plugin's discover_markets
    (family-filtered), not dump the raw Gamma feed. Guards the bug where a
    weather scan showed unrelated political markets.
    """
    from pmlab.core.market_spec import MarketSpec, OutcomeBin
    from pmlab.plugins import registry as registry_mod

    spec = MarketSpec(
        market_id="tmax-1",
        slug="tmax-nyc",
        question="Highest temperature in NYC on May 10?",
        outcome_bins=[OutcomeBin(label="30°C")],
        close_time="2026-05-10T20:00:00Z",
        market_family="range",
        tags=["weather"],
    )

    class _FakePlugin:
        def __init__(self) -> None:
            self.called_with: dict | None = None

        def discover_markets(self, **kwargs):
            self.called_with = kwargs
            return [spec]

    fake = _FakePlugin()
    monkeypatch.setattr(registry_mod, "build_plugin", lambda family, **kw: fake)

    result = runner.invoke(app, ["scan-markets", "--plugin", "weather_tmax", "--limit", "50"])
    assert result.exit_code == 0
    assert "Highest temperature in NYC" in result.output
    assert fake.called_with is not None
    assert fake.called_with.get("limit") == 50
