"""Smoke tests — package importable and version correct."""

from __future__ import annotations


def test_import() -> None:
    import pmlab

    assert pmlab.__version__ == "0.1.0"


def test_cli_version() -> None:
    from typer.testing import CliRunner

    from pmlab.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0, result.output
    assert "0.1.0" in result.output


def test_cli_help() -> None:
    from typer.testing import CliRunner

    from pmlab.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_public_api_exports():
    import pmlab
    assert hasattr(pmlab, 'MarketSpec')
    assert hasattr(pmlab, 'MarketPlugin')
    assert hasattr(pmlab, 'ChampionManifest')
    assert hasattr(pmlab, 'PaperBroker')
    assert hasattr(pmlab, 'SettlementEngine')
    assert hasattr(pmlab, '__all__')
    # All __all__ items must be importable
    for name in pmlab.__all__:
        assert hasattr(pmlab, name), f'{name} in __all__ but not importable'


def test_cli_scan_markets():
    from typer.testing import CliRunner

    from pmlab.cli.main import app
    runner = CliRunner()
    result = runner.invoke(app, ["scan-markets", "--plugin", "weather_tmax"])
    assert result.exit_code == 0
    assert "scan-markets" in result.output


def test_cli_record_trades():
    from typer.testing import CliRunner

    from pmlab.cli.main import app
    runner = CliRunner()
    result = runner.invoke(app, ["record-trades", "--plugin", "weather_tmax"])
    assert result.exit_code == 0


def test_cli_settle_trades():
    from typer.testing import CliRunner

    from pmlab.cli.main import app
    runner = CliRunner()
    result = runner.invoke(app, ["settle-trades", "--plugin", "weather_tmax"])
    assert result.exit_code == 0


def test_cli_backtest_stride_guard():
    from typer.testing import CliRunner

    from pmlab.cli.main import app
    runner = CliRunner()
    result = runner.invoke(app, ["backtest", "--plugin", "weather_tmax", "--stride", "1"])
    assert result.exit_code != 0  # stride too low


def test_cli_backtest_valid():
    from typer.testing import CliRunner

    from pmlab.cli.main import app
    runner = CliRunner()
    result = runner.invoke(app, ["backtest", "--plugin", "weather_tmax", "--stride", "30"])
    assert result.exit_code == 0


def test_cli_promote_champion():
    from typer.testing import CliRunner

    from pmlab.cli.main import app
    runner = CliRunner()
    result = runner.invoke(app, ["promote-champion", "/tmp/model.pkl", "--gate-path", "/tmp/gate.json", "--plugin", "weather_tmax"])
    assert result.exit_code == 0
