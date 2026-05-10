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
