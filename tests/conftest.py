"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _plain_cli_output(monkeypatch):
    """Keep rich/typer output free of ANSI codes, whatever the CI terminal says.

    GitHub Actions sets variables that make rich colour CliRunner output, which
    splits '--policy' into escape sequences and fails substring assertions that
    pass locally.
    """
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.setenv("TERM", "dumb")


@pytest.fixture
def tmp_workspace(tmp_path):
    """Return a tmp_path with standard workspace subdirs created."""
    (tmp_path / "artifacts" / "signals" / "v2").mkdir(parents=True)
    (tmp_path / "data" / "parquet").mkdir(parents=True)
    (tmp_path / "data" / "duckdb").mkdir(parents=True)
    return tmp_path
