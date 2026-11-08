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
