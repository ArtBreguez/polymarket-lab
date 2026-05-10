"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture
def tmp_workspace(tmp_path):
    """Return a tmp_path with standard workspace subdirs created."""
    (tmp_path / "artifacts" / "signals" / "v2").mkdir(parents=True)
    (tmp_path / "data" / "parquet").mkdir(parents=True)
    (tmp_path / "data" / "duckdb").mkdir(parents=True)
    return tmp_path
