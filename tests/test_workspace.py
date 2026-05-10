"""Tests for WorkspaceContext — Phase 3."""
from pathlib import Path

import pytest

from pmlab.workspace.context import WorkspaceContext

FAKE_REPO = Path('/fake/repo')


class TestFromName:
    def test_from_name_creates_correct_paths(self):
        ctx = WorkspaceContext.from_name('myws', FAKE_REPO)
        assert ctx.data_root == FAKE_REPO / 'data' / 'workspaces' / 'myws'
        assert ctx.parquet_root == FAKE_REPO / 'data' / 'workspaces' / 'myws' / 'parquet'
        assert ctx.duckdb_path == FAKE_REPO / 'data' / 'workspaces' / 'myws' / 'duckdb' / 'warehouse.duckdb'

    def test_from_name_artifacts_root_path(self):
        ctx = WorkspaceContext.from_name('myws', FAKE_REPO)
        assert ctx.artifacts_root == FAKE_REPO / 'artifacts' / 'workspaces' / 'myws'

    def test_workspace_name_stored(self):
        ctx = WorkspaceContext.from_name('alpha', FAKE_REPO)
        assert ctx.name == 'alpha'

    def test_duckdb_path_inside_data_root(self):
        ctx = WorkspaceContext.from_name('beta', FAKE_REPO)
        assert str(ctx.duckdb_path).startswith(str(ctx.data_root))


class TestFromEnv:
    def _set_env(self, monkeypatch, name='testws', base='/tmp/ws'):
        monkeypatch.setenv('PMLAB_WORKSPACE_NAME', name)
        monkeypatch.setenv('PMLAB_DATA_DIR', f'{base}/data')
        monkeypatch.setenv('PMLAB_PARQUET_DIR', f'{base}/data/parquet')
        monkeypatch.setenv('PMLAB_DUCKDB_PATH', f'{base}/data/duckdb/warehouse.duckdb')
        monkeypatch.setenv('PMLAB_ARTIFACTS_DIR', f'{base}/artifacts')

    def test_from_env_reads_pmlab_vars(self, monkeypatch):
        self._set_env(monkeypatch, name='envws', base='/tmp/envws')
        ctx = WorkspaceContext.from_env()
        assert ctx.name == 'envws'
        assert ctx.data_root == Path('/tmp/envws/data')
        assert ctx.parquet_root == Path('/tmp/envws/data/parquet')
        assert ctx.duckdb_path == Path('/tmp/envws/data/duckdb/warehouse.duckdb')
        assert ctx.artifacts_root == Path('/tmp/envws/artifacts')

    def test_from_env_raises_if_var_missing(self, monkeypatch):
        # Ensure none of the vars are set
        for var in ['PMLAB_WORKSPACE_NAME', 'PMLAB_DATA_DIR', 'PMLAB_PARQUET_DIR',
                    'PMLAB_DUCKDB_PATH', 'PMLAB_ARTIFACTS_DIR']:
            monkeypatch.delenv(var, raising=False)
        with pytest.raises(EnvironmentError):
            WorkspaceContext.from_env()

    def test_from_env_raises_if_one_var_missing(self, monkeypatch):
        self._set_env(monkeypatch)
        monkeypatch.delenv('PMLAB_DUCKDB_PATH')
        with pytest.raises(EnvironmentError, match='PMLAB_DUCKDB_PATH'):
            WorkspaceContext.from_env()
