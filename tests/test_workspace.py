"""Tests for WorkspaceContext — Phase 3."""

from pathlib import Path

import pytest

from pmlab.config import resolve_workspace_dir
from pmlab.workspace.context import WorkspaceContext

FAKE_REPO = Path("/fake/repo")


class TestFromName:
    def test_from_name_creates_correct_paths(self):
        ctx = WorkspaceContext.from_name("myws", FAKE_REPO)
        assert ctx.data_root == FAKE_REPO / "data" / "workspaces" / "myws"
        assert ctx.parquet_root == FAKE_REPO / "data" / "workspaces" / "myws" / "parquet"
        assert (
            ctx.duckdb_path
            == FAKE_REPO / "data" / "workspaces" / "myws" / "duckdb" / "warehouse.duckdb"
        )

    def test_from_name_artifacts_root_path(self):
        ctx = WorkspaceContext.from_name("myws", FAKE_REPO)
        assert ctx.artifacts_root == FAKE_REPO / "artifacts" / "workspaces" / "myws"

    def test_workspace_name_stored(self):
        ctx = WorkspaceContext.from_name("alpha", FAKE_REPO)
        assert ctx.name == "alpha"

    def test_duckdb_path_inside_data_root(self):
        ctx = WorkspaceContext.from_name("beta", FAKE_REPO)
        assert str(ctx.duckdb_path).startswith(str(ctx.data_root))


class TestFromEnv:
    def _set_env(self, monkeypatch, name="testws", base="/tmp/ws"):
        monkeypatch.setenv("PMLAB_WORKSPACE_NAME", name)
        monkeypatch.setenv("PMLAB_DATA_DIR", f"{base}/data")
        monkeypatch.setenv("PMLAB_PARQUET_DIR", f"{base}/data/parquet")
        monkeypatch.setenv("PMLAB_DUCKDB_PATH", f"{base}/data/duckdb/warehouse.duckdb")
        monkeypatch.setenv("PMLAB_ARTIFACTS_DIR", f"{base}/artifacts")

    def test_from_env_reads_pmlab_vars(self, monkeypatch):
        self._set_env(monkeypatch, name="envws", base="/tmp/envws")
        ctx = WorkspaceContext.from_env()
        assert ctx.name == "envws"
        assert ctx.data_root == Path("/tmp/envws/data")
        assert ctx.parquet_root == Path("/tmp/envws/data/parquet")
        assert ctx.duckdb_path == Path("/tmp/envws/data/duckdb/warehouse.duckdb")
        assert ctx.artifacts_root == Path("/tmp/envws/artifacts")

    def test_from_env_raises_if_var_missing(self, monkeypatch):
        # Ensure none of the vars are set
        for var in [
            "PMLAB_WORKSPACE_NAME",
            "PMLAB_DATA_DIR",
            "PMLAB_PARQUET_DIR",
            "PMLAB_DUCKDB_PATH",
            "PMLAB_ARTIFACTS_DIR",
        ]:
            monkeypatch.delenv(var, raising=False)
        with pytest.raises(EnvironmentError):
            WorkspaceContext.from_env()

    def test_from_env_raises_if_one_var_missing(self, monkeypatch):
        self._set_env(monkeypatch)
        monkeypatch.delenv("PMLAB_DUCKDB_PATH")
        with pytest.raises(EnvironmentError, match="PMLAB_DUCKDB_PATH"):
            WorkspaceContext.from_env()


BAD_NAMES = ["", ".", "..", "../x", "a/b", "a\\b", "/abs", "../../etc", "x\x00y"]


class TestWorkspaceNames:
    """A workspace name is one path segment (#56): it cannot reach another workspace or leave the
    repository, and the empty name cannot stand for the directory that holds every workspace."""

    @pytest.mark.parametrize("name", BAD_NAMES)
    def test_from_name_rejects(self, name):
        with pytest.raises(ValueError, match="workspace"):
            WorkspaceContext.from_name(name, FAKE_REPO)

    @pytest.mark.parametrize("name", BAD_NAMES)
    def test_resolve_workspace_dir_rejects(self, name):
        with pytest.raises(ValueError, match="workspace"):
            resolve_workspace_dir(FAKE_REPO / "artifacts", name)

    @pytest.mark.parametrize("name", ["ops_daily", "historical_real", "my ws", "w-1.2", "Étude"])
    def test_ordinary_names_still_work(self, name):
        ctx = WorkspaceContext.from_name(name, FAKE_REPO)
        assert ctx.artifacts_root == FAKE_REPO / "artifacts" / "workspaces" / name
        assert resolve_workspace_dir(FAKE_REPO / "artifacts", name).name == name

    @pytest.mark.parametrize("name", ["..", "../x", "a/b", "/abs"])
    def test_cli_refuses_with_a_usage_error(self, name, tmp_path):
        from typer.testing import CliRunner

        from pmlab.cli.main import app

        result = CliRunner().invoke(
            app, ["status", "--artifacts-dir", str(tmp_path), "--workspace", name]
        )
        assert result.exit_code == 2
        assert "workspace" in result.output.lower()
        assert "Traceback" not in result.output
