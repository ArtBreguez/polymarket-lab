"""Tests for PmlabSettings."""

from __future__ import annotations

from pathlib import Path

from pmlab.config import PmlabSettings, get_settings


class TestPmlabSettings:
    def test_defaults(self):
        s = PmlabSettings()
        assert s.workspace == "ops_daily"
        assert s.artifacts_dir == Path("artifacts")
        assert s.cache_ttl == 3600
        assert s.log_level == "INFO"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("PMLAB_WORKSPACE", "historical_real")
        monkeypatch.setenv("PMLAB_LOG_LEVEL", "DEBUG")
        s = PmlabSettings()
        assert s.workspace == "historical_real"
        assert s.log_level == "DEBUG"

    def test_workspace_dir_property(self):
        s = PmlabSettings()
        s.artifacts_dir = Path("/tmp/artifacts")
        assert s.workspace_dir == Path("/tmp/artifacts/workspaces/ops_daily")

    def test_champion_json_path(self):
        s = PmlabSettings()
        s.artifacts_dir = Path("/tmp/artifacts")
        assert s.champion_json_path == Path("/tmp/artifacts/public_models/champion.json")

    def test_trades_path(self):
        s = PmlabSettings()
        s.artifacts_dir = Path("/tmp/art")
        assert "forward_paper_trades.json" in str(s.trades_path)

    def test_has_api_credentials_false(self):
        s = PmlabSettings()
        assert s.has_api_credentials is False

    def test_has_api_credentials_true(self, monkeypatch):
        monkeypatch.setenv("POLY_API_KEY", "k")
        monkeypatch.setenv("POLY_API_SECRET", "s")
        monkeypatch.setenv("POLY_API_PASSPHRASE", "p")
        s = PmlabSettings()
        assert s.has_api_credentials is True

    def test_get_settings_returns_instance(self):
        s = get_settings()
        assert isinstance(s, PmlabSettings)


class TestTheWorkspaceDirectoryIsTheSameFromEveryEntryStyle:
    """`pmlab -a artifacts -w ws` and `scripts/pmlab-workspace ws pmlab` name one workspace.

    The script points PMLAB_ARTIFACTS_DIR at the workspace's own artifacts
    (`artifacts/workspaces/ws`). The settings used to append `workspaces/<name>` again, so
    `status`, `record-trades` and `monitor` looked for the trade log in
    `artifacts/workspaces/ws/workspaces/ops_daily/...`, a directory nobody wrote.
    """

    def test_a_plain_artifacts_root_gets_the_workspace_appended(self):
        from pmlab.config import resolve_workspace_dir

        assert resolve_workspace_dir(Path("artifacts"), "ws") == Path("artifacts/workspaces/ws")

    def test_an_artifacts_dir_that_already_is_the_workspace_is_not_appended_to(self):
        from pmlab.config import resolve_workspace_dir

        scoped = Path("/repo/artifacts/workspaces/ws")
        assert resolve_workspace_dir(scoped, "ws") == scoped

    def test_a_different_workspace_name_is_still_appended(self):
        from pmlab.config import resolve_workspace_dir

        other = Path("/repo/artifacts/workspaces/other")
        assert resolve_workspace_dir(other, "ws") == other / "workspaces" / "ws"

    def test_only_a_workspaces_parent_counts_as_the_scoped_form(self):
        from pmlab.config import resolve_workspace_dir

        odd = Path("/data/ws")  # a directory that happens to share the workspace's name
        assert resolve_workspace_dir(odd, "ws") == odd / "workspaces" / "ws"

    def test_both_entry_styles_resolve_to_the_same_trade_log(self):
        direct = PmlabSettings()
        direct.artifacts_dir = Path("/repo/artifacts")
        direct.workspace = "paper"
        scripted = PmlabSettings()
        scripted.artifacts_dir = Path("/repo/artifacts/workspaces/paper")
        scripted.workspace = "paper"
        assert direct.trades_path == scripted.trades_path
        assert direct.trades_path == Path(
            "/repo/artifacts/workspaces/paper/signals/v2/forward_paper_trades.json"
        )

    def test_the_champion_stays_where_each_style_always_put_it(self):
        direct = PmlabSettings()
        direct.artifacts_dir = Path("/repo/artifacts")
        scripted = PmlabSettings()
        scripted.artifacts_dir = Path("/repo/artifacts/workspaces/paper")
        assert direct.champion_json_path == Path("/repo/artifacts/public_models/champion.json")
        assert scripted.champion_json_path == Path(
            "/repo/artifacts/workspaces/paper/public_models/champion.json"
        )

    def test_the_script_names_its_workspace_in_pmlab_workspace_name(self, monkeypatch):
        monkeypatch.delenv("PMLAB_WORKSPACE", raising=False)
        monkeypatch.setenv("PMLAB_WORKSPACE_NAME", "from_the_script")
        assert PmlabSettings().workspace == "from_the_script"

    def test_pmlab_workspace_wins_over_the_scripts_name(self, monkeypatch):
        monkeypatch.setenv("PMLAB_WORKSPACE", "explicit")
        monkeypatch.setenv("PMLAB_WORKSPACE_NAME", "from_the_script")
        assert PmlabSettings().workspace == "explicit"

    def test_the_default_workspace_is_unchanged(self, monkeypatch):
        monkeypatch.delenv("PMLAB_WORKSPACE", raising=False)
        monkeypatch.delenv("PMLAB_WORKSPACE_NAME", raising=False)
        assert PmlabSettings().workspace == "ops_daily"
