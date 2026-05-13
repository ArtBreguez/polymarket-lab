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
