"""Central configuration for pmlab via pydantic-settings.

All settings can be overridden via environment variables or a .env file.
Environment variable names are the field names uppercased with PMLAB_ prefix
(configured via env_prefix).

Example .env:
    PMLAB_ARTIFACTS_DIR=./artifacts
    PMLAB_WORKSPACE=ops_daily
    POLY_API_KEY=your-key-here
    PMLAB_LOG_LEVEL=DEBUG
"""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["PmlabSettings", "get_settings", "resolve_workspace_dir"]


def resolve_workspace_dir(artifacts_dir: Path, workspace: str) -> Path:
    """The directory that holds a workspace's signals and paper trades.

    ``<artifacts_dir>/workspaces/<workspace>``, unless `artifacts_dir` already is that
    directory. ``scripts/pmlab-workspace`` points PMLAB_ARTIFACTS_DIR at the workspace's own
    artifacts (``artifacts/workspaces/<workspace>``); appending ``workspaces/<workspace>``
    again would look for the trade log in a directory nobody wrote, so ``pmlab -a artifacts
    -w ws`` and ``scripts/pmlab-workspace ws pmlab`` name the same files.
    """
    if artifacts_dir.name == workspace and artifacts_dir.parent.name == "workspaces":
        return artifacts_dir
    return artifacts_dir / "workspaces" / workspace


class PmlabSettings(BaseSettings):
    """pmlab runtime configuration.

    All fields can be overridden via environment variables:
        PMLAB_ARTIFACTS_DIR, PMLAB_WORKSPACE, PMLAB_CACHE_DIR,
        PMLAB_CACHE_TTL, PMLAB_LOG_LEVEL,
        POLY_API_KEY, POLY_API_SECRET, POLY_API_PASSPHRASE,
        GAMMA_API_BASE, CLOB_API_BASE
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="PMLAB_",
        extra="ignore",
        populate_by_name=True,
    )

    # Workspace / paths
    artifacts_dir: Path = Field(default=Path("artifacts"), description="Root artifacts directory")
    # PMLAB_WORKSPACE_NAME is what scripts/pmlab-workspace exports (and WorkspaceContext reads);
    # PMLAB_WORKSPACE, the documented one, wins when both are set.
    workspace: str = Field(
        default="ops_daily",
        description="Active workspace name",
        validation_alias=AliasChoices("PMLAB_WORKSPACE", "PMLAB_WORKSPACE_NAME", "workspace"),
    )
    cache_dir: Path = Field(default=Path(".pmlab_cache"), description="Disk cache directory")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")

    # Logging
    log_level: str = Field(default="INFO", description="Log level: DEBUG, INFO, WARNING, ERROR")

    # Polymarket API credentials (no PMLAB_ prefix for these)
    poly_api_key: str = Field(default="", alias="POLY_API_KEY")
    poly_api_secret: str = Field(default="", alias="POLY_API_SECRET")
    poly_api_passphrase: str = Field(default="", alias="POLY_API_PASSPHRASE")

    # API base URLs (for testing / staging overrides)
    gamma_api_base: str = Field(
        default="https://gamma-api.polymarket.com",
        alias="GAMMA_API_BASE",
    )
    clob_api_base: str = Field(
        default="https://clob.polymarket.com",
        alias="CLOB_API_BASE",
    )

    @property
    def workspace_dir(self) -> Path:
        """Resolve the active workspace directory."""
        return resolve_workspace_dir(self.artifacts_dir, self.workspace)

    @property
    def champion_json_path(self) -> Path:
        """Path to the published champion manifest."""
        return self.artifacts_dir / "public_models" / "champion.json"

    @property
    def trades_path(self) -> Path:
        """Path to the forward paper trades file in the active workspace."""
        return self.workspace_dir / "signals" / "v2" / "forward_paper_trades.json"

    @property
    def has_api_credentials(self) -> bool:
        """Return True if all three POLY_API_* credentials are set."""
        return bool(self.poly_api_key and self.poly_api_secret and self.poly_api_passphrase)


def get_settings() -> PmlabSettings:
    """Return a PmlabSettings instance loaded from environment / .env file."""
    return PmlabSettings()
