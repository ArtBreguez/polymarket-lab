import os
from dataclasses import dataclass
from pathlib import Path


def validate_path_segment(name: str, what: str = "name") -> str:
    """``name`` if it is a single, ordinary path component, else ``ValueError`` (#56).

    A name that becomes a directory or file name must not reach another workspace or leave its
    root: not empty, not ``.`` or ``..``, no separator, not absolute, no NUL.
    """
    if not isinstance(name, str):
        raise ValueError(f"{what} must be a string, got {type(name).__name__}")
    if name in ("", ".", "..") or "/" in name or "\\" in name or "\x00" in name:
        raise ValueError(
            f"invalid {what} {name!r}: it must be one path component "
            "(not empty, '.' or '..', and without '/' or '\\')"
        )
    return name


@dataclass
class WorkspaceContext:
    name: str
    data_root: Path
    artifacts_root: Path
    parquet_root: Path
    duckdb_path: Path

    @classmethod
    def from_name(cls, name: str, repo_root: Path) -> "WorkspaceContext":
        validate_path_segment(name, "workspace name")
        data_root = repo_root / "data" / "workspaces" / name
        artifacts_root = repo_root / "artifacts" / "workspaces" / name
        parquet_root = data_root / "parquet"
        duckdb_path = data_root / "duckdb" / "warehouse.duckdb"
        return cls(
            name=name,
            data_root=data_root,
            artifacts_root=artifacts_root,
            parquet_root=parquet_root,
            duckdb_path=duckdb_path,
        )

    @classmethod
    def from_env(cls) -> "WorkspaceContext":
        required = [
            "PMLAB_WORKSPACE_NAME",
            "PMLAB_DATA_DIR",
            "PMLAB_PARQUET_DIR",
            "PMLAB_DUCKDB_PATH",
            "PMLAB_ARTIFACTS_DIR",
        ]
        missing = [v for v in required if not os.environ.get(v)]
        if missing:
            raise OSError(f"Missing required env vars: {missing}")
        return cls(
            name=os.environ["PMLAB_WORKSPACE_NAME"],
            data_root=Path(os.environ["PMLAB_DATA_DIR"]),
            artifacts_root=Path(os.environ["PMLAB_ARTIFACTS_DIR"]),
            parquet_root=Path(os.environ["PMLAB_PARQUET_DIR"]),
            duckdb_path=Path(os.environ["PMLAB_DUCKDB_PATH"]),
        )
