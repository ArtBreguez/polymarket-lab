import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class WorkspaceContext:
    name: str
    data_root: Path
    artifacts_root: Path
    parquet_root: Path
    duckdb_path: Path

    @classmethod
    def from_name(cls, name: str, repo_root: Path) -> 'WorkspaceContext':
        data_root = repo_root / 'data' / 'workspaces' / name
        artifacts_root = repo_root / 'artifacts' / 'workspaces' / name
        parquet_root = data_root / 'parquet'
        duckdb_path = data_root / 'duckdb' / 'warehouse.duckdb'
        return cls(name=name, data_root=data_root, artifacts_root=artifacts_root,
                   parquet_root=parquet_root, duckdb_path=duckdb_path)

    @classmethod
    def from_env(cls) -> 'WorkspaceContext':
        required = ['PMLAB_WORKSPACE_NAME', 'PMLAB_DATA_DIR', 'PMLAB_PARQUET_DIR',
                    'PMLAB_DUCKDB_PATH', 'PMLAB_ARTIFACTS_DIR']
        missing = [v for v in required if not os.environ.get(v)]
        if missing:
            raise OSError(f'Missing required env vars: {missing}')
        return cls(
            name=os.environ['PMLAB_WORKSPACE_NAME'],
            data_root=Path(os.environ['PMLAB_DATA_DIR']),
            artifacts_root=Path(os.environ['PMLAB_ARTIFACTS_DIR']),
            parquet_root=Path(os.environ['PMLAB_PARQUET_DIR']),
            duckdb_path=Path(os.environ['PMLAB_DUCKDB_PATH']),
        )
