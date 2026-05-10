"""TTL disk cache for Polymarket API responses."""
from __future__ import annotations
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

__all__ = ["DiskCache"]
_SENTINEL = object()

class DiskCache:
    def __init__(self, cache_dir: Path | str, ttl_seconds: int = 3600) -> None:
        self.cache_dir = Path(cache_dir)
        self.ttl_seconds = ttl_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str, default: Any = None) -> Any:
        path = self._key_path(key)
        if not path.exists():
            return default
        try:
            entry = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return default
        expires_at = datetime.fromisoformat(entry["expires_at"])
        if datetime.now(UTC) >= expires_at:
            path.unlink(missing_ok=True)
            return default
        return entry["value"]

    def set(self, key: str, value: Any) -> None:
        expires_at = datetime.now(UTC) + timedelta(seconds=self.ttl_seconds)
        entry = {"expires_at": expires_at.isoformat(), "value": value}
        self._key_path(key).write_text(json.dumps(entry))

    def delete(self, key: str) -> bool:
        path = self._key_path(key)
        if path.exists():
            path.unlink()
            return True
        return False

    def clear(self) -> int:
        count = 0
        for p in self.cache_dir.glob("*.json"):
            p.unlink(missing_ok=True)
            count += 1
        return count

    def purge_expired(self) -> int:
        count = 0
        now = datetime.now(UTC)
        for p in self.cache_dir.glob("*.json"):
            try:
                entry = json.loads(p.read_text())
                if now >= datetime.fromisoformat(entry["expires_at"]):
                    p.unlink(missing_ok=True)
                    count += 1
            except (json.JSONDecodeError, KeyError, OSError):
                p.unlink(missing_ok=True)
                count += 1
        return count

    def __contains__(self, key: str) -> bool:
        return self.get(key, _SENTINEL) is not _SENTINEL

    def _key_path(self, key: str) -> Path:
        h = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{h}.json"
