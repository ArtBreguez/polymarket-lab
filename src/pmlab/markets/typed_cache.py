"""TypedCache[T] — type-safe wrapper around DiskCache.

Provides a Generic subclass of DiskCache so callers can annotate the
value type and get IDE completions without losing the disk-cache behaviour.

Example::

    from pmlab.markets.typed_cache import TypedCache

    cache: TypedCache[list[dict]] = TypedCache(".cache", ttl_seconds=3600)
    cache.set("gamma:temp", markets)
    markets = cache.get("gamma:temp")  # type: list[dict] | None
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Generic, TypeVar

from pmlab.markets.cache import DiskCache

__all__ = ["TypedCache"]

T = TypeVar("T")


class TypedCache(DiskCache, Generic[T]):
    """Type-safe TTL disk cache.

    Drop-in replacement for DiskCache with typed get/set signatures.
    No runtime overhead — all typing information is erased at runtime.

    Args:
        cache_dir: Directory to store cache files.
        ttl_seconds: Time-to-live in seconds (default: 3600).
    """

    def __init__(self, cache_dir: Path | str, ttl_seconds: int = 3600) -> None:
        super().__init__(cache_dir=cache_dir, ttl_seconds=ttl_seconds)

    def get(self, key: str, default: Any = None) -> Any:
        """Return typed cached value or default."""
        return super().get(key, default)

    def set(self, key: str, value: T) -> None:
        """Store a typed value under key."""
        super().set(key, value)
