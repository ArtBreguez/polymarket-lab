"""TDD tests for TypedCache[T] — written BEFORE implementation."""
from __future__ import annotations

import pytest

# This import MUST fail before implementation (RED)
from pmlab.markets.typed_cache import TypedCache


class TestTypedCacheStr:
    def test_set_and_get(self, tmp_path):
        c: TypedCache[str] = TypedCache(tmp_path / "c", ttl_seconds=60)
        c.set("key", "hello")
        assert c.get("key") == "hello"

    def test_missing_returns_none(self, tmp_path):
        c: TypedCache[str] = TypedCache(tmp_path / "c", ttl_seconds=60)
        assert c.get("missing") is None

    def test_missing_returns_default(self, tmp_path):
        c: TypedCache[str] = TypedCache(tmp_path / "c", ttl_seconds=60)
        assert c.get("missing", "fallback") == "fallback"


class TestTypedCacheList:
    def test_list_roundtrip(self, tmp_path):
        c: TypedCache[list[dict]] = TypedCache(tmp_path / "c", ttl_seconds=60)
        val = [{"market_id": "m1", "price": 0.6}]
        c.set("markets", val)
        assert c.get("markets") == val


class TestTypedCacheContains:
    def test_contains_existing(self, tmp_path):
        c: TypedCache[int] = TypedCache(tmp_path / "c", ttl_seconds=60)
        c.set("n", 42)
        assert "n" in c

    def test_not_contains_missing(self, tmp_path):
        c: TypedCache[int] = TypedCache(tmp_path / "c", ttl_seconds=60)
        assert "missing" not in c


class TestTypedCacheClear:
    def test_clear(self, tmp_path):
        c: TypedCache[str] = TypedCache(tmp_path / "c", ttl_seconds=60)
        c.set("a", "x")
        c.set("b", "y")
        n = c.clear()
        assert n == 2
        assert c.get("a") is None

    def test_delete(self, tmp_path):
        c: TypedCache[str] = TypedCache(tmp_path / "c", ttl_seconds=60)
        c.set("k", "v")
        assert c.delete("k") is True
        assert c.delete("k") is False


class TestTypedCacheExpiry:
    def test_expired_returns_none(self, tmp_path):
        import time
        c: TypedCache[str] = TypedCache(tmp_path / "c", ttl_seconds=1)
        c.set("k", "v")
        time.sleep(1.1)
        assert c.get("k") is None


class TestTypedCacheIsDiskCache:
    def test_is_diskcache_subclass(self, tmp_path):
        from pmlab.markets.cache import DiskCache
        c = TypedCache[str](tmp_path / "c")
        assert isinstance(c, DiskCache)
