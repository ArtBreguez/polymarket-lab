"""Tests for DiskCache."""

from __future__ import annotations

import time

import pytest

from pmlab.markets.cache import DiskCache


@pytest.fixture
def cache(tmp_path):
    return DiskCache(tmp_path / "cache", ttl_seconds=10)


class TestDiskCache:
    def test_set_and_get(self, cache):
        cache.set("key1", {"data": 42})
        assert cache.get("key1") == {"data": 42}

    def test_missing_key_returns_default(self, cache):
        assert cache.get("nonexistent") is None
        assert cache.get("nonexistent", "fallback") == "fallback"

    def test_contains_existing(self, cache):
        cache.set("k", 1)
        assert "k" in cache

    def test_contains_missing(self, cache):
        assert "missing" not in cache

    def test_delete(self, cache):
        cache.set("k", "v")
        assert cache.delete("k") is True
        assert cache.get("k") is None

    def test_delete_missing(self, cache):
        assert cache.delete("nope") is False

    def test_clear(self, cache):
        cache.set("a", 1)
        cache.set("b", 2)
        count = cache.clear()
        assert count == 2
        assert cache.get("a") is None

    def test_expired_entry_returns_default(self, tmp_path):
        short_cache = DiskCache(tmp_path / "short", ttl_seconds=1)
        short_cache.set("k", "v")
        time.sleep(1.1)
        assert short_cache.get("k") is None

    def test_purge_expired(self, tmp_path):
        short_cache = DiskCache(tmp_path / "purge", ttl_seconds=1)
        short_cache.set("x", 1)
        short_cache.set("y", 2)
        time.sleep(1.1)
        assert short_cache.purge_expired() == 2

    def test_list_values(self, cache):
        cache.set("a", [1, 2, 3])
        assert cache.get("a") == [1, 2, 3]

    def test_creates_directory(self, tmp_path):
        new_dir = tmp_path / "nested" / "cache"
        DiskCache(new_dir)
        assert new_dir.exists()

    def test_corrupt_file_returns_default(self, tmp_path):
        c = DiskCache(tmp_path / "c")
        c._key_path("bad").write_text("not json{{{")
        assert c.get("bad") is None
