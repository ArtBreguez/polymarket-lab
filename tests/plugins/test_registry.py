"""Tests for PluginRegistry."""

from __future__ import annotations

from typing import Any

import pytest

from pmlab.core.market_spec import MarketSpec
from pmlab.plugins.base import MarketPlugin
from pmlab.plugins.registry import PluginRegistry


class _Plugin(MarketPlugin):
    def __init__(self, name: str) -> None:
        self.family = name

    def discover_markets(self, **kw: Any) -> list[MarketSpec]: return []
    def fetch_features(self, s: MarketSpec, h: str, **kw: Any) -> dict[str, float]: return {}
    def fetch_truth(self, s: MarketSpec, **kw: Any) -> float | str | None: return None
    def build_training_row(self, s: MarketSpec, h: str, **kw: Any) -> dict | None: return None


class TestPluginRegistry:
    def test_register_and_get(self) -> None:
        registry = PluginRegistry()
        p = _Plugin("weather_tmax")
        registry.register(p)
        assert registry.get("weather_tmax") is p

    def test_get_unknown_raises_key_error(self) -> None:
        registry = PluginRegistry()
        with pytest.raises(KeyError, match="No plugin registered"):
            registry.get("nonexistent")

    def test_list_families_empty(self) -> None:
        assert PluginRegistry().list_families() == []

    def test_list_families_sorted(self) -> None:
        registry = PluginRegistry()
        registry.register(_Plugin("sports_f1"))
        registry.register(_Plugin("weather_tmax"))
        assert registry.list_families() == ["sports_f1", "weather_tmax"]

    def test_duplicate_register_raises(self) -> None:
        registry = PluginRegistry()
        registry.register(_Plugin("weather_tmax"))
        with pytest.raises(ValueError, match="already registered"):
            registry.register(_Plugin("weather_tmax"))

    def test_unregister_removes_plugin(self) -> None:
        registry = PluginRegistry()
        registry.register(_Plugin("weather_tmax"))
        registry.unregister("weather_tmax")
        with pytest.raises(KeyError):
            registry.get("weather_tmax")

    def test_unregister_nonexistent_is_noop(self) -> None:
        registry = PluginRegistry()
        registry.unregister("never_existed")  # should not raise
