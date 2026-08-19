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

    def discover_markets(self, **kw: Any) -> list[MarketSpec]:
        return []

    def fetch_features(self, s: MarketSpec, h: str, **kw: Any) -> dict[str, float]:
        return {}

    def fetch_truth(self, s: MarketSpec, **kw: Any) -> float | str | None:
        return None

    def build_training_row(self, s: MarketSpec, h: str, **kw: Any) -> dict | None:
        return None


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


class TestBuildPlugin:
    """The resolver the CLI uses to turn a --plugin name into a live plugin."""

    def test_builds_builtin_weather_with_gamma_client(self) -> None:
        from pmlab.plugins.registry import build_plugin
        from pmlab.plugins.weather_tmax.plugin import WeatherTmaxPlugin

        sentinel = object()
        plugin = build_plugin("weather_tmax", gamma_client=sentinel)
        assert isinstance(plugin, WeatherTmaxPlugin)
        assert plugin._gamma is sentinel  # gamma client is injected, not ignored

    def test_builds_builtin_sports_f1_with_gamma_client(self) -> None:
        from pmlab.plugins.registry import build_plugin
        from pmlab.plugins.sports_f1.plugin import SportsF1Plugin

        sentinel = object()
        plugin = build_plugin("sports_f1", gamma_client=sentinel)
        assert isinstance(plugin, SportsF1Plugin)
        assert plugin._gamma is sentinel

    def test_unknown_family_raises_with_available_list(self) -> None:
        from pmlab.plugins.registry import build_plugin

        with pytest.raises(KeyError, match="No plugin registered for family 'banana'"):
            build_plugin("banana")

    def test_available_families_includes_builtins(self) -> None:
        from pmlab.plugins.registry import available_families

        fams = available_families()
        assert "weather_tmax" in fams
        assert "sports_f1" in fams
        assert fams == sorted(fams)  # returned sorted
