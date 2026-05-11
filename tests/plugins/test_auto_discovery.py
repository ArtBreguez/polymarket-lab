"""TDD tests for plugin auto-discovery via entry_points."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

# This import must exist but discover_plugins must not be there yet
from pmlab.plugins.registry import PluginRegistry

# This import MUST fail before implementation (RED)
from pmlab.plugins.discovery import discover_plugins, load_plugins_from_entry_points


class TestDiscoverPlugins:
    def test_discover_plugins_returns_registry(self):
        """discover_plugins() returns a PluginRegistry instance."""
        registry = discover_plugins()
        assert isinstance(registry, PluginRegistry)

    def test_discover_plugins_no_entry_points(self):
        """With no pmlab.plugins entry_points, registry is empty."""
        with patch("pmlab.plugins.discovery.entry_points") as mock_ep:
            mock_ep.return_value = []
            registry = discover_plugins()
        assert registry.list_families() == []

    def test_load_plugins_loads_valid_plugin(self):
        """load_plugins_from_entry_points loads a plugin class and registers it."""
        from pmlab.plugins.base import MarketPlugin
        from pmlab.core.market_spec import MarketSpec

        class _FakePlugin(MarketPlugin):
            family = "fake_discovery"
            def discover_markets(self, **kwargs): return []
            def fetch_features(self, spec, horizon, **kwargs): return {}
            def fetch_truth(self, spec, **kwargs): return None
            def build_training_row(self, spec, horizon, **kwargs): return None

        mock_ep = MagicMock()
        mock_ep.name = "fake_discovery"
        mock_ep.load.return_value = _FakePlugin

        with patch("pmlab.plugins.discovery.entry_points") as mock_eps:
            mock_eps.return_value = [mock_ep]
            registry = load_plugins_from_entry_points()

        assert "fake_discovery" in registry.list_families()

    def test_load_plugins_skips_bad_entry_point(self):
        """A broken entry_point is skipped with a warning, not a crash."""
        mock_ep = MagicMock()
        mock_ep.name = "broken"
        mock_ep.load.side_effect = ImportError("module not found")

        with patch("pmlab.plugins.discovery.entry_points") as mock_eps:
            mock_eps.return_value = [mock_ep]
            registry = load_plugins_from_entry_points()  # must not raise

        assert registry.list_families() == []

    def test_load_plugins_skips_non_plugin_class(self):
        """A class that doesn't subclass MarketPlugin is skipped."""
        class _NotAPlugin:
            family = "not_real"

        mock_ep = MagicMock()
        mock_ep.name = "not_real"
        mock_ep.load.return_value = _NotAPlugin

        with patch("pmlab.plugins.discovery.entry_points") as mock_eps:
            mock_eps.return_value = [mock_ep]
            registry = load_plugins_from_entry_points()  # must not raise

        assert registry.list_families() == []

    def test_discover_plugins_calls_load_entry_points(self):
        """discover_plugins delegates to load_plugins_from_entry_points."""
        with patch("pmlab.plugins.discovery.entry_points") as mock_eps:
            mock_eps.return_value = []
            registry = discover_plugins()
        assert isinstance(registry, PluginRegistry)
