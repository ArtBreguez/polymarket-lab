"""Plugin registry — lookup plugins by family name."""

from __future__ import annotations

from pmlab.plugins.base import MarketPlugin


class PluginRegistry:
    """Registry of installed MarketPlugin instances, keyed by family name."""

    def __init__(self) -> None:
        self._plugins: dict[str, MarketPlugin] = {}

    def register(self, plugin: MarketPlugin) -> None:
        """Register a plugin. Raises if family already registered."""
        if plugin.family in self._plugins:
            raise ValueError(
                f"Plugin for family '{plugin.family}' is already registered. "
                "Unregister it first or use a different family name."
            )
        self._plugins[plugin.family] = plugin

    def get(self, family: str) -> MarketPlugin:
        """Return plugin for *family*. Raises KeyError if not found."""
        if family not in self._plugins:
            raise KeyError(
                f"No plugin registered for family '{family}'. "
                f"Available: {sorted(self._plugins)}"
            )
        return self._plugins[family]

    def list_families(self) -> list[str]:
        """Return sorted list of registered family names."""
        return sorted(self._plugins)

    def unregister(self, family: str) -> None:
        """Remove plugin for *family*. No-op if not registered."""
        self._plugins.pop(family, None)
