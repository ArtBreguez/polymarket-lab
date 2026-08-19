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
                f"No plugin registered for family '{family}'. Available: {sorted(self._plugins)}"
            )
        return self._plugins[family]

    def list_families(self) -> list[str]:
        """Return sorted list of registered family names."""
        return sorted(self._plugins)

    def unregister(self, family: str) -> None:
        """Remove plugin for *family*. No-op if not registered."""
        self._plugins.pop(family, None)


# Built-in plugin families shipped with pmlab. External plugins register via
# entry_points (see plugins.discovery); these are the batteries-included ones so
# the CLI works out of the box with no extra install.
_BUILTIN_FAMILIES = ("weather_tmax", "sports_f1")


def build_plugin(family: str, *, gamma_client: object | None = None) -> MarketPlugin:
    """Instantiate a plugin by family name, injecting a gamma client.

    Resolves built-in plugins first, then entry-point plugins. Raises
    ``KeyError`` with the list of available families if *family* is unknown.
    """
    # Built-ins are instantiated explicitly (not via a type[MarketPlugin] dict)
    # so their gamma_client keyword argument stays statically typed.
    if family == "weather_tmax":
        from pmlab.plugins.weather_tmax.plugin import WeatherTmaxPlugin

        return WeatherTmaxPlugin(gamma_client=gamma_client)
    if family == "sports_f1":
        from pmlab.plugins.sports_f1.plugin import SportsF1Plugin

        return SportsF1Plugin(gamma_client=gamma_client)

    # Fall back to entry-point discovered plugins (external packages).
    from pmlab.plugins.discovery import discover_plugins

    registry = discover_plugins()
    if family not in registry.list_families():
        available = sorted(set(_BUILTIN_FAMILIES) | set(registry.list_families()))
        raise KeyError(f"No plugin registered for family '{family}'. Available: {available}")
    return registry.get(family)


def available_families() -> list[str]:
    """Return sorted family names of all resolvable plugins (built-in + entry-point)."""
    from pmlab.plugins.discovery import discover_plugins

    try:
        discovered = set(discover_plugins().list_families())
    except Exception:
        discovered = set()
    return sorted(set(_BUILTIN_FAMILIES) | discovered)
