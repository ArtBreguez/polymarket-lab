"""Plugin auto-discovery via importlib.metadata entry_points.

Plugins published as Python packages can register themselves by adding
to their pyproject.toml:

    [project.entry-points."pmlab.plugins"]
    weather_tmax = "my_package.plugin:MyPlugin"

Then calling ``discover_plugins()`` (or ``load_plugins_from_entry_points()``)
will automatically instantiate and register each plugin.
"""

from __future__ import annotations

import logging
from importlib.metadata import entry_points
from typing import Any

from pmlab.plugins.base import MarketPlugin
from pmlab.plugins.registry import PluginRegistry

_logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "pmlab.plugins"

__all__ = ["discover_plugins", "load_plugins_from_entry_points"]


def load_plugins_from_entry_points(
    registry: PluginRegistry | None = None,
) -> PluginRegistry:
    """Load all plugins registered under the 'pmlab.plugins' entry_point group.

    Args:
        registry: Optional existing registry to add plugins to.
                  If None, a new PluginRegistry is created.

    Returns:
        Registry populated with all successfully loaded plugins.
    """
    if registry is None:
        registry = PluginRegistry()

    eps = entry_points(group=ENTRY_POINT_GROUP)

    for ep in eps:
        try:
            plugin_cls: Any = ep.load()
        except Exception as exc:
            _logger.warning(
                "Failed to load pmlab plugin entry_point '%s': %s",
                ep.name,
                exc,
            )
            continue

        # Validate it's a proper MarketPlugin subclass
        if not (
            isinstance(plugin_cls, type)
            and issubclass(plugin_cls, MarketPlugin)
            and plugin_cls is not MarketPlugin
        ):
            _logger.warning(
                "Entry_point '%s' did not load a MarketPlugin subclass, skipping.",
                ep.name,
            )
            continue

        try:
            instance = plugin_cls()
            registry.register(instance)
            _logger.debug("Registered plugin family '%s' from entry_point.", instance.family)
        except Exception as exc:
            _logger.warning(
                "Failed to instantiate plugin '%s' from entry_point: %s",
                ep.name,
                exc,
            )

    return registry


def discover_plugins() -> PluginRegistry:
    """Convenience wrapper: create a new registry and populate from entry_points.

    Returns:
        A PluginRegistry with all installed pmlab plugins.
    """
    return load_plugins_from_entry_points()
