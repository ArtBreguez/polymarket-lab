"""Structured logging for pmlab using rich."""

from __future__ import annotations

import logging
import os

from rich.logging import RichHandler

__all__ = ["get_logger", "setup_logging"]

_LOG_LEVEL = os.environ.get("PMLAB_LOG_LEVEL", "INFO").upper()


def setup_logging(level: str = _LOG_LEVEL) -> None:
    """Configure root pmlab logger with rich handler.

    Call once at application startup (CLI entry point).
    Library code should use get_logger() and let the caller configure.
    """
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )
    logging.getLogger("pmlab").setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """Get a logger under the pmlab namespace.

    Args:
        name: Module name, e.g. __name__. Will be prefixed with 'pmlab.'
              if it doesn't already start with 'pmlab'.

    Returns:
        A Logger instance.
    """
    if not name.startswith("pmlab"):
        name = f"pmlab.{name}"
    return logging.getLogger(name)
