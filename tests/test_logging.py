"""Tests for pmlab.logging module."""
from __future__ import annotations
import logging
from pmlab.logging import get_logger, setup_logging


def test_get_logger_returns_logger():
    logger = get_logger("test_module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "pmlab.test_module"


def test_get_logger_already_prefixed():
    logger = get_logger("pmlab.markets.gamma")
    assert logger.name == "pmlab.markets.gamma"


def test_setup_logging_does_not_raise():
    setup_logging(level="WARNING")
