"""Tests for MarketPlugin ABC and basic contract."""

from __future__ import annotations

from typing import Any

import pytest

from pmlab.core.market_spec import MarketSpec
from pmlab.plugins.base import MarketPlugin

# ---------------------------------------------------------------------------
# Minimal concrete implementation for testing
# ---------------------------------------------------------------------------


class DummyPlugin(MarketPlugin):
    family = "dummy"

    def __init__(self, truth: float | str | None = None) -> None:
        self._truth = truth

    def discover_markets(self, **kwargs: Any) -> list[MarketSpec]:
        return []

    def fetch_features(self, spec: MarketSpec, horizon: str, **kwargs: Any) -> dict[str, float]:
        return {"feature_a": 1.0, "feature_b": 2.5}

    def fetch_truth(self, spec: MarketSpec, **kwargs: Any) -> float | str | None:
        return self._truth

    def build_training_row(
        self, spec: MarketSpec, horizon: str, **kwargs: Any
    ) -> dict[str, Any] | None:
        return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMarketPluginABC:
    def test_cannot_instantiate_abc_directly(self) -> None:
        with pytest.raises(TypeError):
            MarketPlugin()  # type: ignore[abstract]

    def test_dummy_plugin_instantiable(self) -> None:
        p = DummyPlugin()
        assert p.family == "dummy"

    def test_discover_markets_returns_list(self) -> None:
        p = DummyPlugin()
        result = p.discover_markets()
        assert isinstance(result, list)

    def test_fetch_features_returns_float_dict(self) -> None:
        p = DummyPlugin()
        spec = MarketSpec(
            market_id="x",
            slug="x",
            question="q",
            outcome_bins=[],
            close_time="2026-01-01T00:00:00Z",
            market_family="binary",
        )
        features = p.fetch_features(spec, horizon="previous_evening")
        assert isinstance(features, dict)
        assert all(isinstance(v, float) for v in features.values())

    def test_fetch_truth_none_when_unresolved(self) -> None:
        p = DummyPlugin(truth=None)
        spec = MarketSpec(
            market_id="x",
            slug="x",
            question="q",
            outcome_bins=[],
            close_time="2026-01-01T00:00:00Z",
            market_family="binary",
        )
        assert p.fetch_truth(spec) is None

    def test_fetch_truth_returns_value_when_resolved(self) -> None:
        p = DummyPlugin(truth=31.5)
        spec = MarketSpec(
            market_id="x",
            slug="x",
            question="q",
            outcome_bins=[],
            close_time="2026-01-01T00:00:00Z",
            market_family="binary",
        )
        assert p.fetch_truth(spec) == 31.5

    def test_is_truth_final_default_follows_fetch_truth(self) -> None:
        resolved = DummyPlugin(truth=30.0)
        unresolved = DummyPlugin(truth=None)
        spec = MarketSpec(
            market_id="x",
            slug="x",
            question="q",
            outcome_bins=[],
            close_time="2026-01-01T00:00:00Z",
            market_family="binary",
        )
        assert resolved.is_truth_final(spec) is True
        assert unresolved.is_truth_final(spec) is False

    def test_build_training_row_can_return_none(self) -> None:
        p = DummyPlugin()
        spec = MarketSpec(
            market_id="x",
            slug="x",
            question="q",
            outcome_bins=[],
            close_time="2026-01-01T00:00:00Z",
            market_family="binary",
        )
        assert p.build_training_row(spec, "morning_of") is None
