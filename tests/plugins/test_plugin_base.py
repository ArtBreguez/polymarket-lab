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


# ---------------------------------------------------------------------------
# build_training_rows: the plural form, and the compatibility of the singular one
# ---------------------------------------------------------------------------


class SingularOnlyPlugin(DummyPlugin):
    """A third-party plugin written before the plural form existed."""

    def __init__(self, row: dict[str, Any] | None) -> None:
        super().__init__()
        self._row = row
        self.seen: list[tuple[str, dict[str, Any]]] = []

    def build_training_row(
        self, spec: MarketSpec, horizon: str, **kwargs: Any
    ) -> dict[str, Any] | None:
        self.seen.append((horizon, kwargs))
        return self._row


_SPEC = MarketSpec(
    market_id="x",
    slug="x",
    question="q",
    outcome_bins=[],
    close_time="2026-01-01T00:00:00Z",
    market_family="binary",
)


class TestBuildTrainingRowsDefault:
    def test_a_singular_only_plugin_still_yields_its_row(self) -> None:
        row = {"market_id": "x", "market_price": 0.4}
        assert SingularOnlyPlugin(row).build_training_rows(_SPEC, "morning_of") == [row]

    def test_an_unbuildable_market_yields_no_rows(self) -> None:
        assert SingularOnlyPlugin(None).build_training_rows(_SPEC, "morning_of") == []

    def test_the_horizon_and_kwargs_reach_the_singular_method(self) -> None:
        p = SingularOnlyPlugin({"market_id": "x"})
        p.build_training_rows(_SPEC, "previous_evening", decision_time="t", flag=True)
        assert p.seen == [("previous_evening", {"decision_time": "t", "flag": True})]

    def test_the_result_is_a_list_of_dicts(self) -> None:
        rows = SingularOnlyPlugin({"market_id": "x"}).build_training_rows(_SPEC, "h")
        assert isinstance(rows, list) and all(isinstance(r, dict) for r in rows)

    def test_a_plugin_may_override_it_with_several_rows(self) -> None:
        class PerBin(DummyPlugin):
            def build_training_rows(
                self, spec: MarketSpec, horizon: str, **kwargs: Any
            ) -> list[dict[str, Any]]:
                return [{"outcome_label": "a"}, {"outcome_label": "b"}]

        assert [r["outcome_label"] for r in PerBin().build_training_rows(_SPEC, "h")] == ["a", "b"]

    def test_the_singular_method_stays_abstract(self) -> None:
        class NoSingular(MarketPlugin):
            family = "n"

            def discover_markets(self, **kwargs: Any) -> list[MarketSpec]:
                return []

            def fetch_features(self, spec: MarketSpec, horizon: str, **kwargs: Any):  # type: ignore[no-untyped-def]
                return {}

            def fetch_truth(self, spec: MarketSpec, **kwargs: Any):  # type: ignore[no-untyped-def]
                return None

        with pytest.raises(TypeError):
            NoSingular()  # type: ignore[abstract]
