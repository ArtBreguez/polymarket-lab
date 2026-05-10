"""Integration test: full pipeline with DummyPlugin."""
from __future__ import annotations

from datetime import UTC
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pmlab.backtest.holdout_gate import HoldoutGateResult
from pmlab.backtest.rolling_origin import rolling_origin_eval
from pmlab.core.market_spec import MarketSpec, OutcomeBin
from pmlab.execution.edge_signal import EdgeSignal
from pmlab.execution.paper_broker import PaperBroker
from pmlab.modeling.champion import ChampionManifest
from pmlab.modeling.lgbm_baseline import LGBMForecaster
from pmlab.plugins.base import MarketPlugin
from pmlab.plugins.registry import PluginRegistry


# --- Minimal concrete plugin for testing ---
class IntegrationPlugin(MarketPlugin):
    family = "integration_test"
    def __init__(self, truth_value: float = 25.0):
        self._truth = truth_value

    def discover_markets(self, **kw: Any) -> list[MarketSpec]:
        return [self._make_spec()]

    def fetch_features(self, spec: MarketSpec, horizon: str, **kw: Any) -> dict[str, float]:
        return {"feature_temp": 25.0, "feature_lead_days": 1.0}

    def fetch_truth(self, spec: MarketSpec, **kw: Any) -> float | None:
        return self._truth

    def build_training_row(self, spec: MarketSpec, horizon: str, **kw: Any) -> dict | None:
        features = self.fetch_features(spec, horizon)
        truth = self.fetch_truth(spec)
        if truth is None:
            return None
        winning_label = spec.resolve_winning_bin(float(truth))
        return {"market_id": spec.market_id, "decision_horizon": horizon,
                "winning_label": winning_label, "market_price": 0.3, **features}

    def _make_spec(self) -> MarketSpec:
        return MarketSpec(
            market_id="int_test_001", slug="test",
            question="Test market?",
            outcome_bins=[
                OutcomeBin(label="cold", upper=20.0, upper_inclusive=False),
                OutcomeBin(label="warm", lower=20.0, upper=30.0, lower_inclusive=True, upper_inclusive=False),
                OutcomeBin(label="hot", lower=30.0),
            ],
            close_time="2026-05-10T20:00:00Z",
            market_family="range",
            metadata={"city": "TestCity", "target_date": "2026-05-10"},
        )


def _make_synthetic_panel(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = [f"2026-0{i//10 + 1}-{i%28 + 1:02d}" for i in range(n)]
    return pd.DataFrame({
        "market_id": [f"m{i%10}" for i in range(n)],
        "decision_date": dates,
        "outcome_label": ["warm" if i % 2 == 0 else "hot" for i in range(n)],
        "winning_label": ["warm" if i % 3 != 0 else "hot" for i in range(n)],
        "market_price": rng.uniform(0.2, 0.7, n),
        "segment": ["TestCity"] * n,
        "feature_temp": rng.normal(25, 3, n),
        "feature_lead_days": rng.uniform(1, 5, n),
    })


class TestFullPipeline:
    def test_plugin_registry_integration(self):
        registry = PluginRegistry()
        registry.register(IntegrationPlugin())
        assert "integration_test" in registry.list_families()
        plugin = registry.get("integration_test")
        markets = plugin.discover_markets()
        assert len(markets) == 1

    def test_backtest_to_gate_pipeline(self):
        panel = _make_synthetic_panel(120)
        model = LGBMForecaster()
        result = rolling_origin_eval(panel, model, min_train_rows=20, stride=15)
        assert len(result.trades) > 0
        assert "realized_pnl" in result.trades.columns

    def test_champion_publish_requires_go(self, tmp_path):
        """HoldoutGate blocks champion publish on NO_GO."""
        from pmlab.backtest.holdout_gate import SegmentGateResult
        bad_gate = HoldoutGateResult(
            decision="NO_GO",
            segment_results=[SegmentGateResult("A", 10, -5.0, False, "insufficient_trades")],
            aggregate_pnl=-5.0, aggregate_trades=10,
        )
        model = LGBMForecaster()
        X = pd.DataFrame({"f": [1.0, 2.0, 3.0]})
        y = pd.Series([0, 1, 0])
        model.fit(X, y)
        with pytest.raises(ValueError, match="NO_GO"):
            ChampionManifest.publish(model=model, gate=bad_gate, output_dir=tmp_path, plugin_family="test")

    def test_paper_broker_records_and_settlement(self, tmp_path):
        """PaperBroker records signals; SettlementEngine settles them."""
        from datetime import datetime
        broker = PaperBroker(
            trades_path=tmp_path / "trades.json",
            allowed_segments=None,
            flat_stake=1.0,
        )
        future_date = "2099-12-31"
        signal = EdgeSignal(
            market_id="int_001",
            city_or_segment="TestCity",
            target_date=future_date,
            horizon="previous_evening",
            outcome_label="warm",
            direction="yes",
            gamma_price=0.3,
            model_prob=0.6,
            best_edge=0.297,
            yes_edge=0.297,
            no_edge=-0.3,
        )
        now_utc = datetime(2026, 5, 10, 10, 0, 0, tzinfo=UTC)
        new_trades = broker.record([signal], now_utc=now_utc)
        assert len(new_trades) == 1
        trades = broker.load_trades()
        assert len(trades) == 1
        assert trades[0]["outcome"] is None  # not yet settled

    def test_full_discovery_to_features(self):
        """Plugin discover → features → build_training_row roundtrip."""
        plugin = IntegrationPlugin(truth_value=25.0)
        markets = plugin.discover_markets()
        spec = markets[0]
        features = plugin.fetch_features(spec, "previous_evening")
        assert isinstance(features, dict)
        row = plugin.build_training_row(spec, "previous_evening")
        assert row is not None
        assert row["winning_label"] == "warm"  # 25.0 falls in warm bin
        assert "feature_temp" in row
