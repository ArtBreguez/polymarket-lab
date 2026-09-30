"""Tests for CostModel unified across the backtest and the live path.

v0.8.3 gave the backtest a cost model; `PaperBroker` kept filling at the quote
with a fee-only adjustment. Two cost definitions for the same trade means the
v0.9.0 watchdog would compare live PnL against backtest PnL without knowing which
difference is the market and which is our own arithmetic.

The deliverable is an equivalence: the same CostModel, the same quote and the same
stake must produce the same fill on both sides.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pmlab.backtest.rolling_origin import rolling_origin_eval
from pmlab.core.costs import CostModel
from pmlab.execution.edge_signal import EdgeSignal
from pmlab.execution.paper_broker import PaperBroker

# Before the morning_of cutoff for target_date 2026-06-15, so signals are fresh.
_NOW = datetime(2026, 6, 15, 5, 0, tzinfo=UTC)


def _signal(price: float = 0.40, direction: str = "yes") -> EdgeSignal:
    return EdgeSignal(
        market_id="mkt1",
        city_or_segment="politics",
        target_date="2026-06-15",
        horizon="morning_of",
        outcome_label="YES",
        direction=direction,
        gamma_price=price,
        model_prob=0.55,
        best_edge=0.12,
        yes_edge=0.12,
        no_edge=-0.12,
    )


class _Flat:
    """Deterministic forecaster so the fill is the only variable."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:  # noqa: N803
        return None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        n = len(X)
        return np.column_stack([np.full(n, 0.2), np.full(n, 0.8)])


def _panel(price: float, n_dates: int = 10) -> pd.DataFrame:
    rows = []
    for d in range(n_dates):
        for label in ("YES", "NO"):
            rows.append(
                {
                    "market_id": "mkt1",
                    "decision_date": f"2026-06-{d + 1:02d}",
                    "outcome_label": label,
                    "winning_label": "YES",
                    "market_price": price,
                    "feature_x": 1.0 if label == "YES" else 0.0,
                    "segment": "politics",
                }
            )
    return pd.DataFrame(rows)


class TestBackwardCompatibility:
    """A user upgrading must see identical paper trades."""

    def test_default_paper_broker_matches_fee_only_behaviour(self, tmp_path: Path) -> None:
        broker = PaperBroker(trades_path=tmp_path / "t.json", flat_stake=10.0, taker_bps=30.0)
        added = broker.record([_signal(0.40)], city_timezones={"politics": "UTC"}, now_utc=_NOW)
        assert len(added) == 1
        trade = added[0]
        # Pre-existing contract: entry at the quote, fee on notional.
        assert trade["gamma_price"] == pytest.approx(0.40)
        assert trade["fee_paid"] == pytest.approx(10.0 * 30.0 / 10_000.0)
        assert trade["size"] == pytest.approx(10.0 / 0.40)

    def test_explicit_default_cost_model_is_the_same(self, tmp_path: Path) -> None:
        legacy = PaperBroker(trades_path=tmp_path / "a.json", flat_stake=10.0, taker_bps=30.0)
        costed = PaperBroker(
            trades_path=tmp_path / "b.json", flat_stake=10.0, costs=CostModel(taker_bps=30.0)
        )
        a = legacy.record([_signal(0.40)], city_timezones={"politics": "UTC"}, now_utc=_NOW)[0]
        b = costed.record([_signal(0.40)], city_timezones={"politics": "UTC"}, now_utc=_NOW)[0]
        for field in ("gamma_price", "fill_price", "size", "fee_paid"):
            assert a[field] == pytest.approx(b[field]), f"{field} diverged"

    def test_both_taker_bps_and_costs_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="taker_bps"):
            PaperBroker(
                trades_path=tmp_path / "t.json",
                taker_bps=50.0,
                costs=CostModel(taker_bps=10.0),
            )


class TestFillPriceOnTheLivePath:
    """The live side must pay the spread too, and record what it paid."""

    def test_trade_records_fill_price_alongside_the_quote(self, tmp_path: Path) -> None:
        broker = PaperBroker(
            trades_path=tmp_path / "t.json",
            flat_stake=10.0,
            costs=CostModel(taker_bps=30.0, slippage_bps=200.0),
        )
        trade = broker.record([_signal(0.40)], city_timezones={"politics": "UTC"}, now_utc=_NOW)[0]
        assert trade["gamma_price"] == pytest.approx(0.40), "quote must be preserved"
        assert trade["fill_price"] > trade["gamma_price"], "fill must be worse than the quote"

    def test_size_is_computed_from_the_fill_not_the_quote(self, tmp_path: Path) -> None:
        """Sizing off the quote buys shares you could not have bought."""
        broker = PaperBroker(
            trades_path=tmp_path / "t.json",
            flat_stake=10.0,
            costs=CostModel(taker_bps=0.0, slippage_bps=500.0),
        )
        trade = broker.record([_signal(0.40)], city_timezones={"politics": "UTC"}, now_utc=_NOW)[0]
        assert trade["size"] == pytest.approx(10.0 / trade["fill_price"])
        assert trade["size"] < 10.0 / 0.40

    def test_no_direction_fills_against_the_complement(self, tmp_path: Path) -> None:
        """Buying NO means paying 1 - quote, and slippage still works against you."""
        broker = PaperBroker(
            trades_path=tmp_path / "t.json",
            flat_stake=10.0,
            costs=CostModel(taker_bps=0.0, slippage_bps=300.0),
        )
        trade = broker.record(
            [_signal(0.40, direction="no")], city_timezones={"politics": "UTC"}, now_utc=_NOW
        )[0]
        # entry on the NO side is 0.60 before costs
        assert trade["fill_price"] > 0.60

    def test_depth_penalty_applies_to_the_live_stake(self, tmp_path: Path) -> None:
        costs = CostModel(taker_bps=0.0, depth_bps_per_unit=100.0, depth_reference_stake=5.0)
        small = PaperBroker(trades_path=tmp_path / "s.json", flat_stake=5.0, costs=costs)
        large = PaperBroker(trades_path=tmp_path / "l.json", flat_stake=50.0, costs=costs)
        s = small.record([_signal(0.40)], city_timezones={"politics": "UTC"}, now_utc=_NOW)[0]
        loud = large.record([_signal(0.40)], city_timezones={"politics": "UTC"}, now_utc=_NOW)[0]
        assert loud["fill_price"] > s["fill_price"]


class TestTheEquivalence:
    """The actual point of this change."""

    @pytest.mark.parametrize("price", [0.10, 0.40, 0.55, 0.90])
    @pytest.mark.parametrize("stake", [1.0, 10.0, 100.0])
    def test_backtest_and_paper_broker_fill_identically(
        self, tmp_path: Path, price: float, stake: float
    ) -> None:
        costs = CostModel(
            taker_bps=30.0,
            slippage_bps=75.0,
            slippage_fixed=0.002,
            depth_bps_per_unit=40.0,
            depth_reference_stake=10.0,
        )
        backtest = rolling_origin_eval(
            _panel(price), _Flat(), min_train_rows=4, stride=2, flat_stake=stake, costs=costs
        )
        broker = PaperBroker(
            trades_path=tmp_path / f"t_{price}_{stake}.json", flat_stake=stake, costs=costs
        )
        live = broker.record([_signal(price)], city_timezones={"politics": "UTC"}, now_utc=_NOW)[0]

        backtest_fill = float(backtest.trades.iloc[0]["fill_price"])
        assert live["fill_price"] == pytest.approx(backtest_fill), (
            f"live and backtest disagree at price={price} stake={stake}: "
            f"{live['fill_price']} vs {backtest_fill}"
        )

    def test_fees_agree_too(self, tmp_path: Path) -> None:
        costs = CostModel(taker_bps=42.0)
        backtest = rolling_origin_eval(
            _panel(0.40), _Flat(), min_train_rows=4, stride=2, flat_stake=25.0, costs=costs
        )
        broker = PaperBroker(trades_path=tmp_path / "t.json", flat_stake=25.0, costs=costs)
        live = broker.record([_signal(0.40)], city_timezones={"politics": "UTC"}, now_utc=_NOW)[0]
        # rolling_origin does not expose fee per trade, so recompute from the model
        assert live["fee_paid"] == pytest.approx(costs.fee(25.0))
        assert len(backtest.trades) > 0


class TestChampionManifestCarriesTheCostModel:
    """Without this the two sides can silently drift apart again."""

    def test_publish_records_the_cost_model(self, tmp_path: Path) -> None:
        from pmlab.modeling.champion import ChampionManifest

        gate, model = _passing_gate_and_model()
        costs = CostModel(taker_bps=30.0, slippage_bps=60.0)
        ChampionManifest.publish(
            model=model,
            gate=gate,
            output_dir=tmp_path,
            plugin_family="weather_tmax",
            costs=costs,
        )
        data = json.loads((tmp_path / "champion.json").read_text())
        assert data["costs"] == costs.to_dict()

    def test_load_round_trips_the_cost_model(self, tmp_path: Path) -> None:
        from pmlab.modeling.champion import ChampionManifest

        gate, model = _passing_gate_and_model()
        costs = CostModel(slippage_bps=15.0, depth_bps_per_unit=5.0)
        ChampionManifest.publish(
            model=model,
            gate=gate,
            output_dir=tmp_path,
            plugin_family="weather_tmax",
            costs=costs,
        )
        loaded = ChampionManifest.load(tmp_path / "champion.json")
        assert loaded.costs == costs

    def test_a_manifest_without_costs_still_loads(self, tmp_path: Path) -> None:
        """Champions published before this change must not become unreadable."""
        from pmlab.modeling.champion import ChampionManifest

        gate, model = _passing_gate_and_model()
        ChampionManifest.publish(
            model=model, gate=gate, output_dir=tmp_path, plugin_family="weather_tmax"
        )
        data = json.loads((tmp_path / "champion.json").read_text())
        data.pop("costs", None)  # simulate an old manifest
        (tmp_path / "champion.json").write_text(json.dumps(data))

        loaded = ChampionManifest.load(tmp_path / "champion.json")
        assert loaded.costs is None


class TestRegistryPreservesTheCostModel:
    """A rollback must restore the friction the gate was computed under.

    Found by mutation testing: `ModelRegistry.record()` rebuilds champion.json
    field by field rather than copying it, so a new manifest field is silently
    dropped on archive unless it is added here too.
    """

    def test_recorded_version_keeps_the_cost_model(self, tmp_path: Path) -> None:
        from pmlab.modeling.champion import ChampionManifest
        from pmlab.modeling.registry import ModelRegistry

        gate, model = _passing_gate_and_model()
        costs = CostModel(taker_bps=30.0, slippage_bps=80.0, depth_bps_per_unit=25.0)
        manifest = ChampionManifest.publish(
            model=model,
            gate=gate,
            output_dir=tmp_path / "live",
            plugin_family="weather_tmax",
            costs=costs,
        )
        registry = ModelRegistry(registry_dir=tmp_path / "registry")
        version_id = registry.record(manifest)

        restored = registry.get(version_id)
        assert restored.costs == costs, "archived version lost its cost model"

    def test_version_without_costs_is_still_retrievable(self, tmp_path: Path) -> None:
        from pmlab.modeling.champion import ChampionManifest
        from pmlab.modeling.registry import ModelRegistry

        gate, model = _passing_gate_and_model()
        manifest = ChampionManifest.publish(
            model=model, gate=gate, output_dir=tmp_path / "live", plugin_family="weather_tmax"
        )
        registry = ModelRegistry(registry_dir=tmp_path / "registry")
        version_id = registry.record(manifest)
        assert registry.get(version_id).costs is None


def _passing_gate_and_model():  # type: ignore[no-untyped-def]
    """A GO gate and a fitted model, the minimum publish() accepts."""
    from pmlab.backtest.holdout_gate import HoldoutGateResult
    from pmlab.modeling.lgbm_baseline import LGBMForecaster

    trades = pd.DataFrame(
        {
            "realized_pnl": [1.0] * 50,
            "outcome": ["won"] * 50,
            "segment": ["politics"] * 50,
            "edge": [0.05] * 50,
        }
    )
    gate = HoldoutGateResult.evaluate(trades, required_segments=["politics"])
    assert gate.decision == "GO", "test fixture must produce a GO gate"

    X = pd.DataFrame({"f": np.arange(40, dtype=float)})
    y = pd.Series([0, 1] * 20)
    model = LGBMForecaster()
    model.fit(X, y)
    return gate, model
