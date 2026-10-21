"""Settled PnL must carry the same slippage the backtest charged.

`PaperBroker` fills at the costed price and records it as `fill_price`, but
`SettlementEngine` used to recompute the entry from `gamma_price`: a trade that
paid 50bps of slippage settled as if it had paid none. Live PnL then read better
than the backtest's by exactly the friction the cost model was meant to share,
which is the divergence the unified cost model exists to remove. The watchdog
(#14) compares the two, so it would have been biased toward green.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pmlab.backtest.rolling_origin import rolling_origin_eval
from pmlab.core.costs import CostModel
from pmlab.core.market_spec import MarketSpec, OutcomeBin
from pmlab.execution.edge_signal import EdgeSignal
from pmlab.execution.paper_broker import PaperBroker
from pmlab.execution.settlement import SettlementEngine
from pmlab.plugins.base import MarketPlugin

_NOW = datetime(2026, 6, 15, 5, 0, tzinfo=UTC)
_COSTS = CostModel(
    taker_bps=30.0,
    slippage_bps=75.0,
    slippage_fixed=0.002,
    depth_bps_per_unit=40.0,
    depth_reference_stake=10.0,
)


class _Truth(MarketPlugin):
    family = "dummy"

    def __init__(self, truth: str) -> None:
        self._truth = truth

    def discover_markets(self, **kwargs: Any) -> list[MarketSpec]:
        return []

    def fetch_features(self, spec: MarketSpec, horizon: str, **kwargs: Any) -> dict[str, float]:
        return {}

    def fetch_truth(self, spec: MarketSpec, **kwargs: Any) -> str | None:
        return self._truth

    def build_training_row(
        self, spec: MarketSpec, horizon: str, **kwargs: Any
    ) -> dict[str, Any] | None:
        return None


class _Flat:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:  # noqa: N803
        return None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        n = len(X)
        return np.column_stack([np.full(n, 0.2), np.full(n, 0.8)])


def _panel(price: float, winning: str) -> pd.DataFrame:
    rows = []
    for d in range(10):
        for label in ("YES", "NO"):
            rows.append(
                {
                    "market_id": "mkt1",
                    "decision_date": f"2026-06-{d + 1:02d}",
                    "outcome_label": label,
                    "winning_label": winning,
                    "market_price": price,
                    "feature_x": 1.0 if label == "YES" else 0.0,
                    "segment": "politics",
                }
            )
    return pd.DataFrame(rows)


def _signal(price: float) -> EdgeSignal:
    return EdgeSignal(
        market_id="mkt1",
        city_or_segment="politics",
        target_date="2026-06-15",
        horizon="morning_of",
        outcome_label="YES",
        direction="yes",
        gamma_price=price,
        model_prob=0.8,
        best_edge=0.2,
        yes_edge=0.2,
        no_edge=-0.2,
    )


def _spec() -> MarketSpec:
    return MarketSpec(
        market_id="mkt1",
        slug="s",
        question="q",
        outcome_bins=[OutcomeBin(label="YES"), OutcomeBin(label="NO")],
        close_time="2026-06-15T20:00:00Z",
        market_family="binary",
        metadata={"city": "politics", "target_date": "2026-06-15"},
    )


def _settle(path: Path, winning: str) -> dict[str, Any]:
    engine = SettlementEngine(plugin=_Truth(winning), trades_path=path)
    engine.settle_all(specs=[_spec()], today_str="2026-06-16")
    trade: dict[str, Any] = json.loads(path.read_text())["trades"][0]
    return trade


class TestSettledPnlCarriesTheSlippage:
    @pytest.mark.parametrize("winning", ["YES", "NO"])
    @pytest.mark.parametrize("price", [0.10, 0.40, 0.90])
    @pytest.mark.parametrize("stake", [1.0, 10.0])
    def test_live_settled_pnl_equals_backtest_pnl(
        self, tmp_path: Path, price: float, winning: str, stake: float
    ) -> None:
        backtest = rolling_origin_eval(
            _panel(price, winning),
            _Flat(),
            min_train_rows=4,
            stride=2,
            flat_stake=stake,
            costs=_COSTS,
        )
        path = tmp_path / "t.json"
        PaperBroker(trades_path=path, flat_stake=stake, costs=_COSTS).record(
            [_signal(price)], city_timezones={"politics": "UTC"}, now_utc=_NOW
        )

        live = _settle(path, winning)

        assert live["realized_pnl"] == pytest.approx(
            float(backtest.trades.iloc[0]["realized_pnl"]), abs=1e-5
        )

    def test_slippage_makes_the_settled_pnl_worse_than_a_frictionless_fill(
        self, tmp_path: Path
    ) -> None:
        paths = {"free": tmp_path / "free.json", "costly": tmp_path / "costly.json"}
        PaperBroker(
            trades_path=paths["free"], flat_stake=10.0, costs=CostModel(taker_bps=30.0)
        ).record([_signal(0.40)], city_timezones={"politics": "UTC"}, now_utc=_NOW)
        PaperBroker(trades_path=paths["costly"], flat_stake=10.0, costs=_COSTS).record(
            [_signal(0.40)], city_timezones={"politics": "UTC"}, now_utc=_NOW
        )

        assert (
            _settle(paths["costly"], "YES")["realized_pnl"]
            < _settle(paths["free"], "YES")["realized_pnl"]
        )

    def test_a_trade_without_a_recorded_fill_settles_as_before(self, tmp_path: Path) -> None:
        """Logs written before fill_price existed must settle to the same number."""
        path = tmp_path / "old.json"
        PaperBroker(trades_path=path, flat_stake=10.0, costs=CostModel(taker_bps=30.0)).record(
            [_signal(0.40)], city_timezones={"politics": "UTC"}, now_utc=_NOW
        )
        trades = json.loads(path.read_text())["trades"]
        for t in trades:
            del t["fill_price"]
        path.write_text(json.dumps({"trades": trades}))

        settled = _settle(path, "YES")

        # (1 - 0.40) * (10 / 0.40) - fee
        assert settled["realized_pnl"] == pytest.approx(15.0 - 10.0 * 30.0 / 10_000.0)
