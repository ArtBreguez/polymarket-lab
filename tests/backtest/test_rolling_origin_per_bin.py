"""A per-bin panel (K rows per market) must not make a market count K times (issue #28).

Everything downstream of `rolling_origin_eval` counts *trades*, and the policies emit one
trade per (market_id, decision_date). These tests hold that at K=5 for every consumer that
reports a per-market number: the trade log, `compute_metrics`, `HoldoutGateResult`, and the
CLI's `Panel: ... markets` line.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from pmlab.backtest.holdout_gate import HoldoutGateResult
from pmlab.backtest.metrics import compute_metrics
from pmlab.backtest.rolling_origin import rolling_origin_eval
from pmlab.cli.main import app
from pmlab.core.costs import CostModel
from pmlab.core.policy import TradePolicy
from pmlab.modeling.lgbm_baseline import LGBMForecaster
from tests.plugins.weather_tmax import synthetic_world as sw

K = len(sw.LABELS)
POLICIES = {
    "live": TradePolicy.live(min_edge=0.05),
    "top-bin-yes": TradePolicy.top_bin_yes(),
}


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return pd.DataFrame(sw.reference_panel_rows(sw.make_world(n_days=40, seed=3)))


def _trades(panel: pd.DataFrame, policy: TradePolicy, stride: int = 5) -> pd.DataFrame:
    return rolling_origin_eval(
        panel,
        LGBMForecaster(),
        min_train_rows=20,
        stride=stride,
        costs=CostModel.frictionless(),
        policy=policy,
    ).trades


@pytest.mark.parametrize("policy_name", list(POLICIES))
class TestAMarketIsOneTradeNotK:
    def test_at_most_one_trade_per_market_and_date(
        self, panel: pd.DataFrame, policy_name: str
    ) -> None:
        trades = _trades(panel, POLICIES[policy_name])
        assert len(trades) > 0
        assert trades.groupby(["market_id", "eval_date"]).size().max() == 1

    def test_the_trade_count_is_bounded_by_markets_not_rows(
        self, panel: pd.DataFrame, policy_name: str
    ) -> None:
        trades = _trades(panel, POLICIES[policy_name])
        evaluated = trades["eval_date"].unique()
        markets_on_those_dates = panel.loc[panel["decision_date"].isin(evaluated), "market_id"]
        assert len(trades) <= markets_on_those_dates.nunique()

    def test_the_gate_counts_trades_per_segment(
        self, panel: pd.DataFrame, policy_name: str
    ) -> None:
        seg = panel.copy()
        seg["segment"] = (
            seg["market_id"].str[-1].map(lambda c: "even" if int(c) % 2 == 0 else "odd")
        )
        trades = _trades(seg, POLICIES[policy_name])
        gate = HoldoutGateResult.evaluate(
            trades, required_segments=["even", "odd"], min_trades_per_segment=1
        )
        by_seg = {r.segment: r.num_trades for r in gate.segment_results}
        for name in ("even", "odd"):
            distinct = trades.loc[trades["segment"] == name, ["market_id", "eval_date"]]
            assert by_seg[name] == len(distinct.drop_duplicates()) > 0
        assert gate.aggregate_trades == len(trades)

    def test_metrics_are_per_trade(self, panel: pd.DataFrame, policy_name: str) -> None:
        trades = _trades(panel, POLICIES[policy_name])
        m = compute_metrics(trades)
        assert m.num_trades == len(trades) < len(panel)
        assert m.hit_rate == pytest.approx((trades["realized_pnl"] > 0).mean())


class TestCliCountsMarketsNotRows:
    def test_the_panel_line_reports_distinct_markets(
        self, panel: pd.DataFrame, tmp_path: Path
    ) -> None:
        path = tmp_path / "p.parquet"
        panel.to_parquet(path)
        result = CliRunner().invoke(
            app,
            ["backtest", "--plugin", "weather_tmax", "--panel", str(path), "--stride", "10"],
        )
        markets = panel["market_id"].nunique()
        dates = panel["decision_date"].nunique()
        assert f"Panel: {len(panel)} rows, {dates} dates, {markets} markets" in " ".join(
            result.output.split()
        )
