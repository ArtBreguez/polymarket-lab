"""TDD: the backtest → metrics → gate lifecycle must compose without manual glue.

These are the frictions found while dogfooding pmlab as a real user: the trades
DataFrame produced by rolling_origin_eval could not be fed into compute_metrics
(missing 'outcome') or HoldoutGateResult.evaluate (missing 'segment'). RED first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pmlab.backtest.holdout_gate import HoldoutGateResult
from pmlab.backtest.metrics import compute_metrics
from pmlab.backtest.rolling_origin import rolling_origin_eval
from pmlab.modeling.sklearn_forecaster import SklearnForecaster


def _panel(n_dates: int = 40, per_date: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dates):
        date = f"2026-01-{d + 1:02d}"
        for _ in range(per_date):
            f = rng.normal(0, 1)
            win = "YES" if f + rng.normal(0, 0.4) > 0 else "NO"
            rows.append(
                {
                    "market_id": f"m{d}",
                    "decision_date": date,
                    "outcome_label": "YES",
                    "winning_label": win,
                    "market_price": 0.5,
                    "feature_f": f,
                }
            )
    return pd.DataFrame(rows)


class TestBacktestEmitsComposableColumns:
    def test_trades_have_outcome_column(self) -> None:
        result = rolling_origin_eval(
            _panel(),
            SklearnForecaster(estimator="random_forest"),
            min_train_rows=100,
            stride=5,
        )
        assert not result.trades.empty
        assert "outcome" in result.trades.columns
        # outcome is derived from realized_pnl sign
        won = result.trades["realized_pnl"] > 0
        assert (result.trades.loc[won, "outcome"] == "won").all()
        assert (result.trades.loc[~won, "outcome"] == "lost").all()

    def test_trades_have_segment_column(self) -> None:
        result = rolling_origin_eval(
            _panel(),
            SklearnForecaster(estimator="random_forest"),
            min_train_rows=100,
            stride=5,
        )
        assert "segment" in result.trades.columns
        # default segment is "all" when none is supplied
        assert (result.trades["segment"] == "all").all()

    def test_segment_column_passthrough_from_panel(self) -> None:
        panel = _panel()
        panel["segment"] = np.where(panel["feature_f"] > 0, "high", "low")
        result = rolling_origin_eval(
            panel,
            SklearnForecaster(estimator="random_forest"),
            min_train_rows=100,
            stride=5,
        )
        assert set(result.trades["segment"].unique()) <= {"high", "low"}


class TestLifecycleComposesEndToEnd:
    def test_backtest_feeds_compute_metrics_directly(self) -> None:
        # The friction: compute_metrics(result.trades) used to KeyError on 'outcome'.
        result = rolling_origin_eval(
            _panel(),
            SklearnForecaster(estimator="random_forest"),
            min_train_rows=100,
            stride=5,
        )
        m = compute_metrics(result.trades)  # must not raise
        assert m.num_trades == len(result.trades)
        assert 0.0 <= m.hit_rate <= 1.0

    def test_backtest_feeds_gate_directly(self) -> None:
        # The friction: no way to get a GO/NO_GO from backtest trades without a
        # 'segment' column and without manually building the dataclass.
        result = rolling_origin_eval(
            _panel(),
            SklearnForecaster(estimator="random_forest"),
            min_train_rows=100,
            stride=5,
        )
        gate = HoldoutGateResult.evaluate(result.trades)  # required_segments optional
        assert gate.decision in ("GO", "NO_GO")
        assert gate.aggregate_trades == len(result.trades)


class TestComputeMetricsBackwardCompatible:
    def test_derives_outcome_when_absent(self) -> None:
        df = pd.DataFrame({"realized_pnl": [0.5, -0.3, 0.2], "edge": [0.1, -0.1, 0.05]})
        m = compute_metrics(df)  # no 'outcome' column
        assert m.num_trades == 3
        assert m.hit_rate == pytest.approx(2 / 3)

    def test_still_honors_explicit_outcome(self) -> None:
        df = pd.DataFrame(
            {
                "realized_pnl": [0.5, -0.3, 0.2],
                "outcome": ["won", "lost", "lost"],  # explicit overrides sign
                "edge": [0.1, -0.1, 0.05],
            }
        )
        m = compute_metrics(df)
        assert m.hit_rate == pytest.approx(1 / 3)


class TestGateAggregateMode:
    def test_empty_required_segments_does_not_auto_pass(self) -> None:
        # Safety regression: evaluate(trades, required_segments=[]) must NOT
        # return GO on an empty segment list (that would let a losing model be
        # published — violates the champion hard-gate). Empty falls back to the
        # aggregate grade, so a -PnL log is correctly NO_GO.
        losing = pd.DataFrame({"realized_pnl": [-1.0] * 100, "segment": ["x"] * 100})
        gate = HoldoutGateResult.evaluate(losing, required_segments=[])
        assert gate.decision == "NO_GO"
        assert len(gate.segment_results) == 1  # aggregate, not zero segments

    def test_evaluate_without_segments_uses_aggregate(self) -> None:
        rows = [{"realized_pnl": 0.1, "outcome": "won"} for _ in range(50)]
        trades = pd.DataFrame(rows)
        gate = HoldoutGateResult.evaluate(trades, min_trades_per_segment=40)
        assert gate.decision == "GO"
        assert len(gate.segment_results) == 1
        assert gate.segment_results[0].segment == "all"

    def test_evaluate_aggregate_negative_pnl_is_nogo(self) -> None:
        rows = [{"realized_pnl": -0.1, "outcome": "lost"} for _ in range(50)]
        trades = pd.DataFrame(rows)
        gate = HoldoutGateResult.evaluate(trades, min_trades_per_segment=40)
        assert gate.decision == "NO_GO"
