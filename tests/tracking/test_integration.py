"""Integration: a real backtest logs params, metrics, and the gate into the
tracker with no manual glue — the objects produced by the lifecycle
(BacktestMetrics, HoldoutGateResult) are accepted directly.

This is the v0.8.0 value proposition: every backtest run becomes auditable and
reproducible evidence on disk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pmlab.backtest.holdout_gate import HoldoutGateResult
from pmlab.backtest.metrics import compute_metrics
from pmlab.backtest.rolling_origin import rolling_origin_eval
from pmlab.modeling.sklearn_forecaster import SklearnForecaster
from pmlab.tracking import LocalJSONTracker


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


def test_backtest_run_is_fully_tracked(tmp_path):
    tracker = LocalJSONTracker(root=tmp_path / "runs")

    stride, taker_bps = 5, 30.0
    with tracker.run(experiment="weather_tmax", run_name="rf_baseline") as run:
        run.log_params({"model": "random_forest", "stride": stride, "taker_bps": taker_bps})
        result = rolling_origin_eval(
            _panel(),
            SklearnForecaster(estimator="random_forest"),
            min_train_rows=100,
            stride=stride,
            taker_bps=taker_bps,
        )
        metrics = compute_metrics(result.trades)  # BacktestMetrics
        gate = HoldoutGateResult.evaluate(
            result.trades, required_segments=None, min_trades_per_segment=1
        )
        run.log_metrics(metrics)  # logged directly, no manual glue
        run.log_gate(gate)
        rid = run.run_id

    # Reload from disk — the run is complete, auditable evidence.
    loaded = tracker.get_run("weather_tmax", rid)
    assert loaded is not None
    assert loaded.status == "finished"
    assert loaded.params["model"] == "random_forest"
    assert loaded.params["stride"] == stride
    assert loaded.metrics["num_trades"] == metrics.num_trades
    assert loaded.metrics["total_pnl"] == metrics.total_pnl
    assert loaded.gate["decision"] in {"GO", "NO_GO"}
    assert loaded.gate["decision"] == gate.decision
