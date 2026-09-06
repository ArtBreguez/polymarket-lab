# Experiment Tracking — pmlab.tracking

Make every backtest **auditable and reproducible**: log the params, metrics, and
gate decision of each run so a result is never a number you can't re-explain.

Two backends share one `ExperimentTracker` protocol:

| Backend | Dependency | Use it for |
|---|---|---|
| `LocalJSONTracker` | none (default) | append-only JSON records on disk, zero setup |
| `MLflowTracker` | `pmlab[track]` extra | teams already on MLflow, richer UI |

---

## Basic Usage

```python
from pmlab import LocalJSONTracker
from pmlab.backtest.rolling_origin import rolling_origin_eval
from pmlab.backtest.metrics import compute_metrics
from pmlab.backtest.holdout_gate import HoldoutGateResult

tracker = LocalJSONTracker(root="runs")

with tracker.run(experiment="weather_tmax", run_name="rf_baseline") as run:
    run.log_params({"model": "random_forest", "stride": 5, "taker_bps": 30.0})

    result = rolling_origin_eval(panel, model, stride=5, taker_bps=30.0)
    metrics = compute_metrics(result.trades)
    gate = HoldoutGateResult.evaluate(result.trades, required_segments=["all"])

    run.log_metrics(metrics)   # BacktestMetrics logs directly — no manual glue
    run.log_gate(gate)         # HoldoutGateResult too
```

`log_metrics` accepts a plain `dict`, a `BacktestMetrics` (any dataclass), or any
object with a `to_dict()`. `log_gate` accepts a `HoldoutGateResult`. On exit the
run is flushed with a `finished` status — or `failed` (plus the error) if the body
raised, so partial runs are never silently lost.

---

## Reading runs back

```python
runs = tracker.list_runs("weather_tmax")   # newest first, typed RunRecord
best = max(runs, key=lambda r: r.metrics["total_pnl"])
print(best.run_id, best.params, best.gate["decision"])

one = tracker.get_run("weather_tmax", best.run_id)   # point-in-time load
```

Each run is one immutable JSON file at `runs/<experiment>/<run_id>.json`. A new
run never overwrites an old one (unique `run_id`), so the history is stable
evidence — the same append-only, point-in-time philosophy as `FeatureSnapshotStore`.

---

## MLflow backend (optional)

```bash
uv sync --extra track     # or: pip install pmlab[track]
```

```python
from pmlab.tracking.mlflow_tracker import MLflowTracker

tracker = MLflowTracker(tracking_uri="file:./mlruns")  # or a remote server
# same run(...) / log_params / log_metrics / log_gate / list_runs API
```

The module always imports safely; MLflow is only needed when you construct
`MLflowTracker`. Without the extra, constructing one raises an actionable
`ImportError` telling you to install `pmlab[track]`.

---

## API

| Symbol | Role |
|---|---|
| `ExperimentTracker` | Protocol: `run`, `list_runs`, `get_run` |
| `LocalJSONTracker` | Zero-dependency JSON backend (default) |
| `MLflowTracker` | Optional MLflow backend (`pmlab[track]`) |
| `ActiveRun` | Live run handle: `log_params`, `log_metrics`, `log_gate` |
| `RunRecord` | Immutable typed record of a completed run |
