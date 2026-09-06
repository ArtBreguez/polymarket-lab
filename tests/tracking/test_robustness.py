"""Robustness tests found by dogfooding the tracker as a real ML user (pre-merge
adversarial review of v0.8.0).

The tracker is the generic logging layer: it must survive the value types that
actually come out of numpy/pandas in a backtest (np.int64, np.float64, NaN/Inf),
and it must not let an experiment name escape its root directory.
"""

from __future__ import annotations

import json

import numpy as np

from pmlab.tracking import LocalJSONTracker


def test_log_numpy_scalar_metrics(tmp_path):
    """np.int64/np.float64 are the NATIVE output of pandas aggregations."""
    tracker = LocalJSONTracker(root=tmp_path)
    with tracker.run(experiment="np", run_name="r") as run:
        run.log_metrics(
            {
                "num_trades": np.int64(42),
                "total_pnl": np.float64(2.5),
                "hit_rate": np.float32(0.75),
            }
        )
    rec = tracker.list_runs("np")[0]
    assert rec.metrics["num_trades"] == 42
    assert rec.metrics["total_pnl"] == 2.5
    # Round-trips as plain JSON numbers.
    raw = (tmp_path / "np" / f"{rec.run_id}.json").read_text()
    json.loads(raw)  # must not raise


def test_log_numpy_in_params_and_arrays(tmp_path):
    tracker = LocalJSONTracker(root=tmp_path)
    with tracker.run(experiment="np2", run_name="r") as run:
        run.log_params({"seed": np.int64(7), "weights": np.array([0.1, 0.2])})
    rec = tracker.list_runs("np2")[0]
    assert rec.params["seed"] == 7
    assert rec.params["weights"] == [0.1, 0.2]


def test_nan_and_inf_metrics_are_written_as_valid_json(tmp_path):
    """A backtest with 0 trades yields NaN hit_rate; the file must be spec-valid
    JSON so external tooling (JS/Rust/strict parsers) can read it."""
    tracker = LocalJSONTracker(root=tmp_path)
    with tracker.run(experiment="nan", run_name="r") as run:
        run.log_metrics({"hit_rate": float("nan"), "avg_edge": float("inf")})
    raw = (tmp_path / "nan").glob("*.json").__next__().read_text()

    # Strict parse: reject the JS-only NaN/Infinity literals.
    def _reject(x: str) -> object:
        raise ValueError(f"non-standard JSON constant: {x}")

    parsed = json.loads(raw, parse_constant=_reject)
    # NaN/Inf are represented as null (the conventional JSON-safe encoding).
    assert parsed["metrics"]["hit_rate"] is None
    assert parsed["metrics"]["avg_edge"] is None


def test_experiment_name_cannot_escape_root(tmp_path):
    tracker = LocalJSONTracker(root=tmp_path)
    with tracker.run(experiment="../escape", run_name="x") as run:
        run.log_params({"a": 1})
    # Nothing was written outside the root.
    assert not (tmp_path.parent / "escape").exists()
    # The run is still retrievable via the same (sanitized) name.
    assert tracker.list_runs("../escape")


def test_slash_in_experiment_name_is_sanitized(tmp_path):
    tracker = LocalJSONTracker(root=tmp_path)
    with tracker.run(experiment="weather/tmax", run_name="x") as run:
        run.log_metrics({"pnl": 1.0})
    runs = tracker.list_runs("weather/tmax")
    assert len(runs) == 1
    assert runs[0].metrics["pnl"] == 1.0
