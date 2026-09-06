"""Tests for the optional MLflow tracking backend.

The whole point is that MLflow stays optional: importing the module is always
safe, and the tracker only needs MLflow when actually constructed. When the
'track' extra is installed, MLflowTracker must satisfy the same ExperimentTracker
protocol and round-trip a run.
"""

from __future__ import annotations

import importlib.util

import pytest

from pmlab.tracking import ExperimentTracker
from pmlab.tracking.mlflow_tracker import MLflowTracker

_HAS_MLFLOW = importlib.util.find_spec("mlflow") is not None


def test_module_imports_without_mlflow_installed():
    # Importing the module (done at top) must never require mlflow.
    assert MLflowTracker is not None


@pytest.mark.skipif(_HAS_MLFLOW, reason="mlflow IS installed; error path not applicable")
def test_construct_without_extra_raises_actionable_error():
    with pytest.raises(ImportError, match="track"):
        MLflowTracker()


@pytest.mark.skipif(not _HAS_MLFLOW, reason="requires the 'track' extra (mlflow)")
def test_mlflow_tracker_round_trip(tmp_path):
    tracker = MLflowTracker(tracking_uri=f"file://{tmp_path}/mlruns")
    assert isinstance(tracker, ExperimentTracker)

    with tracker.run(experiment="exp_mlflow", run_name="r") as run:
        run.log_params({"model": "lgbm", "stride": 10})
        run.log_metrics({"total_pnl": 2.0, "hit_rate": 0.5})

    runs = tracker.list_runs("exp_mlflow")
    assert len(runs) == 1
    assert runs[0].metrics["total_pnl"] == 2.0
    assert runs[0].params["model"] == "lgbm"
