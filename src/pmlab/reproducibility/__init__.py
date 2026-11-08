"""Reproducibility (v0.10.0): capture a run's seeds, versions, config and data; re-run it.

* :class:`ReproducibilityRecord` — what a run records about itself. Stored in the experiment
  tracker's run file (``ActiveRun.log_reproducibility``), in ``champion.json``
  (``ChampionManifest.reproducibility``), in the model registry, and in the metrics file
  ``pmlab backtest --output`` writes.
* :func:`reproduce` / ``pmlab reproduce <run>`` — re-execute the recorded run and verify that
  the same metrics come back.

See ``docs/reproducibility.md``.
"""

from __future__ import annotations

from pmlab.reproducibility.capture import (
    SCHEMA_VERSION,
    DataFingerprint,
    EnvDrift,
    ReproducibilityRecord,
    capture_environment,
    environment_drift,
    fingerprint_panel,
    seed_everything,
    seeded,
)
from pmlab.reproducibility.reproduce import (
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    DataMismatchError,
    MetricComparison,
    ReproduceError,
    ReproduceReport,
    compare_metrics,
    load_record,
    reproduce,
)
from pmlab.reproducibility.runner import (
    DEFAULT_SEED,
    BacktestConfig,
    BacktestOutcome,
    backtest_metrics,
    build_forecaster,
    capture_backtest,
    load_panel,
    resolve_seeds,
    run_backtest,
)

__all__ = [
    "DEFAULT_ATOL",
    "DEFAULT_RTOL",
    "DEFAULT_SEED",
    "SCHEMA_VERSION",
    "BacktestConfig",
    "BacktestOutcome",
    "DataFingerprint",
    "DataMismatchError",
    "EnvDrift",
    "MetricComparison",
    "ReproducibilityRecord",
    "ReproduceError",
    "ReproduceReport",
    "backtest_metrics",
    "build_forecaster",
    "capture_backtest",
    "capture_environment",
    "compare_metrics",
    "environment_drift",
    "fingerprint_panel",
    "load_panel",
    "load_record",
    "reproduce",
    "resolve_seeds",
    "run_backtest",
    "seed_everything",
    "seeded",
]
