"""Live monitoring of a deployed champion.

The backtest gate answers "was this model good on history". This package answers
"is it still good on money at risk" (`CalibrationTracker`) and "is the world it is
seeing still the one it was trained on" (`drift_report`). `health_report` weighs both,
and the champion's backtest, into one verdict a scheduler can act on.
"""

from pmlab.monitoring.calibration_tracker import CalibrationTracker, CalibrationWindow
from pmlab.monitoring.drift import (
    DriftConfig,
    DriftReport,
    DriftThresholds,
    FeatureDrift,
    drift_report,
    drift_report_from_store,
)
from pmlab.monitoring.watchdog import (
    CalibrationThresholds,
    CheckResult,
    HealthReport,
    PnLThresholds,
    WatchdogConfig,
    WatchdogInputError,
    health_report,
)

__all__ = [
    "CalibrationThresholds",
    "CalibrationTracker",
    "CalibrationWindow",
    "CheckResult",
    "DriftConfig",
    "DriftReport",
    "DriftThresholds",
    "FeatureDrift",
    "HealthReport",
    "PnLThresholds",
    "WatchdogConfig",
    "WatchdogInputError",
    "drift_report",
    "drift_report_from_store",
    "health_report",
]
