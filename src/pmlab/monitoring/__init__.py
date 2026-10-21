"""Live monitoring of a deployed champion.

The backtest gate answers "was this model good on history". This package answers
"is it still good on money at risk" (`CalibrationTracker`) and "is the world it is
seeing still the one it was trained on" (`drift_report`).
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

__all__ = [
    "CalibrationTracker",
    "CalibrationWindow",
    "DriftConfig",
    "DriftReport",
    "DriftThresholds",
    "FeatureDrift",
    "drift_report",
    "drift_report_from_store",
]
