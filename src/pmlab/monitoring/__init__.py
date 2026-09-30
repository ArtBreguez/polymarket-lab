"""Live monitoring of a deployed champion.

The backtest gate answers "was this model good on history". This package answers
"is it still good on money at risk".
"""

from pmlab.monitoring.calibration_tracker import CalibrationTracker, CalibrationWindow

__all__ = ["CalibrationTracker", "CalibrationWindow"]
