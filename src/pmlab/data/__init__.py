"""Data & feature-engineering layer: point-in-time snapshots, panel building,
and leakage guards — the reproducibility backbone for pmlab."""

from __future__ import annotations

from pmlab.data.leakage import LeakageError, LeakageReport, check_no_leakage
from pmlab.data.panel_builder import PanelSchemaError, build_panel
from pmlab.data.snapshot_panel import build_panel_from_snapshots
from pmlab.data.snapshot_store import FeatureSnapshotStore

__all__ = [
    "FeatureSnapshotStore",
    "LeakageError",
    "LeakageReport",
    "check_no_leakage",
    "PanelSchemaError",
    "build_panel",
    "build_panel_from_snapshots",
]
