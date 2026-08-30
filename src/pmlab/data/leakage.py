"""Leakage guards.

A feature must be computable from information available *at decision time*. A
column derived (directly or indirectly) from the resolved outcome will predict
the label almost perfectly on historical data and then fail live — the classic
silent trap. These guards score each ``feature_*`` column against the label and
flag any that separate it suspiciously well.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

__all__ = ["LeakageError", "LeakageReport", "check_no_leakage"]

_LABEL_COL = "winning_label"


class LeakageError(Exception):
    """Raised when a feature column separates the label too well to be legitimate."""


@dataclass
class LeakageReport:
    """Per-feature leakage diagnostics.

    Attributes:
        scores: feature column → separation score in [0.5, 1.0] (|AUC| folded so
            that a feature which perfectly predicts *either* class scores 1.0).
        leaked: feature columns whose score exceeds ``max_auc``.
        max_auc: the threshold used.
    """

    scores: dict[str, float] = field(default_factory=dict)
    leaked: list[str] = field(default_factory=list)
    max_auc: float = 0.97


def check_no_leakage(
    df: pd.DataFrame,
    label_col: str = _LABEL_COL,
    max_auc: float = 0.97,
    raise_on_leak: bool = True,
) -> LeakageReport:
    """Check that no ``feature_*`` column leaks the label.

    Each feature is scored by how well it alone separates the (binary) label,
    measured as ``max(auc, 1 - auc)`` so leakage toward either class is caught.
    Features scoring above ``max_auc`` are considered leaking.

    Args:
        df: Panel/trade frame containing ``feature_*`` columns and ``label_col``.
        label_col: Name of the outcome column (default ``winning_label``).
        max_auc: Separation score above which a feature is flagged (default 0.97).
        raise_on_leak: If True (default), raise ``LeakageError`` when any feature
            leaks; otherwise return the report for inspection.

    Returns:
        A :class:`LeakageReport`. When ``raise_on_leak`` is True and leakage is
        found, raises instead of returning.

    Raises:
        ValueError: if ``label_col`` is absent.
        LeakageError: if leakage is found and ``raise_on_leak`` is True.
    """
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame")

    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    report = LeakageReport(max_auc=max_auc)

    # Binary target from the label; AUC is undefined for a single class.
    y = df[label_col]
    classes = y.dropna().unique()
    if len(feature_cols) == 0 or len(classes) < 2:
        return report

    # Map the label to {0, 1}: the most frequent class is 0.
    positive = y.value_counts().index[-1]
    y_bin = (y == positive).astype(int).to_numpy()

    for col in feature_cols:
        x = pd.to_numeric(df[col], errors="coerce").to_numpy()
        mask = ~np.isnan(x)
        # Skip when too few valid points, when the label has one class in-mask,
        # or when the feature is constant (no separating power → cannot leak).
        if mask.sum() < 2 or len(np.unique(y_bin[mask])) < 2 or len(np.unique(x[mask])) < 2:
            continue
        auc = roc_auc_score(y_bin[mask], x[mask])
        score = float(max(auc, 1.0 - auc))
        report.scores[col] = score
        if score > max_auc:
            report.leaked.append(col)

    if report.leaked and raise_on_leak:
        details = ", ".join(f"{c} (score={report.scores[c]:.3f})" for c in report.leaked)
        raise LeakageError(
            f"Potential label leakage in feature column(s): {details}. "
            f"These separate '{label_col}' above max_auc={max_auc}. If legitimate, "
            f"raise max_auc or drop the column."
        )
    return report
