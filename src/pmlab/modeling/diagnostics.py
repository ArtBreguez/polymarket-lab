"""Calibration diagnostics: Brier score decomposition and reliability data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "BrierDecomposition",
    "brier_decomposition",
    "reliability_data",
    "MulticlassBrier",
    "multiclass_brier",
    "reliability_data_multiclass",
]


@dataclass
class BrierDecomposition:
    uncertainty: float
    resolution: float
    reliability: float
    brier_score: float
    skill_score: float
    n_samples: int


def brier_decomposition(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    n_bins: int = 10,
) -> BrierDecomposition:
    """Murphy (1973) Brier score decomposition."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_prob, dtype=float)
    if len(yt) == 0:
        raise ValueError("Empty arrays - cannot compute Brier decomposition")
    if len(yt) != len(yp):
        raise ValueError("y_true and y_prob must have the same length")
    n = len(yt)
    clim = float(yt.mean())
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.clip(np.digitize(yp, bins) - 1, 0, n_bins - 1)
    reliability_term = 0.0
    resolution_term = 0.0
    for k in range(n_bins):
        mask = bin_indices == k
        nk = int(mask.sum())
        if nk == 0:
            continue
        ok = float(yt[mask].mean())
        fk = float(yp[mask].mean())
        reliability_term += nk * (fk - ok) ** 2
        resolution_term += nk * (ok - clim) ** 2
    reliability_term /= n
    resolution_term /= n
    uncertainty_term = clim * (1.0 - clim)
    brier = float(np.mean((yp - yt) ** 2))
    bs_clim = uncertainty_term
    skill = 1.0 - brier / bs_clim if bs_clim > 0 else 0.0
    return BrierDecomposition(
        uncertainty=float(uncertainty_term),
        resolution=float(resolution_term),
        reliability=float(reliability_term),
        brier_score=brier,
        skill_score=float(skill),
        n_samples=n,
    )


def reliability_data(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    n_bins: int = 10,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Reliability diagram data: (bin_centers, mean_predicted_prob, fraction_positive)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.clip(np.digitize(yp, bins) - 1, 0, n_bins - 1)
    bin_centers_list: list[float] = []
    mean_pred_list: list[float] = []
    frac_pos_list: list[float] = []
    for k in range(n_bins):
        mask = bin_indices == k
        if mask.sum() == 0:
            continue
        bin_centers_list.append(float((bins[k] + bins[k + 1]) / 2.0))
        mean_pred_list.append(float(yp[mask].mean()))
        frac_pos_list.append(float(yt[mask].mean()))
    return (
        np.array(bin_centers_list, dtype=np.float64),
        np.array(mean_pred_list, dtype=np.float64),
        np.array(frac_pos_list, dtype=np.float64),
    )


@dataclass
class MulticlassBrier:
    """Multiclass Brier score = mean over rows of sum_k (p_k - onehot_k)^2."""

    brier_score: float
    skill_score: float
    n_samples: int
    n_classes: int


def multiclass_brier(y_true: ArrayLike, y_prob: ArrayLike) -> MulticlassBrier:
    """Multiclass Brier score with a climatology skill score.

    ``y_true`` is a length-n vector of integer class indices (0..k-1); ``y_prob``
    is the (n, k) probability matrix. Skill is measured against the base-rate
    (climatology) forecast that predicts the training class frequencies for every
    row: ``skill = 1 - BS_model / BS_climatology``.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_prob, dtype=float)
    if yt.size == 0 or yp.size == 0:
        raise ValueError("Empty arrays - cannot compute multiclass Brier score")
    if len(yt) != len(yp):
        raise ValueError("y_true and y_prob must have the same length")
    n, k = yp.shape
    # Map labels to column indices (handles non-0-based / unsorted label sets).
    classes = np.unique(yt)
    class_to_col = {c: i for i, c in enumerate(classes)}
    col_idx = np.array([class_to_col[c] for c in yt])
    onehot = np.zeros((n, k), dtype=float)
    onehot[np.arange(n), col_idx] = 1.0

    brier = float(np.mean(np.sum((yp - onehot) ** 2, axis=1)))
    base_rates = onehot.mean(axis=0, keepdims=True)
    bs_clim = float(np.mean(np.sum((base_rates - onehot) ** 2, axis=1)))
    skill = 1.0 - brier / bs_clim if bs_clim > 0 else 0.0
    return MulticlassBrier(
        brier_score=brier,
        skill_score=float(skill),
        n_samples=n,
        n_classes=k,
    )


def reliability_data_multiclass(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    n_bins: int = 10,
) -> list[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]]:
    """Per-class reliability data via one-vs-rest.

    Returns one ``(bin_centers, mean_predicted_prob, fraction_positive)`` tuple
    per class, in the column order of the probability matrix.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_prob, dtype=float)
    classes = np.unique(yt)
    curves: list[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]] = []
    for col, cls in enumerate(classes):
        binary_target = (yt == cls).astype(float)
        curves.append(reliability_data(binary_target, yp[:, col], n_bins=n_bins))
    return curves
