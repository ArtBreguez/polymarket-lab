"""Calibration diagnostics: Brier score decomposition and reliability data."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["BrierDecomposition", "brier_decomposition", "reliability_data"]


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
