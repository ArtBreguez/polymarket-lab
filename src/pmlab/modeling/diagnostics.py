"""Calibration diagnostics: Brier score decomposition and reliability data."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

__all__ = ["BrierDecomposition", "brier_decomposition", "reliability_data"]

@dataclass
class BrierDecomposition:
    uncertainty: float
    resolution: float
    reliability: float
    brier_score: float
    skill_score: float
    n_samples: int

def brier_decomposition(y_true, y_prob, n_bins: int = 10) -> BrierDecomposition:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) == 0:
        raise ValueError("Empty arrays - cannot compute Brier decomposition")
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length")
    n = len(y_true)
    clim = y_true.mean()
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    reliability_term = 0.0
    resolution_term = 0.0
    for k in range(n_bins):
        mask = bin_indices == k
        nk = mask.sum()
        if nk == 0:
            continue
        ok = y_true[mask].mean()
        fk = y_prob[mask].mean()
        reliability_term += nk * (fk - ok) ** 2
        resolution_term += nk * (ok - clim) ** 2
    reliability_term /= n
    resolution_term /= n
    uncertainty_term = clim * (1.0 - clim)
    brier = float(np.mean((y_prob - y_true) ** 2))
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

def reliability_data(y_true, y_prob, n_bins: int = 10):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    bin_centers_list, mean_pred_list, frac_pos_list = [], [], []
    for k in range(n_bins):
        mask = bin_indices == k
        if mask.sum() == 0:
            continue
        bin_centers_list.append((bins[k] + bins[k + 1]) / 2.0)
        mean_pred_list.append(float(y_prob[mask].mean()))
        frac_pos_list.append(float(y_true[mask].mean()))
    return np.array(bin_centers_list), np.array(mean_pred_list), np.array(frac_pos_list)
