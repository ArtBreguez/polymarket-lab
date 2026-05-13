"""Tests for calibration diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from pmlab.modeling.diagnostics import BrierDecomposition, brier_decomposition, reliability_data

RNG = np.random.default_rng(42)


class TestBrierDecomposition:
    def test_perfect_forecast(self):
        y_true = np.array([1, 1, 0, 0, 1])
        y_prob = np.array([1.0, 1.0, 0.0, 0.0, 1.0])
        r = brier_decomposition(y_true, y_prob)
        assert r.brier_score == pytest.approx(0.0, abs=1e-9)
        assert r.skill_score == pytest.approx(1.0, abs=1e-6)

    def test_climatological_forecast(self):
        y_true = np.array([1, 0, 1, 0, 1, 0])
        clim = y_true.mean()
        y_prob = np.full(len(y_true), clim)
        r = brier_decomposition(y_true, y_prob)
        assert r.skill_score == pytest.approx(0.0, abs=1e-6)

    def test_random_forecast(self):
        y_true = RNG.integers(0, 2, size=500).astype(float)
        y_prob = RNG.random(500)
        r = brier_decomposition(y_true, y_prob)
        assert isinstance(r, BrierDecomposition)
        assert r.n_samples == 500

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Empty"):
            brier_decomposition([], [])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            brier_decomposition([1, 0], [0.5])

    def test_decomposition_identity(self):
        y_true = RNG.integers(0, 2, size=200).astype(float)
        y_prob = np.clip(RNG.normal(0.5, 0.2, 200), 0, 1)
        r = brier_decomposition(y_true, y_prob, n_bins=10)
        reconstructed = r.reliability - r.resolution + r.uncertainty
        assert reconstructed == pytest.approx(r.brier_score, abs=0.02)


class TestReliabilityData:
    def test_returns_three_arrays(self):
        y_true = RNG.integers(0, 2, size=200).astype(float)
        y_prob = RNG.random(200)
        centers, preds, fracs = reliability_data(y_true, y_prob)
        assert len(centers) == len(preds) == len(fracs)
        assert len(centers) > 0

    def test_fracs_in_range(self):
        y_true = RNG.integers(0, 2, size=200).astype(float)
        y_prob = RNG.random(200)
        _, _, fracs = reliability_data(y_true, y_prob)
        assert np.all(fracs >= 0) and np.all(fracs <= 1)
