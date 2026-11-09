"""Calibration helpers refuse what is not a probability, a label or a bin count (#58)."""

from __future__ import annotations

import numpy as np
import pytest

from pmlab import (
    IsotonicCalibrator,
    brier_decomposition,
    multiclass_brier,
    reliability_data,
)

Y = np.array([0, 1, 1, 0, 1, 0, 1, 1])
P = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4])


@pytest.mark.parametrize("fn", [brier_decomposition, reliability_data])
class TestBinaryDiagnostics:
    def test_valid_input_still_works(self, fn):
        fn(Y, P)
        fn(Y.astype(bool), P)
        fn(list(Y), list(P))
        fn(Y, np.zeros(8))  # a degenerate forecast is a forecast
        fn(np.ones(8), P)

    def test_labels_must_be_zero_or_one(self, fn):
        with pytest.raises(ValueError, match="y_true"):
            fn(Y * 2, P)
        with pytest.raises(ValueError, match="y_true"):
            fn(np.where(np.arange(8) == 2, np.nan, Y), P)

    def test_probabilities_must_be_in_unit_interval(self, fn):
        with pytest.raises(ValueError, match="y_prob"):
            fn(Y, P * 2)
        with pytest.raises(ValueError, match="y_prob"):
            fn(Y, P - 0.5)
        with pytest.raises(ValueError, match="y_prob"):
            fn(Y, np.where(np.arange(8) == 2, np.nan, P))
        with pytest.raises(ValueError, match="y_prob"):
            fn(Y, np.where(np.arange(8) == 2, np.inf, P))

    def test_inputs_must_be_one_dimensional(self, fn):
        with pytest.raises(ValueError, match="1-D"):
            fn(Y.reshape(2, 4), P.reshape(2, 4))

    @pytest.mark.parametrize("bad", [0, -1, None, 2.5, True, "10"])
    def test_n_bins_must_be_a_positive_int(self, fn, bad):
        with pytest.raises((ValueError, TypeError), match="n_bins"):
            fn(Y, P, n_bins=bad)

    def test_length_mismatch(self, fn):
        with pytest.raises(ValueError, match="same length"):
            fn(Y, P[:5])


def test_swapped_isotonic_fit_fails_loudly():
    """`fit(probs, labels)`: the swapped call used to fit a nonsense map without a word."""
    c = IsotonicCalibrator()
    with pytest.raises(ValueError, match="labels"):
        c.fit(Y, P)
    with pytest.raises(ValueError, match="probs"):
        c.fit(P * 3, Y)
    c.fit(P, Y)
    assert c.transform(P).shape == P.shape


def test_isotonic_accepts_what_the_multiclass_calibrator_passes_it():
    from pmlab import MulticlassCalibrator

    rng = np.random.default_rng(0)
    proba = rng.dirichlet(np.ones(3), size=60)
    labels = rng.integers(0, 3, size=60)
    cal = MulticlassCalibrator(method="isotonic")
    cal.fit(proba, labels)
    out = cal.transform(proba)
    assert np.allclose(out.sum(axis=1), 1.0)


def test_multiclass_brier_needs_probabilities():
    ok = multiclass_brier([0, 1, 2], np.eye(3))
    assert ok.brier_score == 0.0
    with pytest.raises(ValueError, match="sum to 1"):
        multiclass_brier([0, 1, 2], np.ones((3, 3)))
    with pytest.raises(ValueError, match="y_prob"):
        multiclass_brier([0, 1, 2], np.full((3, 3), np.nan))
    with pytest.raises(ValueError, match="2-D"):
        multiclass_brier([0, 1, 2], np.ones(3))
