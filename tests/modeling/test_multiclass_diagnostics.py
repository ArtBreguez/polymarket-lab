"""TDD tests for multiclass Brier + per-class reliability — RED first."""

from __future__ import annotations

import numpy as np
import pytest

from pmlab.modeling.diagnostics import (
    MulticlassBrier,
    multiclass_brier,
    reliability_data_multiclass,
)


@pytest.fixture
def three_class() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n, k = 900, 3
    true_p = rng.dirichlet(np.ones(k) * 1.5, size=n)
    labels = np.array([rng.choice(k, p=row) for row in true_p])
    return labels, true_p


class TestMulticlassBrier:
    def test_returns_dataclass_with_fields(
        self, three_class: tuple[np.ndarray, np.ndarray]
    ) -> None:
        labels, proba = three_class
        res = multiclass_brier(labels, proba)
        assert isinstance(res, MulticlassBrier)
        assert res.n_samples == len(labels)
        assert res.n_classes == 3
        assert res.brier_score >= 0.0

    def test_perfect_prediction_zero_brier(self) -> None:
        labels = np.array([0, 1, 2, 1, 0])
        proba = np.eye(3)[labels]  # one-hot = perfect
        res = multiclass_brier(labels, proba)
        assert res.brier_score == pytest.approx(0.0, abs=1e-12)

    def test_matches_manual_sum_of_squares(
        self, three_class: tuple[np.ndarray, np.ndarray]
    ) -> None:
        labels, proba = three_class
        onehot = np.eye(proba.shape[1])[labels]
        manual = float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))
        assert multiclass_brier(labels, proba).brier_score == pytest.approx(manual, abs=1e-12)

    def test_skill_score_positive_when_better_than_climatology(
        self, three_class: tuple[np.ndarray, np.ndarray]
    ) -> None:
        labels, proba = three_class
        res = multiclass_brier(labels, proba)
        # A model that tracks the generating distribution beats the base rate.
        assert res.skill_score > 0.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length"):
            multiclass_brier(np.array([0, 1]), np.eye(3)[[0, 1, 2]])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="[Ee]mpty"):
            multiclass_brier(np.array([]), np.empty((0, 3)))


class TestReliabilityMulticlass:
    def test_returns_one_curve_per_class(self, three_class: tuple[np.ndarray, np.ndarray]) -> None:
        labels, proba = three_class
        curves = reliability_data_multiclass(labels, proba, n_bins=10)
        assert len(curves) == 3  # one (centers, mean_pred, frac_pos) per class
        for centers, mean_pred, frac_pos in curves:
            assert len(centers) == len(mean_pred) == len(frac_pos)
            assert np.all((frac_pos >= 0) & (frac_pos <= 1))
