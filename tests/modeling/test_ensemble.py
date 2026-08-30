"""TDD tests for EnsembleForecaster — written BEFORE implementation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# This import MUST fail before implementation (RED)
from pmlab.modeling.ensemble import EnsembleForecaster
from pmlab.modeling.sklearn_forecaster import SklearnForecaster


@pytest.fixture
def sample_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "feature_a": rng.normal(0, 1, 200),
            "feature_b": rng.normal(0, 1, 200),
        }
    )
    y = pd.Series((X["feature_a"] + rng.normal(0, 0.1, 200) > 0).astype(int))
    return X, y


def _members() -> list[SklearnForecaster]:
    return [
        SklearnForecaster(estimator="logistic_regression"),
        SklearnForecaster(estimator="random_forest", n_estimators=20),
    ]


class TestEnsembleConstruction:
    def test_requires_at_least_one_member(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            EnsembleForecaster(forecasters=[])

    def test_weights_length_must_match(self, sample_data: tuple[pd.DataFrame, pd.Series]) -> None:
        with pytest.raises(ValueError, match="weights"):
            EnsembleForecaster(forecasters=_members(), weights=[1.0])

    def test_rejects_negative_weights(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            EnsembleForecaster(forecasters=_members(), weights=[1.0, -1.0])

    def test_rejects_all_zero_weights(self) -> None:
        with pytest.raises(ValueError, match="sum"):
            EnsembleForecaster(forecasters=_members(), weights=[0.0, 0.0])


class TestEnsemblePredict:
    def test_predict_proba_shape(self, sample_data: tuple[pd.DataFrame, pd.Series]) -> None:
        X, y = sample_data
        ens = EnsembleForecaster(forecasters=_members())
        ens.fit(X, y)
        proba = ens.predict_proba(X)
        assert proba.shape == (200, 2)

    def test_predict_proba_rows_sum_to_one(
        self, sample_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = sample_data
        ens = EnsembleForecaster(forecasters=_members())
        ens.fit(X, y)
        proba = ens.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)

    def test_equal_weight_is_mean_of_members(
        self, sample_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = sample_data
        members = _members()
        ens = EnsembleForecaster(forecasters=members)
        ens.fit(X, y)
        # Members are fitted by the ensemble; compare its blend to the manual mean.
        manual = np.mean([m.predict_proba(X) for m in ens.forecasters], axis=0)
        np.testing.assert_allclose(ens.predict_proba(X), manual, atol=1e-9)

    def test_weighted_blend_matches_manual(
        self, sample_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = sample_data
        ens = EnsembleForecaster(forecasters=_members(), weights=[3.0, 1.0])
        ens.fit(X, y)
        p0 = ens.forecasters[0].predict_proba(X)
        p1 = ens.forecasters[1].predict_proba(X)
        expected = (3.0 * p0 + 1.0 * p1) / 4.0
        np.testing.assert_allclose(ens.predict_proba(X), expected, atol=1e-9)

    def test_predict_before_fit_raises(self, sample_data: tuple[pd.DataFrame, pd.Series]) -> None:
        X, _ = sample_data
        ens = EnsembleForecaster(forecasters=_members())
        with pytest.raises(RuntimeError, match="fit"):
            ens.predict_proba(X)

    def test_mismatched_member_shapes_raise_clearly(
        self, sample_data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        # Regression: blending members with different class counts must raise a
        # clear error, not a cryptic numpy broadcast failure.
        X, _ = sample_data

        class _Binary(SklearnForecaster):
            def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
                return np.tile([0.5, 0.5], (len(X), 1))

        class _Ternary(SklearnForecaster):
            def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
                return np.tile([0.3, 0.3, 0.4], (len(X), 1))

        ens = EnsembleForecaster(forecasters=[_Binary(), _Ternary()])
        ens._fitted = True
        with pytest.raises(ValueError, match="shape"):
            ens.predict_proba(X)


class TestEnsemblePersistence:
    def test_save_load_roundtrip(
        self, sample_data: tuple[pd.DataFrame, pd.Series], tmp_path: Path
    ) -> None:
        X, y = sample_data
        ens = EnsembleForecaster(forecasters=_members(), weights=[2.0, 1.0])
        ens.fit(X, y)
        before = ens.predict_proba(X)

        path = tmp_path / "ensemble.pkl"
        ens.save(path)
        loaded = EnsembleForecaster.load(path)

        np.testing.assert_allclose(loaded.predict_proba(X), before, atol=1e-12)

    def test_feature_names_from_members(self, sample_data: tuple[pd.DataFrame, pd.Series]) -> None:
        X, y = sample_data
        ens = EnsembleForecaster(forecasters=_members())
        ens.fit(X, y)
        assert ens.feature_names() == ["feature_a", "feature_b"]
