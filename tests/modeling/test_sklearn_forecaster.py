"""TDD tests for SklearnForecaster — written BEFORE implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# This import MUST fail before implementation (RED)
from pmlab.modeling.sklearn_forecaster import SklearnForecaster


@pytest.fixture
def sample_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "feature_a": rng.normal(0, 1, 100),
            "feature_b": rng.normal(0, 1, 100),
        }
    )
    y = pd.Series((X["feature_a"] + rng.normal(0, 0.1, 100) > 0).astype(int))
    return X, y


class TestSklearnForecasterDefaults:
    def test_default_uses_logistic_regression(self, sample_data):
        X, y = sample_data
        f = SklearnForecaster()
        f.fit(X, y)
        proba = f.predict_proba(X)
        assert proba.shape[0] == 100

    def test_predict_proba_shape_binary(self, sample_data):
        X, y = sample_data
        f = SklearnForecaster()
        f.fit(X, y)
        proba = f.predict_proba(X)
        assert proba.ndim == 2
        assert proba.shape[1] == 2

    def test_predict_proba_sums_to_one(self, sample_data):
        X, y = sample_data
        f = SklearnForecaster()
        f.fit(X, y)
        proba = f.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_not_fitted_raises(self):
        f = SklearnForecaster()
        X = pd.DataFrame({"a": [1.0]})
        with pytest.raises(RuntimeError, match="fitted"):
            f.predict_proba(X)


class TestSklearnForecasterEstimators:
    def test_random_forest(self, sample_data):
        X, y = sample_data
        f = SklearnForecaster(estimator="random_forest", n_estimators=10)
        f.fit(X, y)
        proba = f.predict_proba(X)
        assert proba.shape == (100, 2)

    def test_custom_sklearn_estimator(self, sample_data):
        from sklearn.tree import DecisionTreeClassifier

        X, y = sample_data
        f = SklearnForecaster(estimator=DecisionTreeClassifier(max_depth=3))
        f.fit(X, y)
        proba = f.predict_proba(X)
        assert proba.shape[0] == 100

    def test_invalid_estimator_raises(self):
        with pytest.raises(ValueError, match="estimator"):
            SklearnForecaster(estimator="nonexistent_algo")


class TestSklearnForecasterSaveLoad:
    def test_save_and_load(self, tmp_path, sample_data):
        X, y = sample_data
        f = SklearnForecaster()
        f.fit(X, y)
        path = tmp_path / "model.pkl"
        f.save(path)
        loaded = SklearnForecaster.load(path)
        proba = loaded.predict_proba(X)
        assert proba.shape == (100, 2)

    def test_save_creates_parent_dirs(self, tmp_path, sample_data):
        X, y = sample_data
        f = SklearnForecaster()
        f.fit(X, y)
        path = tmp_path / "nested" / "dir" / "model.pkl"
        f.save(path)
        assert path.exists()

    def test_load_preserves_feature_names(self, tmp_path, sample_data):
        X, y = sample_data
        f = SklearnForecaster()
        f.fit(X, y)
        path = tmp_path / "model.pkl"
        f.save(path)
        loaded = SklearnForecaster.load(path)
        assert loaded.feature_names() == ["feature_a", "feature_b"]


class TestSklearnForecasterFeatureNames:
    def test_feature_names_after_fit(self, sample_data):
        X, y = sample_data
        f = SklearnForecaster()
        f.fit(X, y)
        assert f.feature_names() == ["feature_a", "feature_b"]

    def test_feature_names_before_fit(self):
        f = SklearnForecaster()
        assert f.feature_names() == []


class TestSklearnForecasterIsMarketForecaster:
    def test_is_market_forecaster(self):
        from pmlab.modeling.base import MarketForecaster

        f = SklearnForecaster()
        assert isinstance(f, MarketForecaster)
