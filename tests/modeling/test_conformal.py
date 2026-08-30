"""TDD tests for ConformalForecaster (split-conformal classification) — RED first."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# MUST fail before implementation (RED)
from pmlab.modeling.conformal import ConformalForecaster
from pmlab.modeling.sklearn_forecaster import SklearnForecaster


@pytest.fixture
def data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    n = 1200
    X = pd.DataFrame(
        {
            "a": rng.normal(0, 1, n),
            "b": rng.normal(0, 1, n),
        }
    )
    y = pd.Series((X["a"] + rng.normal(0, 0.6, n) > 0).astype(int))
    return X, y


class TestConstruction:
    def test_alpha_must_be_in_0_1(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            ConformalForecaster(SklearnForecaster(), alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            ConformalForecaster(SklearnForecaster(), alpha=1.0)


class TestPredictProba:
    def test_passthrough_proba_shape(self, data: tuple[pd.DataFrame, pd.Series]) -> None:
        X, y = data
        clf = ConformalForecaster(SklearnForecaster(estimator="random_forest", n_estimators=40))
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)

    def test_predict_before_fit_raises(self, data: tuple[pd.DataFrame, pd.Series]) -> None:
        X, _ = data
        clf = ConformalForecaster(SklearnForecaster())
        with pytest.raises(RuntimeError, match="fit"):
            clf.predict_proba(X)


class TestPredictionSets:
    def test_prediction_set_is_list_of_sets(
        self, data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = data
        clf = ConformalForecaster(SklearnForecaster(), alpha=0.1)
        clf.fit(X, y)
        sets = clf.predict_set(X.head(5))
        assert len(sets) == 5
        assert all(isinstance(s, set) for s in sets)
        # Every set is a subset of the label space {0, 1}
        assert all(s <= {0, 1} for s in sets)

    def test_empirical_coverage_meets_guarantee(
        self, data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        # Split-conformal guarantees marginal coverage >= 1 - alpha on exchangeable
        # data. Hold out a fresh test set and check the realized coverage.
        X, y = data
        X_train, y_train = X.iloc[:900], y.iloc[:900]
        X_test, y_test = X.iloc[900:], y.iloc[900:]
        alpha = 0.1
        clf = ConformalForecaster(
            SklearnForecaster(estimator="random_forest", n_estimators=60),
            alpha=alpha,
            random_state=0,
        )
        clf.fit(X_train, y_train)
        sets = clf.predict_set(X_test)
        covered = [yt in s for yt, s in zip(y_test.tolist(), sets, strict=True)]
        coverage = float(np.mean(covered))
        # Allow a small finite-sample slack below the 1 - alpha target.
        assert coverage >= (1 - alpha) - 0.05

    def test_smaller_alpha_gives_larger_sets(
        self, data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = data
        strict = ConformalForecaster(SklearnForecaster(), alpha=0.01, random_state=1)
        loose = ConformalForecaster(SklearnForecaster(), alpha=0.30, random_state=1)
        strict.fit(X, y)
        loose.fit(X, y)
        strict_sz = float(np.mean([len(s) for s in strict.predict_set(X)]))
        loose_sz = float(np.mean([len(s) for s in loose.predict_set(X)]))
        # Higher confidence (smaller alpha) => larger average set size.
        assert strict_sz >= loose_sz

    def test_predict_set_before_fit_raises(
        self, data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, _ = data
        clf = ConformalForecaster(SklearnForecaster())
        with pytest.raises(RuntimeError, match="fit"):
            clf.predict_set(X)


class TestPersistence:
    def test_save_load_roundtrip(
        self, data: tuple[pd.DataFrame, pd.Series], tmp_path: Path
    ) -> None:
        X, y = data
        clf = ConformalForecaster(
            SklearnForecaster(estimator="random_forest", n_estimators=40),
            alpha=0.1,
        )
        clf.fit(X, y)
        before = clf.predict_set(X.head(20))
        path = tmp_path / "conformal.pkl"
        clf.save(path)
        loaded = ConformalForecaster.load(path)
        after = loaded.predict_set(X.head(20))
        assert before == after

    def test_feature_names(self, data: tuple[pd.DataFrame, pd.Series]) -> None:
        X, y = data
        clf = ConformalForecaster(SklearnForecaster())
        clf.fit(X, y)
        assert clf.feature_names() == ["a", "b"]


class TestCategoricalLabels:
    """Regression: predict_set must preserve non-integer labels (F1/politics)."""

    def test_string_labels_returned_verbatim(self) -> None:
        rng = np.random.default_rng(11)
        n = 600
        X = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
        y = pd.Series(np.where(X["a"] > 0, "UP", "DOWN"))
        clf = ConformalForecaster(
            SklearnForecaster(estimator="random_forest", n_estimators=40), alpha=0.2
        )
        clf.fit(X, y)
        sets = clf.predict_set(X.head(10))
        assert len(sets) == 10
        # Labels come back as the original strings, not coerced ints.
        for s in sets:
            assert s <= {"UP", "DOWN"}
            assert all(isinstance(label, str) for label in s)

    def test_multiclass_string_coverage(self) -> None:
        rng = np.random.default_rng(12)
        n = 1500
        X = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
        # 3-way categorical target driven by feature a.
        y = pd.Series(
            np.select(
                [X["a"] < -0.4, X["a"] > 0.4],
                ["LOW", "HIGH"],
                default="MID",
            )
        )
        X_tr, y_tr = X.iloc[:1100], y.iloc[:1100]
        X_te, y_te = X.iloc[1100:], y.iloc[1100:]
        alpha = 0.1
        clf = ConformalForecaster(
            SklearnForecaster(estimator="random_forest", n_estimators=80),
            alpha=alpha,
            random_state=0,
        )
        clf.fit(X_tr, y_tr)
        sets = clf.predict_set(X_te)
        covered = [yt in s for yt, s in zip(y_te.tolist(), sets, strict=True)]
        assert float(np.mean(covered)) >= (1 - alpha) - 0.05
        assert all(s <= {"LOW", "MID", "HIGH"} for s in sets)
