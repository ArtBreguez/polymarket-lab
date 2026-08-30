"""TDD tests for MulticlassCalibrator + multiclass CalibratedForecaster — RED first."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pmlab.modeling.calibration import CalibratedForecaster, MulticlassCalibrator
from pmlab.modeling.sklearn_forecaster import SklearnForecaster


@pytest.fixture
def multiclass_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    n = 1500
    X = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
    # 3-way target driven by feature a.
    y = pd.Series(np.select([X["a"] < -0.4, X["a"] > 0.4], ["LOW", "HIGH"], default="MID"))
    return X, y


class TestMulticlassCalibrator:
    def test_fit_transform_shape_and_normalized(self) -> None:
        rng = np.random.default_rng(0)
        n, k = 600, 3
        raw = rng.dirichlet(np.ones(k), size=n)
        labels = np.array([rng.choice(k, p=row) for row in raw])
        cal = MulticlassCalibrator(method="isotonic")
        cal.fit(raw, labels)
        out = cal.transform(raw)
        assert out.shape == (n, k)
        np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-9)

    def test_transform_before_fit_raises(self) -> None:
        cal = MulticlassCalibrator()
        with pytest.raises(RuntimeError, match="fitted"):
            cal.transform(np.array([[0.3, 0.3, 0.4]]))

    def test_sigmoid_method_supported(self) -> None:
        rng = np.random.default_rng(1)
        raw = rng.dirichlet(np.ones(4), size=300)
        labels = np.array([rng.choice(4, p=row) for row in raw])
        cal = MulticlassCalibrator(method="sigmoid")
        cal.fit(raw, labels)
        assert cal.transform(raw).shape == (300, 4)

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="method"):
            MulticlassCalibrator(method="bogus")

    def test_improves_or_matches_brier_on_miscalibrated(self) -> None:
        # Overconfident 3-class scores; calibration shouldn't worsen multiclass Brier.
        rng = np.random.default_rng(7)
        k = 3
        true_p = rng.dirichlet(np.ones(k) * 2.0, size=3000)
        labels = np.array([rng.choice(k, p=row) for row in true_p])
        sharp = true_p**3
        raw = sharp / sharp.sum(axis=1, keepdims=True)  # push to extremes
        onehot = np.eye(k)[labels]
        cal = MulticlassCalibrator(method="isotonic")
        cal.fit(raw, labels)
        calibrated = cal.transform(raw)
        brier_raw = float(np.mean(np.sum((raw - onehot) ** 2, axis=1)))
        brier_cal = float(np.mean(np.sum((calibrated - onehot) ** 2, axis=1)))
        assert brier_cal <= brier_raw + 1e-9

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(3)
        raw = rng.dirichlet(np.ones(3), size=400)
        labels = np.array([rng.choice(3, p=row) for row in raw])
        cal = MulticlassCalibrator(method="isotonic")
        cal.fit(raw, labels)
        path = tmp_path / "mc.pkl"
        cal.save(path)
        loaded = MulticlassCalibrator.load(path)
        np.testing.assert_allclose(cal.transform(raw), loaded.transform(raw), atol=1e-10)

    def test_wrong_column_count_raises(self) -> None:
        # Regression: transforming a matrix whose class count differs from fit
        # must raise, not silently return a wrong-shaped result.
        rng = np.random.default_rng(4)
        raw = rng.dirichlet(np.ones(3), size=100)
        labels = np.array([rng.choice(3, p=row) for row in raw])
        cal = MulticlassCalibrator()
        cal.fit(raw, labels)
        with pytest.raises(ValueError, match="matching the fitted classes"):
            cal.transform(rng.dirichlet(np.ones(4), size=10))


class TestCalibratedForecasterMulticlass:
    def test_three_class_fit_predict(self, multiclass_data: tuple[pd.DataFrame, pd.Series]) -> None:
        X, y = multiclass_data
        clf = CalibratedForecaster(
            SklearnForecaster(estimator="random_forest", n_estimators=60),
            method="isotonic",
        )
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)

    def test_sigmoid_multiclass(self, multiclass_data: tuple[pd.DataFrame, pd.Series]) -> None:
        X, y = multiclass_data
        clf = CalibratedForecaster(SklearnForecaster(), method="sigmoid")
        clf.fit(X, y)
        assert clf.predict_proba(X).shape == (len(X), 3)

    def test_binary_still_works(self) -> None:
        rng = np.random.default_rng(5)
        X = pd.DataFrame({"a": rng.normal(0, 1, 400), "b": rng.normal(0, 1, 400)})
        y = pd.Series((X["a"] > 0).astype(int))
        clf = CalibratedForecaster(SklearnForecaster(), method="isotonic")
        clf.fit(X, y)
        p = clf.predict_proba(X)
        assert p.shape == (400, 2)
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-9)

    def test_save_load_roundtrip_multiclass(
        self, multiclass_data: tuple[pd.DataFrame, pd.Series], tmp_path: Path
    ) -> None:
        X, y = multiclass_data
        clf = CalibratedForecaster(
            SklearnForecaster(estimator="random_forest", n_estimators=60),
            method="isotonic",
        )
        clf.fit(X, y)
        before = clf.predict_proba(X)
        path = tmp_path / "mc_fc.pkl"
        clf.save(path)
        loaded = CalibratedForecaster.load(path)
        np.testing.assert_allclose(loaded.predict_proba(X), before, atol=1e-10)
