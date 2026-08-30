"""TDD tests for SigmoidCalibrator + CalibratedForecaster — written BEFORE impl."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# These imports MUST fail before implementation (RED)
from pmlab.modeling.calibration import CalibratedForecaster, SigmoidCalibrator
from pmlab.modeling.sklearn_forecaster import SklearnForecaster


class TestSigmoidCalibrator:
    def test_transform_outputs_in_01(self) -> None:
        cal = SigmoidCalibrator()
        rng = np.random.default_rng(0)
        probs = rng.uniform(0, 1, 200)
        labels = (probs > 0.5).astype(int)
        cal.fit(probs, labels)
        out = cal.transform(np.array([-2.0, 0.0, 0.5, 1.0, 3.0]))
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_monotonic_in_input(self) -> None:
        cal = SigmoidCalibrator()
        rng = np.random.default_rng(1)
        probs = rng.uniform(0, 1, 300)
        labels = (rng.uniform(0, 1, 300) < probs).astype(int)
        cal.fit(probs, labels)
        grid = np.linspace(0.01, 0.99, 50)
        out = cal.transform(grid)
        # Platt scaling is monotonic increasing in the raw score.
        assert np.all(np.diff(out) >= -1e-9)

    def test_transform_before_fit_raises(self) -> None:
        cal = SigmoidCalibrator()
        with pytest.raises(RuntimeError, match="fitted"):
            cal.transform(np.array([0.5]))

    def test_improves_calibration_on_miscalibrated_scores(self) -> None:
        # Overconfident scores: true rate is a shrunk version of the score.
        rng = np.random.default_rng(7)
        true_p = rng.uniform(0.1, 0.9, 2000)
        labels = (rng.uniform(0, 1, 2000) < true_p).astype(int)
        # Distort: push probabilities toward the extremes (overconfidence).
        raw = np.clip(true_p**2 / (true_p**2 + (1 - true_p) ** 2), 1e-6, 1 - 1e-6)
        cal = SigmoidCalibrator()
        cal.fit(raw, labels)
        calibrated = cal.transform(raw)
        brier_raw = float(np.mean((raw - labels) ** 2))
        brier_cal = float(np.mean((calibrated - labels) ** 2))
        assert brier_cal <= brier_raw + 1e-9

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        cal = SigmoidCalibrator()
        probs = np.array([0.1, 0.2, 0.3, 0.6, 0.8, 0.9])
        labels = np.array([0, 0, 0, 1, 1, 1])
        cal.fit(probs, labels)
        path = tmp_path / "sig.pkl"
        cal.save(path)
        loaded = SigmoidCalibrator.load(path)
        grid = np.array([0.2, 0.5, 0.8])
        np.testing.assert_allclose(cal.transform(grid), loaded.transform(grid), atol=1e-10)


@pytest.fixture
def data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "a": rng.normal(0, 1, 400),
            "b": rng.normal(0, 1, 400),
        }
    )
    y = pd.Series((X["a"] + rng.normal(0, 0.5, 400) > 0).astype(int))
    return X, y


class TestCalibratedForecaster:
    def test_predict_proba_shape_and_normalized(
        self, data: tuple[pd.DataFrame, pd.Series]
    ) -> None:
        X, y = data
        base = SklearnForecaster(estimator="random_forest", n_estimators=30)
        clf = CalibratedForecaster(base, method="isotonic")
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (400, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)

    def test_sigmoid_method_supported(self, data: tuple[pd.DataFrame, pd.Series]) -> None:
        X, y = data
        clf = CalibratedForecaster(SklearnForecaster(), method="sigmoid")
        clf.fit(X, y)
        assert clf.predict_proba(X).shape == (400, 2)

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="method"):
            CalibratedForecaster(SklearnForecaster(), method="bogus")

    def test_predict_before_fit_raises(self, data: tuple[pd.DataFrame, pd.Series]) -> None:
        X, _ = data
        clf = CalibratedForecaster(SklearnForecaster())
        with pytest.raises(RuntimeError, match="fit"):
            clf.predict_proba(X)

    def test_only_binary_supported(self) -> None:
        rng = np.random.default_rng(3)
        X = pd.DataFrame({"a": rng.normal(0, 1, 90)})
        y = pd.Series(rng.integers(0, 3, 90))  # 3 classes
        clf = CalibratedForecaster(SklearnForecaster())
        with pytest.raises(ValueError, match="binary"):
            clf.fit(X, y)

    def test_save_load_roundtrip(
        self, data: tuple[pd.DataFrame, pd.Series], tmp_path: Path
    ) -> None:
        X, y = data
        clf = CalibratedForecaster(
            SklearnForecaster(estimator="random_forest", n_estimators=30),
            method="isotonic",
        )
        clf.fit(X, y)
        before = clf.predict_proba(X)
        path = tmp_path / "cal_fc.pkl"
        clf.save(path)
        loaded = CalibratedForecaster.load(path)
        np.testing.assert_allclose(loaded.predict_proba(X), before, atol=1e-10)

    def test_feature_names(self, data: tuple[pd.DataFrame, pd.Series]) -> None:
        X, y = data
        clf = CalibratedForecaster(SklearnForecaster())
        clf.fit(X, y)
        assert clf.feature_names() == ["a", "b"]
