"""Probability calibration: isotonic + Platt (sigmoid), plus a forecaster wrapper.

Prediction-market edge is computed from probabilities, so a miscalibrated model
that says "70%" when it means "55%" bleeds money even with good discrimination.
These tools map raw model scores onto empirically-calibrated probabilities.

- ``IsotonicCalibrator`` — non-parametric, monotonic; best with enough data.
- ``SigmoidCalibrator`` — Platt scaling (1-D logistic); robust on small samples.
- ``CalibratedForecaster`` — wraps any ``MarketForecaster`` and calibrates its
  positive-class probability on a held-out split, exposing the same interface.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from pmlab.modeling.base import MarketForecaster


class IsotonicCalibrator:
    """Wraps sklearn IsotonicRegression to calibrate predicted probabilities."""

    def __init__(self) -> None:
        self._model: IsotonicRegression | None = None

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> None:
        """Fit isotonic regression on raw probabilities and true labels."""
        self._model = IsotonicRegression(out_of_bounds="clip")
        self._model.fit(probs, labels)

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Calibrate probabilities, clipping output to [0, 1]."""
        if self._model is None:
            raise RuntimeError("Calibrator has not been fitted yet.")
        calibrated: np.ndarray = np.array(self._model.predict(probs))
        clipped: np.ndarray = np.clip(calibrated, 0.0, 1.0)
        return clipped

    def save(self, path: Path) -> None:
        """Pickle-serialize calibrator to path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self._model}, f)

    @classmethod
    def load(cls, path: Path) -> IsotonicCalibrator:
        """Load calibrator from path."""
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls()
        instance._model = data["model"]
        return instance


class SigmoidCalibrator:
    """Platt scaling — a 1-D logistic regression mapping raw scores to probabilities.

    More sample-efficient than isotonic when calibration data is scarce, at the
    cost of assuming a sigmoidal distortion. Output is monotonic in the input.
    """

    def __init__(self) -> None:
        self._model: LogisticRegression | None = None

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> None:
        """Fit the logistic mapping from raw scores to true labels."""
        model = LogisticRegression(C=1e10, solver="lbfgs")
        model.fit(np.asarray(probs, dtype=float).reshape(-1, 1), np.asarray(labels))
        self._model = model

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Map raw scores onto calibrated probabilities in [0, 1]."""
        if self._model is None:
            raise RuntimeError("SigmoidCalibrator has not been fitted yet.")
        x = np.asarray(probs, dtype=float).reshape(-1, 1)
        calibrated: np.ndarray = np.asarray(self._model.predict_proba(x))[:, 1]
        clipped: np.ndarray = np.clip(calibrated, 0.0, 1.0)
        return clipped

    def save(self, path: Path) -> None:
        """Pickle-serialize calibrator to path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self._model}, f)

    @classmethod
    def load(cls, path: Path) -> SigmoidCalibrator:
        """Load calibrator from path."""
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls()
        instance._model = data["model"]
        return instance


CalibrationMethod = Literal["isotonic", "sigmoid"]


class CalibratedForecaster(MarketForecaster):
    """Wrap a binary ``MarketForecaster`` and calibrate its positive-class output.

    On ``fit``, the data is split into a training part (fits the base model) and a
    held-out calibration part (fits the calibrator on the base model's out-of-fold
    scores). ``predict_proba`` then returns the calibrated two-column matrix.

    Args:
        base: The forecaster to wrap.
        method: ``"isotonic"`` (default) or ``"sigmoid"`` (Platt).
        calib_fraction: Fraction of ``fit`` data held out to fit the calibrator.
        random_state: Seed for the calibration split.
    """

    def __init__(
        self,
        base: MarketForecaster,
        method: CalibrationMethod = "isotonic",
        calib_fraction: float = 0.3,
        random_state: int = 0,
    ) -> None:
        if method not in ("isotonic", "sigmoid"):
            raise ValueError(f"Unknown calibration method '{method}'. Use 'isotonic' or 'sigmoid'.")
        self.base = base
        self.method: CalibrationMethod = method
        self.calib_fraction = calib_fraction
        self.random_state = random_state
        self._calibrator: IsotonicCalibrator | SigmoidCalibrator | None = None
        self._fitted = False
        self._feature_names: list[str] = []

    def _new_calibrator(self) -> IsotonicCalibrator | SigmoidCalibrator:
        return IsotonicCalibrator() if self.method == "isotonic" else SigmoidCalibrator()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit base model on a train split, calibrator on a held-out split."""
        classes = np.unique(np.asarray(y))
        if len(classes) != 2:
            raise ValueError(
                f"CalibratedForecaster supports binary targets only; got {len(classes)} classes."
            )
        X_tr, X_cal, y_tr, y_cal = train_test_split(
            X,
            y,
            test_size=self.calib_fraction,
            random_state=self.random_state,
            stratify=y,
        )
        self.base.fit(X_tr, y_tr)
        raw = np.asarray(self.base.predict_proba(X_cal))[:, 1]
        calibrator = self._new_calibrator()
        calibrator.fit(raw, np.asarray(y_cal))
        self._calibrator = calibrator
        self._feature_names = list(X.columns)
        self._fitted = True

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated (n, 2) probabilities."""
        if not self._fitted or self._calibrator is None:
            raise RuntimeError("CalibratedForecaster has not been fitted yet. Call fit() first.")
        raw_pos = np.asarray(self.base.predict_proba(X))[:, 1]
        pos = np.asarray(self._calibrator.transform(raw_pos), dtype=float)
        pos = np.clip(pos, 0.0, 1.0)
        return np.column_stack([1.0 - pos, pos])

    def save(self, path: Path) -> None:
        """Pickle-serialize the wrapper (base + calibrator) to path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "base": self.base,
                    "method": self.method,
                    "calib_fraction": self.calib_fraction,
                    "random_state": self.random_state,
                    "calibrator": self._calibrator,
                    "fitted": self._fitted,
                    "feature_names": self._feature_names,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> CalibratedForecaster:
        """Load a previously saved CalibratedForecaster from path."""
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls(
            base=data["base"],
            method=data["method"],
            calib_fraction=data["calib_fraction"],
            random_state=data["random_state"],
        )
        instance._calibrator = data["calibrator"]
        instance._fitted = data["fitted"]
        instance._feature_names = data["feature_names"]
        return instance

    def feature_names(self) -> list[str]:
        """Return feature names from the last fit() call."""
        return list(self._feature_names)
