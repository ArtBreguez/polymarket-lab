"""Isotonic calibration wrapper."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression


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
        calibrated = self._model.predict(probs)
        return np.clip(calibrated, 0.0, 1.0)

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
