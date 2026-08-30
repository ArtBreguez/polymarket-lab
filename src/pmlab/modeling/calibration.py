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


class MulticlassCalibrator:
    """One-vs-rest probability calibration for multiclass score matrices.

    Fits an independent 1-D calibrator (isotonic or sigmoid) per class on the
    one-vs-rest problem "is this row class k?", then renormalizes the calibrated
    columns to sum to 1. This is the standard reduction sklearn uses for
    multiclass calibration, and it keeps each class's mapping monotonic.

    Input to ``fit``/``transform`` is a (n, n_classes) probability matrix whose
    columns are ordered by ``np.unique`` of the training labels.
    """

    def __init__(self, method: CalibrationMethod = "isotonic") -> None:
        if method not in ("isotonic", "sigmoid"):
            raise ValueError(f"Unknown calibration method '{method}'. Use 'isotonic' or 'sigmoid'.")
        self.method: CalibrationMethod = method
        self._calibrators: list[IsotonicCalibrator | SigmoidCalibrator] | None = None
        self._classes: np.ndarray | None = None

    def _new_calibrator(self) -> IsotonicCalibrator | SigmoidCalibrator:
        return IsotonicCalibrator() if self.method == "isotonic" else SigmoidCalibrator()

    def fit(self, proba: np.ndarray, labels: np.ndarray) -> None:
        """Fit one calibrator per class on the one-vs-rest target."""
        p = np.asarray(proba, dtype=float)
        y = np.asarray(labels)
        self._classes = np.unique(y)
        calibrators: list[IsotonicCalibrator | SigmoidCalibrator] = []
        for col, cls in enumerate(self._classes):
            binary_target = (y == cls).astype(int)
            cal = self._new_calibrator()
            cal.fit(p[:, col], binary_target)
            calibrators.append(cal)
        self._calibrators = calibrators

    def transform(self, proba: np.ndarray) -> np.ndarray:
        """Calibrate each column, then renormalize rows to sum to 1."""
        if self._calibrators is None:
            raise RuntimeError("MulticlassCalibrator has not been fitted yet.")
        p = np.asarray(proba, dtype=float)
        if p.ndim != 2 or p.shape[1] != len(self._calibrators):
            raise ValueError(
                f"Expected a (n, {len(self._calibrators)}) probability matrix matching the "
                f"fitted classes; got shape {p.shape}."
            )
        cols = [cal.transform(p[:, i]) for i, cal in enumerate(self._calibrators)]
        stacked = np.clip(np.column_stack(cols), 0.0, 1.0)
        row_sums = stacked.sum(axis=1, keepdims=True)
        # Guard the degenerate all-zero row: fall back to a uniform distribution.
        safe = np.where(row_sums > 0, row_sums, 1.0)
        normalized = stacked / safe
        zero_rows = row_sums.ravel() == 0
        if zero_rows.any():
            normalized[zero_rows] = 1.0 / stacked.shape[1]
        return np.asarray(normalized)

    def save(self, path: Path) -> None:
        """Pickle-serialize the per-class calibrators to path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"method": self.method, "calibrators": self._calibrators, "classes": self._classes},
                f,
            )

    @classmethod
    def load(cls, path: Path) -> MulticlassCalibrator:
        """Load a previously saved MulticlassCalibrator from path."""
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls(method=data["method"])
        instance._calibrators = data["calibrators"]
        instance._classes = data["classes"]
        return instance


class CalibratedForecaster(MarketForecaster):
    """Wrap a ``MarketForecaster`` and calibrate its probability output.

    On ``fit``, the data is split into a training part (fits the base model) and a
    held-out calibration part (fits the calibrator on the base model's out-of-fold
    scores). ``predict_proba`` then returns the calibrated matrix. Binary targets
    calibrate the positive class; multiclass targets use one-vs-rest calibration
    (``MulticlassCalibrator``), so F1 / political multi-outcome markets are
    supported too.

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
        self._calibrator: IsotonicCalibrator | SigmoidCalibrator | MulticlassCalibrator | None = (
            None
        )
        self._multiclass = False
        self._fitted = False
        self._feature_names: list[str] = []

    def _new_calibrator(self) -> IsotonicCalibrator | SigmoidCalibrator:
        return IsotonicCalibrator() if self.method == "isotonic" else SigmoidCalibrator()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit base model on a train split, calibrator on a held-out split."""
        classes = np.unique(np.asarray(y))
        if len(classes) < 2:
            raise ValueError(f"CalibratedForecaster needs at least 2 classes; got {len(classes)}.")
        self._multiclass = len(classes) > 2
        X_tr, X_cal, y_tr, y_cal = train_test_split(
            X,
            y,
            test_size=self.calib_fraction,
            random_state=self.random_state,
            stratify=y,
        )
        self.base.fit(X_tr, y_tr)
        raw = np.asarray(self.base.predict_proba(X_cal), dtype=float)
        if self._multiclass:
            mc = MulticlassCalibrator(method=self.method)
            mc.fit(raw, np.asarray(y_cal))
            self._calibrator = mc
        else:
            cal = self._new_calibrator()
            cal.fit(raw[:, 1], np.asarray(y_cal))
            self._calibrator = cal
        self._feature_names = list(X.columns)
        self._fitted = True

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated (n, n_classes) probabilities."""
        if not self._fitted or self._calibrator is None:
            raise RuntimeError("CalibratedForecaster has not been fitted yet. Call fit() first.")
        raw = np.asarray(self.base.predict_proba(X), dtype=float)
        if self._multiclass:
            assert isinstance(self._calibrator, MulticlassCalibrator)
            return self._calibrator.transform(raw)
        pos = np.asarray(self._calibrator.transform(raw[:, 1]), dtype=float)
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
                    "multiclass": self._multiclass,
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
        instance._multiclass = data["multiclass"]
        instance._fitted = data["fitted"]
        instance._feature_names = data["feature_names"]
        return instance

    def feature_names(self) -> list[str]:
        """Return feature names from the last fit() call."""
        return list(self._feature_names)
