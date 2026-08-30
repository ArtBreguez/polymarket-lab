"""Split-conformal classification — prediction sets with a coverage guarantee.

A point probability says *how likely*; a conformal prediction *set* says *which
outcomes can't be ruled out* at a chosen confidence. On exchangeable data,
split-conformal guarantees marginal coverage: the returned set contains the true
label with probability >= 1 - alpha, regardless of whether the base model is
well-specified. That distribution-free guarantee is what makes it valuable for
sizing decisions on noisy prediction markets — a singleton set is a confident
call; a full set {0, 1} is the model honestly saying "toss-up, don't bet".

Method (LAC / "score = 1 - p_true"):
  1. Split fit data into proper-train and calibration.
  2. Fit the base model on proper-train.
  3. Nonconformity score on calibration = 1 - p(model, true label).
  4. qhat = the ceil((n+1)(1-alpha))/n empirical quantile of those scores.
  5. Prediction set for x = { class k : 1 - p_k(x) <= qhat }.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pmlab.modeling.base import MarketForecaster


class ConformalForecaster(MarketForecaster):
    """Wrap a ``MarketForecaster`` to emit calibrated prediction sets.

    Args:
        base: The forecaster to wrap.
        alpha: Miscoverage level in (0, 1). Target coverage is ``1 - alpha``
            (e.g. ``alpha=0.1`` → 90% sets).
        calib_fraction: Fraction of ``fit`` data held out for calibration.
        random_state: Seed for the calibration split.

    ``predict_proba`` passes through the base model's probabilities (so an
    instance is still a drop-in forecaster); ``predict_set`` returns the
    coverage-guaranteed sets.
    """

    def __init__(
        self,
        base: MarketForecaster,
        alpha: float = 0.1,
        calib_fraction: float = 0.3,
        random_state: int = 0,
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1); got {alpha}.")
        self.base = base
        self.alpha = alpha
        self.calib_fraction = calib_fraction
        self.random_state = random_state
        self._qhat: float | None = None
        self._classes: np.ndarray | None = None
        self._fitted = False
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit base on a proper-train split, calibrate qhat on the held-out split."""
        y_arr = np.asarray(y)
        X_tr, X_cal, y_tr, y_cal = train_test_split(
            X,
            y,
            test_size=self.calib_fraction,
            random_state=self.random_state,
            stratify=y_arr if len(np.unique(y_arr)) > 1 else None,
        )
        self.base.fit(X_tr, y_tr)
        self._classes = np.unique(y_arr)

        proba_cal = np.asarray(self.base.predict_proba(X_cal), dtype=float)
        y_cal_arr = np.asarray(y_cal)
        # Column index of each true label within the class ordering.
        class_to_col = {c: i for i, c in enumerate(self._classes)}
        true_cols = np.array([class_to_col[c] for c in y_cal_arr])
        p_true = proba_cal[np.arange(len(y_cal_arr)), true_cols]
        scores = 1.0 - p_true  # nonconformity: higher = worse fit

        n = len(scores)
        # Finite-sample-adjusted quantile level; clip to [0, 1].
        level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self._qhat = float(np.quantile(scores, level, method="higher"))

        self._feature_names = list(X.columns)
        self._fitted = True

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Pass through the base model's probability matrix, shape (n, n_classes)."""
        if not self._fitted:
            raise RuntimeError("ConformalForecaster has not been fitted yet. Call fit() first.")
        return np.asarray(self.base.predict_proba(X), dtype=float)

    def predict_set(self, X: pd.DataFrame) -> list[set[object]]:
        """Return one prediction set per row: classes with 1 - p_k <= qhat.

        Every set is guaranteed non-empty (the argmax class always qualifies),
        so a downstream consumer never has to special-case the empty set. Label
        types are preserved (int, str, etc.) — categorical markets (e.g. F1 driver
        names) get their real labels back, not coerced integers.
        """
        if not self._fitted or self._qhat is None or self._classes is None:
            raise RuntimeError("ConformalForecaster has not been fitted yet. Call fit() first.")
        proba = np.asarray(self.base.predict_proba(X), dtype=float)
        scores = 1.0 - proba  # (n, n_classes)
        classes = self._classes

        def _label(k: int) -> object:
            # Return a native Python scalar (int/str/float), never a numpy type.
            val = classes[k]
            return val.item() if hasattr(val, "item") else val

        out: list[set[object]] = []
        for row_scores in scores:
            members = {_label(k) for k in range(len(classes)) if row_scores[k] <= self._qhat}
            if not members:  # numerical edge: keep the single best class
                members = {_label(int(np.argmin(row_scores)))}
            out.append(members)
        return out

    def save(self, path: Path) -> None:
        """Pickle-serialize the wrapper (base + qhat + classes) to path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "base": self.base,
                    "alpha": self.alpha,
                    "calib_fraction": self.calib_fraction,
                    "random_state": self.random_state,
                    "qhat": self._qhat,
                    "classes": self._classes,
                    "fitted": self._fitted,
                    "feature_names": self._feature_names,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> ConformalForecaster:
        """Load a previously saved ConformalForecaster from path."""
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls(
            base=data["base"],
            alpha=data["alpha"],
            calib_fraction=data["calib_fraction"],
            random_state=data["random_state"],
        )
        instance._qhat = data["qhat"]
        instance._classes = data["classes"]
        instance._fitted = data["fitted"]
        instance._feature_names = data["feature_names"]
        return instance

    def feature_names(self) -> list[str]:
        """Return feature names from the last fit() call."""
        return list(self._feature_names)
