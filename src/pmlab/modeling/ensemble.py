"""Ensemble forecaster — blend several MarketForecasters into one.

Combining independently-trained models is one of the most reliable ways to cut
variance and improve calibration on noisy prediction-market data. This wraps N
member forecasters behind the same ``MarketForecaster`` interface, so an ensemble
is a drop-in anywhere a single model is accepted (backtest, champion, brokers).

The blend is a (weighted) average of member ``predict_proba`` outputs. Members
must agree on the class layout — they are trained on the same ``(X, y)``, so the
column order of their probability matrices matches.
"""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from pmlab.modeling.base import MarketForecaster


class EnsembleForecaster(MarketForecaster):
    """Weighted-average ensemble of member forecasters.

    Args:
        forecasters: One or more fitted-or-unfitted ``MarketForecaster`` members.
            ``fit`` trains every member on the same data; if you prefer to blend
            already-trained models, pass them in and call ``fit`` anyway (cheap)
            or set them up before use — ``predict_proba`` only needs them fitted.
        weights: Optional non-negative blend weights, one per member. Defaults to
            equal weight. Normalized internally, so ``[3, 1]`` == ``[0.75, 0.25]``.

    Example::

        ens = EnsembleForecaster(
            forecasters=[LGBMForecaster(), SklearnForecaster("random_forest")],
            weights=[2.0, 1.0],
        )
        ens.fit(X_train, y_train)
        proba = ens.predict_proba(X_test)
    """

    def __init__(
        self,
        forecasters: Sequence[MarketForecaster],
        weights: Sequence[float] | None = None,
    ) -> None:
        if len(forecasters) == 0:
            raise ValueError("EnsembleForecaster needs at least one member forecaster.")

        if weights is None:
            weights = [1.0] * len(forecasters)
        if len(weights) != len(forecasters):
            raise ValueError(
                f"weights length ({len(weights)}) must match number of "
                f"forecasters ({len(forecasters)})."
            )
        if any(w < 0 for w in weights):
            raise ValueError("weights must be non-negative.")
        total = float(sum(weights))
        if total <= 0.0:
            raise ValueError("weights must sum to a positive value.")

        self.forecasters: list[MarketForecaster] = list(forecasters)
        self._weights: np.ndarray = np.asarray(weights, dtype=float) / total
        self._fitted = False
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit every member on the same training data."""
        for member in self.forecasters:
            member.fit(X, y)
        self._feature_names = list(X.columns)
        self._fitted = True

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Weighted-average of member probability matrices, shape (n, n_classes)."""
        if not self._fitted:
            raise RuntimeError("EnsembleForecaster has not been fitted yet. Call fit() first.")
        blended: np.ndarray | None = None
        for weight, member in zip(self._weights, self.forecasters, strict=True):
            proba = np.asarray(member.predict_proba(X), dtype=float)
            contribution = weight * proba
            blended = contribution if blended is None else blended + contribution
        assert blended is not None  # guaranteed: >=1 member
        # Renormalize defensively so rows sum to exactly 1 despite float drift.
        row_sums = blended.sum(axis=1, keepdims=True)
        return np.asarray(blended / row_sums)

    def save(self, path: Path) -> None:
        """Pickle-serialize the ensemble (members included) to path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "forecasters": self.forecasters,
                    "weights": self._weights,
                    "fitted": self._fitted,
                    "feature_names": self._feature_names,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> EnsembleForecaster:
        """Load a previously saved EnsembleForecaster from path."""
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls(forecasters=data["forecasters"])
        instance._weights = data["weights"]
        instance._fitted = data["fitted"]
        instance._feature_names = data["feature_names"]
        return instance

    def feature_names(self) -> list[str]:
        """Return feature names from the last fit() call."""
        return list(self._feature_names)
