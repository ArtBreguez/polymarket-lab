"""Generic scikit-learn forecaster wrapper.

Provides a MarketForecaster implementation for any sklearn classifier,
with named presets for common algorithms.

Useful as a quick baseline before committing to LightGBM.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from pmlab.modeling.base import MarketForecaster

__all__ = ["SklearnForecaster"]

# Named presets: string -> (class, default_kwargs)
_PRESETS: dict[str, tuple[type[ClassifierMixin], dict[str, Any]]] = {
    "logistic_regression": (LogisticRegression, {"max_iter": 1000, "C": 1.0}),
    "random_forest": (RandomForestClassifier, {"n_estimators": 100}),
    "gradient_boosting": (GradientBoostingClassifier, {"n_estimators": 100}),
}


class SklearnForecaster(MarketForecaster):
    """sklearn-based market forecaster supporting named presets and custom estimators.

    Args:
        estimator: One of 'logistic_regression' (default), 'random_forest',
                   'gradient_boosting', or a pre-instantiated sklearn classifier.
        **kwargs: Extra kwargs forwarded to the preset constructor (ignored when
                  a pre-instantiated estimator is passed).

    Examples::

        # Named preset
        f = SklearnForecaster(estimator="random_forest", n_estimators=50)

        # Custom estimator
        from sklearn.tree import DecisionTreeClassifier
        f = SklearnForecaster(estimator=DecisionTreeClassifier(max_depth=4))
    """

    def __init__(
        self,
        estimator: Union[str, ClassifierMixin] = "logistic_regression",
        **kwargs: Any,
    ) -> None:
        if isinstance(estimator, str):
            if estimator not in _PRESETS:
                raise ValueError(
                    f"Unknown estimator '{estimator}'. "
                    f"Choose from {sorted(_PRESETS)} or pass a sklearn classifier instance."
                )
            cls, defaults = _PRESETS[estimator]
            merged = {**defaults, **kwargs}
            self._estimator: ClassifierMixin = cls(**merged)
        else:
            self._estimator = estimator

        self._fitted = False
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the sklearn estimator on training data."""
        self._estimator.fit(X, y)
        self._feature_names = list(X.columns)
        self._fitted = True

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability estimates, shape (n_samples, n_classes)."""
        if not self._fitted:
            raise RuntimeError(
                "SklearnForecaster has not been fitted yet. Call fit() first."
            )
        result: np.ndarray = np.array(self._estimator.predict_proba(X))
        return result

    def save(self, path: Path) -> None:
        """Pickle-serialize the forecaster to path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "estimator": self._estimator,
                    "fitted": self._fitted,
                    "feature_names": self._feature_names,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> SklearnForecaster:
        """Load a previously saved SklearnForecaster from path."""
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls(estimator=data["estimator"])
        instance._fitted = data["fitted"]
        instance._feature_names = data["feature_names"]
        return instance

    def feature_names(self) -> list[str]:
        """Return feature names from the last fit() call."""
        return list(self._feature_names)
