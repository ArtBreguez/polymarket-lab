"""LightGBM-based market forecaster."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from pmlab.modeling.base import MarketForecaster


class LGBMForecaster(MarketForecaster):
    """LightGBM wrapper supporting binary and multiclass classification."""

    def __init__(self, objective: str = "binary", **lgbm_params: Any) -> None:
        self.objective = objective
        self.lgbm_params = lgbm_params
        self._model: lgb.LGBMClassifier | None = None
        self._feature_names: list[str] = []

    def resolved_params(self) -> dict[str, Any]:
        """The parameters LightGBM is actually given: the defaults, then the caller's overrides.

        This is what a reproducibility record stores as the model config, so a later change of
        a default here does not change what an old record means. The seed is part of it only
        when the caller passed ``random_state`` (LightGBM otherwise uses its own fixed internal
        seeds: deterministic, but not a value pmlab can record).
        """
        return {"objective": self.objective, "verbose": -1, **self.lgbm_params}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit LightGBM classifier."""
        self._model = lgb.LGBMClassifier(**self.resolved_params())
        self._model.fit(X, y)
        self._feature_names = list(X.columns)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability estimates, shape (n_samples, n_classes)."""
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet.")
        if self._feature_names and isinstance(X, pd.DataFrame):
            # LightGBM matches a DataFrame by position, so the same columns in another order
            # would silently give other probabilities (#54): select by the fitted names.
            missing = [c for c in self._feature_names if c not in X.columns]
            if missing:
                raise ValueError(
                    f"predict_proba is missing feature column(s) the model was fitted on: {missing}"
                )
            X = X[self._feature_names]
        result: np.ndarray = np.array(self._model.predict_proba(X))
        return result

    def save(self, path: Path) -> None:
        """Pickle-serialize the model to path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "objective": self.objective,
                    "lgbm_params": self.lgbm_params,
                    "model": self._model,
                    "feature_names": self._feature_names,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> LGBMForecaster:
        """Load a previously saved LGBMForecaster from path."""
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls(objective=data["objective"], **data["lgbm_params"])
        instance._model = data["model"]
        instance._feature_names = data["feature_names"]
        return instance

    def feature_names(self) -> list[str]:
        return list(self._feature_names)
