"""Abstract base class for market forecasters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class MarketForecaster(ABC):
    """Abstract base class for all market forecasters."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model on training data."""
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability estimates."""
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Serialize model to disk."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> MarketForecaster:
        """Deserialize model from disk."""
        ...

    def feature_names(self) -> list[str]:
        """Return list of feature names used by this model."""
        return []
