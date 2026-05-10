"""Tests for modeling.base ABC."""

import numpy as np
import pandas as pd
import pytest

from pmlab.modeling.base import MarketForecaster


def test_cannot_instantiate_abc():
    with pytest.raises(TypeError):
        MarketForecaster()  # type: ignore[abstract]


def test_dummy_fit_predict():
    """A minimal concrete subclass should work."""

    class DummyForecaster(MarketForecaster):
        def fit(self, X, y):
            pass

        def predict_proba(self, X):
            return np.full((len(X), 2), 0.5)

        def save(self, path):
            pass

        @classmethod
        def load(cls, path):
            return cls()

    m = DummyForecaster()
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([0, 1, 0])

    m.fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (3, 2)
    assert m.feature_names() == []
