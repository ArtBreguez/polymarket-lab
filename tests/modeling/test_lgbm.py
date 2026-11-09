"""Tests for modeling.lgbm_baseline."""

import numpy as np
import pandas as pd

from pmlab.modeling.lgbm_baseline import LGBMForecaster


def _make_binary_data(n: int = 100) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "feature_a": rng.random(n),
            "feature_b": rng.random(n),
        }
    )
    y = pd.Series((X["feature_a"] > 0.5).astype(int))
    return X, y


def test_binary_shape():
    X, y = _make_binary_data()
    model = LGBMForecaster(objective="binary")
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)


def test_proba_sums_to_one():
    X, y = _make_binary_data()
    model = LGBMForecaster(objective="binary")
    model.fit(X, y)
    proba = model.predict_proba(X)
    row_sums = proba.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


def test_save_load(tmp_path):
    X, y = _make_binary_data()
    model = LGBMForecaster(objective="binary")
    model.fit(X, y)

    pkl_path = tmp_path / "model.pkl"
    model.save(pkl_path)

    loaded = LGBMForecaster.load(pkl_path)
    proba_orig = model.predict_proba(X)
    proba_loaded = loaded.predict_proba(X)
    np.testing.assert_allclose(proba_orig, proba_loaded, atol=1e-8)


def test_feature_names():
    X, y = _make_binary_data()
    model = LGBMForecaster(objective="binary")
    model.fit(X, y)
    assert model.feature_names() == ["feature_a", "feature_b"]


def test_predict_proba_matches_columns_by_name():
    """The same columns in another order give the same probabilities (#54): LightGBM matches a
    DataFrame positionally, so the forecaster must select by the names it was fitted on."""
    X, y = _make_binary_data()
    model = LGBMForecaster(n_estimators=10, n_jobs=1)
    model.fit(X, y)
    np.testing.assert_allclose(
        model.predict_proba(X[["feature_b", "feature_a"]]), model.predict_proba(X)
    )


def test_predict_proba_ignores_extra_columns():
    X, y = _make_binary_data()
    model = LGBMForecaster(n_estimators=10, n_jobs=1)
    model.fit(X, y)
    wider = X.assign(feature_zzz=1.0)[["feature_zzz", "feature_b", "feature_a"]]
    np.testing.assert_allclose(model.predict_proba(wider), model.predict_proba(X))


def test_predict_proba_names_the_missing_column():
    import pytest

    X, y = _make_binary_data()
    model = LGBMForecaster(n_estimators=10, n_jobs=1)
    model.fit(X, y)
    with pytest.raises(ValueError, match="feature_b"):
        model.predict_proba(X[["feature_a"]])


def test_reordered_columns_after_save_and_load(tmp_path):
    X, y = _make_binary_data()
    model = LGBMForecaster(n_estimators=10, n_jobs=1)
    model.fit(X, y)
    model.save(tmp_path / "m.pkl")
    loaded = LGBMForecaster.load(tmp_path / "m.pkl")
    np.testing.assert_allclose(
        loaded.predict_proba(X[["feature_b", "feature_a"]]), model.predict_proba(X)
    )
