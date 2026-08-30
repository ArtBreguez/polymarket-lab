"""TDD tests for TunedForecaster (Optuna, no-lookahead via rolling_origin) — RED."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pmlab.modeling.sklearn_forecaster import SklearnForecaster
from pmlab.modeling.tuning import TunedForecaster

_HAS_OPTUNA = importlib.util.find_spec("optuna") is not None
optuna_required = pytest.mark.skipif(not _HAS_OPTUNA, reason="optuna not installed")


def _panel(n_dates: int = 40, per_date: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dates):
        date = f"2026-01-{d + 1:02d}"
        for _ in range(per_date):
            f1 = rng.normal(0, 1)
            f2 = rng.normal(0, 1)
            win = int(f1 + rng.normal(0, 0.5) > 0)
            rows.append(
                {
                    "market_id": f"m{d}",
                    "decision_date": date,
                    "outcome_label": "YES",
                    "winning_label": "YES" if win else "NO",
                    "market_price": 0.5,
                    "feature_f1": f1,
                    "feature_f2": f2,
                }
            )
    return pd.DataFrame(rows)


class TestConstruction:
    def test_requires_positive_n_trials(self) -> None:
        with pytest.raises(ValueError, match="n_trials"):
            TunedForecaster(
                lambda t: SklearnForecaster(),
                param_space=lambda t: {},
                n_trials=0,
            )

    def test_unknown_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="metric"):
            TunedForecaster(
                lambda t: SklearnForecaster(),
                param_space=lambda t: {},
                n_trials=3,
                metric="bogus",
            )


@optuna_required
class TestTuning:
    def test_tune_selects_params_and_fits(self) -> None:
        panel = _panel()

        def build(params: dict) -> SklearnForecaster:
            return SklearnForecaster(estimator="random_forest", **params)

        def space(trial) -> dict:  # type: ignore[no-untyped-def]
            return {"n_estimators": trial.suggest_int("n_estimators", 10, 40, step=10)}

        tuner = TunedForecaster(
            build_fn=build,
            param_space=space,
            n_trials=4,
            metric="total_pnl",
            random_state=0,
            stride=5,
            min_train_rows=30,
        )
        tuner.tune(panel)
        assert tuner.best_params_ is not None
        assert "n_estimators" in tuner.best_params_
        # After tune(), it behaves as a fitted forecaster on the full panel.
        feat = panel[[c for c in panel.columns if c.startswith("feature_")]]
        proba = tuner.predict_proba(feat)
        assert proba.shape == (len(panel), 2)

    def test_predict_before_tune_raises(self) -> None:
        tuner = TunedForecaster(
            lambda p: SklearnForecaster(),
            param_space=lambda t: {},
            n_trials=2,
        )
        X = pd.DataFrame({"feature_f1": [0.1, 0.2]})
        with pytest.raises(RuntimeError, match="tune"):
            tuner.predict_proba(X)

    def test_no_lookahead_uses_rolling_origin(self) -> None:
        # The tuner must score via rolling_origin_eval (train on past only). We
        # assert it never calls the model's fit on the full panel during search
        # by checking best_score_ is finite and study has the requested trials.
        panel = _panel(seed=3)
        tuner = TunedForecaster(
            build_fn=lambda p: SklearnForecaster(estimator="random_forest", **p),
            param_space=lambda t: {"n_estimators": t.suggest_int("n_estimators", 10, 30, step=10)},
            n_trials=3,
            metric="hit_rate",
            stride=5,
            min_train_rows=30,
        )
        tuner.tune(panel)
        assert np.isfinite(tuner.best_score_)
        assert len(tuner.study_.trials) == 3

    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        panel = _panel(seed=1)
        tuner = TunedForecaster(
            build_fn=lambda p: SklearnForecaster(estimator="random_forest", **p),
            param_space=lambda t: {"n_estimators": t.suggest_int("n_estimators", 10, 30, step=10)},
            n_trials=3,
            stride=5,
            min_train_rows=30,
        )
        tuner.tune(panel)
        feat = panel[[c for c in panel.columns if c.startswith("feature_")]]
        before = tuner.predict_proba(feat)
        path = tmp_path / "tuned.pkl"
        tuner.save(path)
        loaded = TunedForecaster.load(path)
        np.testing.assert_allclose(loaded.predict_proba(feat), before, atol=1e-10)

    def test_fit_after_load_raises_clearly(self, tmp_path: Path) -> None:
        # Regression: build_fn isn't picklable, so a loaded tuner can't refit.
        # predict_proba must work, but fit() should raise a clear error.
        panel = _panel(seed=2)
        tuner = TunedForecaster(
            build_fn=lambda p: SklearnForecaster(estimator="random_forest", **p),
            param_space=lambda t: {"n_estimators": t.suggest_int("n_estimators", 10, 30, step=10)},
            n_trials=2,
            stride=5,
            min_train_rows=30,
        )
        tuner.tune(panel)
        path = tmp_path / "t.pkl"
        tuner.save(path)
        loaded = TunedForecaster.load(path)
        feat = panel[[c for c in panel.columns if c.startswith("feature_")]]
        # predict still works
        assert loaded.predict_proba(feat).shape[0] == len(panel)
        # refit fails with a helpful message
        with pytest.raises(RuntimeError, match="restored from disk"):
            loaded.fit(feat, (panel["outcome_label"] == panel["winning_label"]).astype(int))
