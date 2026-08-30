"""Hyperparameter tuning that respects the no-lookahead contract.

Naive tuning (random-shuffle CV) leaks the future into model selection on time
series — the single most common way a prediction-market backtest lies to you.
``TunedForecaster`` scores every Optuna trial through ``rolling_origin_eval``,
which trains strictly on ``decision_date < eval_date``, so the search itself is
walk-forward and lookahead-free.

Optuna is an optional dependency: ``pip install pmlab[tune]``. Importing this
module works without it; only ``tune()`` requires it.
"""

from __future__ import annotations

import pickle
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from pmlab.backtest.rolling_origin import rolling_origin_eval
from pmlab.modeling.base import MarketForecaster

if TYPE_CHECKING:
    import optuna

TuneMetric = Literal["total_pnl", "mean_pnl", "hit_rate"]
_METRICS = ("total_pnl", "mean_pnl", "hit_rate")


def _score(trades: pd.DataFrame, metric: TuneMetric) -> float:
    """Reduce a rolling-origin trade log to a single scalar to maximize."""
    if trades.empty:
        return float("-inf")
    pnl = trades["realized_pnl"]
    if metric == "total_pnl":
        return float(pnl.sum())
    if metric == "mean_pnl":
        return float(pnl.mean())
    return float((pnl > 0).mean())  # hit_rate


class TunedForecaster(MarketForecaster):
    """Optuna hyperparameter search scored by walk-forward backtest.

    Args:
        build_fn: ``params -> MarketForecaster``. Given the concrete params Optuna
            picked for a trial, return an (unfitted) forecaster.
        param_space: ``trial -> params dict``. Uses the Optuna ``trial`` API
            (``trial.suggest_int`` etc.) to define the search space.
        n_trials: Number of Optuna trials (> 0).
        metric: What to maximize — ``"total_pnl"`` (default), ``"mean_pnl"``, or
            ``"hit_rate"``.
        random_state: Seed for Optuna's sampler (reproducible search).
        stride / min_train_rows / flat_stake / taker_bps: forwarded to
            ``rolling_origin_eval`` so the scoring backtest matches production.

    After ``tune(panel)``, the instance is a fitted forecaster (best params,
    refit on the full panel) and exposes ``best_params_``, ``best_score_``,
    and ``study_``.
    """

    def __init__(
        self,
        build_fn: Callable[[dict[str, Any]], MarketForecaster],
        param_space: Callable[[optuna.Trial], dict[str, Any]],
        n_trials: int = 20,
        metric: TuneMetric = "total_pnl",
        random_state: int = 0,
        stride: int = 10,
        min_train_rows: int = 20,
        flat_stake: float = 1.0,
        taker_bps: float = 30.0,
    ) -> None:
        if n_trials <= 0:
            raise ValueError(f"n_trials must be > 0; got {n_trials}.")
        if metric not in _METRICS:
            raise ValueError(f"Unknown metric '{metric}'. Choose from {_METRICS}.")
        self.build_fn = build_fn
        self.param_space = param_space
        self.n_trials = n_trials
        self.metric: TuneMetric = metric
        self.random_state = random_state
        self.stride = stride
        self.min_train_rows = min_train_rows
        self.flat_stake = flat_stake
        self.taker_bps = taker_bps

        self.best_params_: dict[str, Any] | None = None
        self.best_score_: float = float("-inf")
        self.study_: optuna.Study | None = None
        self._model: MarketForecaster | None = None
        self._feature_names: list[str] = []

    def tune(self, panel: pd.DataFrame) -> TunedForecaster:
        """Run the Optuna search over ``panel``, then refit best on the full panel."""
        try:
            import optuna
        except ImportError as exc:  # pragma: no cover - exercised only without optuna
            raise ImportError(
                "TunedForecaster.tune requires optuna. Install with: pip install 'pmlab[tune]'."
            ) from exc

        def objective(trial: optuna.Trial) -> float:
            params = self.param_space(trial)
            model = self.build_fn(params)
            result = rolling_origin_eval(
                panel,
                model,
                min_train_rows=self.min_train_rows,
                stride=self.stride,
                flat_stake=self.flat_stake,
                taker_bps=self.taker_bps,
            )
            return _score(result.trades, self.metric)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=self.n_trials)

        self.study_ = study
        self.best_params_ = dict(study.best_params)
        self.best_score_ = float(study.best_value)

        # Refit the winning config on the entire panel so the tuner is usable.
        feature_cols = [c for c in panel.columns if c.startswith("feature_")]
        X = panel[feature_cols]
        y = (panel["outcome_label"] == panel["winning_label"]).astype(int)
        self._model = self.build_fn(self.best_params_)
        self._model.fit(X, y)
        self._feature_names = feature_cols
        return self

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the current best model directly (bypasses search).

        Available after ``tune``; use ``tune`` for the walk-forward search. This
        exists so a TunedForecaster still satisfies the MarketForecaster contract.
        """
        if self.best_params_ is None:
            raise RuntimeError("Call tune() before fit(), or use tune() to run the search.")
        if self.build_fn is None:
            raise RuntimeError(
                "This TunedForecaster was restored from disk (build_fn is not picklable). "
                "Use predict_proba directly; to refit, construct a fresh TunedForecaster."
            )
        self._model = self.build_fn(self.best_params_)
        self._model.fit(X, y)
        self._feature_names = list(X.columns)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Delegate to the tuned-and-fitted model."""
        if self._model is None:
            raise RuntimeError("TunedForecaster has not been tuned yet. Call tune() first.")
        return np.asarray(self._model.predict_proba(X), dtype=float)

    def save(self, path: Path) -> None:
        """Pickle-serialize the fitted best model + search metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self._model,
                    "best_params": self.best_params_,
                    "best_score": self.best_score_,
                    "metric": self.metric,
                    "feature_names": self._feature_names,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> TunedForecaster:
        """Load a tuned forecaster. The Optuna study itself is not restored."""
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls.__new__(cls)
        instance.build_fn = None  # type: ignore[assignment]
        instance.param_space = None  # type: ignore[assignment]
        instance.n_trials = 0
        instance.metric = data["metric"]
        instance.random_state = 0
        instance.stride = 10
        instance.min_train_rows = 20
        instance.flat_stake = 1.0
        instance.taker_bps = 30.0
        instance.best_params_ = data["best_params"]
        instance.best_score_ = data["best_score"]
        instance.study_ = None
        instance._model = data["model"]
        instance._feature_names = data["feature_names"]
        return instance

    def feature_names(self) -> list[str]:
        """Return feature names from the last fit/tune call."""
        return list(self._feature_names)
