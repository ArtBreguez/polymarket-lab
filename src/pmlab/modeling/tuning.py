"""Hyperparameter tuning that respects the no-lookahead contract.

Naive tuning (random-shuffle CV) leaks the future into model selection on time
series — the single most common way a prediction-market backtest lies to you.
``TunedForecaster`` scores every Optuna trial through ``rolling_origin_eval``,
which trains strictly on ``decision_date < eval_date``, so the search itself is
walk-forward and lookahead-free.

What a trial is scored *on* is a choice, and it has to be the choice you will trade with.
By default a trial is the historical backtest: YES on the most probable bin of every market,
no threshold. Live trading does something else (per market the best after-fee edge, YES or
NO, only if it clears ``min_edge``), and a model tuned for one rule can be the wrong model for
the other. Pass ``policy=TradePolicy.live(min_edge=...)`` (and the ``costs`` you will trade
with) and every trial is scored by the same decision rule ``record-trades`` applies.

Optuna is an optional dependency: ``pip install pmlab[tune]``. Importing this
module works without it; only ``tune()`` requires it.
"""

from __future__ import annotations

import math
import pickle
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from pmlab.backtest.rolling_origin import rolling_origin_eval
from pmlab.core.costs import CostModel
from pmlab.core.policy import TradePolicy
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
        stride: Forwarded to ``rolling_origin_eval``.
        min_train_rows: Forwarded to ``rolling_origin_eval``.
        flat_stake: Forwarded to ``rolling_origin_eval``.
        taker_bps: Legacy fee-only knob (default 30). Mutually exclusive with ``costs``.
        policy: The decision rule every trial is scored under, a
            `TradePolicy`. Omit it for the historical objective
            (YES on the top-probability bin, every market: the search that shipped before
            policies existed, unchanged). Pass ``TradePolicy.live(min_edge=...)`` to tune for
            what ``record-trades`` trades: the best after-fee edge per market, YES or NO,
            taken if it clears ``min_edge``. Use the ``min_edge`` you will trade with.
        costs: The `CostModel` a trial's fills are priced with
            (fee, slippage, depth). Under the live policy it decides which trades clear the
            threshold, so tune with the costs you will trade with. Omit it for the 30 bps
            fee, fill at the quote.

    After ``tune(panel)``, the instance is a fitted forecaster (best params,
    refit on the full panel) and exposes ``best_params_``, ``best_score_``,
    and ``study_``. `tuning_config` returns the objective it maximised, resolved,
    for the reproducibility record (``BacktestConfig.resolve(tuning=...)``).

    A trial that makes no trade scores ``-inf``; if *every* trial does (a threshold nothing
    clears), ``tune`` warns, because the search learned nothing and ``best_params_`` is
    just the first trial.
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
        taker_bps: float | None = None,
        policy: TradePolicy | None = None,
        costs: CostModel | None = None,
    ) -> None:
        if n_trials <= 0:
            raise ValueError(f"n_trials must be > 0; got {n_trials}.")
        if metric not in _METRICS:
            raise ValueError(f"Unknown metric '{metric}'. Choose from {_METRICS}.")
        if policy is not None and not isinstance(policy, TradePolicy):
            raise TypeError(
                f"policy must be a TradePolicy (e.g. TradePolicy.live(min_edge=0.05)) or None "
                f"for the historical objective; got {policy!r}."
            )
        if costs is not None and not isinstance(costs, CostModel):
            raise TypeError(f"costs must be a CostModel or None; got {costs!r}.")
        if taker_bps is not None and costs is not None:
            raise ValueError(
                "pass either taker_bps or costs, not both: costs.taker_bps already carries the fee"
            )
        self.build_fn = build_fn
        self.param_space = param_space
        self.n_trials = n_trials
        self.metric: TuneMetric = metric
        self.random_state = random_state
        self.stride = stride
        self.min_train_rows = min_train_rows
        self.flat_stake = flat_stake
        self.policy = policy
        self.costs: CostModel = (
            costs
            if costs is not None
            else (CostModel() if taker_bps is None else CostModel(taker_bps=taker_bps))
        )
        self.taker_bps: float = self.costs.taker_bps

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
                costs=self.costs,
                policy=self.policy,
            )
            return _score(result.trades, self.metric)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=self.n_trials)

        self.study_ = study
        self.best_params_ = dict(study.best_params)
        self.best_score_ = float(study.best_value)
        if self.best_score_ == float("-inf"):
            warnings.warn(
                "no trial produced a trade under the objective "
                f"(policy={self._policy_dict()}): every trial scored -inf, so the search "
                "learned nothing and best_params_ is just the first trial. Lower min_edge, "
                "or give the search more data or a smaller stride.",
                UserWarning,
                stacklevel=2,
            )

        # Refit the winning config on the entire panel so the tuner is usable.
        feature_cols = [c for c in panel.columns if c.startswith("feature_")]
        X = panel[feature_cols]
        y = (panel["outcome_label"] == panel["winning_label"]).astype(int)
        self._model = self.build_fn(self.best_params_)
        self._model.fit(X, y)
        self._feature_names = feature_cols
        return self

    def _policy_dict(self) -> dict[str, Any]:
        """The policy trials are scored under, spelled out (the default is the historical one)."""
        return (self.policy or TradePolicy.top_bin_yes()).to_dict()

    def tuning_config(self) -> dict[str, Any]:
        """The objective the search maximised, resolved and strict-JSON, for the record.

        Pass it to ``BacktestConfig.resolve(tuning=...)`` so the reproducibility record of the
        backtest says how its parameters were chosen: which metric, under which trade policy
        and costs, walking forward how. Every default is spelled out. The sampler seed is not
        in here: a run's seeds live in one place, ``resolve_seeds(seed, optuna=tuner.random_state)``.
        ``best_score`` is ``None`` before ``tune`` and for a search that never traded.
        """
        finite = math.isfinite(self.best_score_)
        return {
            "metric": self.metric,
            "n_trials": self.n_trials,
            "policy": self._policy_dict(),
            "costs": self.costs.to_dict(),
            "stride": self.stride,
            "min_train_rows": self.min_train_rows,
            "flat_stake": self.flat_stake,
            "best_score": self.best_score_ if finite else None,
        }

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
                    "random_state": self.random_state,
                    "tuning": self.tuning_config(),
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
        instance.metric = data["metric"]
        instance.random_state = data.get("random_state", 0)
        # A tuner saved before policies existed has no objective on file: it was tuned by the
        # historical one, which is exactly what the defaults below say.
        tuning = data.get("tuning") or {}
        instance.n_trials = tuning.get("n_trials", 0)
        instance.stride = tuning.get("stride", 10)
        instance.min_train_rows = tuning.get("min_train_rows", 20)
        instance.flat_stake = tuning.get("flat_stake", 1.0)
        policy = tuning.get("policy")
        stored = TradePolicy.from_dict(policy) if policy is not None else None
        instance.policy = None if stored == TradePolicy.top_bin_yes() else stored
        instance.costs = CostModel(**tuning["costs"]) if "costs" in tuning else CostModel()
        instance.taker_bps = instance.costs.taker_bps
        instance.best_params_ = data["best_params"]
        instance.best_score_ = data["best_score"]
        instance.study_ = None
        instance._model = data["model"]
        instance._feature_names = data["feature_names"]
        return instance

    def feature_names(self) -> list[str]:
        """Return feature names from the last fit/tune call."""
        return list(self._feature_names)
