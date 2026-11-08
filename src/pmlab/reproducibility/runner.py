"""The recipe of a backtest run, as data: resolve it, execute it, capture it.

``pmlab backtest`` and ``pmlab reproduce`` call the same functions here, so a re-run executes
exactly the code path of the original run rather than a re-implementation of it. A
:class:`BacktestConfig` holds the *resolved* parameters (every default spelled out), the seeds
live next to it in one dict (:func:`resolve_seeds`), and :func:`run_backtest` applies both.

Seeding, and where it happens:

* ``python_random`` / ``numpy`` — the process-wide generators, pinned by
  :func:`~pmlab.reproducibility.capture.seed_everything` before the run.
* ``lightgbm`` — passed to the forecaster as ``random_state``, the only place LightGBM reads a
  seed (bagging, feature sub-sampling, extra trees). It is *not* a model parameter of the
  config: there is exactly one place a seed can be set, and a second one is refused.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from pmlab.backtest.rolling_origin import RollingOriginResult, rolling_origin_eval
from pmlab.core.costs import CostModel
from pmlab.core.edge import DEFAULT_MIN_EDGE
from pmlab.core.policy import TradePolicy
from pmlab.modeling.lgbm_baseline import LGBMForecaster
from pmlab.reproducibility.capture import (
    ReproducibilityRecord,
    fingerprint_panel,
    seed_everything,
)

__all__ = [
    "DEFAULT_SEED",
    "BacktestConfig",
    "BacktestOutcome",
    "backtest_metrics",
    "build_forecaster",
    "capture_backtest",
    "load_panel",
    "resolve_seeds",
    "run_backtest",
]

DEFAULT_SEED = 0
_MODELS = ("lgbm_baseline",)
_SEED_PARAMS = ("random_state", "seed", "random_seed")


def resolve_seeds(seed: int = DEFAULT_SEED) -> dict[str, int]:
    """One root seed -> the seed of every generator a backtest draws from."""
    return {"python_random": seed, "numpy": seed, "lightgbm": seed}


@dataclass(frozen=True)
class BacktestConfig:
    """The resolved parameters of a backtest: nothing left to a default.

    Build one with :meth:`resolve`, which applies the defaults; ``from_dict`` reads one back
    from a record exactly as it was written.
    """

    plugin: str
    model: str
    model_params: dict[str, Any]
    stride: int
    min_train_rows: int
    flat_stake: float
    costs: dict[str, float]
    policy: dict[str, Any]

    @classmethod
    def resolve(
        cls,
        plugin: str,
        model: str = "lgbm_baseline",
        model_params: Mapping[str, Any] | None = None,
        stride: int = 30,
        min_train_rows: int = 20,
        flat_stake: float = 1.0,
        costs: CostModel | None = None,
        policy: TradePolicy | str = "live",
        min_edge: float | None = None,
    ) -> BacktestConfig:
        """Apply every default and validate. ``ValueError`` says what is wrong."""
        if model not in _MODELS:
            raise ValueError(f"unknown model {model!r}: only {list(_MODELS)} is supported")
        user = dict(model_params or {})
        seeded = [k for k in _SEED_PARAMS if k in user]
        if seeded:
            raise ValueError(
                f"model parameter(s) {seeded} set a seed, but a seed is not a model parameter: "
                "use --seed (or `seeds`), the one place a run's randomness is chosen and recorded"
            )
        if isinstance(policy, str):
            if policy == "live":
                policy = TradePolicy.live(
                    min_edge=DEFAULT_MIN_EDGE if min_edge is None else min_edge
                )
            elif policy == "top-bin-yes":
                if min_edge is not None:
                    raise ValueError("min_edge has no meaning under the top-bin-yes policy")
                policy = TradePolicy.top_bin_yes()
            else:
                raise ValueError(f"unknown policy {policy!r}: choose 'live' or 'top-bin-yes'")
        return cls(
            plugin=plugin,
            model=model,
            model_params=LGBMForecaster(**user).resolved_params(),
            stride=stride,
            min_train_rows=min_train_rows,
            flat_stake=flat_stake,
            costs=(costs or CostModel()).to_dict(),
            policy=policy.to_dict(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "plugin": self.plugin,
            "model": self.model,
            "model_params": dict(self.model_params),
            "stride": self.stride,
            "min_train_rows": self.min_train_rows,
            "flat_stake": self.flat_stake,
            "costs": dict(self.costs),
            "policy": dict(self.policy),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> BacktestConfig:
        missing = [
            k
            for k in (
                "plugin",
                "model",
                "model_params",
                "stride",
                "min_train_rows",
                "flat_stake",
                "costs",
                "policy",
            )
            if k not in d
        ]
        if missing:
            raise ValueError(f"backtest config is missing {missing}")
        return cls(
            plugin=str(d["plugin"]),
            model=str(d["model"]),
            model_params=dict(d["model_params"]),
            stride=int(d["stride"]),
            min_train_rows=int(d["min_train_rows"]),
            flat_stake=float(d["flat_stake"]),
            costs=dict(d["costs"]),
            policy=dict(d["policy"]),
        )


@dataclass
class BacktestOutcome:
    trades: pd.DataFrame
    steps: list[dict[str, Any]]
    metrics: dict[str, Any]


def build_forecaster(config: BacktestConfig, seeds: Mapping[str, int | None]) -> LGBMForecaster:
    """The forecaster of the run, seeded from ``seeds['lightgbm']``."""
    params = dict(config.model_params)
    objective = params.pop("objective", "binary")
    seed = seeds.get("lightgbm")
    if seed is not None:
        params["random_state"] = seed
    return LGBMForecaster(objective=objective, **params)


def backtest_metrics(result: RollingOriginResult) -> dict[str, Any]:
    """The summary a backtest is judged (and reproduced) by."""
    trades = result.trades
    base: dict[str, Any] = {"total_trades": len(trades), "steps": len(result.steps)}
    if trades.empty:
        return {**base, "total_pnl": 0.0, "hit_rate": None, "avg_edge": None}
    pnl = trades["realized_pnl"]
    return {
        **base,
        "total_pnl": float(pnl.sum()),
        "hit_rate": float((pnl > 0).mean()),
        "avg_edge": float(trades["edge"].mean()),
    }


def run_backtest(
    panel: pd.DataFrame, config: BacktestConfig, seeds: Mapping[str, int | None]
) -> BacktestOutcome:
    """Seed, then run the walk-forward backtest the config describes."""
    seed_everything(seeds)
    result = rolling_origin_eval(
        panel,
        build_forecaster(config, seeds),
        min_train_rows=config.min_train_rows,
        stride=config.stride,
        flat_stake=config.flat_stake,
        costs=CostModel(**config.costs),
        policy=TradePolicy.from_dict(config.policy),
    )
    return BacktestOutcome(result.trades, result.steps, backtest_metrics(result))


def capture_backtest(
    panel: pd.DataFrame,
    config: BacktestConfig,
    seeds: Mapping[str, int | None],
    metrics: Mapping[str, Any],
    source: str | None = None,
) -> ReproducibilityRecord:
    """The reproducibility record of a finished backtest."""
    return ReproducibilityRecord.capture(
        kind="backtest",
        seeds=seeds,
        config=config.to_dict(),
        data=fingerprint_panel(panel, source=source),
        metrics=metrics,
    )


def load_panel(path: str | Path) -> pd.DataFrame:
    """Read a ``.parquet`` or ``.csv`` panel; ``ValueError`` for any other extension."""
    file = Path(path)
    if file.suffix == ".parquet":
        return pd.read_parquet(file)
    if file.suffix == ".csv":
        return pd.read_csv(file)
    raise ValueError("Panel must be .parquet or .csv")
