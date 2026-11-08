"""A tuned run is reproducible, and its record says which objective picked the parameters.

`pmlab reproduce` re-executes the *backtest* a record describes, with the parameters the run
used: those are the tuned ones, so a tuned run reproduces like any other. What the record adds
(issue #36) is the `tuning` block: the objective the search maximised (metric, trade policy,
costs, how it walked forward) and, in `seeds`, the sampler seed. Together with the data
fingerprint that is enough to run the search again and land on the same parameters.
"""

from __future__ import annotations

import importlib.util
import json
from typing import Any

import pytest

from pmlab.core.costs import CostModel
from pmlab.core.policy import TradePolicy
from pmlab.modeling.lgbm_baseline import LGBMForecaster
from pmlab.modeling.tuning import TunedForecaster
from pmlab.reproducibility import (
    BacktestConfig,
    ReproducibilityRecord,
    capture_backtest,
    reproduce,
    resolve_seeds,
    run_backtest,
)
from tests.reproducibility.helpers import make_panel

LIVE = TradePolicy.live(min_edge=0.05)
COSTS = CostModel(taker_bps=30.0, slippage_bps=50.0)
needs_optuna = pytest.mark.skipif(
    importlib.util.find_spec("optuna") is None, reason="optuna not installed"
)


def _tuner(policy: TradePolicy | None, random_state: int = 3) -> TunedForecaster:
    def build(params: dict[str, Any]) -> LGBMForecaster:
        return LGBMForecaster(random_state=7, n_jobs=1, **params)

    def space(trial: Any) -> dict[str, Any]:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 10, 30, step=10),
            "num_leaves": trial.suggest_int("num_leaves", 4, 8),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 20),
        }

    return TunedForecaster(
        build,
        space,
        n_trials=4,
        random_state=random_state,
        stride=10,
        min_train_rows=60,
        policy=policy,
        costs=COSTS,
    )


def _tuned_run(policy: TradePolicy | None) -> tuple[ReproducibilityRecord, Any]:
    panel = make_panel(n_dates=80)
    tuner = _tuner(policy).tune(panel)
    config = BacktestConfig.resolve(
        plugin="weather_tmax",
        model_params=tuner.best_params_,
        stride=10,
        min_train_rows=60,
        costs=COSTS,
        policy=LIVE,
        tuning=tuner.tuning_config(),
    )
    seeds = resolve_seeds(11, optuna=tuner.random_state)
    outcome = run_backtest(panel, config, seeds)
    return capture_backtest(panel, config, seeds, outcome.metrics, source="p.parquet"), panel


class TestBacktestConfigTuning:
    def test_an_untuned_config_is_byte_identical_to_before(self) -> None:
        """No `tuning` key: records written without a tuner keep the exact shape they had."""
        config = BacktestConfig.resolve(plugin="weather_tmax")
        assert config.tuning is None
        assert "tuning" not in config.to_dict()
        assert BacktestConfig.from_dict(config.to_dict()) == config

    def test_a_record_written_before_tuning_existed_loads(self) -> None:
        old = BacktestConfig.resolve(plugin="weather_tmax").to_dict()
        assert BacktestConfig.from_dict(old).tuning is None

    def test_the_tuning_block_round_trips_through_json(self) -> None:
        block = {"metric": "total_pnl", "policy": LIVE.to_dict(), "costs": COSTS.to_dict()}
        config = BacktestConfig.resolve(plugin="weather_tmax", tuning=block)
        again = BacktestConfig.from_dict(json.loads(json.dumps(config.to_dict())))
        assert again == config
        assert again.tuning == block

    def test_tuning_must_be_a_mapping(self) -> None:
        with pytest.raises(ValueError, match="tuning"):
            BacktestConfig.resolve(plugin="weather_tmax", tuning="total_pnl")  # type: ignore[arg-type]


class TestSeeds:
    def test_default_seeds_are_unchanged(self) -> None:
        assert resolve_seeds(5) == {"python_random": 5, "numpy": 5, "lightgbm": 5}

    def test_the_sampler_seed_is_recorded_next_to_the_others(self) -> None:
        assert resolve_seeds(5, optuna=3) == {
            "python_random": 5,
            "numpy": 5,
            "lightgbm": 5,
            "optuna": 3,
        }


@needs_optuna
class TestATunedRunReproduces:
    def test_the_record_carries_the_objective_and_the_sampler_seed(self) -> None:
        record, _ = _tuned_run(LIVE)
        tuning = record.config["tuning"]
        assert tuning["policy"] == LIVE.to_dict()
        assert tuning["costs"] == COSTS.to_dict()
        assert tuning["metric"] == "total_pnl"
        assert record.seeds["optuna"] == 3
        # And the backtest it graded: the resolved params are the tuned ones.
        assert record.config["model_params"]["n_estimators"] in (10, 20, 30)

    def test_the_historical_objective_is_spelled_out_not_left_blank(self) -> None:
        record, _ = _tuned_run(None)
        assert record.config["tuning"]["policy"] == TradePolicy.top_bin_yes().to_dict()

    def test_reproduce_gives_the_same_metrics(self) -> None:
        record, panel = _tuned_run(LIVE)
        report = reproduce(ReproducibilityRecord.from_dict(record.to_dict()), panel=panel)
        assert report.reproduced, [c for c in report.comparisons if not c.ok]

    def test_reproduce_survives_the_json_file(self, tmp_path: Any) -> None:
        record, panel = _tuned_run(LIVE)
        path = tmp_path / "metrics.json"
        path.write_text(json.dumps({"reproducibility": record.to_dict()}, allow_nan=False))
        panel_path = tmp_path / "p.parquet"
        panel.to_parquet(panel_path)
        from pmlab.reproducibility import load_record

        report = reproduce(load_record(str(path)), panel_path=panel_path)
        assert report.reproduced

    def test_searching_again_with_the_recorded_objective_finds_the_same_params(self) -> None:
        """The recorded block + seed + data are enough to repeat the search."""
        record, panel = _tuned_run(LIVE)
        block = record.config["tuning"]
        again = TunedForecaster(
            _tuner(LIVE).build_fn,
            _tuner(LIVE).param_space,
            n_trials=block["n_trials"],
            metric=block["metric"],
            random_state=record.seeds["optuna"],
            stride=block["stride"],
            min_train_rows=block["min_train_rows"],
            flat_stake=block["flat_stake"],
            policy=TradePolicy.from_dict(block["policy"]),
            costs=CostModel(**block["costs"]),
        ).tune(panel)
        recorded = record.config["model_params"]
        assert all(recorded[k] == v for k, v in (again.best_params_ or {}).items())
        assert again.best_score_ == pytest.approx(block["best_score"])

    def test_two_objectives_leave_two_different_records(self) -> None:
        live, _ = _tuned_run(LIVE)
        historical, _ = _tuned_run(None)
        assert live.config["tuning"]["policy"] != historical.config["tuning"]["policy"]
