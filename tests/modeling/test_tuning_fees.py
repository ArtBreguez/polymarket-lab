"""`TunedForecaster` scores its trials under the Polymarket fee model, and records which one (#72).

The tuner hands its `CostModel` to every trial's backtest, so the model it holds is the objective it
optimises. A tuner saved before 1.5.0 has the flat 30 bps in its cost dict (or no cost dict at all,
when it is older than the cost model): it must load as the objective it was tuned under, not as the
new default.
"""

from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path
from typing import Any

import pytest

from pmlab import PmlabDeprecationWarning
from pmlab.backtest.rolling_origin import rolling_origin_eval
from pmlab.core.costs import CostModel
from pmlab.core.fee_schedule import FeeSchedule
from pmlab.modeling.tuning import TunedForecaster
from tests.modeling.tuning_helpers import PickFeature, build_pick, crafted_panel, pick_space

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("optuna") is None, reason="optuna not installed"
)

FLAT = CostModel(fee_model="flat-bps")


def _tuner(**kwargs: Any) -> TunedForecaster:
    base: dict[str, Any] = {"n_trials": 4, "stride": 1, "min_train_rows": 20, "random_state": 0}
    return TunedForecaster(build_pick, pick_space, **{**base, **kwargs})


class TestTheTunerHoldsTheNewDefault:
    def test_the_default_objective_is_priced_with_the_polymarket_fee(self) -> None:
        tuner = _tuner()
        assert tuner.costs.fee_model == "polymarket"
        assert tuner.costs == CostModel()

    def test_a_trials_score_is_the_backtest_score_under_those_fees(self) -> None:
        panel = crafted_panel()
        schedule_costs = CostModel(fee_schedule=FeeSchedule(rate=0.02), fee_source="gamma")
        tuner = _tuner(costs=schedule_costs).tune(panel)
        assert tuner.study_ is not None
        for trial in tuner.study_.trials:
            trades = rolling_origin_eval(
                panel,
                PickFeature(trial.params["col"]),
                min_train_rows=20,
                stride=1,
                costs=schedule_costs,
            ).trades
            assert trial.value == pytest.approx(float(trades["realized_pnl"].sum()))

    def test_the_fee_model_changes_the_score(self) -> None:
        panel = crafted_panel()
        curve = _tuner().tune(panel)
        flat = _tuner(costs=FLAT).tune(panel)
        assert curve.best_score_ < flat.best_score_

    def test_the_objective_records_the_fee_model(self) -> None:
        config = _tuner().tune(crafted_panel()).tuning_config()
        assert config["costs"]["fee_model"] == "polymarket"
        assert config["costs"]["fee_schedule"]["rate"] == 0.05

    def test_taker_bps_is_the_flat_model_and_warns(self) -> None:
        with pytest.warns(PmlabDeprecationWarning, match="taker_bps"):
            tuner = _tuner(taker_bps=75.0)
        assert tuner.costs == CostModel(fee_model="flat-bps", taker_bps=75.0)
        assert tuner.taker_bps == 75.0


class TestWhatAnOlderTunerMeans:
    def test_a_saved_polymarket_tuner_round_trips(self, tmp_path: Path) -> None:
        costs = CostModel(fee_schedule=FeeSchedule(rate=0.02), fee_source="gamma")
        tuner = _tuner(costs=costs).tune(crafted_panel())
        tuner.save(tmp_path / "t.pkl")
        loaded = TunedForecaster.load(tmp_path / "t.pkl")
        assert loaded.costs == costs
        assert loaded.tuning_config() == tuner.tuning_config()

    def test_a_tuner_saved_with_the_old_cost_dict_loads_as_flat_bps(self, tmp_path: Path) -> None:
        tuner = _tuner(costs=FLAT).tune(crafted_panel())
        tuner.save(tmp_path / "new.pkl")
        with open(tmp_path / "new.pkl", "rb") as f:
            data = pickle.load(f)
        data["tuning"]["costs"] = {
            "taker_bps": 30.0,
            "slippage_bps": 50.0,
            "slippage_fixed": 0.0,
            "depth_bps_per_unit": 0.0,
            "depth_reference_stake": 1.0,
        }
        with open(tmp_path / "1.4.pkl", "wb") as f:
            pickle.dump(data, f)
        loaded = TunedForecaster.load(tmp_path / "1.4.pkl")
        assert loaded.costs == CostModel(fee_model="flat-bps", slippage_bps=50.0)
        assert loaded.taker_bps == 30.0

    def test_a_tuner_saved_with_no_cost_dict_at_all_loads_as_flat_bps(self, tmp_path: Path) -> None:
        tuner = _tuner(costs=FLAT).tune(crafted_panel())
        tuner.save(tmp_path / "new.pkl")
        with open(tmp_path / "new.pkl", "rb") as f:
            data = pickle.load(f)
        data["tuning"].pop("costs")
        with open(tmp_path / "old.pkl", "wb") as f:
            pickle.dump(data, f)
        loaded = TunedForecaster.load(tmp_path / "old.pkl")
        assert loaded.costs == FLAT
        assert loaded.best_params_ == tuner.best_params_
