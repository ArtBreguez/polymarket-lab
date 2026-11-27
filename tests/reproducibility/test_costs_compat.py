"""A run records the fee model it used, and an old record is replayed with the model it ran with (#72).

The epic's rule (#69): *a run must record which fee schedule it used, so an old backtest reproduces
with the old model.* The `costs` of a record written by 1.4.0 and before has no `fee_model`; it is
the flat-bps model, so `pmlab reproduce` of that record gives the numbers it gave then, not the
numbers of the new default.
"""

from __future__ import annotations

import copy
import json
import warnings

import pytest

from pmlab import PmlabDeprecationWarning
from pmlab.core.costs import CostModel
from pmlab.reproducibility import (
    BacktestConfig,
    ReproducibilityRecord,
    capture_backtest,
    reproduce,
    resolve_seeds,
    run_backtest,
)
from tests.reproducibility.helpers import STOCHASTIC_PARAMS, make_panel

FLAT = CostModel(fee_model="flat-bps")
SEED = 7
OLD_COSTS = {
    "taker_bps": 30.0,
    "slippage_bps": 0.0,
    "slippage_fixed": 0.0,
    "depth_bps_per_unit": 0.0,
    "depth_reference_stake": 1.0,
}


def _config(**kw: object) -> BacktestConfig:
    base: dict[str, object] = {
        "plugin": "demo",
        "model_params": dict(STOCHASTIC_PARAMS),
        "stride": 10,
        "min_train_rows": 40,
        "policy": "top-bin-yes",
    }
    return BacktestConfig.resolve(**{**base, **kw})  # type: ignore[arg-type]


def _record(config: BacktestConfig) -> tuple[ReproducibilityRecord, dict[str, float]]:
    panel = make_panel()
    seeds = resolve_seeds(SEED)
    outcome = run_backtest(panel, config, seeds)
    record = capture_backtest(panel, config, seeds, outcome.metrics, source=None)
    return record, outcome.metrics


class TestTheRecordNamesTheFeeModel:
    def test_the_resolved_config_spells_out_the_default_fee_model(self) -> None:
        costs = _config().to_dict()["costs"]
        assert costs["fee_model"] == "polymarket"
        assert costs["fee_schedule"]["rate"] == 0.05
        assert costs["fee_source"] == "category_default"

    def test_the_config_round_trips_through_json(self) -> None:
        for config in (_config(), _config(costs=FLAT)):
            assert BacktestConfig.from_dict(json.loads(json.dumps(config.to_dict()))) == config

    def test_a_flat_run_records_flat_bps(self) -> None:
        assert _config(costs=FLAT).to_dict()["costs"]["fee_model"] == "flat-bps"

    def test_the_fee_model_changes_the_run_it_records(self) -> None:
        _, polymarket = _record(_config())
        _, flat = _record(_config(costs=FLAT))
        assert polymarket["total_trades"] == flat["total_trades"]
        assert polymarket["total_pnl"] < flat["total_pnl"]

    def test_a_polymarket_record_reproduces(self) -> None:
        record, _ = _record(_config())
        report = reproduce(record, panel=make_panel())
        assert report.status == "reproduced"

    def test_a_record_does_not_reproduce_under_the_other_fee_model(self) -> None:
        record, _ = _record(_config())
        forged = copy.deepcopy(record.to_dict())
        forged["config"]["costs"] = FLAT.to_dict()
        report = reproduce(ReproducibilityRecord.from_dict(forged), panel=make_panel())
        assert report.status != "reproduced"
        assert any(c.name == "total_pnl" and not c.ok for c in report.comparisons)


class TestARecordFromBefore1_5_0:
    def _old(self, record: ReproducibilityRecord) -> ReproducibilityRecord:
        written = copy.deepcopy(record.to_dict())
        written["config"]["costs"] = dict(OLD_COSTS)
        return ReproducibilityRecord.from_dict(written)

    def test_its_costs_are_the_flat_model(self) -> None:
        record, _ = _record(_config(costs=FLAT))
        old = self._old(record)
        assert CostModel.from_dict(old.config["costs"]) == FLAT

    def test_it_is_replayed_with_the_flat_model_and_reproduces(self) -> None:
        record, _ = _record(_config(costs=FLAT))
        report = reproduce(self._old(record), panel=make_panel())
        assert report.status == "reproduced"
        assert all(c.ok for c in report.comparisons)

    def test_replaying_it_does_not_trigger_the_deprecation_it_predates(self) -> None:
        record, _ = _record(_config(costs=FLAT))
        old = self._old(record)
        with warnings.catch_warnings():
            warnings.simplefilter("error", PmlabDeprecationWarning)
            reproduce(old, panel=make_panel())

    def test_replayed_under_the_new_default_it_would_not_reproduce(self) -> None:
        # the reason the flat model must be what an old record means
        record, _ = _record(_config(costs=FLAT))
        forged = copy.deepcopy(record.to_dict())
        forged["config"]["costs"] = CostModel().to_dict()
        report = reproduce(ReproducibilityRecord.from_dict(forged), panel=make_panel())
        assert report.status != "reproduced"

    def test_the_old_config_loads_as_a_backtest_config(self) -> None:
        old = BacktestConfig.from_dict({**_config().to_dict(), "costs": dict(OLD_COSTS)})
        assert old.costs == OLD_COSTS
        outcome = run_backtest(make_panel(), old, resolve_seeds(SEED))
        _, flat = _record(_config(costs=FLAT))
        assert outcome.metrics["total_pnl"] == pytest.approx(flat["total_pnl"])
