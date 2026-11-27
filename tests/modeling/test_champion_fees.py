"""`champion.json` records the fee model the gate was computed under, and reads an older one as flat-bps (#72).

The `costs` key of a champion written by 1.4.0 and before has the five `CostModel` fields and no
`fee_model`. That champion was gated under the flat 30 bps, so that is what it loads as: reading it as
the new default would have `record-trades` trade it under fees its gate never saw.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from pmlab import PmlabDeprecationWarning
from pmlab.backtest.holdout_gate import HoldoutGateResult
from pmlab.core.costs import CostModel
from pmlab.core.fee_schedule import FeeSchedule
from pmlab.modeling.champion import ChampionManifest
from pmlab.modeling.lgbm_baseline import LGBMForecaster

FLAT = CostModel(fee_model="flat-bps")
OLD_COSTS = {
    "taker_bps": 30.0,
    "slippage_bps": 10.0,
    "slippage_fixed": 0.0,
    "depth_bps_per_unit": 0.0,
    "depth_reference_stake": 1.0,
}


def _gate() -> HoldoutGateResult:
    rows = [{"realized_pnl": 0.1, "outcome": "won", "segment": s} for s in "AB" for _ in range(50)]
    return HoldoutGateResult.evaluate(
        pd.DataFrame(rows), required_segments=["A", "B"], min_trades_per_segment=40
    )


def _model() -> LGBMForecaster:
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"f1": rng.random(50), "f2": rng.random(50)})  # noqa: N806
    model = LGBMForecaster(objective="binary")
    model.fit(X, pd.Series((X["f1"] > 0.5).astype(int)))
    return model


def _publish(tmp_path: Path, costs: CostModel | None) -> Path:
    ChampionManifest.publish(
        model=_model(),
        gate=_gate(),
        output_dir=tmp_path / "out",
        plugin_family="test",
        costs=costs,
    )
    return tmp_path / "out" / "champion.json"


class TestTheChampionRecordsTheFeeModel:
    def test_a_polymarket_champion_round_trips(self, tmp_path: Path) -> None:
        costs = CostModel(fee_schedule=FeeSchedule(rate=0.02), fee_source="gamma", slippage_bps=5.0)
        path = _publish(tmp_path, costs)
        assert json.loads(path.read_text())["costs"]["fee_model"] == "polymarket"
        assert ChampionManifest.load(path).costs == costs

    def test_a_flat_champion_round_trips(self, tmp_path: Path) -> None:
        costs = CostModel(fee_model="flat-bps", taker_bps=45.0)
        path = _publish(tmp_path, costs)
        assert json.loads(path.read_text())["costs"]["fee_model"] == "flat-bps"
        assert ChampionManifest.load(path).costs == costs

    def test_a_champion_without_costs_still_has_none(self, tmp_path: Path) -> None:
        assert ChampionManifest.load(_publish(tmp_path, None)).costs is None


class TestAnOlderChampion:
    def test_costs_written_before_1_5_0_load_as_flat_bps(self, tmp_path: Path) -> None:
        path = _publish(tmp_path, FLAT)
        data = json.loads(path.read_text())
        data["costs"] = OLD_COSTS
        path.write_text(json.dumps(data))
        loaded = ChampionManifest.load(path).costs
        assert loaded == CostModel(fee_model="flat-bps", slippage_bps=10.0)

    def test_loading_an_older_champion_does_not_warn(self, tmp_path: Path) -> None:
        path = _publish(tmp_path, FLAT)
        data = json.loads(path.read_text())
        data["costs"] = OLD_COSTS
        path.write_text(json.dumps(data))
        with warnings.catch_warnings():
            warnings.simplefilter("error", PmlabDeprecationWarning)
            ChampionManifest.load(path)

    def test_an_older_champion_written_back_names_its_model(self, tmp_path: Path) -> None:
        path = _publish(tmp_path, FLAT)
        data = json.loads(path.read_text())
        data["costs"] = OLD_COSTS
        path.write_text(json.dumps(data))
        written = ChampionManifest.load(path).to_dict()
        assert written["costs"]["fee_model"] == "flat-bps"
