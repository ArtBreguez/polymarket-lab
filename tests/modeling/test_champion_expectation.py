"""The champion records what it promised, and old champions still load."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pmlab.backtest.expectation import BacktestExpectation
from pmlab.backtest.holdout_gate import HoldoutGateResult
from pmlab.modeling.champion import ChampionManifest
from pmlab.modeling.lgbm_baseline import LGBMForecaster


def _go_gate() -> HoldoutGateResult:
    rows = [{"realized_pnl": 0.1, "outcome": "won", "segment": s} for s in "AB" for _ in range(50)]
    return HoldoutGateResult.evaluate(
        pd.DataFrame(rows), required_segments=["A", "B"], min_trades_per_segment=40
    )


def _model() -> LGBMForecaster:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f1": rng.random(50), "f2": rng.random(50)})
    model = LGBMForecaster()
    model.fit(X, pd.Series((X["f1"] > 0.5).astype(int)))
    return model


def _expectation() -> BacktestExpectation:
    return BacktestExpectation(
        n_trades=120,
        flat_stake=1.0,
        mean_pnl_per_stake=0.05,
        std_pnl_per_stake=0.9,
        brier_score=0.21,
        brier_std=0.17,
    )


def _publish(out: Path, **kwargs) -> ChampionManifest:
    return ChampionManifest.publish(
        model=_model(), gate=_go_gate(), output_dir=out, plugin_family="test", **kwargs
    )


def test_publish_records_the_expectation_and_load_restores_it(tmp_path: Path) -> None:
    manifest = _publish(tmp_path, expectation=_expectation())

    on_disk = json.loads((tmp_path / "champion.json").read_text())
    assert on_disk["backtest_expectation"] == _expectation().to_dict()
    assert manifest.expectation == _expectation()
    assert ChampionManifest.load(tmp_path / "champion.json").expectation == _expectation()


def test_publish_without_an_expectation_records_null(tmp_path: Path) -> None:
    manifest = _publish(tmp_path)

    assert manifest.expectation is None
    on_disk = json.loads((tmp_path / "champion.json").read_text())
    assert on_disk["backtest_expectation"] is None
    assert ChampionManifest.load(tmp_path / "champion.json").expectation is None


def test_a_champion_published_before_expectations_still_loads(tmp_path: Path) -> None:
    """Upgrading must not strand a live champion: no key means 'unknown', not a crash."""
    _publish(tmp_path, expectation=_expectation())
    path = tmp_path / "champion.json"
    data = json.loads(path.read_text())
    del data["backtest_expectation"]
    path.write_text(json.dumps(data))

    assert ChampionManifest.load(path).expectation is None


def test_a_malformed_expectation_is_an_error_not_a_silent_none(tmp_path: Path) -> None:
    """A broken record must not read as 'no expectation' and quietly weaken the watchdog."""
    _publish(tmp_path, expectation=_expectation())
    path = tmp_path / "champion.json"
    data = json.loads(path.read_text())
    for broken in ({"n_trades": 5}, {}):
        data["backtest_expectation"] = broken
        path.write_text(json.dumps(data))

        with pytest.raises((KeyError, ValueError)):
            ChampionManifest.load(path)


def test_the_gate_still_hard_blocks_with_an_expectation(tmp_path: Path) -> None:
    nogo = HoldoutGateResult(
        decision="NO_GO", segment_results=[], aggregate_pnl=-1.0, aggregate_trades=10
    )
    with pytest.raises(ValueError, match="NO_GO"):
        ChampionManifest.publish(
            model=_model(),
            gate=nogo,
            output_dir=tmp_path,
            plugin_family="test",
            expectation=_expectation(),
        )


@pytest.mark.parametrize(
    "broken",
    [
        {"std_pnl_per_stake": float("nan")},
        {"brier_std": float("nan")},
        {"brier_score": float("nan")},
        {"mean_pnl_per_stake": float("inf")},
        {"std_pnl_per_stake": -0.1},
        {"n_trades": 0},
        {"n_trades": 1},
    ],
)
def test_an_unusable_expectation_makes_the_champion_unloadable(
    tmp_path: Path, broken: dict
) -> None:
    """champion.json is hand-editable and json.loads accepts NaN: loading must refuse it."""
    _publish(tmp_path, expectation=_expectation())
    path = tmp_path / "champion.json"
    data = json.loads(path.read_text())
    data["backtest_expectation"] = {**data["backtest_expectation"], **broken}
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError):
        ChampionManifest.load(path)
