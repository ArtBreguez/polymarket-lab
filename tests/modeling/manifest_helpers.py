"""Builders and assertions for "a champion survives the registry intact".

The assertions iterate over `dataclasses.fields(ChampionManifest)` and over the keys of the
`champion.json` on disk, never over a hand-kept list of names. A field added to the manifest
later is therefore held to the same round trip without anyone remembering to add it here.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pmlab.backtest.expectation import BacktestExpectation, GroupExpectation, SegmentExpectation
from pmlab.backtest.holdout_gate import HoldoutGateResult
from pmlab.core.costs import CostModel
from pmlab.core.policy import TradePolicy
from pmlab.modeling.calibration import IsotonicCalibrator
from pmlab.modeling.champion import ChampionManifest
from pmlab.modeling.lgbm_baseline import LGBMForecaster
from pmlab.reproducibility import ReproducibilityRecord, fingerprint_panel
from tests.reproducibility.helpers import make_panel

# The two fields that legitimately change on the way through a registry: they are the paths of the
# artifacts, re-pointed at the archive (record) or at the live directory (rollback).
REPOINTED = frozenset({"model_path", "calibrator_path"})


def full_manifest(out: Path) -> ChampionManifest:
    """A champion published with every optional field set to a non-default value."""
    rows = [{"realized_pnl": 0.1, "outcome": "won", "segment": s} for s in "AB" for _ in range(50)]
    gate = HoldoutGateResult.evaluate(
        pd.DataFrame(rows), required_segments=["A", "B"], min_trades_per_segment=40
    )
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f1": rng.random(60), "f2": rng.random(60)})  # noqa: N806
    model = LGBMForecaster(n_estimators=3, min_child_samples=2)
    model.fit(X, pd.Series((X["f1"] > 0.5).astype(int)))
    calibrator = IsotonicCalibrator()
    calibrator.fit(np.array([0.3, 0.5, 0.7]), np.array([0, 1, 1]))
    expectation = BacktestExpectation(
        n_trades=120,
        flat_stake=1.0,
        mean_pnl_per_stake=0.05,
        std_pnl_per_stake=0.9,
        brier_score=0.21,
        brier_std=0.17,
        by_direction={
            "yes": GroupExpectation(70, 0.12, 1.0, 0.20, 0.16),
            "no": GroupExpectation(50, -0.04, 0.7, 0.23, 0.18),
        },
        policy=TradePolicy.live(min_edge=0.07),
        by_segment={
            "A": SegmentExpectation(60, 0.06, 0.9, 0.21, 0.17),
            "B": SegmentExpectation(60, 0.04, 0.9, 0.21, 0.17),
        },
    )
    record = ReproducibilityRecord.capture(
        kind="backtest",
        seeds={"python_random": 4, "numpy": 4, "lightgbm": 4},
        config={"plugin": "demo", "stride": 10},
        data=fingerprint_panel(make_panel(n_dates=5), source="p.parquet"),
        metrics={"total_trades": 3, "total_pnl": 0.5},
    )
    return ChampionManifest.publish(
        model=model,
        gate=gate,
        output_dir=out,
        plugin_family="test_family",
        model_name="full_champ",
        calibrator=calibrator,
        costs=CostModel(taker_bps=45.0, slippage_bps=20.0, slippage_fixed=0.003),
        expectation=expectation,
        reproducibility=record,
    )


def _is_unset(field: dataclasses.Field[Any], value: Any) -> bool:
    """None, or still at its default: a value that a load would invent on its own."""
    if value is None:
        return True
    if field.default is not dataclasses.MISSING:
        return bool(value == field.default)
    if field.default_factory is not dataclasses.MISSING:
        return bool(value == field.default_factory())
    return False


def assert_every_field_set(manifest: ChampionManifest) -> None:
    """Guard: a field the fixture leaves unset would make the round trip prove nothing for it."""
    unset = [
        f.name for f in dataclasses.fields(manifest) if _is_unset(f, getattr(manifest, f.name))
    ]
    assert not unset, (
        f"full_manifest() leaves {unset} unset, so the round trip cannot show they survive: "
        "give them a value in tests/modeling/manifest_helpers.py"
    )


def assert_manifest_survived(original: ChampionManifest, restored: ChampionManifest) -> None:
    """Every field of `original` is in `restored`, bar the artifact paths (which must exist)."""
    for f in dataclasses.fields(ChampionManifest):
        if f.name in REPOINTED:
            continue
        assert getattr(restored, f.name) == getattr(original, f.name), f"{f.name} was lost"
    assert restored.model_path.exists() and restored.model_path != original.model_path
    assert (original.calibrator_path is None) == (restored.calibrator_path is None)
    if restored.calibrator_path is not None:
        assert restored.calibrator_path.exists()


def _without_paths(champion_json: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(champion_json.read_text())
    return {k: v for k, v in data.items() if k not in REPOINTED}


def assert_same_champion_json(published: Path, other: Path) -> None:
    """The two `champion.json` files hold the same keys and values, bar the artifact paths.

    This is what a reader of the file sees, so it also covers a key that the dataclass does not
    know about (a field added to `publish` but not to the dataclass would still be caught).
    """
    assert _without_paths(other) == _without_paths(published)
