# The champion registry: record, roll back, lose nothing

`ChampionManifest.publish` writes one `champion.json` that every retrain overwrites.
`ModelRegistry` archives each promotion as an immutable version (model, calibrator, manifest)
so you can `list`, `diff` and `rollback` to an earlier champion. A rolled-back champion is the
champion as it was promoted, not a stripped copy of it.

## What a version keeps

Every field of the manifest, and nothing is rebuilt key by key on the way: `publish`,
`ModelRegistry.record()` and `ModelRegistry.rollback()` all write `ChampionManifest.to_dict()`.

| Field | Why a rollback needs it |
|-------|-------------------------|
| `model_name`, `plugin_family`, `published_at`, `gate` | what was promoted, and the evidence for it; the registry still refuses anything but a `GO` |
| `costs` | the friction the gate was computed under, so a live run can trade under the same assumptions |
| `expectation` (`backtest_expectation` in the JSON) | what the backtest promised, which `pmlab monitor` holds live PnL and calibration against |
| `reproducibility` | how the backtest was run, so `pmlab reproduce --registry-dir ...` can re-run it |
| `model_path`, `calibrator_path` | re-pointed at the archived (`record`) or live (`rollback`) artifacts; `get()` finds an archive's artifacts next to its `champion.json`, so a registry can be opened from any directory, or moved |

Releases before 0.11.1 dropped `backtest_expectation` in `record()` and `costs` and
`backtest_expectation` in `rollback()` (#33): a rolled-back champion came back with no
expectation and no costs, and `pmlab monitor` reported the expectation check as unavailable.
Versions archived by those releases still load and roll back. They never held the expectation,
so a rollback to one has none; what they did archive (`costs` from 0.8.4, `reproducibility`
from 0.10.0) a rollback now restores too. If you kept the `champion.json` the version was
promoted from, `ModelRegistry.record(ChampionManifest.load(path))` archives it again, whole.

## Worked example

The whole path as a user runs it: a backtest, a promotion with the CLI, then the registry.
The blocks share one directory and run in order. A synthetic panel first:

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
rows = []
for d in range(80):
    day = f"2026-{1 + d // 28:02d}-{d % 28 + 1:02d}"
    for k in range(3):
        winner = "YES" if rng.random() < 0.5 else "NO"
        for label in ("YES", "NO"):
            signal = 1.0 if label == winner else -1.0
            rows.append(
                {
                    "market_id": f"m{d}_{k}",
                    "decision_date": day,
                    "outcome_label": label,
                    "winning_label": winner,
                    "market_price": 0.5,
                    "feature_a": signal * 0.4 + rng.normal(),
                    "feature_b": rng.normal(),
                }
            )
pd.DataFrame(rows).to_parquet("panel.parquet")
open("costs.json", "w").write('{"taker_bps": 45.0, "slippage_bps": 20.0}')
print(len(rows), "rows written")
```

Backtest under non-default costs. `--output` records the costs, the expectation and the
reproducibility record:

```bash
pmlab backtest --plugin demo --panel panel.parquet --stride 10 --min-train-rows 40 \
    --policy top-bin-yes --seed 7 --costs costs.json --output metrics.json
```

A model and a gate to promote (a real run gets them from its own training and `HoldoutGate`):

```python
import numpy as np
import pandas as pd
import json

from pmlab.backtest.holdout_gate import HoldoutGateResult
from pmlab.modeling.lgbm_baseline import LGBMForecaster

rng = np.random.default_rng(0)
X = pd.DataFrame({"feature_a": rng.normal(size=60)})
model = LGBMForecaster(n_estimators=3, min_child_samples=2)
model.fit(X, pd.Series((X["feature_a"] > 0).astype(int)))
model.save("model.pkl")

trades = pd.DataFrame([{"realized_pnl": 0.1, "outcome": "won", "segment": "A"}] * 50)
gate = HoldoutGateResult.evaluate(trades, required_segments=["A"], min_trades_per_segment=40)
json.dump(gate.to_dict(), open("gate.json", "w"))
```

```bash
pmlab promote-champion model.pkl --gate-path gate.json --plugin demo \
    --output-dir champion --expectation metrics.json
```

Archive it, restore it into another directory, and compare every field of the manifest:

```python
import dataclasses

from pmlab.modeling.champion import ChampionManifest
from pmlab.modeling.registry import ModelRegistry

promoted = ChampionManifest.load("champion/champion.json")
registry = ModelRegistry("registry")
version = registry.record(promoted)
restored = registry.rollback(version, "restored")

for field in dataclasses.fields(ChampionManifest):
    if field.name in ("model_path", "calibrator_path"):  # re-pointed at the restored artifacts
        continue
    same = getattr(restored, field.name) == getattr(promoted, field.name)
    print(f"{field.name:16} {'kept' if same else 'LOST'}")
    assert same, field.name

assert restored.costs is not None and restored.expectation is not None
assert restored.reproducibility is not None
open("version.txt", "w").write(version)
print("rolled back:", version)
```

```text
model_name       kept
gate             kept
published_at     kept
plugin_family    kept
costs            kept
expectation      kept
reproducibility  kept
rolled back: 2026-11-08T05-20-03.944591-00-00__champion
```

The archived version still re-runs the backtest it was promoted on:

```bash
pmlab reproduce "$(cat version.txt)" --registry-dir registry
```

## Writing a new manifest field

Add it to `ChampionManifest`, to `to_dict()` and to `load()`, and nothing else: the registry
and `publish` both write `to_dict()`. `tests/modeling/test_registry_roundtrip.py` walks
`dataclasses.fields(ChampionManifest)` and the keys of the file on disk, so a field you add is
held to the round trip without editing the test. It fails first, by name, until the fixture
in `tests/modeling/manifest_helpers.py` gives the field a value, so a field cannot be added
without being exercised.
