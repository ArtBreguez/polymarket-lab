# Reproducibility — `pmlab reproduce`

> A result you can't re-run isn't a result.

Every backtest records what it needs to be run again, and `pmlab reproduce` runs it again
and checks that the same numbers come back. This page is a walkthrough: every `python` and
`bash` block below is executed, in order, exactly as written, by the end-to-end test against
the installed wheel (`uv run pytest -m e2e`).

## What a run records

| Part | Content | Why |
|---|---|---|
| **seeds** | `python_random`, `numpy`, `lightgbm`: the value each generator was given | the part of a result that is chance |
| **environment** | pmlab, Python, numpy, pandas, lightgbm, scikit-learn, optuna (`null` if absent), OS, CPU architecture and count | a library upgrade can change a number without a line of your code changing |
| **config** | the *resolved* parameters: model and its LightGBM params with defaults spelled out, stride, min train rows, stake, the full cost model, the full trade policy | a later change of a default must not change what an old record means |
| **data** | SHA-256 of the panel's content, row count, column list, where it was read from | tells you the data under a path is no longer the data of the run |
| **metrics** | what the run produced: `total_trades`, `steps`, `total_pnl`, `hit_rate`, `avg_edge` | what a re-run is compared to |

The record is one JSON object (`ReproducibilityRecord`, schema version 1) stored where runs
are already kept: in the experiment tracker's run file, in `champion.json`, in the model
registry's archived versions, and in the metrics file `pmlab backtest --output` writes.

## Worked example: run, then reproduce

A small synthetic panel (480 rows, 80 decision dates), written to `panel.parquet`:

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
                    "feature_c": rng.normal(),
                    "feature_d": rng.normal(),
                }
            )
pd.DataFrame(rows).to_parquet("panel.parquet")
print(len(rows), "rows written")
```

Run the backtest with an explicit seed and a model that actually uses it (bagging and feature
sub-sampling draw random numbers; a model without them would give the same result for any
seed). `--runs-dir` also logs the run in the experiment tracker, `--output` writes the metrics
file:

```bash
pmlab backtest --plugin demo --panel panel.parquet --stride 10 --min-train-rows 40 \
    --policy top-bin-yes --seed 123 \
    --lgbm-param bagging_fraction=0.6 --lgbm-param bagging_freq=1 \
    --lgbm-param feature_fraction=0.6 \
    --runs-dir runs --output metrics.json
```

Now reproduce it: re-execute from the record alone, in a new process, and compare.

```bash
pmlab reproduce metrics.json
```

```text
Reproducing metrics.json: backtest, plugin=demo, seeds {'python_random': 123, 'numpy': 123, 'lightgbm': 123}, recorded 2026-11-08T04:01:47
Data: matches the record (480 rows, sha256 980d1cc0aa5c)
                          Reproduction
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┓
┃ Metric       ┃ Recorded      ┃ Re-run        ┃ |diff| ┃ Match ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━┩
│ total_trades │ 21            │ 21            │ 0      │ yes   │
│ steps        │ 7             │ 7             │ 0      │ yes   │
│ total_pnl    │ 0.937         │ 0.937         │ 0      │ yes   │
│ hit_rate     │ 0.5238095238  │ 0.5238095238  │ 0      │ yes   │
│ avg_edge     │ 0.06744466124 │ 0.06744466124 │ 0      │ yes   │
└──────────────┴───────────────┴───────────────┴────────┴───────┘
REPRODUCED: 5 metrics match (rtol=1e-09, atol=1e-12; counts exact).
```

`pmlab reproduce` accepts whatever holds the record: the metrics file above, a tracker run id
(or a unique prefix of it, with `--runs-dir`), a `champion.json` or the directory holding it,
or a registry version id (with `--registry-dir`). By run id:

```bash
RUN_ID=$(basename runs/demo/*.json .json)
pmlab reproduce "$RUN_ID" --runs-dir runs
```

## When it does not reproduce

Exit status: **0** reproduced, **1** it ran but the metrics differ, **2** it could not run
(no record, data changed, panel missing).

### The data changed: refused

The record holds a hash of the panel's content. A re-run on other data is not a reproduction,
so it is refused. Here a copy of the panel with one value changed, and a copy of the metrics
file claiming an older LightGBM (used below):

```python
import json

import pandas as pd

panel = pd.read_parquet("panel.parquet")
panel.loc[0, "feature_a"] += 0.5
panel.to_parquet("panel_edited.parquet")

record = json.load(open("metrics.json"))
record["reproducibility"]["environment"]["lightgbm"] = "3.3.0"
json.dump(record, open("metrics_older_lightgbm.json", "w"))
```

```bash
pmlab reproduce metrics.json --panel panel_edited.parquet || echo "exit status: $?"
```

```text
Reproducing metrics.json: backtest, plugin=demo, seeds {...}, recorded 2026-11-08T04:01:47
Cannot reproduce: The data differs from the data this run was recorded on (content hash 980d1cc0aa5c (recorded) vs 40a835b489d8 (now); 480 rows (recorded) vs 480 rows (now)). Refusing to compare metrics across different data; pass --allow-data-mismatch to run anyway.
exit status: 2
```

`--allow-data-mismatch` runs it anyway, for when you want to see how far the numbers moved.
The result can then only ever be reported as *not* reproduced:

```bash
pmlab reproduce metrics.json --panel panel_edited.parquet --allow-data-mismatch || echo "exit status: $?"
```

### A library version changed: warned, with the diff

A different version is a warning, not a verdict: only the metrics decide. If they come back,
the run is reproduced and the drift is shown next to it; if they do not, the drift is the first
place to look.

```bash
pmlab reproduce metrics_older_lightgbm.json; echo "exit status: $?"
```

```text
Data: matches the record (480 rows, sha256 980d1cc0aa5c)
Version drift: 1 version(s) differ from the record. A different version can change a result without a change in the code.
         Version drift
┏━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━┓
┃ Library  ┃ Recorded ┃ Now   ┃
┡━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━┩
│ lightgbm │ 3.3.0    │ 4.6.0 │
└──────────┴──────────┴───────┘
  ... the metrics table, as above ...
REPRODUCED: 5 metrics match (rtol=1e-09, atol=1e-12; counts exact).
exit status: 0
```

### There is nothing to re-run

```bash
pmlab reproduce 0123abcd --runs-dir runs || echo "exit status: $?"
```

A run recorded before 0.10.0 (a `champion.json` or a tracker run with no `reproducibility`)
says so and says how to get one: re-run the backtest with `pmlab backtest ... --output`.

## Tolerance

On the machine and library versions of the original run the metrics come back **bit for bit**.
The default tolerance is tight, `--rtol 1e-9 --atol 1e-12` for floating-point metrics
(`total_pnl`, `hit_rate`, `avg_edge`): it absorbs summation-order differences across CPUs and
builds and nothing a different seed produces (in the example above, seed 124 instead of 123
moves `avg_edge` from 0.0674 to 0.0986). Counts (`total_trades`, `steps`) are always compared exactly. `--rtol 0 --atol 0` is
strict.

## Champions and the registry

`pmlab promote-champion` records the backtest's reproducibility record in `champion.json`
(`"reproducibility"`), taken from the `--expectation` file (which `pmlab backtest --output`
wrote) or from `--reproducibility FILE`. Through the API:
`ChampionManifest.publish(..., reproducibility=record)`. `ModelRegistry.record()` archives it with the
version and `rollback()` restores it. Old manifests without the key load as before
(`manifest.reproducibility is None`).

```python
import pandas as pd

from pmlab.backtest.holdout_gate import HoldoutGateResult
from pmlab.modeling.champion import ChampionManifest
from pmlab.reproducibility import BacktestConfig, build_forecaster, load_record, run_backtest

record = load_record("metrics.json")
panel = pd.read_parquet("panel.parquet")
config = BacktestConfig.from_dict(record.config)

trades = run_backtest(panel, config, record.seeds).trades
gate = HoldoutGateResult.evaluate(trades, required_segments=None, min_trades_per_segment=10)

model = build_forecaster(config, record.seeds)
features = [c for c in panel.columns if c.startswith("feature_")]
model.fit(panel[features], (panel["outcome_label"] == panel["winning_label"]).astype(int))

ChampionManifest.publish(
    model, gate, "champion", plugin_family="demo", reproducibility=record
)
print("gate:", gate.decision)
```

The backtest behind the promotion can now be re-run from the champion:

```bash
pmlab reproduce champion/champion.json
```

## Library use

```python
from pmlab.reproducibility import (
    BacktestConfig, capture_backtest, load_record, reproduce, resolve_seeds, run_backtest,
)
from pmlab import LocalJSONTracker

import pandas as pd

panel = pd.read_parquet("panel.parquet")
config = BacktestConfig.resolve(
    plugin="demo", stride=10, min_train_rows=40, policy="top-bin-yes",
    model_params={"bagging_fraction": 0.6, "bagging_freq": 1},
)
seeds = resolve_seeds(7)
outcome = run_backtest(panel, config, seeds)
record = capture_backtest(panel, config, seeds, outcome.metrics, source="panel.parquet")

tracker = LocalJSONTracker("runs")
with tracker.run("demo", "api-example") as run:
    run.log_metrics(outcome.metrics)
    run.log_reproducibility(record)

report = reproduce(record)
print(report.status, [d.name for d in report.drift])
```

`reproduce(record, panel=df)` takes the data in memory instead of a path. For your own
training loop (not `run_backtest`), build the record with `ReproducibilityRecord.capture(
kind=..., seeds=..., config=..., data=fingerprint_panel(df), metrics=...)` and log it: it is
stored and read back like any other, and `pmlab reproduce` will say clearly that it can only
re-execute `backtest` runs.

## Where randomness is drawn, and how it is seeded

| Where | Seeded by | Note |
|---|---|---|
| `pmlab backtest` / `run_backtest` | `--seed` (default 0): python `random`, numpy global, LightGBM `random_state` | one place; a seed in `--lgbm-param` is refused. The global generators are seeded for the run only and restored after, so your own streams are untouched |
| `LGBMForecaster` used directly | `random_state=` if you pass it | unset, LightGBM uses its own fixed internal seeds: deterministic, but not a value that can be recorded |
| `SklearnForecaster` `random_forest` / `gradient_boosting` | `random_state=0` by default (**new in 0.10.0**) | previously unseeded: the numpy global state decided, so two runs differed. Pass `random_state=` to change it |
| `TunedForecaster` | `random_state=0`: the Optuna TPE sampler | unchanged |
| calibrators, conformal | `random_state=0`: the calibration split | unchanged |
| bootstrap stability report | `seed=0` | unchanged |
| target encoding (`kfold`, `group`) | `seed=0` | unchanged |

## Limits

* Only `backtest` runs can be re-executed from their record; a record of another `kind`
  is read, stored and shown, and `reproduce` says it cannot run it.
* The record cannot pin `PYTHONHASHSEED`. The number of threads is not pinned either (the CPU
  count is recorded, for information, and never counted as drift). With LightGBM 4.6.0, 1, 2
  and 8 threads gave identical probabilities on 300 000 rows x 30 features, but a result that
  did depend on them would show up as a miss with no version drift; adding
  `--lgbm-param deterministic=true --lgbm-param force_row_wise=true` removes that source.
* A backtest that produces no trades writes nothing (no `--output` file, no tracked run), as
  before: there is no result to reproduce. A record whose metrics hold no number cannot be
  verified, and `reproduce` says so instead of reporting a vacuous success.
* A CSV panel is hashed as `pandas.read_csv` returns it; the same file always hashes the same,
  a re-exported one may not (float text parsing).
* The MLflow tracker stores the record as the artifact `reproducibility.json`; download it
  and pass the file to `pmlab reproduce`.
