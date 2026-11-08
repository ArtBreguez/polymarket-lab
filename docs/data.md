# Data layer: snapshots, panels, leakage guards

`pmlab.data` is the reproducibility backbone of ingestion: point-in-time snapshots of open markets,
a typed panel builder, and guards against the two ways a backtest lies (a feature that gives the
outcome away, a price seen after the decision it informs).

The examples run offline. The first block imports every name this page uses.

```python
import tempfile
from pathlib import Path

import pandas as pd

from pmlab import (
    FeatureSnapshotStore,
    build_panel,
    build_panel_from_snapshots,
    check_no_leakage,
)
from pmlab.datasets import load_sample
```

## Capture what a decision could have seen

`FeatureSnapshotStore` is append-only: it records the features of **open** markets over time, keyed by
`(market_id, captured_at)` per market family. When a market later resolves, you join its truth to the
snapshot taken *before* the decision, not to whatever the features say today.

```python
store = FeatureSnapshotStore(Path(tempfile.mkdtemp()) / "snapshots")   # any directory, e.g. ~/.pmlab/snapshots
store.append("politics", [
    {"market_id": "m1", "captured_at": "2025-01-05T00:00:00Z",
     "feature_spread": 0.02, "feature_vol": 1.2e6},
    {"market_id": "m1", "captured_at": "2025-01-09T00:00:00Z",
     "feature_spread": 0.05, "feature_vol": 1.9e6},
])

# The latest snapshot of each market strictly before the cutoff: no lookahead.
seen = store.as_of("politics", cutoff="2025-01-06")
print(seen[["market_id", "captured_at", "feature_spread"]])
```

## Close the loop when markets resolve

Once a market resolves, `build_panel_from_snapshots` joins each resolution to the snapshot taken
before its decision date and returns a panel the backtest accepts:

```python
resolutions = pd.DataFrame([
    {"market_id": "m1", "decision_date": "2025-01-10", "winning_label": "Yes"},
])
train = build_panel_from_snapshots(store, "politics", resolutions)   # uses the snapshot < decision_date
print(train[["market_id", "decision_date", "winning_label"]])
```

## Assemble a validated panel from plugin rows

`build_panel` takes the rows a plugin's `build_training_rows` returns and checks them: it drops `None`,
coerces features to float, raises a clear error on a missing column or a bad date, and **refuses a
price observed at or after the decision it informs**. A market with several bins is several rows,
one per bin, each at that bin's own price (see [One row per bin](plugin-authoring.md#one-row-per-bin)).

```python
sample = load_sample()
resolver = sample.plugin()                 # the same plugin once the sample has resolved: it labels rows
rows = [
    resolver.build_training_rows(spec, "previous_evening", decision_time=f"{day}T12:00:00+00:00")
    for day in sample.days[:30]
    for spec in sample.plugin(day).discover_markets()
]
panel = build_panel(rows)
print(panel.shape, "rows (one per market and bin)")
```

## Guard against label leakage

`check_no_leakage` scores each feature's power to separate the label and raises `LeakageError` when
one separates it suspiciously well: a feature derived from the outcome. It once reported an AUC of
1.000 silently, which is how it was found. `build_panel(rows, check_leakage=True)` runs it for you.

```python
report = check_no_leakage(panel)           # raises LeakageError on a leak
print("leaked:", report.leaked or "none")
```

On a panel with one row per bin the label is "did this bin win", the same target the backtest trains
on. What the check scores against, what it can and cannot see, and how to choose the target yourself:
[Leakage](leakage.md).
