# Performance budget

pmlab has two hot paths, the ones a researcher waits on and a scheduled job repeats every day:

- **`rolling_origin_eval`**, the walk-forward backtest (`pmlab.backtest.rolling_origin`). Every
  refit re-masks the panel, trains, predicts and settles one trade per market. Its second
  decision rule, the live best-edge policy, walks the bin rows in Python.
- **`build_panel`**, the validated training panel (`pmlab.data`). It flattens the rows a plugin
  emits, coerces and checks the columns, and runs the price-timing guard.

Both have a **budget**, and CI asserts it on every push (`pytest -m perf`, job `perf`): a change
that makes either of them several times slower, or that makes the cost grow faster than the data,
fails the build instead of being noticed months later. Nothing here is a benchmark of LightGBM:
the workloads use a deliberately cheap model so that what is timed is pmlab's own code.

## The budget

Time is in **calibration units**, i.e. wall seconds divided by the wall time of a small fixed
loop measured in the same run. The loop is the hot paths' own pandas work in miniature (string
masks and boolean takes on object columns, a group-by on two string keys, `iterrows`, dict
records to a DataFrame, an interpreter loop) with none of pmlab's code in it, so it tracks how
fast a machine runs *this kind* of work. A first version that was only a pure-Python loop did
not: the CI runner has a much faster interpreter than the reference VM but the same pandas
speed, and the units differed by 2x between them. The budget is 2.5x the reference, so a 3x
slowdown fails and ordinary noise does not.

| workload | what is timed | reference (units) | budget (units) |
|---|---|---|---|
| `rolling_origin_top_prob` | `rolling_origin_eval`, historical top-bin rule | 6.4 | 16 |
| `rolling_origin_live_policy` | `rolling_origin_eval` with `TradePolicy.live(min_edge=0.02)` | 10.8 | 27 |
| `build_panel` | `build_panel` on timing-stamped rows from a plugin | 4.3 | 11 |

The workloads are seeded and built in-process (no network, no files). Sizes are a season of daily
decisions over a few hundred markets:

| workload | rows | shape | work |
|---|---|---|---|
| `rolling_origin_top_prob` | 72,000 | 100 markets x 120 dates x 6 bins | 29 refits (`stride=4`), ~2,900 trades |
| `rolling_origin_live_policy` | 28,800 | 40 markets x 120 dates x 6 bins | 27 refits (`stride=4`), ~1,080 trades, best (bin, side) per market |
| `build_panel` | 120,000 | 2,000 markets x 10 dates x 6 bins | `price_ts`/`decision_ts`/`market_close_ts` on every row |

Measured on the reference machine (2 vCPU VM, Python 3.12, `uv run`, nothing else running), the
calibration loop takes about 63 ms:

| workload | seconds | units | spread over 6 rounds |
|---|---|---|---|
| `rolling_origin_top_prob` | 0.40 | 6.4 | 6.4 to 6.5 |
| `rolling_origin_live_policy` | 0.68 | 10.8 | 10.7 to 10.9 |
| `build_panel` | 0.28 | 4.3 | 4.3 to 4.4 |

## Scaling guard

A budget in seconds cannot see an accidental O(n^2) until the data is big enough to hurt. So each
workload is also timed at **2x the markets (and so 2x the rows)**, and the guard asserts that it
costs **less than 3x** as much. Linear code costs about 2x (1.7x to 2.1x measured here); quadratic
code costs 4x. This check compares two timings from the same run, so it does not depend on the
machine at all.

## Why it is not flaky

- **Normalised.** Budgets are in calibration units, measured back to back with the workload.
- **Minimum, not mean.** Each timing is the best of 5 runs after a warm-up, with the garbage
  collector paused (as `timeit` advises): noise can only add time, so the minimum is the
  cleanest reading of the code's own cost.
- **Retries before failing.** If a round is over budget, up to two more rounds are measured and
  the guard fails only if every one of them is. A noisy neighbour spoils a round; a regression
  spoils all of them.
- **No threads, no I/O.** The model is single-threaded numpy and nothing touches the disk, so
  runner core counts and disk speed do not enter.
- **Alone, untraced.** The CI job runs by itself and without coverage; the test skips itself if
  a tracer (coverage, a debugger) is active.

## Running it

```bash
uv run pytest -m perf                  # the guard (about a minute); deselected by plain `pytest`
uv run python examples/perf_budget.py  # the table above, measured on your machine
uv run python examples/perf_budget.py --check   # same, exit 1 if a budget or scaling limit breaks
```

Run it with nothing else heavy on the machine. To see how long your own panel takes, time the
two functions directly; this runs against the installed package:

```python
import time

import numpy as np
from sklearn.dummy import DummyClassifier

from pmlab.backtest.rolling_origin import rolling_origin_eval
from pmlab.data import build_panel
```

```python
rng = np.random.default_rng(0)
rows = [
    [
        {
            "market_id": f"m{m}",
            "decision_date": f"2025-01-{d + 1:02d}",
            "outcome_label": label,
            "winning_label": "Yes",
            "market_price": 0.5,
            "price_ts": f"2025-01-{d + 1:02d}T11:00:00+00:00",
            "decision_ts": f"2025-01-{d + 1:02d}T12:00:00+00:00",
            "feature_x": float(rng.normal()),
        }
        for d in range(20)
        for label in ("Yes", "No")
    ]
    for m in range(50)
]
start = time.perf_counter()
panel = build_panel(rows)
built = time.perf_counter() - start
start = time.perf_counter()
result = rolling_origin_eval(panel, DummyClassifier(strategy="prior"), min_train_rows=50, stride=2)
print(f"build_panel: {built:.3f}s for {len(panel)} rows; "
      f"rolling_origin_eval: {time.perf_counter() - start:.3f}s, {len(result.trades)} trades")
```

## Changing a budget on purpose

A slowdown can be the right trade (a safety check, a new column in the trade log). Then:

1. Run `uv run python examples/perf_budget.py` several times on a quiet machine and take the
   median `units` of the affected workload.
2. In `examples/perf_budget.py`, set that workload's `reference_units` to it and `budget_units`
   to about 2.5x (the headroom that lets a 3x slowdown fail and noise pass).
3. Update the table above (a fast test fails if the page and the script disagree).
4. Say why in the changelog and the PR: a budget raised without a reason is the regression the
   guard exists to catch.

If a workload is over budget and you did not mean to slow it down, profile it first
(`cProfile` on `w.run(w.prepare(1))`, with `w` taken from `WORKLOADS` in the script). The
scaling test failing means the cost became super-linear in the rows, which is a bug, not a budget.

Budgets for `LGBMForecaster` training, the Gamma and CLOB clients and
`build_panel_from_snapshots` (which reads the snapshot store once per resolution) are out of
scope: they are bound by native threads, the network or the disk, which a unit-normalised
CPU-bound guard cannot see.
