# Tuning for the policy you trade

`TunedForecaster` picks hyper-parameters by scoring every Optuna trial with a walk-forward
backtest (`rolling_origin_eval`). *What a trial is scored on* is a decision rule, and since
0.12.0 it is yours to choose: pass the [`TradePolicy`](reference/core.md#pmlab.TradePolicy) you will
trade with and the search maximises that, not a stand-in for it.

```text
pip install "pmlab[tune]"        # Optuna is an optional extra
```

## The mismatch this closes

| | decides | threshold | sides |
|---|---|---|---|
| **historical** (default, `policy=None`) | YES on the bin the model rates most likely, in *every* market | none | YES |
| **live** (`TradePolicy.live(min_edge=...)`, what `pmlab record-trades` trades) | per market, the (bin, side) with the best after-fee edge | the edge must reach `min_edge` (default **0.20**) | YES or NO |

The two rules reward different models. Under the historical rule only the model's *argmax* matters,
so a model that is merely right about which bin is likeliest looks good, and its confidence is
irrelevant. Under the live rule the argmax is irrelevant and the *size of the edge against the
price* decides everything: a model with no opinion in most markets and a sharp, correct one in a few
never trades the first and only takes the second.

A search that scores trials with the first rule, and a strategy that trades with the second, are
tuning one thing and shipping another. The block below builds the smallest panel where they
disagree, and scores two candidate models (call them A and B) under both rules; the numbers are
real, from the backtest itself.

```python
import numpy as np
import pandas as pd

from pmlab import TradePolicy, rolling_origin_eval

# Three-bin "n" markets (x, y, z) and a two-bin "z" market (x, y). Each row: label, YES price,
# and the probability model A and model B assign to that bin (as features p_a, p_b).
N_BINS = (("x", 0.50, 0.60, 0.33), ("y", 0.30, 0.20, 0.34), ("z", 0.20, 0.20, 0.33))
Z_BINS = (("x", 0.90, 0.90, 0.55), ("y", 0.10, 0.10, 0.45))


def make_panel(n_dates=48):
    rows = []
    for d, day in enumerate(pd.date_range("2026-01-01", periods=n_dates).strftime("%Y-%m-%d")):
        markets = [(f"n{d}-{j}", N_BINS, "xxxyz"[(d + j) % 5]) for j in range(4)]
        markets.append((f"z{d}", Z_BINS, "y" if d % 6 == 0 else "x"))
        for market_id, bins, winner in markets:
            for label, price, p_a, p_b in bins:
                rows.append(
                    {
                        "market_id": market_id, "decision_date": day, "outcome_label": label,
                        "winning_label": winner, "market_price": price,
                        "feature_p_a": p_a, "feature_p_b": p_b,
                    }
                )
    return pd.DataFrame(rows)


class Pick:
    """A forecaster whose probability is simply the feature `col`."""

    def __init__(self, col):
        self.col = col

    def fit(self, X, y):
        return None

    def predict_proba(self, X):
        p = X[self.col].to_numpy(float)
        return np.column_stack([1 - p, p])


panel = make_panel()
rules = {
    "historical (top-prob)": None,
    "live, min_edge=0.20": TradePolicy.live(min_edge=0.20),
    "live, min_edge=0.05": TradePolicy.live(min_edge=0.05),
}
for rule, policy in rules.items():
    scores = []
    for name, col in (("A", "feature_p_a"), ("B", "feature_p_b")):
        trades = rolling_origin_eval(
            panel, Pick(col), min_train_rows=20, stride=1, policy=policy
        ).trades
        pnl = "no trades" if trades.empty else f"{trades['realized_pnl'].sum():+.1f}"
        scores.append(f"{name}: {pnl}")
    print(f"{rule:24} {'   '.join(scores)}")

panel.to_parquet("panel.parquet")  # the next blocks read it back
```

```text
historical (top-prob)    A: +32.6   B: -64.0
live, min_edge=0.20      A: no trades   B: +23.9
live, min_edge=0.05      A: +25.4   B: -12.7
```

Tuned by the historical rule, the winner is A. Held to the live rule at the default threshold,
A never trades and B is the only model that makes money. At `min_edge=0.05` A wins again: the
threshold you trade with changes the answer, which is why the policy is a parameter and not a
global.

## Tuning with the live policy

```python
import numpy as np
import pandas as pd

from pmlab import TradePolicy, TunedForecaster

panel = pd.read_parquet("panel.parquet")  # the panel of the block above


class Pick:  # the same forecaster as above, so this block runs on its own
    def __init__(self, col):
        self.col = col

    def fit(self, X, y):
        return None

    def predict_proba(self, X):
        p = X[self.col].to_numpy(float)
        return np.column_stack([1 - p, p])


def build(params):
    return Pick(params["col"])


def space(trial):
    return {"col": trial.suggest_categorical("col", ["feature_p_a", "feature_p_b"])}


def search(policy):
    tuner = TunedForecaster(
        build, space, n_trials=12, random_state=0, stride=1, min_train_rows=20, policy=policy
    )
    return tuner.tune(panel)


print("historical :", search(None).best_params_)
print("live  0.20 :", search(TradePolicy.live(min_edge=0.20)).best_params_)
print("live  0.05 :", search(TradePolicy.live(min_edge=0.05)).best_params_)
```

```text
historical : {'col': 'feature_p_a'}
live  0.20 : {'col': 'feature_p_b'}
live  0.05 : {'col': 'feature_p_a'}
```

The arguments that define the objective:

| Argument | Meaning | Default |
|---|---|---|
| `policy` | the decision rule each trial is scored under: `TradePolicy.live(min_edge=...)`, `TradePolicy.top_bin_yes()` or `None` | `None`: the historical rule, exactly the search of 0.11 and before |
| `costs` | a `CostModel` (fee, slippage, depth) pricing each fill. Under the live rule it decides which trades clear `min_edge` | the 30 bps fee, fill at the quote |
| `taker_bps` | the fee alone, as before. Not allowed together with `costs` | `None` (30 bps) |
| `metric` | `"total_pnl"`, `"mean_pnl"` or `"hit_rate"` of the trades the policy takes | `"total_pnl"` |

Use the `min_edge` you pass to `record-trades --min-edge` (the default is 0.20) and the `CostModel`
you will trade with. The default of `policy=None` is kept so existing searches reproduce; it is the
right choice only if you really do trade the top bin of every market.

Still true with a policy: tuning uses only training folds. Every trial is scored by
`rolling_origin_eval`, which fits strictly on `decision_date < eval_date`, whatever the policy.

### Choosing the metric under a threshold

* `total_pnl` rewards edge *and* volume, and is what a book of such trades earns. It is the default
  and the one to use unless you have a reason not to.
* `mean_pnl` is the average per trade, so a configuration that trades once and wins looks perfect.
  Add a floor on the number of trades if you use it.
* `hit_rate` ignores prices: a NO on a 0.95 favourite wins often and pays little.

### A search that never trades

A trial that makes no trade scores `-inf`. If every trial does (a `min_edge` nothing clears on this
panel), `tune()` raises a `UserWarning` instead of returning noise silently: the sampler learned
nothing and `best_params_` is just the first trial. Lower `min_edge`, widen the panel, or reduce
`stride`.

## Reproducibility: the record says which objective picked the parameters

A tuned backtest is an ordinary backtest whose `model_params` are the tuned ones, so
`pmlab reproduce` re-runs it like any other. It does not search again (a search is Python
callables, `build_fn` and `param_space`, and callables do not go in a JSON record). What the record
adds is *how the parameters were chosen*, so you can run the search again and land on the same ones:

* `BacktestConfig.resolve(..., tuning=tuner.tuning_config())` stores the objective, every default
  spelled out: `metric`, `n_trials`, `policy`, `costs`, `stride`, `min_train_rows`, `flat_stake` and
  the `best_score`. A run that was not tuned has no `tuning` key at all: its record is byte-for-byte
  what it was before.
* `resolve_seeds(seed, optuna=tuner.random_state)` records the sampler seed beside the others, in
  `seeds`, the one place a run's randomness is written down.

```python
import json
from pathlib import Path

import pandas as pd

from pmlab import (
    BacktestConfig, CostModel, LGBMForecaster, TradePolicy, TunedForecaster,
    capture_backtest, resolve_seeds, run_backtest,
)

panel = pd.read_parquet("panel.parquet")
policy = TradePolicy.live(min_edge=0.05)
costs = CostModel(taker_bps=30.0, slippage_bps=50.0)


def build(params):
    return LGBMForecaster(random_state=7, n_jobs=1, **params)


def space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 10, 30, step=10),
        "num_leaves": trial.suggest_int("num_leaves", 4, 8),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 20),
    }


tuner = TunedForecaster(
    build, space, n_trials=4, random_state=3, stride=10, min_train_rows=60,
    policy=policy, costs=costs,
).tune(panel)

config = BacktestConfig.resolve(
    plugin="demo", model_params=tuner.best_params_, stride=10, min_train_rows=60,
    costs=costs, policy=policy, tuning=tuner.tuning_config(),
)
seeds = resolve_seeds(11, optuna=tuner.random_state)
outcome = run_backtest(panel, config, seeds)

record = capture_backtest(
    panel, config, seeds, outcome.metrics, source=str(Path("panel.parquet").resolve())
)
Path("metrics.json").write_text(json.dumps({"reproducibility": record.to_dict()}, indent=2))
print("tuned with:", json.dumps(record.config["tuning"]["policy"]))
print("seeds:", record.seeds)
```

```text
tuned with: {"selection": "best_edge", "directions": ["yes", "no"], "min_edge": 0.05}
seeds: {'python_random': 11, 'numpy': 11, 'lightgbm': 11, 'optuna': 3}
```

The record re-runs from the command line, from any directory:

```bash
pmlab reproduce metrics.json
```

```text
REPRODUCED: 5 metrics match (rtol=1e-09, atol=1e-12; counts exact).
```

To repeat the *search*, rebuild the tuner from the record (`TradePolicy.from_dict(block["policy"])`,
`CostModel(**block["costs"])`, `random_state=record.seeds["optuna"]`) and call `tune` on the same
data (the record's data fingerprint tells you whether it is): the same objective, sampler seed and
panel give the same `best_params_`.

## Limits

* `pmlab reproduce` verifies the *backtest* of a tuned run, not the search itself. The tuning block
  is provenance, and a search is repeated from Python, as above.
* The Optuna sampler is deterministic for a given Optuna version; a different version can draw a
  different sequence (the environment block of the record lists the version, and `pmlab reproduce`
  reports library drift).
* `TunedForecaster` scores one panel with one policy. To tune for two thresholds, run two searches.
