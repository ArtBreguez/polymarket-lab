---
template: home.html
title: pmlab
description: A plugin-based ML framework for Polymarket prediction markets. A backtest that cannot look ahead, a gate that no failing model gets past, reproducible runs and a frozen public API.
---

<!-- This page is rendered by overrides/home.html. Its text is the quickstart of the home page, and
     the search index of the home page. The hero, the guarantees and the links live in the template,
     drawn from docs/assets/data/sample-run.json. -->

## Quickstart

This runs offline on a small **synthetic** sample bundled in the package: 750 weather-temperature
markets, no network and no account. The numbers say nothing about real markets.

```python
from pmlab import (ChampionManifest, HoldoutGateResult, LGBMForecaster, TradePolicy,
                   build_panel, rolling_origin_eval)
from pmlab.datasets import load_sample

sample = load_sample()                       # 750 synthetic weather markets, checked against a checksum
resolver = sample.plugin()                   # the real WeatherTmaxPlugin, wired to the bundled file
found = [(day, spec) for day in sample.days for spec in sample.plugin(day).discover_markets()]

# One row per (market, bin), each at its own price. build_panel refuses a price seen at or after
# the decision, and check_leakage=True raises if a feature gives the outcome away.
panel = build_panel(
    [resolver.build_training_rows(s, "previous_evening", decision_time=f"{d}T12:00:00+00:00")
     for d, s in found],
    check_leakage=True,
)
# The forecast alone cannot tell a market's bins apart: add how far each bin sits from it.
bounds = {(s.market_id, b.label): [x for x in (b.lower, b.upper) if x is not None]
          for _, s in found for b in s.outcome_bins}
centre = [sum(bounds[k]) / len(bounds[k]) for k in zip(panel["market_id"], panel["outcome_label"])]
panel["feature_bin_dist"] = centre - panel["feature_fcst_tmax"]

# Walk-forward backtest: every fit sees only rows from before the day it predicts.
model = LGBMForecaster(random_state=7, n_jobs=1)
result = rolling_origin_eval(
    panel, model, stride=10, min_train_rows=2700,
    policy=TradePolicy.live(min_edge=0.05),  # what `pmlab record-trades` trades: best bin and side
)
gate = HoldoutGateResult.evaluate(result.trades)
print(len(result.trades), "trades | PnL", round(result.trades["realized_pnl"].sum(), 2), "|", gate.decision)
panel.to_parquet("panel.parquet")            # the command line below reads this file

# A gate that fails cannot publish a champion.
failed = HoldoutGateResult.evaluate(result.trades, required_segments=["Atlantis"])
try:
    ChampionManifest.publish(model, failed, "champion", plugin_family="weather_tmax")
except ValueError as err:
    print("refused:", err)
```

It prints the number of trades, the total PnL and the gate's decision, then the refusal. The same panel
runs from the command line, and the [tutorial](tutorial.md) continues from here: tuning, the gate per city,
paper trading and the report.

```shell
pip install pmlab            # Python 3.12+; or: uv add pmlab
pip install "pmlab[tune]"    # + Optuna, for TunedForecaster
pip install "pmlab[track]"   # + MLflow, for MLflowTracker
```
