# pmlab

[![CI](https://github.com/ArtBreguez/polymarket-lab/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ArtBreguez/polymarket-lab/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pmlab)](https://pypi.org/project/pmlab/)
[![Python](https://img.shields.io/pypi/pyversions/pmlab)](https://pypi.org/project/pmlab/)
[![License: MIT](https://img.shields.io/pypi/l/pmlab)](https://github.com/ArtBreguez/polymarket-lab/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://pmlab.hiddenterps.si/)

**A plugin-based ML framework for Polymarket prediction markets.** Build, backtest and paper-trade
models for any market family (weather, sports, politics, crypto) with a backtest that cannot look
ahead and a gate that no failing model gets past.

You write a small plugin that says how to find a family's markets, build its features and resolve its
outcomes. pmlab does the rest: edge and cost arithmetic, walk-forward backtesting, a hard GO / NO_GO
gate before a model is promoted, paper trading, settlement, drift monitoring and a cron-friendly
watchdog.

> **Not financial advice.** pmlab is research software; trading prediction markets can lose all the
> money you put in. No `pmlab` command places an order. Only a `LiveBroker` you build yourself, with
> your own Polymarket credentials, does, and `dry_run=True` is how you try it first.

## Why it is safe to build on

These are enforced in code and checked by the test suite, not left to convention.

- **No lookahead.** `rolling_origin_eval` trains only on rows whose label was already known when the
  fold decided: a decision for tomorrow's temperature is not a label until tomorrow is over.
  `build_panel` refuses a price observed at or after the decision it informs, and `check_no_leakage`
  raises when a feature gives the outcome away. [How the checks work](https://pmlab.hiddenterps.si/leakage/)
- **A failed gate cannot ship.** `ChampionManifest.publish` raises unless the gate says GO, and that
  is not configurable. Paper trading reads its allowed segments from `champion.json`, the record made
  at promotion, and a trade is never settled before the plugin says the truth is final.
- **Reproducible.** Every backtest records its seeds, library versions, resolved config and a
  fingerprint of the data; `pmlab reproduce` re-runs it and verifies the metrics come back.
  [Reproducibility](https://pmlab.hiddenterps.si/reproducibility/)
- **Stable.** From 1.0 pmlab follows [semantic versioning](https://semver.org). The public API (the
  names, the `pmlab` command and its options, the file formats) is frozen and compared with a checked-in
  snapshot on every change; nothing public is removed before 2.0 without a deprecation warning first.
  [What is public](https://pmlab.hiddenterps.si/api-stability/)

## Install

```shell
pip install pmlab            # Python 3.12+; or: uv add pmlab
pip install "pmlab[tune]"    # + Optuna, for TunedForecaster
pip install "pmlab[track]"   # + MLflow, for MLflowTracker
```

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
runs from the command line:

```bash
pmlab backtest --plugin weather_tmax --panel panel.parquet \
    --stride 10 --min-train-rows 2700 --min-edge 0.05 --output metrics.json
pmlab reproduce metrics.json     # re-runs the backtest and verifies the same metrics come back
```

The [tutorial](https://pmlab.hiddenterps.si/tutorial/) continues from here: tuning,
a per-city gate, a published champion, paper trading, settlement and an HTML report, in one notebook
that CI executes.

## The command line

| Command | What it does |
|---|---|
| `pmlab backtest` | Walk-forward backtest of a panel, recorded so it can be reproduced |
| `pmlab reproduce` | Re-run a recorded backtest and verify the metrics |
| `pmlab promote-champion` | Publish a model as champion, only if its gate is GO |
| `pmlab scan-markets` | List a family's open markets from the Gamma API |
| `pmlab record-trades` | Paper-trade the scan's signals that clear `--min-edge` |
| `pmlab status` · `report` · `resettle` | Inspect paper trades, render an HTML report, repair old settlements |
| `pmlab monitor` | The watchdog: drift, live calibration and live PnL; silent when healthy |
| `pmlab settle-trades` | A placeholder: prints how to settle with `SettlementEngine`; it settles nothing |
| `pmlab version` | Print the version |

`pmlab --help` and `pmlab <command> --help` list every option. [Command line reference](https://pmlab.hiddenterps.si/cli/)

## Bring your own market family

A plugin is one class with four methods. Everything else is shared.

```python
from pmlab import MarketPlugin

class MyPlugin(MarketPlugin):
    family = "my_markets"
    def discover_markets(self, **kwargs): ...
    def fetch_features(self, spec, horizon, **kwargs): ...
    def fetch_truth(self, spec, **kwargs): ...
    def build_training_row(self, spec, horizon, **kwargs): ...
```

[Plugin authoring guide](https://pmlab.hiddenterps.si/plugin-authoring/)

## Documentation

Everything else lives on the [documentation site](https://pmlab.hiddenterps.si/).

| | |
|---|---|
| Start here | [Tutorial](https://pmlab.hiddenterps.si/tutorial/) · [Architecture](https://pmlab.hiddenterps.si/architecture/) · [Command line](https://pmlab.hiddenterps.si/cli/) · [Plugin authoring](https://pmlab.hiddenterps.si/plugin-authoring/) |
| Build models | [Data layer](https://pmlab.hiddenterps.si/data/) · [Features](https://pmlab.hiddenterps.si/features/) · [Modeling](https://pmlab.hiddenterps.si/modeling/) · [Tuning](https://pmlab.hiddenterps.si/tuning/) · [Costs](https://pmlab.hiddenterps.si/costs/) · [Position sizing](https://pmlab.hiddenterps.si/sizing/) |
| Run it | [Live trading](https://pmlab.hiddenterps.si/live-trading/) · [Drift monitoring](https://pmlab.hiddenterps.si/drift-monitoring/) · [Watchdog](https://pmlab.hiddenterps.si/watchdog/) · [Champion registry](https://pmlab.hiddenterps.si/registry/) |
| Reference | [API reference](https://pmlab.hiddenterps.si/reference/) · [Public API and stability](https://pmlab.hiddenterps.si/api-stability/) · [Deprecation policy](https://pmlab.hiddenterps.si/deprecation/) |
| Project | [Changelog](https://github.com/ArtBreguez/polymarket-lab/blob/main/CHANGELOG.md) · [Roadmap](https://github.com/ArtBreguez/polymarket-lab/blob/main/ROADMAP.md) · [Contributing](https://github.com/ArtBreguez/polymarket-lab/blob/main/CONTRIBUTING.md) · [Security](https://github.com/ArtBreguez/polymarket-lab/blob/main/SECURITY.md) |

## Development

```shell
git clone https://github.com/ArtBreguez/polymarket-lab && cd polymarket-lab
uv sync --extra dev
uv run pytest                  # the suite; the coverage gate is in pyproject.toml
```

[CONTRIBUTING.md](https://github.com/ArtBreguez/polymarket-lab/blob/main/CONTRIBUTING.md) has the rest:
the test-first rule, the end-to-end, performance and notebook suites, and how the public API is changed.

## License

MIT. See [LICENSE](https://github.com/ArtBreguez/polymarket-lab/blob/main/LICENSE).
