# Architecture

pmlab is a core that knows nothing about any one market and a **plugin** per market family that knows
everything about its own. The plugin turns a family's markets into rows; the core backtests, gates,
trades and monitors those rows the same way for every family.

## The pipeline

```text
Polymarket API
      │
      ▼
MarketPlugin.discover_markets()  ──►  list[MarketSpec]
      │
      ├── fetch_features(spec, horizon)       ──►  dict[str, float]
      ├── fetch_truth(spec)                   ──►  float | str | None
      └── build_training_rows(spec, horizon)  ──►  list[dict]        (one row per bin)
                │
                ▼
        build_panel(rows)                      schema check, price-timing check, leakage check
                │
                ▼
        rolling_origin_eval(panel, model)      walk-forward: trains on rows whose label was known at the eval decision
                │
                ├── BacktestMetrics            (PnL, hit rate, average edge)
                ├── brier_decomposition()      calibration diagnostics
                │
                └── HoldoutGateResult.evaluate(required_segments)
                            │
                      GO ───┴─── NO_GO  →  ChampionManifest.publish() raises ValueError
                      │
        ChampionManifest.publish(model, gate)
                      │
              champion.json + champion.pkl         (ModelRegistry archives every version)
                      │
        ┌─────────────┴──────────────┐
        │                            │
   PaperBroker                  LiveBroker
   paper trades,                real CLOB orders,
   segments from champion.json  dry_run=True to test first
        │
   SettlementEngine.settle_all(specs)              waits for plugin.is_truth_final(spec)
        │
   generate_report()  ·  pmlab monitor             HTML report · drift, calibration and PnL watchdog
```

## What each package owns

The names are the public ones: import them from `pmlab` (or from the package named in the row). Module
paths such as `pmlab.core.costs` are internal; see [Public API and stability](api-stability.md).

| Package | Responsibility |
|---|---|
| `pmlab.core` | `MarketSpec`, `OutcomeBin`, `Position`, `CostModel`, `TradePolicy`, `compute_edge`, `estimate_fee`, `flat_stake_size`, `kelly_fraction`, `kelly_stake_size`, `settle_position`. No domain logic. |
| `pmlab.plugins` | `MarketPlugin` (the ABC a family implements), `PluginRegistry`, `WeatherTmaxPlugin` (the reference plugin: temperature markets). `pmlab.plugins.sports_f1` is a skeleton for categorical F1 markets. |
| `pmlab.features` | `add_lags`, `add_rolling_stats`, `encode_cyclical`, `encode_onehot`, `clip_outliers`, `add_missing_indicators`, `add_interactions`, `TargetEncoder`, `FittedTargetEncoder`. See [Features](features.md). |
| `pmlab.data` | `FeatureSnapshotStore` (point-in-time), `build_panel`, `build_panel_from_snapshots`, `check_no_leakage`, `check_price_timing`. See [Data layer](data.md). |
| `pmlab.markets` | `GammaClient`, `ClobClient`, `AsyncGammaClient`, `AsyncClobClient`, `DiskCache`, `TypedCache`. See [Async clients](async-clients.md). |
| `pmlab.backtest` | `rolling_origin_eval`, `compute_metrics`, `BacktestMetrics`, `HoldoutGateResult`, `purged_kfold`, `embargoed_split`. No lookahead. |
| `pmlab.modeling` | `MarketForecaster` (the interface), `LGBMForecaster`, `SklearnForecaster`, `EnsembleForecaster`, `ConformalForecaster`, `CalibratedForecaster`, `TunedForecaster`, the calibrators, `brier_decomposition`, `ChampionManifest`, `ModelRegistry`. See [Modeling](modeling.md). |
| `pmlab.execution` | `EdgeSignal`, `PaperBroker`, `SettlementEngine`, `LiveBroker`, `resettle_trades`. |
| `pmlab.monitoring` | `drift_report`, `CalibrationTracker`, `health_report`: the pure core of `pmlab monitor`. See [Watchdog](watchdog.md). |
| `pmlab.reports` | `generate_report`: a self-contained HTML report. |
| `pmlab.tracking` | `LocalJSONTracker`, `ExperimentTracker`, `MLflowTracker` (the `track` extra). See [Tracking](tracking.md). |
| `pmlab.reproducibility` | `ReproducibilityRecord`, `BacktestConfig`, `run_backtest`, `capture_backtest`, `reproduce`, `load_record`, `fingerprint_panel`: seeds, versions, config and a data hash per run, re-run and verified by `pmlab reproduce`. |
| `pmlab.datasets` | `load_sample()`: the bundled synthetic weather sample and the offline clients that serve it. |
| `pmlab.workspace` | `WorkspaceContext`: every path comes from a workspace, and workspaces never share one. |
| `pmlab.cli` | The `pmlab` command. See [Command line](cli.md). |

## Extending it

A new market family is one class: subclass `MarketPlugin`, implement four methods, register it.
See the [plugin authoring guide](plugin-authoring.md). Optional heavy dependencies (Optuna, MLflow)
stay behind extras and are never imported by the core install.
