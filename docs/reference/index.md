# API reference

Generated from the docstrings of the **public** API, and only that: the names in `pmlab.__all__`
and in the `__all__` of `pmlab.data`, `pmlab.datasets`, `pmlab.features`, `pmlab.monitoring`,
`pmlab.reports`, `pmlab.reproducibility` and `pmlab.tracking`. What counts as public, and what is
promised about it, is on [Public API and stability](../api-stability.md).

Import every name from `pmlab` itself, as these pages show (`from pmlab import CostModel`), except
the few that only a subpackage exports (`pmlab.reproducibility`, `pmlab.datasets`,
`pmlab.tracking.MLflowTracker`). Module paths such as `pmlab.core.costs` are where the code lives
today, not an address: they are internal and may move in any release.

| Page | Covers |
|------|--------|
| [Markets, edge and sizing](core.md) | `MarketSpec`, `compute_edge`, `CostModel`, `TradePolicy`, Kelly sizing |
| [Backtest and gate](backtest.md) | `rolling_origin_eval`, `compute_metrics`, `HoldoutGateResult`, expectations |
| [Data and features](data.md) | `build_panel`, the snapshot store, the leakage guards, the transforms |
| [Modeling](modeling.md) | forecasters, calibration, diagnostics, `ChampionManifest`, `ModelRegistry` |
| [Execution](execution.md) | `EdgeSignal`, brokers, settlement |
| [Monitoring](monitoring.md) | drift, live calibration, the watchdog |
| [Market data clients](markets.md) | Gamma and CLOB clients, sync and async, caches |
| [Plugins and workspaces](plugins.md) | `MarketPlugin`, `PluginRegistry`, `WorkspaceContext` |
| [Reproducibility](reproducibility.md) | capture and re-run a run |
| [Tracking and reports](tracking.md) | experiment trackers, `generate_report` |
| [Sample dataset](datasets.md) | `load_sample` and its offline clients |
| [Configuration](config.md) | `PmlabSettings`, `PmlabDeprecationWarning` |

The command line has no generated reference: run `pmlab --help`, or `pmlab <command> --help`. Its
commands and options are frozen as listed on the [stability page](../api-stability.md#the-command-line).
