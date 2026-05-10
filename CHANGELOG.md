# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] — 2026-05-10

### Added
- `MarketPlugin` ABC — interface for domain-specific plugins
- `PluginRegistry` — register and look up plugins by family name
- `WeatherTmaxPlugin` — reference implementation for temperature markets
- `SportsF1Plugin` — categorical outcome plugin for F1 race markets
- `MarketSpec` + `OutcomeBin` — generic market descriptor
- `Position` + `settle_position` — binary outcome PnL accounting
- `compute_edge` — after-cost probability edge
- `estimate_fee` + `flat_stake_size` — trade sizing utilities
- `rolling_origin_eval` — walk-forward backtest with no-lookahead guarantee
- `BacktestMetrics` + `compute_metrics` — PnL, hit rate, edge statistics
- `HoldoutGateResult` — per-segment GO/NO_GO gate
- `MarketForecaster` ABC — model protocol
- `LGBMForecaster` — LightGBM binary/multiclass implementation
- `IsotonicCalibrator` — probability calibration
- `ChampionManifest` — hard-gate champion publish (raises on NO_GO)
- `EdgeSignal` — typed scan-edge output dataclass
- `PaperBroker` — record trades with segment gate, stale guard, dedup
- `SettlementEngine` — settle open trades via plugin truth resolution
- `WorkspaceContext` + `scripts/pmlab-workspace` — workspace isolation
- `GammaClient` + `fetch_gamma_markets` — Polymarket Gamma API
- `ClobClient` + `fetch_token_prices` — Polymarket CLOB price fetch
- CLI: `pmlab version`, `status`, `scan-markets`, `record-trades`, `settle-trades`, `backtest`, `promote-champion`
- GitHub Actions CI: lint (ruff) + pytest (70% coverage gate) + build
- 123+ tests, 92%+ coverage
