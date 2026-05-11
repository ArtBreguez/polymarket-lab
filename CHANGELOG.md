# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] — 2026-05-10

### Added — Core
- `MarketSpec` + `OutcomeBin` — generic market descriptor with bin resolution logic
- `Position` + `settle_position` — binary outcome PnL accounting
- `compute_edge` — after-cost probability edge calculation
- `estimate_fee` — USDC taker fee estimate
- `flat_stake_size` — share count for a flat-stake bet
- `kelly_fraction` — fractional Kelly bankroll fraction (binary outcome formula)
- `kelly_stake_size` — Kelly-sized USDC stake with `max_exposure` cap

### Added — Features
- `add_lags` — lag features with optional `group_by` for panel data
- `add_rolling_stats` — rolling mean/std/min/max/median with optional `group_by`
- `encode_cyclical` — sin/cos encoding for periodic features (hour, DOY, month)
- `encode_onehot` — one-hot encoding with `drop_first` and dtype control
- `clip_outliers` — IQR and z-score outlier clipping (returns copy, no mutation)

### Added — Markets
- `GammaClient` / `fetch_gamma_markets` — Polymarket Gamma API client with optional `DiskCache`
- `ClobClient` / `fetch_token_prices` — CLOB midpoint price fetcher
- `AsyncGammaClient` — async Gamma API client (`httpx.AsyncClient`) with keyword filter
- `AsyncClobClient` — async CLOB client with `asyncio.Semaphore` concurrency control
- `DiskCache` — TTL disk cache (MD5-keyed JSON files, `get/set/delete/clear/purge_expired`)

### Added — Modeling
- `MarketForecaster` ABC — protocol: `fit`, `predict_proba`, `save`, `load`
- `LGBMForecaster` — LightGBM binary/multiclass implementation
- `IsotonicCalibrator` — isotonic regression probability calibration
- `ChampionManifest` — hard-gate champion publish (raises `ValueError` on NO_GO)
- `brier_decomposition` — Murphy (1973) decomposition: reliability, resolution, uncertainty, skill score
- `reliability_data` — reliability diagram data (bin centers, mean predicted prob, fraction positive)
- `BrierDecomposition` — dataclass for decomposition results

### Added — Backtest
- `rolling_origin_eval` — walk-forward backtest with strict no-lookahead guarantee
- `BacktestMetrics` + `compute_metrics` — PnL, hit rate, avg edge statistics
- `HoldoutGateResult` / `SegmentGateResult` — per-segment GO/NO_GO gate

### Added — Execution
- `EdgeSignal` — typed scan-edge output dataclass
- `PaperBroker` — record paper trades with segment gate, staleness guard, dedup
- `SettlementEngine` — settle open trades via `plugin.fetch_truth + is_truth_final`
- `LiveBroker` — real CLOB order execution: `place_order`, `cancel_order`, `cancel_all_orders`, `get_open_orders`, `get_balance`; HMAC-SHA256 L1 auth; `dry_run=True` mode
- `OrderReceipt` — dataclass returned by `LiveBroker.place_order`
- `LiveBrokerError` — exception for API failures

### Added — Reports
- `generate_report` — self-contained dark-themed HTML report: summary cards, equity curve SVG, per-segment breakdown, full trade log; zero JS/CSS dependencies

### Added — Plugins
- `MarketPlugin` ABC — 4 required methods: `discover_markets`, `fetch_features`, `fetch_truth`, `build_training_row`
- `PluginRegistry` — register and look up plugins by `family` name
- `WeatherTmaxPlugin` — reference implementation for temperature markets
- `SportsF1Plugin` — categorical outcome plugin for F1 race markets

### Added — Workspace & CLI
- `WorkspaceContext` — multi-workspace path isolation via `PMLAB_ARTIFACTS_DIR`
- `scripts/pmlab-workspace` — wrapper to scope CLI commands to a workspace
- CLI commands: `version`, `status`, `scan-markets`, `record-trades`, `settle-trades`, `backtest`, `promote-champion`

### Added — Infrastructure
- GitHub Actions CI: ruff lint + pytest (70% coverage gate) + wheel build
- `py.typed` marker — PEP 561 compliant typed library
- `AGENTS.md` — shared contract for AI coding agents
- `docs/plugin-authoring.md` — complete plugin authoring guide
- `docs/live-trading.md` — LiveBroker setup and paper-to-live transition guide
- `docs/features.md` — feature engineering transforms reference
- `docs/calibration.md` — Brier decomposition and calibration diagnostics guide
- `docs/reports.md` — HTML report generation guide
- `docs/async-clients.md` — async API clients and DiskCache guide
- `CONTRIBUTING.md` — contribution guide with TDD contract
- 237 tests, 95.7% coverage
