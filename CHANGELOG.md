# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com/).

## [0.7.1] — 2026-08-30

### Added — Closed training loop

- **`build_panel_from_snapshots(store, family, resolutions)`** — assembles a
  no-lookahead training panel by joining point-in-time snapshots
  (`FeatureSnapshotStore`) with realized market truth. For each resolved market
  it uses the latest snapshot captured *strictly before* that market's
  `decision_date`, so no feature can originate after the decision — no lookahead
  by construction. This closes the loop the data layer was built for:
  capture open markets over time → markets resolve → build the panel → train →
  backtest → gate.

Validated end-to-end against live Polymarket data (376 political markets, a
two-day snapshot-capture simulation of a cron): the `as_of` join correctly
selected the pre-decision snapshot, the leakage guard passed on honest features,
and the panel flowed through backtest → gate. 100% coverage on the new code.

## [0.7.0] — 2026-08-30

### Added — Data & feature layer (`pmlab.data`)

Turns the previously empty `pmlab.data` package into the reproducibility
backbone. Motivated directly by dogfooding the library against the live
Polymarket API, which exposed that the framework could not honor its own
no-lookahead principle during ingestion, and silently accepted a leaking
feature (AUC 1.000, no warning).

- **`FeatureSnapshotStore`** — append-only, point-in-time store of per-market
  feature snapshots (one parquet file per family), keyed and de-duplicated on
  `(market_id, captured_at)`. `as_of(family, cutoff)` returns the latest snapshot
  per market captured strictly before the cutoff, so a retrain uses the features
  a decision could actually have seen — never the post-resolution state. This is
  the missing piece that lets ingestion be no-lookahead.
- **`check_no_leakage` / `LeakageError` / `LeakageReport`** — scores each
  `feature_*` column by how well it alone separates the label (`max(auc, 1-auc)`)
  and flags any above `max_auc` (default 0.97). Catches the exact trap from the
  dogfood (a feature derived from the outcome) instead of failing silently.
- **`build_panel`** — typed panel builder that assembles plugin rows into a
  validated training panel: enforces the required schema (`market_id`,
  `decision_date` as YYYY-MM-DD, `outcome_label`, `winning_label`,
  `market_price`, ≥1 `feature_*`), coerces features to float, drops `None` rows,
  and optionally runs the leakage guard. Validates `market_price` too (rejects
  non-numeric and all-NaN — it drives PnL/edge). Turns a KeyError-deep-in-the-
  backtest into a clear error at build time.

### Changed
- **`publish.yml` now creates a GitHub Release automatically** on `v*` tag, right
  after the PyPI upload (auto-generated notes, wheel + sdist attached, marked
  latest). Tagging is now the single action that ships a version end to end.

100% test coverage on the new package.

## [0.6.1] — 2026-08-30

### Fixed — Lifecycle composition (found by dogfooding as a real user)
- **`compute_metrics` now accepts the `rolling_origin_eval` trade log directly.**
  It derives the win/loss `outcome` from the sign of `realized_pnl` when the
  column is absent (and treats `edge` as optional). Previously
  `compute_metrics(result.trades)` raised `KeyError: 'outcome'` — the two central
  public functions did not compose, and even the CLI worked around it by hand.
- **`HoldoutGateResult.evaluate` gains an aggregate mode.** `required_segments`
  is now optional; when omitted (or an empty list) the whole trade log is graded
  as a single `"all"` segment, so a backtest can be turned into a GO/NO_GO
  decision without first splitting into segments or hand-building the dataclass.
  An empty `required_segments=[]` now falls back to this aggregate grade instead
  of auto-passing (`all([]) == True` would have returned GO on zero segments —
  a champion hard-gate breach).

### Changed
- **`rolling_origin_eval` trade log now includes `outcome` and `segment`
  columns.** `outcome` is derived from PnL sign; `segment` is passed through from
  the panel when present, else defaults to `"all"`. This lets the trade log flow
  straight into `compute_metrics` and `HoldoutGateResult.evaluate` — closing the
  `backtest → metrics → gate → champion` loop with no manual glue.

---

## [0.6.0] — 2026-08-30

### Added — Modeling
- **Multiclass calibration.** `CalibratedForecaster` now supports >2 classes via
  one-vs-rest (new `MulticlassCalibrator`), so F1 / multi-outcome political
  markets are calibrated too. Binary behavior is unchanged.
- **Multiclass diagnostics.** `multiclass_brier` (with a climatology skill score,
  `MulticlassBrier` dataclass) and `reliability_data_multiclass` (per-class
  reliability curves) complement the existing binary Brier decomposition.
- **`TunedForecaster`** — Optuna hyperparameter search that scores every trial
  through `rolling_origin_eval`, so model selection is **walk-forward and
  lookahead-free** (never a random shuffle). Optimizes `total_pnl`, `mean_pnl`,
  or `hit_rate`. Optuna is an optional dependency: `pip install pmlab[tune]`.

### Added — Backtest
- **Leakage-aware CV.** `purged_kfold` (contiguous test blocks with an embargo
  band, López de Prado style) and `embargoed_split` (expanding-origin walk-forward
  with an embargo gap) in `pmlab.backtest.cv`.

### Changed
- `CalibratedForecaster` raises on <2 classes (was: raised on >2). Multiclass is
  now first-class.

### Infrastructure
- New optional extra `tune` (Optuna). 376 tests, high coverage on new modules.

---

## [0.5.0] — 2026-08-30

### Added — Modeling
- `EnsembleForecaster` — weighted-average blend of any N `MarketForecaster` members
  behind the same interface (drop-in for backtest/champion/brokers). Validates
  weights (non-negative, positive sum), renormalizes rows, pickles members.
- `ConformalForecaster` — split-conformal classification with a distribution-free
  marginal coverage guarantee. `predict_set()` returns per-row prediction sets
  containing the true label with probability ≥ 1 − alpha; `predict_proba()` still
  passes through so it stays a drop-in forecaster. Non-empty sets guaranteed.
- `CalibratedForecaster` — wraps any binary `MarketForecaster` and calibrates its
  positive-class probability on a held-out split (`isotonic` or `sigmoid`).
- `SigmoidCalibrator` — Platt scaling (1-D logistic) calibrator; more
  sample-efficient than isotonic on small calibration sets.

### Infrastructure
- Untracked committed `.pyc` / `__pycache__` files (already in `.gitignore`).
- 333 tests, 85% coverage.

---

## [0.4.0] — 2026-05-13

### Changed — LiveBroker
- Replaced custom HMAC-SHA256 auth with `py-clob-client` (`ClobClient`) — proper L1 ECDSA + L2 API creds, matching Polymarket CLOB requirements
- Constructor now requires `private_key` (L1, `0x`-prefixed hex) in addition to `api_key / api_secret / api_passphrase` (L2)
- `place_order` → uses `create_order + post_order` via `ClobClient`
- `cancel_order / cancel_all_orders` → `client.cancel / client.cancel_all`
- `get_open_orders` → `client.get_orders`; `get_balance` → `client.get_balance_allowance`
- `preflight()` health check added

### Changed — WeatherTmaxPlugin
- `discover_markets`: paginates Gamma API with real tmax regex filter; no fake `temperature` tag
- `fetch_features`: canonical keys matching pmtmax training schema (`forecast_temperature_2m_max/mean/min`, dew_point, humidity, wind, cloud, lead_hours)
- `build_training_row`: single-row output with all `feature_*` keys
- `_build_spec`: fully parses city/date from question regex, token_ids from `clobTokenIds` JSON, outcome_schema with prices, unit C/F

### Added — GammaClient
- `TmaxMarketInfo` dataclass: `market_id`, `city`, `target_date`, `unit`, `token_ids`, `outcome_labels/prices`, `end_date`, `active`
- `GammaClient.discover_tmax_markets()` — paginate all active markets, filter with pmtmax-identical logic, return `list[TmaxMarketInfo]`
- `AsyncGammaClient.discover_tmax_markets()` — async variant

### Infrastructure
- PyPI publish via GitHub Actions OIDC Trusted Publisher — no secrets stored in repo, auto-publishes on `v*` tags
- 301 tests, 83% coverage

---

## [0.3.0] — 2026-05-10

### Added
- `SklearnForecaster` — scikit-learn wrapper implementing `MarketForecaster` ABC (LogisticRegression, RandomForest, etc.)
- `TypedCache[T]` — generic typed wrapper around `DiskCache` with PEP 695 compatibility
- Plugin auto-discovery via `PluginRegistry.discover()` — scans entry points for `pmlab.plugins`
- CLI `report` command — generates HTML report from paper trade DB without re-running backtest
- CI coverage gate raised to 70%

### Fixed
- `TypedCache` PEP 695 syntax compatibility with Python 3.12
- 66 ruff violations resolved across all modules (`ruff format --check` added to CI)

---

## [0.2.0] — 2026-05-09

### Added
- `PmlabSettings` — Pydantic settings class: `PMLAB_ARTIFACTS_DIR`, `PMLAB_LOG_LEVEL`, `PMLAB_DRY_RUN`, CLOB/Gamma API base URLs; `from_env()` classmethod
- Full CLI (`pmlab` entry point): `version`, `status`, `scan-markets`, `record-trades`, `settle-trades`, `backtest`, `promote-champion`
- `WorkspaceContext` — multi-workspace path isolation via `PMLAB_ARTIFACTS_DIR`
- `scripts/pmlab-workspace` — bash wrapper to scope CLI to a named workspace
- Retry + structured logging — all API clients use exponential backoff; `logging.getLogger("pmlab")` hierarchy
- Gamma/CLOB pagination — `GammaClient` auto-paginates `next_cursor`; `AsyncClobClient` respects `asyncio.Semaphore`
- mypy strict clean — `py.typed` marker, all public APIs fully typed

---

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
