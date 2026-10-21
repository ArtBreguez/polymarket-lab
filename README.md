<div align="center">

<img src="https://img.shields.io/pypi/v/pmlab?color=blue" alt="PyPI version">
<img src="https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white" alt="Python 3.12">
<img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
<img src="https://img.shields.io/badge/coverage-85%25-brightgreen" alt="Coverage 85%">
<img src="https://img.shields.io/badge/tests-433%20passing-brightgreen" alt="433 tests">
<img src="https://img.shields.io/badge/code%20style-ruff-black" alt="Ruff">
<img src="https://img.shields.io/badge/typed-py.typed-blue" alt="PEP 561 typed">

# pmlab

**A generic, plugin-based ML framework for Polymarket prediction markets.**

[Installation](#installation) · [Quickstart](#quickstart) · [Architecture](#architecture) · [Writing a Plugin](#writing-a-plugin) · [CLI](#cli-reference) · [API Reference](#api-reference) · [Roadmap](ROADMAP.md) · [Contributing](#contributing)

</div>

---

## What is pmlab?

`pmlab` is a Python library that lets you build, backtest, and paper-trade machine learning models for **any Polymarket prediction market** — weather, sports, politics, crypto, or anything else.

The framework is **completely domain-agnostic**: you write a small plugin that tells pmlab how to find markets, generate features, and resolve outcomes. pmlab handles everything else — edge calculation, walk-forward backtesting, champion promotion with a hard gate, paper trading with stale-signal guards, and settlement.

**Battle-tested design** — the architecture is extracted from two live production systems:
- [`polymarket-tmax-lab`](https://github.com/ArtBreguez/polymarket-tmax-lab) — weather/temperature markets (active since 2025)
- `f1-polymarket-lab` — Formula 1 race outcome markets

---

## Installation

```bash
pip install pmlab
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add pmlab
```

**Requirements:** Python 3.12+

---

## Quickstart

### 1. Implement a plugin

```python
from pmlab import MarketPlugin, MarketSpec, OutcomeBin

class MyPlugin(MarketPlugin):
    family = "my_markets"

    def discover_markets(self, **kwargs) -> list[MarketSpec]:
        return [...]

    def fetch_features(self, spec: MarketSpec, horizon: str, **kwargs) -> dict[str, float]:
        return {"feature_trend": 0.65, "feature_volume": 1200.0}

    def fetch_truth(self, spec: MarketSpec, **kwargs) -> float | str | None:
        return 31.5

    def build_training_row(self, spec: MarketSpec, horizon: str, **kwargs) -> dict | None:
        # One row per (market, bin). This market has one bin, "Yes", so one row.
        # `market_price` is that bin's YES price, observed before the decision; return
        # None when you have none. A made-up default (0.5) would be traded against.
        features = self.fetch_features(spec, horizon)
        truth = self.fetch_truth(spec)
        price = self.price_before(spec, kwargs["decision_time"])   # your own lookup
        if truth is None or price is None:
            return None
        return {"market_id": spec.market_id, "decision_horizon": horizon,
                "decision_date": kwargs["decision_time"][:10],
                "outcome_label": "Yes",
                "winning_label": spec.resolve_winning_bin(float(truth)) or "No",
                "market_price": price, **features}
```

A market with **several bins** (temperature ranges, a race winner) is several rows: one per
bin, each at that bin's own price, so the backtest sees the bins it could have bought and not
only the one that won. Override `build_training_rows` for it; see
[One row per bin](docs/plugin-authoring.md#one-row-per-bin). `weather_tmax` does.

### 2. Run a backtest

```python
import pandas as pd
from pmlab import LGBMForecaster
from pmlab.backtest.rolling_origin import rolling_origin_eval

panel = pd.read_parquet("data/historical_panel.parquet")
model = LGBMForecaster()
# `min_train_rows` counts panel rows: with K rows per market (one per bin) it is reached
# after min_train_rows / K markets, so scale it by K.
result = rolling_origin_eval(panel, model, stride=30, min_train_rows=100)

print(f"Total trades: {len(result.trades)}")
print(f"Total PnL:    {result.trades['realized_pnl'].sum():.2f}")
print(f"Hit rate:     {(result.trades['realized_pnl'] > 0).mean():.1%}")

# By default the backtest is the historical one: YES on each market's most probable bin.
# To backtest what `pmlab record-trades` actually trades (the best-edge bin and side, YES
# or NO, when the after-fee edge clears the threshold), name the policy:
from pmlab import TradePolicy

live = rolling_origin_eval(panel, model, stride=30, min_train_rows=100,
                           policy=TradePolicy.live(min_edge=0.20))
print(live.trades["direction"].value_counts())   # yes / no

# The trade log composes straight into metrics and the go/no-go gate —
# no manual column wrangling.
from pmlab import compute_metrics
from pmlab.backtest.holdout_gate import HoldoutGateResult

metrics = compute_metrics(result.trades)          # derives outcome from PnL
gate = HoldoutGateResult.evaluate(result.trades)  # aggregate GO / NO_GO
print(metrics, "->", gate.decision)
```

### 3. Evaluate calibration

```python
from pmlab import brier_decomposition, reliability_data

# Murphy (1973) decomposition: BS = reliability - resolution + uncertainty
result = brier_decomposition(y_true, y_prob)
print(f"Brier score:  {result.brier_score:.4f}")
print(f"Skill score:  {result.skill_score:.4f}")
print(f"Reliability:  {result.reliability:.4f}")
print(f"Resolution:   {result.resolution:.4f}")

# Data for a reliability diagram
centers, mean_pred, frac_pos = reliability_data(y_true, y_prob, n_bins=10)
```

### 4. Promote a champion

```python
from pmlab import ChampionManifest, HoldoutGateResult

gate = HoldoutGateResult.evaluate(
    trades=result.trades,
    required_segments=["Buenos Aires", "Atlanta", "Dallas"],
    min_trades_per_segment=40,
    min_pnl_per_segment=0.0,
)
print(f"Gate decision: {gate.decision}")  # "GO" or "NO_GO"

if gate.decision == "GO":
    manifest = ChampionManifest.publish(
        model=model,
        gate=gate,
        output_dir="artifacts/public_models",
        plugin_family="my_markets",
    )
    print(f"Allowed segments: {manifest.get_allowed_segments()}")
# If gate.decision == "NO_GO", publish() raises ValueError — by design.
```

### 5. Paper trade with Kelly sizing

```python
from pmlab import EdgeSignal, PaperBroker, kelly_stake_size

# Kelly-sized position (quarter-Kelly, max 5% of bankroll per trade)
stake = kelly_stake_size(
    win_prob=0.65,
    entry_price=0.35,
    bankroll=200.0,
    fraction=0.25,
    max_exposure=0.05,
)

broker = PaperBroker(
    trades_path="artifacts/ops/forward_paper_trades.json",
    allowed_segments=manifest.get_allowed_segments(),
    flat_stake=stake,
)
new_trades = broker.record(signals)
```

### 6. Generate an HTML report

```python
from pmlab import generate_report

trades = broker.load_trades()
generate_report(
    trades,
    output_path="reports/session_report.html",
    title="May 2026 Paper Trading",
    brier_score=result_diagnostics.brier_score,
)
```

Opens a self-contained dark-themed HTML with equity curve, per-segment breakdown, and full trade log.

### 7. Go live (dry-run first)

```python
from pmlab import LiveBroker

# Test without sending real orders (L1 private key + L2 API credentials)
with LiveBroker(
    private_key="0x...",          # L1 ECDSA private key
    api_key="...",
    api_secret="...",
    api_passphrase="...",
    dry_run=True,
) as broker:
    receipt = broker.place_order(token_id="tok_abc", side="BUY", price=0.35, size=14.28)
    print(receipt.status)  # "dry_run"

# Check balance before going live
with LiveBroker(
    private_key="0x...",
    api_key="...",
    api_secret="...",
    api_passphrase="...",
) as broker:
    balance = broker.get_balance()
    print(f"Available: ${balance:.2f} USDC")
```

---

## Architecture

```
Polymarket API
      │
      ▼
MarketPlugin.discover_markets()  ──►  list[MarketSpec]
      │
      ├── fetch_features(spec, horizon)       ──►  dict[str, float]
      ├── fetch_truth(spec)                   ──►  float | str | None
      └── build_training_rows(spec, horizon)  ──►  list[dict]   (one row per bin)
                │
    [pmlab.features]  ← add_lags, rolling_stats, encode_cyclical, onehot, clip_outliers
                │
                ▼
        rolling_origin_eval(panel, model)      ◄── LGBMForecaster
                │
                ├── BacktestMetrics  (PnL, hit_rate, avg_edge)
                ├── brier_decomposition()      ◄── calibration diagnostics
                │
                └── HoldoutGate.evaluate(required_segments)
                            │
                          GO │ NO_GO → ValueError (hard gate)
                            │
                  ChampionManifest.publish(model, gate)
                            │
                     champion.json + champion.pkl
                            │
              ┌─────────────┴──────────────┐
              │                            │
        PaperBroker                   LiveBroker
        (paper trades)                (real CLOB orders)
        flat or kelly sizing          dry_run=True to test
              │
        SettlementEngine.settle_all(specs)
              │
        generate_report() → HTML report
```

### Module map

| Package | Responsibility |
|---|---|
| `pmlab.core` | `MarketSpec`, `OutcomeBin`, `Position`, `compute_edge`, `estimate_fee`, `flat_stake_size`, `kelly_fraction`, `kelly_stake_size` |
| `pmlab.features` | `add_lags`, `add_rolling_stats`, `encode_cyclical`, `encode_onehot`, `clip_outliers` |
| `pmlab.plugins` | `MarketPlugin` ABC, `PluginRegistry`, reference plugins |
| `pmlab.plugins.weather_tmax` | Reference implementation — temperature markets |
| `pmlab.plugins.sports_f1` | Categorical outcome plugin — F1 race markets |
| `pmlab.markets` | `GammaClient`, `ClobClient`, `AsyncGammaClient`, `AsyncClobClient`, `DiskCache` |
| `pmlab.backtest` | `rolling_origin_eval`, `HoldoutGateResult`, `BacktestMetrics`, `purged_kfold`, `embargoed_split` |
| `pmlab.data` | `FeatureSnapshotStore` (point-in-time), `build_panel`, `build_panel_from_snapshots`, `check_no_leakage` |
| `pmlab.modeling` | `MarketForecaster` ABC, `LGBMForecaster`, `SklearnForecaster`, `EnsembleForecaster`, `ConformalForecaster`, `CalibratedForecaster`, `TunedForecaster`, `IsotonicCalibrator`, `SigmoidCalibrator`, `MulticlassCalibrator`, `ChampionManifest`, `brier_decomposition`, `multiclass_brier`, `reliability_data` |
| `pmlab.execution` | `EdgeSignal`, `PaperBroker`, `SettlementEngine`, `LiveBroker` |
| `pmlab.reports` | `generate_report` — self-contained HTML report |
| `pmlab.tracking` | `LocalJSONTracker`, `MLflowTracker` (extra), `ExperimentTracker` — log params/metrics/gate per run |
| `pmlab.workspace` | `WorkspaceContext` — multi-workspace path isolation |

---

## Writing a Plugin

See the full guide at [`docs/plugin-authoring.md`](docs/plugin-authoring.md).

### The four methods

```python
class MarketPlugin(ABC):
    family: str  # unique identifier

    def discover_markets(self, **kwargs) -> list[MarketSpec]: ...
    def fetch_features(self, spec: MarketSpec, horizon: str, **kwargs) -> dict[str, float]: ...
    def fetch_truth(self, spec: MarketSpec, **kwargs) -> float | str | None: ...
    def build_training_row(self, spec: MarketSpec, horizon: str, **kwargs) -> dict | None: ...

    # Optional — override for a market with several bins (one row per bin). The default
    # wraps build_training_row, so a plugin written for the singular form keeps working:
    def build_training_rows(self, spec: MarketSpec, horizon: str, **kwargs) -> list[dict]: ...

    # Optional — override if your data source has a finalization lag:
    def is_truth_final(self, spec: MarketSpec, **kwargs) -> bool: ...
```

### Dependency injection (recommended for testing)

```python
class MyPlugin(MarketPlugin):
    family = "my_markets"

    def __init__(self, gamma_client=None):
        self._gamma = gamma_client or GammaClient()

# In tests — inject a mock, never hit the network:
mock_gamma = MagicMock()
plugin = MyPlugin(gamma_client=mock_gamma)
```

---

## Feature Engineering

`pmlab.features` provides common transforms so plugins don't reimplement them:

```python
from pmlab import add_lags, add_rolling_stats, encode_cyclical, encode_onehot, clip_outliers

# Lag features (with optional group-by for panel data)
df = add_lags(df, cols=["temp"], lags=[1, 3, 7], group_by="city")

# Rolling statistics
df = add_rolling_stats(df, cols=["temp"], windows=[7, 14], stats=["mean", "std"])

# Cyclical encoding (e.g. hour-of-day, day-of-year)
df["hour_sin"], df["hour_cos"] = encode_cyclical(df["hour"], period=24)

# One-hot encoding (drop_first=True avoids multicollinearity)
df = encode_onehot(df, cols=["city"], drop_first=True)

# Outlier clipping (IQR or z-score)
df = clip_outliers(df, cols=["temp", "humidity"], method="iqr", iqr_factor=3.0)
```

All transforms return a copy — originals are never mutated.

---

## Data Layer — reproducibility & no-lookahead ingestion

`pmlab.data` is the reproducibility backbone: point-in-time snapshots, a typed
panel builder, and leakage guards.

```python
from pmlab import FeatureSnapshotStore, build_panel, check_no_leakage

# 1. Capture point-in-time snapshots of OPEN markets over time. When a market
#    later resolves, join its truth to the snapshot taken before decision time —
#    the features a decision could actually have seen.
store = FeatureSnapshotStore("~/.pmlab/snapshots")
store.append("politics", [
    {"market_id": "m1", "captured_at": "2025-01-05T00:00:00Z",
     "feature_spread": 0.02, "feature_vol": 1.2e6},
])
seen = store.as_of("politics", cutoff="2025-01-06")   # latest snapshot per market before the cutoff

# 2. Assemble a schema-validated panel from plugin rows (drops None, coerces
#    features to float, raises a clear error on missing columns / bad dates, and
#    refuses a price observed at or after the decision it informs). A market with
#    several bins is several rows, one per bin, each at that bin's own price.
panel = build_panel(
    plugin.build_training_rows(s, "morning_of", decision_time="2025-01-06T08:00:00Z")
    for s in specs
)

# 3. Guard against label leakage — the silent killer. Raises LeakageError on any
#    feature that separates the outcome suspiciously well.
check_no_leakage(panel)                    # or build_panel(rows, check_leakage=True)

# 4. Close the loop: when markets resolve, build a no-lookahead panel by joining
#    each market's pre-decision snapshot to its realized truth.
from pmlab import build_panel_from_snapshots

resolutions = pd.DataFrame([
    {"market_id": "m1", "decision_date": "2025-01-10", "winning_label": "Yes"},
])
train = build_panel_from_snapshots(store, "politics", resolutions)  # uses snapshot < decision_date
```

---

## Async Market Access

For scanning many markets in parallel, use the async clients:

```python
import asyncio
from pmlab import AsyncGammaClient, AsyncClobClient

async def scan():
    async with AsyncGammaClient() as gamma:
        markets = await gamma.fetch_markets(tag="temperature", keyword="highest")

    token_ids = [m["conditionId"] for m in markets]
    async with AsyncClobClient(concurrency=20) as clob:
        prices = await clob.fetch_prices(token_ids)

    return markets, prices

markets, prices = asyncio.run(scan())
```

### TTL disk cache

Avoid redundant API calls during repeated scans:

```python
from pmlab import GammaClient, DiskCache

cache = DiskCache(cache_dir=".pmlab_cache", ttl_seconds=3600)
client = GammaClient(cache=cache)

markets = client.fetch_markets(tag="temperature")  # fetched from API
markets = client.fetch_markets(tag="temperature")  # served from cache
```

---

## Position Sizing

```python
from pmlab import flat_stake_size, kelly_fraction, kelly_stake_size

# Flat stake — always risk $5 regardless of edge
shares = flat_stake_size(flat_stake=5.0, entry_price=0.35)

# Kelly fraction — what % of bankroll to wager?
f = kelly_fraction(win_prob=0.65, entry_price=0.35, fraction=0.25)  # quarter-Kelly

# Kelly stake — USDC amount, capped at max_exposure % of bankroll
stake = kelly_stake_size(
    win_prob=0.65,
    entry_price=0.35,
    bankroll=500.0,
    fraction=0.25,
    max_exposure=0.05,  # never more than 5% of bankroll per trade
)
```

---

## Advanced Modeling

Every forecaster below implements the same `MarketForecaster` interface
(`fit` / `predict_proba` / `save` / `load`), so they compose freely and drop into
the backtest, champion gate, and brokers unchanged.

### Ensembles

Blend several models to cut variance — a weighted average of member probabilities:

```python
from pmlab import EnsembleForecaster, LGBMForecaster, SklearnForecaster

ens = EnsembleForecaster(
    forecasters=[
        LGBMForecaster(),
        SklearnForecaster(estimator="random_forest", n_estimators=200),
        SklearnForecaster(estimator="logistic_regression"),
    ],
    weights=[2.0, 1.0, 1.0],  # optional; defaults to equal weight
)
ens.fit(X_train, y_train)
proba = ens.predict_proba(X_test)
```

### Probability calibration

Edge is computed from probabilities, so calibration matters. Wrap any forecaster
to calibrate its positive-class output on a held-out split:

```python
from pmlab import CalibratedForecaster, LGBMForecaster

clf = CalibratedForecaster(LGBMForecaster(), method="isotonic")  # or "sigmoid" (Platt)
clf.fit(X_train, y_train)
calibrated = clf.predict_proba(X_test)  # calibrated (n, 2) probabilities
```

Or calibrate raw scores directly:

```python
from pmlab import SigmoidCalibrator, IsotonicCalibrator

cal = SigmoidCalibrator()          # Platt scaling — sample-efficient
cal.fit(raw_scores, true_labels)
p = cal.transform(new_scores)
```

### Conformal prediction sets

Turn a point probability into a **prediction set** with a distribution-free
coverage guarantee: on exchangeable data, the set contains the true label with
probability ≥ 1 − alpha, no matter how well-specified the base model is. A
singleton set is a confident call; a full set is the model saying "toss-up".

```python
from pmlab import ConformalForecaster, LGBMForecaster

clf = ConformalForecaster(LGBMForecaster(), alpha=0.1)  # target 90% coverage
clf.fit(X_train, y_train)

sets = clf.predict_set(X_test)   # list[set[int]] — e.g. {1}, {0, 1}, {0}
proba = clf.predict_proba(X_test)  # still a drop-in forecaster
```

### Multiclass calibration

Calibration extends to multi-outcome markets (F1 winner, multi-candidate races)
via one-vs-rest — just pass a multiclass target:

```python
from pmlab import CalibratedForecaster, SklearnForecaster, multiclass_brier

clf = CalibratedForecaster(SklearnForecaster("random_forest"), method="isotonic")
clf.fit(X_train, y_train)          # y has 3+ classes
proba = clf.predict_proba(X_test)  # calibrated (n, n_classes), rows sum to 1

score = multiclass_brier(y_test, proba)   # Brier + climatology skill score
print(score.brier_score, score.skill_score)
```

### Hyperparameter tuning (no-lookahead)

`TunedForecaster` searches with Optuna but scores every trial through
`rolling_origin_eval`, so model selection is walk-forward — it never shuffles the
future into the past. Requires the `tune` extra (`pip install pmlab[tune]`):

```python
from pmlab import TunedForecaster, SklearnForecaster

tuner = TunedForecaster(
    build_fn=lambda p: SklearnForecaster(estimator="random_forest", **p),
    param_space=lambda t: {"n_estimators": t.suggest_int("n_estimators", 50, 400, step=50)},
    n_trials=30,
    metric="total_pnl",      # or "mean_pnl" / "hit_rate"
)
tuner.tune(panel)            # panel = the same format rolling_origin_eval expects
print(tuner.best_params_, tuner.best_score_)
proba = tuner.predict_proba(X_test)   # refit on the full panel, ready to use
```

### Leakage-aware cross-validation

For time-series model selection outside the backtest, use splitters that respect
time instead of `sklearn.KFold`:

```python
import numpy as np
from pmlab import purged_kfold, embargoed_split

idx = np.arange(len(panel))            # rows sorted by decision_date
for train, test in purged_kfold(idx, n_splits=5, embargo=3):
    ...   # test is a contiguous block; train purges an embargo band around it

for train, test in embargoed_split(idx, n_splits=5, embargo=3):
    ...   # expanding-origin walk-forward: train is always strictly earlier
```

---

## Bundled Plugins

### `WeatherTmaxPlugin`

Handles `"Highest temperature in [city] on [date]?"` markets.
Features: ECMWF forecast temperature, ensemble spread, lead time, city baseline.
Truth: official observations via Wunderground / NOAA / CWA.

### `SportsF1Plugin`

Handles `"F1 [GP] winner?"` and similar categorical race markets.
Categorical bins — winning label is a driver/team name string.

---

## CLI Reference

```bash
pmlab version                                              # print version
pmlab status                                               # paper trade status
pmlab scan-markets --plugin weather_tmax                   # discover open markets
pmlab record-trades --plugin weather_tmax --min-edge 0.20  # record paper trades
pmlab settle-trades --plugin weather_tmax                  # settle resolved positions
pmlab resettle --trades forward_paper_trades.json          # fix NO trades an old release booked wrongly
pmlab backtest --plugin weather_tmax --stride 30           # walk-forward backtest of the live policy
# --policy top-bin-yes reproduces the old YES-only backtest; --min-edge matches record-trades
pmlab promote-champion model.pkl --gate-path gate.json \   # promote if gate is GO
               --plugin weather_tmax
# --costs costs.json on backtest / promote-champion / record-trades: one cost model
# end to end (see docs/costs.md)
pmlab monitor --reference panel.parquet --snapshots snaps/ # watchdog: silent when healthy
```

`pmlab backtest` simulates what `record-trades` trades: per market the best-edge bin and
side, YES or NO, taken when the after-fee edge clears `--min-edge` (default 0.20, the same
default as `record-trades`). The threshold is recorded in `champion.json`, and
`record-trades` uses it by default. See [docs/watchdog.md](docs/watchdog.md#the-backtest-expectation).

`pmlab monitor` is built for cron — it prints nothing and exits 0 when drift,
live calibration and live PnL are all within bounds, and otherwise says what is
wrong and exits non-zero (10 warn, 20 alert, 30 could not check). See
[docs/watchdog.md](docs/watchdog.md).

### Workspace isolation

```bash
scripts/pmlab-workspace ops_daily pmlab scan-markets --plugin weather_tmax
scripts/pmlab-workspace historical_real pmlab backtest --plugin weather_tmax --stride 30
```

---

## API Reference

### Core

| Symbol | Description |
|---|---|
| `MarketSpec` | Market descriptor: bins, token IDs, resolution logic |
| `OutcomeBin` | A single labeled outcome bin (label, lo, hi) |
| `Position` | Open trade: spec, entry_price, stake, direction |
| `settle_position(pos, outcome)` | Compute realized PnL for a settled position |
| `compute_edge(model_prob, market_price, fee_bps)` | After-cost probability edge |
| `estimate_fee(stake, bps)` | USDC fee for a trade |
| `flat_stake_size(stake, price)` | Share count for a flat-stake bet |
| `kelly_fraction(win_prob, price, fraction)` | Fractional Kelly bankroll fraction |
| `kelly_stake_size(win_prob, price, bankroll, ...)` | Kelly-sized USDC stake with cap |

### Features

| Symbol | Description |
|---|---|
| `add_lags(df, cols, lags, group_by)` | Add lag columns (e.g. `temp_lag7`) |
| `add_rolling_stats(df, cols, windows, stats)` | Add rolling mean/std/min/max/median |
| `encode_cyclical(series, period)` | Sin/cos encoding for periodic features |
| `encode_onehot(df, cols, drop_first)` | One-hot encode categorical columns |
| `clip_outliers(df, cols, method)` | Clip outliers via IQR or z-score |

### Markets

| Symbol | Description |
|---|---|
| `GammaClient` | Sync Gamma API client (with optional DiskCache) |
| `ClobClient` | Sync CLOB price client |
| `AsyncGammaClient` | Async Gamma API client (httpx.AsyncClient) |
| `AsyncClobClient` | Async CLOB client with semaphore concurrency |
| `DiskCache` | TTL disk cache (MD5-keyed JSON, auto-purge) |

### Modeling

| Symbol | Description |
|---|---|
| `MarketForecaster` | Abstract base — implement `fit`, `predict_proba`, `save`, `load` |
| `LGBMForecaster` | LightGBM binary/multiclass forecaster |
| `SklearnForecaster` | Any sklearn classifier (presets: logistic/RF/gradient-boosting) |
| `EnsembleForecaster` | Weighted-average blend of N member forecasters |
| `ConformalForecaster` | Split-conformal prediction sets with coverage ≥ 1−alpha |
| `CalibratedForecaster` | Wrap a forecaster; calibrate probabilities (binary or multiclass, isotonic/sigmoid) |
| `TunedForecaster` | Optuna hyperparameter search scored walk-forward (no-lookahead); needs `pmlab[tune]` |
| `IsotonicCalibrator` | Non-parametric monotonic probability calibration |
| `SigmoidCalibrator` | Platt scaling (1-D logistic) probability calibration |
| `MulticlassCalibrator` | One-vs-rest multiclass probability calibration |
| `ChampionManifest` | Publish/load champion model with hard NO_GO gate |
| `brier_decomposition(y_true, y_prob)` | Murphy (1973) Brier score decomposition (binary) |
| `multiclass_brier(y_true, y_prob)` | Multiclass Brier score + climatology skill score |
| `reliability_data(y_true, y_prob)` | Reliability diagram data (bin centers, mean pred, frac pos) |
| `reliability_data_multiclass(...)` | Per-class reliability curves (one-vs-rest) |
| `purged_kfold` / `embargoed_split` | Leakage-aware time-series CV splitters |
| `BrierDecomposition` | Dataclass: uncertainty, resolution, reliability, skill_score |

### Execution

| Symbol | Description |
|---|---|
| `EdgeSignal` | Typed output of a market scan (edge, direction, horizon) |
| `PaperBroker` | Record paper trades with segment gate, stale guard, dedup |
| `SettlementEngine` | Settle open trades via plugin truth resolution |
| `LiveBroker` | Real CLOB order execution (place, cancel, balance, dry_run) |
| `OrderReceipt` | Return value of `LiveBroker.place_order` |

### Reports

| Symbol | Description |
|---|---|
| `generate_report(trades, output_path, ...)` | Self-contained HTML report with equity curve SVG |

---

## Key Design Decisions

**Hard gate on champion promotion** — `ChampionManifest.publish()` raises `ValueError` if `gate.decision != "GO"`. Not configurable. Investigate failures; don't bypass the gate.

**`champion.json` is the source of truth** — `PaperBroker` reads `allowed_segments` from `champion.json`, not from any intermediate benchmark file that could be overwritten by a failed retrain.

**No-lookahead guarantee** — `rolling_origin_eval` trains strictly on rows with `decision_date < eval_date`. This is asserted in the test suite.

**Settlement waits for finalization** — `SettlementEngine` only settles when `plugin.is_truth_final(spec)` returns `True`. Data sources with finalization lags can override this method.

**paper → live transition** — Use `LiveBroker(dry_run=True)` to validate signals against the real CLOB API without sending orders. Flip `dry_run=False` only when validators pass.

---

## Development

```bash
git clone https://github.com/ArtBreguez/polymarket-lab
cd polymarket-lab
uv sync --extra dev

uv run pytest                              # all tests
uv run pytest --cov=src/pmlab             # with coverage report
uv run ruff check src/ tests/             # lint
uv run ruff check src/ tests/ --fix       # auto-fix
uv run mypy src/                          # type check
uv build                                  # build wheel + sdist
```

---

## Project Status

| Module | Status | Coverage |
|---|---|---|
| `core` (PnL, edge, fees, sizing, Kelly) | ✅ Stable | 100% |
| `features` (lags, rolling, cyclical, onehot) | ✅ Stable | 100% |
| `plugins` (ABC, registry) | ✅ Stable | 100% |
| `plugins/weather_tmax` | ✅ Stable | 96% |
| `plugins/sports_f1` | ✅ Skeleton | 94% |
| `backtest` (rolling_origin, gate, metrics) | ✅ Stable | 94–100% |
| `modeling` (LGBM, calibration, champion, diagnostics) | ✅ Stable | 95–100% |
| `execution` (paper broker, settlement, live broker) | ✅ Stable | 84–99% |
| `markets` (Gamma, CLOB, async, cache) | ✅ Stable | 84–98% |
| `reports` (HTML report) | ✅ Stable | 96% |
| `workspace` | ✅ Stable | 100% |
| `cli` | 🔧 Shell | 91% |

**Tests:** 237 passing · **Coverage:** 95.7% · **Python:** 3.12

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

Quick version:
1. Fork the repo
2. Create a branch: `git checkout -b feat/my-plugin`
3. Write your plugin in `src/pmlab/plugins/<family>/`
4. Add tests in `tests/plugins/<family>/` — **TDD required**
5. Run `uv run pytest && uv run ruff check src/ tests/`
6. Open a pull request

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built to trade · Designed to be extended · Tested to be trusted
</div>
