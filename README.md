<div align="center">

<img src="https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white" alt="Python 3.12">
<img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License">
<img src="https://img.shields.io/badge/coverage-94%25-brightgreen" alt="Coverage 94%">
<img src="https://img.shields.io/badge/tests-149%20passing-brightgreen" alt="149 tests">
<img src="https://img.shields.io/badge/code%20style-ruff-black" alt="Ruff">
<img src="https://img.shields.io/badge/typed-py.typed-blue" alt="PEP 561 typed">

# pmlab

**A generic, plugin-based ML framework for Polymarket prediction markets.**

[Installation](#installation) · [Quickstart](#quickstart) · [Architecture](#architecture) · [Writing a Plugin](#writing-a-plugin) · [CLI](#cli-reference) · [Contributing](#contributing)

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
        # Fetch open markets from Polymarket
        return [...]

    def fetch_features(self, spec: MarketSpec, horizon: str, **kwargs) -> dict[str, float]:
        # Return numeric features for this market at this decision point
        return {"feature_trend": 0.65, "feature_volume": 1200.0}

    def fetch_truth(self, spec: MarketSpec, **kwargs) -> float | str | None:
        # Return the realized outcome (or None if unresolved)
        return 31.5  # numeric for range markets, string for categorical

    def build_training_row(self, spec: MarketSpec, horizon: str, **kwargs) -> dict | None:
        features = self.fetch_features(spec, horizon)
        truth = self.fetch_truth(spec)
        if truth is None:
            return None
        winning_label = spec.resolve_winning_bin(float(truth))
        return {"market_id": spec.market_id, "decision_horizon": horizon,
                "winning_label": winning_label, "market_price": 0.3, **features}
```

### 2. Run a backtest

```python
import pandas as pd
from pmlab import LGBMForecaster
from pmlab.backtest.rolling_origin import rolling_origin_eval
from pmlab.backtest.holdout_gate import HoldoutGateResult

# panel: DataFrame with columns market_id, decision_date, outcome_label,
#        winning_label, market_price, segment, + feature_* columns
panel = pd.read_parquet("data/historical_panel.parquet")

model = LGBMForecaster()
result = rolling_origin_eval(panel, model, stride=30, min_train_rows=100)

print(f"Total trades: {len(result.trades)}")
print(f"Total PnL:    {result.trades['realized_pnl'].sum():.2f}")
print(f"Hit rate:     {(result.trades['realized_pnl'] > 0).mean():.1%}")
```

### 3. Promote a champion

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
    print(f"Champion published: {manifest.model_name}")
    print(f"Allowed segments: {manifest.get_allowed_segments()}")
# If gate.decision == "NO_GO", publish() raises ValueError — by design.
```

### 4. Paper trade

```python
from pmlab import EdgeSignal, PaperBroker, SettlementEngine

# Record signals
broker = PaperBroker(
    trades_path="artifacts/ops/forward_paper_trades.json",
    allowed_segments=manifest.get_allowed_segments(),
    flat_stake=1.0,
)

signals = [
    EdgeSignal(
        market_id="m_001",
        city_or_segment="Buenos Aires",
        target_date="2026-05-12",
        horizon="previous_evening",
        outcome_label="warm",
        direction="yes",
        gamma_price=0.35,
        model_prob=0.65,
        best_edge=0.297,
        yes_edge=0.297,
        no_edge=-0.30,
    )
]
new_trades = broker.record(signals)
print(f"Recorded {len(new_trades)} new trades")

# Settle when markets resolve
engine = SettlementEngine(plugin=MyPlugin(), trades_path=broker.trades_path)
summary = engine.settle_all(specs=plugin.discover_markets())
print(f"Settled: {summary['settled']}, PnL: {summary['total_pnl']:+.2f}")
```

---

## Architecture

```
Polymarket API
      │
      ▼
MarketPlugin.discover_markets()  ──►  list[MarketSpec]
      │
      ├── fetch_features(spec, horizon)  ──►  dict[str, float]
      ├── fetch_truth(spec)              ──►  float | str | None
      └── build_training_row(spec, horizon)  ──►  dict | None
                │
                ▼
        rolling_origin_eval(panel, model)      ◄── LGBMForecaster
                │
                ├── BacktestMetrics  (PnL, hit_rate, avg_edge)
                │
                └── HoldoutGate.evaluate(required_segments)
                            │
                          GO │ NO_GO → ValueError (hard gate)
                            │
                  ChampionManifest.publish(model, gate)
                            │
                     champion.json + champion.pkl
                            │
                  PaperBroker.record(signals)
                  (segment gate · stale guard · dedup)
                            │
                  SettlementEngine.settle_all(specs)
                  (calls plugin.fetch_truth + is_truth_final)
```

### Module map

| Package | Responsibility |
|---|---|
| `pmlab.core` | `MarketSpec`, `OutcomeBin`, `Position`, `compute_edge`, `estimate_fee` |
| `pmlab.plugins` | `MarketPlugin` ABC, `PluginRegistry`, reference plugins |
| `pmlab.plugins.weather_tmax` | Reference implementation — temperature markets |
| `pmlab.plugins.sports_f1` | Categorical outcome plugin — F1 race markets |
| `pmlab.markets` | `GammaClient`, `ClobClient` — Polymarket API access |
| `pmlab.backtest` | `rolling_origin_eval`, `HoldoutGateResult`, `BacktestMetrics` |
| `pmlab.modeling` | `MarketForecaster` ABC, `LGBMForecaster`, `ChampionManifest` |
| `pmlab.execution` | `EdgeSignal`, `PaperBroker`, `SettlementEngine` |
| `pmlab.workspace` | `WorkspaceContext` — multi-workspace path isolation |

---

## Writing a Plugin

See the full guide at [`docs/plugin-authoring.md`](docs/plugin-authoring.md).

### The four methods

```python
class MarketPlugin(ABC):
    family: str  # unique identifier

    def discover_markets(self, **kwargs) -> list[MarketSpec]:
        """Fetch open markets from Polymarket for your family."""

    def fetch_features(self, spec: MarketSpec, horizon: str, **kwargs) -> dict[str, float]:
        """Return numeric features for one (market, decision-horizon) pair."""

    def fetch_truth(self, spec: MarketSpec, **kwargs) -> float | str | None:
        """Return the realized outcome, or None if not yet resolved."""

    def build_training_row(self, spec: MarketSpec, horizon: str, **kwargs) -> dict | None:
        """Assemble one labeled training row, or None if data unavailable."""

    # Optional override:
    def is_truth_final(self, spec: MarketSpec, **kwargs) -> bool:
        """Return True when truth is final (not a preliminary reading)."""
```

### Using dependency injection (recommended)

```python
class MyPlugin(MarketPlugin):
    family = "my_markets"

    def __init__(self, gamma_client=None, data_client=None):
        self._gamma = gamma_client or GammaClient()
        self._data = data_client  # None = use real API; mock in tests

    def discover_markets(self, **kwargs):
        if self._gamma is None:
            raise RuntimeError("gamma_client required")
        return [self._build_spec(m) for m in self._gamma.fetch_markets(tag="my_tag")]
```

This makes unit testing trivial — inject mocks, never hit the network:

```python
from unittest.mock import MagicMock

mock_gamma = MagicMock()
mock_gamma.fetch_markets.return_value = [{"id": "m1", "question": "..."}]
plugin = MyPlugin(gamma_client=mock_gamma)
```

### Registering your plugin

```python
from pmlab import PluginRegistry
from my_package import MyPlugin

registry = PluginRegistry()
registry.register(MyPlugin())

plugin = registry.get("my_markets")
```

---

## Bundled Plugins

### `WeatherTmaxPlugin`

Handles `"Highest temperature in [city] on [date]?"` markets.

```python
from pmlab.plugins.weather_tmax.plugin import WeatherTmaxPlugin
from pmlab.markets.gamma_client import GammaClient

plugin = WeatherTmaxPlugin(gamma_client=GammaClient())
markets = plugin.discover_markets()
```

Features: ECMWF forecast temperature, ensemble spread, lead time, city baseline.  
Truth: official observations via Wunderground / NOAA / CWA.

### `SportsF1Plugin`

Handles `"F1 [GP] winner?"` and similar categorical race markets.

```python
from pmlab.plugins.sports_f1.plugin import SportsF1Plugin
plugin = SportsF1Plugin(gamma_client=GammaClient())
```

Categorical bins — winning label is a driver/team name string, not a float.

---

## CLI Reference

```bash
# Print version
pmlab version

# Check paper trade status
pmlab status

# Discover open markets
pmlab scan-markets --plugin weather_tmax

# Record paper trades from latest scan-edge output
pmlab record-trades --plugin weather_tmax --min-edge 0.20

# Settle open positions against resolved markets
pmlab settle-trades --plugin weather_tmax

# Run walk-forward backtest (stride ≥ 10 enforced)
pmlab backtest --plugin weather_tmax --model lgbm_baseline --stride 30

# Promote a model to champion if gate is GO
pmlab promote-champion path/to/model.pkl --gate-path path/to/gate.json --plugin weather_tmax
```

### Workspace isolation

Use `scripts/pmlab-workspace` to scope commands to a workspace, which sets all
`PMLAB_*` environment variables automatically:

```bash
# Scan markets in the ops_daily workspace
scripts/pmlab-workspace ops_daily pmlab scan-markets --plugin weather_tmax

# Run backtest in the historical_real workspace
scripts/pmlab-workspace historical_real pmlab backtest --plugin weather_tmax --stride 30
```

Available workspaces: `ops_daily`, `historical_real`, `recent_core_eval`, `weather_train`.

---

## Key Design Decisions

### Hard gate on champion promotion

`ChampionManifest.publish()` raises `ValueError` if `gate.decision != "GO"`. This is not configurable. A model that failed the holdout gate **cannot** be promoted, even manually. Investigate the failure; don't bypass the gate.

### City/segment gate reads from `champion.json`

The `PaperBroker` reads `allowed_segments` from `champion.json` — the gate at **promotion time** — not from any intermediate benchmark file that could be overwritten by a subsequent failed retrain.

### No-lookahead guarantee

`rolling_origin_eval` trains strictly on rows with `decision_date < eval_date`. Training data never includes the evaluation date or any future date. This is asserted in the test suite.

### Plugin `is_truth_final()`

`SettlementEngine` never settles a trade until `plugin.is_truth_final(spec)` returns `True`. Data sources with finalization lags (e.g. weather stations that issue preliminary then revised readings) can override this method to prevent premature settlement.

---

## Development

### Setup

```bash
git clone https://github.com/ArtBreguez/polymarket-lab
cd polymarket-lab
uv sync --extra dev
```

### Run tests

```bash
uv run pytest                              # all tests
uv run pytest --cov=src/pmlab             # with coverage
uv run pytest tests/integration/ -v       # integration suite only
```

### Lint and type-check

```bash
uv run ruff check src/ tests/             # lint
uv run ruff check src/ tests/ --fix       # auto-fix
uv run mypy src/                          # type check
```

### Build the library

```bash
uv build
# Outputs: dist/pmlab-0.1.0-py3-none-any.whl
#          dist/pmlab-0.1.0.tar.gz
```

---

## Project Status

| Module | Status | Coverage |
|---|---|---|
| `core` (PnL, edge, fees, MarketSpec) | ✅ Stable | 100% |
| `plugins` (ABC, registry) | ✅ Stable | 100% |
| `plugins/weather_tmax` | ✅ Stable | 96% |
| `plugins/sports_f1` | ✅ Skeleton | 94% |
| `backtest` (rolling_origin, gate, metrics) | ✅ Stable | 94–100% |
| `modeling` (LGBM, calibration, champion) | ✅ Stable | 90–95% |
| `execution` (broker, settlement) | ✅ Stable | 72–99% |
| `markets` (Gamma, CLOB) | ✅ Stable | 98–100% |
| `workspace` | ✅ Stable | 100% |
| `cli` | 🔧 Shell | 91% |

**Tests:** 149 passing · **Coverage:** 94% · **Python:** 3.12

---

## Contributing

1. Fork the repo
2. Create a branch: `git checkout -b feat/my-plugin`
3. Write your plugin in `src/pmlab/plugins/<family>/`
4. Add tests in `tests/plugins/<family>/` — **TDD required** (RED → GREEN → commit)
5. Run `uv run pytest && uv run ruff check src/ tests/`
6. Open a pull request

See [`docs/plugin-authoring.md`](docs/plugin-authoring.md) for the complete plugin guide.

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built to trade · Designed to be extended · Tested to be trusted
</div>
