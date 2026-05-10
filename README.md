# pmlab

**pmlab** is a generic, plugin-based ML framework for [Polymarket](https://polymarket.com) prediction markets.

Build, backtest, and paper-trade ML models for any Polymarket market family — weather, sports, politics, crypto — using a unified pipeline.

## Features

- **Plugin architecture** — each market family implements a `MarketPlugin` interface; the core handles everything else
- **Full backtest engine** — no-lookahead rolling-origin evaluation with champion promotion gate
- **Paper trading lifecycle** — scan edge → record trades → settle → PnL tracking
- **TDD-first** — every module has tests; 70%+ coverage enforced by CI

## Quickstart

```bash
pip install pmlab

# Scan for edges using the weather plugin
pmlab scan-edge --plugin weather_tmax --model champion

# Check paper trade status
pmlab status
```

## Writing a Plugin

```python
from pmlab.plugins.base import MarketPlugin
from pmlab.core.market_spec import MarketSpec

class MyPlugin(MarketPlugin):
    family = "my_market_family"

    def discover_markets(self, **kwargs) -> list[MarketSpec]: ...
    def fetch_features(self, spec, horizon, **kwargs) -> dict: ...
    def fetch_truth(self, spec, **kwargs) -> float | str | None: ...
    def build_training_row(self, spec, horizon, **kwargs) -> dict | None: ...
```

See [docs/plugin-authoring.md](docs/plugin-authoring.md) for the full guide.

## License

MIT
