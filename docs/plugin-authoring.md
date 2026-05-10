# Plugin Authoring Guide

## What is a plugin?

A **MarketPlugin** is a Python class that teaches `pmlab` how to work with one Polymarket
market family: how to find markets, generate features, fetch truth, and build training data.

The core framework (backtest, paper trading, settlement) is 100% plugin-agnostic.
You write the domain logic; pmlab handles everything else.

## Minimal example

```python
from pmlab.plugins.base import MarketPlugin
from pmlab.core.market_spec import MarketSpec, OutcomeBin
from typing import Any

class CryptoPricePlugin(MarketPlugin):
    family = "crypto_price"  # unique identifier

    def discover_markets(self, **kwargs: Any) -> list[MarketSpec]:
        # Fetch open "Will BTC be above $X on date Y?" markets
        ...

    def fetch_features(self, spec: MarketSpec, horizon: str, **kwargs: Any) -> dict[str, float]:
        # Return numeric features: {"btc_price": 65000.0, "rsi_14": 52.3, ...}
        ...

    def fetch_truth(self, spec: MarketSpec, **kwargs: Any) -> float | str | None:
        # Return BTC closing price on target date, or None if not yet resolved
        ...

    def build_training_row(self, spec: MarketSpec, horizon: str, **kwargs: Any) -> dict | None:
        features = self.fetch_features(spec, horizon)
        truth = self.fetch_truth(spec)
        if truth is None:
            return None
        winning_label = spec.resolve_winning_bin(float(truth))
        return {"market_id": spec.market_id, "decision_horizon": horizon,
                "winning_label": winning_label, "market_price": 0.5, **features}
```

## The four methods

### `discover_markets(**kwargs) -> list[MarketSpec]`
Fetch currently open markets for your family from Polymarket.
Use `GammaClient` from `pmlab.markets.gamma_client` to query the API.
Return a list of `MarketSpec` objects.

### `fetch_features(spec, horizon, **kwargs) -> dict[str, float]`
Generate numeric features for a single (market, decision horizon) pair.
All feature values must be **float**. Feature names must be consistent across calls.

Horizons: `"market_open"`, `"previous_evening"`, `"morning_of"`.

### `fetch_truth(spec, **kwargs) -> float | str | None`
- Range/numeric markets: return a float (e.g. `31.2` for temperature °C).
- Categorical markets: return the winning label string (e.g. `"Verstappen"`).
- Unresolved: return `None`.

### `build_training_row(spec, horizon, **kwargs) -> dict | None`
Assemble one labeled training row. Required output keys:
```python
{
    "market_id": str,
    "decision_horizon": str,
    "winning_label": str,
    "market_price": float,
    # + all feature columns (keys starting with "feature_" recommended)
}
```
Return `None` if data is unavailable.

## Optional: `is_truth_final(spec, **kwargs) -> bool`
Return `True` when the truth from `fetch_truth()` is **final** (not preliminary).
Default: `True` whenever `fetch_truth()` returns non-None.
Override when your data source has a finalization lag (e.g. weather stations).

## Registering your plugin

```python
from pmlab.plugins.registry import PluginRegistry
from my_package import CryptoPricePlugin

registry = PluginRegistry()
registry.register(CryptoPricePlugin())

plugin = registry.get("crypto_price")
```

## Dependency injection

Don't hardcode API clients in `__init__`. Accept them as parameters:

```python
class CryptoPricePlugin(MarketPlugin):
    def __init__(self, gamma_client=None, price_client=None):
        self._gamma = gamma_client or GammaClient()
        self._prices = price_client
```

This makes your plugin testable without network calls:

```python
from unittest.mock import MagicMock

mock_gamma = MagicMock()
mock_gamma.fetch_markets.return_value = [...]
plugin = CryptoPricePlugin(gamma_client=mock_gamma)
```

## Testing your plugin

```python
from unittest.mock import MagicMock
from pmlab.core.market_spec import MarketSpec

def test_fetch_features_returns_float_dict():
    plugin = CryptoPricePlugin()
    spec = MarketSpec(market_id="btc_001", ...)
    features = plugin.fetch_features(spec, "previous_evening")
    assert isinstance(features, dict)
    assert all(isinstance(v, float) for v in features.values())

def test_truth_none_when_unresolved():
    plugin = CryptoPricePlugin(price_client=None)
    spec = MarketSpec(...)
    assert plugin.fetch_truth(spec) is None
```

## Reference implementation

See `src/pmlab/plugins/weather_tmax/plugin.py` — the most complete plugin example.
It demonstrates dependency injection, spec building from raw API dicts, and
truth resolution for numeric (temperature) markets.
