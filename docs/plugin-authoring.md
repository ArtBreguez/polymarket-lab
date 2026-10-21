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
        # A binary market: one bin ("Yes"), so one row. `price` is the Yes price you
        # observed at decision time: look it up, and return None if you cannot. Never
        # substitute a default such as 0.5; the backtest would trade against it.
        features = self.fetch_features(spec, horizon)
        truth = self.fetch_truth(spec)
        price = self.yes_price_at(spec, kwargs["decision_time"])  # your own lookup
        if truth is None or price is None:
            return None
        winning_label = spec.resolve_winning_bin(float(truth))
        return {"market_id": spec.market_id, "decision_horizon": horizon,
                "decision_date": kwargs["decision_time"][:10],
                "outcome_label": "Yes", "winning_label": winning_label,
                "market_price": price, **features}
```

A market with **more than one bin** (temperature ranges, a race winner) is not one row:
see [One row per bin](#one-row-per-bin) before you write `build_training_row` for it.

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
Assemble one labeled training row: one (market, **bin**, decision). It is the only
row-building method you have to write. Required output keys:
```python
{
    "market_id": str,
    "decision_horizon": str,
    "decision_date": str,       # YYYY-MM-DD, what the backtest orders and splits on
    "outcome_label": str,       # the bin THIS row describes
    "winning_label": str,       # the bin that won
    "market_price": float,      # the YES price of outcome_label, observed before the decision
    # + all feature columns (keys starting with "feature_" required to train on them)
}
```
Return `None` if data is unavailable.

### `build_training_rows(spec, horizon, **kwargs) -> list[dict]`
Every row of one (market, decision): the plural form, and the one a panel is built
from. The default wraps `build_training_row` (`[row]`, or `[]` for `None`), so a plugin
written for the singular form needs no change. Override it for a market with several bins.

```python
from pmlab import build_panel

panel = build_panel(plugin.build_training_rows(s, "morning_of", decision_time=t) for s in specs)
```

`build_panel` takes the per-market lists as they come (and still takes single rows and
`None`).

## One row per bin

A market with several outcome bins is **several rows**: one per bin, the same `market_id`
and `decision_date`, each with that bin's `outcome_label` and *that bin's* `market_price`.

The backtest decides per market which bin to buy and whether to buy YES or NO
(`--policy live`, the default, is what `record-trades` does). It can only do that if it
sees the bins it could have chosen. A panel with a single row per market, labelled with
the bin that won, shows it only winners: every YES it buys wins and every NO it buys
loses, whatever the model says, and the PnL it reports belongs to the panel, not the
model. That was `weather_tmax` before 0.9.2
([#28](https://github.com/ArtBreguez/polymarket-lab/issues/28)): on a synthetic world with
fair prices, where the expected PnL of any policy is exactly zero, it reported a 100% hit
rate under `--policy top-bin-yes` and, with the CLI's LightGBM model, a 0% hit rate under
`--policy live`, each 18 to 20 standard deviations from zero. (The sign under `live` follows
the model: it trained on winners only, LightGBM then reports a probability of about zero for
every label, and the policy buys NO on the winner.)

Rules for the rows of one market:

1. **Every bin that has a usable price gets a row**, winner and losers alike.
   `outcome_label == winning_label` for exactly one of them.
2. **`market_price` is the price of that row's bin**, never the market's maximum or
   favourite, and never the same number on every row.
3. **The price is point-in-time.** It was observed strictly before the decision
   (the rule `FeatureSnapshotStore.as_of` applies to features). Say when and where
   it came from, with `price_ts` (ISO-8601), `decision_ts` and `price_source`;
   `build_panel` then refuses a panel in which a price is at or after its decision
   (`LeakageError`, see `check_price_timing`). A panel without these columns is not
   checked, so a plugin that omits them is trusted, not verified.
4. **A bin with no usable price gets no row.** Never fill one in (0.5 included): the
   backtest would trade against a quote nobody made. Count what you drop, and say why.
5. The model sees only `feature_*` columns. Features that are the same for every bin of a
   market (a forecast) cannot tell the bins apart; if the model should, add bin-aware
   features (the bin's bounds, the forecast's distance from them).

How the counts differ from a one-row panel: a panel has K rows per market, but a
backtest makes one trade per (market, decision date), so `num_trades`, hit rate, the gate's
per-segment `n` and the watchdog's counts are all per trade and need no change. Two
settings are in **rows**, not markets: `--min-train-rows` (20 rows is 4 markets at K=5, so
scale it by K) and any row-wise split you do yourself, where the bins of one market belong
on the same side.

### `weather_tmax`

`WeatherTmaxPlugin.build_training_rows(spec, horizon, decision_time=...)` is the reference
(`decision_time` is required; there is no way to check a price against a decision that
has no time). It reads each bin's price from `spec.metadata["outcome_schema"]`, where
`discover_markets` put the Gamma `outcomePrices` and stamped when it saw them
(`price_observed_at`, `price_source = "discovery"`).

**The only price this plugin has is the discovery-time one.** `pmlab.markets` reads the
CLOB's *current* midpoint (`fetch_token_prices`) and keeps no history, and the snapshot
store is keyed per market and has no per-bin price. So a training row's price is the
Gamma price at the moment `discover_markets` ran, and that must be before the decision. To
train on a better price (a CLOB midpoint at the horizon, say), put it in the spec's
`outcome_schema` with its own `observed_at` and `source` before calling.

```python
from collections import Counter
from pmlab import build_panel
from pmlab.plugins.weather_tmax.plugin import WeatherTmaxPlugin

plugin = WeatherTmaxPlugin(gamma_client=..., forecast_client=..., truth_client=...)
specs = plugin.discover_markets()   # prices are stamped "observed now"

exclusions: Counter[str] = Counter()
rows = [
    row
    for spec in specs
    for row in plugin.build_training_rows(
        spec, "morning_of", decision_time="2026-06-05T08:00:00Z", exclusions=exclusions
    )
]
panel = build_panel(rows)
print(exclusions)   # e.g. Counter({'price_missing': 3, 'price_after_cutoff': 12, ...})
```

A bin is left out, and counted under its reason, when its price is `price_missing`,
`price_invalid` (not a number in [0, 1]), `price_degenerate` (exactly 0 or 1, which is what
a settled market shows), `price_untimed` (no readable observation time) or
`price_after_cutoff`. A market is left out entirely when it is `market_unresolved`, its
truth is in no bin (`market_truth_unbinned`), or no bin is usable
(`market_no_usable_price`). Without an `exclusions` mapping the counts are logged at WARNING
per market, never lost. A market with a bin left out still has rows for the others, so the backtest
chooses among the bins that were priced.

`discover_markets` stamps *its own* time on the prices, so a market discovered **after** it
resolved (or after the decision) has prices that are not decision-time prices, and every
bin of it is excluded as `price_after_cutoff`. Capture the specs at the decision, store
them, and build the rows when the market resolves.

The deprecated `build_training_row` is the single winning-bin row, kept so existing callers
do not break; it emits a `DeprecationWarning`, and a panel made of it is biased the way
described above.

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
It demonstrates dependency injection, spec building from raw API dicts, truth
resolution for numeric (temperature) markets, and one training row per bin at a
point-in-time price.
