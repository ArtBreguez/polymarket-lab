# pmlab

**A generic, plugin-based ML framework for Polymarket prediction markets.**

pmlab lets you build, backtest and paper-trade models for any Polymarket market family: weather,
sports, politics, crypto. You write a small plugin that says how to find markets, build features and
resolve outcomes; pmlab does the rest: edge and cost calculation, walk-forward backtesting without
lookahead, a hard GO / NO_GO gate before a model can be promoted, paper trading with stale-signal
guards, and settlement that waits until the truth is final.

```bash
pip install pmlab        # or: uv add pmlab        (Python 3.12+)
```

```python
from pmlab.datasets import load_sample

sample = load_sample()                                  # 750 synthetic markets, no network
specs = sample.plugin(sample.days[0]).discover_markets()  # the real WeatherTmaxPlugin, offline
```

## Where to start

- **[Tutorial](tutorial.md)**: discover → panel → tune → backtest → gate → paper → report, offline,
  in one notebook that CI executes.
- **[Architecture](architecture.md)**: the pipeline and what each package owns.
- **[Command line](cli.md)**: every `pmlab` command and a first offline session.
- **[Plugin authoring](plugin-authoring.md)**: the four methods a market family implements.
- **[Running it live](live-trading.md)**: live trading, drift monitoring, live calibration and the
  watchdog behind `pmlab monitor`.
- **[API reference](reference/index.md)**: every public name, generated from its docstring.

## Stable from 1.0

pmlab follows [semantic versioning](https://semver.org) from 1.0.0. What is public (the names in
`pmlab` and a few subpackages, the `pmlab` command and its options, the files it writes) is listed
exactly in [Public API and stability](api-stability.md), and a test fails if it changes without the
change being deliberate. Nothing public is removed before 2.0.0, and not before it has been
[deprecated](deprecation.md) with a warning that names the replacement.

## The safety rules

These are properties of the framework, enforced in code and not left to convention:

1. A champion is **never published with a NO_GO gate**: `ChampionManifest.publish` raises.
2. Canonical training data is **never overwritten** without an explicit flag.
3. Workspaces are **never mixed**: live forward evidence and training data stay apart.
4. The champion's segment gate is **always read from `champion.json`**, the record made at promotion.
5. A trade is **never settled before the truth is final** (`MarketPlugin.is_truth_final`).

## Project

* Source, issues and releases: [github.com/ArtBreguez/polymarket-lab](https://github.com/ArtBreguez/polymarket-lab)
* [Changelog](https://github.com/ArtBreguez/polymarket-lab/blob/main/CHANGELOG.md) and
  [roadmap](https://github.com/ArtBreguez/polymarket-lab/blob/main/ROADMAP.md)
* Install from [PyPI](https://pypi.org/project/pmlab/)
