# pmlab Agent Guide

## Purpose

`pmlab` is a **generic, plugin-based ML framework** for Polymarket prediction markets.
It is designed to be published as a Python library (pip/uv installable).

The framework is market-family agnostic: each domain (weather, sports, politics, crypto)
implements a `MarketPlugin` and the core handles everything else.

Use this file as the shared contract for all agents (Codex, Claude, etc).

---

## Module Ownership

| Module | Owns |
|--------|------|
| `src/pmlab/core/` | MarketSpec, OutcomeBin, PnL, edge, fees, sizing — no domain logic |
| `src/pmlab/plugins/` | MarketPlugin ABC, PluginRegistry, domain plugin implementations |
| `src/pmlab/plugins/weather_tmax/` | Weather/temperature market plugin (reference implementation) |
| `src/pmlab/plugins/sports_f1/` | F1 race outcome plugin |
| `src/pmlab/backtest/` | Rolling-origin eval, metrics, HoldoutGate — no lookahead |
| `src/pmlab/modeling/` | MarketForecaster ABC, LGBMForecaster, calibration, ChampionManifest |
| `src/pmlab/execution/` | EdgeSignal, PaperBroker, SettlementEngine |
| `src/pmlab/workspace/` | WorkspaceContext — multi-workspace path isolation |
| `src/pmlab/markets/` | Polymarket Gamma API client, CLOB price client (Phase 4) |
| `src/pmlab/cli/` | Typer CLI entry points |

---

## Safety Rules

1. **Never publish a champion with NO_GO gate.**
   `ChampionManifest.publish()` hard-raises `ValueError` if `gate.decision != "GO"`.
   This is not configurable. A failed gate must be investigated, not bypassed.

2. **Never overwrite canonical training data without explicit `--allow-overwrite`.**
   Experiments use variant output paths. Canonical data is for stable baselines only.

3. **Never mix workspace data.**
   `ops_daily` is live forward evidence only. `historical_real` is training data.
   A model trained on ops_daily data is contaminated — do not promote it.

4. **Always use `scripts/pmlab-workspace <name> <cmd>` for workspace-scoped operations.**
   This sets `PMLAB_ARTIFACTS_DIR` and friends so all paths resolve correctly.

5. **City/segment gate always reads from `champion.json`.**
   The `recent_core_benchmark_summary.json` (or equivalent) is overwritten on each retrain.
   `champion.json` reflects the gate at promotion time and is the authoritative source.

6. **Never settle trades before `plugin.is_truth_final()` returns True.**
   Some data sources have a lag (e.g. weather stations). Wait for finalization.

---

## Adding a New Plugin

1. Create `src/pmlab/plugins/<your_family>/plugin.py`
2. Subclass `MarketPlugin` from `pmlab.plugins.base`
3. Set `family = "your_family"`
4. Implement all 4 abstract methods: `discover_markets`, `fetch_features`, `fetch_truth`, `build_training_row`
5. Optionally override `is_truth_final()` if your data source has a finalization lag
6. Add tests in `tests/plugins/<your_family>/`
7. Register via `PluginRegistry.register(YourPlugin())`

See `src/pmlab/plugins/weather_tmax/plugin.py` as the reference implementation.
See `docs/plugin-authoring.md` for the full guide.

---

## Standard Commands

```bash
# Install in development mode
uv sync --extra dev

# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src/pmlab --cov-report=term-missing

# Lint
uv run ruff check src/ tests/

# Type check
uv run mypy src/

# Workspace-scoped command
scripts/pmlab-workspace ops_daily uv run pmlab scan-edge --plugin weather_tmax

# Build distribution
uv build
```

---

## Commit Convention

Format: `<type>: <subject>` — imperative, no period.

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `ci`, `perf`

Examples:
- `feat: add SportsF1Plugin with categorical outcome bins`
- `fix: champion gate reads from champion.json not retrain summary`
- `test: add rolling_origin no-lookahead assertion`
- `docs: update plugin-authoring guide with is_truth_final`

---

## TDD Contract

Every new module gets tests BEFORE implementation:
1. Write test → run → confirm RED (import error or assertion failure)
2. Implement minimal code → run → confirm GREEN
3. Refactor → run → confirm still GREEN
4. Commit

Coverage gate: 70% minimum (enforced in pyproject.toml).

---

## Architecture Summary

```
Polymarket API ──► MarketPlugin.discover_markets() ──► list[MarketSpec]
                                │
                                ├──► MarketPlugin.fetch_features()   ──► features dict
                                ├──► MarketPlugin.fetch_truth()       ──► realized outcome
                                └──► MarketPlugin.build_training_row() ──► training row
                                              │
                    ┌─────────────────────────┘
                    │
                    ▼
            rolling_origin_eval(panel, model)
                    │
                    ├──► BacktestMetrics (PnL, hit_rate, edge)
                    └──► HoldoutGate.evaluate(required_segments) ──► GO / NO_GO
                                            │
                                           GO
                                            │
                              ChampionManifest.publish(model, gate)
                              (hard-raises on NO_GO)
                                            │
                                     champion.json
                                     champion.pkl
                                            │
                              PaperBroker.record(signals)
                              (reads allowed_segments from champion.json)
                                            │
                              SettlementEngine.settle_all(specs)
                              (calls plugin.fetch_truth + is_truth_final)
```
