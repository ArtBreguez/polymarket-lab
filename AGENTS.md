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
| `src/pmlab/data/` | FeatureSnapshotStore (point-in-time), build_panel, leakage guards |
| `src/pmlab/modeling/` | MarketForecaster ABC, LGBMForecaster, calibration, ChampionManifest |
| `src/pmlab/execution/` | EdgeSignal, PaperBroker, SettlementEngine |
| `src/pmlab/reproducibility/` | ReproducibilityRecord (seeds, versions, resolved config, data fingerprint), backtest runner shared by `backtest` and `reproduce`, `pmlab reproduce` |
| `src/pmlab/datasets/` | The bundled synthetic sample (`load_sample`) and the offline clients that feed the real weather plugin; nothing else may depend on its numbers except the tutorial notebook and its tests |
| `src/pmlab/workspace/` | WorkspaceContext — multi-workspace path isolation |
| `src/pmlab/_deprecation.py` | `deprecated` / `warn_deprecated`, `PmlabDeprecationWarning`: the only way a public name is retired (`docs/deprecation.md`) |
| `scripts/api_surface.py`, `tests/api/` | The public-API snapshot (`tests/api/public_api.json`) and the tests that enforce it |
| `docs/`, `mkdocs.yml` | The documentation site (mkdocs-material, API reference from docstrings), built `--strict` by `.github/workflows/docs.yml` |
| `src/pmlab/markets/` | Polymarket Gamma and CLOB clients (sync and async), `DiskCache` |
| `src/pmlab/features/` | Pure DataFrame transforms and out-of-fold `TargetEncoder` (copy-on-write, no I/O) |
| `src/pmlab/monitoring/` | Drift (PSI/KL), live calibration, `health_report`: the pure core of `pmlab monitor` |
| `src/pmlab/tracking/` | `ExperimentTracker` protocol, `LocalJSONTracker`, `MLflowTracker` (optional extra) |
| `src/pmlab/reports/` | `generate_report`: the self-contained HTML report |
| `src/pmlab/config.py` | `PmlabSettings` (environment-driven settings; the `PMLAB_*` variables are public) |
| `src/pmlab/cli/` | Typer CLI entry points (`pmlab`); `monitor` lives in `cli/monitor.py` |
| `README.md` | The front page **and the PyPI page**: absolute links only, no number that can go stale; `tests/test_readme.py` enforces it and the e2e job runs its code (see `CONTRIBUTING.md`) |

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

7. **The public API is frozen (1.0+).**
   Public = `pmlab.__all__` + the `__all__` of `pmlab.data/datasets/features/monitoring/reports/
   reproducibility/tracking`, the `pmlab` CLI commands and options, and the file formats in
   `docs/api-stability.md`. `tests/api/public_api.json` is the snapshot: change it only with
   `scripts/api_surface.py --write`, on purpose. Never remove or change a public name without a
   deprecation (`pmlab._deprecation.deprecated`) and a major release (`docs/deprecation.md`).
   Docs and examples import only public names (`from pmlab import ...`), never `pmlab.core.x`.

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

# Formatting (CI checks it; drop --check to fix)
uv run ruff format --check src/ tests/

# Performance budget of the hot paths (opt-in, ~1 min, run it alone; see docs/performance.md)
uv run pytest -m perf

# End-to-end: build the wheel, install it in a fresh venv, run the public API (needs uv + network)
uv run pytest -m e2e

# Tutorial notebook: execute notebooks/tutorial.ipynb top to bottom, offline (~15 s; docs/tutorial.md)
uv run pytest -m notebook

# Public API still matches the snapshot (--write to change it on purpose)
uv run python scripts/api_surface.py

# Docs site (needs `uv sync --extra docs`); must pass --strict
uv run mkdocs build --strict

# Lint
uv run ruff check src/ tests/

# Type check
uv run mypy src/

# Workspace-scoped command
scripts/pmlab-workspace ops_daily uv run pmlab scan-markets --plugin weather_tmax

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

Coverage gate: 90% minimum (`fail_under` in pyproject.toml, the single source of truth; CI
runs `pytest --cov` without an override, so a miss fails the job).

Performance budget: `rolling_origin_eval` and `build_panel` are timed against a budget in
calibration units by `uv run pytest -m perf` (own CI job, deselected by default). A slowdown
that is intended means updating the budget on purpose: see `docs/performance.md`.

Tutorial notebook: `notebooks/tutorial.ipynb` is executed by `uv run pytest -m notebook` (own CI
job, deselected by default) and is committed with its outputs stripped. Changing a public API the
notebook uses means updating the notebook in the same PR; regenerating the sample
(`scripts/make_sample_dataset.py`) changes its numbers, so re-run the notebook, e2e and default
suites. See `docs/tutorial.md`.

Docs and README: the code blocks of `README.md` and of the guide pages named in
`tests/e2e/test_installed_wheel.py` are executed as written by `uv run pytest -m e2e`, against the
built wheel, and `tests/test_readme.py` runs the README in the default suite. A block that does not
run is a documentation bug. Use `text` (output) or `shell` (not run) for anything illustrative.

---

---

## Versioning and Releases

Semantic versioning from 1.0 (`docs/deprecation.md`). A PR that changes what the wheel ships (code,
or the README, which is the PyPI page) bumps `version` in `pyproject.toml` **and** `__version__` in
`src/pmlab/__init__.py`, and adds a dated `## [x.y.z]` entry to `CHANGELOG.md`; a test checks that
the three agree. Merging to `main` with a new version publishes to PyPI, tags and creates the
GitHub release (`.github/workflows/publish.yml`); a merge without a bump publishes nothing. The
reviewer merges, not the author.

---

## Architecture

See `docs/architecture.md`: the pipeline from `MarketPlugin.discover_markets()` to
`SettlementEngine.settle_all()`, and what each package owns. The module table above is the
ownership contract; the page is the picture.
