# Contributing to pmlab

Thank you for your interest in contributing! This guide covers how to get started, the TDD contract, commit conventions, and the process for adding new plugins.

---

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Git

### Setup

```bash
git clone https://github.com/ArtBreguez/polymarket-lab
cd polymarket-lab
uv sync --extra dev                # installs all dev dependencies
uv sync --extra dev --extra docs   # ... plus the docs site's (mkdocs, mkdocstrings)
```

### Verify your setup

```bash
uv run pytest                          # the default suite; it should pass
uv run ruff check src/ tests/          # lint
uv run ruff format --check src/ tests/ # formatting (`uv run ruff format src/ tests/` fixes it)
uv run mypy src/                       # strict type check
```

CI runs those four, plus the suites the default run deselects (each is its own job):

| Command | What it checks |
|---|---|
| `uv run pytest -m e2e` | Builds the wheel, installs it in a fresh venv and runs the public API, the CLI, the README and the docs pages from there. Needs `uv` and the network. |
| `uv run pytest -m notebook` | Executes `notebooks/tutorial.ipynb` top to bottom, offline. |
| `uv run pytest -m perf` | Times the hot paths against the budget in [docs/performance.md](docs/performance.md). Run it alone. |
| `uv run python scripts/api_surface.py` | The public API still matches `tests/api/public_api.json`. |
| `uv run mkdocs build --strict` | The docs site builds (needs the `docs` extra). |

---

## TDD Contract — Mandatory

All new code in pmlab follows strict test-driven development:

1. **Write the test first** — import the symbol that doesn't exist yet. The test must fail with an `ImportError` or `AssertionError`.
2. **Implement minimal code** — make the test go GREEN with the simplest correct implementation.
3. **Refactor** — clean up without breaking tests.
4. **Commit** — commit test and implementation together.

```bash
# Confirm RED
uv run pytest tests/path/to/test_new.py -v  # expect FAILED or ImportError

# Implement, then confirm GREEN
uv run pytest tests/path/to/test_new.py -v  # expect PASSED

# Run full suite before pushing
uv run pytest --cov=src/pmlab
```

**Coverage gate:** 90% minimum. The number is `fail_under` in `pyproject.toml`, the single source
of truth: pytest-cov reads it, and CI runs `pytest --cov` without an override, so a miss fails the job.

---

## Adding a New Plugin

Plugins are the primary extension point. See [`docs/plugin-authoring.md`](docs/plugin-authoring.md) for the full guide.

Quick checklist:

1. Create `src/pmlab/plugins/<family>/` directory with `__init__.py` and `plugin.py`
2. Subclass `MarketPlugin` from `pmlab.plugins.base`
3. Set `family = "<your_family>"`
4. Implement all 4 abstract methods
5. Optionally override `is_truth_final()` if your data source has a finalization lag
6. Create `tests/plugins/<family>/` with `__init__.py` and test file(s)
7. Ensure `uv run pytest && uv run ruff check src/ tests/` both pass

---

## Adding a New Module

1. Create the module under the appropriate `src/pmlab/` subpackage
2. Export all public symbols via `__all__` in the module
3. Add to `src/pmlab/__init__.py` if it belongs in the top-level public API
4. Write tests in the corresponding `tests/` directory
5. Add a bullet to `CHANGELOG.md` under `## [Unreleased]` (see [Releases](#releases))

---

## Changing the Public API

From 1.0 the public API is frozen (see [docs/api-stability.md](docs/api-stability.md)): the names in
`pmlab.__all__` and the documented subpackages, the `pmlab` command and its options, and the file
formats. A test compares them with a checked-in snapshot and fails on any difference.

- **Adding** a public name, option or file key: export it, give it a docstring, add it to the page
  under `docs/reference/` (a test checks every public name has one), then
  `uv run python scripts/api_surface.py --write` and commit the snapshot with the change.
- **Removing or changing** one: deprecate it first (`pmlab._deprecation.deprecated`, see
  [docs/deprecation.md](docs/deprecation.md)); it is removed no earlier than the next major release.
- **Docs examples** import only public names (`from pmlab import ...`); a test enforces it.
- `uv run mkdocs build --strict` must pass (CI runs it on every change to `docs/` or `src/`).

### Documentation and the README

The README is also the PyPI page, so it uses absolute links only, and it carries no number that can
go stale (no test count, no coverage figure): `tests/test_readme.py` checks both. The fenced
`python` and `bash` blocks of the README and of the guide pages that
`tests/e2e/test_installed_wheel.py` names are **executed as written** by the end-to-end tests,
against the built wheel; use `text` for sample output and `shell` for commands
that must not run (an install line, a command that needs credentials or the network). A page whose
blocks build on each other is run in order, so write each block to work after the ones above it.

---

## Commit Convention

Format: `<type>: <subject>` — imperative mood, no period, max 72 chars.

| Type | When to use |
|---|---|
| `feat` | New feature or module |
| `fix` | Bug fix |
| `refactor` | Restructuring without behavior change |
| `test` | Adding or improving tests |
| `docs` | Documentation only |
| `chore` | Tooling, repository upkeep, dependencies |
| `ci` | GitHub Actions workflows |
| `perf` | Performance improvement |

Examples:
```
feat: add AsyncClobClient with semaphore concurrency
fix: champion gate reads from champion.json not retrain summary
test: add reliability_data empty bin coverage
docs: add live-trading guide for LiveBroker setup
```

---

## Pull Request Process

1. Fork the repo and create a branch: `git checkout -b feat/my-feature`
2. Follow the TDD contract above
3. Ensure all checks pass:
   ```bash
   uv run pytest --cov=src/pmlab
   uv run ruff check src/ tests/
   uv run ruff format --check src/ tests/
   uv run mypy src/
   ```
4. Update `CHANGELOG.md` — add your changes under `## [Unreleased]`
5. Open a pull request against `main` with a clear description of what and why

---

## Releases

pmlab follows [semantic versioning](https://semver.org) from 1.0 (see
[docs/deprecation.md](docs/deprecation.md) for how a public name is retired). A release is a pull
request, opened by a maintainer, that:

1. bumps `version` in `pyproject.toml` and `__version__` in `src/pmlab/__init__.py` (a test checks
   that the two and the newest `CHANGELOG.md` heading agree);
2. renames `## [Unreleased]` to `## [x.y.z] — YYYY-MM-DD` and adds a fresh empty `## [Unreleased]`
   above it, with the compare links at the bottom of the file.

Merging it to `main` publishes: `.github/workflows/publish.yml` runs the gates, uploads to PyPI,
and creates the `vX.Y.Z` tag and the GitHub release. A merge that does not bump the version
publishes nothing.

---

## Code Style

- **Formatter / linter:** [ruff](https://docs.astral.sh/ruff/) — `uv run ruff check src/ tests/ --fix` and `uv run ruff format src/ tests/`
- **Type checker:** mypy in strict mode — `uv run mypy src/`
- **Line length:** 100 characters
- **Python version:** 3.12+ syntax (use `X | Y` unions, `match`, etc.)
- All public functions and classes must have docstrings
- All public modules must have `__all__`
- No mutable default arguments
- Prefer dataclasses over plain dicts for typed return values

---

## Safety Rules (Non-Negotiable)

These rules are enforced by the framework itself or by code review:

1. **Never publish a champion with a NO_GO gate.** `ChampionManifest.publish()` hard-raises. Do not try to bypass it.
2. **Never mix workspace data.** `ops_daily` is live evidence only; `historical_real` is training data.
3. **Never settle before `is_truth_final()`.** Some data sources issue preliminary readings.
4. **Never store credentials in code.** Use environment variables or a secrets manager.
5. **Never use `LiveBroker` with real capital before paper validators pass.**

---

## Questions

Open an [issue](https://github.com/ArtBreguez/polymarket-lab/issues). To report a security problem, follow
[SECURITY.md](SECURITY.md) instead of opening a public issue.
