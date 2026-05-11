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
uv sync --extra dev        # installs all dev dependencies
pre-commit install         # sets up lint hooks
```

### Verify your setup

```bash
uv run pytest              # should pass 237+ tests
uv run ruff check src/ tests/
uv run mypy src/
```

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
uv run pytest --cov=src/pmlab --cov-fail-under=70
```

**Coverage gate:** 70% minimum (enforced in CI). Aim for 90%+.

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
5. Update `CHANGELOG.md` under `## [Unreleased]`

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
| `chore` | Tooling, CI, dependencies |
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
   uv run pytest --cov=src/pmlab --cov-fail-under=70
   uv run ruff check src/ tests/
   uv run mypy src/
   ```
4. Update `CHANGELOG.md` — add your changes under `## [Unreleased]`
5. Open a pull request against `main` with a clear description of what and why

---

## Code Style

- **Formatter / linter:** [ruff](https://docs.astral.sh/ruff/) — `uv run ruff check src/ tests/ --fix`
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

Open an [issue](https://github.com/ArtBreguez/polymarket-lab/issues) or start a discussion.
