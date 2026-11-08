## What and why

<!-- One or two sentences. Link the issue it closes: "Closes #N". -->

## Checklist

- [ ] Test first: the new test failed before the change (see CONTRIBUTING.md)
- [ ] `uv run pytest`, `uv run ruff check src/ tests/`, `uv run ruff format --check src/ tests/` and `uv run mypy src/` pass
- [ ] Public API: unchanged, or `scripts/api_surface.py --write` run on purpose and a deprecation added for anything removed
- [ ] `CHANGELOG.md` has an entry under `[Unreleased]`
- [ ] Docs and README examples still run (the e2e job executes them)
