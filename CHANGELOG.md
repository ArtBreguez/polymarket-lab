# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

## [1.0.1] — 2026-11-08

A documentation and repository clean-up. **No behaviour changes and no public API changes**
(`tests/api/public_api.json` is identical). It is a release because the README is the PyPI page.

### Documentation

- **The README is rewritten** (863 lines to about 160): what pmlab is, the four guarantees (no
  lookahead, a gate that cannot be bypassed, reproducibility, a frozen API), install, a quickstart
  that runs on the bundled sample, the command line at a glance, and links into the docs site. What
  was wrong in the old one, now gone:
  - static badges and figures (`tests-3109 passing`, `coverage-96%`, "Tests: 3109 passing") that
    were typed in by hand and go stale with every commit; the badges are now computed (CI status,
    PyPI version, Python versions, license) and a test fails if a number comes back;
  - the `pmlab backtest` example was missing the required `--panel`;
  - the API tables had signatures that do not exist (`flat_stake_size(stake, price)`,
    `compute_edge(model_prob, market_price, fee_bps)`, `estimate_fee(stake, bps)`,
    `settle_position(pos, outcome)`);
  - the quickstart used undefined names (`y_true`, `signals`, `result_diagnostics`, `plugin`) and
    could not run; the new one does, from the installed wheel, and says it is synthetic data;
  - relative links and images, which break on PyPI; a link to `polymarket-tmax-lab`, a private
    repository; "active since 2025", which the repository's history does not show;
  - `scripts/pmlab-workspace` presented as part of the installed package (it is not shipped), and
    `settle-trades` listed as if it settled (it is a placeholder that prints instructions);
  - a module status table whose per-package coverage ranges no longer matched the code (it said
    `markets` 84-98% and `cli` 91%).
- **Five new guide pages** take the useful parts of the old README, each with code that runs:
  [Architecture](docs/architecture.md) (the pipeline and what each package owns),
  [Command line](docs/cli.md) (every command, honestly, and a first offline session),
  [Data layer](docs/data.md), [Modeling](docs/modeling.md) (ensembles, calibration, conformal
  sets, multiclass, time-aware CV) and [Position sizing](docs/sizing.md). The plugin guide gains a
  table of the bundled plugins and says `sports_f1` is a skeleton. Sections that duplicated a page
  (feature engineering, async clients and the cache, the API tables) were removed in favour of it.
- **`ROADMAP.md`** says the 1.0 roadmap is done (#15, #16, #17, #18 and the follow-ups #33, #35,
  #36, #39), that nothing is scheduled, and keeps the known gaps as ideas. It was "as of v0.10.0".
- **`CONTRIBUTING.md`** and **`AGENTS.md`** match the repository: the coverage gate is the 90% from
  `pyproject.toml` (`CONTRIBUTING.md` said 70%), `ruff format --check`, the `e2e`/`notebook`/`perf`
  suites and `api_surface.py` are listed, and a Releases section describes the version bump that
  publishes. `CONTRIBUTING.md` no longer tells contributors to `pre-commit install` (there was no
  configuration). In `AGENTS.md`, `pmlab scan-edge` (a command that does not exist) became
  `scan-markets`, and the module table gains the packages it was missing (`features`, `monitoring`,
  `tracking`, `reports`, `config`).
- **`CHANGELOG.md`**: entries for 0.8.0, 0.8.1 and 0.8.2, which were published without one; the
  dates of 0.1.0 to 0.3.0 are the PyPI release dates (0.2.0 was dated before 0.1.0); 0.5.0 and
  0.6.1 say they were never released on their own; compare links for every tagged version;
  an `[Unreleased]` section, which `CONTRIBUTING.md` already told contributors to use.

### Changed

- **Package metadata**: a description that says what it does, more keywords, classifiers
  (`Operating System :: OS Independent`, `Topic :: Office/Business :: Financial`, `Python :: 3 ::
  Only`, among others; the status was already `5 - Production/Stable`), and a `Security` URL.
- **`publish.yml` no longer overrides the coverage gate**: it ran `--cov-fail-under=70`, below the 90%
  that `pyproject.toml` sets and `CONTRIBUTING.md` calls the single source of truth.
- **`pre-commit` removed from the `dev` extra**: no `.pre-commit-config.yaml` ever existed, so it
  installed a tool nothing used. The lockfile loses it and the packages only it needed.
- **An unused test fixture** (`tmp_workspace` in `tests/conftest.py`, referenced nowhere) is removed.
- **`.gitignore`** covers what a local run writes (`coverage.xml`, `artifacts/`, `.pmlab_cache/`,
  editor folders) and its comment is in English.

### Added

- `SECURITY.md` (how to report a vulnerability, and the two things to know about credentials and
  pickled models), short issue and pull-request templates, and `.github/dependabot.yml` (weekly,
  grouped, for the uv lockfile and the GitHub Actions).

### Tests

- `tests/test_readme.py` (default suite): no static badge or figure carries a number, every link
  is absolute (PyPI cannot resolve a relative one) and points at a file or page that exists, every
  `pmlab` command and option shown exists and every required option is shown, and the README's
  `python` and `bash` blocks run as written from an empty directory.
- `tests/test_docs_pages.py`: the command table of `docs/cli.md` is the command set of the
  snapshot, and the package map of `docs/architecture.md` lists real packages and public names.
- `tests/test_package_metadata.py`: classifiers, the Python versions CI tests, the changelog's
  ordering, dates and links, and the repository files this release adds.
- End-to-end (`pytest -m e2e`): the README, `docs/cli.md`, `docs/modeling.md`, `docs/data.md` and
  `docs/sizing.md` run as written from the installed wheel, and the wheel's long description is the
  README and passes `twine check --strict`. `run_doc_walkthrough.py` gains `--session`, which runs
  a page's blocks in one namespace.

---

## [1.0.0] — 2026-11-08

### Added — the stability guarantee: a frozen public API, a deprecation policy, and the docs site

Closes #17, the last issue of the v1.0.0 roadmap. From this release pmlab follows
[semantic versioning](https://semver.org): nothing public is removed or changed incompatibly
before 2.0.0, and not before it has been deprecated with a warning that names the replacement.
**No behaviour changes**: every existing call, command and file works as it did in 0.12.1.

- **What is public is written down**: [docs/api-stability.md](docs/api-stability.md). Public is
  `pmlab.__all__` plus the `__all__` of `pmlab.data`, `pmlab.datasets`, `pmlab.features`,
  `pmlab.monitoring`, `pmlab.reports`, `pmlab.reproducibility` and `pmlab.tracking`; the `pmlab`
  command, its sub-commands and options (flags, defaults, environment variables); and the file
  formats: `champion.json`, the run record, the reproducibility record (`schema_version` 1) and
  the `pmlab.plugins` entry-point group. Internal is everything else: any other module path
  (`pmlab.core.costs`, `pmlab.plugins.base`), underscore names, message text.
- **It is enforced, not just stated.** `tests/api/public_api.json` snapshots 192 public names (kind,
  signature, public members, bases, abstract methods), 11 commands with every option, and the key
  tree of each file format. `tests/api/test_public_api.py` fails when the surface differs and says
  which name, option or key was added, removed or changed; `scripts/api_surface.py --write` changes
  the snapshot on purpose. The end-to-end test runs the same script with the Python of a venv that
  holds only the built wheel. Mutation-checked: renaming a public function, dropping a name from `__all__`, changing a public
  default, renaming a CLI option, renaming a `champion.json` key and adding an abstract method to
  `MarketPlugin` each turn the snapshot test red with the name that changed; so do a deprecation
  timeline that allows removal in the same major or in 1.0.0, a decorator that loses the signature,
  the package reaching a `removed_in`, a public name missing from the reference, a deprecation
  missing from the policy table, an internal import path in the docs, a Pages workflow that deploys
  on pull requests, and an unpinned action.
- **A deprecation policy and its helper**: [docs/deprecation.md](docs/deprecation.md). A name is
  deprecated in a minor release and removed in the next *major* at the earliest, never before
  2.0.0. `pmlab._deprecation.deprecated(since=, removed_in=, alternative=)` emits a
  `PmlabDeprecationWarning` (public, a `DeprecationWarning` subclass) that names the caller's line;
  it keeps signature, `__wrapped__` and `async`-ness, adds a notice to the docstring, and refuses a
  timeline that breaks the rule at import time. `warn_deprecated(...)` covers an argument or a key,
  and `cli=True` a command-line option (one plain `warning:` line on stderr, since Typer cannot flag
  an option). `DeprecationWarning` and not `FutureWarning`: Python shows it to the code that
  triggers it (`__main__`, tests) and hides it from libraries in between, which is the audience
  that can act. A test fails when the package reaches the `removed_in` of any deprecation left in
  the code. Turn pmlab's deprecations into errors with `pytest -W error::pmlab.PmlabDeprecationWarning`
  or `warnings.simplefilter("error", pmlab.PmlabDeprecationWarning)` (the interpreter's own `-W`
  cannot name a third-party category: the page says so).
- **The docs site** (<https://artbreguez.github.io/polymarket-lab/>): mkdocs-material, every page in
  `docs/` in the nav, and an API reference generated from the docstrings of the public names only
  (`mkdocstrings`, static analysis, one page per area; a test fails when a public name has no entry).
  `mkdocs build --strict` fails on a broken link, a missing anchor, a page outside the nav or an entry
  that cannot be rendered. The deprecation notice reaches the reference too (a small Griffe extension
  reads the `@deprecated(...)` call). Published to GitHub Pages by `.github/workflows/docs.yml` on
  every push to `main` that touches `docs/` or `src/` (and on demand); pull requests build it
  without deploying; every action is pinned to a commit SHA. New optional extra: `pmlab[docs]`.
  The tutorial notebook is linked rather than rendered: `mkdocs-jupyter` would take the docs venv
  from 123 MB / 35 packages to 198 MB / 82 (a Jupyter kernel stack) to show the code cells of a
  notebook committed without outputs.

### Changed

- **Names that were only reachable through an internal path are public**: `ModelRegistry` and
  `RegistryEntry` (documented in [docs/registry.md](docs/registry.md)) are now in `pmlab.__all__`;
  `MLflowTracker` is `pmlab.tracking.MLflowTracker` (loaded on first use, so the core install still
  never imports MLflow); `PmlabDeprecationWarning` is new. The old module paths keep working.
- **The documentation teaches the stable paths**: every `from pmlab.<module>.<file> import ...` in
  the docs, the README and `examples/` is now `from pmlab import ...`, and a test fails if a code
  example imports a name that is not public.
- **`WeatherTmaxPlugin.build_training_row`**, deprecated since 0.9.2, now goes through the helper: the
  same `DeprecationWarning` behaviour, now a `PmlabDeprecationWarning` with the timeline
  (**removed in 2.0.0**).
- **Docstrings**: ten public callables that had none now have one (`add_lags`, `add_rolling_stats`,
  `encode_cyclical`, `encode_onehot`, `clip_outliers`, `generate_report`, `DiskCache`,
  `AsyncGammaClient`, `AsyncClobClient`, `LiveBrokerError`); Sphinx roles (`:class:`) became
  Markdown backticks, since the reference is Markdown.
- Classifier `Development Status :: 5 - Production/Stable`; `Documentation` and `Changelog` project URLs.

### Deprecated

- `WeatherTmaxPlugin.build_training_row` (since 0.9.2): use `build_training_rows`. Removed in 2.0.0.

## [0.12.1] — 2026-11-08

### Fixed — the leakage check catches a market-level copy of the realised value

Closes #39. Since 0.11.2 a per-bin panel is scored against the per-row outcome
(`outcome_label == winning_label`). A feature that is the same for every bin of a market scores
exactly 0.5 against that by construction, so a column holding the realised value on every bin row
(`feature_realised_tmax`) was a perfect leak the check could not see; 0.11.2 documented it as a
known limit. It is caught now. Documented in [docs/leakage.md](docs/leakage.md).

- **A market-level feature is also scored by its distance to each bin.** On a per-bin panel, a
  numeric feature that takes one value across the bins of every (`market_id`, `decision_date`) is
  scored by `-distance(feature, bin)` against the per-row outcome, with the same fold and the same
  `max_auc`: the distance is 0 inside the bin and the gap to the nearest end outside it. A copy of
  the realised value is inside the winning bin and outside every other, and scores 1.000; the
  feature's score is the larger of that and the raw score. Features that vary across bins, one-row
  panels and an explicit `label_col` are scored exactly as before.
- **The panel has no numeric bin bounds, so they are read from `outcome_label`.** The issue assumed
  the panel carried them; a per-bin row has the label only (`MarketSpec.outcome_bins` holds the
  bounds, and the row dict does not). Reading the label works on panels already built and saved,
  and does not change the row schema, so no fingerprint or stored panel moves; emitting
  `bin_lower` / `bin_upper` from the plugin would have. The reader (private, in
  `pmlab.data.leakage`, so `data/` does not import a plugin) understands `18°C`, `-3°C`,
  `28-30°C`, `<=14°C`, `10°C or below`, `below 10`, `>=22°C`, `38°C or above`, `38+`, `≥38`,
  `at least 38`; a label it cannot read leaves its row out, and a panel of names is left as it was.
- **Distance to the bin's interval, not to its centre.** Evaluated against the issue's
  `-|feature - bin_center|` and a containment test (is the value inside the bin?), on the whole
  sample and on mixed-width layouts. On 1-degree bins distance and centre rank alike
  (`feature_fcst_tmax` 0.725 vs 0.724), but the centre needs a made-up centre for an open-ended bin
  (a copy deep inside a tail is far from the tail's bound: 0.984 on plain 1-degree bins with tails,
  against 1.000) and ranks a narrow neighbour first when a range is wide (0.901 with mixed widths,
  under the 0.97 line, so the copy escapes; swapping the centre in turns the copy tests red);
  containment lets a copy that is a third of a bin off through
  (0.946 vs 0.995 at 0.3 degrees of noise). Open-ended bins (`<=14°C`, `38°C or above`, in six
  wordings) have an infinite bound, so no centre is needed.
- **No false positive on honest data.** The bundled sample's eight forecast features are scored
  0.51 to 0.70 (`feature_fcst_tmax`, the best forecast, 0.698 on the 60-day panel; 0.725 on the
  whole sample) against 0.97, and `check_no_leakage(panel)` and `build_panel(rows,
  check_leakage=True)` both pass on it. Features in another unit (a percentage, a four-digit
  liquidity) score 0.50 to 0.51.
- **How exact a copy must be**: noise of 0.2 degrees on 1-degree bins is still 1.000, 0.3 is 0.995,
  0.5 is 0.966 (under the line), 1.0 is 0.879. The page has the runnable sweep.
- **`LeakageReport.bin_relative`** (new, default `[]`) lists the features scored this way, and the
  error says so: `feature_realised_tmax (score=1.000, by its distance to each bin)`. The `scores`
  entry of such a feature is no longer always 0.5: that is the visible change.
- **Upgrading**: nothing to change. A forecast that is very good for coarse bins can now score
  close to the line (an honest forecast of a market with 8-degree bins scored 0.93 in the tests);
  if a legitimate one is refused, the error names it and `max_auc` is the knob. The page lists
  what the check still cannot see (a copy in another unit, bins that are names, a blend).
- Mutation-checked: skipping the distance score (25 tests red), a centre instead of the interval,
  an open-ended bound read as a point, and ignoring the decision date when deciding that a
  feature is market-level each turn tests red.

## [0.12.0] — 2026-11-08

### Added — `TunedForecaster` can tune for the trade policy you actually trade

Closes #36. `TunedForecaster` scored every trial with the historical backtest rule (YES on the most
probable bin of every market, no threshold) while live trading uses the edge path (per market the
best after-fee edge, YES or NO, taken only if it clears `min_edge`, 0.20 by default). A model tuned
for one rule can be the wrong model for the other. Documented in [docs/tuning.md](docs/tuning.md).

**No default changed: nothing to do on upgrade.** Without the new arguments a search is the one
0.11 ran, trial for trial (a test pins every trial's score to the historical backtest's).

- **The mismatch, concretely.** Two candidate models on a crafted 48-day panel (the page builds it
  and the numbers come from the backtest itself): scored by the historical rule A earns +32.6 and B
  loses 64.0, so A wins the search; scored by `TradePolicy.live(min_edge=0.20)` A never trades and B
  earns +23.9, so B wins; at `min_edge=0.05` A wins again (+25.4 vs -12.7). The threshold alone
  changes which hyper-parameters win. (On the bundled sample the 8-trial tutorial search picks the
  same parameters under every objective, with different scores: 22.65 historical, 18.13 at 0.05,
  7.91 at 0.20. The sample is too well-behaved to show the divergence.)
- **`policy=` and `costs=`.** `TunedForecaster(..., policy=TradePolicy.live(min_edge=0.05),
  costs=CostModel(...))` forwards both to the scoring `rolling_origin_eval`, so a trial is graded
  by the decision rule and the fills `record-trades` will use. `policy=None` (default) is the
  historical rule; `taker_bps` keeps working for a fee alone (it is now `None` by default and still
  resolves to 30 bps; `tuner.taker_bps` still reads 30.0) and is refused together with `costs`.
  A `policy` that is not a `TradePolicy` is a `TypeError`.
- **No lookahead introduced.** Every trial still goes through `rolling_origin_eval`, which fits on
  `decision_date < eval_date` whatever the policy (a spy forecaster asserts it under the live policy).
- **A search that never trades warns.** Under a threshold every trial can score `-inf`; `tune()` now
  raises a `UserWarning` instead of returning the first trial as if it had won.
- **The record says which objective picked the parameters.** `TunedForecaster.tuning_config()` is the
  resolved objective (metric, policy, costs, walk-forward settings, best score), strict JSON.
  `BacktestConfig.resolve(..., tuning=tuner.tuning_config())` stores it as `config["tuning"]` and
  `resolve_seeds(seed, optuna=tuner.random_state)` adds the sampler seed to `seeds`. A run without a
  tuner has no `tuning` key and no `optuna` seed: its record is byte-for-byte what it was. A tuned
  run reproduces like any other (`pmlab reproduce` re-runs the backtest with the tuned parameters,
  it does not search again) and the recorded objective, seed and data are enough to run the search
  again and land on the same parameters (tested).
- **`save`/`load` keep the objective.** A tuner pickled before this release loads as the historical
  objective, which is what produced it.
- **Tutorial notebook** tunes with the live policy and the costs of its backtest, and records the
  objective. Docs: [docs/tuning.md](docs/tuning.md), README, [docs/reproducibility.md](docs/reproducibility.md).
- **Mutation-checked.** Ignoring the passed policy (7 tests red), ignoring the costs (4), dropping the
  `tuning` block from the record (4) and dropping the sampler seed (3) each turn tests red.

## [0.11.2] — 2026-11-08

### Fixed — the leakage check no longer rejects a ranged-market panel

Closes #35. `build_panel(rows, check_leakage=True)` raised `LeakageError` on panels with one row
per outcome bin, which is what `MarketPlugin.build_training_rows` returns. On the bundled sample it
flagged four of the eight forecast features (`feature_fcst_tmax` scored 0.999), features that were
doing their job. The tutorial worked around it by handing the check a binary label of its own.
Documented in [docs/leakage.md](docs/leakage.md).

- **A false positive, not a leak.** `check_no_leakage` scored each feature against the multiclass
  `winning_label`, taking the rarest label as the positive class. On a per-bin panel `winning_label`
  is the same for every bin of a market, so the score asked which *market* resolved to the rarest
  label (on the sample: the one market, 9 rows, that resolved to 38°C) and an accurate forecast of a
  hot day answers that near-perfectly. Against `outcome_label == winning_label`, per row, the
  label `rolling_origin_eval` actually fits (`y_train`), every forecast feature of the sample scores 0.5.
- **The check scores what the model learns.** On a per-bin panel (some market has more than one
  distinct `outcome_label` at one `decision_date`, the grouping the backtest chooses within) the
  target is the per-row outcome. Any other panel (one row per market) is scored against
  `winning_label` exactly as before, and an explicit `label_col` is used as given on any shape
  (`label_col` now defaults to `None`, meaning "choose"; passing `"winning_label"` gets the old
  target back). `LeakageReport` gains `target`, which says what was scored, and the error names it.
- **Real leaks are still caught, on both shapes.** A per-bin feature copied from the outcome, or
  derived from the realised value (a bin's distance to what happened), scores above 0.97 and
  raises; a one-row-per-market feature that encodes the winner is still caught against
  `winning_label`. Mutation-checked: disabling the check, reverting the target, skipping per-bin
  panels and widening the per-bin rule to every shape each turn a test red.
- **What it cannot see** (documented): a feature identical across a market's bins scores 0.5
  against the per-row outcome by construction. It cannot leak on its own, and a label score cannot
  tell an honest forecast from a copy of the realised value when both are constant across bins; it
  becomes a leak once something makes it bin-relative, and that feature is scored.
- **Tutorial notebook**: the `won` workaround is gone. Cell 12 calls
  `build_panel(rows, check_leakage=True)` and cell 14 calls `check_no_leakage(panel)` on the full
  feature set, with no label.
- **Upgrading**: nothing to change. A call that passed `label_col=` keeps scoring that column; one that
  did not now scores a per-bin panel against "did this bin win" and no longer raises on honest features.

## [0.11.1] — 2026-11-08

### Fixed — a registry round trip loses no manifest field

Closes #33. `ModelRegistry.record()` dropped `backtest_expectation`, and `rollback()` dropped
`costs` and `backtest_expectation`, so a champion rolled back from the registry came back
without the promise `pmlab monitor` holds live PnL and calibration against (the check then
reports "unavailable") and without the friction its gate was computed under. Documented in
[docs/registry.md](docs/registry.md).

- **Root cause**: `champion.json` was written key by key in three places (`publish`, `record`,
  `rollback`), and each new manifest field (`costs` in 0.8.4, `backtest_expectation` in 0.9.1,
  `reproducibility` in 0.10.0) had to be added to all three by hand. Two writers were missed.
- **`ChampionManifest.to_dict()`** is now the one serializer (the inverse of `load()`);
  `publish`, `ModelRegistry.record()` and `ModelRegistry.rollback()` all write it. The JSON it
  produces is byte-for-byte what `publish` wrote before. A field added to the manifest reaches
  every copy of it without touching the registry.
- **Tests that cover a field added later**: `tests/modeling/test_registry_roundtrip.py` publishes
  a manifest with every field set, records it, rolls it back and compares it field by field over
  `dataclasses.fields(ChampionManifest)`, and compares the `champion.json` files key by key. A
  guard fails by name if the fixture leaves a field unset, so a new field cannot go unexercised.
  Also pinned: a `promote-champion` champion (CLI) survives the registry, and a second
  record/rollback of a rolled-back champion still loses nothing.
- **Also fixed, found by the end-to-end run**: a registry opened with a relative path
  (`ModelRegistry("registry")`) archived `model_path` relative to the directory it was recorded
  from, so `rollback()` from any other directory raised `FileNotFoundError`, and so did one on a
  registry that had been moved. `ModelRegistry.get()` now points at the artifacts where the
  registry keeps them (next to the version's `champion.json`) instead of trusting the stored
  path. The on-disk format is unchanged.
- **End to end, from the installed wheel**: the new docs page runs as written (backtest under
  non-default costs, `promote-champion`, record, rollback, `pmlab reproduce` on the archived
  version); the e2e then compares the promoted, archived and restored files.

### Compatibility

- Registry entries written by older releases still load and roll back. They never held
  `backtest_expectation`, so a rollback to one has no expectation (`None`, which the watchdog
  reports as unavailable, never as fine). What they did archive, `costs` (since 0.8.4) and
  `reproducibility` (since 0.10.0), a rollback now restores too. Pinned by a test that strips the
  keys the way 0.10.0 and earlier left them.
- `champion.json` files, the registry's `index.json` layout and every public signature are
  unchanged. `ChampionManifest.to_dict()` is new.

### Notes

Found while implementing #16. This is the second time this root cause has bitten
(0.8.4 noted `record()` "rebuilds `champion.json` field by field instead of copying it").
Checked for the same gap: `promote-champion` passes costs, expectation and reproducibility to
`publish`, which now serializes through `to_dict()`; no other code writes a `champion.json`.

## [0.11.0] — 2026-11-08

### Added — an end-to-end tutorial notebook on a bundled sample dataset

Closes #15. One notebook, [`notebooks/tutorial.ipynb`](notebooks/tutorial.ipynb), walks
discover → panel → tune → backtest → gate → paper → report with the public API only, offline.
Documented in [docs/tutorial.md](docs/tutorial.md).

- **`pmlab.datasets`**: `load_sample()` returns a synthetic weather-temperature sample (750
  markets: 5 cities x 150 days x 9 bins) shipped inside the wheel. 39 KB gzipped, verified
  against a SHA-256 on every load (`SampleDataError` on a missing, corrupt or out-of-step
  file). Generated by `scripts/make_sample_dataset.py`; flagged `meta["synthetic"]`.
- **Offline discover, without weakening the real plugin**: `sample.plugin(day)` is the real
  `WeatherTmaxPlugin` with its three injected clients (`SampleGammaClient`,
  `SampleForecastClient`, `SampleTruthClient`) reading the bundled file. Time is replayed: a
  plugin built for a day stamps prices at 07:30 UTC that morning and only sees observations
  final by then, so the leakage guards and the settlement lag behave as they do live. A
  plugin with no Gamma client still raises.
- **Executed in CI so it cannot rot**: new `notebook` pytest marker and CI job
  (`pytest -m notebook`) run it top to bottom in a fresh kernel from outside the repo with the
  network refused, and assert a 120 s budget (it takes ~12 s). Executor: `nbclient` +
  `nbformat` + `ipykernel` (dev dependencies; what `nbmake` and `nbconvert --execute` drive,
  without their other dependencies). The marker is deselected from the default run.
- **Committed with outputs stripped**; a default-suite test fails otherwise, and also if the
  notebook imports a private module, touches an underscore attribute or names a missing export.
- **The e2e runs it from the installed wheel** in a fresh venv, and runs the quick start of
  docs/tutorial.md as written.
- `pmlab` now also exports `WeatherTmaxPlugin`, `rolling_origin_eval`, `RollingOriginResult`
  and the `datasets` module at the top level (the notebook imports everything from `pmlab`).

## [0.10.0] — 2026-11-08

### Added — reproducibility capture and `pmlab reproduce`

Closes #16. "A result you can't re-run isn't a result" (design principle #5) is now something
the tool checks. Documented in [docs/reproducibility.md](docs/reproducibility.md), whose
examples are executed as written by the end-to-end test.

- **`ReproducibilityRecord`** (`pmlab.reproducibility`) — what a run records about itself:
  every RNG seed used (`python_random`, `numpy`, `lightgbm`), the versions of pmlab, Python,
  numpy, pandas, lightgbm, scikit-learn and optuna (`null` if absent) plus OS and CPU
  architecture, the *resolved* config (every default spelled out: model params, stride, stake,
  the full cost model and trade policy), a fingerprint of the input data (SHA-256 of the panel
  content, row count, columns, source path), and the metrics the run produced.
- **Stored where runs already live**, composing with the existing systems:
  `ActiveRun.log_reproducibility()` and `RunRecord.reproducibility` (tracker run file, and an
  MLflow artifact `reproducibility.json`); `ChampionManifest.reproducibility` /
  `publish(..., reproducibility=)` (`"reproducibility"` in `champion.json`);
  `ModelRegistry.record()` archives it and `rollback()` restores it; `pmlab backtest --output`
  writes it. **Backward compatible**: every file written before 0.10.0 loads
  (`reproducibility` is `None` / `{}`), and `reproduce` says plainly that such a run has no record.
- **`pmlab reproduce <run>`** — `<run>` is a path to any file holding a record (metrics file,
  tracker run file, `champion.json`, a directory with one), a tracker run id or unique prefix
  (`--runs-dir`), or a registry version id (`--registry-dir`). It re-executes the recorded
  backtest through the same code path as `pmlab backtest` and compares the metrics. Exit 0
  reproduced, 1 ran but differs, 2 could not run.
  - **Different data** (content hash) is refused, naming how it differs (hash, rows, columns),
    unless `--allow-data-mismatch`; a run on other data is never reported as reproduced.
  - **Different library versions** are warned about with a diff table; only the metrics decide
    the verdict, and a miss with drift points at the drift.
  - **Tolerance**: counts exactly; floats within `--rtol 1e-9 --atol 1e-12` (bit-for-bit on the
    same machine and versions; `--rtol 0 --atol 0` is strict).
  - A record with no numeric metrics cannot be verified and is an error, never "reproduced".
  - Missing run, no record, unknown run kind, missing or moved panel (`--panel`), malformed or
    newer-schema records: each a one-line error, never a traceback.
- **`pmlab backtest`** gains `--seed` (default 0), `--lgbm-param KEY=VALUE` (repeatable),
  `--runs-dir` / `--run-name` (also log a tracker run). `pmlab promote-champion` records the
  record from the `--expectation` file, or `--reproducibility FILE`.
- `run_backtest` / `reproduce` seed python `random` and numpy for the run only (`seeded()`), then
  restore the caller's generator state. A `champion.json` whose record this pmlab cannot read
  (a newer schema, damage) loads with a warning and no record rather than failing the live loop.
- Library API: `BacktestConfig.resolve`, `resolve_seeds`, `run_backtest`, `capture_backtest`,
  `fingerprint_panel`, `capture_environment`, `environment_drift`, `load_record`, `reproduce`
  (all also exported from `pmlab`). `LGBMForecaster.resolved_params()`.

### Changed — randomness that was not seeded now is

- `SklearnForecaster` presets `random_forest` and `gradient_boosting` default to
  `random_state=0`. They were unseeded: they drew from the global numpy state, so two processes
  gave two different models. Pass `random_state=` to choose another seed. This changes the
  output of code that relied on the old (arbitrary) behaviour.
- `pmlab backtest` seeds python `random`, numpy and LightGBM from `--seed` (default 0). For the
  default LightGBM parameters (no bagging or feature sub-sampling) the seed has no effect, so
  existing backtests give the same numbers. `LGBMForecaster` itself is unchanged: unseeded it
  uses LightGBM's own fixed internal seeds (deterministic, but not recordable).
- **Breaking, small:** `pmlab backtest --model` now rejects a name other than `lgbm_baseline`
  (exit 1). It used to ignore the option and run LightGBM whatever it said, which a
  reproducibility record would then have recorded wrongly. A script that passed another name was
  already getting LightGBM; drop the flag or pass `lgbm_baseline`.
- The backtest summary computation moved from the CLI body to `pmlab.reproducibility`
  (`backtest_metrics`); the numbers are unchanged.

## [0.9.4] — 2026-11-08

### Added — a 90% coverage gate and a CI performance budget for the hot paths

Closes #18. No change to the library's behaviour; the guarantees are in the build.

- **Coverage gate raised from 70% to 90%.** `fail_under = 90` in `pyproject.toml` is now the
  single source of truth: the `--cov-fail-under=70` override in the CI pytest step (which would
  have shadowed it) is gone, so `pytest --cov` exits non-zero below 90% and fails the `test` job.
  Measured total at this release is 95.3%; 90 leaves a 5-point margin for the modules that are
  only partly reachable in CI (the optional MLflow extra, the live Gamma client) while still
  catching a module that ships untested. No test was added to reach it.
- **Performance budget for `rolling_origin_eval` and `build_panel`**, asserted in CI by a new
  `perf` job (`uv run pytest -m perf`, deselected from the default run like `e2e`). Three seeded,
  in-process workloads (72k-row backtest, 28.8k-row backtest under the live best-edge policy,
  120k-row `build_panel` with price-timing stamps) are timed as the best of 5 runs after a warm-up
  and read in **calibration units** (seconds divided by a fixed loop of the hot paths' own pandas work, measured in the same run), so
  the budget holds across machines; a round over budget is retried before it fails. Budgets are
  about 2.3x the reference machine's value, so a 3x slowdown fails and 2x does not, on the CI
  runner as well as on the reference VM. A scaling check (2x the rows must
  cost < 3x) catches super-linear growth independently of the machine. Everything is in
  `examples/perf_budget.py` (runnable by anyone: `python examples/perf_budget.py [--check]`),
  documented in `docs/performance.md`, with the doc's table pinned to the script by a test.
- `tests/e2e` also runs the workloads and `docs/performance.md` from the wheel installed in a
  fresh venv.

## [0.9.3] — 2026-11-08

### Added — out-of-fold target encoding, interactions and missing-value indicators

Closes #19. Three more transforms, copy-on-write like the existing five (the input frame
is never mutated), typed and tested.

- **`TargetEncoder` / `FittedTargetEncoder`** (`pmlab.features.target_encoding`) — smoothed
  (m-estimate) target encoding of one categorical column with a leakage contract:
  - `enc.fit_transform(train)` encodes **training rows out-of-fold**: a row's encoding is
    built from the targets of other folds only (the prior is out-of-fold too), so no row
    sees its own target.
  - `cv="time"` (default, needs `time_col`) is an expanding window over distinct
    timestamps: a row is encoded from strictly **earlier** timestamps only, so there is no
    lookahead. The earliest fold has no history and is NaN. `cv="group"` (seeded group
    K-fold, e.g. all bins of one market in one fold) and `cv="kfold"` (seeded shuffled
    K-fold) exist but must be asked for explicitly; there is no silent fallback to KFold.
  - `enc.fit(train)` returns a frozen, picklable `FittedTargetEncoder` with the full-train
    statistics; `.transform(live)` applies them to new rows (no target column needed).
    Inference never takes the OOF path.
  - Unseen category -> prior; NaN/None category is its own category (prior if never seen
    while fitting); NaN targets are ignored; deterministic given `seed`. `enc.assign_folds(df)`
    exposes the fold ids.
  - `output_col=` names the encoded column (default `f"{col}{suffix}"`). `rolling_origin_eval`
    only feeds `feature_*` columns to the model, and renaming after the fact on the training
    frame but not on the live frame is an easy way to ship mismatched columns; found by the
    end-to-end run below.
- **`add_missing_indicators(df, cols, suffix="_isna")`** — `int8` 0/1 flags.
- **`add_interactions(df, pairs, op="mul")`** — `mul` / `div` / `add` / `sub` with
  deterministic names (`a_x_b`, `a_div_b`, `a_plus_b`, `a_minus_b`); division by zero is NaN,
  never `inf`.

All are exported from `pmlab.features` and `pmlab`. See `docs/features.md` for the leakage
contract and a correct-usage example.

### Added — end-to-end test of the public API on the installed wheel

`examples/e2e_transforms.py` is a standalone user script (public imports only) that builds a
multi-market synthetic panel with NaNs, adds missing indicators and interactions, target-encodes
out-of-fold on the training period and with the fitted mapping on a later live period (unseen and
NaN categories), trains `LGBMForecaster`, runs `rolling_origin_eval`, scores the live rows, and
asserts the user-visible guarantees (inputs unchanged, no `inf`, live rows never use the OOF path,
unseen category gets the prior, finite metrics). `tests/e2e/` builds the wheel, installs it into a
fresh venv, and runs that script, every code block of `docs/features.md` (unmodified) and
`pmlab backtest` on the resulting panel from outside the repo. Opt-in: `pytest -m e2e` (deselected
by default); CI runs it in its own `e2e` job. Running every doc block unmodified found a
syntactically invalid import line in `docs/features.md` (`import add_lags, ...`), now fixed.

### Fixed — `add_rolling_stats(group_by=...)` put values on the wrong rows

`add_rolling_stats` assigned `groupby().rolling().<stat>().values` positionally. That result
is ordered by group, so whenever the frame was **not already sorted by the group key**
(e.g. rows interleaved `a, b, a, b, ...`), every group's rolling values landed on other
rows — silently, with plausible-looking numbers. It now uses `groupby().transform`, which keeps
each value on its own row (and is also correct with duplicate index labels). Frames already sorted
ascending by the group key produce identical output to before (checked); any feature built with
`group_by` on an unsorted frame was wrong and should be regenerated. `add_lags` was not
affected. Regression tests use interleaved groups.

## [0.9.2] — 2026-10-21

### Fixed — the weather panel showed the backtest only the bin that won (#28)

`WeatherTmaxPlugin.build_training_row` emitted **one row per market**, labelled with the
winning bin (`outcome_label == winning_label`) and priced at the market's *highest* bin price
from discovery. The backtest chooses, per market, which bin to buy and whether to buy YES or
NO; a panel holding only the winner lets it make exactly one kind of decision correctly. Every
YES it takes wins, every NO it takes loses, whatever the model says, and the model is trained
on labels that are all 1. The PnL such a panel reports belongs to the panel, not the model.
Since 0.9.1 made `--policy live` the default this decided real YES/NO trades, and
`--policy top-bin-yes` had always been affected.

**Measured, on a synthetic world with fair prices** (60 days x 8 markets x 5 bins, every bin
priced at its true probability, so any policy's expected PnL is exactly 0; frictionless, the
CLI's `LGBMForecaster`, stride 2, seeds 7/11/23). A "z" is the PnL over the standard deviation
it must have if the prices are fair:

| panel | policy | trades | hit rate | PnL | z |
|---|---|---|---|---|---|
| one winning row per market (before) | `top-bin-yes` | 224 | 100% | +377 to +393 | +19.4 to +19.8 |
| one winning row per market (before) | `live` (min edge 0.05) | 224 | 0% (all NO) | -224 | -18.2 to -18.4 |
| one row per bin (this release) | `top-bin-yes` | 232 | 17-19% | -52 to +0.3 | -1.35 to +0.01 |
| one row per bin (this release) | `live` (min edge 0.05) | 226-232 | 47-52% | -8.5 to +10.7 | -0.31 to +0.40 |

The sign under `live` follows the model, the size does not: LightGBM fitted on one class
reports a probability of about 1e-15 for every label, so the policy buys NO on the winner and
loses the stake each time; a model that predicts the training base rate (p = 1 on this panel)
buys YES on the winner and wins each time. Both are the same panel artifact. The regression
test (`tests/plugins/weather_tmax/test_panel_bias.py`) asserts both, and that the per-bin panel
stays inside 4 standard deviations of zero on all three seeds.

- **`MarketPlugin.build_training_rows(spec, horizon, **kwargs) -> list[dict]`** — every row of
  one (market, decision), one per outcome bin. The default wraps `build_training_row`
  (`[row]`, or `[]` for `None`), so a plugin written for the singular form needs no change.
  `build_training_row` stays abstract.
- **`WeatherTmaxPlugin.build_training_rows`** — one row per bin, each with that bin's own
  `outcome_label` and `market_price`, plus `decision_date`, `decision_ts`, `price_ts` and
  `price_source` so a leakage audit can tell where a price came from and when. `decision_time`
  is required. Bins that cannot be priced get no row and are **counted** under a reason in an
  `exclusions` mapping the caller passes (logged at WARNING when it passes none):
  `price_missing`, `price_invalid`, `price_degenerate` (exactly 0 or 1, which is what a settled
  market shows), `price_untimed`, `price_after_cutoff`, `price_stale` (only with
  `max_price_age`), and per market `decision_after_close`, `market_close_unknown`,
  `market_unresolved`, `market_truth_unbinned`, `market_no_usable_price`.
- **No price is invented.** `market_price` no longer defaults to 0.5, and neither does
  `_build_spec`: an absent or unreadable Gamma price is `None` in `outcome_schema`, a short
  price list leaves the remaining bins unpriced, and one bad entry no longer discards the others.
- **Point-in-time.** A price is used only if it was observed strictly before the decision (the
  rule `FeatureSnapshotStore.as_of` applies to features). `discover_markets` now stamps
  `price_observed_at` (one reading of an injectable `clock`) and `price_source = "discovery"`
  on every spec. **The discovery-time Gamma price is the only per-bin price this plugin has**:
  the CLOB client reads the current midpoint and keeps no history, and the snapshot store holds
  no per-bin price. A market discovered after the decision has the wrong prices and is excluded
  as `price_after_cutoff`, not used; one decided on after it resolved gives no rows at all
  (`decision_after_close`, below). A better source can be supplied per bin in `outcome_schema`
  (`observed_at`, `source`).
- **`check_price_timing(df)` / `PriceTimingReport`** in `pmlab.data`: raises `LeakageError` if a
  row's `price_ts` is at or after its `decision_ts`, or has none. `build_panel` runs it always
  (not only with `check_leakage=True`); a panel without the two columns is not checked.
- **A decision after the market ended is no training row** (found in review). A price observed
  before the decision was accepted even when the decision came after the market had resolved:
  a market resolving 2026-05-01, discovered 2026-05-03, decided on 2026-05-04T00:00Z gave 5
  accepted rows, every one a post-resolution price. `build_training_rows` now gives no rows for
  a `decision_time` at or after the market's end (its `close_time`, or the end of its
  `target_date` when that is an ISO date, whichever is first), counted once per market as
  `decision_after_close`, and none for a market whose end cannot be read (`market_close_unknown`).
  A decision on the target day itself, before the close, is unaffected (`morning_of`). Rows carry
  `market_close_ts`, and `check_price_timing(close_col="market_close_ts")` refuses a row whose
  `decision_ts` is at or after it (rows with an unreadable close are counted in `close_unknown`,
  not refused), so the same rule holds for any plugin's panel through `build_panel`.
- **`check_price_timing` no longer fails open on half a schema** (found in review). A panel with
  `decision_ts` but no `price_ts` (or the reverse) was accepted with `n_checked = 0`, as if it
  were an untimed panel. It now raises `LeakageError` naming the column that is there and the
  one that is not, even with `raise_on_leak=False` (there is nothing to report on). A panel with
  **neither** column remains the documented unchecked case. The weather plugin's rows carry the
  pair or neither: a row built without a `decision_time` (the deprecated singular form) no
  longer carries a lone `price_ts`.
- **`build_panel` warns once when prices are untimed** (found in review). `market_price` drives
  every backtest trade, and a panel with neither `price_ts` nor `decision_ts` has nothing that
  says it was known by the decision. `build_panel` still builds it, and now logs a single
  WARNING per call (logger `pmlab.data.panel_builder`; `logging`, never stdout) naming the risk.
  Untimed today: plugins written before the check, `sports_f1`, and `build_panel_from_snapshots`
  (its `market_price` is a snapshot feature taken before the decision by `as_of`, but its panel
  carries no timing columns); each is now documented as such. `pmlab monitor` never builds a
  panel, so its quiet-on-green output is untouched (pinned by a test that fails if `build_panel`
  is referenced from `pmlab.cli` or `pmlab.monitoring`, and by a healthy run with it disabled).
- **`max_price_age` bounds how stale a discovery price may be** (found in review). The discovery
  price is reused for any later decision, so a week-old quote was a training row like any other.
  `build_training_rows(..., max_price_age=timedelta(hours=6))` (or a positive number of seconds)
  leaves out, and counts as `price_stale`, any bin whose price was observed longer before the
  decision than that; a price exactly that old is kept. Default `None` is the previous behaviour.
  It is a plugin argument only: no command builds a weather panel, so there is no CLI option.
  The deprecated singular form takes it only with a `decision_time` and raises without one.
- **`build_panel` takes lists of rows** (the output of `build_training_rows`, per market) as
  well as single rows and `None`.
- **`rolling_origin_eval` ignores the panel's index.** `top_prob` read the chosen rows back by
  index label, so a panel stacked market by market (`pd.concat` of one frame per market, each
  indexed 0..K-1) matched the same label in every market: 448 trades where a clean index gives 56 (a 40-day,
  8-market, 5-bin panel, stride 5). It was latent with one row per market; assembling a per-bin
  panel market by market makes it likely.

### Changed

- `WeatherTmaxPlugin.build_training_row` is **deprecated** (`DeprecationWarning`). It returns
  the winning bin's row at that bin's own price (it used to carry the market maximum), and a
  panel made of it is still biased. It no longer substitutes 0.5 and returns `None` when the
  winning bin has no usable price. Its `decision_time` is optional, as before.
- `decision_date` is the **UTC** date of the decision, now stated on `build_training_rows`
  and in the plugin guide (`2026-05-10T23:30:00-05:00` is `2026-05-11T04:30:00Z`, so its
  `decision_date` is `2026-05-11`). The deprecated `build_training_row` called without a
  `decision_time` yields a row `build_panel` rejects (no `decision_date`); this is documented
  on it and pinned by a test.
- `metadata["market_price"]` on a weather spec stays, as the highest *known* bin price, and is
  `None` when none is known. It is for display; it is never a row's price.
- Feature columns no longer overwrite a row's `market_id`, `outcome_label`, `market_price` and
  the other structural keys (the structural keys win).
- A panel has K rows per market, but the backtest makes one trade per (market, decision date),
  so `num_trades`, hit rate, the gate's per-segment `n`, the expectation and every watchdog
  count are per trade and unchanged in meaning (pinned in
  `tests/backtest/test_rolling_origin_per_bin.py`). Two settings are in rows: `--min-train-rows`
  (20 rows is 4 markets at K = 5; scale it by K) and any row-wise split of your own, where the
  bins of a market belong on one side.

### Upgrading — champions promoted from a weather panel built before 0.9.2

**Their `backtest_expectation` is optimistic, and so is the gate that let them through.** A
champion exists only because its gate said GO, so its backtest PnL was non-negative; on a
panel of winners that number does not measure edge, and `champion.json` records it as the
promise the live book will be held to.

- **Check whether a panel is affected:** `(panel["outcome_label"] == panel["winning_label"]).all()`
  is `True` for a winner-only panel.
- **What the watchdog will do:** `pmlab monitor` compares live PnL per stake with the recorded
  per-stake mean. Against an inflated reference expect a PnL **shortfall** alert (exit 20) once
  there are enough settled trades (30 by default; `insufficient`, exit 30, until then). In this
  release's end-to-end check, a champion that promised +1.778 per stake, held to a live book
  with no edge (+0.021 per stake over 150 trades), alerted: a 99% shortfall at z = 15.5, the
  sequential test having crossed its threshold at trade 13. That alert is the reference being
  wrong, not the market having changed. The recorded calibration reference is no better: it
  comes from the same trades, in which the label always occurred (a Brier of 1.000 in that
  run), so calibration reads `ok` whatever the live book does.
- **What to do:** rebuild the panel with `build_training_rows`, re-run `pmlab backtest --output
  metrics.json` (scale `--min-train-rows` by the bins per market), re-run the gate, and
  re-promote with `pmlab promote-champion ... --expectation metrics.json`. Expect some of those
  to come back `NO_GO`: that is the correct outcome, not a regression.
- **A `GO` on the rebuilt panel is not evidence of edge.** The gate asks for PnL >= 0 over at
  least 40 trades, which a model with no edge passes about half the time. In the end-to-end
  check (a world where no policy can have an edge) `top-bin-yes` came back `NO_GO` and `live`
  `GO` at +4.40 over 41 trades, z = +0.69; the champion recorded +0.107 per stake with a spread
  of 1.0, and `pmlab monitor` then reported OK against a fair live book while stating it could
  only detect a shortfall of 453% or more at n = 150. Read the PnL against its spread
  (`std_pnl_per_stake` in the `--output` JSON and in `champion.json`), not the decision.
- No number printed in the README or `docs/` came from a weather panel; the only backtest output
  there (`docs/watchdog.md`) is a user-written synthetic world.

### Known limits, not changed here

- `sports_f1` has the same defect and was left alone: its row has no `outcome_label` and its
  `market_price` is always the 0.5 default (its spec records no prices), so it cannot pass
  `build_panel` today.
- The per-bin rows share their market-level forecast features, so a model cannot tell the bins
  apart unless the forecast client or a subclass adds bin-aware features. Separately, the
  plugin's stub features (`forecast_*`, `lead_hours`) are not `feature_*`, the only columns the
  backtest passes to the model; a forecast client has to name its keys `feature_*`.
- A training window with a single class makes LightGBM report ~1e-15 for the label, and
  `rolling_origin_eval` trades on it. A per-bin panel does not trigger it (each market adds a
  winner and losers), but it is still there for any panel whose training window has one class,
  and it is the second half of why the old panel lost under `live`.
- `CalibratedForecaster` and the conformal wrapper split by random row, so the bins of one
  market can fall on both sides; drift reports count reference *rows* as samples.

## [0.9.1] — 2026-10-21

### Added — Watchdog CLI

Closes #14. The gate, live calibration and drift each compute numbers and stop;
this is the layer that decides. It stays silent when healthy, so it can run from
cron: `pmlab monitor ... || notify`.

- **`pmlab monitor`** — drift, live calibration and live PnL against the champion,
  each `ok` / `warn` / `alert` / `insufficient` with explicit reasons; the overall
  status is the worst. A healthy run prints nothing (stdout or stderr) and exits 0;
  `--verbose` prints the full report; `--json PATH|-` always writes it.
- **Exit codes**: `0` healthy, `10` warn, `20` alert, `30` insufficient/unavailable,
  `64` bad input, `70` internal error. They avoid 1 (an uncaught Python exception)
  and 2 (a Click usage error) so a crash or a typo'd flag can never read as a
  verdict, and nothing that goes wrong can exit 0.
- **`health_report(champion=, trades=, drift=, as_of=)`** — the pure core: no I/O, no
  clock, byte-identical output for identical inputs. `HealthReport` / `CheckResult` /
  `WatchdogConfig` are frozen, strict-JSON-safe, with `to_dict()` / `from_dict()`.
- **`BacktestExpectation`** — mean, standard deviation and count of per-stake PnL
  (and of the per-trade squared forecast error), recorded in `champion.json` as
  `backtest_expectation`. The gate keeps only totals, so a live run previously had
  nothing to be surprised *against*. `ChampionManifest.publish(expectation=)`,
  `pmlab promote-champion --expectation`, and `pmlab backtest --output` now writes the
  block it consumes.
- **`--costs PATH` on `backtest`, `promote-champion` and `record-trades`.** The CLI
  could not reach `CostModel`: `backtest` ran at the default, `promote-champion`
  published `costs: null` and `record-trades` built a `PaperBroker` without costs, so
  a CLI user never got a champion whose cost basis the watchdog could check.
  `backtest --output` records the costs it ran under, `promote-champion --expectation`
  publishes the champion with them (a contradicting `--costs` is refused), and
  `record-trades` fills under `champion.json`'s costs by default (a contradicting
  `--costs` is refused). That last is a behaviour change for a champion that already
  records costs: its paper trades now carry the slippage and fee the backtest charged.
- `CalibrationTracker.squared_errors()`; `rolling_origin_eval` trades gain
  `label_occurred` (its `outcome` is a profit flag; calibration needs the label).
- `docs/watchdog.md`. Every Python block was re-run after the final edit and diffed
  against its printed output; the CLI transcripts come from the built wheel installed
  into a fresh venv.

### Added — backtest/live parity, per-side and per-segment health, sequential detection

- **The backtest can simulate what `record-trades` trades.** `TradePolicy` and one edge
  computation, `quote_edges`, shared by the backtest, `EdgeSignal.from_quote` and
  `record-trades`: per market the (bin, side) with the best after-fee edge, YES or NO, if it
  clears `--min-edge` (default 0.20); a NO trade is a buy of the NO token at `1 - yes_price`
  under the same cost model and arithmetic as `PaperBroker` and `SettlementEngine`.
  `rolling_origin_eval(..., policy=TradePolicy.live())`; `pmlab backtest --policy
  live|top-bin-yes --min-edge`. A test replays a backtest's signals through
  `PaperBroker` and `SettlementEngine` and gets the same per-trade PnL.
- **The expectation is recorded per side and per segment.** `BacktestExpectation.by_direction`
  (YES, NO), `by_segment` (each with its own sides) and `policy`; a side or segment the
  backtest traded fewer than 30 times is not given a promise it cannot back (`null` for a
  segment). `champion.json` carries all of it. A champion published before this has the
  single record: its NO trades are left out of the PnL comparison and counted (never read as
  green), exactly as before.
- **Calibration and PnL judge the book, each side and each segment.** Each live trade is
  held to the backtest's record for the cell it fell in, so a book that trades another mix of
  sides or segments than the backtest did is not mistaken for a decline (and a gate-rejected
  segment cannot dilute the promise). Sides and segments are each a family corrected with
  Holm-Bonferroni; a thin or unpromised scope is listed as *not assessed* and does not make a
  conclusive check `insufficient`. The failing segment or side is named in the output and in the
  JSON (`metrics.directions`, `metrics.segments`). Calibration scores the model's probability
  that the label occurs whichever side was bought (pinned by a test).
- **A sequential test** (one-sided CUSUM on each trade's standardized shortfall, in settlement
  order, `h = 30`, threshold calibrated by simulation) makes a PnL alert possible when the
  pooled mean dilutes a fade. It alerts while the statistic is above the threshold, so a healthy
  book does not carry a stale alert. No single trade may add more than `cusum_increment_cap`
  (5) to the statistic, so one total loss in a near-certain cell (a spread of almost nothing)
  cannot raise an alert by itself. `PnLThresholds.sequential`, `cusum_*`, `mde_power`.
- **Every PnL result says what it could have detected**: the minimum detectable shortfall at 80%
  power, in `--verbose`, the JSON and the default output whenever a check is not ok.
- **`pmlab resettle --trades PATH`** recomputes `realized_pnl` of settled trades that an older
  release booked wrongly (see Fixed), from the fields they recorded: no network, unsettled
  trades untouched, a timestamped backup first, `--dry-run` for the diff, idempotent. The
  backup and the new log are synced to disk before the rename, and the directory after it.
- `scripts/watchdog_power.py` regenerates every power and false-alarm table in
  `docs/watchdog.md`; `tests/monitoring/test_power.py` holds its fast copies to `health_report`.

### Changed

- **`pmlab backtest` now defaults to `--policy live`** (YES and NO at a best edge of at
  least 0.20, as `record-trades` trades), so its trade count, PnL and gate differ from a 0.9.0 run
  on the same panel. **`--policy top-bin-yes` reproduces the historical backtest exactly**, and
  the library function keeps that as its default (`policy=None`) so published numbers still
  reproduce. `record-trades` takes its threshold from the champion's policy (a `--min-edge`
  that differs is called out), skips signals on a side the policy does not trade, and records the
  best signal per market.
- **A champion promoted on an aggregate gate now records every segment.** Such a gate
  allows `{"all"}`, and `PaperBroker` compared segment names with it literally, so
  `record-trades` recorded nothing; `"all"` now means every segment in `ChampionManifest`,
  `PaperBroker` and the watchdog alike. An empty allowlist still means none. `"all"` is the
  wildcard only on its own: beside other names it is a segment called "all", so a per-segment
  gate that graded such a segment does not widen to segments it never graded.
- **`scripts/pmlab-workspace` and `pmlab -a artifacts -w <name>` now read and write the same
  trade log.** The wrapper's artifacts directory is already `artifacts/workspaces/<name>`, and
  the settings appended `workspaces/<name>` again (and looked up workspace `ops_daily`, because
  the wrapper exported only `PMLAB_WORKSPACE_NAME`). No file moves, so no migration is needed.
  The champion is the one difference between the entry styles: it is
  `<artifacts dir>/public_models/champion.json`, so pass `--champion` if you use both.

### Fixed

- **NO-side realized PnL was wrong, in both directions.** `SettlementEngine` settled a
  `direction="no"` trade as a short YES but handed the formula the *NO* price, which is
  the wrong price for it. At a YES quote of 0.30 and a stake of 1 a NO bet whose label
  occurred (it lost everything) booked -0.4286 instead of -1.0000, and one that won
  booked +1.0000 instead of +0.4286: biased toward profit exactly when the NO side won.
  A NO trade now settles as a buy of the NO token at its own price (`fill_price` when
  recorded, else `1 - gamma_price`), the same per-stake arithmetic the backtest applies
  to a buy, and `Position` gains `side="buy_no"` (an unknown `side` now raises instead
  of falling through to the short-YES formula). `outcome` is unchanged: it still says
  whether the *label* occurred, which is what calibration reads.
  **Trades already settled keep their old number**, because settlement is idempotent
  and never recomputes a settled trade. Every trade settled from now on carries
  `"settlement_version": 1`; a NO trade settled by 0.9.0 or earlier has no such key.
  **`pmlab resettle --trades PATH`** recomputes them (try `--dry-run` first). To list the
  affected ones by hand:
  `jq '[.trades[] | select(.direction == "no" and .outcome != null and .settlement_version == null)]' forward_paper_trades.json`
- **Settlement ignored `fill_price`.** `PaperBroker` fills at the costed price and
  records it, but `SettlementEngine` recomputed the entry from `gamma_price`, so with
  any slippage in the cost model live `realized_pnl` read better than the backtest's
  by exactly that friction (live 1.081 vs backtest 0.974 on the same trade at a quote
  of 0.10). It now settles at `fill_price` and writes `settled_basis`
  (`"fill_price"`, or `"quote"` when none was recorded). A trade with no `fill_price`
  still enters at the quote, exactly as before. **A trade settled by 0.9.0 or earlier
  keeps the number it was given even if it carries a `fill_price`: that number is the
  frictionless one**, and it has no `settled_basis`. `pmlab monitor` leaves such trades
  out of the PnL check (`n_excluded_legacy_settlement`) unless their `fill_price`
  equals the quote, rather than judging them as cost-matched. A live-vs-backtest
  comparison would otherwise have been biased toward green.

### Notes on semantics

- **A shortfall is a finding only if it is material *and* distinguishable from
  noise**: an effect-size threshold, plus a one-sided two-sample Welch z using the
  spread of both the live and the backtest sample. One-sided, so outperformance never
  pages. `z_warn = 2`, `z_alert = 3` are deliberately stricter than the textbook
  values: the watchdog looks every run.
- **The checks have limited power, and the docs say so with numbers** (all from
  `scripts/watchdog_power.py`, 400 simulated champions per cell, a 400-trade backtest).
  The fixed PnL test false-alarms in about 3% of runs or fewer and notices a *completely
  vanished* edge in 25% of runs at 100 live trades, 42% at 400 and 51% at 800; a *halved* edge
  in 9%, 15% and 17%. The sequential test adds 3 points at 400 trades and 16 at 800 for a vanished
  edge (45%, 67%), its median first crossing is trade 415 for a vanished edge and 904 for a
  halved one, and a healthy book has its alert on in at most 2% of runs at any moment (though
  26% of healthy champions cross the threshold at some point within 3000 trades). The Brier
  check is not the earlier signal: at 100 trades it fires in 0.5% of runs where the PnL check fires
  in 17% (a vanished 6-point edge), and a test of the mean forecast error has the PnL
  check's power, no more. Ten segments judged on their own false-alarm in 25% of runs, with Holm
  in 4%; the whole PnL check (book, sides, up to ten segments, sequential) warns or alerts for
  nothing in 3-6% of runs and alerts in at most 1%. A quiet watchdog is weak evidence the
  edge is intact, which is why every PnL result reports the shortfall it could have seen.
- **`insufficient` is not healthy.** It outranks `ok`, loses to any measured problem,
  and has its own exit code. `--insufficient warn` counts it as a warn; there is
  deliberately no `ignore`.
- **Only evidence that belongs to this champion counts**: settled trades in its allowed
  segments (from `champion.json`), recorded at or after its publication, and settled
  by `--as-of` (`SettlementEngine` now writes `settled_at`; a trade from an older
  release has none and is judged by its `target_date`, counted in `n_no_settled_at`
  and noted in the reasons). Every exclusion is counted. PnL also excludes trades on a side
  the champion has no expectation for (a champion published before the per-side record
  only bought YES, as far as its record goes), NO trades booked by an older release until
  `pmlab resettle` recomputes them, trades settled at the quote by an older release
  although they paid a different fill, and trades filled under a different cost model
  than `champion.json` records.
- **Old champions still load.** One published before v0.9.1 has no expectation, so
  calibration and PnL report `insufficient` ("no backtest expectation recorded"); a
  present but unusable record (non-finite number, negative spread, fewer than two
  trades) is an error (exit 64), not a silent `None` and never a silent pass.
- **A book with no trade in the champion's segments says so.** The checks stay
  `insufficient` (exit 30), but the reason now leads with `0 of N settled trades are in
  the champion's allowed segments [...]` instead of burying it in an exclusion tally,
  and `allowed_segments` is in the metrics. A champion whose gate allows no segment at
  all (an empty allowlist means *none*, as in `PaperBroker`) is malformed and exits 64.
- **A zero or negative backtest mean keeps the PnL check `insufficient` for good**, and
  the reason now says so (`... will stay insufficient on every run until the champion is
  re-published on a backtest that earned money`) instead of leaving a cron job to wonder
  why exit 30 never clears. Drift and calibration are still judged.
- **A `model_prob` that is not a probability is bad data, not a Brier alert.** NaN,
  infinite or outside [0, 1] is left out of calibration, counted in
  `n_invalid_forecast`, and named in the reasons; scored, it would page as a
  miscalibrated model or turn every statistic into NaN.
- **A cutoff needs a UTC offset.** `--as-of` (and `--window-start`, and the `as_of`
  argument of `health_report`) refuse a naive timestamp such as `2026-03-01` with exit
  64 and a message asking for `Z` or an explicit offset, instead of reading it as UTC:
  on a cron box in another zone that is a different instant and the report would still
  look right. Timestamps inside the trade log stay lenient.
- **Typo'd configuration is rejected**, not ignored: an unknown threshold key would
  otherwise leave the default in force while the operator believed otherwise.
- A trade log is put in a canonical order before any float sum. Shuffling the file
  changed the report in the last bit (tied `target_date`s were scored in input order);
  found by a test, fixed.

### Not included

- **Acting on an alert.** It reports and exits; pausing or retraining is the caller's.
- **A CUSUM per segment or side.** The sequential test runs on the book; per-segment CUSUMs
  would multiply the false alarms. The sides and segments are judged by the fixed test.
- **Locking the trade log.** `pmlab resettle` (like `record-trades`) assumes nothing else is
  writing to the log while it runs.

## [0.9.0] — 2026-10-21

### Added — Drift monitoring

Closes #12. The gate and live calibration both need outcomes, which arrive days or
weeks after the forecast. A feature that has moved is visible today.

- **`drift_report(reference, current)`** (`pmlab.monitoring`) — a pure function: no
  I/O, no randomness, same inputs give an identical report. PSI and
  KL(current ‖ reference) per feature, plus the same machinery on the model's
  predicted probabilities (`reference_predictions=` / `current_predictions=`),
  reported separately as `report.prediction`.
- **`DriftReport` / `FeatureDrift` / `DriftConfig` / `DriftThresholds`** — frozen,
  `to_dict()`/`from_dict()`, strict-JSON-safe. The report carries flat numeric
  summaries (`n_alert`, `max_psi`, …) because trackers log only top-level numerics,
  and `ExperimentTracker.log_metrics(report)` works as-is.
- **Threshold-classified**: each feature is `ok` / `warn` / `alert` / `insufficient`,
  with `drivers` naming the signal(s) responsible. Thresholds are explicit
  configuration written into every report, not buried constants — this is what the
  watchdog (#14) will consume.
- **`drift_report_from_store(store, family, reference, window_start=, cutoff=)`** —
  builds the live window through `FeatureSnapshotStore.as_of`, so nothing captured
  at or after `cutoff` can leak in. `window_start` is required: without a lower
  bound "current" would include the training-era markets and understate drift.
- `docs/drift-monitoring.md`. Every number in it comes from running its snippets:
  all 15 blocks were re-run after the review fixes below and diffed against their
  printed output.

### Notes on semantics

- **Bins come from the reference only** (its quantiles), reused for the live
  sample. Binning each sample on its *own* quantiles scores a 3-sd wholesale shift
  at PSI 0.00; on reference edges it is 9.21.
- **Out-of-range mass has its own bins.** Everything strictly below the reference
  minimum and strictly above its maximum gets a bin of its own, symmetrically. The
  first draft left the end bins open-ended, so a constant or heavily tied reference
  (whose quantile edges collapse) merged everything below it into its own bin:
  a constant reference scored PSI 0.0 against a sample that was half below it, and
  `[0.7] * 500` vs `[0.3] * 500` predictions scored 0 and read `ok`. They now score
  4.60 and 18.42 and alert. A sample against itself is still exactly 0.0 for
  constant, tied, discrete and continuous references.
- **`unseen_rate` has no thresholds, and needs none.** Because out-of-range mass
  now sits in bins the reference left empty, it moves PSI and KL by itself: at the
  default `epsilon`, 1% beyond the reference range is `ok`, 1.5% `warn`, 3% `alert`.
  It stays on the report because a coarse `epsilon` makes that mass cheap (at 1e-2,
  5% out of range is still `ok`); the docs say so and a test pins it.
- **Threshold comparisons carry 1e-9 of slack** for float round-off, for PSI, KL and
  missing rate alike. A missing rate going 10% → 30% changes by
  `0.19999999999999998` and used to read `warn` against the documented `>= 0.20`
  alert line.
- **Empty bins are floored at `epsilon` (1e-4) and renormalised**, never dropped, so
  a value range the reference never saw scores large but finite — never `inf`/`NaN`.
  Bins empty on both sides are omitted.
- **KL is directional: `kl` = KL(current ‖ reference).** PSI is the sum of both
  directions, so swapping the arguments leaves PSI unchanged and changes `kl`
  (0.3681 vs 0.5108 in the docs example). Asserted against a hand computation.
- **Missing is drift.** NaN/None/±inf are excluded from the distribution but counted,
  and the change in missing rate is classified on its own: 0% → 40% missing is an
  alert even when the remaining values look identical.
- **Unseen categories are drift, not dropped** — pooled into one zero-reference bin
  and listed. Numeric vs categorical is decided from the *reference* column, so a
  feed that dies into all-`None` is still a numeric feature that went missing. Two
  consequences are documented and tested rather than changed: `1`, `1.0` and `True`
  are one category (a `Counter` merges values that hash alike), and one stray string
  in a numeric reference column makes it categorical (`DriftConfig(numeric=...)`
  then fails loudly, naming the feature).
- **Predictions are validated.** Values outside `[0, 1]` (including `±inf`) and
  non-numbers raise `ValueError` naming the side and how many offend; `NaN` is
  missing, as for features. Reference and current arrays may differ in length.
  `drift_report_from_store` requires `predict_fn` to return exactly one number per
  row for both the reference panel and the live window.
- **The live window excludes the training markets.** `drift_report_from_store` drops
  markets whose `market_id` is in the reference panel (`exclude_reference_markets=True`
  by default) and reports the count as `DriftReport.n_current_excluded_overlap`
  (in `to_dict` / `from_dict`). A reference without a `market_id` column raises
  rather than silently skipping the check. The docs also warn that the store gives
  each market its latest snapshot before the cutoff while the panel holds the one at
  decision time, so part of the measured drift can be markets at a different age.
- **Quiet failures now raise:** an unknown `family` (nothing in the store for it at
  all), `window_start` after `cutoff`, and an empty feature list. An
  empty-but-valid window is still `insufficient`.
- **`insufficient` never reads as `ok`.** Below `min_samples` the statistics are
  `None`. Precedence is alert > warn > insufficient > ok; an empty live window is
  `insufficient`, not clean.
- **`min_samples` defaults to 300, not 100**, because PSI of an *unchanged* feature
  has a noise floor of roughly `(n_bins − 1) / n`. Simulated over 500 windows, an
  unchanged feature crossed the 0.10 warn line in 223 at 100 values and in 3 at 300
  (204 and 3 before the outer bins, which add a little noise: a reference's own tail
  is now an empty bin). The first draft used 100; running the numbers showed it
  would false-warn nearly half the windows.
- **Defaults are conventions, labelled as such.** PSI 0.10 / 0.25 is the standard
  credit-scoring convention. KL 0.05 / 0.125 is *derived* (half the PSI bands) and
  the missing-rate bands 0.05 / 0.20 are judgment; neither has an industry standard.
- Classification, not action: deciding what to do about an alert is the watchdog's
  job (#14).

### Verification

786 passed, 1 skipped; mypy `--strict` clean on 66 files; ruff check + format
clean; 100% line and branch coverage on `drift.py` (88.61% overall, gate 70%).

38 mutations tested one at a time (reference vs current bin edges, own-quantile
binning, epsilon removed, smoothing not renormalised, KL direction swapped, lookahead
admitted, `window_start` ignored, missing rate ignored or taken signed, unseen
categories dropped, min-sample checks and boundaries, status precedence, band
edges, feature ordering, defaults, and more); none survives. Three **survived** the
first pass of 29 and each exposed a weak test: the smoothing convention was
unpinned (a skipped renormalisation passed everything), numeric `unseen_rate` was
never asserted, and the unseen-category ranking test used data whose frequency
order equalled alphabetical order. Each now has a test that fails against the
mutant. Separately, an incorrect claim in my own docstring (that taking edges
from the current sample scores ~0 — it is the *own-quantile* binning that does)
was caught by running the counterfactual and corrected.

A second round for the review fixes above: 19 further mutations (each outer bin
removed and both together, the tolerance removed or kept on one line only, the
`[0, 1]` check, NaN and `inf` handling, the one-value-per-row check, overlap
exclusion removed / uncounted / unserialised / not text-normalised / skipped on a
missing `market_id`, and the unknown-family, window-order and empty-feature
checks). Two survived the first pass: a quantile-edge filter that turned out to be
an equivalent mutant (values above the maximum are forced into the upper bin, so the
bin it guarded is always empty on both sides — the line was dead and was removed),
and the store-side `astype(str)` in the overlap check (no test used integer market
ids in the store). That one has a test now. All 18 live mutants are killed.

## [0.8.5] — 2026-09-30

### Added — Live calibration tracking

Closes #13. The gate says a model was calibrated on history; nothing said whether
it still is on money at risk.

- **`CalibrationTracker`** (`pmlab.monitoring`) — rolling Brier, Murphy
  decomposition and reliability curves over realized trades. `overall()`,
  `rolling(window=N)`, `by_segment()`, `from_path()`. Reuses the existing
  `brier_decomposition` / `reliability_data` diagnostics rather than
  reimplementing them.
- **`CalibrationWindow`** — frozen, `to_dict()`/`from_dict()`, JSON-safe (no numpy
  scalars leak into the report).
- **`PaperBroker` now records `model_prob`** in the trade log. Without the forecast
  there is nothing to score an outcome against, so calibration on realized trades
  was impossible before this.

### Notes on semantics

- **`outcome` means "the label occurred", not "this trade profited."**
  `SettlementEngine` computes it without reference to `direction`, so a
  `direction="no"` trade whose label occurs is `"won"` with a negative
  `realized_pnl` (measured: `outcome="won"`, `realized_pnl=-6.6967`). Calibration
  wants the former. `_label_occurred` is named for what it means.
- **Calibration and PnL are independent** — a calibrated model can lose money and
  a miscalibrated one can win for a while. `docs/live-calibration.md` spells this
  out.
- **Unscoreable ≠ zero.** Open trades and pre-0.8.5 rows without `model_prob` are
  excluded and counted in `n_skipped`; treating a missing forecast as `0.0` would
  manufacture miscalibration. `overall()` returns `None` on an empty history.
- **Windows order by `target_date`, not `recorded_at`** — a forecast is scored when
  its truth lands, not when the row was written. Matters for backfilled logs.
- No alerting thresholds here; that is the watchdog's job (#14). Thresholds are a
  policy choice and burying one in a metrics module would hide it.

### Verification

569 passed, mypy `--strict` clean on 65 files, ruff check + format clean, 100%
coverage on `calibration_tracker.py` and `paper_broker.py`.

Five mutations tested. One **survived** initially — sorting by `recorded_at`
instead of `target_date` passed every test, because the fixtures moved both dates
together. Added a backfill case where they disagree; it now fails.

## [0.8.4] — 2026-09-30

### Added — One cost model across backtest and live

v0.8.3 gave the backtest a `CostModel`; the live path still filled at the quote
with a fee-only adjustment. Two cost definitions for the same trade means live-vs-
backtest PnL differs for two reasons at once — the market, and our own arithmetic —
which would make the planned v0.9.0 watchdog unable to tell them apart.

- **`PaperBroker(costs=CostModel(...))`** — fills at the costed price using the same
  arithmetic as `rolling_origin_eval`, sizes off that fill rather than the quote, and
  records a **`fill_price`** field next to `gamma_price`. Works on both sides of a
  market: a `no` direction fills against `1 - gamma_price` with slippage still
  against the buyer.
- **`ChampionManifest.publish(costs=...)`** — persists the cost model in
  `champion.json`, so the friction a gate was computed under travels with the model
  and a live run can read it back instead of having it retyped.
- **`ModelRegistry.record()`** carries `costs` into the archived version, so a
  rollback restores a champion whose cost assumptions are known rather than lost.

### Compatibility

- `PaperBroker` without `costs=` behaves exactly as before (30bps fee, fill at the
  quote); `taker_bps=` still works on its own and `PaperBroker.taker_bps` now reads
  through to `costs.taker_bps` so code inspecting it keeps working. Passing both
  raises `ValueError`.
- `ChampionManifest.load()` uses `.get("costs")`, so manifests written before this
  release still load with `costs=None` — meaning "assumptions unknown", not "no
  costs".

### Notes

The deliverable is an equivalence, not a feature: for the same quote and stake, the
backtest and the paper broker produce the **same fill price**, asserted across a grid
of 4 prices × 3 stakes. `LiveBroker` is deliberately untouched — it takes price and
size from its caller and does no fill accounting of its own.

Found by mutation testing: `ModelRegistry.record()` rebuilds `champion.json` field by
field instead of copying it, so the new field was silently dropped on archive until a
test pinned it.

## [0.8.3] — 2026-09-30

### Added — Backtest realism (`costs=`)

- **`CostModel`** (`pmlab.core.costs`) — execution costs applied to a backtest
  fill: taker fee, price-proportional slippage (bps), fixed slippage in
  probability units, and a depth penalty that grows linearly with stake above a
  reference size. Frozen dataclass, validated on construction, `to_dict()` for
  the experiment tracker. `CostModel.frictionless()` gives a zero-cost upper
  bound.

- **`rolling_origin_eval(..., costs=CostModel(...))`** — fills now happen at the
  costed price instead of the quote, and `edge` is computed against that fill.
  This matters beyond PnL: `compute_metrics` averages `edge` into `avg_edge`,
  which the holdout gate reads, so a book whose edge exists only at the mid can
  no longer pass the gate. Trade logs gain a **`fill_price`** column.

### Compatibility

- **The default is unchanged behaviour.** Calling `rolling_origin_eval` without
  `costs=` applies a 30bps taker fee and fills at the quote, exactly as before —
  every result published prior to this release stays reproducible. Verified by a
  test asserting frame equality between the legacy call and `costs=CostModel()`.
- Passing both `taker_bps=` and `costs=` raises `ValueError` rather than silently
  charging the fee twice.

### Notes

The depth term is a documented linear approximation, not a book walk: pmlab does
not carry L2 depth in the training panel, so the model is explicit about
approximating rather than pretending to measure. Coverage: 100% on `core/costs.py`,
95% on `backtest/rolling_origin.py`.

## [0.8.2] — 2026-09-13

*This entry was written in 1.0.1: the release was published without one.*

### Added — Stability report
- `stability_report(trades)` bootstraps the trade log (resampling with replacement) to put a
  confidence interval around total PnL, average PnL per trade and hit rate, plus
  `prob_positive_pnl`, so a GO can be judged on a distribution and not a point estimate. A pure,
  seeded function with a `to_dict()` for the tracker; it does not touch `rolling_origin_eval`.
  It lives in `pmlab.backtest.stability` and is not part of the 1.0 public API.

---

## [0.8.1] — 2026-09-13

*This entry was written in 1.0.1: the release was published without one.*

### Added — Model registry
- `ModelRegistry` versions every promoted champion (model, calibrator and manifest archived under
  `registry/<version_id>/`) with `record`, `list`, `get`, `diff` and `rollback`, replacing the
  single overwrite-in-place `champion.json` as the history. `record()` re-asserts the hard gate, so
  a NO_GO manifest can never enter the registry, and a rollback preserves history: no version is
  deleted. The existing `ChampionManifest.publish` flow is unchanged.

### Fixed
- Two champions promoted in the same instant no longer share a `version_id`: a numeric suffix is
  appended until it is unique, so neither archive is overwritten.

---

## [0.8.0] — 2026-09-06

*This entry was written in 1.0.1: the release was published without one.*

### Added — Experiment tracking
- An `ExperimentTracker` protocol with a zero-dependency `LocalJSONTracker` (one append-only JSON
  record per run) and an optional `MLflowTracker` behind the new `track` extra
  (`pip install "pmlab[track]"`). A backtest can log its params, metrics and gate decision, and
  runs can be listed and loaded.

---

## [0.7.1] — 2026-08-30

### Added — Closed training loop

- **`build_panel_from_snapshots(store, family, resolutions)`** — assembles a
  no-lookahead training panel by joining point-in-time snapshots
  (`FeatureSnapshotStore`) with realized market truth. For each resolved market
  it uses the latest snapshot captured *strictly before* that market's
  `decision_date`, so no feature can originate after the decision — no lookahead
  by construction. This closes the loop the data layer was built for:
  capture open markets over time → markets resolve → build the panel → train →
  backtest → gate.

Validated end-to-end against live Polymarket data (376 political markets, a
two-day snapshot-capture simulation of a cron): the `as_of` join correctly
selected the pre-decision snapshot, the leakage guard passed on honest features,
and the panel flowed through backtest → gate. 100% coverage on the new code.

## [0.7.0] — 2026-08-30

### Added — Data & feature layer (`pmlab.data`)

Turns the previously empty `pmlab.data` package into the reproducibility
backbone. Motivated directly by dogfooding the library against the live
Polymarket API, which exposed that the framework could not honor its own
no-lookahead principle during ingestion, and silently accepted a leaking
feature (AUC 1.000, no warning).

- **`FeatureSnapshotStore`** — append-only, point-in-time store of per-market
  feature snapshots (one parquet file per family), keyed and de-duplicated on
  `(market_id, captured_at)`. `as_of(family, cutoff)` returns the latest snapshot
  per market captured strictly before the cutoff, so a retrain uses the features
  a decision could actually have seen — never the post-resolution state. This is
  the missing piece that lets ingestion be no-lookahead.
- **`check_no_leakage` / `LeakageError` / `LeakageReport`** — scores each
  `feature_*` column by how well it alone separates the label (`max(auc, 1-auc)`)
  and flags any above `max_auc` (default 0.97). Catches the exact trap from the
  dogfood (a feature derived from the outcome) instead of failing silently.
- **`build_panel`** — typed panel builder that assembles plugin rows into a
  validated training panel: enforces the required schema (`market_id`,
  `decision_date` as YYYY-MM-DD, `outcome_label`, `winning_label`,
  `market_price`, ≥1 `feature_*`), coerces features to float, drops `None` rows,
  and optionally runs the leakage guard. Validates `market_price` too (rejects
  non-numeric and all-NaN — it drives PnL/edge). Turns a KeyError-deep-in-the-
  backtest into a clear error at build time.

### Changed
- **`publish.yml` now creates a GitHub Release automatically** on `v*` tag, right
  after the PyPI upload (auto-generated notes, wheel + sdist attached, marked
  latest). Tagging is now the single action that ships a version end to end.

100% test coverage on the new package.

## [0.6.1] — 2026-08-30

*Merged, not released on its own: no tag, no PyPI release. It shipped inside 0.7.0.*

### Fixed — Lifecycle composition (found by dogfooding as a real user)
- **`compute_metrics` now accepts the `rolling_origin_eval` trade log directly.**
  It derives the win/loss `outcome` from the sign of `realized_pnl` when the
  column is absent (and treats `edge` as optional). Previously
  `compute_metrics(result.trades)` raised `KeyError: 'outcome'` — the two central
  public functions did not compose, and even the CLI worked around it by hand.
- **`HoldoutGateResult.evaluate` gains an aggregate mode.** `required_segments`
  is now optional; when omitted (or an empty list) the whole trade log is graded
  as a single `"all"` segment, so a backtest can be turned into a GO/NO_GO
  decision without first splitting into segments or hand-building the dataclass.
  An empty `required_segments=[]` now falls back to this aggregate grade instead
  of auto-passing (`all([]) == True` would have returned GO on zero segments —
  a champion hard-gate breach).

### Changed
- **`rolling_origin_eval` trade log now includes `outcome` and `segment`
  columns.** `outcome` is derived from PnL sign; `segment` is passed through from
  the panel when present, else defaults to `"all"`. This lets the trade log flow
  straight into `compute_metrics` and `HoldoutGateResult.evaluate` — closing the
  `backtest → metrics → gate → champion` loop with no manual glue.

---

## [0.6.0] — 2026-08-30

### Added — Modeling
- **Multiclass calibration.** `CalibratedForecaster` now supports >2 classes via
  one-vs-rest (new `MulticlassCalibrator`), so F1 / multi-outcome political
  markets are calibrated too. Binary behavior is unchanged.
- **Multiclass diagnostics.** `multiclass_brier` (with a climatology skill score,
  `MulticlassBrier` dataclass) and `reliability_data_multiclass` (per-class
  reliability curves) complement the existing binary Brier decomposition.
- **`TunedForecaster`** — Optuna hyperparameter search that scores every trial
  through `rolling_origin_eval`, so model selection is **walk-forward and
  lookahead-free** (never a random shuffle). Optimizes `total_pnl`, `mean_pnl`,
  or `hit_rate`. Optuna is an optional dependency: `pip install pmlab[tune]`.

### Added — Backtest
- **Leakage-aware CV.** `purged_kfold` (contiguous test blocks with an embargo
  band, López de Prado style) and `embargoed_split` (expanding-origin walk-forward
  with an embargo gap) in `pmlab.backtest.cv`.

### Changed
- `CalibratedForecaster` raises on <2 classes (was: raised on >2). Multiclass is
  now first-class.

### Infrastructure
- New optional extra `tune` (Optuna). 376 tests, high coverage on new modules.

---

## [0.5.0] — 2026-08-30

*Merged, not released on its own: no tag, no PyPI release. It shipped inside 0.6.0.*

### Added — Modeling
- `EnsembleForecaster` — weighted-average blend of any N `MarketForecaster` members
  behind the same interface (drop-in for backtest/champion/brokers). Validates
  weights (non-negative, positive sum), renormalizes rows, pickles members.
- `ConformalForecaster` — split-conformal classification with a distribution-free
  marginal coverage guarantee. `predict_set()` returns per-row prediction sets
  containing the true label with probability ≥ 1 − alpha; `predict_proba()` still
  passes through so it stays a drop-in forecaster. Non-empty sets guaranteed.
- `CalibratedForecaster` — wraps any binary `MarketForecaster` and calibrates its
  positive-class probability on a held-out split (`isotonic` or `sigmoid`).
- `SigmoidCalibrator` — Platt scaling (1-D logistic) calibrator; more
  sample-efficient than isotonic on small calibration sets.

### Infrastructure
- Untracked committed `.pyc` / `__pycache__` files (already in `.gitignore`).
- 333 tests, 85% coverage.

---

## [0.4.0] — 2026-05-13

### Changed — LiveBroker
- Replaced custom HMAC-SHA256 auth with `py-clob-client` (`ClobClient`) — proper L1 ECDSA + L2 API creds, matching Polymarket CLOB requirements
- Constructor now requires `private_key` (L1, `0x`-prefixed hex) in addition to `api_key / api_secret / api_passphrase` (L2)
- `place_order` → uses `create_order + post_order` via `ClobClient`
- `cancel_order / cancel_all_orders` → `client.cancel / client.cancel_all`
- `get_open_orders` → `client.get_orders`; `get_balance` → `client.get_balance_allowance`
- `preflight()` health check added

### Changed — WeatherTmaxPlugin
- `discover_markets`: paginates Gamma API with real tmax regex filter; no fake `temperature` tag
- `fetch_features`: canonical keys matching pmtmax training schema (`forecast_temperature_2m_max/mean/min`, dew_point, humidity, wind, cloud, lead_hours)
- `build_training_row`: single-row output with all `feature_*` keys
- `_build_spec`: fully parses city/date from question regex, token_ids from `clobTokenIds` JSON, outcome_schema with prices, unit C/F

### Added — GammaClient
- `TmaxMarketInfo` dataclass: `market_id`, `city`, `target_date`, `unit`, `token_ids`, `outcome_labels/prices`, `end_date`, `active`
- `GammaClient.discover_tmax_markets()` — paginate all active markets, filter with pmtmax-identical logic, return `list[TmaxMarketInfo]`
- `AsyncGammaClient.discover_tmax_markets()` — async variant

### Infrastructure
- PyPI publish via GitHub Actions OIDC Trusted Publisher — no secrets stored in repo, auto-publishes on `v*` tags
- 301 tests, 83% coverage

---

## [0.3.0] — 2026-05-11

### Added
- `SklearnForecaster` — scikit-learn wrapper implementing `MarketForecaster` ABC (LogisticRegression, RandomForest, etc.)
- `TypedCache[T]` — generic typed wrapper around `DiskCache` with PEP 695 compatibility
- Plugin auto-discovery via `PluginRegistry.discover()` — scans entry points for `pmlab.plugins`
- CLI `report` command — generates HTML report from paper trade DB without re-running backtest
- CI coverage gate raised to 70%

### Fixed
- `TypedCache` PEP 695 syntax compatibility with Python 3.12
- 66 ruff violations resolved across all modules (`ruff format --check` added to CI)

---

## [0.2.0] — 2026-05-11

### Added
- `PmlabSettings` — Pydantic settings class: `PMLAB_ARTIFACTS_DIR`, `PMLAB_LOG_LEVEL`, `PMLAB_DRY_RUN`, CLOB/Gamma API base URLs; `from_env()` classmethod
- Full CLI (`pmlab` entry point): `version`, `status`, `scan-markets`, `record-trades`, `settle-trades`, `backtest`, `promote-champion`
- `WorkspaceContext` — multi-workspace path isolation via `PMLAB_ARTIFACTS_DIR`
- `scripts/pmlab-workspace` — bash wrapper to scope CLI to a named workspace
- Retry + structured logging — all API clients use exponential backoff; `logging.getLogger("pmlab")` hierarchy
- Gamma/CLOB pagination — `GammaClient` auto-paginates `next_cursor`; `AsyncClobClient` respects `asyncio.Semaphore`
- mypy strict clean — `py.typed` marker, all public APIs fully typed

---

## [0.1.0] — 2026-05-11

### Added — Core
- `MarketSpec` + `OutcomeBin` — generic market descriptor with bin resolution logic
- `Position` + `settle_position` — binary outcome PnL accounting
- `compute_edge` — after-cost probability edge calculation
- `estimate_fee` — USDC taker fee estimate
- `flat_stake_size` — share count for a flat-stake bet
- `kelly_fraction` — fractional Kelly bankroll fraction (binary outcome formula)
- `kelly_stake_size` — Kelly-sized USDC stake with `max_exposure` cap

### Added — Features
- `add_lags` — lag features with optional `group_by` for panel data
- `add_rolling_stats` — rolling mean/std/min/max/median with optional `group_by`
- `encode_cyclical` — sin/cos encoding for periodic features (hour, DOY, month)
- `encode_onehot` — one-hot encoding with `drop_first` and dtype control
- `clip_outliers` — IQR and z-score outlier clipping (returns copy, no mutation)

### Added — Markets
- `GammaClient` / `fetch_gamma_markets` — Polymarket Gamma API client with optional `DiskCache`
- `ClobClient` / `fetch_token_prices` — CLOB midpoint price fetcher
- `AsyncGammaClient` — async Gamma API client (`httpx.AsyncClient`) with keyword filter
- `AsyncClobClient` — async CLOB client with `asyncio.Semaphore` concurrency control
- `DiskCache` — TTL disk cache (MD5-keyed JSON files, `get/set/delete/clear/purge_expired`)

### Added — Modeling
- `MarketForecaster` ABC — protocol: `fit`, `predict_proba`, `save`, `load`
- `LGBMForecaster` — LightGBM binary/multiclass implementation
- `IsotonicCalibrator` — isotonic regression probability calibration
- `ChampionManifest` — hard-gate champion publish (raises `ValueError` on NO_GO)
- `brier_decomposition` — Murphy (1973) decomposition: reliability, resolution, uncertainty, skill score
- `reliability_data` — reliability diagram data (bin centers, mean predicted prob, fraction positive)
- `BrierDecomposition` — dataclass for decomposition results

### Added — Backtest
- `rolling_origin_eval` — walk-forward backtest with strict no-lookahead guarantee
- `BacktestMetrics` + `compute_metrics` — PnL, hit rate, avg edge statistics
- `HoldoutGateResult` / `SegmentGateResult` — per-segment GO/NO_GO gate

### Added — Execution
- `EdgeSignal` — typed scan-edge output dataclass
- `PaperBroker` — record paper trades with segment gate, staleness guard, dedup
- `SettlementEngine` — settle open trades via `plugin.fetch_truth + is_truth_final`
- `LiveBroker` — real CLOB order execution: `place_order`, `cancel_order`, `cancel_all_orders`, `get_open_orders`, `get_balance`; HMAC-SHA256 L1 auth; `dry_run=True` mode
- `OrderReceipt` — dataclass returned by `LiveBroker.place_order`
- `LiveBrokerError` — exception for API failures

### Added — Reports
- `generate_report` — self-contained dark-themed HTML report: summary cards, equity curve SVG, per-segment breakdown, full trade log; zero JS/CSS dependencies

### Added — Plugins
- `MarketPlugin` ABC — 4 required methods: `discover_markets`, `fetch_features`, `fetch_truth`, `build_training_row`
- `PluginRegistry` — register and look up plugins by `family` name
- `WeatherTmaxPlugin` — reference implementation for temperature markets
- `SportsF1Plugin` — categorical outcome plugin for F1 race markets

### Added — Workspace & CLI
- `WorkspaceContext` — multi-workspace path isolation via `PMLAB_ARTIFACTS_DIR`
- `scripts/pmlab-workspace` — wrapper to scope CLI commands to a workspace
- CLI commands: `version`, `status`, `scan-markets`, `record-trades`, `settle-trades`, `backtest`, `promote-champion`

### Added — Infrastructure
- GitHub Actions CI: ruff lint + pytest (70% coverage gate) + wheel build
- `py.typed` marker — PEP 561 compliant typed library
- `AGENTS.md` — shared contract for AI coding agents
- `docs/plugin-authoring.md` — complete plugin authoring guide
- `docs/live-trading.md` — LiveBroker setup and paper-to-live transition guide
- `docs/features.md` — feature engineering transforms reference
- `docs/calibration.md` — Brier decomposition and calibration diagnostics guide
- `docs/reports.md` — HTML report generation guide
- `docs/async-clients.md` — async API clients and DiskCache guide
- `CONTRIBUTING.md` — contribution guide with TDD contract
- 237 tests, 95.7% coverage

[Unreleased]: https://github.com/ArtBreguez/polymarket-lab/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/ArtBreguez/polymarket-lab/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/ArtBreguez/polymarket-lab/compare/v0.12.1...v1.0.0
[0.12.1]: https://github.com/ArtBreguez/polymarket-lab/compare/v0.12.0...v0.12.1
[0.12.0]: https://github.com/ArtBreguez/polymarket-lab/compare/v0.11.2...v0.12.0
[0.11.2]: https://github.com/ArtBreguez/polymarket-lab/compare/v0.11.1...v0.11.2
[0.11.1]: https://github.com/ArtBreguez/polymarket-lab/compare/v0.11.0...v0.11.1
[0.11.0]: https://github.com/ArtBreguez/polymarket-lab/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/ArtBreguez/polymarket-lab/compare/v0.9.4...v0.10.0
[0.9.4]: https://github.com/ArtBreguez/polymarket-lab/compare/v0.9.3...v0.9.4
[0.9.3]: https://github.com/ArtBreguez/polymarket-lab/compare/v0.9.2...v0.9.3
[0.9.2]: https://github.com/ArtBreguez/polymarket-lab/compare/v0.9.1...v0.9.2
[0.9.1]: https://github.com/ArtBreguez/polymarket-lab/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/ArtBreguez/polymarket-lab/compare/v0.8.5...v0.9.0
[0.8.5]: https://github.com/ArtBreguez/polymarket-lab/compare/v0.8.4...v0.8.5
[0.8.4]: https://github.com/ArtBreguez/polymarket-lab/compare/v0.8.3...v0.8.4
[0.8.3]: https://github.com/ArtBreguez/polymarket-lab/compare/v0.8.2...v0.8.3
[0.8.2]: https://github.com/ArtBreguez/polymarket-lab/compare/v0.8.1...v0.8.2
[0.8.1]: https://github.com/ArtBreguez/polymarket-lab/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/ArtBreguez/polymarket-lab/compare/v0.7.1...v0.8.0
[0.7.1]: https://github.com/ArtBreguez/polymarket-lab/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/ArtBreguez/polymarket-lab/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/ArtBreguez/polymarket-lab/compare/v0.4.0...v0.6.0
[0.4.0]: https://github.com/ArtBreguez/polymarket-lab/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/ArtBreguez/polymarket-lab/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ArtBreguez/polymarket-lab/releases/tag/v0.2.0
