# Public API and stability

From **1.0.0**, pmlab follows [semantic versioning](https://semver.org). This page says exactly
what that covers, what it does not, and how the promise is enforced rather than just stated.

> **The promise.** Everything listed under [What is public](#what-is-public) keeps working, with
> the same names, signatures, options and file keys, for every release of the 1.x series. It is
> removed or changed incompatibly only in a major release (2.0.0), and only after it has been
> [deprecated](deprecation.md) with a warning in an earlier release.

| Release | May contain |
|---------|-------------|
| **Patch** (1.0.**x**) | Bug fixes. Nothing public is added, removed or changed. |
| **Minor** (1.**x**.0) | Everything a patch may, plus: new public names; new optional parameters (with a default, after the existing ones, or keyword-only); new commands and options; new keys in the files below; new deprecations. |
| **Major** (**x**.0.0) | Removal of what was deprecated, and any other incompatible change, each listed in the changelog. |

A bug fix that changes a result is still a fix: if a function returned a wrong number, it returns
the right one in the next patch. The changelog says so.

## What is public

### Python names

Import from these modules, and only from these:

| Module | What it holds |
|--------|---------------|
| `pmlab` | The core: markets, edge, costs, sizing, backtest, gate, forecasters, champion and registry, execution, brokers, plugins, workspaces, settings. Every name in `pmlab.__all__`. |
| `pmlab.data` | The point-in-time store, the panel builder and the leakage guards. |
| `pmlab.datasets` | The bundled synthetic sample (`load_sample`) and the offline clients that feed the real weather plugin. |
| `pmlab.features` | Feature transforms and the target encoder. |
| `pmlab.monitoring` | Drift, live calibration and the watchdog. |
| `pmlab.reports` | The HTML report. |
| `pmlab.reproducibility` | Capture a run and re-execute it. |
| `pmlab.tracking` | Experiment tracking, including the optional `MLflowTracker`. |

A name is public when it is in the `__all__` of one of these modules. The
[API reference](reference/index.md) documents exactly those names and nothing else.

For a public class or function, the contract is:

* **Functions:** the parameter names, their kinds (positional, keyword-only, `*args`, `**kwargs`),
  their defaults, and the return type's meaning. New parameters come with a default.
* **Classes:** the constructor signature, and every attribute and method whose name does not start
  with an underscore. For a dataclass, that includes the field names and their order.
* **Abstract base classes** (`MarketPlugin`, `MarketForecaster`): the set of abstract methods.
  Adding an abstract method breaks every subclass, so it is a major-release change; adding a
  concrete method is not. A method is never silently turned from concrete to abstract.
* **Exceptions:** the class, its base classes, and the situations that raise it.
* A parameter whose name starts with an underscore is not part of any signature.

### The command line

The `pmlab` command, its sub-commands and their options. Names and flags are frozen; so are the
defaults shown, and the environment variables an option reads. New commands and options are
additions. The full list as of 1.0.0:

| Command | Options |
|---|---|
| `backtest` | `--plugin`, `-p`; `--panel`; `--model`, `-m`; `--stride`; `--min-train-rows`; `--flat-stake`; `--costs`; `--policy`; `--min-edge`; `--seed`; `--lgbm-param`; `--runs-dir`; `--run-name`; `--output`, `-o` |
| `monitor` | `--champion`; `--trades`, `-t`; `--artifacts-dir`, `-a`; `--workspace`, `-w`; `--reference`; `--snapshots`; `--family`; `--window-days`; `--window-start`; `--feature`; `--no-prediction`; `--as-of`; `--config`; `--min-trades`; `--insufficient`; `--skip`; `--json`; `--verbose`, `-v` |
| `promote-champion` | `model_path` (argument); `--gate-path`; `--plugin`, `-p`; `--output-dir`; `--name`; `--expectation`; `--costs`; `--reproducibility` |
| `record-trades` | `--plugin`, `-p`; `--workspace`, `-w`; `--artifacts-dir`, `-a`; `--min-edge`; `--flat-stake`; `--costs` |
| `report` | `--trades`, `-t`; `--output`, `-o`; `--title`; `--brier` |
| `reproduce` | `run` (argument); `--runs-dir`; `--registry-dir`; `--panel`; `--allow-data-mismatch`; `--rtol`; `--atol` |
| `resettle` | `--trades`, `-t`; `--dry-run`; `--show` |
| `scan-markets` | `--plugin`, `-p`; `--tag`; `--keyword`; `--limit` |
| `settle-trades` | `--plugin`, `-p`; `--workspace`, `-w`; `--artifacts-dir`, `-a` |
| `status` | `--artifacts-dir`, `-a`; `--workspace`, `-w` |
| `version` | (none) |

The text a command prints (tables, log lines, help strings) is for people and may change in any
release. What is scripted should use `--json` (`monitor`) or the files below.

### Environment variables

`PMLAB_ARTIFACTS_DIR`, `PMLAB_CACHE_DIR`, `PMLAB_CACHE_TTL`, `PMLAB_DATA_DIR`,
**Exit codes.** A command that cannot do what its name says must not exit 0. One command changed on
those grounds: `settle-trades` settles nothing (settlement needs a plugin wired to real data, which a
bare command cannot build) yet always exited 0, so `pmlab record-trades ... && pmlab settle-trades ...`
looked healthy while every trade stayed open ([#43](https://github.com/ArtBreguez/polymarket-lab/issues/43)).
From **1.1.0** it exits **1**, says on stderr what it cannot do and prints the `SettlementEngine` call
that does settle. Its options are unchanged (see the table above). This is a change of behaviour of a
frozen command, made in a minor release on purpose because the old status was the defect: a script that
relied on that 0 now stops at that step, which is where it was not settling anything.

`PMLAB_DUCKDB_PATH`, `PMLAB_LOG_LEVEL`, `PMLAB_PARQUET_DIR`, `PMLAB_WORKSPACE` and
`PMLAB_WORKSPACE_NAME` (the one `scripts/pmlab-workspace` exports; `PMLAB_WORKSPACE` wins when
both are set). Their names, and what each one means, are frozen. The Polymarket credentials
(`POLY_API_KEY`, `POLY_API_SECRET`, `POLY_API_PASSPHRASE`) and the two API base URLs
(`GAMMA_API_BASE`, `CLOB_API_BASE`) are settings of `PmlabSettings`, hence public as well.

### File formats

A file pmlab writes today can be read by every later 1.x release. A key that a later release adds
is optional when reading, so older files keep loading; a key that exists is never removed or given
a different meaning. Readers ignore keys they do not know.

**`champion.json`** is written by `ChampionManifest.publish`, `ModelRegistry.record` and
`ModelRegistry.rollback`, always through `ChampionManifest.to_dict`:

| Key | Holds |
|-----|-------|
| `model_name` | The name the champion was published under. |
| `model_path` | Path of the pickled model. |
| `calibrator_path` | Path of the calibrator, or `null`. |
| `published_at` | ISO-8601 timestamp of the promotion. |
| `plugin_family` | The market family it was trained for. |
| `publish_gate` | The `HoldoutGateResult` it passed: `decision`, `segment_results`, `aggregate_pnl`, `aggregate_trades`. |
| `costs` | The `CostModel` the gate was computed under, or `null`. |
| `backtest_expectation` | What the backtest promised (`BacktestExpectation`), or `null`. |
| `reproducibility` | The run's reproducibility record, or `null`. |

`champion.json` carries no version number of its own: it evolves only by adding optional keys, as
`costs`, `backtest_expectation` and `reproducibility` were added. A change that cannot be made that
way would wait for 2.0.0.

**Run records** (`LocalJSONTracker`, one `<run_id>.json` per run under `<root>/<experiment>/`) have
the keys `run_id`, `experiment`, `run_name`, `params`, `metrics`, `gate`, `reproducibility`,
`status`, `started_at`, `ended_at` and `error`.

**The reproducibility record** (the `reproducibility` block above, and what `pmlab reproduce`
re-executes) has the keys `schema_version`, `kind`, `created_at`, `seeds`, `environment`,
`config`, `data` and `metrics`. `schema_version` is `1`. A reader refuses a record whose
`schema_version` is newer than it understands, with a message that says to upgrade; a record never
changes meaning silently.

**Plugin discovery:** third-party plugins register under the `pmlab.plugins` entry-point group.

The formats not listed here (the layout of a feature-snapshot store, the paper-trade log, the
panel parquet, the sample dataset's contents) are **not** frozen by 1.0. They are documented where
they are described, and a change to one is announced in the changelog.

## What is internal

Everything else, and in particular:

* any module path other than the eight above: `pmlab.core.costs`, `pmlab.modeling.champion`,
  `pmlab.plugins.base` and the rest are where the code lives *today*, not an address. The same
  class is available from `pmlab`, and that is the path the documentation uses;
* names that start with an underscore, and modules whose name does (`pmlab._deprecation`);
* anything not in an `__all__`, even if a public module happens to import it;
* the text of messages, `repr` and `str` output, log lines and the wording of warnings (the warning
  *class*, `PmlabDeprecationWarning`, is public);
* the numbers inside the bundled sample (`pmlab.datasets` is public; what `load_sample()` contains
  is not);
* the performance budget (see [Performance budget](performance.md)): it is re-calibrated on purpose.

If your code depends on something that is not on this page, open an issue: it will either be made
public, with the same guarantees, or you will be told it should not be relied on.

## How the freeze is enforced

The promise is a test, not a paragraph.

* `scripts/api_surface.py` reads the public API off the installed package: every public name with
  its kind and signature (and, for a class, its public members, bases and abstract methods), every
  CLI command and option, and the keys of the files above.
* `tests/api/public_api.json` is what it read when the surface last changed *on purpose*.
* `tests/api/test_public_api.py` fails when the two differ, and says which name, option or key was
  added, removed or changed. The end-to-end test runs the same script with the Python of a venv
  that holds only the built wheel, so the *installed* package is held to the snapshot too.
* `tests/api/test_docs_use_public_api.py` fails when a code example in the documentation, the
  README, `examples/` or the tutorial notebook imports a name that is not public.
* `tests/test_docs_site.py` fails when a public name has no entry in the API reference.

### Changing the public API on purpose

1. Make the change. The snapshot test now fails and lists the difference.
2. If it is an **addition** (a new public name, option, key): export it (`__all__`), give it a
   docstring and an entry in `docs/reference/`, then regenerate the snapshot with
   `uv run python scripts/api_surface.py --write`. Commit the snapshot with the change; its diff is
   what a reviewer reads. Add a changelog line under *Added*.
3. If it is a **removal or an incompatible change**: it needs a [deprecation](deprecation.md) first
   and is made only in a major release. The snapshot is regenerated in that release, and the
   changelog lists it under *Removed* or *Changed*.
4. A **deprecation** does not change the snapshot: the signature is preserved. It changes the
   registry that [the deprecation page](deprecation.md#currently-deprecated) is checked against.

## Related

* [Deprecation policy](deprecation.md): how a name is retired, the timeline, and the warning.
* [API reference](reference/index.md): generated from the docstrings of the public names.
* The [changelog](https://github.com/ArtBreguez/polymarket-lab/blob/main/CHANGELOG.md): every
  release, including each addition and deprecation.
