# pmlab Roadmap

Vision: **the complete, best-engineered ML lifecycle library for Polymarket**: from raw market
discovery to a monitored live champion, every stage typed, tested and reproducible.

**The 1.0 roadmap is done.** pmlab 1.0.0 shipped on 2026-11-08 with a frozen public API, and nothing
is scheduled beyond it. This file lists what shipped, the gaps that are known, and the ideas that
would close them. None of those is a commitment or has a date. The exact history is in the
[changelog](CHANGELOG.md); open work is in the [issue tracker](https://github.com/ArtBreguez/polymarket-lab/issues).

---

## Lifecycle coverage

| Stage | What exists | Known gap |
|---|---|---|
| 1. Data & ingestion | Gamma/CLOB clients (sync and async), DiskCache, plugins, point-in-time `FeatureSnapshotStore` | No panel versioning; no historical per-bin CLOB price (a weather panel's price is the one seen at discovery) |
| 2. Feature engineering | 9 transforms (incl. out-of-fold target encoding), per-plugin features, `build_panel`, leakage and price-timing guards | No feature registry |
| 3. Modeling | LightGBM, scikit-learn, ensembles, conformal sets, calibration (binary and multiclass), walk-forward tuning | |
| 4. Validation | Rolling-origin backtest, holdout gate, Brier decomposition, purged/embargoed CV, bootstrap stability report, one cost model | |
| 5. Model management | `ChampionManifest` hard gate, experiment tracking, versioned `ModelRegistry` with rollback | No automated retrain trigger |
| 6. Execution | Paper and live brokers, settlement, the backtest's cost model reused for fills | No latency model; costs approximate, they do not walk the book |
| 7. Monitoring | Drift (PSI/KL), live calibration, backtest-expectation comparison, `pmlab monitor` with cron-friendly exit codes | The watchdog reports; nothing acts on it (no pause, no retrain) |
| 8. Reproducibility & DX | Typed, frozen public API, docs site, seeds/versions/config/data hash per run and `pmlab reproduce`, an executed offline tutorial | The tutorial runs on synthetic data, not a real market |

## Shipped

| Milestone | What |
|---|---|
| 0.1 – 0.4 | Core primitives, plugins, LightGBM and scikit-learn forecasters, paper and live brokers, HTML report, async clients |
| 0.5 – 0.6 | Ensembles, conformal sets, calibration wrappers, multiclass calibration, walk-forward tuning, purged CV |
| 0.7 | `pmlab.data`: point-in-time snapshot store, panel builder, leakage guards; more transforms |
| 0.8 | Experiment tracking, model registry, bootstrap stability report, the cost model behind the backtest |
| 0.9 | One cost model for backtest and live, drift monitoring, live calibration, the `pmlab monitor` watchdog |
| 0.10 – 0.12 | Reproducibility capture and `pmlab reproduce`, the bundled sample and tutorial notebook, tuning for the policy you trade |
| **1.0** | The stability guarantee: [#17](https://github.com/ArtBreguez/polymarket-lab/issues/17) API freeze, deprecation policy and docs site; [#16](https://github.com/ArtBreguez/polymarket-lab/issues/16) reproducibility; [#15](https://github.com/ArtBreguez/polymarket-lab/issues/15) tutorial notebook; [#18](https://github.com/ArtBreguez/polymarket-lab/issues/18) coverage gate and performance budget |
| 1.0 follow-ups | Found by using it, fixed before the release: [#33](https://github.com/ArtBreguez/polymarket-lab/issues/33) registry round trip, [#35](https://github.com/ArtBreguez/polymarket-lab/issues/35) and [#39](https://github.com/ArtBreguez/polymarket-lab/issues/39) leakage check on ranged markets, [#36](https://github.com/ArtBreguez/polymarket-lab/issues/36) tuning under the live policy |

## What could come next

Ideas, in no order, each a candidate for a minor release that keeps the public API:

- **Close the paper-trading loop from the command line.** `pmlab settle-trades` is a placeholder that
  exits 1 (settlement needs a plugin wired to real data) and nothing writes the signals file that
  `pmlab record-trades` reads, so both steps are done from Python today.
- **A second reference plugin.** `sports_f1` is a skeleton (untimed price, one row per market). A
  finished categorical plugin would exercise the per-bin machinery on something other than weather.
- **A watchdog that acts.** A hook that pauses the broker or starts a retrain on an alert.
- **A real-market tutorial.** The bundled sample is synthetic on purpose; a recorded real market
  would be a better demonstration, if one can be shipped under its terms.
- **Panel versioning and a feature registry**, from the table above.

Breaking changes wait for 2.0 and follow the [deprecation policy](docs/deprecation.md).

---

## Design principles (non-negotiable)

1. **No lookahead, ever.** Any new validation or tuning path trains strictly on
   `decision_date < eval_date`. Asserted in tests.
2. **One interface.** New models implement `MarketForecaster`; new calibrators share
   `fit/transform/save/load`. Everything stays drop-in.
3. **Optional heavy dependencies stay optional.** MLflow and Optuna are behind extras, never in the
   core install.
4. **Typed and tested first.** The TDD contract is in [AGENTS.md](AGENTS.md); mypy `--strict` is clean.
5. **Reproducible over clever.** A result you can't re-run isn't a result.
6. **The public API is a promise.** What is public is listed in
   [docs/api-stability.md](docs/api-stability.md) and a test fails when it changes by accident.

---

*This file is a living contract. Propose changes via PR.*
