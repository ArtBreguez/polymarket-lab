# pmlab Roadmap

Vision: **the complete, best-engineered ML lifecycle library for Polymarket** —
from raw market discovery to a monitored live champion, every stage typed,
tested, and reproducible.

This roadmap is organized by the ML lifecycle. Each stage lists what **exists
today** (✅) and what's **planned** (▢), with the target release. Status current
as of **v0.5.0**.

---

## Lifecycle coverage at a glance

| Stage | Today (v0.6.0) | Biggest gap |
|---|---|---|
| 1. Data & ingestion | Gamma/CLOB clients, DiskCache, plugins | No feature store / panel versioning |
| 2. Feature engineering | 5 transforms, per-plugin features | No feature registry, no leakage checks |
| 3. Modeling | LGBM, sklearn, ensemble, conformal, calibration (binary+multiclass), tuning | ✅ core complete |
| 4. Validation | rolling-origin, holdout gate, Brier, purged/embargoed CV; trade log composes into metrics + gate (0.6.1) | No drift/stability report |
| 5. Model management | ChampionManifest hard gate | No experiment tracking / model registry |
| 6. Execution | Paper + Live broker, settlement | No realistic slippage/latency model |
| 7. Monitoring | — | No drift/calibration monitoring in prod |
| 8. Reproducibility & DX | CLI (8 cmds), typed, docs | No end-to-end tutorial, no seeds/config capture |

---

## v0.6.0 — Finish the modeling layer ✅ SHIPPED

The v0.5.0 additions were binary-first. This release closed that gap and made
tuning first-class.

- ✅ **Multiclass calibration.** `CalibratedForecaster` now supports >2 classes
  via one-vs-rest (`MulticlassCalibrator`), isotonic/sigmoid — F1 and
  multi-outcome political markets are calibrated too.
- ✅ **Multiclass Brier / calibration diagnostics.** `multiclass_brier` (with a
  climatology skill score) and `reliability_data_multiclass` (per-class curves).
- ✅ **`TunedForecaster` / `tune()`** — Optuna hyperparameter search that
  **respects the no-lookahead contract** by scoring trials through
  `rolling_origin_eval`, never a random shuffle. Optuna is an optional extra.
- ✅ **Cross-validation utilities** — `purged_kfold` / `embargoed_split` for time
  series (López de Prado style) so tuning and model selection don't leak.

## v0.7.0 — Data & feature engineering as a first-class stage

Turn the empty `pmlab.data` package into the backbone of reproducibility.

- ▢ **Feature store** — persist/version the `feature_*` rows a plugin produces
  (parquet + manifest), keyed by (family, market, decision_horizon), so a retrain
  reuses exactly the features a prior run saw. Kills silent feature drift between
  training and serving.
- ▢ **Panel builder** — a typed `build_panel(plugin, specs)` that assembles the
  training panel from plugin rows with schema validation and dtype enforcement.
- ▢ **Leakage guards** — assertions that no feature column is computed from
  post-decision data; wire into `build_training_row` review and CI.
- ▢ **More transforms** — target encoding (with CV folds), interaction terms,
  missing-value indicators; all copy-on-write like the existing five.

## v0.8.0 — Validation & model management

- ▢ **Experiment tracking (pluggable)** — a thin `ExperimentTracker` protocol with
  a local-JSON default and optional MLflow backend, so every backtest logs params,
  metrics, and the gate decision. No heavy dependency forced on users.
- ▢ **Model registry** — extend `ChampionManifest` into a versioned registry
  (list, diff, roll back champions) rather than a single `champion.json`.
- ▢ **Stability report** — bootstrap confidence intervals on backtest PnL / hit
  rate so a "GO" is judged on a distribution, not a point estimate.
- ▢ **Backtest realism** — model slippage, order-book depth, and fees in
  `rolling_origin_eval` (today fills are frictionless), gated behind a `costs=`
  argument so existing results stay reproducible.

## v0.9.0 — Monitoring & the production loop

- ▢ **Drift monitoring** — PSI / KL on feature and prediction distributions
  vs the training panel; surfaces when the world has moved under the champion.
- ▢ **Live calibration tracking** — rolling Brier / reliability on realized paper
  and live trades, so miscalibration is caught before it drains the bankroll.
- ▢ **Watchdog CLI** — `pmlab monitor` producing a health report (drift, calib,
  PnL vs backtest), designed to run on a schedule and stay silent when healthy.

## v1.0.0 — Polish, DX, and stability guarantee

- ▢ **End-to-end tutorial** — one notebook: discover → panel → tune → backtest →
  gate → paper → report, on a bundled sample dataset (no network needed).
- ▢ **Reproducibility capture** — every run records seeds, library versions, and
  config into the manifest; `pmlab reproduce <run>` re-executes it.
- ▢ **API stability** — freeze the public API, add deprecation policy, publish the
  docs site (mkdocs-material) from `docs/`.
- ▢ **Coverage ≥ 90%** and a documented performance budget for the hot paths.

---

## Design principles (non-negotiable)

1. **No lookahead, ever.** Any new validation/tuning path trains strictly on
   `decision_date < eval_date`. Asserted in tests.
2. **One interface.** New models implement `MarketForecaster`; new calibrators
   share `fit/transform/save/load`. Everything stays drop-in.
3. **Optional heavy deps stay optional.** MLflow/Optuna behind extras, never in
   the core install.
4. **Typed + tested first.** TDD contract in `AGENTS.md`; mypy `--strict` clean.
5. **Reproducible over clever.** A result you can't re-run isn't a result.

---

*This file is a living contract. Propose changes via PR; tick items as they ship
and move the "status as of" version marker.*
