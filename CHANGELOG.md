# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com/).

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
  To list the affected ones:
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
- **The checks have limited power, and the docs say so with numbers.** Simulated: the
  PnL check false-alarms 1.5-4.8% of the time, but notices a *completely vanished*
  edge in about 19% of runs at 100 live trades and 37% at 400, because per-trade PnL
  is far noisier than its mean. The Brier check fires in most runs at 400 trades only
  once forecasts are off by about 0.2 in probability (squared error is insensitive to
  smaller shifts). A quiet watchdog is weak evidence the edge is intact.
- **`insufficient` is not healthy.** It outranks `ok`, loses to any measured problem,
  and has its own exit code. `--insufficient warn` counts it as a warn; there is
  deliberately no `ignore`.
- **Only evidence that belongs to this champion counts**: settled trades in its allowed
  segments (from `champion.json`), recorded at or after its publication, and settled
  by `--as-of` (`SettlementEngine` now writes `settled_at`; a trade from an older
  release has none and is judged by its `target_date`, counted in `n_no_settled_at`
  and noted in the reasons). Every exclusion is counted. PnL also excludes NO-side
  trades (the backtest only buys YES), trades settled at the quote by an older release
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

- **Per-segment health.** The checks pool the allowed segments; the expectation is not
  recorded per segment. `n_by_segment` is reported and `CalibrationTracker.by_segment()`
  gives the per-segment numbers.
- **Acting on an alert.** It reports and exits; pausing or retraining is the caller's.
- `scripts/pmlab-workspace` exports `PMLAB_ARTIFACTS_DIR` already inside the workspace
  while `PmlabSettings` appends `workspaces/<name>` again, so default trade-log paths
  resolve wrongly there. Pre-existing and shared with `status`/`record-trades`; not
  changed here. Pass `--trades` explicitly under the script (see `docs/watchdog.md`).

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

## [0.3.0] — 2026-05-10

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

## [0.2.0] — 2026-05-09

### Added
- `PmlabSettings` — Pydantic settings class: `PMLAB_ARTIFACTS_DIR`, `PMLAB_LOG_LEVEL`, `PMLAB_DRY_RUN`, CLOB/Gamma API base URLs; `from_env()` classmethod
- Full CLI (`pmlab` entry point): `version`, `status`, `scan-markets`, `record-trades`, `settle-trades`, `backtest`, `promote-champion`
- `WorkspaceContext` — multi-workspace path isolation via `PMLAB_ARTIFACTS_DIR`
- `scripts/pmlab-workspace` — bash wrapper to scope CLI to a named workspace
- Retry + structured logging — all API clients use exponential backoff; `logging.getLogger("pmlab")` hierarchy
- Gamma/CLOB pagination — `GammaClient` auto-paginates `next_cursor`; `AsyncClobClient` respects `asyncio.Semaphore`
- mypy strict clean — `py.typed` marker, all public APIs fully typed

---

## [0.1.0] — 2026-05-10

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
