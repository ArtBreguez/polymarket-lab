# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com/).

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
  market shows), `price_untimed`, `price_after_cutoff`, and per market `market_unresolved`,
  `market_truth_unbinned`, `market_no_usable_price`.
- **No price is invented.** `market_price` no longer defaults to 0.5, and neither does
  `_build_spec`: an absent or unreadable Gamma price is `None` in `outcome_schema`, a short
  price list leaves the remaining bins unpriced, and one bad entry no longer discards the others.
- **Point-in-time.** A price is used only if it was observed strictly before the decision (the
  rule `FeatureSnapshotStore.as_of` applies to features). `discover_markets` now stamps
  `price_observed_at` (one reading of an injectable `clock`) and `price_source = "discovery"`
  on every spec. **The discovery-time Gamma price is the only per-bin price this plugin has**:
  the CLOB client reads the current midpoint and keeps no history, and the snapshot store holds
  no per-bin price. A market discovered after it resolved or after the decision has the wrong
  prices and is excluded as `price_after_cutoff`, not used. A better source can be supplied per
  bin in `outcome_schema` (`observed_at`, `source`).
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
