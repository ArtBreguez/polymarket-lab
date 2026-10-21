# Watchdog: `pmlab monitor`

The holdout gate answers *was this model good on history*. [Live
calibration](live-calibration.md) answers *is it still calibrated on money at risk*.
[Drift monitoring](drift-monitoring.md) answers *is the world it is fed still the
one it was trained on*. Each of them computes numbers and stops there.

This is the layer that decides: it weighs all three against what the champion
promised when it was published, and turns the result into something a scheduler can
act on. **It prints nothing when everything is fine**, and otherwise says in a few
lines what is wrong and exits with a code that says what kind.

The thresholds live here and nowhere else. The earlier layers deliberately ship
without them: where a number becomes "bad enough to act on" is a policy choice, and
burying one inside a calculation would hide it.

---

## The quick version

```bash
pmlab monitor \
  --champion artifacts/public_models/champion.json \
  --trades   artifacts/workspaces/ops_daily/signals/v2/forward_paper_trades.json \
  --reference data/panel.parquet \
  --snapshots data/snapshots
```

Healthy: no output at all, exit code 0. Everything below is a real run against a small
workspace built through the public API and the installed wheel: one champion, and three
live worlds that each differ from training in exactly one respect.

```text
$ pmlab monitor -a artifacts -w paper_healthy --reference data/panel.parquet --as-of 2026-12-31T00:00:00Z --snapshots data/snaps/paper_healthy
[exit code: 0] [bytes printed: 0]
```

Problems come out as a few lines. A feature moved (only that feature is named, and the
model's own prediction, which depends on it; the one that did not move is not):

```text
$ pmlab monitor -a artifacts -w paper_drifted --reference data/panel.parquet --as-of 2026-12-31T00:00:00Z --snapshots data/snaps/paper_drifted
pmlab monitor: ALERT  (champion e2e-lgbm, as of 2026-12-31T00:00:00Z)
  drift       alert         feature_spread: alert (psi 5.121 >= 0.25, kl 3.440 >= 0.125)
                            prediction: alert (psi 0.316 >= 0.25)
[exit code: 20]
```

The model is still calibrated but the market stopped paying for it. Calibration is
fine, PnL is not:

```text
$ pmlab monitor -a artifacts -w paper_degraded --reference data/panel.parquet --as-of 2026-12-31T00:00:00Z --snapshots data/snaps/paper_degraded
pmlab monitor: ALERT  (champion e2e-lgbm, as of 2026-12-31T00:00:00Z)
  pnl         alert         mean PnL/stake -0.266 vs backtest +0.351 (shortfall 176%, z=3.7, n=160); alert needs shortfall >= 100%, z >= 3
[exit code: 20]
```

Run it from cron, and let the exit code do the work:

```cron
*/30 * * * * cd /srv/pmlab && .venv/bin/pmlab monitor --champion ... --trades ... --reference ... --snapshots ... || /usr/local/bin/notify "pmlab monitor exited $?"
```

```text
$ pmlab monitor ... || echo "NOTIFY: exit $rc"
NOTIFY: exit 20 -- pmlab monitor: ALERT  (champion e2e-lgbm, as of 2026-12-31T00:00:00Z)
```

The same verdict from Python. `health_report` is pure: no I/O and no clock (`as_of` is
required), so the same inputs give a byte-identical report on any day.

```python
from pathlib import Path

import numpy as np
import pandas as pd

from pmlab import (
    BacktestExpectation,
    ChampionManifest,
    HoldoutGateResult,
    SegmentGateResult,
    health_report,
)

rng = np.random.default_rng(7)


def simulate(n, edge):
    """n trades: buy a label at price q; it occurs with probability q + edge."""
    q = rng.uniform(0.30, 0.60, n)
    occurred = rng.random(n) < q + 0.06 + edge
    return pd.DataFrame({
        "predicted_prob": q + 0.06,                       # what the model said
        "label_occurred": occurred,
        "realized_pnl": np.where(occurred, 1 / q - 1, -1.0) - 0.003,
    })


# What the champion promised at promotion: mean, spread and size of its backtest PnL
# and Brier. `pmlab promote-champion --expectation` records this in champion.json.
expectation = BacktestExpectation.from_trades(simulate(400, edge=0.0))
champion = ChampionManifest(          # in production: ChampionManifest.load("champion.json")
    model_name="demo", model_path=Path("champion.pkl"), calibrator_path=None,
    gate=HoldoutGateResult("GO", [SegmentGateResult("politics", 400, 30.0, True, "ok")], 30.0, 400),
    published_at="2026-01-01T00:00:00+00:00", plugin_family="politics",
    expectation=expectation,
)


def paper_trades(live):
    """The trade log rows `PaperBroker` + `SettlementEngine` leave behind."""
    return [{
        "recorded_at": "2026-02-01T09:00:00+00:00", "target_date": "2026-02-02",
        "settled_at": "2026-02-02T12:00:00+00:00",
        "city_or_segment": "politics", "flat_stake": 1.0, "outcome": "won" if won else "lost",
        "model_prob": float(p), "realized_pnl": float(pnl),
    } for p, won, pnl in live[["predicted_prob", "label_occurred", "realized_pnl"]].itertuples(index=False)]


for label, edge in (("as promised", 0.0), ("edge gone", -0.12)):
    report = health_report(
        champion=champion, trades=paper_trades(simulate(120, edge)), drift=None,
        as_of="2026-03-01T00:00:00Z", checks=("calibration", "pnl"),
    )
    print(f"--- {label}: {report.status}, exit code {report.exit_code}")
    print(report.render(verbose=True))
```

```text
--- as promised: ok, exit code 0
pmlab monitor: OK  (champion demo, as of 2026-03-01T00:00:00Z)
  calibration ok            Brier 0.235 vs backtest 0.244 (-0.009 abs, -3.8% rel, z=-1.0, n=120); within thresholds
  pnl         ok            mean PnL/stake +0.162 vs backtest +0.161 (shortfall -1%, z=0.0, n=120); within thresholds
  skipped: drift
--- edge gone: alert, exit code 20
pmlab monitor: ALERT  (champion demo, as of 2026-03-01T00:00:00Z)
  calibration ok            Brier 0.246 vs backtest 0.244 (+0.001 abs, +0.5% rel, z=0.1, n=120); within thresholds
  pnl         alert         mean PnL/stake -0.207 vs backtest +0.161 (shortfall 229%, z=3.2, n=120); alert needs shortfall >= 100%, z >= 3
  skipped: drift
```

The first run is a champion trading as promised. In the second its picks win 12 points
less often than before while the model keeps forecasting as it did: the PnL check
alerts, and calibration barely moves. That is not a flaw in the example: a squared error
is insensitive to a modest shift in confidence (see [the statistics](#the-statistics-and-their-limits)).

---

## Exit codes

| code | meaning | when |
|---|---|---|
| `0` | healthy | every check ran and is within bounds. **Nothing is printed.** |
| `10` | warn | at least one check is at its warn level, and none is worse |
| `20` | alert | at least one check is at its alert level |
| `30` | insufficient | nothing measured is wrong, but at least one check **could not be measured** (too few trades, no backtest expectation recorded, no drift inputs) |
| `64` | bad input | missing or unreadable `champion.json`, trade log, reference panel or config; a `NO_GO` gate recorded in `champion.json`; an unwritable `--json`; a bad option value |
| `70` | internal error | an unexpected exception. One line on stderr; never a traceback |
| `2` | bad command line | rejected by the CLI parser before `pmlab` runs (unknown flag) |

The numbers are chosen so that nothing that goes wrong can look like a verdict. An
uncaught Python exception exits **1**, and a malformed command line exits **2**; if
`warn` were 1 and `alert` were 2, a crashed job would page as a warning and a typo in
a flag would page as an alert. So the verdicts are 0/10/20/30 and the failures
are 64/70 (the `sysexits.h` values for usage and software errors). A crash cannot exit 0.

If you only want "did anything go wrong", `|| notify` is enough. If you branch on the
code, treat `30` separately: it means *not enough evidence yet*, which is expected for
a champion that was published this week.

```sh
pmlab monitor ...; rc=$?
case $rc in
  0)  ;;                                           # healthy
  30) echo "not enough evidence yet" >> "$LOG" ;;   # new champion: not an incident
  *)  notify "pmlab monitor exited $rc" ;;
esac
```

The overall status is the **worst** check, ranked **alert > warn > insufficient >
ok**. `insufficient` outranks `ok` on purpose: *we could not check* must never read as
*we checked and it is fine*. It does not outrank a measured problem, so an alert is
never hidden by another check that is merely unmeasurable.

---

## What a run does

1. **Loads `champion.json`** and treats it as the authority: its allowed segments,
   its publication time, its cost model and its backtest expectation. It is never
   substituted by a retrain summary, which is overwritten on every retrain. A
   `champion.json` that records a `NO_GO` gate is refused (exit 64): such a file
   cannot have been produced by `publish()`.
2. **Selects the evidence** from the trade log (below).
3. **Builds the drift report** from your reference panel and the live snapshot window,
   point-in-time, if you gave it one.
4. **Runs the three checks** and takes the worst.
5. **Prints only the problems**, and writes the full report to `--json` if asked.

It only reads. It never changes `champion.json`, the trade log or the snapshot store.

---

## Which trades count

A trade log accumulates across champions and across segments the gate rejected, and
judging a champion on trades that are not its own produces confident nonsense. The
calibration and PnL checks therefore use only **settled trades that are evidence about
this champion, as of `--as-of`**:

- **settled**: open trades have no truth yet;
- in a segment the champion's gate **allowed** (`get_allowed_segments()`, read from
  `champion.json`), the same filter `PaperBroker` applies when recording;
- **dated**: a trade with no usable `recorded_at` or `target_date` cannot be attributed, so
  it is not assumed to be current;
- **recorded at or after the champion was published** (a re-promotion must not be
  judged on the previous champion's trades);
- **settled by `--as-of`**: recorded by then, and `settled_at` (written by
  `SettlementEngine` when it settles the trade) not after it. Replaying an old `--as-of`
  must not see truth that had not arrived: a trade for 2026-03-01 settled at 06:00 that
  day is not evidence at a midnight `--as-of` on 2026-03-01.
  **A trade from a release before 0.9.1 has no `settled_at`.** It falls back to the
  weaker rule, `target_date` not after the as-of *date*, which can admit a trade that
  settled later on the as-of day itself. Those trades are counted in `n_no_settled_at`
  (a subset of `n_in_scope`, not an exclusion) and the check's reasons say so whenever the
  count is not zero. A `settled_at` that cannot be parsed is `n_unattributed`.

Each excluded trade is counted under the first rule above that excludes it, so the counts
add up to `n_total`, and they are in the report: nothing disappears silently.

**When no trade is in the champion's segments** (a book recorded under another champion, or a
segment name that does not match) the checks are `insufficient` (exit 30) and the reason leads with the
cause. Here every settled trade of a workspace was tagged with a segment the gate did not pass:

```text
$ pmlab monitor -a artifacts -w paper_stray --skip drift --as-of 2026-12-31T00:00:00Z
pmlab monitor: INSUFFICIENT  (champion e2e-lgbm, as of 2026-12-31T00:00:00Z)
  calibration insufficient  0 of 160 settled trades are in the champion's allowed segments ['A', 'B', 'C', 'D'], need >= 30 (excluded: 160 outside the champion's allowed segments)
  pnl         insufficient  0 of 160 settled trades are in the champion's allowed segments ['A', 'B', 'C', 'D'], need >= 30 (excluded: 160 outside the champion's allowed segments)
  skipped: drift
[exit code: 30]
```

The report's metrics list `allowed_segments`. A champion that allows **no** segment at all is a different
thing: an empty allowlist means *none*, exactly as in `PaperBroker`, so it could never have recorded a
trade. `publish()` cannot produce it, so such a `champion.json` is malformed and exits 64.

**A log settled by an older release.** The `paper_degraded` workspace from the quick version, with the
`settled_at`, `settled_basis` and `settlement_version` keys removed, which is what a 0.9.0 log looks like.
Those trades were settled at the quote although they paid a slipped fill, so the PnL check leaves them out
and says so; calibration does not care how the PnL was settled and is unaffected:

```text
$ pmlab monitor -a artifacts -w paper_legacy --skip drift --as-of 2026-12-31T00:00:00Z
pmlab monitor: INSUFFICIENT  (champion e2e-lgbm, as of 2026-12-31T00:00:00Z)
  pnl         insufficient  0 scoreable settled trades, need >= 30 (excluded: 160 settled at the quote by an older release although they paid a different fill)
                            note: 160 of 160 in-scope trades have no settled_at (settled by an older release), so they are judged by target_date and may include truth that arrived after as_of
  skipped: drift
[exit code: 30]
```

```python
from pathlib import Path

from pmlab import (
    BacktestExpectation,
    ChampionManifest,
    HoldoutGateResult,
    SegmentGateResult,
    health_report,
)

champion = ChampionManifest(
    model_name="demo", model_path=Path("champion.pkl"), calibrator_path=None,
    gate=HoldoutGateResult(
        "GO",
        [SegmentGateResult("politics", 120, 9.0, True, "ok"),
         SegmentGateResult("sports", 120, -3.0, False, "negative_pnl")],
        aggregate_pnl=6.0, aggregate_trades=240),
    published_at="2026-01-01T00:00:00+00:00", plugin_family="politics",
    expectation=BacktestExpectation(240, 1.0, 0.05, 0.9, 0.22, 0.1),
)


def trade(**over):
    row = {"recorded_at": "2026-02-01T09:00:00+00:00", "target_date": "2026-02-02",
           "city_or_segment": "politics", "flat_stake": 1.0, "outcome": "won",
           "model_prob": 0.6, "realized_pnl": 0.4}
    return {**row, **over}


trades = (
    [trade() for _ in range(60)]                                          # evidence
    + [trade(outcome=None, realized_pnl=None) for _ in range(5)]          # not settled yet
    + [trade(city_or_segment="sports") for _ in range(8)]                 # segment failed the gate
    + [trade(recorded_at="2025-12-20T09:00:00+00:00") for _ in range(10)] # the previous champion's
    + [trade(target_date="2026-03-04") for _ in range(4)]                 # truth not known at as_of
    + [{k: v for k, v in trade().items() if k != "recorded_at"} for _ in range(3)]  # undatable
)
report = health_report(champion=champion, trades=trades, drift=None,
                       as_of="2026-03-01T00:00:00Z", checks=("pnl",))
metrics = report.checks[0].metrics
for key in ("n_total", "n_in_scope", "n_open", "n_out_of_segment", "n_before_champion",
            "n_after_as_of", "n_unattributed"):
    print(f"{key:<19}{metrics[key]:>4}")
```

```text
n_total              90
n_in_scope           60
n_open                5
n_out_of_segment      8
n_before_champion    10
n_after_as_of         4
n_unattributed        3
```

The PnL check also leaves out, counted in the report:

- **NO-side trades** (`n_excluded_no_side`, anything whose `direction` is not `"yes"`).
  The backtest only ever buys YES, so its mean and spread describe that strategy; a NO
  trade's per-stake PnL is the same arithmetic, but it is not a draw from the
  distribution being tested against, so the z test's null does not cover it.
  Calibration still counts them: a forecast is about the label, whichever side was taken.
- trades **settled at the quote although they paid a different fill**
  (`n_excluded_legacy_settlement`). Releases up to 0.9.0 entered every settlement at the
  quote, so such a trade's `realized_pnl` carries none of the slippage its `fill_price`
  says it paid, and would otherwise be judged as cost-matched. Settlement now writes
  `settled_basis` on every trade; one without it is kept only if its `fill_price` equals
  the quote (or none was recorded), because then both arithmetics enter at the same
  price. It is excluded rather than recomputed so the number the watchdog uses is always
  the one in your log.
- trades filled under a **different cost model** than `champion.json` records
  (`n_excluded_cost_basis`), because live and backtest PnL are only comparable if both
  paid the same friction (see [Costs](costs.md)). A trade with no cost fields is kept and
  counted in `n_cost_unverified`; if the champion records no costs at all, the report
  says `cost_basis: "unknown"`.

---

## The three checks

### Drift

Reuses the `DriftReport` classification as it is: the status of the check is the report's
own (worst of the features and, if present, the prediction), and the thresholds are the
ones the report was built with. The reasons name each offending feature, worst first, with
the signal that fired and the line it crossed:

```text
feature_spread: alert (psi 5.121 >= 0.25, kl 3.440 >= 0.125)
```

- **The live window** is `[--window-start, --as-of)`; `--window-start` defaults to
  `--window-days` (7) before `--as-of`. It is read through `FeatureSnapshotStore.as_of`, so
  nothing captured at or after `--as-of` can leak in, and markets the reference panel already
  contains are excluded (see [Drift monitoring](drift-monitoring.md)).
- **The snapshot family** is the key the store files the live snapshots under: `--family`, defaulting
  to the champion's `plugin_family`. A family the store has nothing for is an input error (exit 64),
  not "nothing live yet".
- **Prediction drift** is on by default: the champion model scores the reference panel and the
  live window. `--no-prediction` skips it (and does not load the model). The model is found
  at the path `champion.json` records, or next to `champion.json` if that directory has moved.
- **Features** default to the model's own `feature_names()`; with `--no-prediction`, to the
  reference panel's `feature_*` columns. `--feature NAME` (repeatable) overrides.
- **No drift inputs** (`--reference` and `--snapshots` not given) is `insufficient`
  ("not evaluated"), not green. Pass `--skip drift` to leave the check out deliberately.
- Too few live rows for a feature is `insufficient`, exactly as in the drift report.

### Calibration

Compares the **live Brier score** on realized trades with the **backtest Brier** recorded in
`champion.json`. A level (warn or alert) is reached only if the live Brier is worse by *at
least the absolute margin* **and** *the relative margin* **and** the gap clears the noise
(`z`, below). Absolute alone over-fires on a tiny Brier; relative alone over-fires on a large one.

```python
from pathlib import Path

from pmlab import (
    BacktestExpectation,
    ChampionManifest,
    HoldoutGateResult,
    SegmentGateResult,
    health_report,
)

champion = ChampionManifest(
    model_name="demo", model_path=Path("champion.pkl"), calibrator_path=None,
    gate=HoldoutGateResult("GO", [SegmentGateResult("politics", 120, 9.0, True, "ok")], 9.0, 120),
    published_at="2026-01-01T00:00:00+00:00", plugin_family="politics",
    expectation=BacktestExpectation(240, 1.0, 0.05, 0.9, brier_score=0.22, brier_std=0.10),
)

# 60 settled trades where the model said 0.9 and the label occurred half the time.
trades = [{
    "recorded_at": "2026-02-01T09:00:00+00:00", "target_date": "2026-02-02",
    "city_or_segment": "politics", "flat_stake": 1.0, "model_prob": 0.9,
    "outcome": "won" if i % 2 else "lost", "realized_pnl": 0.1 if i % 2 else -1.0,
} for i in range(60)]

report = health_report(champion=champion, trades=trades, drift=None,
                       as_of="2026-03-01T00:00:00Z", checks=("calibration",))
print(report.status, "->", report.checks[0].reasons[0])
```

```text
alert -> Brier 0.410 vs backtest 0.220 (+0.190 abs, +86.4% rel, z=3.6, n=60); alert needs >= +0.05 abs, >= +25% rel, z >= 3
```

It uses `CalibrationTracker`, so `outcome` is read as *the label occurred*, not *the trade made
money* (see [Live calibration](live-calibration.md)). Trades without a `model_prob` are not
scored and are counted in `n_skipped_no_forecast`. A `model_prob` that is a number but not a
probability (NaN, infinite, below 0 or above 1) is **bad data, not a bad forecast**: it says what
writes the trade log is broken, and scoring it would either page as a Brier alert or poison every
statistic with NaN. It is left out, counted in `n_invalid_forecast`, and a note in the reasons names
the count. `reliability_live` is reported in the metrics for context but does not drive the status.

### PnL against the backtest

Compares the **mean realized PnL per unit stake** on live trades with the backtest's. Per unit
stake (`realized_pnl / flat_stake`) so that trading 0.1 per trade against a backtest at 1.0 does not
read as a ten-fold shortfall. The shortfall is `(backtest - live) / backtest`: 0.5 is earning half of
what was promised, 1.0 is earning nothing.

```text
mean PnL/stake -0.266 vs backtest +0.351 (shortfall 176%, z=3.7, n=160); alert needs shortfall >= 100%, z >= 3
```

The test is **one-sided**: only a shortfall can warn. Beating the backtest is never a problem, and a
watchdog that pages on good luck gets muted. A non-positive backtest mean cannot anchor a shortfall
(there is no "half of what was promised" of a promise of nothing), so that case is `insufficient`
(`invalid_expectation`) **on every run, permanently**: exit 30 does not clear by itself and the
reason says so, `... this check will stay insufficient on every run until the champion is
re-published on a backtest that earned money`. Drift and calibration are still judged. It is kept
rather than waved through because a PnL check that quietly never runs must not look like health.

---

## The backtest expectation

"Live PnL is 0.02 per trade" is neither good nor bad until you know how noisy per-trade PnL is. The
gate records only totals (`aggregate_pnl` over `aggregate_trades`), so the champion also records a
**`BacktestExpectation`**: the mean, standard deviation and count of per-stake PnL, plus the mean and
standard deviation of the per-trade squared forecast error (Brier), all over the backtest's *traded*
rows. Traded rows, not the whole panel, because live trades are a selected subset and a calibration
comparison is only fair between like subsets.

```bash
pmlab backtest --plugin my_family --panel panel.parquet --flat-stake 1.0 --costs costs.json --output metrics.json
pmlab promote-champion model.pkl --gate-path gate.json --plugin my_family --expectation metrics.json
pmlab record-trades --plugin my_family
```

`backtest --output` writes the expectation under an `expectation` key and the cost model it ran
under (`--costs`, else the default) under `costs`. `--expectation` reads either that file or a
bare expectation object, and `promote-champion` publishes the champion with the file's costs, so
`champion.json` records the friction the promise was made under; `record-trades` then fills the
paper trades under the champion's costs. That is what lets the PnL check report a known
`cost_basis` instead of `"unknown"`; see [Costs](costs.md#from-the-command-line). From Python:

```python
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from pmlab import BacktestExpectation, ChampionManifest, HoldoutGateResult, LGBMForecaster

# What `rolling_origin_eval(...).trades` gives you (here: 200 synthetic trades).
rng = np.random.default_rng(0)
trades = pd.DataFrame({
    "realized_pnl": rng.normal(0.12, 0.9, 200),        # USDC per trade, at flat_stake 2.0
    "predicted_prob": rng.uniform(0.4, 0.8, 200),
    "label_occurred": rng.random(200) < 0.6,
    "segment": "politics",
})
expectation = BacktestExpectation.from_trades(trades, flat_stake=2.0)
print({k: round(v, 4) if isinstance(v, float) else v for k, v in expectation.to_dict().items()})

# Publish it with the champion. The gate still hard-blocks NO_GO exactly as before.
X = pd.DataFrame({"f": rng.random(60)})
model = LGBMForecaster()
model.fit(X, (X["f"] > 0.5).astype(int))
gate = HoldoutGateResult.evaluate(
    trades.assign(outcome="won"), required_segments=["politics"], min_trades_per_segment=40
)
out = Path(tempfile.mkdtemp())
ChampionManifest.publish(model, gate, out, plugin_family="politics", expectation=expectation)
on_disk = json.loads((out / "champion.json").read_text())
print("champion.json key:", "backtest_expectation" in on_disk)

# A champion published before v0.9.1 has no such key. It still loads; the watchdog then
# reports calibration and PnL as unavailable rather than guessing.
del on_disk["backtest_expectation"]
(out / "champion.json").write_text(json.dumps(on_disk))
print("old champion loads, expectation =", ChampionManifest.load(out / "champion.json").expectation)
```

```text
{'n_trades': 200, 'flat_stake': 2.0, 'mean_pnl_per_stake': 0.0669, 'std_pnl_per_stake': 0.4336, 'brier_score': 0.2509, 'brier_std': 0.1587}
champion.json key: True
old champion loads, expectation = None
```

A champion published before v0.9.1 has no expectation. It still loads, and the watchdog reports
**calibration and PnL as `insufficient` ("no backtest expectation recorded")**: not green and not a
crash. Drift needs no expectation and is unaffected. Re-publish the champion with an expectation to
turn the other two on:

```text
$ pmlab monitor -a artifacts_old -w paper_healthy --reference data/panel.parquet --as-of 2026-12-31T00:00:00Z --snapshots data/snaps/paper_healthy
pmlab monitor: INSUFFICIENT  (champion e2e-lgbm, as of 2026-12-31T00:00:00Z)
  calibration insufficient  no backtest Brier expectation recorded in champion.json: re-publish the champion with a BacktestExpectation to enable this check
  pnl         insufficient  no backtest PnL expectation recorded in champion.json: re-publish the champion with a BacktestExpectation to enable this check
[exit code: 30]
```

A *present but unusable* record is different: it raises (exit 64), so it can never read as "unknown"
and quietly weaken the watchdog. `BacktestExpectation` refuses to be built, or loaded from
`champion.json`, with a non-finite number, a negative spread, a stake of zero or less, a Brier score
outside [0, 1], or fewer than two trades. The non-finite case matters most: `NaN` is a legal token for
Python's `json.loads` and compares false with everything, so an unchecked `NaN` spread would not fail the
z test, it would switch the check off and let a badly degraded champion exit 0. As a second line of
defence `health_report` reports a check whose expectation (cause `invalid_expectation`) or live
statistic (cause `invalid_statistic`, for example a variance that overflowed) is not finite as
`insufficient` with the offending number named, never as `ok`.

---

## Costs and NO trades, from the CLI

The whole chain through the installed `pmlab`: a backtest under a cost model, a champion promoted on it,
paper trades recorded under the champion's costs (a third of them NO-side), settled, then watched. The
champion carries the costs without `promote-champion` being told them again (it reads them from
`metrics.json`), and the paper trades carry the slippage and fee the backtest charged:

```text
$ pmlab promote-champion model.pkl --gate-path gate.json --plugin e2e_family --output-dir artifacts/public_models --name e2e-lgbm --expectation metrics.json
Gate decision: GO
Aggregate PnL: $+100.07 over 285 trades
Champion published!
  Model: artifacts/public_models/champion.pkl
  Manifest: artifacts/public_models/champion.json
  Allowed segments: {'A', 'C', 'D', 'B'}
  Costs: {'taker_bps': 30.0, 'slippage_bps': 50.0, 'slippage_fixed': 0.0, 
'depth_bps_per_unit': 0.0, 'depth_reference_stake': 1.0}
```

One YES trade and two NO trades from that log. The YES quote 0.328 became a fill of 0.33003 (+50 bps); a NO
trade buys the NO token at `1 - gamma_price` plus the same friction. `outcome` says whether the *label*
occurred, so a NO trade on a label that occurred is `"won"` while its PnL is the whole stake and the fee
lost; the other NO trade's token paid 1 per share, `1 / 0.60666 - 1 - 0.003`:

```text
{"direction":"yes","gamma_price":0.32839121062291904,"fill_price":0.33003317,"fee_paid":0.003,"outcome":"lost","realized_pnl":-1.003,"settled_basis":"fill_price","settlement_version":1,"settled_at":"2026-12-30T12:00:00+00:00"}
{"direction":"no","gamma_price":0.6105203467794211,"fill_price":0.39142705,"fee_paid":0.003,"outcome":"won","realized_pnl":-1.003,"settled_basis":"fill_price","settlement_version":1,"settled_at":"2026-12-30T12:00:00+00:00"}
{"direction":"no","gamma_price":0.3963629312295738,"fill_price":0.60665525,"fee_paid":0.003,"outcome":"lost","realized_pnl":0.645383,"settled_basis":"fill_price","settlement_version":1,"settled_at":"2026-12-30T12:00:00+00:00"}
```

Recomputing every settled trade in that log from its own recorded fields (shares bought at `fill_price`,
paying 1 if the token wins, minus the stake and the fee) agrees with `realized_pnl` to 5e-7, and each of the
37 NO trades whose label occurred lost exactly stake plus fee.

The PnL check uses the costs: the cost basis is known and matches, nothing was unverified, and the 60 NO
trades are counted out of the comparison rather than judged against a strategy the backtest never ran:

```text
$ pmlab monitor -a artifacts -w paper_with_no ... --json - | jq '.checks[] | select(.name == "pnl") | .metrics | {cost_basis, n_in_scope, n_trades, n_excluded_no_side, n_excluded_cost_basis, n_cost_unverified}'
{"cost_basis":"match","n_in_scope":180,"n_trades":120,"n_excluded_no_side":60,"n_excluded_cost_basis":0,"n_cost_unverified":0}
```

Without costs anywhere (a champion promoted with no `--costs` and no costs in its expectation file) the
same check reports `"cost_basis": "unknown"` and compares nothing.

---

## `insufficient` is not healthy

A check that cannot be measured (fewer settled trades than `min_trades`, no expectation, no drift
inputs) is `insufficient`, and by default the report gets its own status and its own exit code,
`30`:

```text
$ pmlab monitor -a artifacts -w paper_new --reference data/panel.parquet --as-of 2026-12-31T00:00:00Z --snapshots data/snaps/paper_healthy
pmlab monitor: INSUFFICIENT  (champion e2e-lgbm, as of 2026-12-31T00:00:00Z)
  calibration insufficient  12 scoreable settled trades, need >= 30 (none excluded)
  pnl         insufficient  12 scoreable settled trades, need >= 30 (none excluded)
[exit code: 30]
```

`--insufficient warn` (or `insufficient = "warn"` in the config file) counts it as a warn (exit 10)
instead; the individual checks still say `insufficient`:

```text
$ pmlab monitor -a artifacts -w paper_new --reference data/panel.parquet --as-of 2026-12-31T00:00:00Z --snapshots data/snaps/paper_healthy --insufficient warn
pmlab monitor: WARN  (champion e2e-lgbm, as of 2026-12-31T00:00:00Z)
  calibration insufficient  12 scoreable settled trades, need >= 30 (none excluded)
  pnl         insufficient  12 scoreable settled trades, need >= 30 (none excluded)
[exit code: 10]
```

There is deliberately no `ignore`. If a new champion is quiet for a week because it has not traded
yet, you want to be able to see that, not to read it as health.

---

## Thresholds

Every threshold is configuration, written into every report. The defaults are conventions, not
standards; tune them on your own history.

| check | setting | default | meaning |
|---|---|---|---|
| calibration | `min_trades` | 30 | scoreable settled trades needed before a verdict |
| | `brier_warn_abs` / `brier_alert_abs` | 0.02 / 0.05 | live Brier minus backtest Brier |
| | `brier_warn_rel` / `brier_alert_rel` | 0.10 / 0.25 | the same gap as a fraction of the backtest Brier |
| | `z_warn` / `z_alert` | 2.0 / 3.0 | one-sided z the gap must clear |
| | `n_bins` | 10 | reliability bins for the reported `reliability_live` |
| pnl | `min_trades` | 30 | scoreable settled trades needed before a verdict |
| | `shortfall_warn` / `shortfall_alert` | 0.5 / 1.0 | `(backtest - live) / backtest` |
| | `z_warn` / `z_alert` | 2.0 / 3.0 | one-sided z the shortfall must clear |
| drift | everything in `DriftConfig` | see [Drift monitoring](drift-monitoring.md) | PSI 0.10/0.25, KL 0.05/0.125, missing 0.05/0.20, `min_samples` 300 |
| policy | `insufficient` | `separate` | `separate` (exit 30) or `warn` (exit 10) |

A value at or above a threshold is in that band, with 1e-9 of slack for float round-off: `0.25 - 0.20`
is `0.04999999999999999` in floating point, and that is an alert at `0.05`, not a warn.

Set them in a TOML (or JSON) file with `--config`; absent keys keep their defaults:

```toml
# watchdog.toml
insufficient = "warn"

[pnl]
min_trades = 50
z_warn = 1.645

[calibration]
brier_warn_abs = 0.03

[drift]
min_samples = 500

[drift.thresholds]
psi_warn = 0.15
```

`--min-trades N` (both checks) and `--insufficient separate|warn` override the file. **Unknown keys
are rejected**, not ignored: a typo'd threshold would otherwise leave the default in force while you
believed otherwise.

```python
from pmlab import WatchdogConfig, WatchdogInputError

# A partial config fills in the rest from the defaults.
cfg = WatchdogConfig.from_dict({"pnl": {"min_trades": 50, "z_warn": 1.645}})
print(cfg.pnl)
print("calibration untouched:", cfg.calibration == WatchdogConfig().calibration)

# A typo is an error, not a silently ignored threshold.
for bad in ({"pnl": {"min_trade": 50}}, {"calibration": {"z_warn": 4, "z_alert": 3}}):
    try:
        WatchdogConfig.from_dict(bad)
    except WatchdogInputError as exc:
        print("rejected:", exc)
```

```text
PnLThresholds(min_trades=50, shortfall_warn=0.5, shortfall_alert=1.0, z_warn=1.645, z_alert=3.0)
calibration untouched: True
rejected: unknown pnl setting(s) ['min_trade']; valid settings: ['min_trades', 'shortfall_alert', 'shortfall_warn', 'z_alert', 'z_warn']
rejected: z_warn (4) must not exceed z_alert (3): warn is the lower bar and alert the higher
```

---

## Output

**Quiet on green.** A healthy run prints nothing, to stdout or stderr. `--verbose` prints the full
report even then:

```text
$ pmlab monitor -a artifacts -w paper_healthy --reference data/panel.parquet --as-of 2026-12-31T00:00:00Z --snapshots data/snaps/paper_healthy --verbose
pmlab monitor: OK  (champion e2e-lgbm, as of 2026-12-31T00:00:00Z)
  drift       ok            2 feature(s) and the prediction within thresholds (max psi 0.020, 600 live rows)
  calibration ok            Brier 0.187 vs backtest 0.205 (-0.019 abs, -9.0% rel, z=-0.9, n=160); within thresholds
  pnl         ok            mean PnL/stake +0.601 vs backtest +0.351 (shortfall -71%, z=-0.8, n=160); within thresholds
[exit code: 0]
```

**`--json PATH`** always writes the full report, healthy or not, for logs and alerting pipelines.
`--json -` writes it to stdout, and then the human text moves to stderr so that stdout stays
machine-readable:

```text
$ pmlab monitor ... --json - 2>/dev/null | python3 -c '...'
{'status': 'alert', 'exit_code': 20}
[('drift', 'ok'), ('calibration', 'ok'), ('pnl', 'alert')]
```

| stream | content |
|---|---|
| stdout | the problem report (nothing when healthy); with `--json -`, the JSON instead |
| stderr | usage and internal errors (one line); with `--json -`, the problem report |

The text is plain ASCII with no colour codes: it lands in cron mail and log files.

The JSON is `HealthReport.to_dict()`, strict JSON (no `NaN` or `inf`; an infinite ratio is `null`),
and `HealthReport.from_dict()` restores it exactly. `status` and `exit_code` are derived, not trusted,
when reading it back.

```python
import json
from pathlib import Path

from pmlab import BacktestExpectation, ChampionManifest, HealthReport, HoldoutGateResult
from pmlab import SegmentGateResult, health_report

champion = ChampionManifest(
    model_name="demo", model_path=Path("champion.pkl"), calibrator_path=None,
    gate=HoldoutGateResult("GO", [SegmentGateResult("politics", 120, 9.0, True, "ok")], 9.0, 120),
    published_at="2026-01-01T00:00:00+00:00", plugin_family="politics",
    expectation=BacktestExpectation(240, 1.0, 0.05, 0.9, 0.22, 0.1),
)
report = health_report(champion=champion, trades=[], drift=None, as_of="2026-03-01T00:00:00Z")

data = report.to_dict()
print(list(data))
print({c["name"]: (c["status"], c["cause"]) for c in data["checks"]})
print(json.dumps(data["checks"][1]["metrics"]))
print("round trip:", HealthReport.from_dict(json.loads(json.dumps(data))) == report)
print("exit code:", data["exit_code"])
```

```text
['status', 'exit_code', 'as_of', 'champion', 'skipped', 'checks', 'config']
{'drift': ('insufficient', 'not_evaluated'), 'calibration': ('insufficient', 'too_few_samples'), 'pnl': ('insufficient', 'too_few_samples')}
{"n_total": 0, "n_open": 0, "n_out_of_segment": 0, "n_unattributed": 0, "n_before_champion": 0, "n_after_as_of": 0, "n_no_settled_at": 0, "n_in_scope": 0, "n_invalid_forecast": 0, "allowed_segments": ["politics"], "n_by_segment": {}, "n_trades": 0, "n_skipped_no_forecast": 0, "min_trades": 30}
round trip: True
exit code: 30
```

Checks are always in the order `drift`, `calibration`, `pnl`, metrics in a fixed order, and the
trade order in the log does not matter (rows are put in a canonical order before any float sum), so two
runs on the same inputs diff cleanly.

---

## The statistics, and their limits

A shortfall is a **finding** only if it is both *material* (the effect-size thresholds above) and
*distinguishable from noise*. The noise test is a **one-sided two-sample Welch z**:

```
z = (reference - live) / sqrt( live_variance / n_live  +  backtest_std² / n_backtest )
```

with a normal approximation. The sign is chosen so that **a positive z always means worse than the
reference**: for Brier, where higher is worse, it is `(live - reference)` in the numerator. Both terms of the standard
error matter. The live sample is noisy; so is the backtest mean, which is itself an estimate: a promise
made on 20 trades is a weak promise, and falling short of it means less than falling short of one made on
2000. The two guards catch different mistakes: effect size alone pages on noise at small n, significance
alone pages on trivia at large n.

`z_warn = 2` and `z_alert = 3` are stricter than the textbook one-sided 1.64 and 2.33 on purpose. The
watchdog looks every run, and repeated looks at one growing sample inflate the false-alarm rate far above
the nominal level.

**How often does the PnL check fire?** 400 simulated champions per cell; every live trade buys a label
at a price between 0.30 and 0.60, and the backtest promised a 6-point edge. "Edge gone" means the market
now prices the label correctly; "reversed" means it is overpriced by 6 points:

```python
from pathlib import Path

import numpy as np

from pmlab import (
    BacktestExpectation,
    ChampionManifest,
    HoldoutGateResult,
    PnLThresholds,
    SegmentGateResult,
    WatchdogConfig,
    health_report,
)

COST = 0.003


def draw(rng, n, edge):
    """Per-unit-stake returns of n trades that each buy a label at q (0.30-0.60)."""
    q = rng.uniform(0.30, 0.60, n)
    occurred = rng.random(n) < q + 0.06 + edge
    return np.where(occurred, 1 / q - 1, -1.0) - COST


def verdicts(live_edge, n_live, sims=400):
    """How often the PnL check says warn/alert, over `sims` independent champions."""
    rng = np.random.default_rng(1000 + n_live)
    cfg = WatchdogConfig(pnl=PnLThresholds(min_trades=2))
    hits = 0
    for _ in range(sims):
        backtest = draw(rng, 400, 0.0)
        expectation = BacktestExpectation(
            400, 1.0, float(backtest.mean()), float(backtest.std(ddof=1))
        )
        champion = ChampionManifest(
            "c", Path("x"), None,
            HoldoutGateResult("GO", [SegmentGateResult("s", 400, 1.0, True, "ok")], 1.0, 400),
            "2026-01-01T00:00:00+00:00", "f", expectation=expectation,
        )
        trades = [
            {"recorded_at": "2026-02-01T00:00:00+00:00", "target_date": "2026-02-02",
             "city_or_segment": "s", "outcome": "lost", "realized_pnl": float(r)}
            for r in draw(rng, n_live, live_edge)
        ]
        report = health_report(champion=champion, trades=trades, drift=None,
                               as_of="2026-03-01T00:00:00Z", config=cfg, checks=("pnl",))
        hits += report.status in ("warn", "alert")
    return hits / sims


print("live trades   as promised   edge gone   edge reversed")
for n in (10, 20, 30, 50, 100, 200, 400):
    row = [verdicts(edge, n) for edge in (0.0, -0.06, -0.12)]
    print(f"{n:>11}   {row[0]:>10.1%}   {row[1]:>9.1%}   {row[2]:>13.1%}")
```

```text
live trades   as promised   edge gone   edge reversed
         10         4.5%        8.2%           16.8%
         20         3.5%        9.0%           20.2%
         30         4.8%       12.8%           27.5%
         50         2.0%       10.8%           38.8%
        100         2.2%       18.8%           60.8%
        200         3.8%       29.2%           81.2%
        400         1.5%       37.2%           91.5%
```

Read it both ways. **False alarms stay low** (a few percent) at every size. **Power is modest**: with 100
live trades the check notices a *completely vanished* edge about one run in five, and 400 trades only get
that to about 37%, because per-trade PnL is very noisy (here its standard deviation is several times its
mean). That is the arithmetic of the quantity, not a defect of the threshold: lowering `z_warn` buys
power at the price of false alarms. A quiet PnL check is therefore **weak evidence that the edge is
intact**. Calibration and drift are the earlier warnings; PnL is the confirmation.

**How large a calibration error does the Brier check see?** The defaults (+0.02 Brier to warn) correspond
to forecasts that are off by roughly 0.15 in probability. Same setup, 300 simulated champions per cell,
share of runs that warn or alert:

```python
from pathlib import Path

import numpy as np

from pmlab import (
    BacktestExpectation,
    CalibrationThresholds,
    ChampionManifest,
    HoldoutGateResult,
    SegmentGateResult,
    WatchdogConfig,
    health_report,
)


def draw(rng, n, miscalibration):
    """Squared errors of n forecasts whose probabilities are off by `miscalibration`."""
    p_true = rng.uniform(0.35, 0.75, n)
    occurred = rng.random(n) < p_true
    forecast = np.clip(p_true + miscalibration, 0.0, 1.0)
    return forecast, occurred


def verdicts(miscalibration, n_live, sims=300):
    rng = np.random.default_rng(2000 + n_live)
    cfg = WatchdogConfig(calibration=CalibrationThresholds(min_trades=2))
    counts = {"ok": 0, "warn": 0, "alert": 0}
    for _ in range(sims):
        f, y = draw(rng, 400, 0.0)
        terms = (f - y) ** 2
        expectation = BacktestExpectation(
            400, 1.0, 0.1, 0.5, float(terms.mean()), float(terms.std(ddof=1))
        )
        champion = ChampionManifest(
            "c", Path("x"), None,
            HoldoutGateResult("GO", [SegmentGateResult("s", 400, 1.0, True, "ok")], 1.0, 400),
            "2026-01-01T00:00:00+00:00", "f", expectation=expectation,
        )
        f, y = draw(rng, n_live, miscalibration)
        trades = [
            {"recorded_at": "2026-02-01T00:00:00+00:00", "target_date": "2026-02-02",
             "city_or_segment": "s", "outcome": "won" if won else "lost",
             "model_prob": float(p), "realized_pnl": 0.0}
            for p, won in zip(f, y, strict=True)
        ]
        report = health_report(champion=champion, trades=trades, drift=None,
                               as_of="2026-03-01T00:00:00Z", config=cfg, checks=("calibration",))
        counts[report.status] += 1
    return counts


print("forecasts off by   n=30      n=100     n=400      (share of runs that warn or alert)")
for delta in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30):
    row = []
    for n in (30, 100, 400):
        c = verdicts(delta, n)
        row.append((c["warn"] + c["alert"]) / sum(c.values()))
    print(f"{delta:>+16.2f}   {row[0]:>6.1%}   {row[1]:>7.1%}   {row[2]:>7.1%}")
```

```text
forecasts off by   n=30      n=100     n=400      (share of runs that warn or alert)
           +0.00     2.0%      2.0%      0.0%
           +0.05     1.3%      2.7%      1.0%
           +0.10     2.0%      6.7%     13.0%
           +0.15     5.3%     14.3%     43.7%
           +0.20     8.0%     28.7%     78.0%
           +0.30    21.0%     73.7%    100.0%
```

A model whose probabilities are off by 0.05 to 0.10 is miscalibrated, but not by enough, in squared error,
for this check to notice (the Brier term is the *square* of the error). If you need to catch smaller
miscalibration, lower `brier_warn_abs` and read `reliability_live` in the JSON; both need more trades.

### Limits

- **Aggregate, not per segment.** The checks pool every allowed segment. A segment can be broken while
  the aggregate looks fine, which is the failure [Live calibration](live-calibration.md) warns about, and
  the backtest expectation is not recorded per segment. `n_by_segment` is in the metrics so you can see
  how thin a segment is; `CalibrationTracker.by_segment()` gives the per-segment numbers.
- **The backtest mean is optimistic.** A champion is chosen *because* its backtest looked good (a winner's
  curse), so its live mean tends to fall below the promise even when nothing is wrong. The warn level of
  "half of what was promised" leaves room for that: at the sample sizes in the simulation above a
  shortfall of that size cannot be told apart from noise, and is correctly not flagged.
- **Trades are treated as independent.** They are not quite: markets that resolve together, or a regime,
  correlate them, which makes the true noise larger than the test assumes.
- **Skew needs volume.** A payoff that wins small often and loses big rarely (or the reverse) has a
  skewed mean, and a normal approximation is shakiest at small n. `min_trades = 30` is the usual floor,
  not a guarantee: the simulation above shows no clear trend in false alarms between 10 and 400 trades
  (each cell has about a percentage point of sampling error), but a more skewed payoff than this one
  needs more. Raise `min_trades` for strongly skewed strategies.
- **Calibrate like with like.** The expectation's Brier is scored on the backtest's raw `predicted_prob`.
  If your live `model_prob` is passed through a calibrator, or your live selection rule differs from the
  backtest's, the two Briers are not on the same footing.
- **Depth-sensitive costs.** PnL is compared per unit stake. If your `CostModel` has a depth term, a live
  stake that differs from the backtest's pays a different price, and per-stake PnL is not exactly
  stake-free.
- **Drift is not degradation.** An alert says the inputs moved, not that the model got worse. Prediction
  drift compares the model on the training panel (in-sample) with the live window, so a flexible, overfit
  model can look different on its own training data; pass `--no-prediction`, or a holdout panel as
  `--reference`, if you see that.
- **Truth lags.** Settled trades are the only evidence on money at risk, and they arrive days or weeks after
  the forecast. Drift is visible today; calibration and PnL are not.
- **Reports and exits; does not act.** Pausing trading, retraining or paging someone is the caller's job.

---

## Running it

- **Paths.** Without `--champion` / `--trades`, paths resolve like `pmlab status`: from
  `--artifacts-dir` / `PMLAB_ARTIFACTS_DIR` and `--workspace` / `PMLAB_WORKSPACE`. Under
  `scripts/pmlab-workspace` the artifacts directory is already the workspace's, and the shared settings
  append `workspaces/<name>` again, so the default trade-log path comes out wrong there; pass `--trades`
  explicitly (the failure is the real output of the copied script; the fix is the second command):

```text
$ scripts/pmlab-workspace paper_healthy pmlab monitor --skip drift --as-of 2026-12-31T00:00:00Z
pmlab monitor: error: trade log not found: /tmp/pmlab-e2e-drift/fake_repo/artifacts/workspaces/paper_healthy/workspaces/ops_daily/signals/v2/forward_paper_trades.json
[exit code: 64]

$ scripts/pmlab-workspace paper_healthy pmlab monitor --skip drift --as-of 2026-12-31T00:00:00Z --trades artifacts/workspaces/paper_healthy/signals/v2/forward_paper_trades.json -v
pmlab monitor: OK  (champion e2e-lgbm, as of 2026-12-31T00:00:00Z)
  calibration ok            Brier 0.187 vs backtest 0.205 (-0.019 abs, -9.0% rel, z=-0.9, n=160); within thresholds
  pnl         ok            mean PnL/stake +0.601 vs backtest +0.351 (shortfall -71%, z=-0.8, n=160); within thresholds
  skipped: drift
[exit code: 0]
```

- **The clock.** `--as-of` defaults to now (UTC) and is the only place the wall clock is read. Pass it
  explicitly to replay an earlier moment; everything is point-in-time as of it. It must carry a UTC
  offset (`2026-03-01T00:00:00Z`, `2026-03-01T00:00:00+01:00`): `2026-03-01` or `2026-03-01T00:00:00` is
  refused with exit 64, because on a cron box in another zone that would be a different instant and
  the report would still look right. The same goes for `--window-start` and the `as_of` argument of
  `health_report`. (The timestamps *inside* the trade log are read leniently: a naive `recorded_at`
  is taken as UTC.)
- **Cron.** Give it absolute paths and a `PATH` that finds `pmlab`; log the JSON for a record
  (`--json /var/log/pmlab/health.json`); and remember that a `%` in a crontab line must be escaped.
- **Bad input is never green.** A missing champion, a corrupt trade log, a typo'd snapshot directory (it is
  not created), an unwritable `--json`, a `--as-of` with no UTC offset, a `champion.json` whose
  expectation holds a `NaN`: all exit 64 with one line on stderr.

```text
$ pmlab monitor -a /nonexistent -w paper_healthy --reference data/panel.parquet --as-of 2026-12-31T00:00:00Z --snapshots data/snaps/paper_healthy
pmlab monitor: error: champion not found: /nonexistent/public_models/champion.json
[exit code: 64]

$ pmlab monitor -a artifacts_bad -w paper_healthy --reference data/panel.parquet --as-of 2026-12-31T00:00:00Z --snapshots data/snaps/paper_healthy
pmlab monitor: error: champion.json is unreadable (artifacts_bad/public_models/champion.json): JSONDecodeError: Expecting value: line 1 column 16 (char 15)
[exit code: 64]

$ pmlab monitor -a artifacts -w paper_healthy --reference data/panel.parquet --as-of 2026-12-31T00:00:00Z --snapshots data/snaps/paper_helthy
pmlab monitor: error: snapshot directory not found: data/snaps/paper_helthy
[exit code: 64]

$ pmlab monitor -a artifacts -w paper_healthy --skip drift --as-of 2026-12-31T00:00:00
pmlab monitor: error: --as-of '2026-12-31T00:00:00' has no UTC offset, so it could be anyone's local time; give one explicitly, e.g. 2026-03-01T00:00:00Z or 2026-03-01T00:00:00+01:00
[exit code: 64]

$ pmlab monitor -a artifacts_nan -w paper_degraded --skip drift --as-of 2026-12-31T00:00:00Z
pmlab monitor: error: champion.json is unreadable (artifacts_nan/public_models/champion.json): ValueError: std_pnl_per_stake must be a finite number, got nan
[exit code: 64]

$ pmlab monitor --bogus
Usage: pmlab monitor [OPTIONS]
Try 'pmlab monitor --help' for help.
╭─ Error ──────────────────────────────────────────────────────────────────────╮
│ No such option: --bogus (Possible options: --verbose)                        │
╰──────────────────────────────────────────────────────────────────────────────╯
[exit code: 2]
```
