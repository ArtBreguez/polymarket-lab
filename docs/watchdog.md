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
workspace built through the public API and the installed wheel: one champion, and six
live worlds that each differ from training in exactly one respect (the guide builds the world in
[the whole chain](#the-whole-chain-from-the-cli)).

```text
$ pmlab monitor -a artifacts -w paper_healthy --reference data/panel.parquet --snapshots data/snaps/paper_healthy --family e2e_family --as-of 2028-12-31T00:00:00Z
[exit code: 0] [bytes printed: 0]
```

Problems come out as a few lines. A feature moved (only that feature is named, and the
model's own prediction, which depends on it; the one that did not move is not):

```text
$ pmlab monitor -a artifacts -w paper_drifted --reference data/panel.parquet --snapshots data/snaps/paper_drifted --family e2e_family --as-of 2028-12-31T00:00:00Z
pmlab monitor: ALERT  (champion e2e-lgbm, as of 2028-12-31T00:00:00Z)
  drift       alert         feature_x1: alert (psi 2.820 >= 0.25, kl 1.627 >= 0.125)
                            prediction: alert (psi 2.217 >= 0.25, kl 1.134 >= 0.125)
[exit code: 20]
```

Every pick the model makes now wins 30 points less often than it said. Calibration and PnL both
say so, and name where: two segments, both sides, and the book as a whole, with the sequential test on top
(the lines are long because each part states what it needed to be flagged; segment B, at a shortfall of
exactly 100%, is only a warn):

```text
$ pmlab monitor -a artifacts -w paper_degraded --reference data/panel.parquet --snapshots data/snaps/paper_degraded --family e2e_family --as-of 2028-12-31T00:00:00Z
pmlab monitor: ALERT  (champion e2e-lgbm, as of 2028-12-31T00:00:00Z)
  calibration alert         segment C: Brier 0.315 vs backtest 0.220 (+0.095 abs, +43.2% rel, z=3.9, Holm z=3.6, n=154); alert needs >= +0.05 abs, >= +25% rel, z >= 3
                            NO side: Brier 0.290 vs backtest 0.219 (+0.071 abs, +32.6% rel, z=3.8, Holm z=3.6, n=238); alert needs >= +0.05 abs, >= +25% rel, z >= 3
                            overall: Brier 0.282 vs backtest 0.228 (+0.054 abs, +23.7% rel, z=3.9, n=458); warn needs >= +0.02 abs, >= +10% rel, z >= 2
  pnl         alert         segment A: mean PnL/stake -0.113 vs backtest +1.358 (shortfall 108%, z=4.1, Holm z=3.9, n=161); alert needs shortfall >= 100%, z >= 3 (could only detect a shortfall >= 87% of the backtest edge at n=161)
                            segment C: mean PnL/stake -0.530 vs backtest +1.457 (shortfall 136%, z=4.8, Holm z=4.6, n=154); alert needs shortfall >= 100%, z >= 3 (could only detect a shortfall >= 92% of the backtest edge at n=154)
                            YES side: mean PnL/stake -0.181 vs backtest +1.039 (shortfall 117%, z=5.0, Holm z=4.9, n=220); alert needs shortfall >= 100%, z >= 3 (could only detect a shortfall >= 74% of the backtest edge at n=220)
                            NO side: mean PnL/stake -0.252 vs backtest +1.396 (shortfall 118%, z=5.1, Holm z=4.9, n=238); alert needs shortfall >= 100%, z >= 3 (could only detect a shortfall >= 73% of the backtest edge at n=238)
                            overall: mean PnL/stake -0.218 vs backtest +1.225 (shortfall 118%, z=7.0, n=458); alert needs shortfall >= 100%, z >= 3 (could only detect a shortfall >= 50% of the backtest edge at n=458)
                            sequential: CUSUM at 91.1, above 30 (first crossed 30 at trade 132 of 458, peak 91.1); alert while it stays above
                            segment B: mean PnL/stake +0.001 vs backtest +0.824 (shortfall 100%, z=3.0, n=143); warn needs shortfall >= 50%, z >= 2 (could only detect a shortfall >= 108% of the backtest edge at n=143)
[exit code: 20]
```

Run it from cron, and let the exit code do the work:

```cron
*/30 * * * * cd /srv/pmlab && .venv/bin/pmlab monitor --champion ... --trades ... --reference ... --snapshots ... || /usr/local/bin/notify "pmlab monitor exited $?"
```

```text
$ pmlab monitor -a artifacts -w paper_degraded --reference data/panel.parquet --snapshots data/snaps/paper_degraded --family e2e_family --as-of 2028-12-31T00:00:00Z || echo "NOTIFY: exit $rc -- $first_line"
NOTIFY: exit 20 -- pmlab monitor: ALERT  (champion e2e-lgbm, as of 2028-12-31T00:00:00Z)
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
  pnl         ok            mean PnL/stake +0.162 vs backtest +0.161 (shortfall -1%, z=0.0, n=120); within thresholds (could only detect a shortfall >= 217% of the backtest edge at n=120)
                            sequential: CUSUM 0.0 of 30 (peak 0.0, n=120); within thresholds
  skipped: drift
--- edge gone: alert, exit code 20
pmlab monitor: ALERT  (champion demo, as of 2026-03-01T00:00:00Z)
  calibration ok            Brier 0.246 vs backtest 0.244 (+0.001 abs, +0.5% rel, z=0.1, n=120); within thresholds
  pnl         alert         mean PnL/stake -0.207 vs backtest +0.161 (shortfall 229%, z=3.2, n=120); alert needs shortfall >= 100%, z >= 3 (could only detect a shortfall >= 202% of the backtest edge at n=120)
                            sequential: CUSUM 26.6 of 30 (peak 26.6, n=120); within thresholds
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

Inside a check the same ranking applies to its parts (the book, each side, each segment,
the sequential test), with one difference: **a part that could not be measured is listed
as not assessed and does not make the check `insufficient` when the book itself could be
measured** (see [the verdict rule](#what-decides-a-checks-verdict)). A check is
`insufficient` only when nothing in it could be measured.

---

## What a run does

1. **Loads `champion.json`** and treats it as the authority: its allowed segments
   (`"all"` meaning every segment), its publication time, its cost model, the trade policy
   it was backtested under and its backtest expectation (for the book, for each side and
   for each segment). It is never substituted by a retrain summary, which is overwritten on
   every retrain. A `champion.json` that records a `NO_GO` gate is refused (exit 64): such a
   file cannot have been produced by `publish()`.
2. **Selects the evidence** from the trade log (below).
3. **Builds the drift report** from your reference panel and the live snapshot window,
   point-in-time, if you gave it one.
4. **Runs the three checks** and takes the worst. Calibration and PnL each judge the book,
   each side that traded and each segment; PnL also runs the sequential test.
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
  `champion.json`), the same filter `PaperBroker` applies when recording. A champion
  promoted on an **aggregate gate** (`HoldoutGateResult.evaluate(trades)` without
  `required_segments`, which grades the whole log as one segment called `"all"`) allows
  `{"all"}`, and `"all"` on its own means *every segment*: it never made a per-segment claim. Beside
  other names it is just a segment called "all", so a per-segment gate that graded such a segment does not
  widen to the ones it never graded (and a per-segment gate whose only segment is called "all" cannot be
  told from an aggregate one, so that name is reserved). An **empty** allowlist still means none;
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
$ pmlab monitor -a artifacts -w paper_stray --skip drift --as-of 2028-12-31T00:00:00Z
pmlab monitor: INSUFFICIENT  (champion e2e-lgbm, as of 2028-12-31T00:00:00Z)
  calibration insufficient  0 of 458 settled trades are in the champion's allowed segments ['A', 'B', 'C'], need >= 30 (excluded: 458 outside the champion's allowed segments)
  pnl         insufficient  0 of 458 settled trades are in the champion's allowed segments ['A', 'B', 'C'], need >= 30 (excluded: 458 outside the champion's allowed segments)
  skipped: drift
[exit code: 30]
```

The report's metrics list `allowed_segments`. A champion that allows **no** segment at all is a different
thing: an empty allowlist means *none*, exactly as in `PaperBroker`, so it could never have recorded a
trade. `publish()` cannot produce it, so such a `champion.json` is malformed and exits 64.

**A log settled by an older release.** The healthy workspace of the guide's world with its trade log
rewritten the way 0.9.0 left it: the `settled_at`, `settled_basis` and `settlement_version` keys removed, a
YES trade entered at the quote although it paid a slipped fill, and a NO trade booked as a short YES at the NO
price. The PnL check leaves both kinds out and says so, because their `realized_pnl` is not the number the
fill implies; calibration does not care how the PnL was settled and is unaffected. `pmlab resettle`
([below](#re-settling-an-old-log)) is what repairs such a log:

```text
$ pmlab monitor -a artifacts -w paper_legacy --reference data/panel.parquet --snapshots data/snaps/paper_legacy --family e2e_family --as-of 2028-12-31T00:00:00Z --verbose
pmlab monitor: INSUFFICIENT  (champion e2e-lgbm, as of 2028-12-31T00:00:00Z)
  drift       ok            2 feature(s) and the prediction within thresholds (max psi 0.040, 500 live rows)
  calibration ok            overall: Brier 0.211 vs backtest 0.228 (-0.018 abs, -7.8% rel, z=-1.4, n=458); within thresholds
                            note: 458 of 458 in-scope trades have no settled_at (settled by an older release), so they are judged by target_date and may include truth that arrived after as_of
  pnl         insufficient  0 scoreable settled trades, need >= 30 (excluded: 238 NO trades settled by a release that booked them wrongly (run `pmlab resettle`); 220 settled at the quote by an older release although they paid a different fill)
                            note: 238 NO trades were settled by a release that booked them as a short YES, so their realized_pnl is wrong and they were left out; run `pmlab resettle --trades PATH` to recompute them
                            note: 458 of 458 in-scope trades have no settled_at (settled by an older release), so they are judged by target_date and may include truth that arrived after as_of
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

- trades on a **side the champion has no expectation for** (`n_excluded_no_side`, and
  `n_excluded_yes_side` for the mirror case; the side is read as settlement reads it, so
  anything whose `direction` is not `"yes"` is NO). A champion promoted on a live-policy
  backtest records one per side, and its NO trades are judged against its NO numbers. One
  published before the split existed (or promoted on `--policy top-bin-yes`, or whose NO
  side traded too rarely in the backtest) only ever bought YES as far as its record goes,
  and a NO trade is not a draw from the distribution being tested against: it is left out
  and counted, never judged by a promise made about something else. Calibration follows
  the same rule for a champion that recorded a split; for one that recorded none it
  keeps scoring every trade against the single Brier it has.
- **NO trades booked by an older release** (`n_excluded_unfixed_no`). Before 0.9.1 a NO
  trade was settled as a short YES at the NO price, so its `realized_pnl` is wrong in both
  directions (a lost NO bet booked a fraction of its stake). A NO trade without
  `settlement_version` is not evidence about the NO strategy until [`pmlab
  resettle`](#re-settling-an-old-log) has recomputed it, and the report
  says so in a note. Calibration does not care how the PnL was settled and counts them.
- trades **settled at the quote although they paid a different fill**
  (`n_excluded_legacy_settlement`). Releases up to 0.9.0 entered every settlement at the
  quote, so such a trade's `realized_pnl` carries none of the slippage its `fill_price`
  says it paid, and would otherwise be judged as cost-matched. Settlement now writes
  `settled_basis` on every trade; one without it is kept only if its `fill_price` equals
  the quote of the token it bought (the YES price, or `1 - gamma_price` for a NO trade),
  or none was recorded, because then both arithmetics enter at the same price. It is
  excluded rather than recomputed here so the number the watchdog uses is always the one
  in your log; `pmlab resettle` is the tool that recomputes it.
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
`champion.json`: the backtest's Brier for the cells the live trades fell in, weighted by how many
fell in each (see [what the book is held to](#what-the-book-is-held-to)). A level (warn or alert) is
reached only if the live Brier is worse by *at least the absolute margin* **and** *the relative
margin* **and** the gap clears the noise (`z`, below). Absolute alone over-fires on a tiny Brier;
relative alone over-fires on a large one.

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
money* (see [Live calibration](live-calibration.md)). That makes a NO trade's forecast the
probability that the label occurs, whichever side was bought: a NO bet on a label that occurred
is a miss for the forecast however little the bet paid, and `model_prob` is never flipped to the
NO token's `1 - p`. Trades without a `model_prob` are not
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
mean PnL/stake -0.218 vs backtest +1.225 (shortfall 118%, z=7.0, n=458); alert needs shortfall >= 100%, z >= 3 (could only detect a shortfall >= 50% of the backtest edge at n=458)
```

The test is **one-sided**: only a shortfall can warn. Beating the backtest is never a problem, and a
watchdog that pages on good luck gets muted. A non-positive backtest mean cannot anchor a shortfall
(there is no "half of what was promised" of a promise of nothing), so that case is `insufficient`
(`invalid_expectation`) **on every run, permanently**: exit 30 does not clear by itself and the
reason says so, `... this check will stay insufficient on every run until the champion is
re-published on a backtest that earned money`. Drift and calibration are still judged. It is kept
rather than waved through because a PnL check that quietly never runs must not look like health.
(A side or a segment whose own promise is not positive is *not assessed* instead; the book is
still judged. See [the verdict rule](#what-decides-a-checks-verdict).)

The parenthesis at the end is [what a quiet run is worth](#what-a-quiet-run-is-worth): the smallest
shortfall this comparison could have shown at this size. And alongside the fixed-sample test runs a
[sequential test](#the-sequential-test), which is the other way a PnL alert is raised.

---

## Judging the book, each side and each segment

The book is not the only thing a champion can fail at. A segment can bleed while the others hide it in
the aggregate, and a NO strategy is a different strategy from a YES one. So calibration and PnL are
each judged on several **scopes**:

| scope | held to | named in the output as |
|---|---|---|
| the book | the backtest's numbers for the cells its trades fell in | `overall` |
| each side that traded (YES, NO), if both did | the same cells, for that side's trades | `YES side` / `NO side` |
| each segment that traded | that segment's own record, with its own mix of sides | `segment <name>` |
| (PnL only) the sequence of trades | each trade's cell, in settlement order | `sequential` |

### What the book is held to

A trade is held to the backtest's numbers for the **cell it fell in**: its segment and its side if the
champion recorded that, else the backtest's record for its side, else it has no promise, is left out
and is counted (`n_excluded_no_side`). The reference for the book is the average of those cells weighted
by how many live trades fell in each:

```
reference = sum over cells c of  (live trades in c / live trades)  *  backtest mean of c
variance  = sum over cells c of  (live trades in c / live trades)^2  *  backtest std of c ^2 / backtest n of c
```

In one line: pooling the backtest would be right only if the live book traded YES and NO, and each
segment, in the proportions the backtest did, so the reference is the backtest's mean for each cell the
live trades fell in, weighted by how many fell in each, and the question it asks is whether every part
earned what it promised, not whether the mix stayed put.

Two failures the pooled number has, both from the same cause and both found by simulation and fixed:

- *A false alarm.* Segment A promised +0.35, segment B +0.05, 200 backtest trades each, so the pooled
  mean is +0.20. A live book of 100 A trades and 900 B trades, each segment earning exactly what it
  promised, averages +0.08: a 60% "shortfall" against the pooled number, at z = 2.4.
- *A false green.* A champion allowed only in segment A, promised +0.40, with a gate-rejected segment B
  that lost money in the backtest dragging the pooled mean to +0.15. Live A earning +0.20, half of what
  A promised, reads as 33% *better* than the pooled promise.

Both are tests in `tests/monitoring/test_watchdog_strata.py`.

A champion published before the split existed has one record for the whole backtest, which describes the
historical backtest, and that only bought YES. It behaves as it always did: its NO trades are left out of
the PnL comparison and counted, and the report says so in a note rather than reading ok on the YES trades
alone. A champion with no per-segment record is judged as a book and by side, and
`segments_recorded: false` says so in the metrics.

### Many scopes, one false-alarm rate

Each side and each segment is a hypothesis test, and testing many of them at once raises the chance
that one alarms by luck: ten segments judged on their own, each at the 2.3% a `z` of 2 allows, is a
false alarm in about a quarter of runs. The sides are a family, and so are the segments. Within a family
the one-sided p-values are corrected with **Holm-Bonferroni** (`pmlab.monitoring.stats.holm_adjust`): sort
them, multiply the smallest by the number of tests, the next by one fewer, and so on, keep the running
maximum, cap at 1. The adjusted p-value is turned back into a `z`, so the same `z_warn` / `z_alert` apply
and the effect-size thresholds are unchanged; with a family of one nothing changes. Holm is uniformly
more powerful than plain Bonferroni and valid under any dependence between the tests. The output shows
both `z`: `z=3.9, Holm z=3.6`.

```text
False alarms from the per-segment checks when nothing has changed (400 simulated champions per row)
each segment has 60 live trades; share of champions with at least one segment warned or alerted

segments     each judged alone   with Holm
2                         8.0%        6.0%
5                        14.0%        3.2%
10                       25.2%        4.0%
20                       38.2%        2.8%
```

Only scopes that were actually tested count towards the size of the family: a segment that is not
assessed (below) does not make the others harder to flag.

The book, the sides, the segments and the sequential test are separate families and the verdict is the
worst of them, so the false-alarm rate of the whole check is higher than any one part's. It is measured
below under "all the parts together" in [the statistics](#the-statistics-and-their-limits).

### What decides a check's verdict

A check's status is the **worst of its conclusive results** (alert > warn > ok). A result that cannot be
measured is *not assessed*: a side or a segment with fewer than `min_trades` live trades, a segment the
backtest recorded no expectation for (it was traded too rarely, or never), a side with no positive
promise, a non-finite statistic. It is listed (`not assessed: segment C (10 scoreable settled trades,
need >= 30)`), it appears in the JSON with status `insufficient` and a `cause`, and it **does not make a
check that could be measured insufficient**. A check is `insufficient` only when nothing in it could be
measured (the book itself has too few trades, no expectation, an unusable one).

Why: with many small segments the alternative would leave the exit code at 30 for good, and the book
already contains their trades. The price is a blind spot, which is why it is named in the report (shown
with `--verbose` and beside any problem, never silent) and counted in `n_by_segment`. Where a more
cautious reading is wanted, `--insufficient warn` still applies to a whole check that could not be
measured.

At equal severity the more specific result leads the output (a segment, then a side, then the book): it
names what is wrong, and the `overall:` line only says that the book as a whole is too.

### The sequential test

The fixed-sample test looks at the pooled mean. Edges do not usually vanish on the day a champion is
published: they fade, or stop working after a regime change, and a pooled mean dilutes the recent trades
with the healthy ones before them. The PnL check therefore also runs a **one-sided CUSUM** on the
sequence of trades:

```
x_i = (promised mean - return_i) / promised std          how far trade i fell short, in units of its own spread
k_i = a * (promised mean / promised std)  +  b / sqrt(n)  what it may fall short by without counting
S_i = max(0, S_{i-1} + x_i - k_i)                        the statistic; alert while S_i >= h
```

with `a = cusum_allowance = 0.5` (half the promised edge, the textbook reference value for an edge that
has vanished), `b = cusum_backtest_se = 0.5` standard errors of the promise (`1/sqrt(n)` of the backtest
trades behind it: the backtest mean is an estimate, and a CUSUM that ignored its error drifts to a
standing alert on any champion whose backtest flattered it) and `h = cusum_threshold = 30`.

- **Order.** Trades are processed in the order their truth arrived: `settled_at`, then the trade's
  identity (segment, target date, horizon, recording time, label, side). A trade from before 0.9.1 has no
  `settled_at` and is placed by its `target_date`. Trades that tie on both are one step, so the result
  cannot depend on the order the log happens to list them in.
- **Each trade is standardized by its own record**: its cell, else the book's record for its side, but
  only if that rests on at least `cusum_min_backtest_trades` (100) backtest trades; otherwise the trade is
  skipped and the reason says how many. So a book that trades more NO, or more of a weaker segment, than
  the backtest did is not mistaken for a decline.
- **No single trade may add more than `cusum_increment_cap` = 5.** A cell whose backtest almost never
  lost has a tiny spread (+0.0279 +/- 0.002 over 100 trades), so one ordinary loss of the whole stake is some
  500 of its standard deviations: uncapped, that one trade put the statistic at 508 and raised an alert at
  trade 61 (found in review; `TestOneTradeCannotTripTheSequentialTest` pins it). The statistic alerts at
  30, so a cap of 5 means no alert from fewer than six trades in a row, however small the spread. The cap
  is on the increment (shortfall minus allowance), so a deterministic cell whose edge vanished still
  reaches the threshold in six trades. It never binds in a cell with a realistic spread, where even a total
  loss is a standard deviation or two, so the tables below are the same with or without it.
- **It alerts while the statistic is at or above the threshold**, and clears when the bad run ends. It
  does not stay an alert for ever once it has crossed: the chance that a healthy book has crossed at least
  once grows with its length, so a remembered signal would turn every long-lived champion into a standing
  alert (an earlier version did exactly that, and was caught by review). The first crossing and the peak
  stay in the report: `sequential: CUSUM at 45.7, above 30 (first crossed 30 at trade 417 of 458, peak 45.7)`.
- It reports and alerts for the book only; the sides and the segments are covered by the fixed test, and a
  CUSUM per segment would multiply the false alarms.
- `sequential = false` switches it off.

The world built for it: the edge is intact for 430 of 600 days, then the traded token wins 60 points less
often. Over the whole book the shortfall is 34% at z = 1.7, which the fixed test rightly ignores; the
sequential test has the run of bad trades isolated and alerts, naming where it first crossed:

```text
$ pmlab monitor -a artifacts -w paper_decay --reference data/panel.parquet --snapshots data/snaps/paper_decay --family e2e_family --as-of 2028-12-31T00:00:00Z
pmlab monitor: ALERT  (champion e2e-lgbm, as of 2028-12-31T00:00:00Z)
  pnl         alert         sequential: CUSUM at 45.7, above 30 (first crossed 30 at trade 417 of 458, peak 45.7); alert while it stays above
                            overall: mean PnL/stake +0.814 vs backtest +1.225 (shortfall 34%, z=1.7, n=458); within thresholds (could only detect a shortfall >= 56% of the backtest edge at n=458)
[exit code: 20]
```

The PnL check alerts if either the fixed test or the sequential test fires. What it buys, from
simulations (every number below is regenerated by `scripts/watchdog_power.py`; a 400-trade backtest unless
the table says otherwise):

```text
The sequential test alone (allowance 0.5 x the promised edge + 0.5 standard errors of the promise, threshold 30, a 400-trade backtest, 10000 simulated champions per row)

share of champions whose alert is on at the run after N live trades
live edge                N=100   N=200   N=400   N=800  N=1500  N=3000
as promised               0.0%    0.2%    0.7%    1.1%    1.7%    1.9%
halved                    0.1%    1.7%    7.5%   16.0%   23.6%   29.2%
vanished                  0.6%    9.4%   33.6%   61.7%   79.3%   89.2%
reversed                  9.4%   60.6%   95.5%   99.8%  100.0%  100.0%
halved after 150          0.0%    0.5%    5.2%   14.6%   23.2%   29.5%
vanished after 150        0.0%    0.8%   20.8%   56.4%   77.5%   88.6%

share that crossed the threshold at least once within the first N live trades
live edge                N=100   N=200   N=400   N=800  N=1500  N=3000
as promised               0.0%    0.4%    2.3%    6.5%   13.5%   25.1%
halved                    0.2%    3.1%   14.8%   35.5%   57.3%   78.8%
vanished                  0.9%   14.2%   47.8%   79.9%   94.4%   99.2%
reversed                 11.8%   68.2%   97.4%  100.0%  100.0%  100.0%
halved after 150          0.0%    0.9%   10.1%   32.0%   55.1%   77.7%
vanished after 150        0.0%    1.2%   30.6%   73.9%   93.5%   99.2%

median trade at which the threshold was first crossed, among those that crossed
  as promised             1385
  halved                   904
  vanished                 415
  reversed                 161
  halved after 150         972
  vanished after 150       524
```

Read the first block as what a cron job sees: the share of champions whose alert is on at the run after N
trades. Under "as promised" it is at most a couple of percent however long the book is, because a false
alert clears; the second block is the chance that *some* run along the way saw one, which only grows with
the length of the book and is the number to quote if you ask "how often will I be paged for nothing". The
sequential test is slow by construction: it is looking for a shift of a fraction of the noise, and
`median` is the trade at which it first crossed.

The backtest's size matters, because the promise is an estimate:

```text
The sequential test when the backtest is short (the promised mean is an estimate)
5000 simulated champions per row

backtest trades     crossed within 400  alert at 3000  vanished: alert at 400   at 800
30                                4.2%           6.9%                   18.4%    28.9%
50                                3.1%           5.4%                   20.8%    33.9%
100                               2.7%           3.8%                   24.5%    43.6%
200                               2.4%           2.2%                   28.2%    52.7%
400                               2.6%           2.1%                   33.3%    62.2%
1000                              2.9%           1.9%                   39.3%    70.7%
```

Hence the floor of 100 backtest trades per record, and the `b` term in the allowance.

And what the two tests together do, run once on a book of N trades, 400 champions per cell (the first
block is the fixed test alone, the second adds the sequential test):

```text
How often the PnL check warns or alerts (400 simulated champions per cell)
                   fixed test only                    fixed test or sequential test
live trades     as promised       halved     vanished     reversed   |  as promised       halved     vanished     reversed
         30            3.0%         5.2%        13.0%        27.0%   |         3.0%         5.2%        13.0%        27.0%
         50            3.0%         5.2%        18.2%        37.8%   |         3.0%         5.2%        18.2%        37.8%
        100            2.5%         8.8%        24.8%        56.5%   |         2.5%         8.8%        24.8%        56.5%
        200            2.5%        12.0%        34.0%        73.5%   |         2.5%        12.0%        34.0%        74.0%
        400            2.5%        14.8%        41.5%        92.2%   |         2.8%        15.5%        44.5%        95.2%
        800            1.5%        17.0%        50.7%        96.5%   |         2.5%        22.5%        66.8%        98.8%
```

### Near-certain cells

The cap exists for cells like this one, and so does a limit of the sequential test. Tokens priced at 0.96-0.98
that win all but a handful of times; a 100-trade backtest that promised a 0.5% chance of losing the stake, and
that, in most simulated champions, saw no loss at all and so recorded a spread of almost nothing:

```text
Near-certain cells (5000 simulated champions per row)
tokens priced 0.96-0.98, a 100-trade backtest that promised a 0.5% chance
of losing the stake. Most backtests saw no loss at all, so their spread is tiny and
one loss of the stake is hundreds of standard deviations. Share of champions whose
alert is on at the run after N live trades (the sequential test with and without the
cap on what one trade may add, the fixed test, and the check as shipped)

live book                       N   fixed test   sequential, no cap   sequential, capped   fixed or capped
healthy (loses 0.5%)           60         0.0%                15.2%                 0.0%              0.0%
healthy (loses 0.5%)          400         1.1%                18.1%                 0.0%              1.1%
edge gone (loses 3%)           60         7.0%                53.1%                 0.0%              7.0%
edge gone (loses 3%)          400        66.4%                80.9%                 0.2%             66.5%
collapsed (loses 10%)          60        74.4%                89.0%                 6.2%             74.4%
collapsed (loses 10%)         400       100.0%               100.0%                37.9%            100.0%
```

Uncapped, a healthy near-certain cell false-alarms in 15% of champions at 60 trades and 18% at 400, because
its first loss is hundreds of standard deviations; capped, in none. The price is that the sequential test has
almost no power there, since a loss adds at most 5 and every win takes back more than that (a cell that loses
3% instead of 0.5% is flagged by it in 0.2% of runs at 400 trades, one that loses 10% in 38%). Detection in such
a cell is the fixed test's job, which is why "fixed or capped" is, to a tenth of a point, the fixed column. A floor on the spread used for standardization was
considered and not added: it would let the sequential test see a collapse sooner, but the fixed test already
sees the collapse (last rows), and a floor is one more number to tune.

### What a quiet run is worth

A check that finds nothing is weak evidence of health when it had little power to find anything. Every
PnL result (the book, each side, each segment) therefore reports the **minimum detectable shortfall**: the
smallest shortfall, as a fraction of the backtest edge, that a test of this size would have flagged with
80% power at the configured significance level:

```
mde = (z_warn + z_power) * se / backtest_mean          z_power = 0.84 for 80% power; se is the test's own
```

`se` is the standard error the test itself used, `z_warn` is the `z` a warn must clear, and inside a
family of tests each must clear the level divided by the family's size, so a segment pays for being one of
several. The figure shown is never below `shortfall_warn`, because a smaller shortfall is not a finding
however clear it is (the unrounded figure is `mde_shortfall` in the JSON, the one shown
`mde_shortfall_effective`).

```text
mean PnL/stake +1.481 vs backtest +1.225 (shortfall -21%, z=-1.0, n=458); within thresholds (could only detect a shortfall >= 62% of the backtest edge at n=458)
```

That reads: nothing was found, and a shortfall under 62% of the backtest edge would mostly have gone
unnoticed at this size. It is in `--verbose`, in the JSON, and in the default output whenever a check is
not ok. `mde_power` sets the 80%.

---

## The backtest expectation

"Live PnL is 0.02 per trade" is neither good nor bad until you know how noisy per-trade PnL is. The
gate records only totals (`aggregate_pnl` over `aggregate_trades`), so the champion also records a
**`BacktestExpectation`**: the mean, standard deviation and count of per-stake PnL, plus the mean and
standard deviation of the per-trade squared forecast error (Brier), all over the backtest's *traded*
rows. Traded rows, not the whole panel, because live trades are a selected subset and a calibration
comparison is only fair between like subsets.

It records them for **the whole backtest, for each side (YES and NO), and for each segment with that
segment's own split by side**, and the **trade policy** the backtest ran. A side or a segment the backtest
traded fewer than 30 times is not given a record of its own (`min_group_trades`): a promise made on fewer
trades is not one a live book can be held to. A segment left out for that reason is written as `null`, so
"could not be promised anything" stays distinguishable from "nobody looked", and publishing a champion
marks every allowed segment the backtest never traded the same way. A side that holds *every* trade is
kept whatever its count, because it is the whole-backtest record.

```bash
pmlab backtest --plugin my_family --panel panel.parquet --flat-stake 1.0 --costs costs.json --output metrics.json
pmlab promote-champion model.pkl --gate-path gate.json --plugin my_family --expectation metrics.json
pmlab record-trades --plugin my_family
```

`pmlab backtest` simulates what `record-trades` trades. Per market it takes the (bin, side) with the best
after-fee edge, **YES or NO**, and trades it if that edge is at least `--min-edge` (default 0.20, the same
default as `record-trades`); a NO trade is a buy of the NO token at `1 - market_price`, priced by the same
cost model, fee and per-stake arithmetic as `PaperBroker` and `SettlementEngine` (a test replays a backtest's
signals through both and compares the PnL trade by trade). `--policy top-bin-yes` reproduces the historical
backtest: YES on the most probable bin of every market, no threshold. The library's `rolling_origin_eval`
keeps that historical rule as its default so numbers published before policies existed reproduce;
`rolling_origin_eval(..., policy=TradePolicy.live())` runs the live one.

`backtest --output` writes the expectation under an `expectation` key, the policy under `policy`, and the
cost model it ran under (`--costs`, else the default) under `costs`. `--expectation` reads either that file
or a bare expectation object, and `promote-champion` publishes the champion with the file's costs and
policy, so `champion.json` records the friction and the strategy the promise was made under;
`record-trades` then fills the paper trades under the champion's costs, takes the champion's `--min-edge` as
its default, skips signals on a side the policy does not trade, and records the best signal per market
(the backtest takes the best bin; the broker keeps the first signal it is given for a segment, date and
horizon, so they are ordered best first). That is what lets the PnL check report a known `cost_basis` instead
of `"unknown"`; see [Costs](costs.md#from-the-command-line).

What the backtest of the guide's world (below) recorded, from its `champion.json`:

```text
policy        {'selection': 'best_edge', 'directions': ['yes', 'no'], 'min_edge': 0.2}
book          {'n_trades': 391, 'mean_pnl_per_stake': 1.219, 'std_pnl_per_stake': 3.651, 'brier_score': 0.228}
side yes      {'n_trades': 192, 'mean_pnl_per_stake': 1.048, 'std_pnl_per_stake': 2.935, 'brier_score': 0.238}
side no       {'n_trades': 199, 'mean_pnl_per_stake': 1.385, 'std_pnl_per_stake': 4.231, 'brier_score': 0.219}
segment A     {'n_trades': 132, 'mean_pnl_per_stake': 1.37} sides {'yes': 85, 'no': 47}
segment B     {'n_trades': 121, 'mean_pnl_per_stake': 0.82} sides {'yes': 61, 'no': 60}
segment C     {'n_trades': 138, 'mean_pnl_per_stake': 1.425} sides {'yes': 46, 'no': 92}
```

From Python:

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
    "direction": rng.choice(["yes", "no"], 200),        # the side each trade bought
    "segment": "politics",
})
expectation = BacktestExpectation.from_trades(trades, flat_stake=2.0)
print({k: round(v, 4) if isinstance(v, float) else v for k, v in expectation.to_dict().items()
       if k in ("n_trades", "flat_stake", "mean_pnl_per_stake", "std_pnl_per_stake", "brier_score")})
print("sides recorded:", sorted(expectation.by_direction))
print("segments recorded:", {k: v.n_trades for k, v in expectation.by_segment.items()})

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
{'n_trades': 200, 'flat_stake': 2.0, 'mean_pnl_per_stake': 0.0669, 'std_pnl_per_stake': 0.4336, 'brier_score': 0.2509}
sides recorded: ['no', 'yes']
segments recorded: {'politics': 200}
champion.json key: True
old champion loads, expectation = None
```

A champion published before v0.9.1 has no expectation. It still loads, and the watchdog reports
**calibration and PnL as `insufficient` ("no backtest expectation recorded")**: not green and not a
crash. Drift needs no expectation and is unaffected. Re-publish the champion with an expectation to
turn the other two on:

```text
$ pmlab monitor -a artifacts_old -w paper_healthy --reference data/panel.parquet --snapshots data/snaps/paper_healthy --family e2e_family --as-of 2028-12-31T00:00:00Z
pmlab monitor: INSUFFICIENT  (champion e2e-lgbm, as of 2028-12-31T00:00:00Z)
  calibration insufficient  no backtest Brier expectation recorded in champion.json: re-publish the champion with a BacktestExpectation to enable this check
  pnl         insufficient  no backtest PnL expectation recorded in champion.json: re-publish the champion with a BacktestExpectation to enable this check
                            note: 238 NO trades are not judged: champion.json records no NO expectation (its backtest only buys YES); re-publish the champion on a backtest of the live policy to judge them
[exit code: 30]
```

An expectation recorded before the per-side and per-segment records existed (0.9.1 development builds)
loads with both as "not recorded": its NO trades are left out of the PnL comparison and counted, and it is
judged as one book.

A *present but unusable* record is different: it raises (exit 64), so it can never read as "unknown"
and quietly weaken the watchdog. `BacktestExpectation` refuses to be built, or loaded from
`champion.json`, with a non-finite number (in the whole-backtest record, in any side's or any segment's), a
negative spread, a stake of zero or less, a Brier score outside [0, 1], fewer than two trades, or a side or
segment with more trades than the backtest it belongs to. The non-finite case matters most: `NaN` is a
legal token for Python's `json.loads` and compares false with everything, so an unchecked `NaN` spread
would not fail the z test, it would switch the check off and let a badly degraded champion exit 0. As a
second line of defence `health_report` reports a check whose expectation (cause `invalid_expectation`) or
live statistic (cause `invalid_statistic`, for example a variance that overflowed) is not finite as
`insufficient` with the offending number named, never as `ok`.

---

## The whole chain, from the CLI

The guide's examples come from one small world, built so that each failure can be made on purpose. The
user's side of it is ordinary Python written for the guide and not part of the library: a market
generator (three segments `A`, `B`, `C`; the market's price strays from the true probability by a noise the
model can exploit), a LightGBM model trained on a 450-day panel, the gate, a scan-edge step that turns the
model and the live prices into `EdgeSignal`s (`EdgeSignal.from_quote`), and a settlement step (the CLI's
`settle-trades` is a stub, so settlement calls `SettlementEngine`). Everything else goes through the
installed `pmlab`: a backtest of the live policy under a cost model, a champion promoted on it, paper
trades recorded under the champion's costs and policy, then watched.

```text
$ pmlab backtest --plugin e2e_family --panel data/panel.parquet --stride 10 --min-train-rows 200 --costs costs.json --output metrics.json
Backtest plugin=e2e_family model=lgbm_baseline stride=10
Loading panel from data/panel.parquet...
Panel: 13500 rows, 450 dates, 13500 markets
Policy: live (best edge >= 0.2, YES+NO)
       Backtest Results        
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Metric          ┃ Value     ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ Total trades    │ 391       │
│ YES / NO trades │ 192 / 199 │
│ Steps evaluated │ 44        │
│ Total PnL       │ $+476.76  │
│ Hit rate        │ 54.7%     │
│ Avg edge        │ 0.3177    │
└─────────────────┴───────────┘
Metrics saved to metrics.json
[exit code: 0]
```

Both sides are traded, at the threshold `record-trades` uses. The expectation that backtest left in
`metrics.json` (the numbers above in the section on [the expectation](#the-backtest-expectation)) is
promoted with the champion, which carries the costs and the policy without `promote-champion` being told
them again:

```text
$ pmlab promote-champion artifacts/model.pkl --gate-path gate.json --plugin e2e_family --output-dir artifacts/public_models --name e2e-lgbm --expectation metrics.json
Gate decision: GO
Aggregate PnL: $+476.76 over 391 trades
Champion published!
  Model: artifacts/public_models/champion.pkl
  Manifest: artifacts/public_models/champion.json
  Allowed segments: {'B', 'C', 'A'}
  Costs: {'taker_bps': 30.0, 'slippage_bps': 50.0, 'slippage_fixed': 0.0, 'depth_bps_per_unit': 0.0, 'depth_reference_stake': 1.0}
  Policy: live (best edge >= 0.2, YES+NO)
[exit code: 0]
```

Each live world is 600 days of markets, one per segment per day. `record-trades` records the signals that
clear the champion's threshold (458 trades in each world, YES and NO), a nightly settlement stamps
`settled_at`, and the worlds differ only in what the truth did. One YES trade and two NO trades from the
healthy world's log: the YES quote became a fill 50 bps worse; a NO trade buys the NO token at
`1 - gamma_price` plus the same friction. `outcome` says whether the *label* occurred, so a NO trade on a
label that occurred is `"won"` while its PnL is the whole stake and the fee lost:

```text
# a YES trade, a NO trade whose label occurred (NO loses), a NO trade whose label did not (NO wins)
{"city_or_segment": "B", "target_date": "2028-04-21", "direction": "yes", "gamma_price": 0.266261, "fill_price": 0.267592, "flat_stake": 1.0, "fee_paid": 0.003, "outcome": "won", "realized_pnl": 2.734026, "settled_basis": "fill_price", "settlement_version": 1}
{"city_or_segment": "C", "target_date": "2028-04-22", "direction": "no", "gamma_price": 0.853613, "fill_price": 0.147119, "flat_stake": 1.0, "fee_paid": 0.003, "outcome": "won", "realized_pnl": -1.003, "settled_basis": "fill_price", "settlement_version": 1}
{"city_or_segment": "C", "target_date": "2027-08-10", "direction": "no", "gamma_price": 0.758975, "fill_price": 0.24223, "flat_stake": 1.0, "fee_paid": 0.003, "outcome": "lost", "realized_pnl": 3.125313, "settled_basis": "fill_price", "settlement_version": 1}
# 458 settled trades recomputed from their own fields: largest difference from realized_pnl 5.0e-07
```

Recomputing every settled trade in that log from its own recorded fields (shares bought at `fill_price`,
paying 1 if the token wins, minus the stake and the fee) agrees with `realized_pnl` to 5e-7, and each NO
trade whose label occurred lost exactly stake plus fee.

The PnL check uses the costs and judges both sides: the cost basis is known and matches, nothing was
unverified, no NO trade was left out:

```text
$ python3 -c '...'   # the PnL check of the healthy world, from --json
{'n_total': 458, 'n_trades': 458, 'n_excluded_no_side': 0, 'n_excluded_yes_side': 0, 'n_excluded_unfixed_no': 0, 'n_excluded_legacy_settlement': 0, 'n_excluded_cost_basis': 0, 'n_cost_unverified': 0, 'cost_basis': 'match'}
```

Without costs anywhere (a champion promoted with no `--costs` and no costs in its expectation file) the
same check reports `"cost_basis": "unknown"` and compares nothing.

---

## Re-settling an old log

Releases before 0.9.1 settled a NO trade as a short YES at the NO price, so a NO bet that lost booked a
fraction of its stake instead of the whole of it, and settlement is idempotent, so those numbers are still in
the log. A trade settled by the current release carries `settlement_version`; one that does not, and is a
NO trade, is **left out of the PnL check and counted** (`n_excluded_unfixed_no`, and a note in the report).
It is never read as green, and it is never trusted. `pmlab resettle` repairs the log:

```bash
pmlab resettle --trades artifacts/workspaces/ops_daily/signals/v2/forward_paper_trades.json --dry-run
pmlab resettle --trades artifacts/workspaces/ops_daily/signals/v2/forward_paper_trades.json
```

For every **settled** trade without the current `settlement_version` it recomputes `realized_pnl` from what
the trade recorded (its side, the fill price, else the quote, the stake and fee, and whether its label
occurred; `outcome` is about the label), using the same arithmetic settlement uses, and stamps
`settlement_version` and `resettled_at` on it (a YES trade settled before 0.9.1 is recomputed and stamped
too; if its number does not move it is not listed among the changes). That is all it does:

- **It never touches an unsettled trade**, and never calls the network or a plugin: the truth is the one
  the trade already holds, so safety rule 6 (never settle before `is_truth_final()`) is not in play, and
  nothing is re-asked.
- **It writes a timestamped backup** (`forward_paper_trades.json.bak-<UTC timestamp>`) next to the log
  before it replaces it, atomically. A backup is never overwritten.
- **`--dry-run` writes nothing** and prints the count, the total PnL change and the first `--show` (10)
  per-trade changes, old value to new.
- **It is idempotent.** A trade already at the current version is not touched, so a second run reports
  "nothing to re-settle" and writes no backup.
- A settled trade whose recorded fields are unusable (an `outcome` that is not `won` or `lost`, no
  `outcome_label`, a price outside [0, 1], a stake that is not positive) is left as it was and listed, and
  the watchdog keeps leaving it out.
- **Nothing locks the log.** Run it when nothing else is writing to it, as you would a second
  `record-trades`: a row appended between its read and its replace would be lost.

A real run on a log written the way 0.9.0 left it (every NO trade booked as a short YES; no
`settled_at`, no `settlement_version`):

```text
$ ls artifacts/workspaces/paper_legacy/signals/v2/
forward_paper_trades.json

$ pmlab resettle --trades artifacts/workspaces/paper_legacy/signals/v2/forward_paper_trades.json --dry-run --show 4
pmlab resettle (dry run): artifacts/workspaces/paper_legacy/signals/v2/forward_paper_trades.json
  458 settled trades below settlement version 1 would be re-settled (0 open trades left alone).
  376 of them change.
  total realized PnL of those trades: -32.032217 would become +678.490528 (delta +710.522745)
  first 4 of 376 changes:
    B 2028-04-21 morning_of yes: +2.752712 -> +2.734026
    C 2027-08-10 morning_of no: +0.997000 -> +3.125313
    C 2028-04-22 morning_of no: -5.834206 -> -1.003000
    A 2028-04-23 morning_of yes: +1.795743 -> +1.781819
    ... and 372 more
  Nothing was written.
[exit code: 0]

$ pmlab resettle --trades artifacts/workspaces/paper_legacy/signals/v2/forward_paper_trades.json --show 3
pmlab resettle: artifacts/workspaces/paper_legacy/signals/v2/forward_paper_trades.json
  458 settled trades below settlement version 1 re-settled (0 open trades left alone).
  376 changed.
  total realized PnL of those trades: was -32.032217, now +678.490528 (delta +710.522745)
  first 3 of 376 changes:
    B 2028-04-21 morning_of yes: +2.752712 -> +2.734026
    C 2027-08-10 morning_of no: +0.997000 -> +3.125313
    C 2028-04-22 morning_of no: -5.834206 -> -1.003000
    ... and 373 more
  Backup of the original: /home/arthur/work/tmp-pmlab/e2e/work/artifacts/workspaces/paper_legacy/signals/v2/forward_paper_trades.json.bak-20261021T212321Z
[exit code: 0]

$ ls artifacts/workspaces/paper_legacy/signals/v2/
forward_paper_trades.json
forward_paper_trades.json.bak-20261021T212321Z

$ pmlab resettle --trades artifacts/workspaces/paper_legacy/signals/v2/forward_paper_trades.json
pmlab resettle: artifacts/workspaces/paper_legacy/signals/v2/forward_paper_trades.json
  Nothing to re-settle: every settled trade is already at settlement version 1 (458 settled, 0 open trades left alone).
[exit code: 0]

$ ls artifacts/workspaces/paper_legacy/signals/v2/ | wc -l
2

458 re-settled trades vs the same trades settled fresh by the engine: largest difference in realized_pnl 0.0e+00
```

And what the watchdog says about that log before and after:

```text
$ pmlab monitor -a artifacts -w paper_legacy --reference data/panel.parquet --snapshots data/snaps/paper_legacy --family e2e_family --as-of 2028-12-31T00:00:00Z --verbose
pmlab monitor: OK  (champion e2e-lgbm, as of 2028-12-31T00:00:00Z)
  drift       ok            2 feature(s) and the prediction within thresholds (max psi 0.040, 500 live rows)
  calibration ok            overall: Brier 0.211 vs backtest 0.228 (-0.018 abs, -7.8% rel, z=-1.4, n=458); within thresholds
                            note: 458 of 458 in-scope trades have no settled_at (settled by an older release), so they are judged by target_date and may include truth that arrived after as_of
  pnl         ok            overall: mean PnL/stake +1.481 vs backtest +1.225 (shortfall -21%, z=-1.0, n=458); within thresholds (could only detect a shortfall >= 62% of the backtest edge at n=458)
                            sequential: CUSUM 0.0 of 30 (peak 4.0, n=458); within thresholds
                            note: 458 of 458 in-scope trades have no settled_at (settled by an older release), so they are judged by target_date and may include truth that arrived after as_of
[exit code: 0]
```

---

## `insufficient` is not healthy

A check that cannot be measured (fewer settled trades than `min_trades`, no expectation, no drift
inputs) is `insufficient`, and by default the report gets its own status and its own exit code,
`30`:

```text
$ pmlab monitor -a artifacts -w paper_new --reference data/panel.parquet --snapshots data/snaps/paper_new --family e2e_family --as-of 2028-12-31T00:00:00Z
pmlab monitor: INSUFFICIENT  (champion e2e-lgbm, as of 2028-12-31T00:00:00Z)
  calibration insufficient  12 scoreable settled trades, need >= 30 (none excluded)
  pnl         insufficient  12 scoreable settled trades, need >= 30 (none excluded)
[exit code: 30]
```

`--insufficient warn` (or `insufficient = "warn"` in the config file) counts it as a warn (exit 10)
instead; the individual checks still say `insufficient`:

```text
$ pmlab monitor -a artifacts -w paper_new --reference data/panel.parquet --snapshots data/snaps/paper_new --family e2e_family --as-of 2028-12-31T00:00:00Z --insufficient warn
pmlab monitor: WARN  (champion e2e-lgbm, as of 2028-12-31T00:00:00Z)
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
| calibration | `min_trades` | 30 | scoreable settled trades a scope needs before it is judged |
| | `brier_warn_abs` / `brier_alert_abs` | 0.02 / 0.05 | live Brier minus backtest Brier |
| | `brier_warn_rel` / `brier_alert_rel` | 0.10 / 0.25 | the same gap as a fraction of the backtest Brier |
| | `z_warn` / `z_alert` | 2.0 / 3.0 | one-sided z the gap must clear (Holm-adjusted within a family) |
| | `n_bins` | 10 | reliability bins for the reported `reliability_live` |
| pnl | `min_trades` | 30 | scoreable settled trades a scope needs before it is judged |
| | `shortfall_warn` / `shortfall_alert` | 0.5 / 1.0 | `(backtest - live) / backtest` |
| | `z_warn` / `z_alert` | 2.0 / 3.0 | one-sided z the shortfall must clear (Holm-adjusted within a family) |
| | `sequential` | `true` | run the [sequential test](#the-sequential-test) as well |
| | `cusum_threshold` | 30 | `h`: the statistic at which the sequential test alerts |
| | `cusum_allowance` | 0.5 | `a`: the shortfall each trade may have, as a fraction of the promised edge, without counting |
| | `cusum_backtest_se` | 0.5 | `b`: the standard errors of the promise added to that allowance |
| | `cusum_min_backtest_trades` | 100 | fewest backtest trades a record needs to standardize live trades for the sequential test |
| | `cusum_increment_cap` | 5 | `c`: the most one trade may add to the statistic (see [the sequential test](#the-sequential-test)) |
| | `mde_power` | 0.8 | power the [minimum detectable shortfall](#what-a-quiet-run-is-worth) is quoted at |
| drift | everything in `DriftConfig` | see [Drift monitoring](drift-monitoring.md) | PSI 0.10/0.25, KL 0.05/0.125, missing 0.05/0.20, `min_samples` 300 |
| policy | `insufficient` | `separate` | `separate` (exit 30) or `warn` (exit 10) |

`min_trades` applies to every scope: the book, each side and each segment is judged only once it has that
many scoreable settled trades, and is otherwise [not assessed](#what-decides-a-checks-verdict).

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
PnLThresholds(min_trades=50, shortfall_warn=0.5, shortfall_alert=1.0, z_warn=1.645, z_alert=3.0, sequential=True, cusum_allowance=0.5, cusum_backtest_se=0.5, cusum_threshold=30.0, cusum_min_backtest_trades=100, cusum_increment_cap=5.0, mde_power=0.8)
calibration untouched: True
rejected: unknown pnl setting(s) ['min_trade']; valid settings: ['cusum_allowance', 'cusum_backtest_se', 'cusum_increment_cap', 'cusum_min_backtest_trades', 'cusum_threshold', 'mde_power', 'min_trades', 'sequential', 'shortfall_alert', 'shortfall_warn', 'z_alert', 'z_warn']
rejected: z_warn (4) must not exceed z_alert (3): warn is the lower bar and alert the higher
```

---

## Output

**Quiet on green.** A healthy run prints nothing, to stdout or stderr. `--verbose` prints the full
report even then:

```text
$ pmlab monitor -a artifacts -w paper_healthy --reference data/panel.parquet --snapshots data/snaps/paper_healthy --family e2e_family --as-of 2028-12-31T00:00:00Z --verbose
pmlab monitor: OK  (champion e2e-lgbm, as of 2028-12-31T00:00:00Z)
  drift       ok            2 feature(s) and the prediction within thresholds (max psi 0.040, 500 live rows)
  calibration ok            overall: Brier 0.211 vs backtest 0.228 (-0.018 abs, -7.8% rel, z=-1.4, n=458); within thresholds
  pnl         ok            overall: mean PnL/stake +1.481 vs backtest +1.225 (shortfall -21%, z=-1.0, n=458); within thresholds (could only detect a shortfall >= 62% of the backtest edge at n=458)
                            sequential: CUSUM 0.7 of 30 (peak 4.4, n=458); within thresholds
[exit code: 0]
```

**Which part failed is in the line.** The calibration and PnL lines lead with the most specific problem,
and name it: a segment, a side, or the sequential test, and then say that the book as a whole is too if it is.
A segment that is failing while the aggregate looks fine, from the world built for it (segment B's picks
win 45 points less often than the model said; A and C are untouched):

```text
$ pmlab monitor -a artifacts -w paper_segment --reference data/panel.parquet --snapshots data/snaps/paper_segment --family e2e_family --as-of 2028-12-31T00:00:00Z
pmlab monitor: ALERT  (champion e2e-lgbm, as of 2028-12-31T00:00:00Z)
  calibration alert         segment B: Brier 0.309 vs backtest 0.217 (+0.093 abs, +42.7% rel, z=3.9, Holm z=3.6, n=143); alert needs >= +0.05 abs, >= +25% rel, z >= 3
                            overall: Brier 0.241 vs backtest 0.228 (+0.013 abs, +5.7% rel, z=1.0, n=458); within thresholds
  pnl         alert         segment B: mean PnL/stake -0.477 vs backtest +0.824 (shortfall 158%, z=5.3, Holm z=5.1, n=143); alert needs shortfall >= 100%, z >= 3 (could only detect a shortfall >= 97% of the backtest edge at n=143)
                            NO side: mean PnL/stake +0.470 vs backtest +1.396 (shortfall 66%, z=2.7, Holm z=2.5, n=238); warn needs shortfall >= 50%, z >= 2 (could only detect a shortfall >= 76% of the backtest edge at n=238)
                            overall: mean PnL/stake +0.844 vs backtest +1.225 (shortfall 31%, z=1.5, n=458); within thresholds (could only detect a shortfall >= 60% of the backtest edge at n=458)
                            sequential: CUSUM 1.1 of 30 (peak 8.1, n=458); within thresholds
[exit code: 20]
```

And a NO side that is failing while the YES side is fine (the NO tokens win 25 points less often than the
model said):

```text
$ pmlab monitor -a artifacts -w paper_noside --reference data/panel.parquet --snapshots data/snaps/paper_noside --family e2e_family --as-of 2028-12-31T00:00:00Z
pmlab monitor: ALERT  (champion e2e-lgbm, as of 2028-12-31T00:00:00Z)
  calibration alert         NO side: Brier 0.282 vs backtest 0.219 (+0.064 abs, +29.1% rel, z=3.4, Holm z=3.2, n=238); alert needs >= +0.05 abs, >= +25% rel, z >= 3
                            overall: Brier 0.251 vs backtest 0.228 (+0.023 abs, +9.9% rel, z=1.7, n=458); within thresholds
  pnl         warn          segment C: mean PnL/stake +0.161 vs backtest +1.457 (shortfall 89%, z=3.0, Holm z=2.6, n=154); warn needs shortfall >= 50%, z >= 2 (could only detect a shortfall >= 97% of the backtest edge at n=154)
                            NO side: mean PnL/stake +0.003 vs backtest +1.396 (shortfall 100%, z=4.2, Holm z=4.0, n=238); warn needs shortfall >= 50%, z >= 2 (could only detect a shortfall >= 74% of the backtest edge at n=238)
                            overall: mean PnL/stake +0.920 vs backtest +1.225 (shortfall 25%, z=1.1, n=458); within thresholds (could only detect a shortfall >= 62% of the backtest edge at n=458)
                            sequential: CUSUM 0.7 of 30 (peak 8.2, n=458); within thresholds
[exit code: 20]
```

The scopes that could not be judged are listed, not hidden (`not assessed: ...`), up to five and then
"and N more"; with `--verbose` every check also prints the lines of the scopes that were fine.

**`--json PATH`** always writes the full report, healthy or not, for logs and alerting pipelines.
`--json -` writes it to stdout, and then the human text moves to stderr so that stdout stays
machine-readable:

```text
$ pmlab monitor -a artifacts -w paper_degraded --reference data/panel.parquet --snapshots data/snaps/paper_degraded --family e2e_family --as-of 2028-12-31T00:00:00Z --json - 2>/dev/null | python3 -c '...'
{'status': 'alert', 'exit_code': 20}
[('drift', 'ok'), ('calibration', 'alert'), ('pnl', 'alert')]
```

| stream | content |
|---|---|
| stdout | the problem report (nothing when healthy); with `--json -`, the JSON instead |
| stderr | usage and internal errors (one line); with `--json -`, the problem report |

The text is plain ASCII with no colour codes: it lands in cron mail and log files.

Each of the calibration and PnL checks carries its parts in `metrics`: `directions` (`yes`, `no`: only the
sides that traded) and `segments` (one entry per segment that traded), each holding the same numbers the
book does (calibration: `brier_live`, `brier_backtest`, `degradation_abs`, `degradation_rel`; PnL:
`mean_live`, `std_live`, `mean_backtest`, `shortfall_rel`, `mde_shortfall`, `mde_shortfall_effective`; both:
`z`, `p_value`, and `z_adjusted` / `p_adjusted` inside a family of more than one) plus its own `status`,
`cause` (set when it could not be judged, e.g. `too_few_samples`) and `n_trades`. The PnL check also has
`sequential` with `status`, `signalled`, `statistic`, `peak`, `crossed_at`, `threshold`, `allowance`,
`increment_cap`, `n` and `n_skipped`. A scope that was not assessed has status `insufficient` there, and still does not change the
check's own status unless nothing in the check could be measured.

```text
$ python3 -c '...'   # the PnL check of the segment world, from --json
status alert
segments['A'] {'status': 'ok', 'n_trades': 161, 'mean_live': 1.96, 'mean_backtest': 1.358, 'shortfall_rel': -0.443, 'z': -1.12, 'mde_shortfall_effective': 1.294, 'z_adjusted': -1.12}
segments['B'] {'status': 'alert', 'n_trades': 143, 'mean_live': -0.477, 'mean_backtest': 0.824, 'shortfall_rel': 1.579, 'z': 5.327, 'mde_shortfall_effective': 0.969, 'z_adjusted': 5.124}
segments['C'] {'status': 'ok', 'n_trades': 154, 'mean_live': 0.903, 'mean_backtest': 1.457, 'shortfall_rel': 0.38, 'z': 1.246, 'mde_shortfall_effective': 0.998, 'z_adjusted': 0.797}
directions['yes'] {'status': 'ok', 'n_trades': 220, 'mean_live': 1.248, 'mean_backtest': 1.039, 'z': -0.538, 'z_adjusted': -0.538}
directions['no'] {'status': 'warn', 'n_trades': 238, 'mean_live': 0.47, 'mean_backtest': 1.396, 'z': 2.735, 'z_adjusted': 2.498}
sequential {'threshold': 30.0, 'n_skipped': 0, 'status': 'ok', 'signalled': False, 'statistic': 1.15, 'peak': 8.15, 'crossed_at': None, 'n': 458}
```

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
{"n_total": 0, "n_open": 0, "n_out_of_segment": 0, "n_unattributed": 0, "n_before_champion": 0, "n_after_as_of": 0, "n_no_settled_at": 0, "n_in_scope": 0, "n_invalid_forecast": 0, "allowed_segments": ["politics"], "n_by_segment": {}, "n_trades": 0, "n_by_direction": {}, "n_skipped_no_forecast": 0, "min_trades": 30, "segments_recorded": false}
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
z = (reference - live) / sqrt( live_variance / n_live  +  reference_variance )
```

with a normal approximation. The sign is chosen so that **a positive z always means worse than the
reference**: for Brier, where higher is worse, it is `(live - reference)` in the numerator. The reference
is the [cell mixture](#what-the-book-is-held-to): its mean is the weighted mean of the backtest means of the
cells the live trades fell in, and its variance, the sum of `weight^2 * std^2 / n` over those cells, is the
uncertainty of that promise. Both terms of the standard error matter. The live sample is noisy; so is the
backtest mean, which is itself an estimate: a promise made on 20 trades is a weak promise, and falling
short of it means less than falling short of one made on 2000. The two guards catch different mistakes:
effect size alone pages on noise at small n, significance alone pages on trivia at large n.

`z_warn = 2` and `z_alert = 3` are stricter than the textbook one-sided 1.64 and 2.33 on purpose. The
watchdog looks every run, and repeated looks at one growing sample inflate the false-alarm rate far above
the nominal level.

**Holm-Bonferroni** (`pmlab.monitoring.stats.holm_adjust`) corrects the sides as a family and the segments
as a family. It is checked against a worked example by hand in `tests/monitoring/test_stats.py`: for the
p-values `0.01, 0.04, 0.03, 0.005` it gives `0.03, 0.06, 0.06, 0.02`.

**How often does the PnL check fire?** The tables below are produced by `scripts/watchdog_power.py`, which
drives `health_report` itself (`uv run python scripts/watchdog_power.py firing`), and a vectorised copy of
the same checks for the larger sweeps; `tests/monitoring/test_power.py` holds the copy to `health_report`
trade by trade, so the tables cannot drift from the code. Every live trade buys a label at a price between
0.30 and 0.60 and the backtest promised a 6-point edge (400 trades). "Halved" means the market now prices
the label 3 points better than the model thinks, "vanished" 6 points, "reversed" 12:

```text
How often the PnL check warns or alerts (400 simulated champions per cell)
                   fixed test only                    fixed test or sequential test
live trades     as promised       halved     vanished     reversed   |  as promised       halved     vanished     reversed
         30            3.0%         5.2%        13.0%        27.0%   |         3.0%         5.2%        13.0%        27.0%
         50            3.0%         5.2%        18.2%        37.8%   |         3.0%         5.2%        18.2%        37.8%
        100            2.5%         8.8%        24.8%        56.5%   |         2.5%         8.8%        24.8%        56.5%
        200            2.5%        12.0%        34.0%        73.5%   |         2.5%        12.0%        34.0%        74.0%
        400            2.5%        14.8%        41.5%        92.2%   |         2.8%        15.5%        44.5%        95.2%
        800            1.5%        17.0%        50.7%        96.5%   |         2.5%        22.5%        66.8%        98.8%
```

Read it both ways. **False alarms stay low** (3% or less at every size). **Power is modest**: with 100
live trades the check notices a *completely vanished* edge about one run in four, 400 trades get that to
about two in five (the sequential test adds three points there and sixteen at 800), and a *halved* edge
is mostly unnoticed, because per-trade PnL is very noisy (here its standard deviation is several times
its mean). That is the arithmetic of the quantity, not a defect of the threshold: lowering `z_warn` buys power
at the price of false alarms. A quiet PnL check is therefore **weak evidence that the edge is intact**,
which is why every PnL result says what it could have detected
([what a quiet run is worth](#what-a-quiet-run-is-worth)).

**Running it after every settled trade.** The table above is one look at a book of N trades. A job that
runs often looks again and again at a growing sample, and the chance that *some* run along the way alarmed
for nothing grows with the number of looks:

```text
False alarms when the check is run after every settled trade (4000 simulated champions)
nothing has changed. For each test: the share of champions that raised an alarm at
some run within the first N trades (events), and the share whose alarm is on at the
run after trade N (what a cron job sees at that moment)

warn or alert (exit 10 or 20)
live trades    fixed: events    at N   sequential: events    at N   either: events
100                    10.7%    2.4%                 0.0%    0.0%            10.7%
200                    13.6%    2.7%                 0.7%    0.3%            13.6%
400                    15.4%    2.4%                 2.8%    0.7%            16.0%
800                    16.4%    2.5%                 7.3%    1.5%            19.2%
1500                   17.0%    2.5%                13.7%    2.0%            24.2%
3000                   17.4%    1.9%                25.8%    2.0%            34.2%

alert only (exit 20)
live trades    fixed: events    at N   sequential: events    at N   either: events
100                     1.6%    0.1%                 0.0%    0.0%             1.6%
200                     2.1%    0.2%                 0.7%    0.3%             2.4%
400                     2.4%    0.1%                 2.8%    0.7%             4.3%
800                     2.4%    0.0%                 7.3%    1.5%             8.7%
1500                    2.4%    0.0%                13.7%    2.0%            14.9%
3000                    2.4%    0.0%                25.8%    2.0%            26.8%
```

Read the two kinds of column apart. **Events** is the chance that some run along the way alarmed for
nothing; it only grows with the number of looks, for both tests. The fixed test, a fixed-sample test looked at
over and over, reaches 10% of healthy champions as warn-or-alert within the first 100 trades and 17% by 3000.
The sequential test is quiet early (nothing in the first 100 trades) but its events keep accumulating,
to 26% by 3000 trades. **At N** is what a cron job sees at that moment: the sequential alert is on for at most
2% of healthy champions however long the book, because it clears when a run of bad luck ends. Whoever
reads the exit code only sees the second kind; whoever counts how often the job ever paged sees the first.
That is the other reason `z_warn` is 2 and not 1.64, and why only an *alert* (exit 20) is worth paging on: its
events are 1.6-2.4% for the fixed test.

**All the parts together.** The book, each side, each segment and the sequential test are separate families
and the check alerts if any of them does. With nothing wrong, the share of champions that warn or alert
anywhere, on 100 live trades per segment:

```text
False alarms from the whole PnL check, every part together (400 simulated champions per row)
nothing has changed; each segment has 100 live trades, judged once at the end
(book, both sides, every segment, the sequential test)

segments    warn or alert   alert only   alerts from the sequential test
1                    2.8%         0.0%                              0.0%
3                    4.2%         0.2%                              0.2%
5                    3.8%         0.8%                              0.2%
10                   5.5%         1.0%                              1.0%
```

**Is calibration the earlier warning?** It is natural to expect it to move first. For a vanishing edge it
does not, and the numbers are the reason. The edge is a few points of probability; a squared error is the *square* of the
forecast error, so it barely moves for a few points, while PnL moves by the whole edge. Same trades, same
n, share of runs that warn or alert:

```text
PnL check against calibration check on the same trades (400 simulated champions per cell)
share of runs that warn or alert, at a live edge 6 points below the forecast (vanished)
and 3 points below it (halved); a mean forecast error test is shown for comparison

prices 0.30-0.60
                          n=50   n=100   n=200   n=400
as promised, PnL          2.2%    2.0%    2.8%    2.0%
as promised, Brier        2.8%    0.2%    0.0%    0.0%
as promised, mean error    2.0%    2.2%    2.8%    2.0%
halved, PnL               5.8%    8.5%    9.5%   10.5%
halved, Brier             2.0%    0.5%    0.0%    0.0%
halved, mean error        6.0%    8.0%   11.0%   11.8%
vanished, PnL            11.2%   17.0%   26.0%   37.5%
vanished, Brier           2.8%    0.5%    0.5%    0.0%
vanished, mean error      9.8%   16.0%   25.0%   38.8%

cheap tokens 0.05-0.25
                          n=50   n=100   n=200   n=400
as promised, PnL          6.2%    4.0%    3.0%    2.8%
as promised, Brier        1.0%    2.8%    2.0%    1.8%
as promised, mean error    3.8%    2.2%    2.2%    2.8%
halved, PnL              18.0%   16.0%   17.2%   21.0%
halved, Brier             0.0%    0.0%    0.0%    0.0%
halved, mean error       13.0%   14.0%   16.0%   21.2%
vanished, PnL            25.8%   29.0%   42.0%   55.5%
vanished, Brier           0.0%    0.0%    0.0%    0.0%
vanished, mean error     21.0%   27.8%   39.8%   57.0%
```

A test of the **mean forecast error** (`mean(occurred - forecast)` against what the backtest saw) has the
power PnL has, almost row for row, because it uses the same information: it is not a way to find the edge
going earlier, only a different way to say it has. **Drift is the earlier warning**: the inputs move before
any truth arrives. Calibration is the check for a model that has *gone wrong* (forecasts off by 0.15 and
more), PnL the check for an edge that has gone, and the sequential test the check for one that is
fading:

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

- **Power is limited by the information, not by the test.** A few trades' worth of PnL cannot tell a
  6-point edge from none. The sequential test helps where the change starts late and the pooled mean
  dilutes it; it does not make a flat decline visible any sooner than a test of the mean would. The
  detectable shortfall is printed so that a quiet run is not read as more than it is.
- **Thin segments are a blind spot, on purpose.** A segment with fewer than `min_trades` live trades is
  not judged on its own and does not make the check insufficient (see [the verdict
  rule](#what-decides-a-checks-verdict)). It is named in the report, but until it has traded enough a
  failure there is seen only through the book.
- **The cells are a scope, not a test.** A (segment, side) cell is the reference a trade is held to, not a
  separate hypothesis: judging every cell on its own would multiply the false alarms by the number of
  cells. The sides and the segments are judged, the cells they are made of are not.
- **Near-certain cells.** Where the backtest almost never lost, the sequential test is blunt on purpose
  (one trade cannot trip it, [above](#near-certain-cells)) and the fixed test carries the detection.
- **The backtest mean is optimistic.** A champion is chosen *because* its backtest looked good (a winner's
  curse), so its live mean tends to fall below the promise even when nothing is wrong. The warn level of
  "half of what was promised" leaves room for that: at the sample sizes in the tables a shortfall of that
  size cannot be told apart from noise, and is correctly not flagged. The sequential test's `b` term is
  there for the same reason.
- **Trades are treated as independent.** They are not quite: markets that resolve together, or a regime,
  correlate them, which makes the true noise larger than the test assumes, and the false-alarm rates above
  are the rates for independent trades.
- **Skew needs volume.** A payoff that wins small often and loses big rarely (or the reverse) has a
  skewed mean, and a normal approximation is shakiest at small n. `min_trades = 30` is the usual floor,
  not a guarantee; raise it for strongly skewed strategies.
- **Calibrate like with like.** The expectation's Brier is scored on the backtest's raw `predicted_prob`.
  If your live `model_prob` is passed through a calibrator, or your live selection rule differs from the
  backtest's, the two Briers are not on the same footing. Calibration scores the model's probability that
  the label occurs whichever side was bought, in the backtest and live alike.
- **Depth-sensitive costs.** PnL is compared per unit stake. If your `CostModel` has a depth term, a live
  stake that differs from the backtest's pays a different price, and per-stake PnL is not exactly
  stake-free.
- **NO trades settled before 0.9.1 are left out until they are re-settled.** They were booked as a short
  YES; see [Re-settling an old log](#re-settling-an-old-log).
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
  `--artifacts-dir` / `PMLAB_ARTIFACTS_DIR` and `--workspace` / `PMLAB_WORKSPACE`. The trade log is
  `<workspace dir>/signals/v2/forward_paper_trades.json`, where the workspace directory is
  `<artifacts dir>/workspaces/<name>`, unless the artifacts directory already *is* that directory.
  `scripts/pmlab-workspace <name> ...` points the artifacts directory at `artifacts/workspaces/<name>`
  itself and sets both `PMLAB_WORKSPACE_NAME` and `PMLAB_WORKSPACE`, so it and `pmlab -a artifacts -w <name>`
  read and write the same trade log; before 0.9.1 the wrapper looked in `.../workspaces/<name>/workspaces/ops_daily/...`
  and found nothing. No file moves: logs are where they have always been. **The champion is the one
  difference.** It lives at `<artifacts dir>/public_models/champion.json`, so under the wrapper it is
  `artifacts/workspaces/<name>/public_models/champion.json`, and with `-a artifacts` it is
  `artifacts/public_models/champion.json`; `--champion` names it explicitly either way:

```text
$ scripts/pmlab-workspace paper_healthy pmlab monitor --skip drift --as-of 2028-12-31T00:00:00Z -v
  pmlab monitor: OK  (champion e2e-lgbm, as of 2028-12-31T00:00:00Z)
    calibration ok            overall: Brier 0.211 vs backtest 0.228 (-0.018 abs, -7.8% rel, z=-1.4, n=458); within thresholds
    pnl         ok            overall: mean PnL/stake +1.481 vs backtest +1.225 (shortfall -21%, z=-1.0, n=458); within thresholds (could only detect a shortfall >= 62% of the backtest edge at n=458)
                              sequential: CUSUM 0.7 of 30 (peak 4.4, n=458); within thresholds
    skipped: drift
[exit code: 0]

$ pmlab monitor -a artifacts -w paper_healthy --skip drift --as-of 2028-12-31T00:00:00Z -v
  pmlab monitor: OK  (champion e2e-lgbm, as of 2028-12-31T00:00:00Z)
    calibration ok            overall: Brier 0.211 vs backtest 0.228 (-0.018 abs, -7.8% rel, z=-1.4, n=458); within thresholds
    pnl         ok            overall: mean PnL/stake +1.481 vs backtest +1.225 (shortfall -21%, z=-1.0, n=458); within thresholds (could only detect a shortfall >= 62% of the backtest edge at n=458)
                              sequential: CUSUM 0.7 of 30 (peak 4.4, n=458); within thresholds
    skipped: drift
[exit code: 0]

$ diff <(wrapper report) <(plain report)
identical
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
$ pmlab monitor -a /nonexistent -w paper_healthy --reference data/panel.parquet --as-of 2028-12-31T00:00:00Z --snapshots data/snaps/paper_healthy
pmlab monitor: error: champion not found: /nonexistent/public_models/champion.json
[exit code: 64]

$ pmlab monitor -a artifacts_bad -w paper_healthy --reference data/panel.parquet --as-of 2028-12-31T00:00:00Z --snapshots data/snaps/paper_healthy
pmlab monitor: error: champion.json is unreadable (artifacts_bad/public_models/champion.json): JSONDecodeError: Expecting value: line 1 column 16 (char 15)
[exit code: 64]

$ pmlab monitor -a artifacts -w paper_healthy --reference data/panel.parquet --as-of 2028-12-31T00:00:00Z --snapshots data/snaps/paper_helthy
pmlab monitor: error: snapshot directory not found: data/snaps/paper_helthy
[exit code: 64]

$ pmlab monitor -a artifacts -w paper_healthy --skip drift --as-of 2028-12-31T00:00:00
pmlab monitor: error: --as-of '2028-12-31T00:00:00' has no UTC offset, so it could be anyone's local time; give one explicitly, e.g. 2026-03-01T00:00:00Z or 2026-03-01T00:00:00+01:00
[exit code: 64]

$ pmlab monitor -a artifacts_nan -w paper_degraded --skip drift --as-of 2028-12-31T00:00:00Z
pmlab monitor: error: champion.json is unreadable (artifacts_nan/public_models/champion.json): ValueError: std_pnl_per_stake must be a finite number, got nan
[exit code: 64]

$ pmlab monitor --bogus
Usage: pmlab monitor [OPTIONS]
Try 'pmlab monitor --help' for help.
╭─ Error ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ No such option: --bogus (Possible options: --verbose)                                                                                                                                                │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
[exit code: 2]
```
