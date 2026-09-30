# Live calibration tracking

The holdout gate answers *was this model calibrated on history*. This answers
*is it still calibrated on money that was actually at risk*.

They are different questions, and the second one is the one that costs you money
when the answer changes.

---

## The quick version

```python
from pmlab import CalibrationTracker

tracker = CalibrationTracker.from_path("artifacts/paper_trades.json")

overall = tracker.overall()
print(f"{overall.n_trades} realized  brier {overall.brier_score:.4f}  "
      f"reliability {overall.reliability:.4f}")

# Has it drifted? Compare early windows to late ones.
for w in tracker.rolling(window=50):
    print(w.window_end, round(w.brier_score, 4))

# The gate promotes per segment, so check per segment.
for segment, w in tracker.by_segment().items():
    print(segment, round(w.brier_score, 4), round(w.reliability, 4))
```

---

## Which number to read

`brier_score` is the headline, but it moves with the base rate: a market that
resolves YES 95% of the time is easy to score well on. Two numbers are more
honest about calibration specifically:

- **`reliability`** — the probability-weighted mean squared gap between what you
  forecast and what occurred. **Lower is better. This is the miscalibration
  number.** It does not improve just because the base rate is extreme.
- **`skill_score`** — Brier relative to always predicting the base rate. `0.0`
  means you added nothing over the climatology; negative means you did worse
  than a constant.

`resolution` (higher is better) says how much your forecasts actually vary with
the outcome. A model that always predicts the base rate has reliability 0 —
perfectly calibrated, and perfectly useless. Read it alongside reliability.

---

## `outcome` is about the label, not your profit

This one is worth stopping on, because the field name invites the wrong reading.
`SettlementEngine` writes:

```python
outcome = "won" if trade["outcome_label"] == winning_label else "lost"
```

There is no reference to `direction`. So a `direction="no"` trade whose label
*does* occur is recorded as `"won"` with a **negative** `realized_pnl`:

```
direction=no, outcome_label=YES, winning_label=YES
  outcome      = "won"
  realized_pnl = -6.6967
```

For calibration that is exactly what we want — we are scoring a forecast about
the label, not the profitability of a position. But if you write your own
reporting on this field, `"won"` means *the thing happened*, not *we made money*.

Consequence worth internalising: **calibration and PnL are independent.** A model
can be perfectly calibrated and lose money (fees, adverse selection, sizing), or
be badly calibrated and win for a while (luck, or a favourable base rate). You
need both numbers; neither one substitutes for the other.

---

## What gets scored, and what doesn't

Only trades that carry **both** a `model_prob` and a settled `outcome`:

- **Open trades are excluded.** No truth has arrived, and including them would be
  lookahead.
- **Trades without `model_prob` are excluded** and counted in `n_skipped`. Trade
  logs written before v0.8.5 don't carry the forecast, so those rows are
  *unscoreable*, not evidence of a bad forecast. Treating a missing prediction as
  `0.0` would manufacture miscalibration that never happened.

```python
tracker = CalibrationTracker.from_path(path)
if tracker.n_skipped:
    print(f"{tracker.n_skipped} rows have no forecast recorded — pre-0.8.5 log")
```

`overall()` returns `None` when nothing is scoreable, rather than raising or
reporting a zero. An empty history is not a calibrated model.

---

## Ordering: truth time, not write time

Windows are ordered by **`target_date`**, not `recorded_at`. A trade recorded in
December about a June event belongs in June's calibration history — a forecast is
scored when its truth lands, not when someone wrote the row. This matters for any
backfilled or replayed log, where the two dates disagree.

---

## Why per-segment

Because an aggregate can hide a broken segment, and the gate promotes per
segment. 18 well-forecast politics trades plus 2 badly-forecast sports trades:

```
overall            brier 0.0855    looks fine
politics           brier 0.0025
sports             brier 0.9025    broken
```

The aggregate is the number you'd put in a dashboard. The segment breakdown is
the number that tells you to stop trading sports.

---

## Limits

- **Binary only.** One `model_prob` per trade against one label occurrence.
  Multiclass markets are scored per-outcome-label row, not jointly.
- **No alerting.** This computes numbers; deciding when a number is bad enough to
  act on is the watchdog's job (#14). Deliberately separate — thresholds are a
  policy choice, and burying one here would hide it.
- **Reliability needs volume.** With 10 bins and 20 trades, most bins hold one or
  two samples and the curve is noise. Each curve point carries `n` so you can see
  when you are reading noise; prefer `reliability` over eyeballing the curve
  until you have a few hundred realized trades.
