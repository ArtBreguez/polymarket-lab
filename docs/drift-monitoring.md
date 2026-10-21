# Drift monitoring

The holdout gate answers *was this model good on history*. [Live
calibration](live-calibration.md) answers *is it still calibrated on money that
was actually at risk*. This answers a third question, and the only one that needs
no outcomes: *is the world the model is being fed still the one it was trained
on*.

That is why it is worth having. A market resolves days or weeks after the
forecast; a feature that has moved is visible today. It is also not the same as
the model being wrong — inputs can move without hurting a robust model, and a
model can degrade with its inputs unchanged — so read it beside calibration, not
instead of it.

---

## The quick version

```python
import numpy as np
import pandas as pd

from pmlab import drift_report

rng = np.random.default_rng(0)
n = 2_000

# The panel the champion was trained on.
reference = pd.DataFrame({
    "feature_price": rng.normal(0.50, 0.10, n),
    "feature_volume": rng.lognormal(3.0, 0.5, n),
    "feature_sector": rng.choice(["politics", "sports", "crypto"], n, p=[0.5, 0.3, 0.2]),
})

# What it is being fed this week: price has drifted up, a new sector has
# appeared, and volume has started arriving blank 30% of the time.
current = pd.DataFrame({
    "feature_price": rng.normal(0.58, 0.10, n),
    "feature_volume": rng.lognormal(3.0, 0.5, n),
    "feature_sector": rng.choice(["politics", "sports", "crypto", "weather"], n,
                                 p=[0.4, 0.25, 0.15, 0.2]),
})
current.loc[rng.random(n) < 0.30, "feature_volume"] = np.nan

report = drift_report(reference, current)

print(f"overall: {report.status}  "
      f"({report.n_alert} alert, {report.n_warn} warn, {report.n_ok} ok)")
for f in report.features:
    psi = "n/a" if f.psi is None else f"{f.psi:.3f}"
    print(f"{f.name:16} {f.status:7} psi={psi:>6}  drivers={f.drivers}")

# Look inside the two that are not a plain distribution shift.
sector = next(f for f in report.features if f.name == "feature_sector")
volume = next(f for f in report.features if f.name == "feature_volume")
print("sector: unseen", sector.unseen_categories, f"= {sector.unseen_rate:.1%} of live rows")
print(f"volume: {volume.reference_missing_rate:.1%} missing -> {volume.current_missing_rate:.1%}"
      f"  (change {volume.missing_rate_change:+.3f}),  psi on what remains {volume.psi:.3f}")
```

```text
overall: alert  (3 alert, 0 warn, 0 ok)
feature_price    alert   psi= 0.649  drivers=('psi', 'kl')
feature_sector   alert   psi= 1.595  drivers=('psi', 'kl')
feature_volume   alert   psi= 0.017  drivers=('missing_rate',)
sector: unseen ('weather',) = 20.3% of live rows
volume: 0.0% missing -> 28.1%  (change +0.281),  psi on what remains 0.017
```

Three different things went wrong, and the report says which kind:

- **`feature_price`** moved: the whole distribution shifted. `psi` and `kl` fire.
- **`feature_sector`** has a category the training panel never had.
- **`feature_volume`** looks *fine* — its PSI on the values that remain is tiny —
  but about 30% of it is blank. Only the missing-rate signal fires, which is why
  it is a separate signal and not folded into the distribution.

---

## Reading a report

`drift_report` returns a `DriftReport`; each feature is a `FeatureDrift`. Both
are frozen dataclasses with `to_dict()` / `from_dict()`.

Every feature gets one status, and so does the report (the worst of its
features and, if you supplied one, the prediction):

| status | meaning |
|---|---|
| `ok` | measured, and below the warn band on every signal |
| `warn` | at or above the warn threshold on at least one signal |
| `alert` | at or above the alert threshold on at least one signal |
| `insufficient` | could not be measured — see [Too little data](#too-little-data) |

Precedence is **alert > warn > insufficient > ok**. `insufficient` outranks `ok`
on purpose: *we could not check* must never read as *we checked and it is fine*.
It does not outrank a measured problem, so one broken feature is never hidden by
another that is merely unmeasurable.

`drivers` names the signals that set the status — any of `"psi"`, `"kl"`,
`"missing_rate"` — and is empty when the status is `ok`.

| field | what it is |
|---|---|
| `psi` | Population Stability Index of the non-missing values. `None` if insufficient. |
| `kl` | KL(current ‖ reference), in nats. `None` if insufficient. |
| `reference_missing_rate`, `current_missing_rate`, `missing_rate_change` | share of missing values on each side, and `current − reference` (signed) |
| `unseen_rate` | share of live values the reference never contained (see below) |
| `n_unseen_categories`, `unseen_categories` | for categorical features: how many, and the most frequent (listing capped at 20) |
| `n_reference`, `n_current` | rows on each side |
| `n_reference_valid`, `n_current_valid` | non-missing values on each side — what the statistics were computed from |
| `kind` | `"numeric"` or `"categorical"` |

On the report itself: `features` (ordered by name), `prediction`,
`absent_from_current` (assessed features the live table does not have at all —
scored as 100% missing), `new_in_current` (columns the reference lacks —
informational), and `config`, so a stored report records the thresholds it was
judged against.

---

## Which number to read

**`psi`** is the headline. It is `0` for identical distributions, grows with the
shift, and has conventional bands: below 0.10 nothing meaningful, 0.10–0.25
moderate, above 0.25 significant.

**`kl`** says which *direction* the shift went. It is **KL(current ‖ reference)**
— `Σ cᵢ ln(cᵢ / rᵢ)` over bins, where `c` is the live distribution and `r` the
training one. Read it as: *how surprised would a model trained on the reference
be by this live data?* It punishes live mass where training had little, and
barely notices training mass that has vanished. PSI is the two directions added
together, `psi = KL(cur ‖ ref) + KL(ref ‖ cur)`, so PSI is direction-free and `kl`
is the half that matters for a model that has to cope with what it is fed.

```python
from pmlab import drift_report

ref = ["a"] * 500 + ["b"] * 500
cur = ["a"] * 900 + ["b"] * 100

fwd = drift_report({"x": ref}, {"x": cur}).features[0]
rev = drift_report({"x": cur}, {"x": ref}).features[0]   # arguments swapped
print(f"drift_report(ref, cur):  psi {fwd.psi:.4f}  kl {fwd.kl:.4f}")
print(f"drift_report(cur, ref):  psi {rev.psi:.4f}  kl {rev.kl:.4f}")
print(f"the two KLs add up to PSI: {fwd.kl + rev.kl:.4f}")

# Live traffic has a category the training panel never saw.
new = ["a"] * 400 + ["b"] * 400 + ["new"] * 200
fwd = drift_report({"x": ref}, {"x": new}).features[0]
rev = drift_report({"x": new}, {"x": ref}).features[0]
print(f"unseen category:  kl(cur || ref) {fwd.kl:.4f}   kl(ref || cur) {rev.kl:.4f}")
```

```text
drift_report(ref, cur):  psi 0.8789  kl 0.3681
drift_report(cur, ref):  psi 0.8789  kl 0.5108
the two KLs add up to PSI: 0.8789
unseen category:  kl(cur || ref) 1.3418   kl(ref || cur) 0.2223
```

**`missing_rate_change`** is `current − reference`, classified by its absolute
value: a feature that *stopped* being missing is a change in the pipeline too.

**`unseen_rate`** is how much of the live data the reference never contained.
For a categorical feature that is the share in unseen categories, and it is
already inside `psi` and `kl`. For a numeric feature it is the share outside the
reference's min/max, and it is **informational** — see
[Bins come from the reference only](#bins-come-from-the-reference-only) for why
it exists.

---

## Bins come from the reference only

Numeric features are cut at the **quantiles of the reference sample** (ten bins by
default, open-ended at both ends), and those same edges are reused for the live
sample. This is not a detail. There are two ways to get it wrong:

- Bin each sample on **its own** quantiles, and both come out roughly uniform. A
  distribution shifted wholesale then scores about zero — drift disappears exactly
  when it is largest.
- Take the shared edges from the **current** sample, and the statistic depends on
  the very sample being judged.

```python
import numpy as np

from pmlab import drift_report

rng = np.random.default_rng(0)
ref = rng.normal(0, 1, 2000)
cur = rng.normal(3, 1, 2000)            # same shape, shifted by 3 standard deviations

fd = drift_report({"x": ref}, {"x": cur}).features[0]
print(f"reference bins:    psi {fd.psi:.2f}  ({fd.status})")

def own_quantile_shares(x):             # the trap: each sample on its own quantiles
    edges = np.quantile(x, np.linspace(0, 1, 11)[1:-1])
    return np.bincount(np.searchsorted(edges, x), minlength=10) / len(x)

r, c = own_quantile_shares(ref), own_quantile_shares(cur)
print(f"own-quantile bins: psi {np.sum((c - r) * np.log(c / r)):.2f}")
```

```text
reference bins:    psi 6.59  (alert)
own-quantile bins: psi 0.00
```

The cost of open-ended end bins is that a tail beyond anything in training is
folded into the first or last bin. `unseen_rate` is there so that is not
invisible: it is the share of live values outside the reference's min/max.
A feature with a modest PSI and a large `unseen_rate` is telling you the model is
being asked to extrapolate.

---

## Empty bins: large, never infinite

`p · ln(p / 0)` is infinite, and a value range the reference never saw is
precisely the drift that matters most. So each bin's proportion is **floored at
`epsilon`** (default `1e-4`) and the vector is rescaled to sum to one. A range
the training panel never contained scores large but finite — never `inf` or `NaN`
— and the report stays valid JSON.

```python
import numpy as np

from pmlab import DriftConfig, drift_report

rng = np.random.default_rng(0)
ref = rng.normal(0, 1, 2000)
cur = rng.normal(0, 1, 2000) + 100      # nowhere near anything in training

for eps in (1e-2, 1e-4, 1e-6):
    fd = drift_report({"x": ref}, {"x": cur}, config=DriftConfig(epsilon=eps)).features[0]
    print(f"epsilon {eps:g}:  psi {fd.psi:5.2f}   kl {fd.kl:.2f}   unseen_rate {fd.unseen_rate:.0%}")
```

```text
epsilon 0.01:  psi  3.76   kl 1.84   unseen_rate 100%
epsilon 0.0001:  psi  8.28   kl 2.29   unseen_rate 100%
epsilon 1e-06:  psi 12.43   kl 2.30   unseen_rate 100%
```

`epsilon` is a knob on how loud "never seen" is: PSI grows roughly with
`ln(1/epsilon)`. Bins that are empty on *both* sides are omitted rather than
floored, so ties in the reference (a feature that is 70% zeros) do not add noise.

---

## Missing values

`NaN`, `None` and (for numeric features) `±inf` are **excluded from the
distribution but counted**. A feature going from 0% to 40% missing is drift even
though every remaining value looks fine; dropping the blanks silently would report
"no drift" on a broken feed. The missing-rate signal is classified separately, by
the absolute change in the missing share (defaults: warn at 0.05, alert at 0.20).

A column that goes *entirely* blank is an `alert` with `psi = None` — there is no
distribution left to compare, and that is not the same as "insufficient". A column
absent from the live table is treated the same way and also listed in
`absent_from_current`.

---

## Categorical features

Categorical features are compared by category frequency. A category the reference
never had is **drift, not noise to discard**: all unseen categories are pooled
into one bin whose reference proportion is zero (so floored at `epsilon`), and
`unseen_categories` lists them by live frequency. A category that *vanished* from
live data is drift too — it shows up as reference mass with nothing opposite it.

**Numeric vs categorical is decided from the reference column**, never from the
live one — a feed that dies and starts emitting all-`None` is still a numeric
feature that went missing, not a categorical one:

- integers and floats are numeric (including object columns of numbers with
  `None`);
- strings, booleans, categoricals and everything else are categorical.

Override it where the dtype lies — integer-coded categories are the usual case:

```python
from pmlab import DriftConfig, drift_report

codes = [0, 1, 2, 3] * 250
for cfg in (DriftConfig(), DriftConfig(categorical=("sector_code",))):
    fd = drift_report({"sector_code": codes}, {"sector_code": codes}, config=cfg).features[0]
    print(f"{fd.name}: {fd.kind}")
```

```text
sector_code: numeric
sector_code: categorical
```

A feature named in `numeric` or `categorical` that the reference does not have
raises `ValueError` rather than being ignored: a typo there would otherwise leave
the real feature silently on auto-detect.

---

## Too little data

Below `min_samples` usable values on either side the statistics are **`None`**,
not a small number, and the status is `insufficient`. The default is **300**, and
it is not arbitrary: PSI of an *unchanged* feature is not zero, it has a noise
floor of roughly `(n_bins − 1) / n` for `n` live values. Simulating 500 windows of
a feature that did not change at all:

```python
import numpy as np

from pmlab import DriftConfig, drift_report

print("live values   median PSI   p95 PSI   windows at or above the 0.10 warn line")
for n_cur in (100, 200, 300, 500, 1000):
    scores = []
    for seed in range(500):
        rng = np.random.default_rng(seed)
        ref, cur = rng.normal(0, 1, 2000), rng.normal(0, 1, n_cur)
        fd = drift_report({"x": ref}, {"x": cur}, config=DriftConfig(min_samples=50)).features[0]
        scores.append(fd.psi)
    s = np.array(scores)
    print(f"{n_cur:>11}   {np.median(s):10.3f}   {np.percentile(s, 95):7.3f}   "
          f"{int((s >= 0.10).sum()):>3} of 500")
```

```text
live values   median PSI   p95 PSI   windows at or above the 0.10 warn line
        100        0.090     0.179   204 of 500
        200        0.045     0.092    14 of 500
        300        0.032     0.068     3 of 500
        500        0.021     0.042     0 of 500
       1000        0.012     0.025     0 of 500
```

At 100 live values, two windows in five would warn about a feature that never
moved. At the default of 300, three in five hundred. If you change `n_bins`,
change `min_samples` with it — the floor scales with the number of bins.

Two details. Only *usable* values count: 500 rows of which 20 are non-blank is
`insufficient` for the distribution, but the 96% missing rate is still measured
and still alerts. And the rows-vs-values distinction applies to the missing-rate
signal too, which needs `min_samples` *rows* on each side before it classifies.

---

## Prediction drift

Inputs can each look unchanged while the model's *output* moves — small shifts
compound through a model — and the output is what you trade on. Pass the model's
predicted probabilities on the training panel and on the live window and the same
machinery runs on them, reported separately as `report.prediction`:

```python
import numpy as np
import pandas as pd

from pmlab import drift_report

rng = np.random.default_rng(0)
cols = ["feature_a", "feature_b", "feature_c", "feature_d"]
reference = pd.DataFrame({c: rng.normal(0.50, 0.10, 2000) for c in cols})
current = pd.DataFrame({c: rng.normal(0.52, 0.10, 2000) for c in cols})  # each moved a fifth of an sd

def model(df):                          # stand-in for champion.predict_proba(...)[:, 1]
    return 1 / (1 + np.exp(-10 * (df[cols] - 0.5).sum(axis=1)))

report = drift_report(
    reference, current,
    reference_predictions=model(reference),
    current_predictions=model(current),
)
for f in report.features:
    print(f"input      {f.name}: psi {f.psi:.3f}  {f.status}")
p = report.prediction
print(f"prediction {p.name}: psi {p.psi:.3f}  {p.status}")
print(f"report: {report.status}")
```

```text
input      feature_a: psi 0.082  ok
input      feature_b: psi 0.047  ok
input      feature_c: psi 0.033  ok
input      feature_d: psi 0.052  ok
prediction prediction: psi 0.165  warn
report: warn
```

Here every input is below the warn line, and the report's status of `warn` comes
from the prediction alone. (A single-feature model that is monotone in its input
would show identical PSI for input and output — quantile bins map one-to-one
through a monotone function — so the prediction signal earns its keep once a model
combines features.)

Both arrays are required together. Score the **whole live window** with the model.
Do not use `model_prob` from the paper-trade log as the live predictions: the log
holds only the markets the strategy chose to trade, a selected subset, and
comparing it with predictions on every training row would flag the selection as
drift.

---

## Reading the live window from the snapshot store

`drift_report` is a pure function over two tables. `drift_report_from_store`
builds the live table for you from a `FeatureSnapshotStore`, and it does so
**point-in-time**: the window is read through the store's own `as_of`, so for each
market it holds the latest snapshot captured **strictly before `cutoff`**.
Nothing captured at or after the cutoff can leak in.

`window_start` is required, not optional. Without a lower bound "current" would
include the very markets the model was trained on, and drift would be understated.

```python
import tempfile

import numpy as np
import pandas as pd

from pmlab import drift_report_from_store
from pmlab.data import FeatureSnapshotStore, build_panel_from_snapshots

rng = np.random.default_rng(0)
store = FeatureSnapshotStore(tempfile.mkdtemp())

# Training era: 400 markets captured in January, resolved after the 10th.
store.append("politics", [
    {"market_id": f"t{i}", "captured_at": "2025-01-05T00:00:00Z",
     "feature_price": float(rng.normal(0.50, 0.10))} for i in range(400)])
resolutions = pd.DataFrame({
    "market_id": [f"t{i}" for i in range(400)],
    "decision_date": "2025-01-10",
    "winning_label": rng.choice(["Yes", "No"], 400),
})
panel = build_panel_from_snapshots(store, "politics", resolutions)

# Live era: 400 markets captured in early March, then re-captured in April
# after a price jump.
store.append("politics", [
    {"market_id": f"l{i}", "captured_at": "2025-03-05T00:00:00Z",
     "feature_price": float(rng.normal(0.50, 0.10))} for i in range(400)])
store.append("politics", [
    {"market_id": f"l{i}", "captured_at": "2025-04-05T00:00:00Z",
     "feature_price": float(rng.normal(0.90, 0.02))} for i in range(400)])

for cutoff in ("2025-03-10", "2025-04-10"):
    report = drift_report_from_store(
        store, "politics", panel, window_start="2025-03-01", cutoff=cutoff)
    f = report.features[0]
    print(f"cutoff {cutoff}: {report.n_current} live markets  {f.name} {f.status}  psi {f.psi:.3f}")
```

```text
cutoff 2025-03-10: 400 live markets  feature_price ok  psi 0.039
cutoff 2025-04-10: 400 live markets  feature_price alert  psi 8.281
```

At the March cutoff the April re-captures do not exist as far as the report is
concerned; move the cutoff past them and the same call sees the jump. Features
default to the panel's `feature_*` columns; pass `features=` to override. An empty
window gives a report whose features are all `insufficient` rather than an error —
no live data is not evidence the world is fine.

To get prediction drift here, pass `predict_fn`, called with the full reference
panel and the full live window and returning one probability per row. It is not
called on an empty window. The champion manifest does not record the feature list
its model was trained on, so pass `features=` explicitly if the panel carries
`feature_*` columns the model does not use — a loaded forecaster's
`feature_names()` is the natural source.

---

## Thresholds are configuration, not constants

The classifier is in `DriftThresholds`, carried on `DriftConfig`, and written into
every report:

| signal | warn | alert | where the default comes from |
|---|---|---|---|
| PSI | 0.10 | 0.25 | the widely used credit-scoring convention |
| KL(cur ‖ ref) | 0.05 | 0.125 | **derived, not a standard**: half the PSI bands, because for small shifts PSI ≈ 2 · KL |
| missing-rate change (absolute) | 0.05 | 0.20 | **judgment**; there is no convention |

A value at or above a threshold is in that band. These are starting points:
features differ in how noisy they are, so one set of bands will suit some better
than others — tune on your own history. Validation rejects negative, non-finite or
inverted (`warn > alert`) thresholds. To take a signal out of the decision, set
both its thresholds high — it is still reported.

```python
import numpy as np

from pmlab import DriftConfig, DriftThresholds, drift_report

rng = np.random.default_rng(0)
ref, cur = rng.normal(0.50, 0.10, 2000), rng.normal(0.58, 0.10, 2000)

lenient = DriftConfig(thresholds=DriftThresholds(
    psi_warn=1.0, psi_alert=2.0, kl_warn=1.0, kl_alert=2.0))
for label, cfg in (("default", DriftConfig()), ("lenient", lenient)):
    f = drift_report({"price": ref}, {"price": cur}, config=cfg).features[0]
    print(f"{label:8} psi {f.psi:.3f}  -> {f.status:6} drivers={f.drivers}")
```

```text
default  psi 0.655  -> alert  drivers=('psi', 'kl')
lenient  psi 0.655  -> ok     drivers=()
```

---

## Logging to an experiment tracker

`to_dict()` is strict JSON — no `inf`, `NaN` or numpy scalars — and
`DriftReport.from_dict()` restores it exactly. The top level carries flat numbers
(`n_alert`, `n_warn`, `n_ok`, `n_insufficient`, `max_psi`, `max_kl`,
`prediction_psi`, `prediction_kl`) because trackers such as MLflow log only
top-level numerics; the per-feature detail rides along as nested lists.

```python
import tempfile

import numpy as np

from pmlab import DriftReport, LocalJSONTracker, drift_report

rng = np.random.default_rng(0)
report = drift_report(
    {"price": rng.normal(0.50, 0.10, 2000), "volume": rng.lognormal(3, 0.5, 2000)},
    {"price": rng.normal(0.58, 0.10, 2000), "volume": rng.lognormal(3, 0.5, 2000)},
)

tracker = LocalJSONTracker(tempfile.mkdtemp())
with tracker.run("drift", "nightly") as run:
    run.log_metrics(report)

(record,) = tracker.list_runs("drift")
flat = {k: round(v, 3) if isinstance(v, float) else v
        for k, v in record.metrics.items()
        if k != "prediction" and not isinstance(v, (list, dict))}
print(flat)
print("restored exactly:", DriftReport.from_dict(report.to_dict()) == report)
```

```text
{'status': 'alert', 'n_reference': 2000, 'n_current': 2000, 'n_ok': 1, 'n_warn': 0, 'n_alert': 1, 'n_insufficient': 0, 'max_psi': 0.686, 'max_kl': 0.33, 'prediction_psi': None, 'prediction_kl': None}
restored exactly: True
```

The report is deterministic — same inputs, identical report, no randomness — and
features are ordered by name whatever order the columns arrived in, so two runs
diff cleanly.

---

## Limits

- **Classification, not action.** This computes numbers and sorts them into bands;
  deciding what to *do* about an alert — pause trading, retrain, page someone — is
  the watchdog's job (#14). The bands are a policy choice, which is why they are
  explicit configuration.
- **Drift is not degradation.** An alert says the inputs moved, not that the model
  got worse. Pair it with [live calibration](live-calibration.md), which says
  whether the forecasts are still right once outcomes arrive.
- **One feature at a time.** Each feature is compared on its own. Two features can
  each look unchanged while the relationship between them has broken; PSI will not
  see that.
- **Tails fold into the end bins.** Watch `unseen_rate` on numeric features.
- **Do not include identifiers or timestamps.** A `market_id` or a date is
  all-unseen by construction and will read as total drift. Pass `features=`. The
  same goes for very high-cardinality categoricals, where most categories have too
  few rows to compare.
- **The live window should not overlap training.** `window_start` is yours to
  choose; a window that includes training-era markets dilutes the signal.
- **A statistic on a sample.** PSI has a noise floor (see
  [Too little data](#too-little-data)). A single `warn` just above the line on a
  small window is weak evidence; a `warn` that persists across windows is not.
- **Binary forecasts.** Prediction drift is one probability column; a multiclass
  market is monitored per outcome label.
