# The leakage check on a ranged-market panel

`check_no_leakage` scores each `feature_*` column on its own against a binary target and flags
any that separate it better than `max_auc` (0.97): a feature that predicts the outcome that well
was almost certainly derived from it. `build_panel(rows, check_leakage=True)` runs it on the
panel it assembles.

## What it scores against

The target is the one the model is trained on, which depends on the shape of the panel:

| Panel | Target | `LeakageReport.target` |
|-------|--------|------------------------|
| **One row per bin**: a market is several rows at one decision date, one per outcome bin (what `MarketPlugin.build_training_rows` returns) | `outcome_label == winning_label`, per row: did *this bin* win. It is what `rolling_origin_eval` fits | `"outcome_label == winning_label"` |
| One row per market | `winning_label` | `"winning_label"` |
| Any shape, with `label_col="..."` | that column, as given | the column name |

Releases before 0.11.2 scored every panel against `winning_label`, and on a one-row-per-bin panel
that is the wrong question (#35). `winning_label` is the same for all nine bins of a market, so
the score became "does this feature tell *which market* resolved to the rarest label", and an
honest forecast answers that well: on the whole bundled sample `build_panel(check_leakage=True)`
raised `LeakageError` on four of its eight forecast features (`feature_fcst_tmax` scored 0.999),
features that were doing their job. The check now asks what the model learns, "did this bin win",
and an honest forecast is no longer taken for a leak. A panel is "one row per bin" when some
market has more than one distinct `outcome_label` at one `decision_date` (the grouping the
backtest chooses within).

## Worked example

The bundled sample (`pmlab.datasets.load_sample`, offline) is a ranged-market panel: 9 bins per
market. The blocks run in order, as written.

```python
from pmlab import build_panel
from pmlab.datasets import load_sample

sample = load_sample()
resolver = sample.plugin()  # the plugin once everything has resolved: it labels the panel
rows = []
for day in sample.days[:60]:
    for spec in sample.plugin(day).discover_markets():
        rows.append(
            resolver.build_training_rows(spec, "previous_evening", decision_time=f"{day}T12:00:00+00:00")
        )

# One row per bin; check_leakage=True scores each feature against "did this bin win".
panel = build_panel(rows, check_leakage=True)
print(len(panel), "rows,", panel["market_id"].nunique(), "markets, 9 bins each")
```

```text
2700 rows, 300 markets, 9 bins each
```

The report says what was scored and how each feature did. The plugin's forecast features are the
same for every bin of a market, so they are scored by their distance to each bin (next section);
the market price varies across the bins and is scored as it is:

```python
from pmlab import build_panel, check_no_leakage
from pmlab.datasets import load_sample

sample = load_sample()
resolver = sample.plugin()
rows = [
    resolver.build_training_rows(spec, "previous_evening", decision_time=f"{day}T12:00:00+00:00")
    for day in sample.days[:60]
    for spec in sample.plugin(day).discover_markets()
]
panel = build_panel(rows)
panel["feature_market_price"] = panel["market_price"]  # varies across a market's bins

report = check_no_leakage(panel, raise_on_leak=False)
print("scored against:", report.target)
for name, score in report.scores.items():
    how = "distance to the bins" if name in report.bin_relative else "as is"
    print(f"  {name:22s} {score:.3f}  ({how})")
print("leaked:", report.leaked or "none")
```

```text
scored against: outcome_label == winning_label
  feature_climo_tmax     0.512  (distance to the bins)
  feature_cloud_cover    0.540  (distance to the bins)
  feature_dew_point      0.527  (distance to the bins)
  feature_ens_spread     0.535  (distance to the bins)
  feature_fcst_tmax      0.698  (distance to the bins)
  feature_fcst_tmin      0.527  (distance to the bins)
  feature_humidity       0.522  (distance to the bins)
  feature_wind_speed     0.540  (distance to the bins)
  feature_market_price   0.646  (as is)
leaked: none
```

A real leak is still caught. Here a feature is a copy of the outcome (the winning bin gets 1, the
others 0), and the check names it:

```python
from pmlab import LeakageError, build_panel, check_no_leakage
from pmlab.datasets import load_sample

sample = load_sample()
resolver = sample.plugin()
rows = [
    resolver.build_training_rows(spec, "previous_evening", decision_time=f"{day}T12:00:00+00:00")
    for day in sample.days[:60]
    for spec in sample.plugin(day).discover_markets()
]
panel = build_panel(rows)

# A feature that is a copy of the outcome: the winning bin gets 1, the others 0.
panel["feature_oops"] = (panel["outcome_label"] == panel["winning_label"]).astype(float)
try:
    check_no_leakage(panel)
except LeakageError as err:
    print("LeakageError:", err)
```

```text
LeakageError: Potential label leakage in feature column(s): feature_oops (score=1.000). These separate 'outcome_label == winning_label' above max_auc=0.97. If legitimate, raise max_auc or drop the column.
```

## A market-level copy of the realised value

A feature that is **the same for every bin of a market** (a forecast, the plugin's own features
above) cannot separate the per-row outcome by its raw value: it scores exactly 0.5, because it
cannot tell a market's winning bin from its losing ones. Through 0.12.0 that left a hole. A
column holding the *realised* value on every bin row (`feature_realised_tmax`, say) is a perfect
leak and scored 0.5 too (#39).

Since 0.12.1 such a feature is also scored by where it sits relative to each bin: its distance to
the bin, which is 0 when the value is inside the bin and otherwise the gap to the bin's nearest
end. That distance is scored against the per-row outcome exactly like any other feature (same
fold, same `max_auc`). A copy of the realised value is inside the winning bin and outside every
other, so it scores about 1.0; an honest forecast is only sometimes nearest to the bin that
wins, and scores well below (`feature_fcst_tmax`, the sample's best forecast, 0.70). The larger
of the two scores is the feature's score, and `LeakageReport.bin_relative` lists the features
scored this way.

```python
from pmlab import LeakageError, build_panel, check_no_leakage
from pmlab.datasets import load_sample

sample = load_sample()
resolver = sample.plugin()
rows = [
    resolver.build_training_rows(spec, "previous_evening", decision_time=f"{day}T12:00:00+00:00")
    for day in sample.days[:60]
    for spec in sample.plugin(day).discover_markets()
]
panel = build_panel(rows)

# A market-level copy of the realised value: the same number on every bin row of a market.
panel["feature_realised_tmax"] = (
    panel["winning_label"].str.extract(r"(-?\d+)", expand=False).astype(float)
)
try:
    check_no_leakage(panel)
except LeakageError as err:
    print("LeakageError:", err)
```

```text
LeakageError: Potential label leakage in feature column(s): feature_realised_tmax (score=1.000, by its distance to each bin). These separate 'outcome_label == winning_label' above max_auc=0.97. If legitimate, raise max_auc or drop the column.
```

**Which features.** Those that take one value across the bins of every (`market_id`,
`decision_date`), the grouping the backtest chooses within. A feature that varies across the bins
is scored as before, by its raw value. A frame that is not per-bin, or an explicit `label_col`,
gets no distance score at all: that target is not "did this bin win".

**Where the bins come from.** The panel carries each bin's `outcome_label`, not numeric bounds,
so the numbers are read from the label. These forms are understood (units, `%`, currency signs
and thousands separators are ignored, `-` and `−` both read as a minus):

| Label | Read as |
|-------|---------|
| `18°C`, `-3°C`, `18.5` | the point 18, -3, 18.5 |
| `28-30°C`, `28 to 30` | the range 28 to 30 |
| `<=14°C`, `<14`, `10°C or below`, `below 10°C`, `at most 10` | everything up to that number |
| `>=22°C`, `>22`, `38°C or above`, `38°C or higher`, `38+`, `≥38`, `at least 38` | everything from that number up |

An open-ended bin needs no centre, which it does not have: the distance to `>=22°C` is 0 for any
value from 22 up and the gap to 22 below it. A label that reads as none of these leaves its row
out of the distance score (a panel whose labels are all names is left as it was).

**Why distance to the bin, and not to its centre.** The issue that asked for this proposed
`-|feature - bin_center|`. On bins of one width the two rank alike (`feature_fcst_tmax` scores
0.72 on the whole sample either way). The centre needs a number for an open-ended bin, and ranks
the wrong bin first when widths differ: a range 8 degrees across has its centre 4 degrees from
either end, so a copy of the realised value at the edge of the range is nearer the centre of the
1-degree bin beside it (1 degree) than the centre of its own bin (4). The distance to the bin is
0 there. The same goes for a copy deep inside an open-ended bin, which is far from the bin's
bound. On a panel of 1-degree bins with open tails the centre scores such a copy 0.984, and with a
wide range among narrow bins 0.901, under the 0.97 line, where the distance scores 1.000.
A plain containment test, "is the value inside the bin?", needs neither but is coarser: a copy
that is a third of a bin off scores about 0.95 and escapes, where the distance scores 0.995.

**How exact the copy must be.** The check measures how well a feature picks the winning bin,
so a copy with noise is caught while the noise is small next to the bin width, and an honest
forecast, which is wrong by more than that, is not. On the 1-degree bins of the sample:

```python
import numpy as np

from pmlab import build_panel, check_no_leakage
from pmlab.datasets import load_sample

sample = load_sample()
resolver = sample.plugin()
rows = [
    resolver.build_training_rows(spec, "previous_evening", decision_time=f"{day}T12:00:00+00:00")
    for day in sample.days[:60]
    for spec in sample.plugin(day).discover_markets()
]
panel = build_panel(rows)

# The realised value on every bin row, plus a different error for each market.
ids = panel["market_id"]
exact = panel["winning_label"].str.extract(r"(-?\d+)", expand=False).astype(float)
shift = dict(zip(ids.unique(), np.random.default_rng(0).normal(size=ids.nunique())))
for sigma in (0.0, 0.2, 0.3, 0.5, 1.0):
    noisy = panel.assign(feature_realised_tmax=exact + sigma * ids.map(shift))
    score = check_no_leakage(noisy, raise_on_leak=False).scores["feature_realised_tmax"]
    print(f"  copy + noise of {sigma:.1f} degrees: {score:.3f}")
```

```text
  copy + noise of 0.0 degrees: 1.000
  copy + noise of 0.2 degrees: 1.000
  copy + noise of 0.3 degrees: 0.995
  copy + noise of 0.5 degrees: 0.966
  copy + noise of 1.0 degrees: 0.879
```

## What it still cannot see

- **A feature not in the labels' unit.** The distance is measured in the labels' own numbers. The
  realised temperature in Fahrenheit against bins in Celsius, or a rescaled copy, is far from
  every bin and scores near 0.5. The check cannot guess a transformation.
- **Labels that are not numbers.** A per-bin panel whose bins are names has nothing to measure a
  value against; its market-level features stay at 0.5.
- **A blend.** A feature that is only partly the realised value scores in between, and the
  threshold decides.
- **A forecast that is very good for coarse bins.** The score rises as the bins get wide
  compared with the forecast's error: an honest forecast then lands in the right bin most of
  the time. 0.97 was set with 1-degree bins in mind. If a legitimate forecast of a market with
  wide bins is refused, the error says which feature and its score; raise `max_auc` for it
  knowingly.

Whether a market-level feature is honest is, in the end, a question about *when* it was captured,
not about its values. That is what `FeatureSnapshotStore.as_of` (features strictly before the
cutoff) and the price-timing guard (`check_price_timing`, which `build_panel` always runs on
stamped rows) enforce, and the distance score is a second line behind them.

## Labels arrive after the decision

A row is a decision and its label. The two are not at the same time: the `weather_tmax`
`previous_evening` horizon decides on day D about the temperature of D+1, so the label of the row
decided on D is not known until D+1 is over. A walk-forward that trains on "every row decided before
today" therefore trains, today, on a label that does not exist yet: the row decided yesterday is about
today.

`rolling_origin_eval` (and so `TunedForecaster`, which scores every trial with it) trains a fold only
on rows whose label was **known when the fold decided**:

* a row's label is known at its `market_close_ts`, the instant its market ends (the plugin writes
  it on every row, and `build_panel` already checks it);
* the fold decides at its earliest `decision_ts` (the start of its `decision_date` when the panel
  has none);
* a row with no readable `market_close_ts`, or a panel without the column, is assumed to resolve the
  day after its decision, so it trains a fold from two days later. If your labels arrive later than
  that, put the time the label became known in `market_close_ts`: the rule then follows the data and
  holds back as many days as your markets need.

**What the rule trusts.** `market_close_ts` must be no earlier than the moment the label can be
computed. The weather plugin takes the earlier of the market's own close and the end of its target
day, so a market that closes at noon on its target day counts as known from noon, although that day's
maximum is final only at midnight. That matters only to a fold that decides later in that same day;
a fold that decides in the morning, as the sample's does, is exact. If your markets close before their
outcome exists and you decide in the afternoon, write the later time in `market_close_ts`.

Releases before 1.0.3 trained on `decision_date < eval_date` and so carried one decision day of
lookahead with the sample's next-day labels; the
[changelog](https://github.com/ArtBreguez/polymarket-lab/blob/main/CHANGELOG.md) says what changed.

The same applies to a model you train to go live, which is not a fold, so nothing holds the row back
for you: fit it on the rows known when you will decide, not on every row you have.

```python
import pandas as pd
from pmlab import build_panel
from pmlab.datasets import load_sample

sample = load_sample()
resolver = sample.plugin()
rows = [
    resolver.build_training_rows(spec, "previous_evening", decision_time=f"{day}T12:00:00+00:00")
    for day in sample.days[:20]
    for spec in sample.plugin(day).discover_markets()
]
panel = build_panel(rows)

# What a model that decides at the last decision of this panel may train on:
decide_at = pd.to_datetime(panel["decision_ts"], utc=True).max()
known = pd.to_datetime(panel["market_close_ts"], utc=True) <= decide_at
print(f"{int(known.sum())} of {len(panel)} rows are labelled by {decide_at:%Y-%m-%d %H:%M} UTC")
last_known = panel.loc[known, "decision_date"].max()
print("latest decision day with a label:", last_known, "| this decision's own day:", panel["decision_date"].max())
```

```text
810 of 900 rows are labelled by 2025-04-20 12:00 UTC
latest decision day with a label: 2025-04-18 | this decision's own day: 2025-04-20
```

The day before the decision (April 19) is missing too: its market ends at 23:00 that evening, after
the 12:00 decision. This is the row the old rule trained on.

## Choosing the target yourself

Pass `label_col` to score against a column of your own. It is used as given, on any shape, so
code that built a binary label to get around #35 keeps working, and `label_col="winning_label"`
asks for the pre-0.11.2 target:

```python
from pmlab import build_panel, check_no_leakage
from pmlab.datasets import load_sample

sample = load_sample()
resolver = sample.plugin()
rows = [
    resolver.build_training_rows(spec, "previous_evening", decision_time=f"{day}T12:00:00+00:00")
    for day in sample.days
    for spec in sample.plugin(day).discover_markets()
]
panel = build_panel(rows)

# A label of your own, scored as given (what the tutorial did before 0.11.2):
won = (panel["outcome_label"] == panel["winning_label"]).astype(int)
print("won:", check_no_leakage(panel.assign(won=won), label_col="won").leaked or "none")

# The pre-0.11.2 target, on request: the forecasts look like leaks again.
old = check_no_leakage(panel, label_col="winning_label", raise_on_leak=False)
print("winning_label:", old.leaked)
```

```text
won: none
winning_label: ['feature_cloud_cover', 'feature_fcst_tmax', 'feature_fcst_tmin', 'feature_humidity']
```
