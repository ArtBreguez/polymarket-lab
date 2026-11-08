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

The report says what was scored and how each feature did:

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
    print(f"  {name:22s} {score:.3f}")
print("leaked:", report.leaked or "none")
```

```text
scored against: outcome_label == winning_label
  feature_climo_tmax     0.500
  feature_cloud_cover    0.500
  feature_dew_point      0.500
  feature_ens_spread     0.500
  feature_fcst_tmax      0.500
  feature_fcst_tmin      0.500
  feature_humidity       0.500
  feature_wind_speed     0.500
  feature_market_price   0.646
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

## What it cannot see

The check is univariate, and that has one consequence on this shape. A feature that is **the same
for every bin of a market** (a forecast, the plugin's own features above) scores exactly 0.5
against the per-row outcome, because it cannot tell a market's winning bin from its losing ones.
It cannot leak on its own, and the check cannot tell an honest forecast from a copy of the
realised value if both are constant across a market's bins. It becomes a leak when something
makes it bin-relative (a bin's distance to the *realised* temperature, say), and that feature
varies across bins, so it is scored and caught.

Whether a market-level feature is honest is a question about *when* it was captured, not about
its values. That is what `FeatureSnapshotStore.as_of` (features strictly before the cutoff) and
the price-timing guard (`check_price_timing`, which `build_panel` always runs on stamped rows)
enforce.

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
