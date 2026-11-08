# Feature Engineering — pmlab.features

`pmlab.features` provides common data transforms so plugins don't reimplement them. All functions return a **copy** — originals are never mutated.

---

## Installation

`pmlab.features` is included in `pmlab`. No extra dependencies.

```python
from pmlab import (
    add_lags, add_rolling_stats, encode_cyclical, encode_onehot, clip_outliers,
    add_missing_indicators, add_interactions, TargetEncoder, FittedTargetEncoder,
)
# or
from pmlab.features.transforms import add_lags, ...
from pmlab.features.target_encoding import TargetEncoder, FittedTargetEncoder
```

---

## add_lags

Adds lag features. New columns are named `{col}_lag{n}`.

```python
df = add_lags(
    df,
    cols=["temp", "humidity"],
    lags=[1, 3, 7],
    group_by="city",     # optional — shift within each group
    fill_value=float("nan"),
)
# Adds: temp_lag1, temp_lag3, temp_lag7, humidity_lag1, ...
```

**Panel data tip:** always pass `group_by` when your DataFrame contains multiple entities (cities, markets) — otherwise lag-1 of city B's first row will be city A's last row.

---

## add_rolling_stats

Adds rolling statistics. New columns are named `{col}_roll{window}_{stat}`.

```python
df = add_rolling_stats(
    df,
    cols=["temp"],
    windows=[7, 14, 30],
    stats=["mean", "std", "min", "max"],   # default: ["mean", "std"]
    group_by="city",
    min_periods=1,
)
# Adds: temp_roll7_mean, temp_roll7_std, temp_roll14_mean, ...
```

With `group_by`, each row's window is computed over the rows of its own group, in the frame's row order, and the value lands on that same row — the frame does **not** need to be sorted by the group key (before 0.9.3 it did, silently). Sort by time within each group first; rolling windows follow row order.

---

## encode_cyclical

Encodes a periodic feature as sin/cos pair. Both columns should be used as features together — a single sin or cos is ambiguous.

```python
# Hour of day (period=24)
df["hour_sin"], df["hour_cos"] = encode_cyclical(df["hour"], period=24)

# Day of year (period=365)
df["doy_sin"], df["doy_cos"] = encode_cyclical(df["day_of_year"], period=365)

# Month (period=12)
df["month_sin"], df["month_cos"] = encode_cyclical(df["month"], period=12)
```

Output series names: `{original_name}_sin`, `{original_name}_cos`.

**Why not one-hot encode time?** Cyclical encoding preserves the circular topology (hour 23 is close to hour 0). One-hot encoding loses this.

---

## encode_onehot

One-hot encodes categorical columns. Original columns are dropped; dummy columns are appended.

```python
df = encode_onehot(
    df,
    cols=["city", "season"],
    drop_first=True,   # drop first dummy to avoid multicollinearity (default)
    dtype=float,       # LightGBM expects floats (default)
)
```

**`drop_first=True`** is the default and is appropriate for tree models and linear models. Use `drop_first=False` if you need all dummies (e.g. for neural networks without a bias term).

---

## clip_outliers

Clips extreme values. Returns a copy with clipped values — no new columns added.

```python
# IQR method: clip at [Q1 - k*IQR, Q3 + k*IQR]
df = clip_outliers(df, cols=["temp"], method="iqr", iqr_factor=3.0)

# Z-score method: clip beyond N standard deviations
df = clip_outliers(df, cols=["temp"], method="zscore", zscore_threshold=4.0)
```

| Method | When to use |
|---|---|
| `iqr` | Robust to heavy tails; works without assuming normality |
| `zscore` | Assumes roughly normal distribution; faster |

**Recommended:** apply `clip_outliers` before `add_rolling_stats` to prevent outliers from contaminating rolling windows.

---

## add_missing_indicators

Adds one `int8` 0/1 column per input column: 1 where the value is missing (NaN/None/NaT).

```python
df = add_missing_indicators(df, cols=["wind_speed", "pressure"])           # suffix="_isna"
# Adds: wind_speed_isna, pressure_isna
```

Call it **before** imputing — once the NaNs are filled, the missingness is gone. Row-local, so it cannot leak.

---

## add_interactions

Adds one pairwise column per `(a, b)`. `op` is `"mul"` (default), `"div"`, `"add"` or `"sub"`.

```python
df = add_interactions(df, pairs=[("temp", "humidity"), ("wind", "temp")])              # temp_x_humidity, wind_x_temp
df = add_interactions(df, pairs=[("price", "liquidity")], op="div")                    # price_div_liquidity
# names: a_x_b (mul), a_div_b (div), a_plus_b (add), a_minus_b (sub)
```

Names follow the order of the pair, so `(a, b)` and `(b, a)` are different columns. **Division by zero gives NaN, never `inf`.** Row-local, so it cannot leak.

---

## Target encoding — `TargetEncoder` / `FittedTargetEncoder`

Replaces a categorical value by a smoothed mean of the target for that value: `(sum + m * prior) / (n + m)` (m-estimate; `m = smoothing`). A category unseen while fitting gets the global `prior`; a missing (NaN/None) category is its own category (its own smoothed estimate if missing values were seen, otherwise `prior`); rows with a NaN target contribute to no statistic. Deterministic given `seed`.

### The leakage contract

Computed naively on the training frame, a row's own target sits inside its own encoding: the model learns to read it back and the backtest overstates live performance. So training and inference are **two different computations**, on purpose (this mirrors `sklearn.preprocessing.TargetEncoder`):

| Use | Call | What a row's encoding is built from |
|---|---|---|
| **Training rows** | `enc.fit_transform(train)` | Targets of **other folds only** (out-of-fold). The prior is out-of-fold too. A row never sees its own target. |
| **New / live rows** | `enc.fit(train).transform(live)` | Full-train smoothed statistics. No target column needed on `live`. |

Never call `fit(train).transform(train)` to build training features: that is the leaky in-sample path.

`cv` selects the folds — there is no silent fallback to shuffled K-fold:

| `cv` | Folds | Use when |
|---|---|---|
| `"time"` (default, needs `time_col`) | Expanding window: distinct timestamps are cut into `n_folds` contiguous chunks; chunk `k` is encoded from chunks `< k` only, i.e. **strictly earlier timestamps — no lookahead**. Rows sharing a timestamp share a fold. The earliest chunk has no history and is NaN (trees handle it natively). | Anything with a time order — i.e. almost every market panel. |
| `"group"` (needs `group_col`) | Seeded group K-fold: all rows of a group (e.g. the bins of one market) stay in one fold and never encode each other. | Rows exchangeable across groups, but not within a group. |
| `"kfold"` | Seeded shuffled K-fold. Must be asked for explicitly. | Rows fully exchangeable (rare). |

`n_folds` is capped by the number of distinct timestamps / groups / rows. `time_col` should order rows by *when the target became known* (e.g. market resolution time) if that differs from the feature timestamp; no extra purge/embargo is applied.

### Correct usage

```python
from pmlab import TargetEncoder

enc = TargetEncoder(col="city", target="outcome", time_col="date", n_folds=5, smoothing=10.0)

# training: out-of-fold -> adds city_te (no row sees its own or any later target)
X_train = enc.fit_transform(train)

# inference: full-train statistics, applied to rows whose target is unknown
fitted = enc.fit(train)              # fit on the *same* rows the model trained on
X_live = fitted.transform(live)      # adds city_te; unseen city -> prior

# the fitted encoder is a frozen dataclass of plain floats + a dict: pickle it with the model
```

To inspect or align with your own CV, `enc.assign_folds(df)` returns each row's fold id. Wrong usage to avoid: encoding the whole panel with `fit(df).transform(df)` and *then* splitting train/test, or fitting on a window that includes the test rows.

The train/serve distributions differ slightly (OOF statistics see less history than full-train ones, most visibly in the first folds); this is the standard cost of leakage-free target encoding, and the reason to keep `n_folds` moderate.

---

## Full Pipeline Example

```python
import pandas as pd
from pmlab import add_lags, add_rolling_stats, encode_cyclical, encode_onehot, clip_outliers

# Raw panel: city, date, temp, humidity, hour
df = pd.read_parquet("data/historical_panel.parquet")

# 1. Clip outliers first
df = clip_outliers(df, cols=["temp", "humidity"], method="iqr", iqr_factor=3.0)

# 2. Lag features
df = add_lags(df, cols=["temp"], lags=[1, 3, 7], group_by="city")

# 3. Rolling stats
df = add_rolling_stats(df, cols=["temp"], windows=[7, 14], stats=["mean", "std"], group_by="city")

# 4. Cyclical time encoding
df["doy_sin"], df["doy_cos"] = encode_cyclical(df["day_of_year"], period=365)

# 5. One-hot encode categoricals
df = encode_onehot(df, cols=["city"], drop_first=True)
```

Missing-value indicators and target encoding belong *before* step 5 (they read the raw categorical / NaNs), with the target-encoded column built out-of-fold as above:

```python
df = add_missing_indicators(df, cols=["humidity"])       # before any imputation
df = add_interactions(df, pairs=[("temp", "humidity")])
df = TargetEncoder("city", "outcome", time_col="date").fit_transform(df)
```
