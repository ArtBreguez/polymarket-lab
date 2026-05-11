# Feature Engineering — pmlab.features

`pmlab.features` provides common data transforms so plugins don't reimplement them. All functions return a **copy** — originals are never mutated.

---

## Installation

`pmlab.features` is included in `pmlab`. No extra dependencies.

```python
from pmlab import add_lags, add_rolling_stats, encode_cyclical, encode_onehot, clip_outliers
# or
from pmlab.features.transforms import add_lags, ...
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
