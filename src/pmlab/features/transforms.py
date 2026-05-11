"""Feature engineering transforms."""
from __future__ import annotations
from typing import Literal
import numpy as np
import pandas as pd

__all__ = ["add_lags", "add_rolling_stats", "encode_cyclical", "encode_onehot", "clip_outliers"]

def add_lags(
    df: pd.DataFrame,
    cols: list[str],
    lags: list[int],
    group_by: str | None = None,
    fill_value: float = float("nan"),
) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame")
        for lag in lags:
            new_col = f"{col}_lag{lag}"
            if group_by is not None:
                out[new_col] = out.groupby(group_by)[col].shift(lag).fillna(fill_value)
            else:
                out[new_col] = out[col].shift(lag).fillna(fill_value)
    return out

def add_rolling_stats(
    df: pd.DataFrame,
    cols: list[str],
    windows: list[int],
    stats: list[Literal["mean", "std", "min", "max", "median"]] | None = None,
    group_by: str | None = None,
    min_periods: int = 1,
) -> pd.DataFrame:
    if stats is None:
        stats = ["mean", "std"]
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame")
        for window in windows:
            for stat in stats:
                new_col = f"{col}_roll{window}_{stat}"
                if group_by is not None:
                    rolled_group = out.groupby(group_by)[col].rolling(window, min_periods=min_periods)
                    out[new_col] = getattr(rolled_group, stat)().values
                else:
                    rolled_plain = out[col].rolling(window, min_periods=min_periods)
                    out[new_col] = getattr(rolled_plain, stat)().values
    return out

def encode_cyclical(series: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    angle = 2.0 * np.pi * series / period
    name = str(series.name) if series.name is not None else "x"
    sin_s = pd.Series(np.sin(angle).values, index=series.index, name=f"{name}_sin")
    cos_s = pd.Series(np.cos(angle).values, index=series.index, name=f"{name}_cos")
    return sin_s, cos_s

def encode_onehot(
    df: pd.DataFrame,
    cols: list[str],
    drop_first: bool = True,
    dtype: type = float,
) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame")
    dummies = pd.get_dummies(out[cols], drop_first=drop_first, dtype=dtype)
    out = out.drop(columns=cols)
    out = pd.concat([out, dummies], axis=1)
    return out

def clip_outliers(
    df: pd.DataFrame,
    cols: list[str],
    method: Literal["iqr", "zscore"] = "iqr",
    iqr_factor: float = 3.0,
    zscore_threshold: float = 4.0,
) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame")
        s = out[col]
        if method == "iqr":
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - iqr_factor * iqr, q3 + iqr_factor * iqr
        else:
            mean, std = s.mean(), s.std()
            lo = mean - zscore_threshold * std
            hi = mean + zscore_threshold * std
        out[col] = s.clip(lower=lo, upper=hi)
    return out
