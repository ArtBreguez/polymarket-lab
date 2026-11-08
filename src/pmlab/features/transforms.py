"""Feature engineering transforms."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

__all__ = [
    "add_interactions",
    "add_lags",
    "add_missing_indicators",
    "add_rolling_stats",
    "clip_outliers",
    "encode_cyclical",
    "encode_onehot",
]

InteractionOp = Literal["mul", "div", "add", "sub"]

# Infix used in the generated column name, per op: ``{a}_x_{b}``, ``{a}_div_{b}``, ...
_INTERACTION_INFIX: dict[str, str] = {"mul": "x", "div": "div", "add": "plus", "sub": "minus"}


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
                    # ``transform`` returns the result aligned to the input rows. The previous
                    # ``groupby().rolling().<stat>().values`` is ordered by group, so it landed on
                    # the wrong rows whenever the frame was not already sorted by the group key.
                    out[new_col] = out.groupby(group_by)[col].transform(
                        lambda s, w=window, st=stat: getattr(
                            s.rolling(w, min_periods=min_periods), st
                        )()
                    )
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


def add_missing_indicators(
    df: pd.DataFrame,
    cols: list[str],
    suffix: str = "_isna",
) -> pd.DataFrame:
    """Add one ``int8`` 0/1 column per entry of ``cols`` flagging missing values.

    ``{col}{suffix}`` is 1 where ``col`` is NaN/None/NaT and 0 elsewhere. Row-local (it looks
    at no other row), so it cannot leak. Apply it *before* imputing the column, otherwise the
    missingness is already gone. The input frame is never mutated.
    """
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame")
        out[f"{col}{suffix}"] = out[col].isna().astype("int8")
    return out


def add_interactions(
    df: pd.DataFrame,
    pairs: list[tuple[str, str]],
    op: InteractionOp = "mul",
) -> pd.DataFrame:
    """Add one pairwise interaction column per ``(a, b)`` in ``pairs``.

    ``op`` is one of ``"mul"`` (``a * b``, column ``a_x_b``), ``"div"`` (``a / b``, ``a_div_b``),
    ``"add"`` (``a + b``, ``a_plus_b``) or ``"sub"`` (``a - b``, ``a_minus_b``). Names follow the
    order of the pair, so ``(a, b)`` and ``(b, a)`` are different columns. Division by zero gives
    NaN (never ``inf``), so downstream models and scalers see a missing value rather than an
    infinity. Row-local, so it cannot leak. The input frame is never mutated.
    """
    if op not in _INTERACTION_INFIX:
        raise ValueError(f"Unknown op '{op}'; expected one of {sorted(_INTERACTION_INFIX)}")
    out = df.copy()
    for a, b in pairs:
        for col in (a, b):
            if col not in out.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame")
        left = out[a].astype(float)
        right = out[b].astype(float)
        if op == "mul":
            result = left * right
        elif op == "add":
            result = left + right
        elif op == "sub":
            result = left - right
        else:
            result = left / right.where(right != 0)
        out[f"{a}_{_INTERACTION_INFIX[op]}_{b}"] = result
    return out
