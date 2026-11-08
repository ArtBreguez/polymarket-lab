"""Leakage-safe target encoding: out-of-fold for training rows, fitted mapping for new rows.

Target encoding replaces a categorical value by a smoothed mean of the target for that value.
Computed naively on the training frame, every row's own target is inside its own encoding, the
model learns to read it back, and the backtest looks better than live trading will ever be.

API choice. This mirrors ``sklearn.preprocessing.TargetEncoder``: ``fit_transform`` and
``fit(...).transform(...)`` are *deliberately different computations*.

* :meth:`TargetEncoder.fit_transform` encodes the **training** rows out-of-fold. Rows are split
  into K folds and the rows of fold ``k`` are encoded from the targets of *other* folds only,
  so no row's encoding contains its own target. The prior is also computed out-of-fold.
* :meth:`TargetEncoder.fit` computes the full-train statistics once and returns a
  :class:`FittedTargetEncoder`, whose :meth:`~FittedTargetEncoder.transform` encodes **new/live**
  rows (inference never takes the OOF path: a live row has no fold, and its target is unknown).

A frozen config object with ``fit_transform``/``fit`` keeps the two paths on one set of
parameters (same smoothing, same NaN handling), so the feature the model trains on and the
feature it is served have the same definition. The fitted state is a separate frozen object
holding only plain floats and a dict, so it pickles next to the model.

Fold schemes (``cv``). There is no silent default to shuffled K-fold:

* ``"time"`` (default, needs ``time_col``): expanding window. Distinct timestamps are cut into
  ``n_folds`` contiguous chunks and the rows of chunk ``k`` are encoded from chunks ``< k``
  only, i.e. from strictly earlier timestamps, so there is no lookahead. Rows sharing a
  timestamp share a fold. The earliest fold has no history at all and is encoded as NaN.
  Pass the column that says *when the target became known* if that differs from the row time
  (e.g. market resolution time); no purge/embargo is applied beyond that.
* ``"kfold"``: seeded shuffled K-fold. Use only when rows are exchangeable (no time order and
  no rows that share an outcome); the caller must ask for it explicitly.
* ``"group"`` (needs ``group_col``): seeded group K-fold; all rows of a group (e.g. the bins of
  one market) land in the same fold, so they never encode each other.

Smoothing is the m-estimate ``(sum + m * prior) / (n + m)``. A category unseen in the fitting
rows gets ``prior``. A missing (NaN/None) category is its own category: it gets its own
smoothed estimate when missing values were seen while fitting, otherwise ``prior``. Rows whose
target is NaN contribute to no statistic. Everything is deterministic given ``seed``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

__all__ = ["FittedTargetEncoder", "TargetEncoder"]

CVScheme = Literal["time", "kfold", "group"]
_CV_SCHEMES = ("time", "kfold", "group")

_Stats = tuple[float, dict[Any, float], float]


def _smoothed_stats(
    cat: npt.NDArray[np.object_], y: npt.NDArray[np.float64], smoothing: float
) -> _Stats:
    """Return ``(prior, {category: encoding}, nan_category_encoding)`` from the given rows."""
    valid = ~np.isnan(y)
    if not valid.any():
        return float("nan"), {}, float("nan")
    cats = pd.Series(cat[valid], dtype=object)
    target = pd.Series(y[valid])
    prior = float(target.mean())
    agg = target.groupby(cats, dropna=False).agg(["sum", "count"])
    smoothed = (agg["sum"] + smoothing * prior) / (agg["count"] + smoothing)
    is_nan_key = np.asarray(smoothed.index.isna())
    nan_value = float(smoothed[is_nan_key].iloc[0]) if is_nan_key.any() else prior
    mapping = {key: float(val) for key, val in smoothed[~is_nan_key].items()}
    return prior, mapping, nan_value


def _encode(
    cat: npt.NDArray[np.object_], prior: float, mapping: dict[Any, float], nan_value: float
) -> npt.NDArray[np.float64]:
    """Look categories up in ``mapping``: unseen -> ``prior``, missing -> ``nan_value``."""
    series = pd.Series(cat, dtype=object)
    looked_up = series.map(mapping).astype(float).to_numpy()
    encoded = np.where(np.isnan(looked_up), prior, looked_up)
    return np.where(series.isna().to_numpy(), nan_value, encoded).astype(np.float64)


def _require(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame")


def _category_array(df: pd.DataFrame, col: str) -> npt.NDArray[np.object_]:
    return np.asarray(df[col].astype(object).to_numpy(), dtype=object)


@dataclass(frozen=True)
class FittedTargetEncoder:
    """Full-train smoothed statistics, applied to new rows. Build it with ``TargetEncoder.fit``.

    ``mapping`` is read-only by convention (it is a plain dict so the object pickles).
    """

    col: str
    output_col: str
    smoothing: float
    prior: float
    nan_value: float
    mapping: dict[Any, float]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``df`` with ``output_col`` added; ``df`` is not mutated.

        Unseen categories get ``prior``; missing categories get ``nan_value``. The target
        column is not needed (nor read) here, so this works on live rows.
        """
        _require(df, [self.col])
        out = df.copy()
        out[self.output_col] = _encode(
            _category_array(df, self.col), self.prior, self.mapping, self.nan_value
        )
        return out


@dataclass(frozen=True)
class TargetEncoder:
    """Configuration of a leakage-safe target encoding of one categorical column.

    See the module docstring for the leakage contract. ``col`` is encoded into
    ``f"{col}{suffix}"`` using ``target``. ``n_folds`` is capped by the number of distinct
    timestamps / groups / rows, so a tiny frame degrades to fewer folds rather than failing.
    """

    col: str
    target: str
    smoothing: float = 10.0
    cv: CVScheme = "time"
    n_folds: int = 5
    time_col: str | None = None
    group_col: str | None = None
    seed: int = 0
    suffix: str = "_te"

    def __post_init__(self) -> None:
        if self.cv not in _CV_SCHEMES:
            raise ValueError(f"Unknown cv '{self.cv}'; expected one of {list(_CV_SCHEMES)}")
        if self.n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {self.n_folds}")
        if not np.isfinite(self.smoothing) or self.smoothing < 0:
            raise ValueError(f"smoothing must be a finite number >= 0, got {self.smoothing}")
        if self.cv == "time" and self.time_col is None:
            raise ValueError(
                'cv="time" needs time_col; pass cv="kfold" or cv="group" explicitly if the '
                "rows are exchangeable"
            )
        if self.cv == "group" and self.group_col is None:
            raise ValueError('cv="group" needs group_col')

    @property
    def output_col(self) -> str:
        return f"{self.col}{self.suffix}"

    def assign_folds(self, df: pd.DataFrame) -> npt.NDArray[np.int64]:
        """Fold id (0..K-1) of every row of ``df``, positionally. Deterministic given ``seed``."""
        _require(df, [self.col, self.target])
        n = len(df)
        if self.cv == "time":
            assert self.time_col is not None
            _require(df, [self.time_col])
            if n == 0:
                return np.empty(0, dtype=np.int64)
            times = df[self.time_col]
            if times.isna().any():
                raise ValueError(f"time column '{self.time_col}' contains missing values")
            codes, uniques = pd.factorize(times, sort=True)
            return self._chunk_ids(len(uniques), shuffled=False)[codes]
        if self.cv == "group":
            assert self.group_col is not None
            _require(df, [self.group_col])
            if n == 0:
                return np.empty(0, dtype=np.int64)
            codes, uniques = pd.factorize(df[self.group_col], use_na_sentinel=False)
            return self._chunk_ids(len(uniques), shuffled=True)[codes]
        if n == 0:
            return np.empty(0, dtype=np.int64)
        return self._chunk_ids(n, shuffled=True)

    def _chunk_ids(self, n_units: int, *, shuffled: bool) -> npt.NDArray[np.int64]:
        """Fold id for each of ``n_units`` units: contiguous chunks, or seeded round-robin."""
        k = min(self.n_folds, n_units)
        if not shuffled:
            ids = np.empty(n_units, dtype=np.int64)
            for fold, chunk in enumerate(np.array_split(np.arange(n_units), k)):
                ids[chunk] = fold
            return ids
        perm = np.random.default_rng(self.seed).permutation(n_units)
        ids = np.empty(n_units, dtype=np.int64)
        ids[perm] = np.arange(n_units, dtype=np.int64) % k
        return ids

    def _target_array(self, df: pd.DataFrame) -> npt.NDArray[np.float64]:
        return np.asarray(df[self.target].to_numpy(dtype=float, na_value=np.nan), dtype=np.float64)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode the **training** rows out-of-fold; returns a copy with ``output_col`` added.

        The row's own target (and every target of its own fold) is never used, and with
        ``cv="time"`` neither is any target at the same or a later timestamp. ``df`` is not
        mutated. Do not use this on live rows: use ``fit(train).transform(live)``.
        """
        folds = self.assign_folds(df)
        cat = _category_array(df, self.col)
        y = self._target_array(df)
        encoded = np.full(len(df), np.nan, dtype=np.float64)
        for fold in np.unique(folds):
            in_fold = folds == fold
            fit_rows = folds < fold if self.cv == "time" else folds != fold
            prior, mapping, nan_value = _smoothed_stats(cat[fit_rows], y[fit_rows], self.smoothing)
            encoded[in_fold] = _encode(cat[in_fold], prior, mapping, nan_value)
        out = df.copy()
        out[self.output_col] = encoded
        return out

    def fit(self, df: pd.DataFrame) -> FittedTargetEncoder:
        """Compute the full-train statistics, for encoding new/live rows at inference time."""
        _require(df, [self.col, self.target])
        prior, mapping, nan_value = _smoothed_stats(
            _category_array(df, self.col), self._target_array(df), self.smoothing
        )
        if np.isnan(prior):
            raise ValueError(f"no rows with a non-missing target '{self.target}' to fit on")
        return FittedTargetEncoder(
            col=self.col,
            output_col=self.output_col,
            smoothing=self.smoothing,
            prior=prior,
            nan_value=nan_value,
            mapping=mapping,
        )
