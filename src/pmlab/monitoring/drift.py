"""Drift monitoring: has the world moved under the champion?

The holdout gate says a model was good on a historical panel. This compares what
the model is being fed *now* — and what it is predicting now — against that
panel, one feature at a time, with PSI and KL divergence.

`drift_report` is a pure function over ``(reference, current)``: no I/O, no
randomness, same inputs give an identical report. `drift_report_from_store` is a
thin wrapper that reads the current window from a `FeatureSnapshotStore` through
its point-in-time read.

Conventions worth knowing before reading the code:

**Bins come from the reference only.** Numeric features are cut at the reference
sample's quantiles and those same edges are reused for the current sample. Two
ways to get this wrong, both avoided: binning each sample on its *own* quantiles
makes both come out ~uniform, so a distribution shifted wholesale scores ~0 —
drift vanishes exactly when it is largest; and taking the shared edges from the
*current* sample makes the statistic depend on the very sample being judged.
Everything strictly below the reference minimum and everything strictly above its
maximum gets a bin of its own, whatever the reference looks like. A constant or
heavily tied reference collapses its quantile edges, and with open-ended end bins
a value below the reference would then share a bin with the reference's own mass
and score 0. Those outer bins are empty in the reference, so out-of-range mass is
floored (see below) and moves PSI by itself; `unseen_rate` reports its share.

**Empty bins are floored, not dropped.** ``p * ln(p / 0)`` is infinite, and a
value range the reference never saw is the drift that matters most. Each bin's
proportion is floored at `epsilon` and the vector is renormalised, so such a
range scores large but finite. Bins empty on *both* sides are omitted.

**KL is directional.** ``kl`` is KL(current || reference) =
``sum(c * ln(c / r))`` in nats: how surprised a model trained on the reference
would be by live data. It punishes current mass where the reference had little,
and barely notices reference mass that has vanished. PSI is the symmetric sum of
both directions: ``psi = KL(cur || ref) + KL(ref || cur)``.

**Missing is drift.** NaN (and +/-inf for numeric features) are excluded from the
distribution but counted: a feature going from 0% to 40% missing changes the
world even though every remaining value looks fine.

**Too little data is "insufficient", not a number.** Below `min_samples` the
statistics are `None` and the status says so. "insufficient" never reads as "ok".

**This computes and classifies; it does not act.** Thresholds are explicit
configuration (`DriftThresholds`), not buried constants. What to do about an
alert is the watchdog's job.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from pmlab.data.snapshot_store import FeatureSnapshotStore

__all__ = [
    "DriftConfig",
    "DriftReport",
    "DriftThresholds",
    "FeatureDrift",
    "drift_report",
    "drift_report_from_store",
]

DriftStatus = Literal["ok", "warn", "alert", "insufficient"]
FeatureKind = Literal["numeric", "categorical"]

# alert > warn > insufficient > ok: "we could not check" must outrank "we checked
# and it is fine", but never mask a measured problem.
_SEVERITY: dict[str, int] = {"ok": 0, "insufficient": 1, "warn": 2, "alert": 3}
_SIGNALS = ("psi", "kl", "missing_rate")

_NUMERIC_KINDS = frozenset({"integer", "floating", "mixed-integer-float", "decimal"})
# Display cap only; `n_unseen_categories` always counts every one.
_MAX_LISTED_UNSEEN = 20
_FEATURE_PREFIX = "feature_"
_PREDICTION = "prediction"


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DriftThresholds:
    """Where each signal turns from ``ok`` to ``warn`` to ``alert``.

    A value at or above a threshold is in that band, with 1e-9 of slack for float
    round-off (a missing rate going 10% -> 30% changes by 0.19999999999999998, and
    is still an alert at 0.20). The defaults are conventions, not laws — tune them
    on your own data:

    - **PSI 0.10 / 0.25** is the widely used credit-scoring convention: below 0.10
      no meaningful shift, 0.10-0.25 moderate, above 0.25 significant.
    - **KL 0.05 / 0.125** has no industry convention. These are half the PSI bands:
      for small shifts both KL directions are about equal, so PSI ~ 2 * KL.
    - **Missing-rate 0.05 / 0.20** is the *absolute* change in the share of missing
      values (either direction). No convention exists; a 0% -> 40% jump is an alert.
    """

    psi_warn: float = 0.10
    psi_alert: float = 0.25
    kl_warn: float = 0.05
    kl_alert: float = 0.125
    missing_warn: float = 0.05
    missing_alert: float = 0.20

    def __post_init__(self) -> None:
        for name in (f.name for f in fields(self)):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite number >= 0, got {value}")
            object.__setattr__(self, name, value)
        for signal in ("psi", "kl", "missing"):
            warn, alert = getattr(self, f"{signal}_warn"), getattr(self, f"{signal}_alert")
            if warn > alert:
                raise ValueError(f"{signal}_warn ({warn}) must not exceed {signal}_alert ({alert})")

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DriftThresholds:
        return cls(**{f.name: float(data[f.name]) for f in fields(cls)})


@dataclass(frozen=True)
class DriftConfig:
    """Everything that is a policy choice rather than arithmetic.

    `n_bins` is the number of reference-quantile bins for numeric features (10 is
    the PSI convention). `min_samples` is the least number of *usable* values each
    side needs before a distribution statistic is reported, and the least number of
    rows each side needs before a missing-rate change is classified. PSI of an
    *unchanged* feature is not zero: its noise floor is roughly ``(n_bins - 1) / n``
    for ``n`` current values. The default of 300 keeps that near 0.03 at ten bins —
    in 500 simulated windows of an unchanged feature, 3 crossed the 0.10 warn line
    at 300 values and 204 did at 100. Raise it if you raise `n_bins`.
    `epsilon` is the floor on a bin's proportion — see the module docstring.

    Numeric vs categorical is inferred from the *reference* column's contents
    (integers and floats are numeric; strings, booleans and everything else are
    categorical). Name a feature in `numeric` or `categorical` to override — for
    example integer-coded categories.
    """

    n_bins: int = 10
    min_samples: int = 300
    epsilon: float = 1e-4
    thresholds: DriftThresholds = field(default_factory=DriftThresholds)
    numeric: tuple[str, ...] = ()
    categorical: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_bins", int(self.n_bins))
        object.__setattr__(self, "min_samples", int(self.min_samples))
        object.__setattr__(self, "epsilon", float(self.epsilon))
        if self.n_bins < 2:
            raise ValueError(f"n_bins must be at least 2, got {self.n_bins}")
        if self.min_samples < 1:
            raise ValueError(f"min_samples must be at least 1, got {self.min_samples}")
        if not 0.0 < self.epsilon < 1.0:
            raise ValueError(f"epsilon must be in (0, 1), got {self.epsilon}")
        object.__setattr__(self, "numeric", tuple(sorted({str(n) for n in self.numeric})))
        object.__setattr__(self, "categorical", tuple(sorted({str(c) for c in self.categorical})))
        both = sorted(set(self.numeric) & set(self.categorical))
        if both:
            raise ValueError(f"features declared both numeric and categorical: {both}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_bins": self.n_bins,
            "min_samples": self.min_samples,
            "epsilon": self.epsilon,
            "thresholds": self.thresholds.to_dict(),
            "numeric": list(self.numeric),
            "categorical": list(self.categorical),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DriftConfig:
        return cls(
            n_bins=int(data["n_bins"]),
            min_samples=int(data["min_samples"]),
            epsilon=float(data["epsilon"]),
            thresholds=DriftThresholds.from_dict(data["thresholds"]),
            numeric=tuple(data["numeric"]),
            categorical=tuple(data["categorical"]),
        )


# --------------------------------------------------------------------------- #
# Report types
# --------------------------------------------------------------------------- #


def _opt_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _worst(statuses: Iterable[str]) -> DriftStatus:
    """Worst status by severity; nothing assessed is "insufficient", never "ok"."""
    worst: DriftStatus | None = None
    for status in statuses:
        if status not in _SEVERITY:
            raise ValueError(f"unknown drift status: {status!r}")
        if worst is None or _SEVERITY[status] > _SEVERITY[worst]:
            worst = status  # type: ignore[assignment]
    return worst if worst is not None else "insufficient"


@dataclass(frozen=True)
class FeatureDrift:
    """Drift of one feature (or of the model's predictions) against the reference.

    `psi` and `kl` are `None` — not zero — when either side has fewer than
    `min_samples` usable values. `kl` is KL(current || reference), in nats.
    `missing_rate_change` is ``current - reference`` (signed); it is classified by
    its absolute value. `unseen_rate` is the share of usable current values the
    reference never contained: outside the reference's min/max for a numeric
    feature, in a category absent from the reference for a categorical one. It is
    already inside `psi` / `kl` in both cases (out-of-range mass has its own bins),
    so it has no thresholds; it stays on the report because a coarse `epsilon`
    makes that mass cheap and can hide it from the status.

    `drivers` names the signals (``"psi"``, ``"kl"``, ``"missing_rate"``) that set
    `status`; empty when ``ok``.
    """

    name: str
    kind: FeatureKind
    status: DriftStatus
    drivers: tuple[str, ...]
    n_reference: int
    n_current: int
    n_reference_valid: int
    n_current_valid: int
    reference_missing_rate: float | None
    current_missing_rate: float | None
    missing_rate_change: float | None
    psi: float | None
    kl: float | None
    unseen_rate: float | None
    n_unseen_categories: int
    unseen_categories: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "status": self.status,
            "drivers": list(self.drivers),
            "n_reference": self.n_reference,
            "n_current": self.n_current,
            "n_reference_valid": self.n_reference_valid,
            "n_current_valid": self.n_current_valid,
            "reference_missing_rate": self.reference_missing_rate,
            "current_missing_rate": self.current_missing_rate,
            "missing_rate_change": self.missing_rate_change,
            "psi": self.psi,
            "kl": self.kl,
            "unseen_rate": self.unseen_rate,
            "n_unseen_categories": self.n_unseen_categories,
            "unseen_categories": list(self.unseen_categories),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureDrift:
        return cls(
            name=str(data["name"]),
            kind=data["kind"],
            status=_worst([data["status"]]),
            drivers=tuple(str(d) for d in data["drivers"]),
            n_reference=int(data["n_reference"]),
            n_current=int(data["n_current"]),
            n_reference_valid=int(data["n_reference_valid"]),
            n_current_valid=int(data["n_current_valid"]),
            reference_missing_rate=_opt_float(data["reference_missing_rate"]),
            current_missing_rate=_opt_float(data["current_missing_rate"]),
            missing_rate_change=_opt_float(data["missing_rate_change"]),
            psi=_opt_float(data["psi"]),
            kl=_opt_float(data["kl"]),
            unseen_rate=_opt_float(data["unseen_rate"]),
            n_unseen_categories=int(data["n_unseen_categories"]),
            unseen_categories=tuple(str(c) for c in data["unseen_categories"]),
        )


@dataclass(frozen=True)
class DriftReport:
    """Drift of every assessed feature, plus the model's predictions if supplied.

    `features` is ordered by name, whatever order the columns arrived in.
    `prediction` is reported separately so a model whose *inputs* look unchanged
    but whose *output* moved is visible. `status` is the worst of all of them.

    `n_reference` / `n_current` are the row counts of the feature tables.
    `absent_from_current` lists assessed features the current table does not have
    at all (they are scored as 100% missing); `new_in_current` lists columns the
    reference does not have (informational — they are not assessed).

    The flat counts and `max_psi` / `max_kl` exist for trackers that only log
    top-level numbers; they are `None`, not zero, when nothing was computed.
    """

    config: DriftConfig
    features: tuple[FeatureDrift, ...]
    prediction: FeatureDrift | None
    n_reference: int
    n_current: int
    absent_from_current: tuple[str, ...]
    new_in_current: tuple[str, ...]

    @property
    def _assessed(self) -> tuple[FeatureDrift, ...]:
        extra = (self.prediction,) if self.prediction is not None else ()
        return self.features + extra

    @property
    def status(self) -> DriftStatus:
        return _worst(f.status for f in self._assessed)

    def _count(self, status: str) -> int:
        return sum(1 for f in self._assessed if f.status == status)

    @property
    def n_ok(self) -> int:
        return self._count("ok")

    @property
    def n_warn(self) -> int:
        return self._count("warn")

    @property
    def n_alert(self) -> int:
        return self._count("alert")

    @property
    def n_insufficient(self) -> int:
        return self._count("insufficient")

    @property
    def max_psi(self) -> float | None:
        """Largest PSI over the features (not the prediction); None if none computed."""
        return max((f.psi for f in self.features if f.psi is not None), default=None)

    @property
    def max_kl(self) -> float | None:
        """Largest KL(current || reference) over the features; None if none computed."""
        return max((f.kl for f in self.features if f.kl is not None), default=None)

    @property
    def prediction_psi(self) -> float | None:
        return None if self.prediction is None else self.prediction.psi

    @property
    def prediction_kl(self) -> float | None:
        return None if self.prediction is None else self.prediction.kl

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "n_reference": self.n_reference,
            "n_current": self.n_current,
            "n_ok": self.n_ok,
            "n_warn": self.n_warn,
            "n_alert": self.n_alert,
            "n_insufficient": self.n_insufficient,
            "max_psi": self.max_psi,
            "max_kl": self.max_kl,
            "prediction_psi": self.prediction_psi,
            "prediction_kl": self.prediction_kl,
            "features": [f.to_dict() for f in self.features],
            "prediction": None if self.prediction is None else self.prediction.to_dict(),
            "absent_from_current": list(self.absent_from_current),
            "new_in_current": list(self.new_in_current),
            "config": self.config.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DriftReport:
        prediction = data["prediction"]
        return cls(
            config=DriftConfig.from_dict(data["config"]),
            features=tuple(FeatureDrift.from_dict(f) for f in data["features"]),
            prediction=None if prediction is None else FeatureDrift.from_dict(prediction),
            n_reference=int(data["n_reference"]),
            n_current=int(data["n_current"]),
            absent_from_current=tuple(str(n) for n in data["absent_from_current"]),
            new_in_current=tuple(str(n) for n in data["new_in_current"]),
        )


# --------------------------------------------------------------------------- #
# Arithmetic
# --------------------------------------------------------------------------- #


def _smooth(counts: npt.NDArray[np.int64], epsilon: float) -> npt.NDArray[np.float64]:
    """Counts -> proportions, floored at `epsilon`, renormalised to sum to 1."""
    p = counts / counts.sum()
    p = np.maximum(p, epsilon)
    return np.asarray(p / p.sum(), dtype=np.float64)


def _psi_kl(
    ref_counts: npt.NDArray[np.int64], cur_counts: npt.NDArray[np.int64], epsilon: float
) -> tuple[float, float]:
    """(PSI, KL(current || reference)) over bins, omitting bins empty on both sides."""
    keep = (ref_counts + cur_counts) > 0
    r = _smooth(ref_counts[keep], epsilon)
    c = _smooth(cur_counts[keep], epsilon)
    log_ratio = np.log(c / r)
    psi = float(np.sum((c - r) * log_ratio))
    # Gibbs' inequality makes KL >= 0; the clamp only absorbs float round-off.
    kl = max(float(np.sum(c * log_ratio)), 0.0)
    return psi, kl


def _numeric_counts(
    ref: npt.NDArray[np.float64], cur: npt.NDArray[np.float64], n_bins: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], float]:
    """Bin counts on reference-quantile edges, plus the out-of-reference-range rate.

    Three kinds of bin, in order: ``x < min(ref)``; the in-range bins; ``x > max(ref)``.
    The in-range bins are cut at the interior quantiles of `ref`, de-duplicated
    (ties collapse bins) and dropped where they sit at the maximum (nothing is left
    above them inside the range); a value falls in range-bin ``i`` when
    ``edges[i-1] < x <= edges[i]``.

    The two outer bins are what makes a constant or heavily tied reference safe.
    Were the first and last bins open-ended instead, a constant reference would
    leave only ``x <= c`` and ``x > c``, and everything below ``c`` would share a
    bin with the reference's own mass and score 0.
    """
    lo, hi = ref.min(), ref.max()
    edges = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, n_bins + 1)[1:-1]))
    edges = edges[edges < hi]
    size = len(edges) + 3

    def counts(x: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        index = np.searchsorted(edges, x, side="left") + 1
        index[x < lo] = 0
        index[x > hi] = size - 1
        return np.bincount(index, minlength=size).astype(np.int64)

    outside = (cur < lo) | (cur > hi)
    return counts(ref), counts(cur), float(outside.mean())


def _category_key(value: Any) -> tuple[str, str]:
    return (str(value), type(value).__name__)


def _categorical_counts(
    ref: list[Any], cur: list[Any]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], float, int, tuple[str, ...]]:
    """Bin counts per reference category, plus one pooled bin for unseen categories."""
    ref_counter, cur_counter = Counter(ref), Counter(cur)
    categories = sorted(ref_counter, key=_category_key)
    unseen = {c: n for c, n in cur_counter.items() if c not in ref_counter}
    ref_counts = np.array([ref_counter[c] for c in categories] + [0], dtype=np.int64)
    cur_counts = np.array(
        [cur_counter.get(c, 0) for c in categories] + [sum(unseen.values())], dtype=np.int64
    )
    ranked = sorted(unseen, key=lambda c: (-unseen[c], _category_key(c)))
    listed = tuple(str(c) for c in ranked[:_MAX_LISTED_UNSEEN])
    return ref_counts, cur_counts, sum(unseen.values()) / len(cur), len(unseen), listed


# PSI, KL and the missing-rate change are float arithmetic: 0.30 - 0.10 is
# 0.19999999999999998, which would read as just short of the documented
# ``>= 0.20`` alert line. A value within this of a threshold is at it.
_TOLERANCE = 1e-9


def _band(value: float, warn: float, alert: float) -> DriftStatus:
    if value >= alert - _TOLERANCE:
        return "alert"
    if value >= warn - _TOLERANCE:
        return "warn"
    return "ok"


def _classify(
    psi: float | None,
    kl: float | None,
    missing_change: float | None,
    *,
    distribution_ok: bool,
    rate_ok: bool,
    t: DriftThresholds,
) -> tuple[DriftStatus, tuple[str, ...]]:
    levels: dict[str, DriftStatus] = {}
    if distribution_ok and psi is not None and kl is not None:
        levels["psi"] = _band(psi, t.psi_warn, t.psi_alert)
        levels["kl"] = _band(kl, t.kl_warn, t.kl_alert)
    else:
        levels["psi"] = levels["kl"] = "insufficient"
    if rate_ok and missing_change is not None:
        levels["missing_rate"] = _band(abs(missing_change), t.missing_warn, t.missing_alert)
    else:
        levels["missing_rate"] = "insufficient"
    worst = _worst(levels.values())
    drivers = () if worst == "ok" else tuple(s for s in _SIGNALS if levels[s] == worst)
    return worst, drivers


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #


def _columns(
    data: Mapping[str, Any] | pd.DataFrame, label: str
) -> tuple[dict[str, pd.Series], int]:
    """Normalise a DataFrame or a name -> sequence mapping to ``{name: Series}``."""
    if isinstance(data, pd.DataFrame):
        if not data.columns.is_unique:
            raise ValueError(f"{label} has duplicate column names")
        cols = {str(c): data[c].reset_index(drop=True) for c in data.columns}
    else:
        cols = {str(k): pd.Series(v).reset_index(drop=True) for k, v in data.items()}
    lengths = {len(s) for s in cols.values()}
    if len(lengths) > 1:
        raise ValueError(f"{label} columns have unequal length: {sorted(lengths)}")
    return cols, (lengths.pop() if lengths else 0)


def _infer_kind(reference: pd.Series) -> FeatureKind:
    kind = pd.api.types.infer_dtype(reference, skipna=True)
    return "numeric" if kind in _NUMERIC_KINDS else "categorical"


def _valid_numeric(series: pd.Series, name: str, label: str) -> npt.NDArray[np.float64]:
    """Finite values only; NaN and +/-inf are missing."""
    try:
        values = pd.to_numeric(series, errors="raise").to_numpy(dtype=float, na_value=np.nan)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"feature {name!r} is numeric but its {label} values are not: {exc}"
        ) from exc
    return np.asarray(values[np.isfinite(values)], dtype=np.float64)


def _valid_categorical(series: pd.Series) -> list[Any]:
    return list(series[~series.isna()].tolist())


def _assess(
    name: str, kind: FeatureKind, ref: pd.Series, cur: pd.Series, cfg: DriftConfig
) -> FeatureDrift:
    n_ref, n_cur = len(ref), len(cur)
    ref_num = cur_num = ref_cat = cur_cat = None
    if kind == "numeric":
        ref_num = _valid_numeric(ref, name, "reference")
        cur_num = _valid_numeric(cur, name, "current")
        n_ref_valid, n_cur_valid = len(ref_num), len(cur_num)
    else:
        ref_cat, cur_cat = _valid_categorical(ref), _valid_categorical(cur)
        n_ref_valid, n_cur_valid = len(ref_cat), len(cur_cat)

    ref_rate = None if n_ref == 0 else (n_ref - n_ref_valid) / n_ref
    cur_rate = None if n_cur == 0 else (n_cur - n_cur_valid) / n_cur
    change = None if ref_rate is None or cur_rate is None else cur_rate - ref_rate

    distribution_ok = n_ref_valid >= cfg.min_samples and n_cur_valid >= cfg.min_samples
    rate_ok = n_ref >= cfg.min_samples and n_cur >= cfg.min_samples

    psi = kl = unseen_rate = None
    n_unseen = 0
    unseen_listed: tuple[str, ...] = ()
    if distribution_ok:
        if ref_num is not None and cur_num is not None:
            ref_counts, cur_counts, unseen_rate = _numeric_counts(ref_num, cur_num, cfg.n_bins)
        else:
            assert ref_cat is not None and cur_cat is not None
            ref_counts, cur_counts, unseen_rate, n_unseen, unseen_listed = _categorical_counts(
                ref_cat, cur_cat
            )
        psi, kl = _psi_kl(ref_counts, cur_counts, cfg.epsilon)

    status, drivers = _classify(
        psi,
        kl,
        change,
        distribution_ok=distribution_ok,
        rate_ok=rate_ok,
        t=cfg.thresholds,
    )
    return FeatureDrift(
        name=name,
        kind=kind,
        status=status,
        drivers=drivers,
        n_reference=n_ref,
        n_current=n_cur,
        n_reference_valid=n_ref_valid,
        n_current_valid=n_cur_valid,
        reference_missing_rate=ref_rate,
        current_missing_rate=cur_rate,
        missing_rate_change=change,
        psi=psi,
        kl=kl,
        unseen_rate=unseen_rate,
        n_unseen_categories=n_unseen,
        unseen_categories=unseen_listed,
    )


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #


def drift_report(
    reference: Mapping[str, Any] | pd.DataFrame,
    current: Mapping[str, Any] | pd.DataFrame,
    *,
    features: Sequence[str] | None = None,
    reference_predictions: Sequence[float] | npt.ArrayLike | None = None,
    current_predictions: Sequence[float] | npt.ArrayLike | None = None,
    config: DriftConfig | None = None,
) -> DriftReport:
    """Compare `current` against `reference`, feature by feature.

    Pure: no I/O, no randomness; the same inputs give an identical report.

    Args:
        reference: The training panel, as a DataFrame or ``{name: values}``.
            Bins, numeric/categorical kind and the category set all come from here.
        current: What the model is being fed now, same shape.
        features: Columns to assess; defaults to every reference column. Pass the
            model's feature list so identifiers and labels are not assessed.
        reference_predictions: The model's predicted probabilities on the
            reference panel. Supply together with `current_predictions`.
        current_predictions: The model's live predicted probabilities.
        config: Bins, minimum sample, smoothing and thresholds.

    Returns:
        A `DriftReport`. Features are ordered by name. A feature missing from
        `current` is scored as entirely missing and listed in
        ``absent_from_current``.

    Raises:
        ValueError: if `features` names a column the reference lacks, columns are
            of unequal length, only one of the prediction arrays is given, or a
            numeric feature holds values that cannot be read as numbers.
    """
    cfg = config if config is not None else DriftConfig()
    if (reference_predictions is None) != (current_predictions is None):
        raise ValueError(
            "reference_predictions and current_predictions must be given together (both or neither)"
        )

    ref_cols, n_ref = _columns(reference, "reference")
    cur_cols, n_cur = _columns(current, "current")

    names = sorted(set(features)) if features is not None else sorted(ref_cols)
    unknown = [n for n in names if n not in ref_cols]
    if unknown:
        raise ValueError(f"features not found in the reference: {unknown}")
    undeclared = sorted((set(cfg.numeric) | set(cfg.categorical)) - set(ref_cols))
    if undeclared:
        raise ValueError(f"config declares the kind of features not in the reference: {undeclared}")

    absent = tuple(n for n in names if n not in cur_cols)
    missing_column = pd.Series([np.nan] * n_cur, dtype=float)

    assessed: list[FeatureDrift] = []
    for name in names:
        if name in cfg.numeric:
            kind: FeatureKind = "numeric"
        elif name in cfg.categorical:
            kind = "categorical"
        else:
            kind = _infer_kind(ref_cols[name])
        assessed.append(
            _assess(name, kind, ref_cols[name], cur_cols.get(name, missing_column), cfg)
        )

    prediction = None
    if reference_predictions is not None and current_predictions is not None:
        prediction = _assess(
            _PREDICTION,
            "numeric",
            pd.Series(reference_predictions),
            pd.Series(current_predictions),
            cfg,
        )

    return DriftReport(
        config=cfg,
        features=tuple(assessed),
        prediction=prediction,
        n_reference=n_ref,
        n_current=n_cur,
        absent_from_current=absent,
        new_in_current=tuple(sorted(set(cur_cols) - set(ref_cols))),
    )


def _utc(timestamp: str) -> pd.Timestamp:
    ts = pd.Timestamp(timestamp)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts


def drift_report_from_store(
    store: FeatureSnapshotStore,
    family: str,
    reference: pd.DataFrame,
    *,
    window_start: str,
    cutoff: str,
    features: Sequence[str] | None = None,
    predict_fn: Callable[[pd.DataFrame], Sequence[float] | npt.ArrayLike] | None = None,
    config: DriftConfig | None = None,
) -> DriftReport:
    """Drift of a live window in a `FeatureSnapshotStore` against a training panel.

    The current window is built with `store.as_of(family, cutoff)` — the store's
    own point-in-time read — so it holds, per market, the latest snapshot captured
    **strictly before** `cutoff`; nothing later can leak in. It is then restricted
    to snapshots captured at or after `window_start`. The lower bound is
    required: without it "current" would include the very markets the model was
    trained on and drift would be understated.

    Args:
        store: The point-in-time snapshot store.
        family: Market family to read.
        reference: The training panel (e.g. from `build_panel_from_snapshots`).
        window_start: ISO-8601 start of the live window (inclusive). A naive
            timestamp is read as UTC.
        cutoff: ISO-8601 end of the live window (exclusive), same convention as
            `FeatureSnapshotStore.as_of`.
        features: Columns to assess; defaults to the reference's ``feature_*``
            columns.
        predict_fn: Optional. Called with the full reference panel and the full
            current window, it returns predicted probabilities (one per row, the
            probability of the label you trade). Use it to enable prediction drift.
            It is not called on an empty window.
        config: See `DriftConfig`.

    An empty window yields a report whose features are all "insufficient" rather
    than an error: no live data is not evidence the world is fine.
    """
    feats = (
        list(features)
        if features is not None
        else [str(c) for c in reference.columns if str(c).startswith(_FEATURE_PREFIX)]
    )

    snapshots = store.as_of(family, cutoff)
    if not snapshots.empty:
        snapshots = snapshots[snapshots["captured_at"] >= _utc(window_start)]
    # A zero-row frame with the reference's columns keeps the report's shape stable.
    current = reference.iloc[0:0] if snapshots.empty else snapshots.reset_index(drop=True)

    reference_predictions = current_predictions = None
    if predict_fn is not None:
        reference_predictions = np.asarray(predict_fn(reference), dtype=float)
        current_predictions = (
            np.empty(0, dtype=float)
            if current.empty
            else np.asarray(predict_fn(current), dtype=float)
        )

    wanted = set(feats) | {c for c in current.columns if str(c).startswith(_FEATURE_PREFIX)}
    return drift_report(
        reference,
        current[[c for c in current.columns if c in wanted]],
        features=feats,
        reference_predictions=reference_predictions,
        current_predictions=current_predictions,
        config=config,
    )
