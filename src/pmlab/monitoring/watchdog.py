"""The watchdog: one health verdict over drift, calibration and PnL.

`drift_report`, `CalibrationTracker` and `BacktestExpectation` compute numbers and
leave the decision to someone else. This is that someone: it owns the thresholds,
the policy for "we could not check", and the exit code a cron job acts on. It
consumes the others; it does not recompute them.

`health_report` is pure. It takes inputs that are already loaded — a drift report,
the trade log, the champion manifest — and does no I/O and reads no clock (`as_of`
is a required argument), so the same inputs give a byte-identical report on any day.
`pmlab monitor` is a thin loader and printer around it.

Four things worth knowing before reading the code:

**Only evidence that belongs to this champion counts.** A trade log accumulates
across champions and across segments the gate rejected. The calibration and PnL
checks therefore see only settled trades in the champion's allowed segments,
recorded at or after it was published, whose truth had arrived by `as_of`. Anything
else is counted and reported, never silently folded in.

**A shortfall needs two things to be a finding.** It has to be *material* (an
effect-size threshold) and *distinguishable from noise* (a one-sided two-sample z
test of the live mean against the backtest's, using both sides' spread). The first
alone pages on noise at small n; the second alone pages on trivia at large n.

**`insufficient` is not healthy.** "We could not check" outranks "ok" and loses to
any measured problem, and it exits with its own code. A new champion with no trades
yet is not a healthy champion.

**Nothing here is statistical certainty.** See `docs/watchdog.md` for what the
test does and does not assume.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from typing import Any, Literal, TypeGuard, cast

import numpy as np

from pmlab.backtest.expectation import BacktestExpectation
from pmlab.core.costs import CostModel
from pmlab.modeling.champion import ChampionManifest
from pmlab.monitoring.calibration_tracker import CalibrationTracker
from pmlab.monitoring.drift import DriftConfig, DriftReport, DriftStatus, FeatureDrift

__all__ = [
    "CHECKS",
    "EXIT_ALERT",
    "EXIT_INSUFFICIENT",
    "EXIT_INTERNAL",
    "EXIT_OK",
    "EXIT_USAGE",
    "EXIT_WARN",
    "CalibrationThresholds",
    "ChampionRef",
    "CheckResult",
    "HealthReport",
    "PnLThresholds",
    "WatchdogConfig",
    "WatchdogInputError",
    "exit_code_for",
    "health_report",
    "parse_timestamp",
]

HealthStatus = DriftStatus
InsufficientPolicy = Literal["separate", "warn"]

CHECKS: tuple[str, ...] = ("drift", "calibration", "pnl")

# Exit codes. 0/10/20/30 are verdicts; 64/70 follow sysexits.h (EX_USAGE,
# EX_SOFTWARE). They deliberately avoid 1 and 2: an uncaught Python exception exits
# 1 and a bad command line exits 2 (Click), and neither may be mistaken for a
# warn or an alert by a script that branches on the code.
EXIT_OK = 0
EXIT_WARN = 10
EXIT_ALERT = 20
EXIT_INSUFFICIENT = 30
EXIT_USAGE = 64
EXIT_INTERNAL = 70

_EXIT_BY_STATUS: dict[str, int] = {
    "ok": EXIT_OK,
    "warn": EXIT_WARN,
    "alert": EXIT_ALERT,
    "insufficient": EXIT_INSUFFICIENT,
}

# alert > warn > insufficient > ok. Same precedence as `DriftReport`, for the same
# reason: "could not check" must never read as "checked and fine", and must not
# hide a measured problem either.
_SEVERITY: dict[str, int] = {"ok": 0, "insufficient": 1, "warn": 2, "alert": 3}

# Slack for float round-off at a threshold: 0.25 - 0.20 is 0.04999999999999999.
_TOLERANCE = 1e-9

# Quote/fill arithmetic is rounded when the broker writes it (fill to 8 places,
# fee to 6), so equality with the cost model is only up to that rounding.
_FILL_TOLERANCE = 2e-6
_FEE_TOLERANCE = 2e-6


class WatchdogInputError(ValueError):
    """The inputs or configuration are unusable. The CLI exits 64 on this."""


def exit_code_for(status: HealthStatus) -> int:
    """The process exit code for a health status."""
    try:
        return _EXIT_BY_STATUS[status]
    except KeyError:
        raise ValueError(f"unknown health status {status!r}") from None


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


def _check_int(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise WatchdogInputError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def _check_float(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise WatchdogInputError(f"{name} must be a number, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise WatchdogInputError(f"{name} must be a number, got {value!r}") from None
    if not math.isfinite(number) or number < 0:
        raise WatchdogInputError(f"{name} must be a finite number >= 0, got {value!r}")
    return number


def _check_pair(low_name: str, low: float, high_name: str, high: float) -> None:
    if low > high:
        raise WatchdogInputError(
            f"{low_name} ({low:g}) must not exceed {high_name} ({high:g}): "
            "warn is the lower bar and alert the higher"
        )


def _from_dict_strict(cls: type[Any], data: Mapping[str, Any], label: str) -> Any:
    """Build a thresholds dataclass from a partial dict; unknown keys are an error."""
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise WatchdogInputError(
            f"unknown {label} setting(s) {unknown}; valid settings: {sorted(known)}"
        )
    return cls(**dict(data))


@dataclass(frozen=True)
class CalibrationThresholds:
    """When live Brier has fallen far enough behind the backtest's to act on.

    A level (warn or alert) is reached only when the live Brier is worse than the
    backtest's by at least the absolute **and** the relative margin, **and** the
    gap clears the noise (`z`). Absolute alone over-fires on a tiny Brier; relative
    alone over-fires on a large one. All values are conventions, not standards.
    """

    min_trades: int = 30
    brier_warn_abs: float = 0.02
    brier_alert_abs: float = 0.05
    brier_warn_rel: float = 0.10
    brier_alert_rel: float = 0.25
    z_warn: float = 2.0
    z_alert: float = 3.0
    n_bins: int = 10

    def __post_init__(self) -> None:
        object.__setattr__(self, "min_trades", _check_int("min_trades", self.min_trades, 2))
        object.__setattr__(self, "n_bins", _check_int("n_bins", self.n_bins, 2))
        for name in (
            "brier_warn_abs",
            "brier_alert_abs",
            "brier_warn_rel",
            "brier_alert_rel",
            "z_warn",
            "z_alert",
        ):
            object.__setattr__(self, name, _check_float(name, getattr(self, name)))
        _check_pair("brier_warn_abs", self.brier_warn_abs, "brier_alert_abs", self.brier_alert_abs)
        _check_pair("brier_warn_rel", self.brier_warn_rel, "brier_alert_rel", self.brier_alert_rel)
        _check_pair("z_warn", self.z_warn, "z_alert", self.z_alert)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CalibrationThresholds:
        result: CalibrationThresholds = _from_dict_strict(cls, data, "calibration")
        return result


@dataclass(frozen=True)
class PnLThresholds:
    """When live PnL per unit stake has fallen far enough behind the backtest's.

    `shortfall_*` is the fraction of the backtest mean that is missing:
    ``(backtest - live) / backtest``. 0.5 means earning half of what was promised;
    1.0 means earning nothing. A level needs the shortfall **and** a one-sided z of
    at least `z_*`. The defaults of 2 and 3 are stricter than the textbook 1.64 and
    2.33 on purpose: the watchdog looks every run, and repeated looks at the same
    growing sample inflate the false-alarm rate far above the nominal one.
    """

    min_trades: int = 30
    shortfall_warn: float = 0.5
    shortfall_alert: float = 1.0
    z_warn: float = 2.0
    z_alert: float = 3.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "min_trades", _check_int("min_trades", self.min_trades, 2))
        for name in ("shortfall_warn", "shortfall_alert", "z_warn", "z_alert"):
            object.__setattr__(self, name, _check_float(name, getattr(self, name)))
        _check_pair("shortfall_warn", self.shortfall_warn, "shortfall_alert", self.shortfall_alert)
        _check_pair("z_warn", self.z_warn, "z_alert", self.z_alert)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PnLThresholds:
        result: PnLThresholds = _from_dict_strict(cls, data, "pnl")
        return result


@dataclass(frozen=True)
class WatchdogConfig:
    """Every threshold the watchdog applies, written into every report.

    `insufficient` is the policy for a check that could not be measured:
    ``"separate"`` (default) gives the report its own status and exit code, so a
    new champion with no trades yet is distinguishable from a healthy one;
    ``"warn"`` counts it as a warn. There is deliberately no ``"ignore"``.
    """

    drift: DriftConfig = field(default_factory=DriftConfig)
    calibration: CalibrationThresholds = field(default_factory=CalibrationThresholds)
    pnl: PnLThresholds = field(default_factory=PnLThresholds)
    insufficient: InsufficientPolicy = "separate"

    def __post_init__(self) -> None:
        if self.insufficient not in ("separate", "warn"):
            raise WatchdogInputError(
                f"insufficient must be 'separate' or 'warn', got {self.insufficient!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "drift": self.drift.to_dict(),
            "calibration": self.calibration.to_dict(),
            "pnl": self.pnl.to_dict(),
            "insufficient": self.insufficient,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WatchdogConfig:
        """Build from a possibly partial dict: absent keys keep their defaults.

        Unknown keys raise. A typo'd threshold that was quietly ignored would leave
        the default in force while the operator believed otherwise.
        """
        known = {"drift", "calibration", "pnl", "insufficient"}
        unknown = sorted(set(data) - known)
        if unknown:
            raise WatchdogInputError(
                f"unknown setting(s) {unknown}; valid sections: {sorted(known)}"
            )
        drift_data = dict(data.get("drift", {}))
        drift_known = {"n_bins", "min_samples", "epsilon", "thresholds", "numeric", "categorical"}
        drift_unknown = sorted(set(drift_data) - drift_known)
        if drift_unknown:
            raise WatchdogInputError(
                f"unknown drift setting(s) {drift_unknown}; valid settings: {sorted(drift_known)}"
            )
        default_thresholds = DriftConfig().thresholds.to_dict()
        given = dict(drift_data.get("thresholds", {}))
        bad = sorted(set(given) - set(default_thresholds))
        if bad:
            raise WatchdogInputError(
                f"unknown drift threshold(s) {bad}; valid: {sorted(default_thresholds)}"
            )
        drift_data["thresholds"] = {**default_thresholds, **given}
        merged = {**DriftConfig().to_dict(), **drift_data}
        try:
            drift = DriftConfig.from_dict(merged)
        except (ValueError, TypeError, KeyError) as exc:
            raise WatchdogInputError(f"invalid drift settings: {exc}") from exc
        return cls(
            drift=drift,
            calibration=CalibrationThresholds.from_dict(data.get("calibration", {})),
            pnl=PnLThresholds.from_dict(data.get("pnl", {})),
            insufficient=data.get("insufficient", "separate"),
        )


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


def _clean(value: Any) -> Any:
    """Make a metrics value strict-JSON: no NaN/inf, no numpy scalars, no tuples."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


@dataclass(frozen=True)
class CheckResult:
    """One check's verdict, the reasons for it, and the numbers behind it.

    `reasons` is never empty: a problem says what is wrong and against which
    threshold, a healthy check says what it measured. `cause` is set only for
    ``insufficient`` and says why it could not be measured:
    ``too_few_samples``, ``no_expectation``, ``invalid_expectation`` or
    ``not_evaluated``.
    """

    name: str
    status: HealthStatus
    reasons: tuple[str, ...]
    metrics: dict[str, Any]
    cause: str | None = None

    def __post_init__(self) -> None:
        # Sanitised once, here, so what is held is what is serialised: an infinite
        # ratio becomes None in memory too, and a report equals its own round trip.
        object.__setattr__(self, "metrics", _clean(self.metrics))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "cause": self.cause,
            "reasons": list(self.reasons),
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CheckResult:
        status = data["status"]
        if status not in _SEVERITY:
            raise ValueError(f"unknown check status {status!r}")
        cause = data.get("cause")
        return cls(
            name=str(data["name"]),
            status=cast(HealthStatus, status),
            reasons=tuple(str(r) for r in data["reasons"]),
            metrics=dict(data["metrics"]),
            cause=None if cause is None else str(cause),
        )


@dataclass(frozen=True)
class ChampionRef:
    """Which champion this report is about, and what it was judged against."""

    model_name: str
    plugin_family: str
    published_at: str
    allowed_segments: tuple[str, ...]
    has_expectation: bool
    costs: dict[str, float] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "plugin_family": self.plugin_family,
            "published_at": self.published_at,
            "allowed_segments": list(self.allowed_segments),
            "has_expectation": self.has_expectation,
            "costs": None if self.costs is None else dict(self.costs),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ChampionRef:
        costs = data.get("costs")
        return cls(
            model_name=str(data["model_name"]),
            plugin_family=str(data["plugin_family"]),
            published_at=str(data["published_at"]),
            allowed_segments=tuple(str(s) for s in data["allowed_segments"]),
            has_expectation=bool(data["has_expectation"]),
            costs=None if costs is None else {str(k): float(v) for k, v in costs.items()},
        )


@dataclass(frozen=True)
class HealthReport:
    """The watchdog's verdict. Frozen, strict-JSON-safe, deterministic."""

    as_of: str
    champion: ChampionRef
    checks: tuple[CheckResult, ...]
    skipped: tuple[str, ...]
    config: WatchdogConfig

    @property
    def status(self) -> HealthStatus:
        """The worst check, after the configured policy for `insufficient`."""
        effective = [
            "warn"
            if c.status == "insufficient" and self.config.insufficient == "warn"
            else c.status
            for c in self.checks
        ]
        return max(effective, key=lambda s: _SEVERITY[s])

    @property
    def exit_code(self) -> int:
        return exit_code_for(self.status)

    @property
    def problems(self) -> tuple[CheckResult, ...]:
        """Every check that is not plainly ok, `insufficient` included."""
        return tuple(c for c in self.checks if c.status != "ok")

    def render(self, *, verbose: bool = False) -> str:
        """Short, actionable text for a human or a cron mail; "" when healthy.

        Quiet on green is a property of the report, not of the CLI: a healthy
        report renders to nothing unless `verbose` asks for the full picture.
        """
        if self.status == "ok" and not verbose:
            return ""
        lines = [
            f"pmlab monitor: {self.status.upper()}  "
            f"(champion {self.champion.model_name}, as of {self.as_of})"
        ]
        shown = self.checks if verbose else self.problems
        for check in shown:
            first, *rest = check.reasons
            lines.append(f"  {check.name:<12}{check.status:<14}{first}")
            lines.extend(f"  {'':<26}{reason}" for reason in rest)
        if self.skipped:
            lines.append(f"  skipped: {', '.join(self.skipped)}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "exit_code": self.exit_code,
            "as_of": self.as_of,
            "champion": self.champion.to_dict(),
            "skipped": list(self.skipped),
            "checks": [c.to_dict() for c in self.checks],
            "config": self.config.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HealthReport:
        """Restore a report. `status` and `exit_code` are derived, not trusted."""
        return cls(
            as_of=str(data["as_of"]),
            champion=ChampionRef.from_dict(data["champion"]),
            checks=tuple(CheckResult.from_dict(c) for c in data["checks"]),
            skipped=tuple(str(s) for s in data["skipped"]),
            config=WatchdogConfig.from_dict(data["config"]),
        )


# --------------------------------------------------------------------------
# Selecting the evidence
# --------------------------------------------------------------------------


def parse_timestamp(value: Any) -> datetime | None:
    """An ISO-8601 string as an aware UTC datetime; a naive one is read as UTC."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _is_number(value: Any) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass(frozen=True)
class _Selection:
    trades: tuple[Mapping[str, Any], ...]
    counts: dict[str, int]
    by_segment: dict[str, int]


def _select_trades(
    trades: Sequence[Mapping[str, Any]],
    *,
    allowed_segments: set[str],
    published_at: datetime,
    as_of: datetime,
) -> _Selection:
    """The settled trades that are evidence about *this* champion as of `as_of`.

    Each excluded trade is counted under the first reason that applies, so the
    counts add up to `n_total`.
    """
    counts = {
        "n_total": len(trades),
        "n_open": 0,
        "n_out_of_segment": 0,
        "n_unattributed": 0,
        "n_before_champion": 0,
        "n_after_as_of": 0,
    }
    kept: list[Mapping[str, Any]] = []
    by_segment: dict[str, int] = {}
    for index, trade in enumerate(trades):
        if not isinstance(trade, Mapping):
            raise WatchdogInputError(f"trade #{index} is not an object")
        if trade.get("outcome") not in ("won", "lost") or not _is_number(trade.get("realized_pnl")):
            counts["n_open"] += 1
            continue
        segment = str(trade.get("city_or_segment", ""))
        if segment not in allowed_segments:
            counts["n_out_of_segment"] += 1
            continue
        recorded = parse_timestamp(trade.get("recorded_at"))
        target = parse_timestamp(trade.get("target_date"))
        if recorded is None or target is None:
            counts["n_unattributed"] += 1
            continue
        if recorded < published_at:
            counts["n_before_champion"] += 1
            continue
        if recorded > as_of or target.date() > as_of.date():
            counts["n_after_as_of"] += 1
            continue
        kept.append(trade)
        by_segment[segment] = by_segment.get(segment, 0) + 1
    counts["n_in_scope"] = len(kept)
    # Canonical order, so the report does not depend on the order rows sit in the
    # file: float sums differ in the last bit when tied rows are visited in a
    # different order, and a byte-identical report is part of the contract.
    kept.sort(key=lambda t: json.dumps(t, sort_keys=True, default=str))
    return _Selection(tuple(kept), counts, dict(sorted(by_segment.items())))


_EXCLUSION_LABELS = (
    ("n_open", "open"),
    ("n_out_of_segment", "outside the champion's allowed segments"),
    ("n_unattributed", "with no usable recorded_at/target_date"),
    ("n_before_champion", "recorded before this champion was published"),
    ("n_after_as_of", "not yet knowable at as_of"),
    ("n_unscoreable", "with no usable stake or PnL"),
    ("n_excluded_cost_basis", "filled under different costs than champion.json records"),
)


def _excluded_summary(metrics: Mapping[str, Any]) -> str:
    parts = [f"{metrics[key]} {label}" for key, label in _EXCLUSION_LABELS if metrics.get(key)]
    return "excluded: " + "; ".join(parts) if parts else "none excluded"


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Evidence:
    """How far the live mean is behind the reference, and whether that is noise."""

    gap: float  # reference - live for "higher is better", live - reference otherwise
    z: float | None  # None when both samples have zero spread
    p_value: float | None

    def clears(self, z_threshold: float) -> bool:
        """One-sided: is the gap in the bad direction and above the noise?"""
        if self.gap <= 0:
            return False
        return True if self.z is None else self.z >= z_threshold - _TOLERANCE


def _one_sided_welch(
    *,
    live_mean: float,
    live_var: float,
    n_live: int,
    ref_mean: float,
    ref_std: float,
    ref_n: int,
    worse_when_higher: bool,
) -> _Evidence:
    """Welch's z for "live is worse than the reference", normal approximation.

    The standard error combines both sides: the live sample's own noise and the
    backtest's, because the backtest mean is itself an estimate. A promise made on
    20 trades is a weak promise and falling short of it means less than falling
    short of one made on 2000.
    """
    gap = (live_mean - ref_mean) if worse_when_higher else (ref_mean - live_mean)
    se = math.sqrt(live_var / n_live + ref_std**2 / ref_n)
    if se <= 0:
        return _Evidence(gap=gap, z=None, p_value=None)
    z = gap / se
    return _Evidence(gap=gap, z=z, p_value=0.5 * math.erfc(z / math.sqrt(2.0)))


def _fmt_z(z: float | None) -> str:
    return "inf" if z is None else f"{z:.1f}"


# --------------------------------------------------------------------------
# The three checks
# --------------------------------------------------------------------------


def _insufficient(
    name: str, cause: str, reasons: Sequence[str], metrics: dict[str, Any]
) -> CheckResult:
    return CheckResult(
        name=name,
        status="insufficient",
        reasons=tuple(reasons),
        metrics=metrics,
        cause=cause,
    )


def _drift_check(report: DriftReport | None) -> CheckResult:
    if report is None:
        return _insufficient(
            "drift",
            "not_evaluated",
            [
                "not evaluated: no drift report was supplied "
                "(needs a reference panel and a live snapshot window, or skip this check)"
            ],
            {},
        )
    assessed: list[FeatureDrift] = list(report.features)
    if report.prediction is not None:
        assessed.append(report.prediction)
    thresholds = report.config.thresholds

    offenders = sorted(
        (f for f in assessed if f.status in ("warn", "alert")),
        key=lambda f: (-_SEVERITY[f.status], f.name),
    )
    unmeasurable = sorted(f.name for f in assessed if f.status == "insufficient")
    metrics: dict[str, Any] = {
        "n_ok": report.n_ok,
        "n_warn": report.n_warn,
        "n_alert": report.n_alert,
        "n_insufficient": report.n_insufficient,
        "max_psi": report.max_psi,
        "max_kl": report.max_kl,
        "prediction_psi": report.prediction_psi,
        "prediction_kl": report.prediction_kl,
        "n_reference": report.n_reference,
        "n_current": report.n_current,
        "n_current_excluded_overlap": report.n_current_excluded_overlap,
        "offending": [f.name for f in offenders],
        "insufficient_features": unmeasurable,
        "absent_from_current": list(report.absent_from_current),
    }
    status = report.status

    if status in ("warn", "alert"):
        reasons = [_drift_reason(f, thresholds) for f in offenders]
        return CheckResult("drift", status, tuple(reasons), metrics)

    if status == "insufficient":
        by_name = {f.name: f for f in assessed}
        reasons = [
            f"{name}: insufficient ({by_name[name].n_current_valid} usable live values, "
            f"{by_name[name].n_reference_valid} reference; need {report.config.min_samples})"
            for name in unmeasurable
        ]
        return _insufficient("drift", "too_few_samples", reasons, metrics)

    max_psi = report.max_psi
    psi_text = "n/a" if max_psi is None else f"{max_psi:.3f}"
    reason = (
        f"{len(assessed)} feature(s) within thresholds "
        f"(max psi {psi_text}, {report.n_current} live rows)"
    )
    return CheckResult("drift", "ok", (reason,), metrics)


def _drift_reason(f: FeatureDrift, thresholds: Any) -> str:
    level = f.status
    parts: list[str] = []
    for driver in f.drivers:
        if driver == "psi" and f.psi is not None:
            limit = thresholds.psi_alert if level == "alert" else thresholds.psi_warn
            parts.append(f"psi {f.psi:.3f} >= {limit:g}")
        elif driver == "kl" and f.kl is not None:
            limit = thresholds.kl_alert if level == "alert" else thresholds.kl_warn
            parts.append(f"kl {f.kl:.3f} >= {limit:g}")
        elif driver == "missing_rate" and f.missing_rate_change is not None:
            limit = thresholds.missing_alert if level == "alert" else thresholds.missing_warn
            parts.append(f"missing-rate change {f.missing_rate_change:+.3f} >= {limit:g}")
    detail = ", ".join(parts) if parts else "see report"
    return f"{f.name}: {level} ({detail})"


def _calibration_check(
    selection: _Selection, expectation: BacktestExpectation | None, cfg: CalibrationThresholds
) -> CheckResult:
    tracker = CalibrationTracker([dict(t) for t in selection.trades], n_bins=cfg.n_bins)
    metrics: dict[str, Any] = {
        **selection.counts,
        "n_by_segment": selection.by_segment,
        "n_trades": tracker.n_realized,
        "n_skipped_no_forecast": tracker.n_skipped,
        "min_trades": cfg.min_trades,
    }

    if expectation is None or expectation.brier_score is None or expectation.brier_std is None:
        return _insufficient(
            "calibration",
            "no_expectation",
            [
                "no backtest Brier expectation recorded in champion.json: re-publish the "
                "champion with a BacktestExpectation to enable this check"
            ],
            metrics,
        )
    if tracker.n_realized < cfg.min_trades:
        return _insufficient(
            "calibration",
            "too_few_samples",
            [
                f"{tracker.n_realized} scoreable settled trades, need >= {cfg.min_trades} "
                f"({_excluded_summary(metrics)})"
            ],
            metrics,
        )

    terms = np.array(tracker.squared_errors(), dtype=float)
    overall = tracker.overall()
    assert overall is not None  # n_realized >= min_trades >= 2
    live = float(terms.mean())
    reference = expectation.brier_score
    evidence = _one_sided_welch(
        live_mean=live,
        live_var=float(terms.var(ddof=1)),
        n_live=len(terms),
        ref_mean=reference,
        ref_std=expectation.brier_std,
        ref_n=expectation.n_trades,
        worse_when_higher=True,
    )
    gap = evidence.gap
    rel = gap / reference if reference > 0 else math.inf
    metrics.update(
        {
            "brier_live": live,
            "brier_backtest": reference,
            "degradation_abs": gap,
            "degradation_rel": rel,
            "z": evidence.z,
            "p_value": evidence.p_value,
            "reliability_live": overall.reliability,
        }
    )

    summary = (
        f"Brier {live:.3f} vs backtest {reference:.3f} "
        f"({gap:+.3f} abs, {rel:+.1%} rel, z={_fmt_z(evidence.z)}, n={len(terms)})"
    )
    if (
        gap >= cfg.brier_alert_abs - _TOLERANCE
        and rel >= cfg.brier_alert_rel - _TOLERANCE
        and evidence.clears(cfg.z_alert)
    ):
        needs = (
            f"alert needs >= +{cfg.brier_alert_abs:g} abs, >= +{cfg.brier_alert_rel:.0%} rel, "
            f"z >= {cfg.z_alert:g}"
        )
        return CheckResult("calibration", "alert", (f"{summary}; {needs}",), metrics)
    if (
        gap >= cfg.brier_warn_abs - _TOLERANCE
        and rel >= cfg.brier_warn_rel - _TOLERANCE
        and evidence.clears(cfg.z_warn)
    ):
        needs = (
            f"warn needs >= +{cfg.brier_warn_abs:g} abs, >= +{cfg.brier_warn_rel:.0%} rel, "
            f"z >= {cfg.z_warn:g}"
        )
        return CheckResult("calibration", "warn", (f"{summary}; {needs}",), metrics)
    return CheckResult("calibration", "ok", (f"{summary}; within thresholds",), metrics)


def _cost_basis(trade: Mapping[str, Any], costs: CostModel, stake: float) -> str:
    """ "match", "mismatch" or "unverified" for one trade against the champion's costs."""
    verdicts: list[bool] = []
    fee = trade.get("fee_paid")
    if _is_number(fee):
        verdicts.append(abs(float(fee) - costs.fee(stake)) <= _FEE_TOLERANCE)
    fill = trade.get("fill_price")
    gamma = trade.get("gamma_price")
    if _is_number(fill) and _is_number(gamma):
        quote = float(gamma) if trade.get("direction", "yes") == "yes" else 1.0 - float(gamma)
        verdicts.append(abs(float(fill) - costs.fill_price(quote, stake=stake)) <= _FILL_TOLERANCE)
    if not verdicts:
        return "unverified"
    return "match" if all(verdicts) else "mismatch"


def _pnl_check(
    selection: _Selection,
    expectation: BacktestExpectation | None,
    costs: CostModel | None,
    cfg: PnLThresholds,
) -> CheckResult:
    returns: list[float] = []
    n_unscoreable = 0
    n_cost_excluded = 0
    n_cost_unverified = 0
    for trade in selection.trades:
        stake = trade.get("flat_stake", 1.0)
        pnl = trade.get("realized_pnl")
        if not _is_number(stake) or not _is_number(pnl) or stake <= 0 or not math.isfinite(pnl):
            n_unscoreable += 1
            continue
        if costs is not None:
            verdict = _cost_basis(trade, costs, float(stake))
            if verdict == "mismatch":
                n_cost_excluded += 1
                continue
            if verdict == "unverified":
                n_cost_unverified += 1
        returns.append(float(pnl) / float(stake))

    metrics: dict[str, Any] = {
        **selection.counts,
        "n_by_segment": selection.by_segment,
        "n_trades": len(returns),
        "n_unscoreable": n_unscoreable,
        "n_excluded_cost_basis": n_cost_excluded,
        "n_cost_unverified": n_cost_unverified if costs is not None else None,
        "cost_basis": (
            "unknown" if costs is None else ("mismatch" if n_cost_excluded else "match")
        ),
        "min_trades": cfg.min_trades,
    }

    if expectation is None:
        return _insufficient(
            "pnl",
            "no_expectation",
            [
                "no backtest PnL expectation recorded in champion.json: re-publish the "
                "champion with a BacktestExpectation to enable this check"
            ],
            metrics,
        )
    if (
        not math.isfinite(expectation.mean_pnl_per_stake)
        or expectation.mean_pnl_per_stake <= 0
        or expectation.n_trades < 2
    ):
        return _insufficient(
            "pnl",
            "invalid_expectation",
            [
                f"backtest mean PnL per stake is {expectation.mean_pnl_per_stake:+.3f} over "
                f"{expectation.n_trades} trades: a shortfall needs a positive expectation to "
                "be measured against"
            ],
            metrics,
        )
    if len(returns) < cfg.min_trades:
        return _insufficient(
            "pnl",
            "too_few_samples",
            [
                f"{len(returns)} scoreable settled trades, need >= {cfg.min_trades} "
                f"({_excluded_summary(metrics)})"
            ],
            metrics,
        )

    arr = np.array(returns, dtype=float)
    live = float(arr.mean())
    reference = expectation.mean_pnl_per_stake
    evidence = _one_sided_welch(
        live_mean=live,
        live_var=float(arr.var(ddof=1)),
        n_live=len(arr),
        ref_mean=reference,
        ref_std=expectation.std_pnl_per_stake,
        ref_n=expectation.n_trades,
        worse_when_higher=False,
    )
    shortfall = evidence.gap / reference
    metrics.update(
        {
            "mean_live": live,
            "std_live": float(arr.std(ddof=1)),
            "mean_backtest": reference,
            "shortfall_rel": shortfall,
            "z": evidence.z,
            "p_value": evidence.p_value,
        }
    )

    summary = (
        f"mean PnL/stake {live:+.3f} vs backtest {reference:+.3f} "
        f"(shortfall {shortfall:.0%}, z={_fmt_z(evidence.z)}, n={len(arr)})"
    )
    if shortfall >= cfg.shortfall_alert - _TOLERANCE and evidence.clears(cfg.z_alert):
        needs = f"alert needs shortfall >= {cfg.shortfall_alert:.0%}, z >= {cfg.z_alert:g}"
        return CheckResult("pnl", "alert", (f"{summary}; {needs}",), metrics)
    if shortfall >= cfg.shortfall_warn - _TOLERANCE and evidence.clears(cfg.z_warn):
        needs = f"warn needs shortfall >= {cfg.shortfall_warn:.0%}, z >= {cfg.z_warn:g}"
        return CheckResult("pnl", "warn", (f"{summary}; {needs}",), metrics)
    return CheckResult("pnl", "ok", (f"{summary}; within thresholds",), metrics)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def health_report(
    *,
    champion: ChampionManifest,
    trades: Sequence[Mapping[str, Any]],
    drift: DriftReport | None,
    as_of: str,
    config: WatchdogConfig | None = None,
    checks: Sequence[str] = CHECKS,
) -> HealthReport:
    """Judge a deployed champion: drift, live calibration, live PnL.

    Pure: no I/O, no clock. Everything is passed in already loaded.

    Args:
        champion: The authoritative record of what was promoted — its allowed
            segments, publish time, cost model and backtest expectation. Never
            substituted by a retrain summary (AGENTS.md safety rule 5).
        trades: The paper/live trade log rows (`{"trades": [...]}`'s list).
        drift: A `DriftReport`, or None if drift was not evaluated (reported as
            unavailable, not as fine).
        as_of: ISO-8601 cutoff. Nothing recorded after it, and no trade whose
            `target_date` is after it, is evidence. Naive means UTC.
        config: Thresholds and the `insufficient` policy; defaults if omitted.
        checks: Which of `CHECKS` to run. The rest are listed in `skipped`.

    Raises:
        WatchdogInputError: Unparseable `as_of` or publish time, an unknown or
            empty `checks`, or a malformed trade row.
    """
    config = config or WatchdogConfig()

    unknown = sorted(set(checks) - set(CHECKS))
    if unknown:
        raise WatchdogInputError(f"unknown check(s) {unknown}; valid checks: {list(CHECKS)}")
    if not checks:
        raise WatchdogInputError(f"run at least one of {list(CHECKS)}")
    selected_checks = [name for name in CHECKS if name in checks]
    skipped = tuple(name for name in CHECKS if name not in checks)

    as_of_ts = parse_timestamp(as_of)
    if as_of_ts is None:
        raise WatchdogInputError(f"as_of {as_of!r} is not an ISO-8601 timestamp")
    published_ts = parse_timestamp(champion.published_at)
    if published_ts is None:
        raise WatchdogInputError(
            f"champion published_at {champion.published_at!r} is not an ISO-8601 timestamp"
        )

    allowed = champion.get_allowed_segments()
    selection = _select_trades(
        trades, allowed_segments=allowed, published_at=published_ts, as_of=as_of_ts
    )

    results: list[CheckResult] = []
    for name in selected_checks:
        if name == "drift":
            results.append(_drift_check(drift))
        elif name == "calibration":
            results.append(_calibration_check(selection, champion.expectation, config.calibration))
        else:
            results.append(_pnl_check(selection, champion.expectation, champion.costs, config.pnl))

    ref = ChampionRef(
        model_name=champion.model_name,
        plugin_family=champion.plugin_family,
        published_at=champion.published_at,
        allowed_segments=tuple(sorted(allowed)),
        has_expectation=champion.expectation is not None,
        costs=None if champion.costs is None else champion.costs.to_dict(),
    )
    return HealthReport(
        as_of=as_of,
        champion=ref,
        checks=tuple(results),
        skipped=skipped,
        config=config,
    )
