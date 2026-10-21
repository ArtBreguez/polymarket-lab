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
recorded at or after it was published, and settled by `as_of` (`settled_at`, which
`SettlementEngine` writes; a trade from an older release has none and is judged by
its `target_date` instead, counted and noted in the reasons). Anything else is
counted and reported, never silently folded in.

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

import itertools
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from datetime import UTC, datetime
from typing import Any, Literal, TypeGuard, cast

import numpy as np

from pmlab.backtest.expectation import BacktestExpectation, SegmentExpectation
from pmlab.core.costs import CostModel
from pmlab.core.segments import ALL_SEGMENTS, segment_allowed
from pmlab.execution.settlement import MIN_TRUSTED_NO_SETTLEMENT_VERSION
from pmlab.modeling.champion import ChampionManifest
from pmlab.monitoring.calibration_tracker import CalibrationTracker
from pmlab.monitoring.drift import DriftConfig, DriftReport, DriftStatus, FeatureDrift
from pmlab.monitoring.stats import (
    cusum,
    detectable_gap,
    holm_adjust,
    mixture_reference,
    normal_isf,
    normal_sf,
)

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
    "parse_as_of",
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

    The **sequential test** (`sequential`) is a one-sided CUSUM on each trade's
    standardized shortfall, in settlement order. It is the other way a PnL alert is
    raised: it signals a run of shortfalls that the pooled mean dilutes with the healthy
    trades before it. `cusum_allowance` is the part of the promised edge a trade may fall
    short by without it counting (a fraction of the edge, in units of the trade's own
    spread: 0.5 is tuned to an edge that has vanished) and `cusum_threshold` is the
    height at which it signals. The defaults were calibrated by simulation, see
    `docs/watchdog.md`.

    A promise made on few trades is a noisy one, and unlike the fixed test the CUSUM
    cannot count that noise: `cusum_min_backtest_trades` is the fewest backtest trades a
    side may have been promised on for its trades to feed the sequential test.

    `mde_power` is the power the *minimum detectable shortfall* is quoted at.
    """

    min_trades: int = 30
    shortfall_warn: float = 0.5
    shortfall_alert: float = 1.0
    z_warn: float = 2.0
    z_alert: float = 3.0
    sequential: bool = True
    cusum_allowance: float = 0.5
    cusum_threshold: float = 40.0
    cusum_min_backtest_trades: int = 100
    mde_power: float = 0.8

    def __post_init__(self) -> None:
        object.__setattr__(self, "min_trades", _check_int("min_trades", self.min_trades, 2))
        for name in ("shortfall_warn", "shortfall_alert", "z_warn", "z_alert", "cusum_allowance"):
            object.__setattr__(self, name, _check_float(name, getattr(self, name)))
        _check_pair("shortfall_warn", self.shortfall_warn, "shortfall_alert", self.shortfall_alert)
        _check_pair("z_warn", self.z_warn, "z_alert", self.z_alert)
        if not isinstance(self.sequential, bool):
            raise WatchdogInputError(f"sequential must be true or false, got {self.sequential!r}")
        threshold = _check_float("cusum_threshold", self.cusum_threshold)
        if threshold <= 0:
            raise WatchdogInputError(f"cusum_threshold must be > 0, got {self.cusum_threshold!r}")
        object.__setattr__(self, "cusum_threshold", threshold)
        object.__setattr__(
            self,
            "cusum_min_backtest_trades",
            _check_int("cusum_min_backtest_trades", self.cusum_min_backtest_trades, 2),
        )
        power = _check_float("mde_power", self.mde_power)
        if not 0.0 < power < 1.0:
            raise WatchdogInputError(f"mde_power must be strictly between 0 and 1, got {power!r}")
        object.__setattr__(self, "mde_power", power)

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
    ``too_few_samples``, ``no_expectation``, ``invalid_expectation`` (the recorded
    expectation cannot be compared against), ``invalid_statistic`` (a number computed
    from the live data is not finite) or ``not_evaluated``.
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
    """An ISO-8601 string as an aware UTC datetime; a naive one is read as UTC.

    Lenient on purpose, for the timestamps *in the data* (a hand-written trade log, a
    bare ``target_date``). A cutoff the operator types goes through `parse_as_of`.
    """
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def parse_as_of(value: Any, *, name: str = "as_of") -> datetime:
    """A cutoff as an aware UTC datetime. It must carry an explicit UTC offset.

    A cutoff with none ("2026-03-01", "2026-03-01T00:00:00") is whoever's local
    midnight; reading it as UTC would silently move the window by hours on a cron box
    in another zone, and the report would still look right. So it is refused.

    Raises:
        WatchdogInputError: Not a string, not ISO-8601, or naive.
    """
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        raise WatchdogInputError(f"{name} {value!r} is not an ISO-8601 timestamp") from None
    if parsed.tzinfo is None:
        raise WatchdogInputError(
            f"{name} {value!r} has no UTC offset, so it could be anyone's local time; "
            "give one explicitly, e.g. 2026-03-01T00:00:00Z or 2026-03-01T00:00:00+01:00"
        )
    return parsed.astimezone(UTC)


def _is_number(value: Any) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass(frozen=True)
class _Selection:
    trades: tuple[Mapping[str, Any], ...]
    counts: dict[str, int]
    by_segment: dict[str, int]
    allowed_segments: tuple[str, ...]


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
        "n_no_settled_at": 0,
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
        if not segment_allowed(segment, allowed_segments):
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
        if recorded > as_of:
            counts["n_after_as_of"] += 1
            continue
        # When did the truth arrive? `settled_at` says so exactly. Without it (a trade
        # from an older release) the best available bound is the target date, which
        # is a *date*: it admits a trade settled later on the as-of day itself.
        if trade.get("settled_at") is None:
            if target.date() > as_of.date():
                counts["n_after_as_of"] += 1
                continue
            counts["n_no_settled_at"] += 1
        else:
            settled = parse_timestamp(trade["settled_at"])
            if settled is None:
                counts["n_unattributed"] += 1
                continue
            if settled > as_of:
                counts["n_after_as_of"] += 1
                continue
        kept.append(trade)
        by_segment[segment] = by_segment.get(segment, 0) + 1
    counts["n_in_scope"] = len(kept)
    # Canonical order, so the report does not depend on the order rows sit in the
    # file: float sums differ in the last bit when tied rows are visited in a
    # different order, and a byte-identical report is part of the contract.
    kept.sort(key=lambda t: json.dumps(t, sort_keys=True, default=str))
    return _Selection(
        tuple(kept), counts, dict(sorted(by_segment.items())), tuple(sorted(allowed_segments))
    )


_EXCLUSION_LABELS = (
    ("n_open", "open"),
    ("n_out_of_segment", "outside the champion's allowed segments"),
    ("n_unattributed", "with no usable recorded_at/target_date"),
    ("n_before_champion", "recorded before this champion was published"),
    ("n_after_as_of", "not yet settled at as_of"),
    ("n_unscoreable", "with no usable stake or PnL"),
    ("n_invalid_forecast", "with a model_prob that is not a probability"),
    ("n_excluded_no_side", "on the NO side (champion.json records no NO expectation)"),
    ("n_excluded_yes_side", "on the YES side (champion.json records no YES expectation)"),
    (
        "n_excluded_unfixed_no",
        "NO trades settled by a release that booked them wrongly (run `pmlab resettle`)",
    ),
    (
        "n_excluded_legacy_settlement",
        "settled at the quote by an older release although they paid a different fill",
    ),
    ("n_excluded_cost_basis", "filled under different costs than champion.json records"),
)


def _excluded_summary(metrics: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key, label in _EXCLUSION_LABELS:
        if not metrics.get(key):
            continue
        if key == "n_excluded_no_side" and metrics.get("expectation_directions") is None:
            # A champion with no per-side record: its backtest only ever bought YES.
            label = "on the NO side (the backtest only buys YES)"
        parts.append(f"{metrics[key]} {label}")
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
    ref_variance: float,
    worse_when_higher: bool,
) -> _Evidence:
    """Welch's z for "live is worse than the reference", normal approximation.

    The standard error combines both sides: the live sample's own noise and the
    backtest's, because the backtest mean is itself an estimate. A promise made on
    20 trades is a weak promise and falling short of it means less than falling
    short of one made on 2000. `ref_variance` is the sampling variance of the
    reference mean: ``std**2 / n`` for one group, and for a mix of sides the weighted
    sum `pmlab.monitoring.stats.mixture_reference` returns.
    """
    gap = (live_mean - ref_mean) if worse_when_higher else (ref_mean - live_mean)
    se = math.sqrt(live_var / n_live + ref_variance)
    if se <= 0:
        return _Evidence(gap=gap, z=None, p_value=None)
    z = gap / se
    return _Evidence(gap=gap, z=z, p_value=0.5 * math.erfc(z / math.sqrt(2.0)))


def _fmt_z(z: float | None) -> str:
    # `+ 0.0` turns -0.0 into 0.0: a z that rounds to zero must not print a sign.
    return "inf" if z is None else f"{round(z, 1) + 0.0:.1f}"


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


def _unusable_numbers(group: Any, *, calibration: bool, prefix: str) -> list[str]:
    """What stops one group of backtest numbers being compared against."""
    problems: list[str] = []
    n_trades = group.n_trades
    if isinstance(n_trades, bool) or not isinstance(n_trades, int) or n_trades < 2:
        problems.append(f"{prefix}n_trades is {n_trades!r} (need an integer >= 2)")
    fields = (
        ("brier_score", "brier_std") if calibration else ("mean_pnl_per_stake", "std_pnl_per_stake")
    )
    for name in fields:
        value = getattr(group, name)
        if not _is_number(value) or not math.isfinite(value):
            problems.append(f"{prefix}{name} is {value!r} (need a finite number)")
        elif "std" in name and value < 0:
            problems.append(f"{prefix}{name} is {value:g} (a spread cannot be negative)")
    return problems


def _unusable_expectation(expectation: BacktestExpectation, *, calibration: bool) -> list[str]:
    """What stops `expectation` being compared against, or [] if nothing does.

    `BacktestExpectation` refuses all of this when it is built, so this is the
    second line of defence for one that skipped that (mutated, duck-typed). NaN
    compares false with everything, so an unchecked NaN spread turns a z test into
    "no evidence" and the check into a silent pass; a zero count divides by zero.
    """
    problems = _unusable_numbers(expectation, calibration=calibration, prefix="")
    if expectation.by_direction is not None:
        for side, group in expectation.by_direction.items():
            problems.extend(_unusable_numbers(group, calibration=calibration, prefix=f"{side}."))
    for name, node in (expectation.by_segment or {}).items():
        if node is None:
            continue
        where = f"segment {name} "
        problems.extend(_unusable_numbers(node, calibration=calibration, prefix=where))
        for side, group in (node.by_direction or {}).items():
            problems.extend(
                _unusable_numbers(group, calibration=calibration, prefix=f"{where}{side}.")
            )
    return problems


def _invalid_expectation(
    name: str, problems: Sequence[str], metrics: dict[str, Any]
) -> CheckResult:
    return _insufficient(
        name,
        "invalid_expectation",
        [
            f"the backtest expectation in champion.json cannot be compared against: "
            f"{'; '.join(problems)}; re-publish the champion with a valid BacktestExpectation"
        ],
        metrics,
    )


def _bad_statistics(**statistics: float | None) -> list[str]:
    """Names of the statistics that are NaN or infinite."""
    return sorted(k for k, v in statistics.items() if v is not None and not math.isfinite(v))


def _invalid_statistic_reason(bad: Sequence[str]) -> str:
    return (
        f"non-finite statistic(s) {', '.join(bad)}: the live data or the expectation cannot "
        "be compared, so this check is unavailable rather than healthy; inspect the trade log"
    )


def _invalid_statistic(
    name: str, metrics: dict[str, Any], **statistics: float | None
) -> CheckResult | None:
    """An `insufficient` result if any statistic is NaN or infinite, else None."""
    bad = _bad_statistics(**statistics)
    if not bad:
        return None
    return _insufficient(name, "invalid_statistic", [_invalid_statistic_reason(bad)], metrics)


def _too_few_samples(name: str, n: int, min_trades: int, metrics: dict[str, Any]) -> CheckResult:
    """`insufficient` for want of evidence, saying so plainly when none is in segment."""
    settled = metrics["n_total"] - metrics["n_open"]
    if settled and metrics["n_out_of_segment"] == settled:
        # The usual cause of an empty check on a live book: the trades are all in
        # segments the gate did not pass. Say that, not just a tally of exclusions.
        head = (
            f"0 of {settled} settled trades are in the champion's allowed segments "
            f"{list(metrics['allowed_segments'])}"
        )
    else:
        head = f"{n} scoreable settled trades"
    return _insufficient(
        name,
        "too_few_samples",
        [f"{head}, need >= {min_trades} ({_excluded_summary(metrics)})"],
        metrics,
    )


def _with_settled_at_note(result: CheckResult, counts: Mapping[str, int]) -> CheckResult:
    """Say, on the result, that some evidence could only be dated by its target_date."""
    missing = counts["n_no_settled_at"]
    if not missing:
        return result
    note = (
        f"note: {missing} of {counts['n_in_scope']} in-scope trades have no settled_at (settled "
        "by an older release), so they are judged by target_date and may include truth that "
        "arrived after as_of"
    )
    return replace(result, reasons=(*result.reasons, note))


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
    subject = f"{len(report.features)} feature(s)"
    if report.prediction is not None:
        subject += " and the prediction"
    reason = f"{subject} within thresholds (max psi {psi_text}, {report.n_current} live rows)"
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


def _is_invalid_forecast(trade: Mapping[str, Any]) -> bool:
    """A `model_prob` that is a number but not a probability. Absent is a different case."""
    prob = trade.get("model_prob")
    return _is_number(prob) and not (math.isfinite(prob) and 0.0 <= prob <= 1.0)


def _with_unfixed_no_note(result: CheckResult) -> CheckResult:
    """Say, on the PnL result, that NO trades with a wrong PnL were left out (and how to fix it)."""
    unfixed = result.metrics.get("n_excluded_unfixed_no", 0)
    if not unfixed:
        return result
    note = (
        f"note: {unfixed} NO trades were settled by a release that booked them as a short YES, "
        "so their realized_pnl is wrong and they were left out; run `pmlab resettle --trades "
        "PATH` to recompute them"
    )
    return replace(result, reasons=(*result.reasons, note))


def _with_bad_forecast_note(result: CheckResult) -> CheckResult:
    """Say, on the calibration result, that some forecasts were not scored as data errors."""
    bad = result.metrics.get("n_invalid_forecast", 0)
    if not bad:
        return result
    note = (
        f"note: {bad} settled trades carry a model_prob that is not a probability (non-finite "
        "or outside [0, 1]) and were not scored; check what writes the trade log"
    )
    return replace(result, reasons=(*result.reasons, note))


# --------------------------------------------------------------------------
# Sides and scopes
# --------------------------------------------------------------------------
#
# A check judges the whole book (the "overall" scope) and, when both sides trade, each
# side on its own. A scope is one comparison of live numbers with the backtest's: its
# status, the sentence that says why, and the numbers behind it. The check's verdict is
# the worst of the *conclusive* scopes; a scope that cannot be measured (too few trades
# on a thin side) is reported as not assessed and does not make the check insufficient
# when the overall comparison is conclusive.

_SIDES = ("yes", "no")
_SIDE_LABEL = {"yes": "YES side", "no": "NO side"}

# (mean, std, n) of one group of backtest trades.
_Group = tuple[float, float, int]


@dataclass(frozen=True)
class _Scope:
    """One comparison of live numbers with the backtest's, and its verdict.

    `metrics` holds the numbers only; `status`, `cause` and the trade count are on the
    scope itself and are added when it is reported.
    """

    key: str
    label: str  # "" for the whole book
    n: int
    status: str
    cause: str | None
    reason: str
    metrics: dict[str, Any]
    kind: str = "overall"  # "overall", "side" or "segment"

    def entry(self) -> dict[str, Any]:
        """This scope as it appears under the check's `directions` metric."""
        return {"status": self.status, "cause": self.cause, "n_trades": self.n, **self.metrics}


@dataclass(frozen=True)
class _Row:
    """One scoreable settled trade's per-stake return, and where it sits in settlement order."""

    side: str
    segment: str
    value: float
    order: tuple[Any, ...]


def _settlement_order(trade: Mapping[str, Any]) -> tuple[Any, ...]:
    """A total order on trades: when the truth arrived, then which trade it was.

    The time the truth arrived (`settled_at`; a trade from an older release has none and
    is placed by its `target_date`), then the trade's identity: segment, target date,
    horizon, recording time, label and side. `PaperBroker` records one trade per segment,
    date and horizon, so in a real log identity decides every tie. Records that still tie
    are indistinguishable in time and identity; the sequential test takes them as one
    step rather than order them by anything that could follow their outcome.
    """
    settled = parse_timestamp(trade.get("settled_at")) or parse_timestamp(trade.get("target_date"))
    return (
        settled or datetime.min.replace(tzinfo=UTC),
        _segment_of(trade),
        str(trade.get("target_date", "")),
        str(trade.get("horizon", "")),
        str(trade.get("recorded_at", "")),
        str(trade.get("outcome_label", "")),
        _direction_of(trade),
    )


@dataclass(frozen=True)
class _Welch:
    live: float
    live_var: float
    ref_mean: float
    ref_variance: float
    evidence: _Evidence

    def problems(self, **extra: float | None) -> list[str]:
        """Names of the non-finite statistics, the live ones first (they explain the rest)."""
        bad = _bad_statistics(live_mean=self.live, live_variance=self.live_var)
        return bad or _bad_statistics(gap=self.evidence.gap, z=self.evidence.z, **extra)


def _direction_of(trade: Mapping[str, Any]) -> str:
    """The side a trade bought, read as settlement reads it: anything but "yes" is NO."""
    return "yes" if trade.get("direction", "yes") == "yes" else "no"


def _no_settlement_is_trusted(trade: Mapping[str, Any]) -> bool:
    """Was this NO trade booked by the arithmetic that settles it correctly?

    Before 0.9.1 a NO trade was settled as a short YES at the NO price, so its
    `realized_pnl` is wrong in both directions. The settlement version says which
    arithmetic booked it; a trade without one is not evidence about the NO strategy
    until `pmlab resettle` has recomputed it.
    """
    version = trade.get("settlement_version")
    return _is_number(version) and version >= MIN_TRUSTED_NO_SETTLEMENT_VERSION


def _pnl_groups(expectation: BacktestExpectation | SegmentExpectation) -> dict[str, _Group]:
    """The backtest's (mean, std, n) of per-stake PnL for each side it can speak for."""
    split = expectation.by_direction
    if split is None:
        # No split recorded: the pooled numbers describe the historical backtest, which
        # only ever bought YES.
        return {
            "yes": (
                expectation.mean_pnl_per_stake,
                expectation.std_pnl_per_stake,
                expectation.n_trades,
            )
        }
    return {
        side: (g.mean_pnl_per_stake, g.std_pnl_per_stake, g.n_trades) for side, g in split.items()
    }


def _brier_groups(
    expectation: BacktestExpectation | SegmentExpectation,
) -> dict[str, _Group] | None:
    """The backtest's (Brier, std, n) per side; None when no split was recorded.

    None means the pooled Brier stands for every trade, as it always did.
    """
    split = expectation.by_direction
    if split is None:
        return None
    groups: dict[str, _Group] = {}
    for side, g in split.items():
        if g.brier_score is not None and g.brier_std is not None:
            groups[side] = (g.brier_score, g.brier_std, g.n_trades)
    return groups


def _welch(
    values: Sequence[float],
    *,
    counts: Mapping[str, int],
    groups: Mapping[str, _Group] | None,
    pooled: _Group | None,
    worse_when_higher: bool,
) -> _Welch:
    """Hold a live sample against the backtest.

    With per-side `groups` the reference is the backtest's side means weighted by the
    live `counts` (`mixture_reference`); with none, the `pooled` group stands for every
    trade.
    """
    arr = np.array(values, dtype=float)
    with np.errstate(all="ignore"):  # an overflow is reported by the caller, not printed
        live = float(arr.mean())
        live_var = float(arr.var(ddof=1))
    if groups is not None:
        ref_mean, ref_variance = mixture_reference(counts, groups)
    else:
        assert pooled is not None  # callers pass one or the other
        ref_mean, ref_std, ref_n = pooled
        ref_variance = ref_std**2 / ref_n
    evidence = _one_sided_welch(
        live_mean=live,
        live_var=live_var,
        n_live=len(arr),
        ref_mean=ref_mean,
        ref_variance=ref_variance,
        worse_when_higher=worse_when_higher,
    )
    return _Welch(live, live_var, ref_mean, ref_variance, evidence)


def _holm_evidence(evidences: Mapping[str, _Evidence]) -> dict[str, _Evidence]:
    """Each scope's evidence after Holm-Bonferroni adjustment across the family.

    Judging several sibling scopes (the two sides) on their own multiplies the chance
    that one of them alarms by luck. Holm's adjusted p-value is what each scope would
    have to clear, and it is turned back into a z so the same thresholds apply: with
    one scope in the family nothing changes.
    """
    keys = list(evidences)
    raw: list[float] = []
    for key in keys:
        evidence = evidences[key]
        if evidence.p_value is not None:
            raw.append(evidence.p_value)
        else:
            # No spread at all: the gap is certain, in one direction or the other.
            raw.append(0.0 if evidence.gap > 0 else 1.0)
    result: dict[str, _Evidence] = {}
    for key, p in zip(keys, holm_adjust(raw), strict=True):
        z = None if p <= 0.0 else normal_isf(p)
        result[key] = _Evidence(gap=evidences[key].gap, z=z, p_value=p)
    return result


def _pnl_level(shortfall: float, evidence: _Evidence, cfg: PnLThresholds) -> tuple[str, str]:
    """(status, the sentence about the threshold) for a PnL shortfall."""
    if shortfall >= cfg.shortfall_alert - _TOLERANCE and evidence.clears(cfg.z_alert):
        return (
            "alert",
            f"; alert needs shortfall >= {cfg.shortfall_alert:.0%}, z >= {cfg.z_alert:g}",
        )
    if shortfall >= cfg.shortfall_warn - _TOLERANCE and evidence.clears(cfg.z_warn):
        return "warn", f"; warn needs shortfall >= {cfg.shortfall_warn:.0%}, z >= {cfg.z_warn:g}"
    return "ok", "; within thresholds"


def _brier_level(
    gap: float, rel: float, evidence: _Evidence, cfg: CalibrationThresholds
) -> tuple[str, str]:
    """(status, the sentence about the threshold) for a Brier gap."""
    if (
        gap >= cfg.brier_alert_abs - _TOLERANCE
        and rel >= cfg.brier_alert_rel - _TOLERANCE
        and evidence.clears(cfg.z_alert)
    ):
        return "alert", (
            f"; alert needs >= +{cfg.brier_alert_abs:g} abs, >= +{cfg.brier_alert_rel:.0%} rel, "
            f"z >= {cfg.z_alert:g}"
        )
    if (
        gap >= cfg.brier_warn_abs - _TOLERANCE
        and rel >= cfg.brier_warn_rel - _TOLERANCE
        and evidence.clears(cfg.z_warn)
    ):
        return "warn", (
            f"; warn needs >= +{cfg.brier_warn_abs:g} abs, >= +{cfg.brier_warn_rel:.0%} rel, "
            f"z >= {cfg.z_warn:g}"
        )
    return "ok", "; within thresholds"


def _z_text(raw: _Evidence, used: _Evidence, family: int) -> str:
    """`z=3.4` or, inside a family of scopes, `z=3.4, Holm z=3.1`."""
    text = f"z={_fmt_z(raw.z)}"
    if family > 1 and _fmt_z(used.z) != _fmt_z(raw.z):
        text += f", Holm z={_fmt_z(used.z)}"
    return text


def _unmeasurable(
    key: str, label: str, n: int, cause: str, reason: str, kind: str = "overall"
) -> _Scope:
    return _Scope(key, label, n, "insufficient", cause, reason, {}, kind)


def _too_few(key: str, label: str, n: int, min_trades: int, kind: str) -> _Scope:
    return _unmeasurable(
        key,
        label,
        n,
        "too_few_samples",
        f"{n} scoreable settled trades, need >= {min_trades}",
        kind,
    )


def _too_few_for_a_side(key: str, n: int, min_trades: int) -> _Scope:
    return _too_few(key, _SIDE_LABEL[key], n, min_trades, "side")


def _segments_recorded(expectation: BacktestExpectation) -> bool:
    """Did the backtest record numbers for individual segments (not just the aggregate)?"""
    return expectation.by_segment is not None and set(expectation.by_segment) != {ALL_SEGMENTS}


def _segments_are_judged(expectation: BacktestExpectation, allowed: Sequence[str]) -> bool:
    """Is there a per-segment record to judge by, and a reason to use it?

    One allowed segment *is* the whole book: judging it again against the same trades
    would only repeat the overall line.
    """
    if not _segments_recorded(expectation):
        return False
    return not (len(allowed) == 1 and allowed[0] != ALL_SEGMENTS)


_KIND_RANK = {"segment": 0, "side": 1, "overall": 2, "sequential": 3}
_NOT_ASSESSED_SHOWN = 5


def _assemble(
    name: str,
    overall: _Scope,
    families: Mapping[str, Sequence[_Scope]],
    metrics: dict[str, Any],
    extras: Sequence[_Scope] = (),
) -> CheckResult:
    """The check's verdict: the worst of the conclusive scopes.

    `insufficient` is reserved for a check in which nothing could be measured. A scope
    with too few trades (or nothing promised for it) is listed as not assessed and does
    not turn an otherwise conclusive check insufficient: with many thin segments that
    would make the exit code permanently 30, and the overall comparison already covers
    their trades.

    `families` maps the metrics key a family of sibling scopes is reported under
    (``directions``, ``segments``) to its scopes. `extras` are further scopes the caller
    reports in its own place in the metrics (the sequential test).
    """
    scopes = [overall]
    for key, family in families.items():
        if family:
            metrics[key] = {sc.key: sc.entry() for sc in family}
            scopes.extend(family)
    scopes.extend(extras)
    conclusive = [sc for sc in scopes if sc.status != "insufficient"]
    if not conclusive:
        return _insufficient(name, overall.cause or "too_few_samples", [overall.reason], metrics)
    # Worst first. At equal severity the more specific scope leads (a segment, then a
    # side, then the book): it names what is wrong, and the overall line only repeats
    # that the book as a whole is.
    problems = sorted(
        (sc for sc in conclusive if sc.status != "ok"),
        key=lambda sc: (-_SEVERITY[sc.status], _KIND_RANK[sc.kind]),
    )
    status = problems[0].status if problems else "ok"
    # With sibling scopes in the report the whole-book line says so, so that it is not
    # mistaken for the sentence about a side or a segment.
    tag = "overall: " if any(families.values()) else ""
    reasons = [(tag + sc.reason) if sc is overall else sc.reason for sc in problems]
    if overall not in problems:
        # The whole story when nothing is wrong, context for the lines above otherwise.
        reasons.append(tag + overall.reason)
    # What the sequential test saw when it saw nothing wrong: shown with --verbose.
    reasons.extend(sc.reason for sc in extras if sc.status == "ok")
    unmeasured = [sc for sc in scopes if sc.status == "insufficient"]
    if unmeasured:
        shown = [f"{sc.label or 'overall'} ({sc.reason})" for sc in unmeasured]
        extra = len(shown) - _NOT_ASSESSED_SHOWN
        listed = ", ".join(shown[:_NOT_ASSESSED_SHOWN])
        reasons.append(f"not assessed: {listed}" + (f", and {extra} more" if extra > 0 else ""))
    return CheckResult(name, cast(HealthStatus, status), tuple(reasons), metrics)


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def _brier_scope(
    key: str,
    label: str,
    n: int,
    welch: _Welch,
    used: _Evidence | None,
    cfg: CalibrationThresholds,
    *,
    family: int,
    kind: str = "overall",
) -> _Scope:
    """The verdict on one measured calibration scope.

    `used` is the evidence to judge by (Holm-adjusted inside a family); None judges the
    scope on its own evidence.
    """
    bad = welch.problems()
    if bad:
        return _unmeasurable(
            key, label, n, "invalid_statistic", _invalid_statistic_reason(bad), kind
        )
    evidence = used or welch.evidence
    gap = welch.evidence.gap
    rel = gap / welch.ref_mean if welch.ref_mean > 0 else math.inf
    status, tail = _brier_level(gap, rel, evidence, cfg)
    prefix = f"{label}: " if label else ""
    summary = (
        f"{prefix}Brier {welch.live:.3f} vs backtest {welch.ref_mean:.3f} "
        f"({gap:+.3f} abs, {rel:+.1%} rel, {_z_text(welch.evidence, evidence, family)}, n={n})"
    )
    metrics: dict[str, Any] = {
        "brier_live": welch.live,
        "brier_backtest": welch.ref_mean,
        "degradation_abs": gap,
        "degradation_rel": rel,
        "z": welch.evidence.z,
        "p_value": welch.evidence.p_value,
    }
    if used is not None:
        metrics["p_adjusted"] = used.p_value
        metrics["z_adjusted"] = used.z
    return _Scope(key, label, n, status, None, f"{summary}{tail}", metrics, kind)


def _brier_family(
    by_side: Mapping[str, Sequence[float]],
    groups: Mapping[str, _Group],
    cfg: CalibrationThresholds,
) -> list[_Scope]:
    """Each side's calibration on its own, Holm-adjusted across the sides assessed."""
    measured: dict[str, _Welch] = {}
    scopes: dict[str, _Scope] = {}
    for side in _SIDES:
        terms = by_side.get(side)
        if terms is None:
            continue
        n = len(terms)
        if n < cfg.min_trades:
            scopes[side] = _too_few_for_a_side(side, n, cfg.min_trades)
            continue
        welch = _welch(
            terms,
            counts={side: n},
            groups={side: groups[side]},
            pooled=None,
            worse_when_higher=True,
        )
        bad = welch.problems()
        if bad:
            reason = _invalid_statistic_reason(bad)
            scopes[side] = _unmeasurable(
                side, _SIDE_LABEL[side], n, "invalid_statistic", reason, "side"
            )
            continue
        measured[side] = welch
    adjusted = _holm_evidence({side: welch.evidence for side, welch in measured.items()})
    for side, welch in measured.items():
        scopes[side] = _brier_scope(
            side,
            _SIDE_LABEL[side],
            len(by_side[side]),
            welch,
            adjusted[side],
            cfg,
            family=len(measured),
            kind="side",
        )
    return [scopes[side] for side in _SIDES if side in scopes]


def _segment_of(trade: Mapping[str, Any]) -> str:
    return str(trade.get("city_or_segment", ""))


def _brier_segment_family(
    trades: Sequence[Mapping[str, Any]],
    expectation: BacktestExpectation,
    cfg: CalibrationThresholds,
) -> list[_Scope]:
    """Each live segment's calibration against that segment's own backtest.

    The segments are a family of tests: Holm-Bonferroni across the ones assessed keeps
    the chance of a false alarm somewhere in the family at what one test would have.
    """
    assert expectation.by_segment is not None
    by_segment: dict[str, list[Mapping[str, Any]]] = {}
    for trade in trades:
        by_segment.setdefault(_segment_of(trade), []).append(trade)
    measured: dict[str, _Welch] = {}
    sizes: dict[str, int] = {}
    scopes: dict[str, _Scope] = {}
    for name in sorted(by_segment):
        label = f"segment {name}"
        node = expectation.by_segment.get(name)
        if node is None or node.brier_score is None or node.brier_std is None:
            scopes[name] = _unmeasurable(
                name,
                label,
                len(by_segment[name]),
                "no_expectation",
                "no backtest Brier expectation recorded for this segment",
                "segment",
            )
            continue
        groups = _brier_groups(node)
        items = [dict(t) for t in by_segment[name] if groups is None or _direction_of(t) in groups]
        tracker = CalibrationTracker(items, n_bins=cfg.n_bins)
        n = tracker.n_realized
        if n < cfg.min_trades:
            scopes[name] = _too_few(name, label, n, cfg.min_trades, "segment")
            continue
        counts = {
            side: CalibrationTracker(
                [t for t in items if _direction_of(t) == side], n_bins=cfg.n_bins
            ).n_realized
            for side in _SIDES
        }
        welch = _welch(
            tracker.squared_errors(),
            counts=counts,
            groups=groups,
            pooled=(node.brier_score, node.brier_std, node.n_trades),
            worse_when_higher=True,
        )
        bad = welch.problems()
        if bad:
            scopes[name] = _unmeasurable(
                name, label, n, "invalid_statistic", _invalid_statistic_reason(bad), "segment"
            )
            continue
        measured[name] = welch
        sizes[name] = n
    adjusted = _holm_evidence({name: welch.evidence for name, welch in measured.items()})
    for name, welch in measured.items():
        scopes[name] = _brier_scope(
            name,
            f"segment {name}",
            sizes[name],
            welch,
            adjusted[name],
            cfg,
            family=len(measured),
            kind="segment",
        )
    return [scopes[name] for name in sorted(scopes)]


def _calibration_check(
    selection: _Selection, expectation: BacktestExpectation | None, cfg: CalibrationThresholds
) -> CheckResult:
    # A model_prob that is NaN, infinite or outside [0, 1] is not a forecast that
    # turned out wrong: the recording step is broken. Scoring it would page as a Brier
    # alert, or (NaN) poison the statistics. Count it as bad data and leave it out.
    scoreable = [t for t in selection.trades if not _is_invalid_forecast(t)]
    n_invalid = len(selection.trades) - len(scoreable)
    groups = _brier_groups(expectation) if expectation is not None else None
    n_no_side = n_yes_side = 0
    if groups is not None:
        # A champion that recorded a per-side split can only speak for the sides it
        # recorded. (One without a split pools every trade against its single Brier.)
        covered: list[Mapping[str, Any]] = []
        for trade in scoreable:
            side = _direction_of(trade)
            if side in groups:
                covered.append(trade)
            elif side == "no":
                n_no_side += 1
            else:
                n_yes_side += 1
        scoreable = covered
    tracker = CalibrationTracker([dict(t) for t in scoreable], n_bins=cfg.n_bins)
    side_terms: dict[str, list[float]] = {}
    for side in _SIDES:
        side_tracker = CalibrationTracker(
            [dict(t) for t in scoreable if _direction_of(t) == side], n_bins=cfg.n_bins
        )
        if side_tracker.n_realized:
            side_terms[side] = side_tracker.squared_errors()
    counts = {side: len(terms) for side, terms in side_terms.items()}

    metrics: dict[str, Any] = {
        **selection.counts,
        "n_invalid_forecast": n_invalid,
        "allowed_segments": list(selection.allowed_segments),
        "n_by_segment": selection.by_segment,
        "n_trades": tracker.n_realized,
        "n_by_direction": counts,
        "n_skipped_no_forecast": tracker.n_skipped,
        "min_trades": cfg.min_trades,
    }
    if groups is not None:
        metrics["n_excluded_no_side"] = n_no_side
        metrics["n_excluded_yes_side"] = n_yes_side
        metrics["expectation_directions"] = sorted(groups)
    if expectation is not None:
        metrics["segments_recorded"] = _segments_recorded(expectation)

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
    if groups is not None and not groups:
        return _insufficient(
            "calibration",
            "no_expectation",
            [
                "champion.json records no per-side Brier expectation (the backtest traded each "
                "side too rarely to promise anything): re-publish the champion on a backtest "
                "with more trades"
            ],
            metrics,
        )
    problems = _unusable_expectation(expectation, calibration=True)
    if problems:
        return _invalid_expectation("calibration", problems, metrics)
    if tracker.n_realized < cfg.min_trades:
        return _too_few_samples("calibration", tracker.n_realized, cfg.min_trades, metrics)

    pooled: _Group = (expectation.brier_score, expectation.brier_std, expectation.n_trades)
    terms = tracker.squared_errors()
    welch = _welch(terms, counts=counts, groups=groups, pooled=pooled, worse_when_higher=True)
    overall = _brier_scope("overall", "", len(terms), welch, None, cfg, family=1)
    if overall.cause == "invalid_statistic":
        # Nothing about this book can be measured.
        return _insufficient("calibration", "invalid_statistic", [overall.reason], metrics)
    overall_window = tracker.overall()
    assert overall_window is not None  # n_realized >= min_trades >= 2
    metrics.update(overall.metrics)
    metrics["reliability_live"] = overall_window.reliability

    sides: list[_Scope] = []
    if len(counts) > 1 and groups is not None:
        sides = _brier_family(side_terms, groups, cfg)
    segments: list[_Scope] = []
    if _segments_are_judged(expectation, selection.allowed_segments):
        segments = _brier_segment_family(scoreable, expectation, cfg)
    return _assemble("calibration", overall, {"directions": sides, "segments": segments}, metrics)


# --------------------------------------------------------------------------
# PnL
# --------------------------------------------------------------------------


def _pnl_is_on_the_recorded_fill(trade: Mapping[str, Any]) -> bool:
    """Does this trade's realized_pnl carry the price its fill_price says it paid?

    Settlement writes `settled_basis` when it settles. A trade without it was settled
    by 0.9.0 or earlier, which entered at the quote even when a `fill_price` was
    recorded; one marked "quote" while carrying a different fill is just as
    inconsistent. Either is exact only if the fill equals the quote (or none was
    recorded), because then both arithmetics enter at the same price. Recomputing the
    rest would replace a number the operator can see in their log with one derived
    here, so they are left out instead.

    The quote is the price of the token bought: the YES price, or for a NO trade
    ``1 - gamma_price``.
    """
    if trade.get("settled_basis") == "fill_price":
        return True
    fill = trade.get("fill_price")
    if fill is None:
        return True
    gamma = trade.get("gamma_price")
    if not (_is_number(fill) and _is_number(gamma)):
        return False
    quote = float(gamma) if _direction_of(trade) == "yes" else 1.0 - float(gamma)
    return abs(float(fill) - quote) <= 1e-8


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


def _describe_promise(groups: Mapping[str, _Group]) -> str:
    """`+0.100 over 2000 trades`, or per side when the backtest recorded several."""
    if len(groups) == 1:
        ((mean, _, n),) = groups.values()
        return f"{mean:+.3f} over {n} trades"
    return ", ".join(
        f"{_SIDE_LABEL[s]} {m:+.3f} over {n} trades" for s, (m, _, n) in groups.items()
    )


def _minimum_detectable_shortfall(
    welch: _Welch, n: int, cfg: PnLThresholds, family: int
) -> tuple[float, float]:
    """The smallest shortfall this comparison could have shown, with `mde_power` power.

    ``(z_alpha + z_power) * se / backtest_mean``, with the standard error the test itself
    used. `z_alpha` is the z a warn must clear (`z_warn`); inside a family of `family`
    sibling tests the level each must meet is divided by the family size, so a side or a
    segment pays for being one of several. The second figure is what the check can
    actually flag: never below `shortfall_warn`, because a smaller shortfall is not a
    finding however clear it is.
    """
    z_alpha = cfg.z_warn if family <= 1 else normal_isf(normal_sf(cfg.z_warn) / family)
    se = math.sqrt(welch.live_var / n + welch.ref_variance)
    gap = detectable_gap(se, z_alpha, cfg.mde_power)
    shortfall = gap / welch.ref_mean
    return shortfall, max(shortfall, cfg.shortfall_warn)


def _pnl_scope(
    key: str,
    label: str,
    n: int,
    welch: _Welch,
    used: _Evidence | None,
    cfg: PnLThresholds,
    *,
    family: int,
    kind: str = "overall",
) -> _Scope:
    """The verdict on one measured PnL scope. `used` is the (adjusted) evidence to judge by."""
    shortfall = welch.evidence.gap / welch.ref_mean
    bad = welch.problems(shortfall=shortfall)
    if bad:
        return _unmeasurable(
            key, label, n, "invalid_statistic", _invalid_statistic_reason(bad), kind
        )
    evidence = used or welch.evidence
    status, tail = _pnl_level(shortfall, evidence, cfg)
    mde, mde_effective = _minimum_detectable_shortfall(welch, n, cfg, family)
    prefix = f"{label}: " if label else ""
    summary = (
        f"{prefix}mean PnL/stake {welch.live:+.3f} vs backtest {welch.ref_mean:+.3f} "
        f"(shortfall {shortfall:.0%}, {_z_text(welch.evidence, evidence, family)}, n={n})"
    )
    metrics: dict[str, Any] = {
        "mean_live": welch.live,
        "std_live": math.sqrt(welch.live_var),
        "mean_backtest": welch.ref_mean,
        "shortfall_rel": shortfall,
        "z": welch.evidence.z,
        "p_value": welch.evidence.p_value,
        "mde_shortfall": mde,
        "mde_shortfall_effective": mde_effective,
    }
    tail += f" (could only detect a shortfall >= {mde_effective:.0%} of the backtest edge at n={n})"
    if used is not None:
        metrics["p_adjusted"] = used.p_value
        metrics["z_adjusted"] = used.z
    return _Scope(key, label, n, status, None, f"{summary}{tail}", metrics, kind)


def _pnl_family(
    by_side: Mapping[str, Sequence[float]], groups: Mapping[str, _Group], cfg: PnLThresholds
) -> list[_Scope]:
    """Each side's PnL on its own, Holm-adjusted across the sides assessed."""
    measured: dict[str, _Welch] = {}
    scopes: dict[str, _Scope] = {}
    for side in _SIDES:
        values = by_side.get(side)
        if values is None:
            continue
        n, label = len(values), _SIDE_LABEL[side]
        mean, _, n_backtest = groups[side]
        if n < cfg.min_trades:
            scopes[side] = _too_few_for_a_side(side, n, cfg.min_trades)
            continue
        if mean <= 0:
            scopes[side] = _unmeasurable(
                side,
                label,
                n,
                "invalid_expectation",
                f"backtest mean PnL per stake is {mean:+.3f} over {n_backtest} trades: a "
                "shortfall needs a positive expectation to be measured against",
                "side",
            )
            continue
        welch = _welch(
            values,
            counts={side: n},
            groups={side: groups[side]},
            pooled=None,
            worse_when_higher=False,
        )
        bad = welch.problems()
        if bad:
            scopes[side] = _unmeasurable(
                side, label, n, "invalid_statistic", _invalid_statistic_reason(bad), "side"
            )
            continue
        measured[side] = welch
    adjusted = _holm_evidence({side: welch.evidence for side, welch in measured.items()})
    for side, welch in measured.items():
        scopes[side] = _pnl_scope(
            side,
            _SIDE_LABEL[side],
            len(by_side[side]),
            welch,
            adjusted[side],
            cfg,
            family=len(measured),
            kind="side",
        )
    return [scopes[side] for side in _SIDES if side in scopes]


def _pnl_segment_family(
    rows: Sequence[_Row], expectation: BacktestExpectation, cfg: PnLThresholds
) -> list[_Scope]:
    """Each live segment's PnL against that segment's own backtest.

    A segment is held to its own per-side means, weighted by its own live mix of sides,
    exactly as the whole book is. The segments are a family of tests: Holm-Bonferroni
    across the ones assessed keeps the chance of a false alarm somewhere in the family
    at what one test would have, however many segments there are.
    """
    assert expectation.by_segment is not None
    by_segment: dict[str, list[tuple[str, float]]] = {}
    for row in rows:
        by_segment.setdefault(row.segment, []).append((row.side, row.value))
    measured: dict[str, _Welch] = {}
    sizes: dict[str, int] = {}
    scopes: dict[str, _Scope] = {}
    for name in sorted(by_segment):
        label = f"segment {name}"
        node = expectation.by_segment.get(name)
        if node is None:
            scopes[name] = _unmeasurable(
                name,
                label,
                len(by_segment[name]),
                "no_expectation",
                "no backtest expectation recorded for this segment",
                "segment",
            )
            continue
        groups = _pnl_groups(node)
        usable = [(side, value) for side, value in by_segment[name] if side in groups]
        n = len(usable)
        if n < cfg.min_trades:
            scopes[name] = _too_few(name, label, n, cfg.min_trades, "segment")
            continue
        counts = Counter(side for side, _ in usable)
        welch = _welch(
            [value for _, value in usable],
            counts=counts,
            groups=groups,
            pooled=None,
            worse_when_higher=False,
        )
        bad = welch.problems()
        if bad:
            scopes[name] = _unmeasurable(
                name, label, n, "invalid_statistic", _invalid_statistic_reason(bad), "segment"
            )
            continue
        if welch.ref_mean <= 0:
            scopes[name] = _unmeasurable(
                name,
                label,
                n,
                "invalid_expectation",
                f"backtest mean PnL per stake for this segment is {welch.ref_mean:+.3f}: a "
                "shortfall needs a positive expectation to be measured against",
                "segment",
            )
            continue
        measured[name] = welch
        sizes[name] = n
    adjusted = _holm_evidence({name: welch.evidence for name, welch in measured.items()})
    for name, welch in measured.items():
        scopes[name] = _pnl_scope(
            name,
            f"segment {name}",
            sizes[name],
            welch,
            adjusted[name],
            cfg,
            family=len(measured),
            kind="segment",
        )
    return [scopes[name] for name in sorted(scopes)]


def _sequential_scope(
    rows: Sequence[_Row], groups: Mapping[str, _Group], cfg: PnLThresholds
) -> tuple[_Scope, dict[str, Any]]:
    """The CUSUM over the trades in settlement order, and the metrics that describe it.

    Each trade is standardized by its own side's promise: its shortfall is
    ``(mean - return) / std`` of that side, and the allowance is `cusum_allowance` times
    the side's ``mean / std``, so a book that trades more NO than the backtest did is not
    mistaken for a decline. A side with no spread, no positive promise, or a promise made on
    fewer than `cusum_min_backtest_trades` trades cannot be standardized: its trades are
    skipped and counted. Trades that tie on settlement time
    and identity are one step (their shortfalls and allowances add), which keeps the
    result independent of the order the log happens to list them in.
    """
    steps: list[tuple[float, float, int]] = []  # (shortfall, allowance, trades) per step
    for _, tied in itertools.groupby(sorted(rows, key=lambda r: r.order), key=lambda r: r.order):
        shortfall = allowance = 0.0
        count = 0
        for row in tied:
            mean, std, n_backtest = groups[row.side]
            if mean > 0 and std > 0 and n_backtest >= cfg.cusum_min_backtest_trades:
                shortfall += (mean - row.value) / std
                allowance += cfg.cusum_allowance * mean / std
                count += 1
        if count:
            steps.append((shortfall, allowance, count))
    n = sum(count for _, _, count in steps)
    skipped = len(rows) - n
    common = {
        "threshold": cfg.cusum_threshold,
        "allowance_fraction": cfg.cusum_allowance,
        "n_skipped": skipped,
    }
    if not steps:
        reason = (
            "no trade could be standardized: the backtest spread is zero, the promised mean is "
            f"not positive, or it was made on fewer than {cfg.cusum_min_backtest_trades} trades"
        )
        metrics = {
            **common,
            "status": "not_applicable",
            "signalled": False,
            "n": 0,
            "reason": reason,
        }
        scope = _unmeasurable(
            "sequential", "sequential test", 0, "not_applicable", reason, "sequential"
        )
        return scope, metrics
    result = cusum([x for x, _, _ in steps], [k for _, k, _ in steps], cfg.cusum_threshold)
    crossed_at = (
        None
        if result.crossed_at is None
        else sum(count for _, _, count in steps[: result.crossed_at])
    )
    signalled = crossed_at is not None
    if signalled:
        status = "alert"
        reason = (
            f"sequential: CUSUM crossed {cfg.cusum_threshold:g} at trade {crossed_at} of "
            f"{n} (now {result.statistic:.1f}, peak {result.peak:.1f}); alert whenever it "
            f"crosses {cfg.cusum_threshold:g}"
        )
    else:
        status = "ok"
        reason = (
            f"sequential: CUSUM {result.statistic:.1f} of {cfg.cusum_threshold:g} "
            f"(peak {result.peak:.1f}, n={n}); within thresholds"
        )
    metrics = {
        **common,
        "status": status,
        "signalled": signalled,
        "statistic": result.statistic,
        "peak": result.peak,
        "crossed_at": crossed_at,
        "allowance": sum(k for _, k, _ in steps) / n,
        "n": n,
        "reason": reason,
    }
    scope = _Scope("sequential", "sequential test", n, status, None, reason, {}, "sequential")
    return scope, metrics


def _pnl_check(
    selection: _Selection,
    expectation: BacktestExpectation | None,
    costs: CostModel | None,
    cfg: PnLThresholds,
) -> CheckResult:
    groups = _pnl_groups(expectation) if expectation is not None else {"yes": (0.0, 0.0, 2)}
    rows: list[_Row] = []
    n_unscoreable = 0
    n_no_side = 0
    n_yes_side = 0
    n_unfixed_no = 0
    n_legacy = 0
    n_cost_excluded = 0
    n_cost_unverified = 0
    for trade in selection.trades:
        stake = trade.get("flat_stake", 1.0)
        pnl = trade.get("realized_pnl")
        if not _is_number(stake) or not _is_number(pnl) or stake <= 0 or not math.isfinite(pnl):
            n_unscoreable += 1
            continue
        side = _direction_of(trade)
        # A side the backtest recorded no expectation for is not a draw from any
        # distribution being tested against: leave it out, counted.
        if side not in groups:
            if side == "no":
                n_no_side += 1
            else:
                n_yes_side += 1
            continue
        # A NO trade settled before the NO arithmetic was fixed carries a wrong PnL.
        if side == "no" and not _no_settlement_is_trusted(trade):
            n_unfixed_no += 1
            continue
        if not _pnl_is_on_the_recorded_fill(trade):
            n_legacy += 1
            continue
        if costs is not None:
            verdict = _cost_basis(trade, costs, float(stake))
            if verdict == "mismatch":
                n_cost_excluded += 1
                continue
            if verdict == "unverified":
                n_cost_unverified += 1
        rows.append(
            _Row(
                side,
                _segment_of(trade),
                float(pnl) / float(stake),
                _settlement_order(trade),
            )
        )

    by_side: dict[str, list[float]] = {}
    for row in rows:
        by_side.setdefault(row.side, []).append(row.value)
    counts = {side: len(by_side[side]) for side in _SIDES if side in by_side}
    metrics: dict[str, Any] = {
        **selection.counts,
        "allowed_segments": list(selection.allowed_segments),
        "n_by_segment": selection.by_segment,
        "n_trades": len(rows),
        "n_by_direction": counts,
        "expectation_directions": (
            sorted(groups)
            if expectation is not None and expectation.by_direction is not None
            else None
        ),
        "n_unscoreable": n_unscoreable,
        "n_excluded_no_side": n_no_side,
        "n_excluded_yes_side": n_yes_side,
        "n_excluded_unfixed_no": n_unfixed_no,
        "n_excluded_legacy_settlement": n_legacy,
        "n_excluded_cost_basis": n_cost_excluded,
        "n_cost_unverified": n_cost_unverified if costs is not None else None,
        "cost_basis": (
            "unknown" if costs is None else ("mismatch" if n_cost_excluded else "match")
        ),
        "min_trades": cfg.min_trades,
    }
    if expectation is not None:
        metrics["segments_recorded"] = _segments_recorded(expectation)

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
    if not groups:
        return _insufficient(
            "pnl",
            "no_expectation",
            [
                "champion.json records no per-side PnL expectation (the backtest traded each "
                "side too rarely to promise anything): re-publish the champion on a backtest "
                "with more trades"
            ],
            metrics,
        )
    problems = _unusable_expectation(expectation, calibration=False)
    if problems:
        return _invalid_expectation("pnl", problems, metrics)
    if not any(mean > 0 for mean, _, _ in groups.values()):
        return _insufficient(
            "pnl",
            "invalid_expectation",
            [
                f"backtest mean PnL per stake is {_describe_promise(groups)}: a shortfall needs "
                "a positive expectation to be measured against, so this check will stay "
                "insufficient on every run until the champion is re-published on a backtest "
                "that earned money"
            ],
            metrics,
        )
    if len(rows) < cfg.min_trades:
        return _too_few_samples("pnl", len(rows), cfg.min_trades, metrics)

    values = [row.value for row in rows]
    welch = _welch(values, counts=counts, groups=groups, pooled=None, worse_when_higher=False)
    if welch.ref_mean <= 0:
        # Each side that traded has a promise, but this mix of them averages to nothing.
        return _insufficient(
            "pnl",
            "invalid_expectation",
            [
                f"backtest mean PnL per stake for this mix of sides is {welch.ref_mean:+.3f}: a "
                "shortfall needs a positive expectation to be measured against"
            ],
            metrics,
        )
    overall = _pnl_scope("overall", "", len(values), welch, None, cfg, family=1)
    if overall.cause == "invalid_statistic":
        return _insufficient("pnl", "invalid_statistic", [overall.reason], metrics)
    metrics.update(overall.metrics)

    sides = _pnl_family(by_side, groups, cfg) if len(counts) > 1 else []
    segments: list[_Scope] = []
    if _segments_are_judged(expectation, selection.allowed_segments):
        segments = _pnl_segment_family(rows, expectation, cfg)
    extras: list[_Scope] = []
    if cfg.sequential:
        sequential, metrics["sequential"] = _sequential_scope(rows, groups, cfg)
        extras.append(sequential)
    return _assemble("pnl", overall, {"directions": sides, "segments": segments}, metrics, extras)


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
        as_of: ISO-8601 cutoff. Nothing recorded after it, and no trade settled
            after it (`settled_at`; `target_date` for a trade from an older release
            that has none), is evidence. Must carry a UTC offset (``Z`` is fine);
            a naive timestamp is refused, not read as UTC.
        config: Thresholds and the `insufficient` policy; defaults if omitted.
        checks: Which of `CHECKS` to run. The rest are listed in `skipped`.

    Raises:
        WatchdogInputError: An unparseable or offset-less `as_of`, an unparseable
            publish time, a champion that allows no segment, an unknown or empty
            `checks`, or a malformed trade row.
    """
    config = config or WatchdogConfig()

    unknown = sorted(set(checks) - set(CHECKS))
    if unknown:
        raise WatchdogInputError(f"unknown check(s) {unknown}; valid checks: {list(CHECKS)}")
    if not checks:
        raise WatchdogInputError(f"run at least one of {list(CHECKS)}")
    selected_checks = [name for name in CHECKS if name in checks]
    skipped = tuple(name for name in CHECKS if name not in checks)

    as_of_ts = parse_as_of(as_of)
    published_ts = parse_timestamp(champion.published_at)
    if published_ts is None:
        raise WatchdogInputError(
            f"champion published_at {champion.published_at!r} is not an ISO-8601 timestamp"
        )

    allowed = champion.get_allowed_segments()
    if not allowed:
        # An empty allowlist means "none", as in PaperBroker, so this champion could
        # never have recorded a trade. publish() cannot produce it (a GO gate has a
        # passing segment): the file was edited or is not a real champion.
        raise WatchdogInputError(
            "champion.json allows no segment (no segment in its gate passes), so it "
            "could never have traded; a GO champion always has at least one"
        )
    selection = _select_trades(
        trades, allowed_segments=allowed, published_at=published_ts, as_of=as_of_ts
    )

    results: list[CheckResult] = []
    for name in selected_checks:
        if name == "drift":
            results.append(_drift_check(drift))
        elif name == "calibration":
            calibration = _calibration_check(selection, champion.expectation, config.calibration)
            calibration = _with_bad_forecast_note(calibration)
            results.append(_with_settled_at_note(calibration, selection.counts))
        else:
            pnl = _pnl_check(selection, champion.expectation, champion.costs, config.pnl)
            pnl = _with_unfixed_no_note(pnl)
            results.append(_with_settled_at_note(pnl, selection.counts))

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
