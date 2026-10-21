"""The arithmetic under the health checks: pure functions, no policy.

Thresholds and verdicts live in `pmlab.monitoring.watchdog`; this module only computes.
Each function here is checked in `tests/monitoring/test_stats.py` against something
that does not share its code (scipy, a hand-worked example, a brute-force version of
the textbook procedure, or a simulation).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist

__all__ = [
    "CusumResult",
    "cusum",
    "detectable_gap",
    "holm_adjust",
    "mixture_reference",
    "normal_isf",
    "normal_sf",
]

_STANDARD_NORMAL = NormalDist()


def normal_sf(z: float) -> float:
    """P(Z >= z) for a standard normal Z: the one-sided p-value of a z statistic."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def normal_isf(p: float) -> float:
    """The z whose one-sided p-value is `p`: the inverse of `normal_sf`.

    0 maps to +inf and 1 to -inf. The lower tail is inverted directly, because
    ``1 - p`` for a small p throws away the digits that matter.
    """
    if not isinstance(p, (int, float)) or not 0.0 <= p <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {p!r}")
    if p == 0.0:
        return math.inf
    if p == 1.0:
        return -math.inf
    if p <= 0.5:
        return -_STANDARD_NORMAL.inv_cdf(p)
    return _STANDARD_NORMAL.inv_cdf(1.0 - p)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the order given.

    Testing several hypotheses at once, each at its own level, raises the chance that
    some true null is rejected: ten true segments each tested at 5% give a 40% chance
    of at least one false alarm. Holm's step-down procedure keeps that chance at or below
    the nominal level for any dependence between the tests, and is uniformly more
    powerful than plain Bonferroni:

        sort ascending, p(1) <= ... <= p(m)
        adjusted(i) = max over j <= i of  min(1, (m - j + 1) * p(j))

    A hypothesis is rejected at family-wise level alpha exactly when its adjusted
    p-value is at most alpha. One test is returned unchanged.

    Raises:
        ValueError: A p-value that is not a number in [0, 1].
    """
    for p in p_values:
        if not isinstance(p, (int, float)) or isinstance(p, bool) or not 0.0 <= p <= 1.0:
            raise ValueError(f"p-value must be a number in [0, 1], got {p!r}")
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def mixture_reference[K](
    counts: Mapping[K, int], groups: Mapping[K, tuple[float, float, int]]
) -> tuple[float, float]:
    """What a live sample with this mix of groups is expected to average, and how sure.

    The live trades come in a mix (so many YES, so many NO). Each group has its own
    backtest mean, spread and size. Holding the live average against the backtest's
    *pooled* mean is only fair if the live mix equals the backtest's; trade more NO than
    the backtest did and a book that is exactly as promised, side by side, looks short.
    The fair benchmark is the backtest's per-group means weighted by the **live** mix:

        mean     = sum_g w_g * mean_g                  w_g = live share of group g
        variance = sum_g w_g**2 * std_g**2 / n_g       the sampling variance of that mean

    The groups' backtest estimates are independent, so their variances add with squared
    weights. A group with no live trades has no weight and needs no reference.

    Args:
        counts: Live trades per group (a group is any hashable key: a side, a segment's cell).
        groups: Per group, the backtest ``(mean, std, n)``.

    Returns:
        ``(mean, variance_of_the_mean)`` of the reference.

    Raises:
        ValueError: No live trades, a negative count, live trades in a group that has no
            reference, or a group whose reference rests on no backtest trades.
    """
    if any(c < 0 for c in counts.values()):
        raise ValueError(f"counts must not be negative, got {dict(counts)}")
    live = {g: c for g, c in counts.items() if c > 0}
    total = sum(live.values())
    if total == 0:
        raise ValueError("a mixture needs at least one live trade")
    missing = [group for group in live if group not in groups]
    if missing:
        raise ValueError(f"no reference for group(s) {missing}")
    unbacked = [group for group in live if groups[group][2] < 1]
    if unbacked:
        raise ValueError(f"group(s) {unbacked} are promised on fewer than one backtest trade")
    mean = 0.0
    variance = 0.0
    for group, count in live.items():
        weight = count / total
        group_mean, group_std, group_n = groups[group]
        mean += weight * group_mean
        variance += weight**2 * group_std**2 / group_n
    return mean, variance


def detectable_gap(se: float, z_alpha: float, power: float) -> float:
    """The smallest true gap a one-sided z test finds with probability `power`.

    A test that needs ``gap / se >= z_alpha`` to fire sees ``(true_gap + noise) / se``,
    with noise standard normal. It fires with probability `power` when the true gap is
    ``(z_alpha + z_power) * se``, where ``z_power`` is the `power` quantile. A quiet run
    only rules out gaps larger than this; a gap smaller than it is mostly missed.

    Args:
        se: The standard error of the gap, as the test computes it.
        z_alpha: The z the test must clear (its significance level, one-sided).
        power: The probability of firing at the returned gap, strictly between 0 and 1.

    Raises:
        ValueError: A negative or non-finite `se`, a non-finite `z_alpha`, or a `power`
            outside (0, 1).
    """
    if not math.isfinite(se) or se < 0:
        raise ValueError(f"se must be a finite number >= 0, got {se!r}")
    if not math.isfinite(z_alpha):
        raise ValueError(f"z_alpha must be finite, got {z_alpha!r}")
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must be strictly between 0 and 1, got {power!r}")
    return (z_alpha + normal_isf(1.0 - power)) * se


@dataclass(frozen=True)
class CusumResult:
    """What a one-sided CUSUM saw: where it ended, how high it got, when it first signalled."""

    statistic: float  # S after the last trade
    peak: float  # the highest S reached
    crossed_at: int | None  # 1-based index of the first trade with S >= threshold
    n: int


def cusum(
    shortfalls: Sequence[float], allowances: Sequence[float], threshold: float
) -> CusumResult:
    """One-sided CUSUM (Page): ``S_i = max(0, S_{i-1} + x_i - k_i)``, signalling at ``S_i >= h``.

    Each `x_i` is how far trade i fell short of what was promised, in units of its own
    spread (positive is worse than promised); each `k_i` is the allowance, the part of a
    shortfall that is not yet evidence. Good luck cannot bank credit, because S stops at
    zero, so a run of losses is judged on its own. The sum is over the trades in the
    order given, which is why the caller must put them in a deterministic order.

    A signal is remembered (`crossed_at`) even if S later falls back: the watchdog keeps
    no state between runs, so "has it ever signalled since the champion was published" is
    what a stateless run can answer. Re-publishing the champion starts a fresh run.

    Raises:
        ValueError: The sequences differ in length or hold a non-finite number, or the
            threshold is not a finite positive number.
    """
    if len(shortfalls) != len(allowances):
        raise ValueError(
            f"one allowance per trade: {len(shortfalls)} shortfalls, {len(allowances)} allowances"
        )
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError(f"threshold must be a finite number > 0, got {threshold!r}")
    statistic = 0.0
    peak = 0.0
    crossed_at: int | None = None
    for index, (shortfall, allowance) in enumerate(zip(shortfalls, allowances, strict=True), 1):
        if not (math.isfinite(shortfall) and math.isfinite(allowance)):
            raise ValueError(f"trade {index} has a non-finite shortfall or allowance")
        statistic = max(0.0, statistic + shortfall - allowance)
        peak = max(peak, statistic)
        if crossed_at is None and statistic >= threshold:
            crossed_at = index
    return CusumResult(statistic=statistic, peak=peak, crossed_at=crossed_at, n=len(shortfalls))
