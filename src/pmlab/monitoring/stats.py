"""The arithmetic under the health checks: pure functions, no policy.

Thresholds and verdicts live in `pmlab.monitoring.watchdog`; this module only computes.
Each function here is checked in `tests/monitoring/test_stats.py` against something
that does not share its code (scipy, a hand-worked example, a brute-force version of
the textbook procedure, or a simulation).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from statistics import NormalDist

__all__ = ["holm_adjust", "mixture_reference", "normal_isf", "normal_sf"]

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


def mixture_reference(
    counts: Mapping[str, int], groups: Mapping[str, tuple[float, float, int]]
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
        counts: Live trades per group.
        groups: Per group, the backtest ``(mean, std, n)``.

    Returns:
        ``(mean, variance_of_the_mean)`` of the reference.

    Raises:
        ValueError: No live trades, a negative count, or live trades in a group that has
            no reference.
    """
    if any(c < 0 for c in counts.values()):
        raise ValueError(f"counts must not be negative, got {dict(counts)}")
    live = {g: c for g, c in counts.items() if c > 0}
    total = sum(live.values())
    if total == 0:
        raise ValueError("a mixture needs at least one live trade")
    missing = sorted(set(live) - set(groups))
    if missing:
        raise ValueError(f"no reference for group(s) {missing}")
    mean = 0.0
    variance = 0.0
    for group, count in live.items():
        weight = count / total
        group_mean, group_std, group_n = groups[group]
        mean += weight * group_mean
        variance += weight**2 * group_std**2 / group_n
    return mean, variance
