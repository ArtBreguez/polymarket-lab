"""Total fees of a trade log, and what share of the stake they are.

One reading of ``fee_paid`` and ``flat_stake`` for every place that shows the cost of fees: the HTML
report, the ``pmlab report`` and ``pmlab backtest`` summaries. Internal: nothing here is public API.
A trade counts when it recorded a ``fee_paid`` that is a finite number >= 0 and has a positive stake
(its own ``flat_stake``, or ``default_stake``); a log written before 1.5.0 may have none that does,
and then there is no summary rather than a zero, because "no fee recorded" is not "no fee paid".
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

__all__ = ["FeeSummary", "summarize_fees"]


@dataclass(frozen=True)
class FeeSummary:
    """Fees over the trades that recorded one.

    Attributes:
        total: Sum of ``fee_paid``, in USDC.
        stake: Sum of the stakes of those same trades, in USDC.
        n_trades: How many trades are in the two sums.
    """

    total: float
    stake: float
    n_trades: int

    @property
    def pct_of_stake(self) -> float:
        """The fees as a percentage of the stake (2.75 is 2.75%)."""
        return 100.0 * self.total / self.stake


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    return float(value) if math.isfinite(value) else None


def summarize_fees(
    trades: Iterable[Mapping[Any, Any]], *, default_stake: float | None = None
) -> FeeSummary | None:
    """Total fees and stake over the trades that recorded a fee, or ``None`` if none did.

    Args:
        trades: Trade records (mappings with ``fee_paid`` and ``flat_stake``).
        default_stake: The stake of a trade that has no ``flat_stake`` of its own (a backtest's
            trades do not repeat it).

    Returns:
        The summary, or ``None`` when no trade has a usable fee and stake.
    """
    total = stake = 0.0
    n = 0
    for trade in trades:
        fee = _number(trade.get("fee_paid"))
        own = _number(trade.get("flat_stake")) if "flat_stake" in trade else default_stake
        if fee is None or fee < 0 or own is None or own <= 0:
            continue
        total += fee
        stake += own
        n += 1
    return FeeSummary(total=total, stake=stake, n_trades=n) if n else None
