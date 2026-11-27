"""Fee estimation for Polymarket CLOB trades."""

from __future__ import annotations

import math

# The default rate of the flat-bps fee model: 30 basis points on the flat stake. It is the fee every
# release before 1.5.0 charged a Polymarket taker; since then `CostModel` charges the price-dependent
# fee of `pmlab.core.fee_schedule` unless it is given fee_model="flat-bps".
DEFAULT_TAKER_BPS: float = 30.0


def estimate_fee(flat_stake: float, taker_bps: float = DEFAULT_TAKER_BPS) -> float:
    """Estimate the flat-bps taker fee for a trade sized at *flat_stake* USDC.

    This is the flat model of 1.4.0 and before, ``taker_bps / 10_000`` of the notional stake at any
    price (``CostModel(fee_model="flat-bps")``). It is **not** what Polymarket charges a taker: that
    fee depends on the price and the market's schedule, and ``CostModel`` charges it by default
    since 1.5.0 (see ``CostModel.fee`` and ``pmlab.core.fee_schedule``).

    Args:
        flat_stake: Notional USDC being risked (e.g. 1.0 USDC).
        taker_bps: Taker fee in basis points (default 30 = 0.30%).

    Returns:
        Fee amount in USDC.
    """
    for name, value in (("flat_stake", flat_stake), ("taker_bps", taker_bps)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite number >= 0, got {value!r}")
    return flat_stake * taker_bps / 10_000.0
