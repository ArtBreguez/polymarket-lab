"""Fee estimation for Polymarket CLOB trades."""

from __future__ import annotations

# Polymarket CLOB taker fee: 30 basis points on the flat stake.
DEFAULT_TAKER_BPS: float = 30.0


def estimate_fee(flat_stake: float, taker_bps: float = DEFAULT_TAKER_BPS) -> float:
    """Estimate taker fee for a trade sized at *flat_stake* USDC.

    Polymarket charges taker_bps / 10_000 on the notional stake (not per share).

    Args:
        flat_stake: Notional USDC being risked (e.g. 1.0 USDC).
        taker_bps: Taker fee in basis points (default 30 = 0.30%).

    Returns:
        Fee amount in USDC.
    """
    return flat_stake * taker_bps / 10_000.0
