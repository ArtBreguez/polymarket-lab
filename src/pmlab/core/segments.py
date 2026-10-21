"""What a champion's allowed segments mean.

A champion promoted on a per-segment gate allows the segments that passed. One promoted on
an aggregate gate (`HoldoutGateResult.evaluate(trades)` with no ``required_segments``)
was graded on the whole trade log as a single segment called ``"all"``: it makes no
per-segment claim, so ``"all"`` means *every segment*. Signals carry real segment names (a
city, a race), so reading the allowlist as a plain set would match none of them and the
champion would never record a trade.

The broker, the champion and the watchdog all read the allowlist through `segment_allowed`.
"""

from __future__ import annotations

from collections.abc import Collection

__all__ = ["ALL_SEGMENTS", "segment_allowed"]

# The name the aggregate gate gives its single segment, and so the wildcard. A segment that
# is literally called "all" is indistinguishable from it; that name is reserved.
ALL_SEGMENTS = "all"


def segment_allowed(segment: str, allowed: Collection[str] | None) -> bool:
    """Is `segment` allowed to trade?

    ``None`` means no filtering. An empty collection means *none*: a champion that allows
    nothing could never have recorded a trade. ``"all"`` in the collection allows every
    segment.
    """
    if allowed is None:
        return True
    return ALL_SEGMENTS in allowed or segment in allowed
