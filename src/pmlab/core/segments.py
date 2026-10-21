"""What a champion's allowed segments mean.

A champion promoted on a per-segment gate allows the segments that passed. One promoted on
an aggregate gate (`HoldoutGateResult.evaluate(trades)` with no ``required_segments``)
was graded on the whole trade log as a single segment called ``"all"``: it makes no
per-segment claim, so ``"all"`` means *every segment*. Signals carry real segment names (a
city, a race), so reading the allowlist as a plain set would match none of them and the
champion would never record a trade.

``"all"`` is the wildcard only **on its own**, which is all an aggregate gate produces. A
per-segment gate that grades a segment literally called ``"all"`` beside others allows those
segments, ``"all"`` among them, and nothing else: reading it as the wildcard would widen the
champion to segments no gate ever graded. A per-segment gate whose only segment is called
``"all"`` is indistinguishable from an aggregate one, so that name stays reserved.

The broker, the champion and the watchdog all read the allowlist through `segment_allowed`.
"""

from __future__ import annotations

from collections.abc import Collection

__all__ = ["ALL_SEGMENTS", "allows_every_segment", "segment_allowed"]

# The name the aggregate gate gives its single segment, and so the wildcard when it stands
# alone.
ALL_SEGMENTS = "all"


def allows_every_segment(allowed: Collection[str]) -> bool:
    """Does this allowlist make no per-segment claim (``"all"`` and nothing else)?"""
    return bool(allowed) and all(name == ALL_SEGMENTS for name in allowed)


def segment_allowed(segment: str, allowed: Collection[str] | None) -> bool:
    """Is `segment` allowed to trade?

    ``None`` means no filtering. An empty collection means *none*: a champion that allows
    nothing could never have recorded a trade. ``"all"`` on its own allows every segment;
    beside other names it is just a segment called "all".
    """
    if allowed is None:
        return True
    return allows_every_segment(allowed) or segment in allowed
