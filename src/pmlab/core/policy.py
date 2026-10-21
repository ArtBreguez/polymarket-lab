"""The trade policy: which trade a strategy takes in a market.

Live trading takes, per market, the side and bin with the best after-fee edge, and only
if that edge clears `record-trades --min-edge` (`EdgeSignal.best_edge`). The backtest
used to do something else: buy YES on the bin the model rated most likely, always. A
champion promoted on that backtest was not the strategy that trades, and the live NO
trades it never simulated could not be held to any expectation.

`TradePolicy` names the decision rule so the backtest can run either one:

- ``top_prob`` is the historical backtest: YES on the bin with the highest predicted
  probability, every market, no threshold. It stays the default of
  `rolling_origin_eval` so numbers published before the policy existed reproduce.
- ``best_edge`` is the live rule: the (bin, side) with the best after-fee edge in each
  market, taken if that edge is at least `min_edge`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pmlab.core.edge import DEFAULT_MIN_EDGE

__all__ = ["TradePolicy"]

_SELECTIONS = ("top_prob", "best_edge")
_DIRECTIONS = ("yes", "no")


@dataclass(frozen=True)
class TradePolicy:
    """Which (bin, side) a strategy trades in a market, and when.

    Attributes:
        selection: ``"top_prob"`` (YES on the most probable bin) or ``"best_edge"``
            (the best after-fee edge over bins and the allowed sides).
        directions: The sides that may be bought, a subset of ``("yes", "no")``.
        min_edge: The smallest after-fee edge worth trading under ``best_edge``;
            ``None`` under ``top_prob``, which has no threshold.
    """

    selection: Literal["top_prob", "best_edge"] = "best_edge"
    directions: tuple[str, ...] = ("yes", "no")
    min_edge: float | None = DEFAULT_MIN_EDGE

    def __post_init__(self) -> None:
        if self.selection not in _SELECTIONS:
            raise ValueError(f"selection must be one of {_SELECTIONS}, got {self.selection!r}")
        directions = tuple(self.directions)
        object.__setattr__(self, "directions", directions)
        if not directions or len(set(directions)) != len(directions):
            raise ValueError(f"directions must be a non-empty set of sides, got {directions!r}")
        bad = [d for d in directions if d not in _DIRECTIONS]
        if bad:
            raise ValueError(f"unknown direction(s) {bad}; valid: {list(_DIRECTIONS)}")
        if self.selection == "top_prob":
            if directions != ("yes",) or self.min_edge is not None:
                raise ValueError(
                    "the top_prob policy buys YES on the most probable bin with no threshold; "
                    "use selection='best_edge' for NO trades or a min_edge"
                )
        else:
            if (
                isinstance(self.min_edge, bool)
                or not isinstance(self.min_edge, (int, float))
                or not math.isfinite(self.min_edge)
            ):
                raise ValueError(f"min_edge must be a finite number, got {self.min_edge!r}")
            # Always YES-first, so the order a caller wrote never changes the policy.
            object.__setattr__(self, "directions", tuple(d for d in _DIRECTIONS if d in directions))

    @classmethod
    def top_bin_yes(cls) -> TradePolicy:
        """The historical backtest: YES on the most probable bin of every market."""
        return cls(selection="top_prob", directions=("yes",), min_edge=None)

    @classmethod
    def live(
        cls,
        min_edge: float = DEFAULT_MIN_EDGE,
        directions: tuple[str, ...] = ("yes", "no"),
    ) -> TradePolicy:
        """What `record-trades` trades: the best-edge (bin, side), if it clears `min_edge`."""
        return cls(selection="best_edge", directions=directions, min_edge=min_edge)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selection": self.selection,
            "directions": list(self.directions),
            "min_edge": self.min_edge,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TradePolicy:
        unknown = sorted(set(data) - {"selection", "directions", "min_edge"})
        if unknown:
            raise ValueError(f"unknown trade policy field(s) {unknown}")
        return cls(
            selection=data["selection"],
            directions=tuple(data["directions"]),
            min_edge=data["min_edge"],
        )
