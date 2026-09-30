"""Backtest cost model — fees, slippage and a depth-aware fill penalty.

`rolling_origin_eval` historically filled at `market_price` and charged a flat
taker fee. That is the best case: it assumes the quote you saw is the price you
got, at any size. Real fills pay the spread and walk the book, and on
mutually-exclusive Polymarket fields the gap is structural rather than
incidental — reading a mid price makes coherent markets look arbitrageable.

`CostModel` makes the friction explicit and **opt-in**. The default instance
reproduces the previous behaviour exactly (30bps taker fee, fill at the quote),
so results published before this module stay reproducible; friction only appears
when a caller asks for it.

Cost arithmetic, in order of application:

    slip_bps_total = slippage_bps + depth_penalty_bps(stake)
    fill = price * (1 + slip_bps_total / 10_000) + slippage_fixed
    fill = min(fill, 1.0)            # a probability price cannot exceed certainty

    depth_penalty_bps(stake) = depth_bps_per_unit * max(0, stake - ref) / ref

The depth term is deliberately linear in *excess* stake rather than a real book
walk: pmlab does not carry L2 depth in the training panel, so a configurable
linear penalty is honest about being an approximation. When depth data is
available, prefer measuring it.

All costs move the fill **against** the buyer. There is no configuration in which
friction improves a trade.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from pmlab.core.fees import DEFAULT_TAKER_BPS

_BPS = 10_000.0


@dataclass(frozen=True)
class CostModel:
    """Execution costs applied to a backtest fill.

    Attributes:
        taker_bps: Taker fee in basis points of notional stake. Defaults to
            Polymarket's 30bps so the default model matches historical results.
        slippage_bps: Price-proportional slippage in basis points. Models the
            half-spread you cross on entry.
        slippage_fixed: Absolute slippage added to the fill price, in probability
            units (0.01 = one cent of probability).
        depth_bps_per_unit: Extra slippage in basis points for each multiple of
            `depth_reference_stake` above that reference. 0 disables the term.
        depth_reference_stake: Stake size assumed to fill without walking the
            book. Must be positive.
    """

    taker_bps: float = DEFAULT_TAKER_BPS
    slippage_bps: float = 0.0
    slippage_fixed: float = 0.0
    depth_bps_per_unit: float = 0.0
    depth_reference_stake: float = 1.0

    def __post_init__(self) -> None:
        for name in ("taker_bps", "slippage_bps", "slippage_fixed", "depth_bps_per_unit"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if self.depth_reference_stake <= 0:
            raise ValueError(
                f"depth_reference_stake must be > 0, got {self.depth_reference_stake}"
            )

    @classmethod
    def frictionless(cls) -> CostModel:
        """A model with no costs at all — useful as a backtest upper bound."""
        return cls(taker_bps=0.0)

    def fee(self, stake: float) -> float:
        """Taker fee charged on *stake* notional."""
        _require_positive_stake(stake)
        return stake * self.taker_bps / _BPS

    def depth_penalty_bps(self, stake: float) -> float:
        """Extra slippage in bps from walking the book at this size."""
        _require_positive_stake(stake)
        if self.depth_bps_per_unit == 0.0:
            return 0.0
        excess = max(0.0, stake - self.depth_reference_stake) / self.depth_reference_stake
        return self.depth_bps_per_unit * excess

    def fill_price(self, price: float, stake: float) -> float:
        """Price actually paid per share when buying *stake* notional at *price*.

        Always >= `price`, and never above 1.0: costs must not manufacture a
        position that pays more than certainty.
        """
        if not 0.0 <= price <= 1.0:
            raise ValueError(f"price must be within [0, 1], got {price}")
        _require_positive_stake(stake)

        total_bps = self.slippage_bps + self.depth_penalty_bps(stake)
        fill = price * (1.0 + total_bps / _BPS) + self.slippage_fixed
        return min(fill, 1.0)

    def to_dict(self) -> dict[str, float]:
        """Flat mapping for the experiment tracker; round-trips via ``CostModel(**d)``."""
        return asdict(self)


def _require_positive_stake(stake: float) -> None:
    if stake <= 0:
        raise ValueError(f"stake must be > 0, got {stake}")
