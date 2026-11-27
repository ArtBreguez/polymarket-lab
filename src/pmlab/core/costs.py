"""Backtest cost model — fees, slippage and a depth-aware fill penalty.

`rolling_origin_eval` historically filled at `market_price` and charged a flat
taker fee. That is the best case: it assumes the quote you saw is the price you
got, at any size, and that the fee is the same at every price. Real fills pay
the spread and walk the book, a Polymarket taker pays a fee that depends on the
price, and on mutually-exclusive Polymarket fields the gap is structural rather
than incidental — reading a mid price makes coherent markets look arbitrageable.

`CostModel` makes the friction explicit. It has two fee models:

* ``fee_model="polymarket"`` (the default since 1.5.0) charges what a Polymarket taker pays: per
  fill ``shares x rate x price x (1 - price)``, rounded to 5 decimals, that is ``rate x (1 -
  price)`` of the stake: 3.5% of the stake at 0.30 and 2.5% at 0.50 for a 0.05-rate market. The
  rate comes from a `FeeSchedule` (``fee_schedule=``); without one it is the ``"other"``
  category's 0.05, the highest published rate, so the default errs on the side of costing more
  and the trades say so (``fee_source="category_default"``).
* ``fee_model="flat-bps"`` is the model of every release before 1.5.0: ``stake x taker_bps /
  10_000``, the same 0.30% at any price. It stays so a backtest, a champion or a tuner written
  before 1.5.0 reproduces with the model it ran with: a cost dict with no ``fee_model`` *is* this
  model (`CostModel.from_dict`).

Slippage and depth are the same under both. Cost arithmetic, in order of application:

    slip_bps_total = slippage_bps + depth_penalty_bps(stake)
    fill = price * (1 + slip_bps_total / 10_000) + slippage_fixed
    fill = min(fill, 1.0)            # a probability price cannot exceed certainty
    fee  = fee(stake, fill)          # the fee is charged at the price the fill happens at

    depth_penalty_bps(stake) = depth_bps_per_unit * max(0, stake - ref) / ref

The depth term is deliberately linear in *excess* stake rather than a real book
walk: pmlab does not carry L2 depth in the training panel, so a configurable
linear penalty is honest about being an approximation. When depth data is
available, prefer measuring it.

All costs move the fill **against** the buyer. There is no configuration in which
friction improves a trade.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from pmlab._deprecation import declare_deprecated, warn_deprecated
from pmlab.core.fee_schedule import FeeSchedule, category_fee_schedule, taker_fee_on_stake
from pmlab.core.fees import DEFAULT_TAKER_BPS

__all__ = ["CostModel"]

_BPS = 10_000.0


class _DeprecationArgs(TypedDict):
    """The arguments `declare_deprecated` and `warn_deprecated` are both given."""

    what: str
    since: str
    removed_in: str
    alternative: str
    reason: str


# The deprecations of the bare ``taker_bps`` (#72), declared here so they are registered at import
# (the docs table is checked against the registry) and warned with the very same arguments.
_ALTERNATIVE_TO_TAKER_BPS = '`costs=CostModel(fee_model="flat-bps", taker_bps=...)`'
_BARE_TAKER_BPS: _DeprecationArgs = {
    "what": "CostModel(taker_bps=...) without a fee_model",
    "since": "1.5.0",
    "removed_in": "2.0.0",
    "alternative": '`CostModel(fee_model="flat-bps", taker_bps=...)`, or the default polymarket '
    "fee model",
    "reason": "the fee depends on the price on Polymarket, and a bare taker_bps silently picks "
    "the model of 1.4.0 and before",
}
_TAKER_BPS_ARGUMENT_OWNERS = ("rolling_origin_eval", "TunedForecaster", "PaperBroker")

# A costs file with a `taker_bps` and no `fee_model` (what every release before 1.5.0 wrote). The
# warning is the CLI's (a line on stderr); the declaration lives here, where `import pmlab` reaches it.
LEGACY_COSTS_FILE_DEPRECATION: _DeprecationArgs = {
    "what": "a --costs file with a taker_bps and no fee_model",
    "since": "1.5.0",
    "removed_in": "2.0.0",
    "alternative": '"fee_model": "flat-bps" next to it, or the default polymarket model',
    "reason": "it names no fee model, so it is read as the flat-bps model of 1.4.0",
}


def _taker_bps_argument(owner: str) -> _DeprecationArgs:
    return {
        "what": f"the taker_bps argument of {owner}",
        "since": "1.5.0",
        "removed_in": "2.0.0",
        "alternative": _ALTERNATIVE_TO_TAKER_BPS,
        "reason": "the fee on Polymarket depends on the price, and a bare number cannot say so",
    }


declare_deprecated(**_BARE_TAKER_BPS)
declare_deprecated(**LEGACY_COSTS_FILE_DEPRECATION)
for _owner in _TAKER_BPS_ARGUMENT_OWNERS:
    declare_deprecated(**_taker_bps_argument(_owner))


def warn_taker_bps_argument(owner: str) -> None:
    """Warn that ``owner`` (``rolling_origin_eval``, ``TunedForecaster``, ``PaperBroker``) was given
    a ``taker_bps``, naming the caller of ``owner``. Internal: the one place the text lives."""
    warn_deprecated(**_taker_bps_argument(owner), stacklevel=4)


FeeModelName = Literal["polymarket", "flat-bps"]
_FEE_MODELS: tuple[str, ...] = ("polymarket", "flat-bps")

# The category whose published rate a CostModel charges when it is given no schedule: the highest
# rate of the table (0.05), so an unknown market is costed too high rather than too low.
_DEFAULT_FEE_CATEGORY = "other"

_SOURCE_DEFAULT = "category_default"
_SOURCE_GIVEN = "user"
_SOURCE_FLAT = "flat_bps"


class _DefaultTakerBps(float):
    """The 30.0 that ``taker_bps`` defaults to, told apart from a 30.0 somebody passed.

    Until 1.5.0 ``CostModel(taker_bps=x)`` was how a fee was set, and a ``taker_bps`` that is given
    still means the flat-bps model. That needs the constructor to know whether ``taker_bps`` was
    passed, and the signature is frozen (``taker_bps: float = 30.0``), so the default is this float:
    equal to 30.0, printed as 30.0, but recognisable. It never survives construction: the field is
    stored as a plain ``float``.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return repr(float(self))


_TAKER_BPS_NOT_GIVEN = _DefaultTakerBps(DEFAULT_TAKER_BPS)


def _is_finite_number(value: object) -> bool:
    """A real number that is not NaN or infinite (a bool or a string is not one here)."""
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class CostModel:
    """Execution costs applied to a backtest fill.

    ``CostModel()`` is the Polymarket fee model with the default schedule and no slippage. Name the
    model you mean with ``fee_model``; ``taker_bps`` alone still selects ``"flat-bps"``, with a
    deprecation warning (it will have to be named from 2.0.0).

    Attributes:
        taker_bps: Taker fee in basis points of notional stake, **read only by the flat-bps model**
            (``fee_model="flat-bps"``), whose default is Polymarket's 30bps, the model before 1.5.0.
            Giving it without a ``fee_model`` selects that model (and warns). The polymarket model
            ignores it; the attribute then holds the unused default, 30.0.
        slippage_bps: Price-proportional slippage in basis points. Models the
            half-spread you cross on entry.
        slippage_fixed: Absolute slippage added to the fill price, in probability
            units (0.01 = one cent of probability).
        depth_bps_per_unit: Extra slippage in basis points for each multiple of
            `depth_reference_stake` above that reference. 0 disables the term.
        depth_reference_stake: Stake size assumed to fill without walking the
            book. Must be positive.
        fee_model: ``"polymarket"`` or ``"flat-bps"`` (see the module docs). ``None``, the default,
            is resolved on construction: ``"flat-bps"`` when ``taker_bps`` is given, else
            ``"polymarket"``. The attribute is never ``None`` afterwards.
        fee_schedule: The `FeeSchedule` the polymarket model charges, or the mapping of its fields
            that ``to_dict`` writes. ``None`` is the ``"other"`` category default (rate 0.05).
            The flat-bps model has none, and refuses one.
        fee_source: Where the schedule came from, as `FeeSnapshot.source` says it
            (``"gamma"``, ``"clob"``, ``"category_default"``), ``"user"`` for a schedule that was
            given without a source, ``"flat_bps"`` for the flat model. Each trade carries it, so a
            number in a report can be traced to what priced it. ``None`` is resolved on
            construction.

    Raises:
        ValueError: If a numeric field is not a finite number >= 0 (``depth_reference_stake`` > 0),
            ``fee_model`` is not one of the two, a ``fee_schedule`` is not a `FeeSchedule` (or a
            mapping of its fields), a flat-bps model is given a schedule or a source of its own,
            or a schedule is given together with a ``taker_bps`` and no ``fee_model``.
    """

    taker_bps: float = _TAKER_BPS_NOT_GIVEN
    slippage_bps: float = 0.0
    slippage_fixed: float = 0.0
    depth_bps_per_unit: float = 0.0
    depth_reference_stake: float = 1.0
    fee_model: Literal["polymarket", "flat-bps"] | None = None
    fee_schedule: FeeSchedule | None = None
    fee_source: str | None = None

    def __post_init__(self) -> None:
        taker_given = not isinstance(self.taker_bps, _DefaultTakerBps)
        for name in ("taker_bps", "slippage_bps", "slippage_fixed", "depth_bps_per_unit"):
            value = getattr(self, name)
            if not _is_finite_number(value) or value < 0:
                raise ValueError(f"{name} must be a finite number >= 0, got {value!r}")
        stake = self.depth_reference_stake
        if not _is_finite_number(stake) or stake <= 0:
            raise ValueError(f"depth_reference_stake must be a finite number > 0, got {stake!r}")

        model = self._resolve_model(taker_given)
        schedule, source = self._resolve_schedule(model)
        object.__setattr__(self, "taker_bps", float(self.taker_bps))
        object.__setattr__(self, "fee_model", model)
        object.__setattr__(self, "fee_schedule", schedule)
        object.__setattr__(self, "fee_source", source)

    def _resolve_model(self, taker_given: bool) -> FeeModelName:
        model = self.fee_model
        if model is None:
            if self.fee_schedule is not None:
                if taker_given:
                    raise ValueError(
                        "a fee_schedule is the polymarket model and a taker_bps the flat-bps one: "
                        "give one of them, or name the model with fee_model"
                    )
                return "polymarket"
            if taker_given:
                # warn_deprecated, here, __post_init__, __init__: the caller of CostModel(...)
                warn_deprecated(**_BARE_TAKER_BPS, stacklevel=5)
                return "flat-bps"
            return "polymarket"
        if not isinstance(model, str) or model not in _FEE_MODELS:
            raise ValueError(f"fee_model must be one of {list(_FEE_MODELS)}, got {model!r}")
        return model

    def _resolve_schedule(self, model: FeeModelName) -> tuple[FeeSchedule | None, str]:
        schedule = self.fee_schedule
        source = self.fee_source
        if source is not None and (not isinstance(source, str) or not source.strip()):
            raise ValueError(f"fee_source must be a non-empty string, got {source!r}")
        if model == "flat-bps":
            if schedule is not None:
                raise ValueError(
                    'fee_model="flat-bps" charges taker_bps of the stake and takes no '
                    'fee_schedule; use fee_model="polymarket" for a schedule'
                )
            if source not in (None, _SOURCE_FLAT):
                raise ValueError(
                    f'a flat-bps model has no schedule to source; fee_source must be None or "{_SOURCE_FLAT}", '
                    f"got {source!r}"
                )
            return None, _SOURCE_FLAT
        if schedule is None:
            return category_fee_schedule(_DEFAULT_FEE_CATEGORY), source or _SOURCE_DEFAULT
        if isinstance(schedule, Mapping):
            schedule = FeeSchedule(**schedule)
        elif not isinstance(schedule, FeeSchedule):
            raise ValueError(
                f"fee_schedule must be a FeeSchedule or a mapping of its fields, got {schedule!r}"
            )
        return schedule, source or _SOURCE_GIVEN

    @classmethod
    def frictionless(cls) -> CostModel:
        """A model with no costs at all — useful as a backtest upper bound."""
        return cls(fee_model="flat-bps", taker_bps=0.0)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CostModel:
        """The model ``to_dict`` wrote, and the one 1.4.0 and before wrote.

        A dict with a ``taker_bps`` and no ``fee_model`` (and no ``fee_schedule``) is what every
        release before 1.5.0 wrote, and it is the flat-bps model, without the deprecation warning
        the constructor gives a bare ``taker_bps``: it is a file, not a call somebody can change.
        So a champion, a reproducibility record or a tuner saved then is read with the model it ran
        under. A dict with neither a ``taker_bps`` nor a ``fee_model`` is a partial one (only
        slippage, say) and gets the default model, as ``CostModel(**data)`` would.

        Args:
            data: The mapping of fields.

        Returns:
            The model.

        Raises:
            TypeError: If ``data`` is not a mapping, or has a key that is not a field.
            ValueError: If a value is out of range (see the class).
        """
        if not isinstance(data, Mapping):
            raise TypeError(f"CostModel.from_dict takes a mapping, got {type(data).__name__}")
        values = dict(data)
        if (
            "fee_model" not in values
            and "taker_bps" in values
            and values.get("fee_schedule") is None
        ):
            values["fee_model"] = "flat-bps"
        return cls(**values)

    def fee(self, stake: float, price: float | None = None) -> float:
        """Taker fee, in USDC, for buying *stake* notional at *price*.

        The flat-bps model charges ``stake x taker_bps / 10_000`` and ignores the price. The
        polymarket model charges the fee of the fill: *stake* USDC at *price* buys ``stake / price``
        shares and pays ``shares x rate x price x (1 - price)``, rounded as the schedule says (5
        decimals for Polymarket). Pass the price the fill happens at, `fill_price` (a fill at 0 buys
        no shares and pays nothing).

        Args:
            stake: Notional USDC bought, > 0.
            price: The fill price of one share, in [0, 1]. Required by the polymarket model.

        Returns:
            The fee in USDC.

        Raises:
            ValueError: If *stake* is not > 0, or *price* is missing (polymarket model) or not a
                price in [0, 1].
        """
        _require_positive_stake(stake)
        if self.fee_model == "flat-bps":
            return stake * self.taker_bps / _BPS
        if price is None:
            raise ValueError(
                "the polymarket fee depends on the price paid: call fee(stake, price) with the "
                "fill price, or use fee_model='flat-bps', whose fee does not"
            )
        if not _is_finite_number(price) or not 0.0 <= price <= 1.0:
            raise ValueError(f"price must be within [0, 1], got {price!r}")
        if price == 0.0:
            return 0.0  # a token at 0 is stake / 0 shares: nothing is bought, nothing is charged
        assert self.fee_schedule is not None  # resolved on construction for this model
        return taker_fee_on_stake(stake, price, self.fee_schedule)

    def fee_fields(self, stake: float, price: float | None = None) -> dict[str, Any]:
        """What a trade records about its fee: ``fee_paid``, ``fee_rate``, ``liquidity``, ``fee_source``.

        The one place those four keys are made, so a backtest trade and a paper trade carry them
        under the same names. ``fee_paid`` is `fee`. ``fee_rate`` is the rate of the model that
        charged it: the schedule's ``rate`` (0.0 for a fee-free market) for polymarket, whose fee is
        ``shares x rate x p x (1 - p)``, and ``taker_bps / 10_000`` for flat-bps, whose fee is
        ``stake x rate``. ``liquidity`` is ``"taker"``: this model fills at the book (maker fills
        are not modelled yet). ``fee_source`` is the schedule's source (see the class).

        Args:
            stake: Notional USDC bought, > 0.
            price: The fill price (see `fee`).

        Returns:
            The four fields as a dict.

        Raises:
            ValueError: As for `fee`.
        """
        if self.fee_model == "flat-bps":
            rate = self.taker_bps / _BPS
        else:
            assert self.fee_schedule is not None
            rate = self.fee_schedule.rate if self.fee_schedule.enabled else 0.0
        return {
            "fee_paid": self.fee(stake, price),
            "fee_rate": rate,
            "liquidity": "taker",
            "fee_source": self.fee_source,
        }

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

    def to_dict(self) -> dict[str, Any]:
        """Mapping for the experiment tracker and the JSON files; round-trips via `from_dict`.

        Names its ``fee_model``, and for the polymarket model carries the ``fee_schedule`` as the
        mapping of its fields (``None`` for flat-bps) and its ``fee_source``. A dict from before
        1.5.0 had only the five numbers; `from_dict` reads it as the flat-bps model.
        """
        return {
            "fee_model": self.fee_model,
            "taker_bps": self.taker_bps,
            "slippage_bps": self.slippage_bps,
            "slippage_fixed": self.slippage_fixed,
            "depth_bps_per_unit": self.depth_bps_per_unit,
            "depth_reference_stake": self.depth_reference_stake,
            "fee_schedule": None if self.fee_schedule is None else self.fee_schedule.to_dict(),
            "fee_source": self.fee_source,
        }


def _require_positive_stake(stake: float) -> None:
    if stake <= 0:
        raise ValueError(f"stake must be > 0, got {stake}")
