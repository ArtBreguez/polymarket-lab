"""Fee model: a ``FeeSchedule``, the taker curve, rounding and Polymarket's category defaults.

Polymarket charges a **taker** fee per fill, in USDC, that depends on the price and on the market::

    fee = shares x rate x (price x (1 - price)) ** exponent        (exponent is 1 on every market today)

rounded to 5 decimals, with a smallest fee of 0.00001 (anything smaller is not charged). Makers pay
nothing. The curve peaks at a price of 0.50 and is symmetric around it, so a taker pays the same
dollar fee at 30 cents as at 70. As a fraction of the stake the fee is ``rate x (1 - price)`` for
exponent 1: 3.5% at 0.30 and 2.5% at 0.50 for a 0.05-rate market, against the flat 0.30% that
``CostModel`` and ``estimate_fee`` still assume.

This module is the core only: pure functions and one frozen value, no I/O and no network. It changes
no existing number: ``CostModel``, the backtest and the brokers still use ``taker_bps``.

**What a schedule can say.** ``FeeSchedule`` also carries a ``maker_rate``, one rounding rule
(``rounding``, ``rounding_unit``) and whether it is applied to each fill or to the sum of an order
(``rounding_scope``). That describes a venue whose taker fee is ``rate x shares x (price x
(1 - price)) ** exponent`` rounded by one rule. It is not shown to describe every venue: a venue
that rounds in two steps (a unit per fill, then a settlement per order, with rebates paid out of a
per-order accumulator across taker and maker fills; reported for Kalshi, not read from its own page
here) cannot be written as one unit and one scope, and ``maker_fee`` has no per-order form. Whoever
builds a second venue on this class (#75) must verify that venue's rounding page first.

Verification record (what was checked before ``exponent`` was coded, 2026-11-25)
--------------------------------------------------------------------------------

*Where does the exponent go?* On the whole product ``price x (1 - price)``, not on ``price`` alone.

* The official V2 clients compute the fee: ``platform_fee_rate = rate * (price * (1 - price)) **
  exponent`` and ``platform_fee = (amount / price) * platform_fee_rate`` (``amount`` is the USDC
  stake, so ``amount / price`` is the share count). Read at these commits:
  ``Polymarket/py-clob-client-v2`` 9c7c9e89cbbc11528daae90dc66ef65c9380cbdf (2026-11-12),
  ``py_clob_client_v2/fees.py`` lines 31 and 34; ``Polymarket/clob-client-v2`` v1.3.1
  2a3f0b02b46de77afee2364881e0256eab4d75cb, ``src/fees/index.ts``; ``Polymarket/rs-clob-client-v2``
  v0.9.0 a27e39c68109828c7bd95cc0a9828affe1a437d7, ``src/clob/utilities.rs``
  (``adjust_market_buy_amount``); ``Polymarket/ts-sdk`` d1ee414086ea5c14c0b367e90b06d4c51eb9c871,
  ``packages/client/src/actions/orders/market.ts`` lines 431-438. All four hold the same formula.
* They do not round. The 5-decimal rule comes from the docs alone.
* The client pmlab depends on, ``py-clob-client`` (0.34.6, archived 2026-05-25), has no fee
  arithmetic: it passes ``base_fee`` through as ``feeRateBps``. ``Polymarket/py-sdk`` 0.20.1
  (81e8abfb2a131ef3874f8fe405c9d95b2123fcdf) models the Gamma ``feeSchedule`` (``rate``,
  ``exponent``, ``takerOnly``, ``rebateRate``) and does no arithmetic either.
* https://docs.polymarket.com/trading/fees prints ``fee = C x feeRate x p x (1 - p)``: the exponent-1
  case, with no exponent in the formula. https://docs.polymarket.com/market-data/market-details
  describes ``feeSchedule.exponent`` only as "exponent applied to the price component of the fee
  curve". The Wayback Machine copies of the fees page give the older form: 2026-02-20
  ``fee = C x feeRate x (p x (1 - p))^exponent`` (crypto rate 0.25 exponent 2, sports 0.0175
  exponent 1) and 2026-03-05 ``fee = C x p x feeRate x (p x (1 - p))^exponent``. The clients agree
  with the first, which is the form implemented here; the second carries one more factor of ``p``
  that no client has.
* Live Gamma ``/markets`` on 2026-11-25, 800 active markets: every ``feeSchedule`` has
  ``exponent`` 1 and ``takerOnly`` true. So the exponent matters only if Polymarket brings back
  a market with exponent above 1; for exponent 1 the docs and the clients are the same formula.

*Rounding.* The docs say: "Fees are rounded to 5 decimal places. The smallest fee charged is 0.00001
USDC. Anything smaller rounds to zero." They do not say which way a tie goes, and no client rounds
a fee. This module rounds half up on the exact decimal value (a fee of exactly 0.000015 is 0.00002),
and does not charge a raw fee below one unit (``rounding="nearest"``). The only effect of a different
tie rule would be one unit (0.00001 USDC) on a fee that lands exactly on a tie. Note that
``"nearest"`` is this minimum-fee rule and not the plain nearest unit: on a cent unit a raw fee of
0.007 is 0, not 0.01. ``"up"`` and ``"down"`` are plain; a plain nearest unit has no value yet.

*Exactness.* The arithmetic is exact decimal arithmetic for exponent 1, which is every market today.
For another exponent the curve is a float power (``math.pow``) read to 15 significant digits, so the
tie guarantee above holds for exponent 1 only.

*The category defaults.* ``POLYMARKET_CATEGORY_FEE_RATES`` is the docs' table as read on 2026-11-25;
a test parses the page saved in ``tests/core/data`` and holds the constant and the 100-share tables
to it. The rates changed after the issue that asked for this module was written: the Wayback copy of
2026-11-12 still had Crypto 0.07, Finance/Politics/Mentions/Tech 0.04 and Economics/Culture/Weather
0.05. Both the live docs and the live Gamma ``feeSchedule`` of every market sampled carry the table
below. Polymarket sets them ("set by the protocol") and has changed them without notice, so a
schedule read from the market itself, not a default, is the right input for a backtest of a past
period.
"""

from __future__ import annotations

import math
import numbers
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Context, Decimal
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

# Six public names. ``POLYMARKET_CATEGORY_FEE_RATES``, ``POLYMARKET_FEE_RATES_AS_OF``,
# ``POLYMARKET_FEE_RATES_SOURCE_URL`` and ``category_fee_schedule`` are module-internal in 1.2.0: the
# table's keys are ours, not Gamma's, and the way a market's tags pick a fallback is still open
# (#71), so they are exported from ``pmlab`` only once that is settled.
__all__ = [
    "FeeSchedule",
    "maker_fee",
    "stake_fee_fraction",
    "taker_fee",
    "taker_fee_for_fills",
    "taker_fee_on_stake",
]

POLYMARKET_FEE_RATES_SOURCE_URL = "https://docs.polymarket.com/trading/fees"
"""The page ``POLYMARKET_CATEGORY_FEE_RATES`` was read from."""

POLYMARKET_FEE_RATES_AS_OF = "2026-11-25"
"""The date (ISO) the table was read: Polymarket sets and changes these rates."""

POLYMARKET_CATEGORY_FEE_RATES: Mapping[str, float] = MappingProxyType(
    {
        "crypto_btc_5m_15m": 0.05,  # Crypto: BTC 5-minute and 15-minute up/down
        "crypto": 0.01,  # Crypto: all other markets
        "sports": 0.05,  # Sports, including NFL and college football
        "finance": 0.05,
        "tech": 0.05,
        "politics_elections": 0.05,  # Politics: elections
        "politics": 0.01,  # Politics: excluding elections and geopolitics
        "economics": 0.01,
        "culture": 0.01,
        "weather": 0.01,
        "mentions": 0.01,
        "geopolitics": 0.0,  # fee-free
        "combos": 0.05,  # every combo, whatever its legs' categories
        "other": 0.05,  # "Other / General"
    }
)
"""Taker fee rate by category, as published at ``POLYMARKET_FEE_RATES_SOURCE_URL`` on
``POLYMARKET_FEE_RATES_AS_OF``. Read-only. Makers pay 0 in every category. Defaults only: the rate
a market actually carries is in its Gamma ``feeSchedule``."""

# The docs call the last row "Other / General"; accept that spelling next to ours.
_CATEGORY_ALIASES = {"other_general": "other"}

_ROUNDING_MODES: dict[str, str] = {
    "nearest": ROUND_HALF_UP,
    "up": ROUND_CEILING,
    "down": ROUND_FLOOR,
}
_ROUNDING_SCOPES = ("fill", "order")

# Exact decimal arithmetic, wide enough that no product of two floats is cut: a fee that is a tie
# in decimal (0.000015) is a tie here, where binary floating point would see 1.4999999999999999e-05.
# Exact for exponent 1; another exponent is a float power (see _raw_fee).
_CTX = Context(prec=60)
_ZERO = Decimal(0)
_ONE = Decimal(1)


def _decimal(value: float) -> Decimal:
    """The decimal a float stands for: ``0.05``, not its binary expansion, and ``0.07 * 3`` is
    0.21, not 0.21000000000000002. A binary double always keeps 15 significant digits, so digits
    16 and 17 are float noise, not something the caller wrote."""
    return Decimal(format(value, ".15g"))


def _number(name: str, value: object) -> float:
    """A finite real number as a float; a bool, a string or NaN is not one."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    try:
        number = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be a finite number, got {value!r}") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return number


def _field_number(name: str, value: object) -> float:
    """A numeric field of ``FeeSchedule``, as the float stored: what ``_number`` takes, or a
    ``Decimal`` or a numeric ``str``. It is cut to 15 significant digits (what a binary double
    always keeps), so ``0.07 * 3`` and ``Decimal("0.07") * 3`` are the same 0.21 and two schedules
    that charge alike compare equal."""
    if isinstance(value, Decimal | str):
        try:
            number = float(Decimal(value.strip() if isinstance(value, str) else value))
        except (ValueError, ArithmeticError):  # InvalidOperation is an ArithmeticError
            raise ValueError(f"{name} must be a finite number, got {value!r}") from None
        if not math.isfinite(number):
            raise ValueError(f"{name} must be a finite number, got {value!r}")
    else:
        number = _number(name, value)
    return float(_decimal(number))


def _fraction(name: str, value: object, *, closed_above: bool) -> float:
    """A finite number in ``[0, 1]`` (``closed_above``) or ``[0, 1)``."""
    number = _field_number(name, value)
    if not (0.0 <= number <= 1.0 if closed_above else 0.0 <= number < 1.0):
        interval = "[0, 1]" if closed_above else "[0, 1)"
        raise ValueError(f"{name} must be in {interval}, got {value!r}")
    return number


@dataclass(frozen=True)
class FeeSchedule:
    """How a venue charges a trade: the rate, the curve, who pays and how the fee is rounded.

    Polymarket's ``feeSchedule`` is ``rate``, ``exponent``, ``taker_only`` and ``rebate_rate``
    (with ``enabled`` for ``feesEnabled``); the other fields describe a venue that rounds or charges
    makers differently, with defaults that mean Polymarket. The taker fee of one fill is ``shares x
    rate x (price x (1 - price)) ** exponent``, then rounded as described below (see ``taker_fee``).

    The numeric fields take a float or int (numpy scalars too) or a ``Decimal`` or a numeric ``str``;
    each is stored as a float. A float is read as the decimal it was written as, to 15 significant
    digits, so ``rate=0.07 * 3`` is 0.21 and not 0.21000000000000002, and a ``Decimal`` or ``str``
    is exact to the same 15 digits. The flags take a ``bool`` or a ``numpy.bool_``.

    Attributes:
        rate: Fee rate, in ``[0, 1)``. ``0.05`` is the Sports rate.
        exponent: Exponent of the curve ``price x (1 - price)``, finite and above 0. Every
            Polymarket market has 1 today; see the module docstring for what was checked. Fees
            are exact decimals for exponent 1 only (another exponent goes through a float power).
        taker_only: True when makers pay nothing. When False, makers pay ``maker_rate``.
        rebate_rate: The market-level rebate fraction (``0.25`` is 25%), in ``[0, 1]``. Stored for
            the maker-rebate estimate; it does not change a fee. It is not an account's tier rate.
        enabled: False when the market charges no fee (Gamma's ``feesEnabled``): every fee is 0.
        maker_rate: Rate makers pay when ``taker_only`` is False, in ``[0, 1)``. ``None`` means the
            same as ``rate`` (both sides pay the schedule). Must be ``None`` or 0 when
            ``taker_only`` is True, so a rate that is never charged cannot be set by mistake.
        rounding: ``"nearest"`` is Polymarket's minimum-fee rule, not the plain nearest unit: half up,
            and a raw fee smaller than one ``rounding_unit`` is not charged (a 0.007 fee on a cent
            unit is 0, not 0.01), as Polymarket's docs say. ``"up"`` (any positive fee pays at
            least one unit) and ``"down"`` are plain.
        rounding_unit: The unit fees are rounded to, in ``(0, 1]``. ``0.00001`` is 5 decimals,
            ``0.01`` a cent.
        rounding_scope: ``"fill"`` when each fill is rounded and the fills added, ``"order"`` when
            the fees of the fills of one order are added and rounded once
            (see ``taker_fee_for_fills``).

    Raises:
        ValueError: If a field is not a finite real number (or flag, or one of the listed
            strings) in its range, or ``maker_rate`` is set on a ``taker_only`` schedule.
    """

    rate: float
    exponent: float = 1.0
    taker_only: bool = True
    rebate_rate: float = 0.0
    enabled: bool = True
    maker_rate: float | None = None
    # "nearest" is Polymarket's rule (half up, and nothing below one unit), not the plain nearest unit
    rounding: Literal["nearest", "up", "down"] = "nearest"
    rounding_unit: float = 0.00001
    rounding_scope: Literal["fill", "order"] = "fill"

    def __post_init__(self) -> None:
        # numbers are stored as plain floats (and flags as plain bools), so a numpy scalar, a
        # Decimal or a string still hashes, compares and serialises like the value it stands for
        object.__setattr__(self, "rate", _fraction("rate", self.rate, closed_above=False))
        exponent = _field_number("exponent", self.exponent)
        if exponent <= 0.0:
            raise ValueError(f"exponent must be above 0, got {self.exponent!r}")
        object.__setattr__(self, "exponent", exponent)
        object.__setattr__(
            self, "rebate_rate", _fraction("rebate_rate", self.rebate_rate, closed_above=True)
        )
        for flag in ("taker_only", "enabled"):
            value = getattr(self, flag)
            if not isinstance(value, bool | np.bool_):  # a DataFrame row gives numpy bools
                raise ValueError(f"{flag} must be a bool, got {value!r}")
            object.__setattr__(self, flag, bool(value))
        if self.maker_rate is not None:
            object.__setattr__(
                self, "maker_rate", _fraction("maker_rate", self.maker_rate, closed_above=False)
            )
            if self.taker_only and self.maker_rate != 0.0:
                raise ValueError(
                    "maker_rate must be None or 0 when taker_only is True "
                    f"(makers pay nothing), got {self.maker_rate!r}"
                )
        if not isinstance(self.rounding, str) or self.rounding not in _ROUNDING_MODES:
            raise ValueError(
                f"rounding must be one of {sorted(_ROUNDING_MODES)}, got {self.rounding!r}"
            )
        unit = _field_number("rounding_unit", self.rounding_unit)
        if not 0.0 < unit <= 1.0:
            raise ValueError(f"rounding_unit must be in (0, 1], got {self.rounding_unit!r}")
        object.__setattr__(self, "rounding_unit", unit)
        if not isinstance(self.rounding_scope, str) or self.rounding_scope not in _ROUNDING_SCOPES:
            raise ValueError(
                f"rounding_scope must be one of {list(_ROUNDING_SCOPES)}, got {self.rounding_scope!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Flat mapping for the experiment tracker; round-trips via ``FeeSchedule(**d)``."""
        return asdict(self)


# ---------------------------------------------------------------------------
# arithmetic (all in Decimal; a float is only the last step)
# ---------------------------------------------------------------------------


def _require_schedule(schedule: object) -> FeeSchedule:
    if not isinstance(schedule, FeeSchedule):
        raise TypeError(f"schedule must be a FeeSchedule, got {type(schedule).__name__}")
    return schedule


def _shares_and_price(shares: object, price: object) -> tuple[Decimal, Decimal]:
    """A fill's inputs: shares finite and >= 0, price a probability in ``[0, 1]``."""
    n = _number("shares", shares)
    p = _number("price", price)
    if n < 0.0:
        raise ValueError(f"shares must be >= 0, got {shares!r}")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"price must be in [0, 1], got {price!r}")
    return _decimal(n), _decimal(p)


def _raw_fee(shares: Decimal, price: Decimal, rate: float, exponent: float) -> Decimal:
    """``shares x rate x (price x (1 - price)) ** exponent``, unrounded."""
    base = _CTX.multiply(price, _CTX.subtract(_ONE, price))
    curve = base if exponent == 1.0 or base == 0 else _decimal(math.pow(float(base), exponent))
    return _CTX.multiply(_CTX.multiply(shares, _decimal(rate)), curve)


def _round_fee(raw: Decimal, schedule: FeeSchedule) -> Decimal:
    """Apply the schedule's rounding rule to a raw fee."""
    unit = _decimal(schedule.rounding_unit)
    if schedule.rounding == "nearest" and raw < unit:
        return _ZERO  # "anything smaller than the smallest fee rounds to zero"
    units = _CTX.divide(raw, unit).to_integral_value(rounding=_ROUNDING_MODES[schedule.rounding])
    return _CTX.multiply(units, unit)


def _maker_rate(schedule: FeeSchedule) -> float:
    if schedule.taker_only:
        return 0.0
    return schedule.rate if schedule.maker_rate is None else schedule.maker_rate


def _taker_fee(shares: Decimal, price: Decimal, schedule: FeeSchedule) -> float:
    if not schedule.enabled:
        return 0.0
    return float(_round_fee(_raw_fee(shares, price, schedule.rate, schedule.exponent), schedule))


# ---------------------------------------------------------------------------
# public functions
# ---------------------------------------------------------------------------


def taker_fee(shares: float, price: float, schedule: FeeSchedule) -> float:
    """The fee, in USDC, a taker pays for one fill of ``shares`` at ``price``.

    ``shares x rate x (price x (1 - price)) ** exponent``, rounded by the schedule (for Polymarket to
    5 decimals, half up, and a fee below 0.00001 is 0). A disabled schedule charges 0. The price is
    the fill price of one share, in ``[0, 1]``; a fee is 0 at both ends and peaks at 0.50.

    For exponent 1 the arithmetic is exact decimal arithmetic on the values as written (``0.05``, not
    its binary expansion), so a fee that is a tie in decimal is rounded as a tie. For another exponent
    the curve is a float power read to 15 significant digits, and that guarantee does not hold.
    ``shares`` and ``price`` are floats or ints (numpy scalars too); a ``Decimal`` or a string is
    refused, only the schedule's own fields take one.

    Args:
        shares: Number of shares filled, finite and >= 0.
        price: Fill price per share, in ``[0, 1]``.
        schedule: The market's fee schedule.

    Returns:
        The fee in USDC, a multiple of the schedule's ``rounding_unit``.

    Raises:
        ValueError: If ``shares`` or ``price`` is not a finite number in range (checked even when
            the fee is 0).
        TypeError: If ``schedule`` is not a ``FeeSchedule``.
    """
    _require_schedule(schedule)
    n, p = _shares_and_price(shares, price)
    return _taker_fee(n, p, schedule)


def maker_fee(shares: float, price: float, schedule: FeeSchedule) -> float:
    """The fee, in USDC, a maker pays for one fill of ``shares`` at ``price``.

    0 when the schedule is ``taker_only`` (Polymarket: "makers are never charged fees") or disabled.
    Otherwise the same curve at ``maker_rate`` (the taker ``rate`` when ``maker_rate`` is ``None``),
    rounded by the schedule. Makers' rebates are a separate matter and are not netted here.

    Args:
        shares: Number of shares filled, finite and >= 0.
        price: Fill price per share, in ``[0, 1]``.
        schedule: The market's fee schedule.

    Returns:
        The fee in USDC.

    Raises:
        ValueError: As for ``taker_fee``.
        TypeError: If ``schedule`` is not a ``FeeSchedule``.
    """
    _require_schedule(schedule)
    n, p = _shares_and_price(shares, price)
    rate = _maker_rate(schedule)
    if not schedule.enabled or rate == 0.0:
        return 0.0
    return float(_round_fee(_raw_fee(n, p, rate, schedule.exponent), schedule))


def taker_fee_for_fills(fills: Iterable[tuple[float, float]], schedule: FeeSchedule) -> float:
    """The taker fee, in USDC, of an order that was filled in several parts.

    ``fills`` are ``(shares, price)`` pairs, for instance the levels a market order walked. With
    ``rounding_scope="fill"`` each fill is rounded as ``taker_fee`` does and the results are added.
    With ``"order"`` the unrounded fees are added and the sum is rounded once, so an order filled at
    three levels pays one rounding, not three.

    Args:
        fills: ``(shares, price)`` pairs; any iterable, read once. Empty means no fee.
        schedule: The market's fee schedule.

    Returns:
        The fee in USDC.

    Raises:
        ValueError: If ``fills`` is not an iterable of pairs, or a pair is out of range
            (see ``taker_fee``).
        TypeError: If ``schedule`` is not a ``FeeSchedule``.
    """
    _require_schedule(schedule)
    try:
        items = list(fills)
    except TypeError:
        raise ValueError(
            f"fills must be an iterable of (shares, price) pairs, got {fills!r}"
        ) from None
    parsed: list[tuple[Decimal, Decimal]] = []
    for item in items:
        try:
            shares, price = item
        except (TypeError, ValueError):
            raise ValueError(f"each fill must be a (shares, price) pair, got {item!r}") from None
        parsed.append(_shares_and_price(shares, price))
    if not schedule.enabled or not parsed:
        return 0.0
    raws = [_raw_fee(n, p, schedule.rate, schedule.exponent) for n, p in parsed]
    total = _ZERO
    if schedule.rounding_scope == "fill":
        for raw in raws:
            total = _CTX.add(total, _round_fee(raw, schedule))
    else:
        for raw in raws:
            total = _CTX.add(total, raw)
        total = _round_fee(total, schedule)
    return float(total)


def stake_fee_fraction(price: float, schedule: FeeSchedule) -> float:
    """The taker fee as a fraction of the stake, before rounding: ``fee / (shares x price)``.

    For exponent 1 this is ``rate x (1 - price)``: for a 0.05-rate market 3.5% of the stake at 0.30
    and 2.5% at 0.50. In general ``rate x (price x (1 - price)) ** exponent / price``. For code that
    sizes by notional (a flat stake, Kelly): ``fee = stake x stake_fee_fraction(price, schedule)``,
    which differs from ``taker_fee_on_stake`` only by its rounding (under one unit per fill).

    Args:
        price: Fill price per share, in ``(0, 1]``: a stake buys no shares at 0.
        schedule: The market's fee schedule.

    Returns:
        The fraction (0 for a disabled schedule).

    Raises:
        ValueError: If ``price`` is not a finite number in ``(0, 1]``.
        TypeError: If ``schedule`` is not a ``FeeSchedule``.
    """
    _require_schedule(schedule)
    p = _number("price", price)
    if not 0.0 < p <= 1.0:
        raise ValueError(f"price must be in (0, 1], got {price!r}")
    if not schedule.enabled:
        return 0.0
    if schedule.exponent == 1.0:
        return schedule.rate * (1.0 - p)
    return schedule.rate * math.pow(p * (1.0 - p), schedule.exponent) / p


def taker_fee_on_stake(stake: float, price: float, schedule: FeeSchedule) -> float:
    """The taker fee, in USDC, of buying ``stake`` USDC of shares at ``price``.

    ``stake / price`` shares are filled, and the fee is ``taker_fee`` of that fill. The fee is on top
    of the stake: it does not reduce the number of shares bought. This is the fee the official
    clients estimate for a market buy of ``amount`` USDC, ``(amount / price) x rate x
    (price x (1 - price)) ** exponent``, with the rounding they leave out.

    Args:
        stake: USDC of shares bought, finite and >= 0.
        price: Fill price per share, in ``(0, 1]``.
        schedule: The market's fee schedule.

    Returns:
        The fee in USDC.

    Raises:
        ValueError: If ``stake`` or ``price`` is out of range.
        TypeError: If ``schedule`` is not a ``FeeSchedule``.
    """
    _require_schedule(schedule)
    amount = _number("stake", stake)
    p = _number("price", price)
    if amount < 0.0:
        raise ValueError(f"stake must be >= 0, got {stake!r}")
    if not 0.0 < p <= 1.0:
        raise ValueError(f"price must be in (0, 1], got {price!r}")
    price_dec = _decimal(p)
    shares = _CTX.divide(_decimal(amount), price_dec)
    return _taker_fee(shares, price_dec, schedule)


def category_fee_schedule(category: str) -> FeeSchedule:
    """The default ``FeeSchedule`` of a Polymarket category (taker only, exponent 1, 5 decimals).

    ``category`` is one of the keys of ``POLYMARKET_CATEGORY_FEE_RATES``; case, spaces and dashes
    are ignored, and the docs' own ``"Other / General"`` is accepted. A rate of 0 (Geopolitics)
    gives a disabled schedule.

    Prefer the schedule a market itself carries: these are the published defaults as of
    ``POLYMARKET_FEE_RATES_AS_OF``, and Polymarket changes them.

    Args:
        category: A category key such as ``"sports"`` or ``"weather"``.

    Returns:
        The schedule for that category.

    Raises:
        ValueError: If ``category`` is not a string or is not in the table.
    """
    if not isinstance(category, str):
        raise ValueError(f"category must be a string, got {category!r}")
    key = re.sub(r"[^a-z0-9]+", "_", category.strip().lower()).strip("_")
    key = _CATEGORY_ALIASES.get(key, key)
    rate = POLYMARKET_CATEGORY_FEE_RATES.get(key)
    if rate is None:
        raise ValueError(
            f"unknown fee category {category!r}; known: {', '.join(sorted(POLYMARKET_CATEGORY_FEE_RATES))} "
            f"(the table is {POLYMARKET_FEE_RATES_SOURCE_URL}, as of {POLYMARKET_FEE_RATES_AS_OF})"
        )
    return FeeSchedule(rate=rate, enabled=rate > 0.0)
