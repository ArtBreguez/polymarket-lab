"""The fee model (1.2.0), used from the installed wheel.

Standalone:  python audit_fee_model.py <polymarket_fees_docs_2026-11-25.md>
Prints ``audit fee model: OK`` and exits 0, or lists every check that failed and exits 1.

The saved copy of Polymarket's fee page is the checked-in ``tests/core/data`` file (byte for byte);
the 100-share tables are read from it, so what is compared with the wheel is what the docs print
and not a second copy typed here.

Six names are public and come from ``pmlab``. Polymarket's category table and the lookup over it
(``POLYMARKET_CATEGORY_FEE_RATES``, ``POLYMARKET_FEE_RATES_AS_OF``, ``POLYMARKET_FEE_RATES_SOURCE_URL``,
``category_fee_schedule``) are module-internal in 1.2.0 (they become public with #71), so this
script reads them from ``pmlab.core.fee_schedule`` of the installed wheel, the one place in the
repo's e2e that does, and checks that ``pmlab`` itself does not export them.
"""

from __future__ import annotations

import dataclasses
import math
import operator
import re
import sys
import traceback
from collections.abc import Callable
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pmlab
import pmlab.core.fee_schedule as fee_module
from pmlab import (
    CostModel,
    FeeSchedule,
    estimate_fee,
    maker_fee,
    stake_fee_fraction,
    taker_fee,
    taker_fee_for_fills,
    taker_fee_on_stake,
)
from pmlab.core.fee_schedule import (
    POLYMARKET_CATEGORY_FEE_RATES,
    POLYMARKET_FEE_RATES_AS_OF,
    POLYMARKET_FEE_RATES_SOURCE_URL,
    category_fee_schedule,
)

DOCS_PAGE = Path(sys.argv[1]) if len(sys.argv) > 1 else None
NAMES = [
    "FeeSchedule",
    "maker_fee",
    "stake_fee_fraction",
    "taker_fee",
    "taker_fee_for_fills",
    "taker_fee_on_stake",
]
INTERNAL = [
    "POLYMARKET_CATEGORY_FEE_RATES",
    "POLYMARKET_FEE_RATES_AS_OF",
    "POLYMARKET_FEE_RATES_SOURCE_URL",
    "category_fee_schedule",
]

# The label of each row of the docs' category table -> the key of pmlab's table.
LABEL_TO_KEY = {
    "Crypto — BTC 5-minute and 15-minute up/down": "crypto_btc_5m_15m",
    "Crypto — all other markets": "crypto",
    "Sports, including NFL and college football": "sports",
    "Finance": "finance",
    "Tech": "tech",
    "Politics — elections": "politics_elections",
    "Politics — excluding elections and geopolitics": "politics",
    "Economics": "economics",
    "Culture": "culture",
    "Weather": "weather",
    "Mentions": "mentions",
    "Geopolitics": "geopolitics",
    "Combos": "combos",
    "Other / General": "other",
}
# The categories each "Fee Tables (100 Shares)" tab of the page is about (Combos has no tab).
TAB_CATEGORIES = {
    "Crypto": ["crypto_btc_5m_15m", "crypto"],
    "Sports": ["sports"],
    "Finance / Politics / Mentions / Tech": [
        "finance",
        "tech",
        "politics_elections",
        "politics",
        "mentions",
    ],
    "Economics / Culture / Weather / Other": ["other", "economics", "culture", "weather"],
}

CHECKS: list[tuple[str, Callable[[], None]]] = []


def check(fn: Callable[[], None]) -> Callable[[], None]:
    CHECKS.append((fn.__name__, fn))
    return fn


def refused(
    exc_type: type[Exception], fn: Callable[..., object], *args: object, **kw: object
) -> str:
    try:
        fn(*args, **kw)
    except exc_type as error:
        return str(error)
    raise AssertionError(f"{getattr(fn, '__name__', fn)}{args}{kw} was accepted")


def cents(value: float) -> Decimal:
    return Decimal(repr(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def docs_text() -> str:
    assert DOCS_PAGE is not None and DOCS_PAGE.is_file(), "pass the saved fee page as argument 1"
    return DOCS_PAGE.read_text(encoding="utf-8")


def docs_category_rates() -> dict[str, float]:
    """The ``Taker Fee Rate`` column of the page's category table, by pmlab's key."""
    rates: dict[str, float] = {}
    for line in docs_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 3 and cells[0] in LABEL_TO_KEY:
            assert cells[2] == "0", f"{cells[0]}: the page charges makers {cells[2]}"
            rates[LABEL_TO_KEY[cells[0]]] = float(cells[1])
    return rates


def docs_fee_tables() -> dict[str, dict[float | None, dict[float, Decimal]]]:
    """Each tab's table: ``{tab: {rate column (None: the tab names no rate): {price: fee}}}``."""
    tables: dict[str, dict[float | None, dict[float, Decimal]]] = {}
    for block in re.split(r"^> \[tab\] ", docs_text(), flags=re.MULTILINE)[1:]:
        title, _, rest = block.partition("\n")
        columns: list[float | None] = []
        rows: dict[float | None, dict[float, Decimal]] = {}
        for line in rest.splitlines():
            if not line.startswith("> |"):
                continue
            cells = [c.strip() for c in line[2:].strip().strip("|").split("|")]
            if cells[0] == "Price":
                found = [float(m) for m in re.findall(r"\(([0-9.]+), USDC\)", " ".join(cells[2:]))]
                columns = list(found) if found else [None]
                rows = {c: {} for c in columns}
            elif cells[0].startswith("\\$"):
                price = float(cells[0].lstrip("\\$"))
                for column, cell in zip(columns, cells[2:], strict=True):
                    rows[column][price] = Decimal(cell.lstrip("\\$"))
        tables[title.strip()] = rows
    return tables


@check
def the_six_names_are_public_and_the_category_table_is_not_yet() -> None:
    assert "site-packages" in pmlab.__file__, pmlab.__file__
    assert "site-packages" in fee_module.__file__, fee_module.__file__
    assert set(NAMES) <= set(pmlab.__all__), set(NAMES) - set(pmlab.__all__)
    for name in NAMES:
        assert getattr(pmlab, name) is getattr(fee_module, name), name
    assert sorted(fee_module.__all__) == sorted(NAMES), fee_module.__all__
    for name in INTERNAL:  # module-internal until #71 settles how a market's tags pick a rate
        assert name not in pmlab.__all__ and not hasattr(pmlab, name), name
        assert name not in fee_module.__all__ and hasattr(fee_module, name), name
    assert POLYMARKET_FEE_RATES_SOURCE_URL == "https://docs.polymarket.com/trading/fees"
    assert POLYMARKET_FEE_RATES_AS_OF == "2026-11-25"


@check
def the_category_table_is_the_docs_table_and_read_only() -> None:
    page = docs_category_rates()
    assert len(page) == len(LABEL_TO_KEY) == 14, sorted(page)
    assert dict(POLYMARKET_CATEGORY_FEE_RATES) == page
    assert category_fee_schedule("weather").rate == 0.01
    assert category_fee_schedule("Politics - Elections").rate == 0.05  # spelled as a person would
    assert category_fee_schedule("Other / General") == category_fee_schedule("other")
    refused(TypeError, operator.setitem, POLYMARKET_CATEGORY_FEE_RATES, "weather", 0.5)
    assert POLYMARKET_CATEGORY_FEE_RATES["weather"] == 0.01
    message = refused(ValueError, category_fee_schedule, "astrology")
    assert "astrology" in message and "weather" in message and POLYMARKET_FEE_RATES_AS_OF in message
    assert "string" in refused(ValueError, category_fee_schedule, 5)


@check
def a_hundred_shares_cost_what_the_docs_table_prints_to_the_cent() -> None:
    tables = docs_fee_tables()
    assert set(tables) == set(TAB_CATEGORIES), sorted(tables)
    checked = 0
    for tab, keys in TAB_CATEGORIES.items():
        by_rate = tables[tab]
        for key in keys:
            schedule = category_fee_schedule(key)
            # the Sports tab prints one unlabelled column: it is the Sports rate
            rows = by_rate[None] if None in by_rate else by_rate[schedule.rate]
            assert len(rows) == 21, (tab, key, len(rows))
            for price, printed in rows.items():
                fee = taker_fee(100, price, schedule)
                assert cents(fee) == printed, f"{key} @ {price}: {fee} vs the docs' {printed}"
                checked += 1
    assert checked == 21 * 12, checked  # 12 categories have a tab; Combos has none
    geopolitics = category_fee_schedule("geopolitics")
    assert geopolitics.enabled is False and geopolitics.rate == 0.0
    assert all(taker_fee(100, p / 100, geopolitics) == 0.0 for p in range(101))
    assert taker_fee(100, 0.5, category_fee_schedule("sports")) == 1.25  # the page's stated peak
    assert taker_fee(100, 0.5, category_fee_schedule("weather")) == 0.25


@check
def the_exponent_applies_to_the_whole_price_curve() -> None:
    schedule = FeeSchedule(rate=0.25, exponent=2)
    assert taker_fee(100, 0.3, schedule) == 1.1025
    assert taker_fee(100, 0.3, schedule) not in (3.675, 1.575)  # on p only, or on (1 - p) only
    for exponent in (1, 2, 3, 1.5, 0.5):  # the official clients: rate * (p * (1 - p)) ** exponent
        for price in (0.1, 0.3, 0.5, 0.77):
            raw = 100 * 0.07 * (price * (1 - price)) ** exponent  # shares x the clients' rate
            fee = taker_fee(100, price, FeeSchedule(rate=0.07, exponent=exponent))
            assert abs(fee - raw) <= 1e-5, (exponent, price, fee, raw)


@check
def a_fee_below_one_unit_is_not_charged_and_a_tie_rounds_up() -> None:
    schedule = FeeSchedule(rate=0.05)  # at price 0.5 the raw fee is shares x 0.0125
    for shares, expected in [
        (0.0007999, 0.0),
        (0.0008, 0.00001),
        (0.00119999, 0.00001),
        (0.0012, 0.00002),  # raw 0.000015 exactly: a tie, in decimal, rounds up
        (0.0016, 0.00002),
        (0.0024, 0.00003),
        (0.0, 0.0),
    ]:
        assert taker_fee(shares, 0.5, schedule) == expected, (
            shares,
            taker_fee(shares, 0.5, schedule),
        )
    assert taker_fee(0.0012, 0.5, schedule) == 0.00002 != round(0.0012 * 0.05 * 0.25, 5)
    assert taker_fee(0.01, 0.01, schedule) == 0.0
    assert taker_fee(100, 0.0001, schedule) == 0.0005
    assert taker_fee(3.3333, 0.5, schedule) == 0.04167
    up = FeeSchedule(rate=0.07, rounding="up", rounding_unit=0.01)
    assert taker_fee(1, 0.5, up) == 0.02  # raw 0.0175


@check
def a_maker_pays_nothing_when_the_schedule_is_taker_only() -> None:
    schedule = FeeSchedule(rate=0.05)
    assert schedule.taker_only is True
    for shares, price in [(100, 0.5), (1, 0.01), (1e6, 0.3), (0, 0.5)]:
        assert maker_fee(shares, price, schedule) == 0.0, (shares, price)
    both = FeeSchedule(rate=0.05, taker_only=False)  # no maker_rate: makers pay the taker's
    assert maker_fee(100, 0.3, both) == taker_fee(100, 0.3, both) == 1.05
    split = FeeSchedule(rate=0.07, taker_only=False, maker_rate=0.0175)
    assert (maker_fee(100, 0.5, split), taker_fee(100, 0.5, split)) == (0.4375, 1.75)
    refused(ValueError, FeeSchedule, rate=0.07, maker_rate=0.0175)  # taker-only: no maker rate
    assert FeeSchedule(rate=0.07, maker_rate=0.0).maker_rate == 0.0


@check
def a_rate_may_be_a_decimal_or_a_string_and_a_float_product_carries_no_noise() -> None:
    import numpy as np  # a dependency of pmlab; a DataFrame row hands over numpy scalars

    for rate in (Decimal("0.07"), "0.07", " 0.07 ", "7e-2", np.float64(0.07)):
        schedule = FeeSchedule(rate=rate)
        assert schedule.rate == 0.07 and type(schedule.rate) is float
        assert schedule == FeeSchedule(rate=0.07) and taker_fee(100, 0.5, schedule) == 1.75
    up = {"rounding": "up", "rounding_unit": 0.01}
    for multiplier in (3, 5, 6, 7, 0.1, 0.5, 1):  # 0.07 * 3 is 0.21000000000000002 in binary
        noisy = FeeSchedule(rate=0.07 * multiplier, **up)
        exact = FeeSchedule(rate=Decimal("0.07") * Decimal(str(multiplier)), **up)
        assert noisy == exact, multiplier
        assert taker_fee(100, 0.5, noisy) == taker_fee(100, 0.5, exact), multiplier
    assert taker_fee(100, 0.5, FeeSchedule(rate=0.07 * 3, **up)) == 5.25  # 100 x 0.21 x 0.25
    for flag in ("taker_only", "enabled"):  # numpy bools are bools, stored as plain bools
        kwargs = {"rate": 0.05, flag: np.True_}
        assert type(getattr(FeeSchedule(**kwargs), flag)) is bool
    refused(ValueError, taker_fee, Decimal("10"), 0.5, FeeSchedule(rate=0.05))  # a fill is numbers


@check
def nearest_is_the_minimum_fee_rule_and_says_so() -> None:
    cent = FeeSchedule(rate=0.07, rounding="nearest", rounding_unit=0.01)
    assert taker_fee(0.4, 0.5, cent) == 0.0  # raw 0.007: below one unit, not charged
    assert taker_fee(0.6, 0.5, cent) == 0.01  # raw 0.0105
    doc = FeeSchedule.__doc__ or ""
    assert "minimum" in doc and "nearest" in doc, doc
    plain = FeeSchedule(rate=0.07, rounding="up", rounding_unit=0.01)
    assert taker_fee(0.4, 0.5, plain) == 0.01  # "up" is the plain rule; "nearest" is not


@check
def invalid_input_is_refused_with_the_name_of_the_argument() -> None:
    nan, inf = math.nan, math.inf
    for field, bad in [
        ("rate", [-0.01, 1.0, 1.5, nan, inf, True, "abc", "nan", Decimal("NaN"), None]),
        ("exponent", [0, -1, -0.5, nan, inf, False, "two"]),
        ("rebate_rate", [-0.1, 1.5, nan, True]),
        ("rounding_unit", [0, -1e-5, nan, inf, 2.0, True]),
        ("rounding", ["ceil", "UP", "", None, 1]),
        ("rounding_scope", ["trade", "FILL", None]),
        ("taker_only", [1, 0, "yes", None]),
        ("enabled", [1, "yes", None]),
    ]:
        for value in bad:
            kwargs = {"rate": 0.05, **{field: value}}
            message = refused(ValueError, FeeSchedule, **kwargs)
            assert field in message, (field, value, message)
    schedule = FeeSchedule(rate=0.05)
    for fn, args in [
        (taker_fee, (-1, 0.5)),
        (taker_fee, (1, 1.5)),
        (taker_fee, (1, -0.1)),
        (taker_fee, (nan, 0.5)),
        (taker_fee, (1, nan)),
        (taker_fee, (10**400, 0.5)),
        (maker_fee, (-1, 0.5)),
        (stake_fee_fraction, (0.0,)),
        (stake_fee_fraction, (1.5,)),
        (taker_fee_on_stake, (-1, 0.5)),
        (taker_fee_on_stake, (10, 0.0)),
    ]:
        refused(ValueError, fn, *args, schedule)
    assert "shares" in refused(ValueError, taker_fee, nan, 0.5, FeeSchedule(rate=0.0))  # even at 0
    for fn, args in [(taker_fee, (1, 0.5)), (maker_fee, (1, 0.5)), (stake_fee_fraction, (0.5,))]:
        assert "FeeSchedule" in refused(TypeError, fn, *args, {"rate": 0.05})
    refused(ValueError, taker_fee_for_fills, [(1, 0.5), (2,)], schedule)
    refused(ValueError, taker_fee_for_fills, 5, schedule)
    frozen = refused(dataclasses.FrozenInstanceError, setattr, schedule, "rate", 0.5)
    assert "rate" in frozen and schedule.rate == 0.05


@check
def the_stake_helpers_agree_with_the_per_share_fee() -> None:
    schedule = FeeSchedule(rate=0.05)
    for price, fraction in [(0.3, 0.035), (0.5, 0.025), (0.9, 0.005)]:
        assert abs(stake_fee_fraction(price, schedule) - fraction) < 1e-12, price
    assert stake_fee_fraction(1.0, schedule) == 0.0
    general = FeeSchedule(rate=0.25, exponent=2)
    assert abs(stake_fee_fraction(0.3, general) - 0.25 * 0.21**2 / 0.3) < 1e-12
    sports = category_fee_schedule("sports")
    assert taker_fee_on_stake(30, 0.3, sports) == taker_fee(100, 0.3, sports) == 1.05
    # the flat 30 bps the library still charges, against the curve, on 100 USDC at 0.30
    legacy = estimate_fee(100.0, 30.0)
    assert (legacy, taker_fee_on_stake(100, 0.3, category_fee_schedule("weather"))) == (0.3, 0.7)
    assert taker_fee_on_stake(100, 0.3, sports) == 3.5
    assert taker_fee_on_stake(0, 0.3, sports) == 0.0
    off = dataclasses.replace(sports, enabled=False)
    assert taker_fee_on_stake(30, 0.3, off) == 0.0 == taker_fee(100, 0.3, off)


@check
def fills_are_rounded_per_fill_or_per_order() -> None:
    fills = [(0.0012, 0.5), (0.0012, 0.5)]
    assert taker_fee_for_fills(fills, FeeSchedule(rate=0.05)) == 0.00004
    assert taker_fee_for_fills(fills, FeeSchedule(rate=0.05, rounding_scope="order")) == 0.00003
    cent_up = {"rate": 0.07, "rounding": "up", "rounding_unit": 0.01}
    assert taker_fee_for_fills([(3, 0.5), (2, 0.5)], FeeSchedule(**cent_up)) == 0.10
    assert (
        taker_fee_for_fills([(3, 0.5), (2, 0.5)], FeeSchedule(**cent_up, rounding_scope="order"))
        == 0.09
    )
    for scope in ("fill", "order"):
        one = FeeSchedule(rate=0.05, rounding_scope=scope)  # type: ignore[arg-type]
        assert taker_fee_for_fills([(100, 0.3)], one) == taker_fee(100, 0.3, one)
    assert taker_fee_for_fills(iter([(100, 0.3), (50, 0.4)]), FeeSchedule(rate=0.05)) == 1.65
    assert taker_fee_for_fills([], FeeSchedule(rate=0.05)) == 0.0


@check
def a_negative_zero_never_reaches_a_field_or_a_fee() -> None:
    for field in ("rate", "rebate_rate", "maker_rate"):
        for zero in (-0.0, Decimal("-0"), "-0"):
            schedule = FeeSchedule(**{"rate": 0.05, field: zero})
            value = getattr(schedule, field)
            assert value == 0.0 and math.copysign(1.0, value) == 1.0, (field, zero, value)
            assert repr(schedule.to_dict()[field]) == "0.0", (field, zero)  # what the tracker logs
    up = FeeSchedule(rate=0.05, taker_only=False, rounding="up")
    for what, fee in {
        "taker, -0.0 shares": taker_fee(-0.0, 0.5, up),
        "taker, -0.0 price": taker_fee(10, -0.0, up),
        "maker, -0.0 shares": maker_fee(-0.0, 0.5, up),
        "stake, -0.0": taker_fee_on_stake(-0.0, 0.5, up),
        "fills, -0.0 shares": taker_fee_for_fills([(-0.0, 0.5)], up),
    }.items():
        assert fee == 0.0 and math.copysign(1.0, fee) == 1.0, (what, fee)


@check
def a_fee_that_does_not_fit_a_float_is_refused_and_not_returned_as_inf() -> None:
    schedule = FeeSchedule(rate=0.05)
    assert taker_fee(1e308, 0.5, schedule) == 1.25e306  # large and finite: still returned
    order = FeeSchedule(rate=0.05, rounding_scope="order")
    for fn, args in (
        (taker_fee_for_fills, ([(1.7e308, 0.5)] * 100, order)),  # each fill fits, the sum does not
        (taker_fee_for_fills, ([(1.7e308, 0.5)] * 100, schedule)),
        (taker_fee_on_stake, (1e300, 5e-324, FeeSchedule(rate=0.05, exponent=0.5))),
        (stake_fee_fraction, (5e-324, FeeSchedule(rate=0.05, exponent=1e-10))),
    ):
        message = refused(ValueError, fn, *args)
        assert "overflows a float" in message, (fn.__name__, message)


@check
def the_fifteen_digit_rule_holds_for_a_tiny_rate_and_a_float_product_exponent() -> None:
    for tiny in (1e-7, "1e-7", Decimal("1E-7")):
        assert FeeSchedule(rate=tiny).rate == 1e-7
        for rounding in ("nearest", "up", "down"):
            assert taker_fee(1e6, 0.5, FeeSchedule(rate=tiny, rounding=rounding)) == 0.025
    assert taker_fee(100, 0.5, FeeSchedule(rate=1e-7)) == 0.0  # 2.5e-6: below one unit
    product = FeeSchedule(rate=0.07 * 3, exponent=0.1 * 3)  # 0.30000000000000004 in binary
    assert (product.rate, product.exponent) == (0.21, 0.3)
    assert product == FeeSchedule(rate=Decimal("0.21"), exponent=Decimal("0.3"))
    assert taker_fee(100, 0.5, product) == taker_fee(100, 0.5, FeeSchedule(rate=0.21, exponent=0.3))


@check
def float32_is_read_as_its_float64_value_and_the_class_docstring_says_so() -> None:
    import numpy as np  # a dependency of pmlab

    narrow = np.float32(0.07)
    assert FeeSchedule(rate=narrow).rate == 0.0700000002980232 != 0.07
    assert FeeSchedule(rate=float(str(narrow))).rate == 0.07  # the conversion the docs name
    up = {"rounding": "up", "rounding_unit": 0.01}
    assert taker_fee(100, 0.5, FeeSchedule(rate=narrow * 3, **up)) == 5.26  # noisy before pmlab
    assert taker_fee(100, 0.5, FeeSchedule(rate=0.07 * 3, **up)) == 5.25
    doc = FeeSchedule.__doc__ or ""
    for words in ("float32", "float16", "float(str(x))", "type: ignore", "negative zero"):
        assert words in doc, (words, doc)


@check
def a_schedule_round_trips_and_the_flat_model_stays_reachable() -> None:
    schedule = FeeSchedule(rate=0.07, exponent=2.0, taker_only=False, maker_rate=0.01)
    assert FeeSchedule(**schedule.to_dict()) == schedule
    assert hash(schedule) == hash(FeeSchedule(**schedule.to_dict()))
    # wired in since 1.5.0 (#72): the default model charges the curve, and the flat 30 bps of
    # 1.4.0 and before is `fee_model="flat-bps"`, which is also what `estimate_fee` still computes
    assert CostModel().fee_model == "polymarket"
    assert CostModel().taker_bps == 30.0
    assert CostModel(fee_model="flat-bps").fee(1.0) == 0.003
    assert estimate_fee(1.0) == 0.003


def main() -> int:
    failures: list[str] = []
    for name, fn in CHECKS:
        try:
            fn()
        except Exception:
            failures.append(name)
            print(f"FAIL {name}\n{traceback.format_exc(limit=3)}")
        else:
            print(f"ok   {name}")
    if failures:
        print(f"audit fee model: {len(failures)} of {len(CHECKS)} checks FAILED: {failures}")
        return 1
    print(f"audit fee model: OK ({len(CHECKS)} checks, pmlab {pmlab.__version__})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
