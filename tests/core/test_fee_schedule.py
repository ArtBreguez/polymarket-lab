"""The fee model: ``FeeSchedule``, the taker curve, rounding, category defaults.

Two things pin it to what Polymarket publishes, offline:

* ``data/polymarket_fees_docs_2026-11-25.md`` is https://docs.polymarket.com/trading/fees.md as
  downloaded on 2026-11-25, byte for byte. The category table and the 100-share fee tables are
  parsed out of it, so the constants and the arithmetic are held against the page's own text and
  not against numbers retyped here.
* ``_official_client_platform_fee`` transcribes the fee line of the official V2 clients, the only
  place that says where ``exponent`` goes (see the module docstring of ``pmlab.core.fee_schedule``).
"""

from __future__ import annotations

import dataclasses
import math
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import pmlab
from pmlab.core.fee_schedule import (
    POLYMARKET_CATEGORY_FEE_RATES,
    POLYMARKET_FEE_RATES_AS_OF,
    POLYMARKET_FEE_RATES_SOURCE_URL,
    FeeSchedule,
    category_fee_schedule,
    maker_fee,
    stake_fee_fraction,
    taker_fee,
    taker_fee_for_fills,
    taker_fee_on_stake,
)

DOCS_PAGE = Path(__file__).parent / "data" / "polymarket_fees_docs_2026-11-25.md"
UNIT = 1e-5  # "Fees are rounded to 5 decimal places"

# The label of each row of the docs' category table -> the key of our table.
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

# Which categories each "Fee Tables (100 Shares)" tab of the page is about.
TAB_CATEGORIES = {
    "Crypto": ["crypto_btc_5m_15m", "crypto"],
    "Sports": ["sports"],
    "Finance / Politics / Mentions / Tech": [
        "finance",
        "tech",
        "politics_elections",
        "politics",
        "mentions",
        "geopolitics",
    ],
    "Economics / Culture / Weather / Other": ["other", "economics", "culture", "weather"],
}


def _official_client_platform_fee(
    amount: float, price: float, rate: float, exponent: float
) -> float:
    """The platform fee of the official clients, unrounded.

    Transcribed from ``adjust_buy_amount_for_fees`` in Polymarket/py-clob-client-v2
    (``py_clob_client_v2/fees.py``, commit 9c7c9e89cbbc11528daae90dc66ef65c9380cbdf, lines 31 and 35);
    the TypeScript, Rust and ts-sdk clients hold the same two lines.
    """
    platform_fee_rate = rate * (price * (1 - price)) ** exponent
    return (amount / price) * platform_fee_rate


def _docs_text() -> str:
    return DOCS_PAGE.read_text(encoding="utf-8")


def _docs_category_rates() -> dict[str, float]:
    """The ``Taker Fee Rate`` column of the page's category table, by our key."""
    rates: dict[str, float] = {}
    for line in _docs_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 3 and cells[0] in LABEL_TO_KEY:
            assert cells[2] == "0", f"{cells[0]}: the page now charges makers {cells[2]}"
            rates[LABEL_TO_KEY[cells[0]]] = float(cells[1])
    return rates


def _docs_fee_tables() -> dict[str, dict[float | None, dict[float, Decimal]]]:
    """Each tab's table: ``{tab: {rate column (None: the tab names no rate): {price: fee}}}``."""
    tables: dict[str, dict[float | None, dict[float, Decimal]]] = {}
    for block in re.split(r"^> \[tab\] ", _docs_text(), flags=re.MULTILINE)[1:]:
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


def _cents(value: float) -> Decimal:
    return Decimal(repr(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# The docs' own tables
# ---------------------------------------------------------------------------


class TestCategoryDefaultsArePinnedToTheDocs:
    def test_every_row_of_the_docs_table_is_in_the_constant_with_the_same_rate(self) -> None:
        docs = _docs_category_rates()
        assert len(docs) == len(LABEL_TO_KEY), "a row of the page's table was not read"
        assert dict(POLYMARKET_CATEGORY_FEE_RATES) == docs

    def test_the_constant_is_dated_and_points_at_the_page_it_was_read_from(self) -> None:
        assert POLYMARKET_FEE_RATES_SOURCE_URL == "https://docs.polymarket.com/trading/fees"
        assert POLYMARKET_FEE_RATES_AS_OF == "2026-11-25"
        assert DOCS_PAGE.name.endswith(f"{POLYMARKET_FEE_RATES_AS_OF}.md")

    def test_the_constant_cannot_be_edited_in_place(self) -> None:
        with pytest.raises(TypeError):
            POLYMARKET_CATEGORY_FEE_RATES["sports"] = 0.0  # type: ignore[index]

    def test_the_page_still_says_what_the_model_assumes(self) -> None:
        """If one of these sentences changes, the model needs another look, not just the table."""
        text = _docs_text().replace("**", "")  # the page bolds some of these
        assert "fee = C × feeRate × p × (1 - p)" in text
        assert "Makers are never charged fees. Only takers pay fees." in text
        assert "Fees are rounded to 5 decimal places." in text
        assert "The smallest fee charged is 0.00001 USDC" in text
        assert "Geopolitics markets are fee-free" in text
        assert "symmetric around 50% probability" in text

    @pytest.mark.parametrize("key", sorted(LABEL_TO_KEY.values()))
    def test_category_schedule_matches_its_row(self, key: str) -> None:
        schedule = category_fee_schedule(key)
        assert schedule.rate == POLYMARKET_CATEGORY_FEE_RATES[key]
        assert schedule.exponent == 1
        assert schedule.taker_only is True
        assert schedule.enabled is (schedule.rate > 0)
        assert schedule.rounding_unit == UNIT


class TestTheHundredShareTables:
    """100 shares at every price of the page, every category, to the cent."""

    TABLES = _docs_fee_tables()

    def test_the_four_tabs_were_all_read_in_full(self) -> None:
        assert set(self.TABLES) == set(TAB_CATEGORIES)
        for title, by_rate in self.TABLES.items():
            for rate, rows in by_rate.items():
                assert len(rows) == 21, f"{title} {rate}: {len(rows)} price rows"
        # every category has a tab but Combos, which the page gives a rate and no table
        tabbed = {k for ks in TAB_CATEGORIES.values() for k in ks}
        assert tabbed == set(LABEL_TO_KEY.values()) - {"combos"}

    def test_combos_have_their_own_rate_and_the_same_curve_as_every_0_05_category(self) -> None:
        combos, sports = category_fee_schedule("combos"), category_fee_schedule("sports")
        assert combos.rate == sports.rate == 0.05
        assert all(
            taker_fee(100, p / 100, combos) == taker_fee(100, p / 100, sports) for p in range(101)
        )

    @pytest.mark.parametrize(
        ("tab", "key"),
        [(t, k) for t, ks in TAB_CATEGORIES.items() for k in ks if k != "geopolitics"],
    )
    def test_taker_fee_of_100_shares_is_the_printed_fee_at_every_price(
        self, tab: str, key: str
    ) -> None:
        schedule = category_fee_schedule(key)
        by_rate = self.TABLES[tab]
        # the Sports tab prints one unlabelled column: it is the Sports rate
        rows = by_rate[None] if None in by_rate else by_rate[schedule.rate]
        assert len(rows) == 21
        for price, printed in rows.items():
            fee = taker_fee(100, price, schedule)
            assert _cents(fee) == printed, f"{key} @ {price}: {fee} vs docs {printed}"

    def test_geopolitics_is_fee_free_at_every_price(self) -> None:
        schedule = category_fee_schedule("geopolitics")
        assert schedule.rate == 0.0 and schedule.enabled is False
        assert all(taker_fee(100, p / 100, schedule) == 0.0 for p in range(0, 101))

    def test_the_peak_the_page_states_is_1_25_and_0_25(self) -> None:
        assert taker_fee(100, 0.5, category_fee_schedule("sports")) == 1.25
        assert taker_fee(100, 0.5, category_fee_schedule("weather")) == 0.25


# ---------------------------------------------------------------------------
# The curve, and what exponent means
# ---------------------------------------------------------------------------


class TestTakerFeeFormula:
    def test_exponent_one_is_shares_times_rate_times_p_times_one_minus_p(self) -> None:
        # 100 x 0.05 x 0.30 x 0.70 = 1.05 (the docs' row for 30 cents)
        assert taker_fee(100, 0.30, FeeSchedule(rate=0.05)) == 1.05

    def test_a_fee_is_in_usdc_and_scales_with_the_shares(self) -> None:
        schedule = FeeSchedule(rate=0.05)
        assert taker_fee(200, 0.5, schedule) == 2 * taker_fee(100, 0.5, schedule) == 2.5
        assert taker_fee(0, 0.5, schedule) == 0.0

    def test_the_exponent_applies_to_the_whole_p_times_one_minus_p(self) -> None:
        # (0.3 x 0.7) ** 2 = 0.0441; 100 x 0.25 x 0.0441 = 1.1025
        assert taker_fee(100, 0.3, FeeSchedule(rate=0.25, exponent=2)) == 1.1025
        # not p x (1-p)**2 (= 0.147 -> 3.675) and not (p**2) x (1-p) (= 0.063 -> 1.575)
        assert taker_fee(100, 0.3, FeeSchedule(rate=0.25, exponent=2)) not in (3.675, 1.575)

    @pytest.mark.parametrize("exponent", [1, 2, 3, 1.5, 0.5])
    @pytest.mark.parametrize("rate", [0.01, 0.05, 0.25, 0.9])
    @pytest.mark.parametrize("price", [0.01, 0.07, 0.3, 0.5, 0.64, 0.93, 0.99])
    def test_it_is_the_official_clients_fee_up_to_the_rounding_the_clients_do_not_do(
        self, exponent: float, rate: float, price: float
    ) -> None:
        schedule = FeeSchedule(rate=rate, exponent=exponent)
        for amount in (1.0, 10.0, 250.0):
            official = _official_client_platform_fee(amount, price, rate, exponent)
            ours = taker_fee_on_stake(amount, price, schedule)
            assert abs(ours - official) <= UNIT, (amount, ours, official)
            # the same fee, asked in shares: a float share count can sit on the other side of a
            # tie, so the two may differ by one unit (and no more)
            in_shares = taker_fee(amount / price, price, schedule)
            assert abs(ours - in_shares) <= UNIT * (1 + 1e-6)

    def test_a_disabled_schedule_charges_nothing(self) -> None:
        schedule = FeeSchedule(rate=0.05, enabled=False)
        assert taker_fee(100, 0.5, schedule) == 0.0
        assert maker_fee(100, 0.5, schedule) == 0.0
        assert stake_fee_fraction(0.5, schedule) == 0.0
        assert taker_fee_on_stake(50, 0.5, schedule) == 0.0

    def test_a_zero_rate_charges_nothing(self) -> None:
        assert taker_fee(100, 0.5, FeeSchedule(rate=0.0)) == 0.0

    def test_the_extremes_pay_nothing(self) -> None:
        schedule = FeeSchedule(rate=0.07)
        assert taker_fee(1000, 0.0, schedule) == 0.0
        assert taker_fee(1000, 1.0, schedule) == 0.0

    def test_symmetry_around_one_half_on_the_cent_grid_is_exact(self) -> None:
        for exponent in (1, 2):
            schedule = FeeSchedule(rate=0.07, exponent=exponent)
            for cents in range(0, 101):
                p, q = cents / 100, round(1 - cents / 100, 2)
                assert taker_fee(123.456, p, schedule) == taker_fee(123.456, q, schedule), cents

    def test_a_trade_at_30_cents_pays_the_same_dollar_fee_as_one_at_70(self) -> None:
        # the docs' own sentence
        schedule = category_fee_schedule("sports")
        assert taker_fee(100, 0.3, schedule) == taker_fee(100, 0.7, schedule) == 1.05

    def test_the_default_30bps_model_charges_far_less_than_the_curve_does_mid_price(self) -> None:
        """The reason for the epic, in numbers: 30 bps of stake vs the weather rate of the docs."""
        stake, price = 100.0, 0.30
        legacy = stake * 30 / 10_000
        weather = taker_fee_on_stake(stake, price, category_fee_schedule("weather"))
        sports = taker_fee_on_stake(stake, price, category_fee_schedule("sports"))
        assert (legacy, weather, sports) == (0.3, 0.7, 3.5)


# ---------------------------------------------------------------------------
# Rounding
# ---------------------------------------------------------------------------


class TestRounding:
    SCHEDULE = FeeSchedule(rate=0.05)  # 0.05 x 0.25 = 0.0125 per share at 0.50

    def test_the_fee_is_rounded_to_five_decimals(self) -> None:
        # 3.3333 sh x 0.0125 = 0.04166625 -> 0.04167
        assert taker_fee(3.3333, 0.5, self.SCHEDULE) == 0.04167

    @pytest.mark.parametrize(
        ("shares", "expected"),
        [
            (0.0007999, 0.0),  # raw 0.0000099988: smaller than the smallest fee
            (0.0008, 0.00001),  # raw 0.00001 exactly: the smallest fee
            (0.00119999, 0.00001),  # raw 0.0000149999 -> down
            (0.0012, 0.00002),  # raw 0.000015: a tie, half up
            (0.0016, 0.00002),  # raw 0.00002 exactly
            (0.0, 0.0),
        ],
    )
    def test_the_smallest_fee_is_one_unit_and_anything_smaller_is_zero(
        self, shares: float, expected: float
    ) -> None:
        assert taker_fee(shares, 0.5, self.SCHEDULE) == expected

    def test_a_tie_is_decided_on_the_decimal_value_not_on_float_noise(self) -> None:
        # 0.0012 x 0.05 x 0.5 x 0.5 is 0.000015 in decimal; in binary floating point it is
        # 1.4999999999999999e-05, which a naive round() sends down.
        assert 0.0012 * 0.05 * (0.5 * (1 - 0.5)) < 0.000015
        assert taker_fee(0.0012, 0.5, self.SCHEDULE) == 0.00002
        # and 1.0e-05 x 3 = 3.0000000000000004e-05 style noise never decides a round-up
        assert taker_fee(0.0024, 0.5, self.SCHEDULE) == 0.00003

    def test_small_fees_near_the_extremes(self) -> None:
        # 100 shares at 0.0001: raw 100 x 0.05 x 0.0001 x 0.9999 = 0.00049995 -> 0.0005
        assert taker_fee(100, 0.0001, self.SCHEDULE) == 0.0005
        # 0.01 share at 0.01: raw 0.01 x 0.05 x 0.0099 = 0.00000495, smaller than one unit -> zero
        assert taker_fee(0.01, 0.01, self.SCHEDULE) == 0.0

    def test_up_rounds_any_positive_fee_to_the_next_unit(self) -> None:
        cent_up = FeeSchedule(rate=0.07, rounding="up", rounding_unit=0.01)
        assert taker_fee(1, 0.5, cent_up) == 0.02  # raw 0.0175
        assert taker_fee(100, 0.5, cent_up) == 1.75  # raw exactly 1.75, already on the unit
        assert taker_fee(0.0001, 0.5, cent_up) == 0.01  # the smallest positive amount pays a unit
        assert taker_fee(0, 0.5, cent_up) == 0.0
        assert taker_fee(10, 0.5, FeeSchedule(rate=0.07, rounding="up", rounding_unit=0.01)) == 0.18

    def test_up_does_not_round_a_value_that_is_already_on_the_unit_to_the_next_one(self) -> None:
        # 24 x 0.05 x 0.25 is 0.3 in decimal but 0.30000000000000004 in binary floating point,
        # which a naive ceil() would push up a full cent
        cent_up = FeeSchedule(rate=0.05, rounding="up", rounding_unit=0.01)
        assert taker_fee(8, 0.5, cent_up) == 0.1
        assert taker_fee(24, 0.5, cent_up) == 0.3

    def test_down_floors(self) -> None:
        cent_down = FeeSchedule(rate=0.07, rounding="down", rounding_unit=0.01)
        assert taker_fee(1, 0.5, cent_down) == 0.01  # raw 0.0175
        assert taker_fee(0.5, 0.5, cent_down) == 0.0  # raw 0.00875
        assert taker_fee(100, 0.5, cent_down) == 1.75

    def test_nearest_is_polymarkets_minimum_fee_rule_not_the_nearest_unit(self) -> None:
        """``"nearest"`` charges nothing below one unit, as the docs say ("anything smaller
        rounds to zero"); a raw fee of 0.007 on a cent unit is 0, though 0.01 is the nearest cent.
        A venue that wants the plain nearest unit has no value for it yet (only ``"up"`` and
        ``"down"`` are plain), so the rule is documented on the field."""
        cent = FeeSchedule(rate=0.07, rounding="nearest", rounding_unit=0.01)
        assert taker_fee(0.4, 0.5, cent) == 0.0  # raw 0.007: below one cent, not charged
        assert taker_fee(0.5714285714, 0.5, cent) == 0.0  # raw 0.0099999..., still below
        assert taker_fee(0.5714285715, 0.5, cent) == 0.01  # raw above 0.01: charged one cent
        assert taker_fee(0.6, 0.5, cent) == 0.01  # raw 0.0105 rounds to the cent
        doc = FeeSchedule.__doc__ or ""
        assert "minimum" in doc and "nearest" in doc

    def test_nearest_is_half_up_on_a_coarse_unit(self) -> None:
        cent = FeeSchedule(rate=0.07, rounding="nearest", rounding_unit=0.01)
        assert taker_fee(2, 0.5, cent) == 0.04  # raw 0.035: a tie, up
        assert taker_fee(1, 0.5, cent) == 0.02  # raw 0.0175
        assert taker_fee(1.2, 0.5, cent) == 0.02  # raw 0.021
        assert taker_fee(0.5, 0.5, cent) == 0.0  # raw 0.00875: smaller than the unit

    def test_a_unit_that_is_not_a_power_of_ten(self) -> None:
        quarter_cent = FeeSchedule(rate=0.07, rounding="up", rounding_unit=0.0025)
        assert taker_fee(1, 0.5, quarter_cent) == 0.0175  # 7 quarter-cents exactly
        assert taker_fee(1.1, 0.5, quarter_cent) == 0.02  # raw 0.01925 -> 8 quarter-cents

    def test_result_is_a_plain_float_on_the_unit(self) -> None:
        fee = taker_fee(np.float64(100), np.float64(0.25), FeeSchedule(rate=np.float64(0.05)))
        assert type(fee) is float
        assert fee == 0.9375


class TestFillsAndOrders:
    def test_per_fill_rounds_each_fill_then_adds(self) -> None:
        schedule = FeeSchedule(rate=0.05)  # rounding_scope="fill" is the default
        fills = [(0.0012, 0.5), (0.0012, 0.5)]  # each raw 0.000015 -> 0.00002
        assert taker_fee_for_fills(fills, schedule) == 0.00004
        assert taker_fee_for_fills(fills, schedule) == pytest.approx(
            sum(taker_fee(s, p, schedule) for s, p in fills), abs=1e-12
        )

    def test_per_order_adds_the_raw_fees_then_rounds_once(self) -> None:
        schedule = FeeSchedule(rate=0.05, rounding_scope="order")
        fills = [(0.0012, 0.5), (0.0012, 0.5)]  # raw 0.000015 + 0.000015 = 0.00003
        assert taker_fee_for_fills(fills, schedule) == 0.00003

    def test_per_order_up_pays_one_unit_per_order_and_per_fill_pays_one_per_fill(self) -> None:
        # An illustrative cent-up schedule, not a statement about any venue's rule (see #75).
        base: dict[str, Any] = {"rate": 0.07, "rounding": "up", "rounding_unit": 0.01}
        fills = [(3, 0.5), (2, 0.5)]  # raw 0.0525 and 0.035
        per_fill = FeeSchedule(**base, rounding_scope="fill")
        per_order = FeeSchedule(**base, rounding_scope="order")
        assert taker_fee_for_fills(fills, per_fill) == 0.10  # 0.06 + 0.04
        assert taker_fee_for_fills(fills, per_order) == 0.09  # up(0.0875)

    def test_a_single_fill_is_the_same_either_way(self) -> None:
        for scope in ("fill", "order"):
            schedule = FeeSchedule(rate=0.05, rounding_scope=scope)  # type: ignore[arg-type]
            assert taker_fee_for_fills([(100, 0.3)], schedule) == taker_fee(100, 0.3, schedule)

    def test_no_fills_no_fee(self) -> None:
        assert taker_fee_for_fills([], FeeSchedule(rate=0.05)) == 0.0

    def test_accepts_any_iterable_once(self) -> None:
        fills = iter([(100, 0.3), (50, 0.4)])
        assert taker_fee_for_fills(fills, FeeSchedule(rate=0.05)) == 1.65

    @pytest.mark.parametrize(
        "bad", [[(1.0,)], [(1, 2, 3)], [("a", 0.5)], [(1, 1.5)], [(-1, 0.5)], 5]
    )
    def test_malformed_fills_are_refused(self, bad: Any) -> None:
        with pytest.raises(ValueError):
            taker_fee_for_fills(bad, FeeSchedule(rate=0.05))

    def test_a_disabled_schedule_charges_no_fill(self) -> None:
        schedule = FeeSchedule(rate=0.05, enabled=False)
        assert taker_fee_for_fills([(100, 0.5)], schedule) == 0.0


# ---------------------------------------------------------------------------
# Makers
# ---------------------------------------------------------------------------


class TestMakerFee:
    def test_makers_pay_nothing_on_a_taker_only_schedule(self) -> None:
        schedule = FeeSchedule(rate=0.05)
        assert schedule.taker_only is True
        for shares, price in ((100, 0.5), (1, 0.01), (1e6, 0.3), (0, 0.5)):
            assert maker_fee(shares, price, schedule) == 0.0

    def test_both_sides_pay_the_schedule_when_it_is_not_taker_only(self) -> None:
        schedule = FeeSchedule(rate=0.05, taker_only=False)
        assert maker_fee(100, 0.3, schedule) == taker_fee(100, 0.3, schedule) == 1.05

    def test_an_explicit_maker_rate_replaces_the_taker_rate_for_makers_only(self) -> None:
        schedule = FeeSchedule(rate=0.07, taker_only=False, maker_rate=0.0175)
        assert maker_fee(100, 0.5, schedule) == 0.4375
        assert taker_fee(100, 0.5, schedule) == 1.75

    def test_a_maker_rate_needs_a_schedule_that_charges_makers(self) -> None:
        with pytest.raises(ValueError, match="maker_rate"):
            FeeSchedule(rate=0.07, taker_only=True, maker_rate=0.0175)

    def test_maker_rate_zero_is_allowed_on_a_taker_only_schedule(self) -> None:
        assert FeeSchedule(rate=0.07, maker_rate=0.0).maker_rate == 0.0

    def test_the_maker_fee_uses_the_same_rounding(self) -> None:
        schedule = FeeSchedule(
            rate=0.07, taker_only=False, maker_rate=0.0175, rounding="up", rounding_unit=0.01
        )
        assert maker_fee(1, 0.5, schedule) == 0.01  # raw 0.004375

    def test_validates_its_arguments_like_the_taker_side(self) -> None:
        schedule = FeeSchedule(rate=0.05)
        with pytest.raises(ValueError, match="shares"):
            maker_fee(-1, 0.5, schedule)
        with pytest.raises(ValueError, match="price"):
            maker_fee(1, 1.5, schedule)


# ---------------------------------------------------------------------------
# Stake helpers
# ---------------------------------------------------------------------------


class TestStakeHelpers:
    @pytest.mark.parametrize(("price", "expected"), [(0.3, 0.035), (0.5, 0.025), (0.9, 0.005)])
    def test_fee_as_a_fraction_of_stake_is_rate_times_one_minus_p(
        self, price: float, expected: float
    ) -> None:
        # the epic's example: a 0.05-rate market at 0.30 costs 3.5% of the stake, at 0.50 2.5%
        assert stake_fee_fraction(price, FeeSchedule(rate=0.05)) == pytest.approx(expected)

    def test_at_price_one_there_is_nothing_to_pay(self) -> None:
        assert stake_fee_fraction(1.0, FeeSchedule(rate=0.05)) == 0.0

    def test_general_exponent(self) -> None:
        # rate x (p(1-p))**e / p
        got = stake_fee_fraction(0.3, FeeSchedule(rate=0.25, exponent=2))
        assert got == pytest.approx(0.25 * 0.21**2 / 0.3)

    def test_fraction_times_stake_is_the_fee_before_rounding(self) -> None:
        schedule = FeeSchedule(rate=0.05)
        for price in (0.05, 0.3, 0.5, 0.8):
            for stake in (10.0, 123.45, 5000.0):
                exact = stake * stake_fee_fraction(price, schedule)
                assert abs(taker_fee_on_stake(stake, price, schedule) - exact) <= UNIT / 2 + 1e-12

    def test_fee_on_a_stake_buys_stake_over_price_shares(self) -> None:
        schedule = category_fee_schedule("sports")
        assert taker_fee_on_stake(30, 0.3, schedule) == taker_fee(100, 0.3, schedule) == 1.05
        assert taker_fee_on_stake(0, 0.3, schedule) == 0.0

    @pytest.mark.parametrize("price", [0.0, -0.1, 1.01, math.nan, math.inf])
    def test_price_must_be_in_zero_exclusive_to_one(self, price: float) -> None:
        with pytest.raises(ValueError, match="price"):
            stake_fee_fraction(price, FeeSchedule(rate=0.05))
        with pytest.raises(ValueError, match="price"):
            taker_fee_on_stake(10, price, FeeSchedule(rate=0.05))

    @pytest.mark.parametrize("stake", [-1.0, math.nan, math.inf, True, "10", None])
    def test_stake_must_be_a_finite_number_at_least_zero(self, stake: Any) -> None:
        with pytest.raises(ValueError, match="stake"):
            taker_fee_on_stake(stake, 0.5, FeeSchedule(rate=0.05))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestFeeScheduleValidation:
    def test_the_defaults_are_the_issues(self) -> None:
        s = FeeSchedule(rate=0.05)
        assert (s.exponent, s.taker_only, s.rebate_rate, s.enabled) == (1, True, 0.0, True)
        assert (s.maker_rate, s.rounding, s.rounding_unit, s.rounding_scope) == (
            None,
            "nearest",
            UNIT,
            "fill",
        )

    def test_the_gamma_payload_from_the_issue_builds(self) -> None:
        """atp-vallejo-humbert-2026-11-09: feeSchedule {rate 0.05, exponent 1, takerOnly true, ...}."""
        s = FeeSchedule(rate=0.05, exponent=1, taker_only=True, rebate_rate=0.15)
        assert taker_fee(100, 0.5, s) == 1.25

    def test_it_is_a_frozen_value(self) -> None:
        s = FeeSchedule(rate=0.05)
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.rate = 0.07  # type: ignore[misc]
        assert s == FeeSchedule(rate=0.05) and hash(s) == hash(FeeSchedule(rate=0.05))
        assert s != FeeSchedule(rate=0.07)

    def test_to_dict_round_trips(self) -> None:
        s = FeeSchedule(
            rate=0.07,
            exponent=2,
            taker_only=False,
            rebate_rate=0.25,
            enabled=True,
            maker_rate=0.0175,
            rounding="up",
            rounding_unit=0.01,
            rounding_scope="order",
        )
        d = s.to_dict()
        assert FeeSchedule(**d) == s
        assert set(d) == {f.name for f in dataclasses.fields(FeeSchedule)}

    @pytest.mark.parametrize("rate", [-0.01, 1.0, 1.5, math.nan, math.inf, -math.inf])
    def test_rate_must_be_finite_and_in_zero_to_one_exclusive(self, rate: float) -> None:
        with pytest.raises(ValueError, match="rate"):
            FeeSchedule(rate=rate)

    @pytest.mark.parametrize("rate", [0.0, 0.0175, 0.999999])
    def test_rate_bounds_that_are_allowed(self, rate: float) -> None:
        assert FeeSchedule(rate=rate).rate == rate

    @pytest.mark.parametrize("exponent", [0, 0.0, -1, -0.5, math.nan, math.inf])
    def test_exponent_must_be_finite_and_positive(self, exponent: float) -> None:
        with pytest.raises(ValueError, match="exponent"):
            FeeSchedule(rate=0.05, exponent=exponent)

    @pytest.mark.parametrize("rebate", [-0.1, 1.1, math.nan, math.inf])
    def test_rebate_rate_must_be_a_fraction(self, rebate: float) -> None:
        with pytest.raises(ValueError, match="rebate_rate"):
            FeeSchedule(rate=0.05, rebate_rate=rebate)

    @pytest.mark.parametrize("maker_rate", [-0.1, 1.0, math.nan, math.inf])
    def test_maker_rate_must_be_finite_and_in_zero_to_one_exclusive(
        self, maker_rate: float
    ) -> None:
        with pytest.raises(ValueError, match="maker_rate"):
            FeeSchedule(rate=0.05, taker_only=False, maker_rate=maker_rate)

    @pytest.mark.parametrize("unit", [0, 0.0, -1e-5, math.nan, math.inf, 2.0])
    def test_rounding_unit_must_be_a_positive_finite_number_up_to_one(self, unit: float) -> None:
        with pytest.raises(ValueError, match="rounding_unit"):
            FeeSchedule(rate=0.05, rounding_unit=unit)

    @pytest.mark.parametrize("rounding", ["ceil", "UP", "", None, 1])
    def test_rounding_is_nearest_up_or_down(self, rounding: Any) -> None:
        with pytest.raises(ValueError, match="rounding"):
            FeeSchedule(rate=0.05, rounding=rounding)

    @pytest.mark.parametrize("scope", ["trade", "FILL", "", None, 1])
    def test_rounding_scope_is_fill_or_order(self, scope: Any) -> None:
        with pytest.raises(ValueError, match="rounding_scope"):
            FeeSchedule(rate=0.05, rounding_scope=scope)

    @pytest.mark.parametrize("field", ["rate", "exponent", "rebate_rate", "rounding_unit"])
    @pytest.mark.parametrize("bad", [True, False, "abc", None, [0.05], 1j])
    def test_numbers_must_be_real_numbers_not_bools_or_non_numeric_strings(
        self, field: str, bad: Any
    ) -> None:
        kwargs: dict[str, Any] = {"rate": 0.05, field: bad}
        with pytest.raises(ValueError, match=field):
            FeeSchedule(**kwargs)

    @pytest.mark.parametrize("field", ["taker_only", "enabled"])
    @pytest.mark.parametrize("bad", [1, 0, "yes", None])
    def test_flags_must_be_bools(self, field: str, bad: Any) -> None:
        with pytest.raises(ValueError, match=field):
            FeeSchedule(rate=0.05, **{field: bad})

    def test_numpy_scalars_are_numbers(self) -> None:
        s = FeeSchedule(rate=np.float64(0.05), exponent=np.int64(1))
        assert taker_fee(100, 0.5, s) == 1.25

    @pytest.mark.parametrize("flag", ["taker_only", "enabled"])
    def test_numpy_bools_are_bools_and_are_stored_as_plain_bools(self, flag: str) -> None:
        """A DataFrame column of flags yields ``numpy.bool_``: refusing it would break the route
        that builds a schedule from a row."""
        for value in (np.True_, np.False_):
            kwargs: dict[str, Any] = {"rate": 0.05, flag: value}
            if flag == "taker_only" and value is np.False_:
                kwargs["maker_rate"] = 0.01
            s = FeeSchedule(**kwargs)
            assert type(getattr(s, flag)) is bool
            assert getattr(s, flag) is bool(value)
            assert FeeSchedule(**s.to_dict()) == s and hash(s) == hash(FeeSchedule(**s.to_dict()))
        assert FeeSchedule(rate=0.05, taker_only=np.True_) == FeeSchedule(rate=0.05)

    @pytest.mark.parametrize("bad", [np.int64(1), np.float64(1.0), np.array([True])])
    def test_other_numpy_values_are_still_not_flags(self, bad: Any) -> None:
        for flag in ("taker_only", "enabled"):
            with pytest.raises(ValueError, match=flag):
                FeeSchedule(rate=0.05, **{flag: bad})

    @pytest.mark.parametrize(
        "rate",
        [Decimal("0.07"), "0.07", " 0.07 ", "7e-2", Decimal("0.0700")],
    )
    def test_a_decimal_or_a_numeric_string_is_a_rate_and_is_exact(self, rate: Any) -> None:
        s = FeeSchedule(rate=rate)
        assert s.rate == 0.07 and type(s.rate) is float
        assert s == FeeSchedule(rate=0.07) and hash(s) == hash(FeeSchedule(rate=0.07))
        assert taker_fee(100, 0.5, s) == 1.75

    def test_the_other_numeric_fields_take_a_decimal_or_a_string_too(self) -> None:
        s = FeeSchedule(
            rate=Decimal("0.07"),
            exponent="2",
            rebate_rate=Decimal("0.25"),
            maker_rate="0.0175",
            taker_only=False,
            rounding_unit=Decimal("0.01"),
        )
        assert (s.rate, s.exponent, s.rebate_rate, s.maker_rate, s.rounding_unit) == (
            0.07,
            2.0,
            0.25,
            0.0175,
            0.01,
        )

    @pytest.mark.parametrize(
        "bad",
        ["", " ", "abc", "0,07", "nan", "inf", "-inf", "0x10", Decimal("NaN"), Decimal("Infinity")],
    )
    def test_a_string_or_decimal_that_is_not_a_finite_number_is_refused(self, bad: Any) -> None:
        for field in ("rate", "exponent", "rebate_rate", "rounding_unit"):
            with pytest.raises(ValueError, match=field):
                FeeSchedule(**{"rate": 0.05, field: bad})
        with pytest.raises(ValueError, match="maker_rate"):
            FeeSchedule(rate=0.05, taker_only=False, maker_rate=bad)

    def test_a_decimal_or_string_out_of_range_names_the_field(self) -> None:
        with pytest.raises(ValueError, match="rate"):
            FeeSchedule(rate=Decimal("1.5"))
        with pytest.raises(ValueError, match="rate"):
            FeeSchedule(rate="-0.01")

    def test_shares_and_price_are_floats_and_ints_not_decimals_or_strings(self) -> None:
        """Only the schedule's own fields take a Decimal or a string; a fill is numbers."""
        for bad in (Decimal("10"), "10"):
            with pytest.raises(ValueError, match="shares"):
                taker_fee(bad, 0.5, FeeSchedule(rate=0.05))
            with pytest.raises(ValueError, match="price"):
                taker_fee(10, bad, FeeSchedule(rate=0.05))

    @pytest.mark.parametrize("multiplier", [3, 5, 6, 7, 0.1, 0.5, 1])
    def test_a_rate_written_as_a_float_product_carries_no_binary_noise(
        self, multiplier: float
    ) -> None:
        """``0.07 * 3`` is 0.21000000000000002 in binary: it must not push a cent-up fee to 5.26."""
        up = {"rounding": "up", "rounding_unit": 0.01}
        noisy = FeeSchedule(rate=0.07 * multiplier, **up)
        exact = FeeSchedule(rate=Decimal("0.07") * Decimal(str(multiplier)), **up)
        assert noisy == exact
        for shares in (100, 40, 7, 250):
            assert taker_fee(shares, 0.5, noisy) == taker_fee(shares, 0.5, exact), (
                multiplier,
                shares,
            )
        assert taker_fee(100, 0.5, FeeSchedule(rate=0.07 * 3, **up)) == 5.25  # 100 x 0.21 x 0.25

    def test_float_noise_in_the_fill_does_not_decide_a_round_up_either(self) -> None:
        up = FeeSchedule(rate=0.05, rounding="up", rounding_unit=0.01)
        assert 0.1 + 0.2 != 0.3
        assert taker_fee(0.1 + 0.2, 0.5, up) == taker_fee(0.3, 0.5, up)
        assert taker_fee(24.000000000000004, 0.5, up) == 0.3  # 24 x 0.05 x 0.25 is 0.3

    def test_a_float_is_read_as_the_decimal_it_was_written_as(self) -> None:
        """Up to 15 significant digits, the most a binary double always keeps."""
        for text in ("0.05", "0.0175", "0.123456789012345", "1e-05", "0.999999"):
            assert FeeSchedule(rate=float(text), rounding_unit=1e-5).rate == float(text)
        assert FeeSchedule(rate=float("0.123456789012345")).rate == 0.123456789012345


def _sign(value: float) -> float:
    """1.0 for 0.0 and -1.0 for -0.0 (``-0.0 == 0.0``, so ``==`` cannot tell them apart)."""
    return math.copysign(1.0, value)


class TestNegativeZero:
    """``-0.0 == 0.0``, so no comparison or hash was wrong, but a negative zero leaked: into
    ``to_dict()`` (the tracker serialises it as ``-0.0``) and out of a round-up fee."""

    @pytest.mark.parametrize("field", ["rate", "rebate_rate", "maker_rate"])
    @pytest.mark.parametrize("zero", [-0.0, np.float64(-0.0), Decimal("-0"), "-0", "-0.0"])
    def test_a_negative_zero_field_is_stored_as_a_positive_zero(
        self, field: str, zero: Any
    ) -> None:
        schedule = FeeSchedule(**{"rate": 0.05, field: zero})
        assert getattr(schedule, field) == 0.0
        assert _sign(getattr(schedule, field)) == 1.0
        assert repr(schedule.to_dict()[field]) == "0.0"
        assert schedule == FeeSchedule(**{**schedule.to_dict(), field: 0.0})

    @pytest.mark.parametrize("rounding", ["nearest", "up", "down"])
    @pytest.mark.parametrize("scope", ["fill", "order"])
    def test_no_fee_is_a_negative_zero(self, rounding: str, scope: str) -> None:
        schedule = FeeSchedule(
            rate=0.05,
            taker_only=False,
            rounding=rounding,
            rounding_scope=scope,  # type: ignore[arg-type]
        )
        fees = {
            "taker, -0.0 shares": taker_fee(-0.0, 0.5, schedule),
            "taker, -0.0 price": taker_fee(10, -0.0, schedule),
            "taker, price 1": taker_fee(10, 1.0, schedule),
            "maker, -0.0 shares": maker_fee(-0.0, 0.5, schedule),
            "maker, -0.0 price": maker_fee(10, -0.0, schedule),
            "stake, -0.0": taker_fee_on_stake(-0.0, 0.5, schedule),
            "fills, -0.0 shares": taker_fee_for_fills([(-0.0, 0.5)], schedule),
            "fills, -0.0 price": taker_fee_for_fills([(10, -0.0)], schedule),
            "fraction at price 1": stake_fee_fraction(1.0, schedule),
        }
        for what, fee in fees.items():
            assert fee == 0.0 and _sign(fee) == 1.0, (what, fee)

    def test_a_negative_zero_rate_charges_nothing_like_a_zero_rate(self) -> None:
        zero, negative = FeeSchedule(rate=0.0, rounding="up"), FeeSchedule(rate=-0.0, rounding="up")
        assert negative == zero and hash(negative) == hash(zero)
        assert repr(taker_fee(100, 0.5, negative)) == repr(taker_fee(100, 0.5, zero)) == "0.0"


class TestAFeeThatDoesNotFitAFloat:
    """The inputs are finite; at magnitudes no market produces the result may not be. It is
    refused, not returned as ``inf``."""

    def test_a_stake_fraction_that_overflows_is_refused(self) -> None:
        with pytest.raises(ValueError, match="overflows a float"):
            stake_fee_fraction(5e-324, FeeSchedule(rate=0.05, exponent=1e-10))

    @pytest.mark.parametrize("scope", ["fill", "order"])
    def test_an_order_total_that_overflows_is_refused(self, scope: Any) -> None:
        """Each fill (1.7e308 shares) is a finite 2.1e306; a hundred of them are not."""
        schedule = FeeSchedule(rate=0.05, rounding_scope=scope)
        assert taker_fee(1.7e308, 0.5, schedule) == pytest.approx(2.125e306)
        with pytest.raises(ValueError, match="overflows a float"):
            taker_fee_for_fills([(1.7e308, 0.5)] * 100, schedule)

    def test_a_stake_fee_that_overflows_is_refused(self) -> None:
        """A stake of 1e300 at a price of 5e-324 buys 2e623 shares: the fee is not a float."""
        with pytest.raises(ValueError, match="overflows a float"):
            taker_fee_on_stake(1e300, 5e-324, FeeSchedule(rate=0.05, exponent=0.5))

    def test_a_single_fill_that_overflows_is_refused_on_both_sides(self) -> None:
        """The largest float as shares, a rate of 0.999999999999999 and a curve of 1: the product
        of the 15-digit reading of the shares and the rate is above the largest float."""
        schedule = FeeSchedule(rate=0.999999999999999, exponent=1e-300, taker_only=False)
        with pytest.raises(ValueError, match="overflows a float"):
            taker_fee(sys.float_info.max, 0.5, schedule)
        with pytest.raises(ValueError, match="overflows a float"):
            maker_fee(sys.float_info.max, 0.5, schedule)

    def test_large_but_finite_results_are_still_returned(self) -> None:
        schedule = FeeSchedule(rate=0.05)
        assert taker_fee(1e308, 0.5, schedule) == 1.25e306
        assert taker_fee_for_fills([(1e308, 0.5)] * 10, schedule) == pytest.approx(1.25e307)
        assert taker_fee_on_stake(1e300, 0.5, schedule) == pytest.approx(2.5e298)


class TestTheContractThatGetsFrozen:
    """Rate 1e-7, a float product exponent, float32: the inputs the 15-digit rule is held to."""

    @pytest.mark.parametrize(
        "spelling", [1e-7, "1e-7", Decimal("1E-7"), np.float64(1e-7), "0.0000001", " 1E-7 "]
    )
    def test_a_rate_of_one_in_ten_million_is_kept_and_charges(self, spelling: Any) -> None:
        schedule = FeeSchedule(rate=spelling)
        assert schedule.rate == 1e-7 and schedule == FeeSchedule(rate=1e-7)
        assert FeeSchedule(**schedule.to_dict()) == schedule
        for rounding in ("nearest", "up", "down"):
            priced = FeeSchedule(rate=spelling, rounding=rounding)  # type: ignore[arg-type]
            assert taker_fee(1e6, 0.5, priced) == 0.025  # 1e6 x 1e-7 x 0.25
        assert taker_fee(1e8, 0.5, FeeSchedule(rate=1e-9)) == 0.025
        assert taker_fee(1e6, 0.5, FeeSchedule(rate=1e-9)) == 0.00025

    def test_a_tiny_raw_fee_is_not_charged_unless_it_rounds_up(self) -> None:
        raw = 100 * 1e-7 * 0.25  # 2.5e-6, below one unit
        assert taker_fee(100, 0.5, FeeSchedule(rate=1e-7)) == 0.0
        assert taker_fee(100, 0.5, FeeSchedule(rate=1e-7, rounding="up")) == UNIT
        assert raw < UNIT

    def test_a_float_product_exponent_is_read_as_the_decimal_it_was_written_as(self) -> None:
        """``0.1 * 3`` is 0.30000000000000004: the exponent is 0.3, as for the rate."""
        assert 0.1 * 3 != 0.3
        product = FeeSchedule(rate=0.07 * 3, exponent=0.1 * 3)
        written = FeeSchedule(rate=0.21, exponent=0.3)
        assert product == written and hash(product) == hash(written)
        assert (product.rate, product.exponent) == (0.21, 0.3)
        assert FeeSchedule(rate=0.05, exponent=Decimal("0.3")).exponent == 0.3
        assert FeeSchedule(**product.to_dict()) == product
        # 100 x 0.21 x 0.25 ** 0.3 = 13.8548...; the power is a float read to 15 digits
        expected = Decimal(100 * 0.21 * 0.25**0.3).quantize(Decimal("0.00001"), ROUND_HALF_UP)
        assert taker_fee(100, 0.5, product) == taker_fee(100, 0.5, written) == float(expected)

    def test_float32_and_float16_are_read_as_their_float64_value_not_as_what_they_print(
        self,
    ) -> None:
        """The 15-digit rule drops binary noise below 1e-15. A float32 is exact to 1e-8 only, so
        its value is kept, not the shorter decimal it prints as. Documented in ``FeeSchedule``:
        convert with ``float(str(x))`` or compute in float64."""
        for narrow in (np.float32(0.07), np.float16(0.07)):
            assert FeeSchedule(rate=narrow).rate != 0.07
            assert FeeSchedule(rate=narrow).rate == float(f"{float(narrow):.15g}")
            assert str(narrow) == "0.07"
            assert FeeSchedule(rate=float(str(narrow))).rate == 0.07  # the documented conversion
        assert FeeSchedule(rate=np.float32(0.07)).rate == 0.0700000002980232
        assert FeeSchedule(rate=np.float16(0.07)).rate == 0.07000732421875
        assert FeeSchedule(rate=np.float32(0.5)).rate == 0.5  # exact in float32: nothing to drop

    def test_a_float32_product_is_already_noisy_so_it_charges_as_written_in_float32(self) -> None:
        """Not the 5.25 of ``0.07 * 3``: ``np.float32(0.07) * 3`` is 0.21000000834..., and a
        cent-up fee on 100 shares at 0.50 is 5.26. Compute the rate in float64."""
        up: dict[str, Any] = {"rounding": "up", "rounding_unit": 0.01}
        assert taker_fee(100, 0.5, FeeSchedule(rate=np.float32(0.07) * 3, **up)) == 5.26
        assert taker_fee(100, 0.5, FeeSchedule(rate=float(np.float32(0.07)) * 3, **up)) == 5.26
        assert taker_fee(100, 0.5, FeeSchedule(rate=float(str(np.float32(0.07))) * 3, **up)) == 5.25
        assert taker_fee(100, 0.5, FeeSchedule(rate=0.07 * 3, **up)) == 5.25


class TestFunctionArgumentValidation:
    SCHEDULE = FeeSchedule(rate=0.05)

    @pytest.mark.parametrize("shares", [-1, -1e-9, math.nan, math.inf, -math.inf])
    def test_shares_must_be_finite_and_not_negative(self, shares: float) -> None:
        with pytest.raises(ValueError, match="shares"):
            taker_fee(shares, 0.5, self.SCHEDULE)

    @pytest.mark.parametrize("price", [-0.01, 1.01, 2, math.nan, math.inf, -math.inf])
    def test_price_must_be_a_probability(self, price: float) -> None:
        with pytest.raises(ValueError, match="price"):
            taker_fee(10, price, self.SCHEDULE)

    def test_an_integer_too_large_for_a_float_is_refused_not_an_overflow_error(self) -> None:
        with pytest.raises(ValueError, match="shares"):
            taker_fee(10**400, 0.5, self.SCHEDULE)

    @pytest.mark.parametrize("bad", [True, "10", None, [1], 1j])
    def test_shares_and_price_must_be_real_numbers(self, bad: Any) -> None:
        with pytest.raises(ValueError, match="shares"):
            taker_fee(bad, 0.5, self.SCHEDULE)
        with pytest.raises(ValueError, match="price"):
            taker_fee(10, bad, self.SCHEDULE)

    @pytest.mark.parametrize("bad", [None, 0.05, "sports", {"rate": 0.05}])
    def test_the_schedule_must_be_a_fee_schedule(self, bad: Any) -> None:
        for fn in (taker_fee, maker_fee):
            with pytest.raises(TypeError, match="FeeSchedule"):
                fn(10, 0.5, bad)
        with pytest.raises(TypeError, match="FeeSchedule"):
            stake_fee_fraction(0.5, bad)
        with pytest.raises(TypeError, match="FeeSchedule"):
            taker_fee_on_stake(10, 0.5, bad)
        with pytest.raises(TypeError, match="FeeSchedule"):
            taker_fee_for_fills([(1, 0.5)], bad)

    def test_a_bad_argument_is_refused_even_when_the_fee_would_be_zero(self) -> None:
        with pytest.raises(ValueError, match="shares"):
            taker_fee(math.nan, 0.5, FeeSchedule(rate=0.0))
        with pytest.raises(ValueError, match="price"):
            taker_fee(1, math.nan, FeeSchedule(rate=0.05, enabled=False))


class TestCategoryLookup:
    @pytest.mark.parametrize(
        ("name", "key"),
        [
            ("sports", "sports"),
            ("Sports", "sports"),
            ("  WEATHER ", "weather"),
            ("Other / General", "other"),
            ("other_general", "other"),
            ("politics-elections", "politics_elections"),
            ("Politics elections", "politics_elections"),
        ],
    )
    def test_lookup_ignores_case_spacing_and_the_docs_label_for_other(
        self, name: str, key: str
    ) -> None:
        assert category_fee_schedule(name) == category_fee_schedule(key)

    @pytest.mark.parametrize("name", ["", "   ", "nope", "politics_excluding", "sport"])
    def test_an_unknown_category_names_the_known_ones(self, name: str) -> None:
        with pytest.raises(ValueError, match="sports") as err:
            category_fee_schedule(name)
        assert "weather" in str(err.value) and POLYMARKET_FEE_RATES_SOURCE_URL in str(err.value)

    @pytest.mark.parametrize("bad", [None, 5, ["sports"]])
    def test_the_category_must_be_a_string(self, bad: Any) -> None:
        with pytest.raises(ValueError, match="category"):
            category_fee_schedule(bad)


# ---------------------------------------------------------------------------
# Property tests (seeded; the project has no hypothesis dependency)
# ---------------------------------------------------------------------------


class TestProperties:
    N = 3000

    @staticmethod
    def _draws(seed: int, n: int) -> list[tuple[float, float, float, float]]:
        rng = np.random.default_rng(seed)
        shares = 10 ** rng.uniform(-3, 6, n)
        price = np.round(rng.uniform(0, 1, n), 4)
        rate = rng.choice([0.0, 0.01, 0.0175, 0.05, 0.07, 0.25, 0.5, 0.99], n)
        exponent = rng.choice([1.0, 1.0, 1.0, 2.0, 3.0, 1.5], n)
        return list(
            zip(shares.tolist(), price.tolist(), rate.tolist(), exponent.tolist(), strict=True)
        )

    def test_a_fee_is_never_negative_and_never_above_shares_times_rate_over_four(self) -> None:
        """max of p(1-p) is 1/4, so (p(1-p))**e <= 1/4 for e >= 1; rounding adds at most half a unit."""
        for shares, price, rate, exponent in self._draws(70, self.N):
            fee = taker_fee(shares, price, FeeSchedule(rate=rate, exponent=exponent))
            assert fee >= 0.0
            bound = shares * rate * 0.25**exponent
            assert fee <= bound + UNIT / 2 + 1e-9 * max(1.0, bound), (shares, price, rate)
            if exponent >= 1:
                assert fee <= shares * rate / 4 + UNIT / 2 + 1e-9 * max(1.0, bound)

    def test_a_fee_is_on_the_unit_and_within_one_unit_of_the_raw_curve(self) -> None:
        for shares, price, rate, exponent in self._draws(71, self.N):
            fee = taker_fee(shares, price, FeeSchedule(rate=rate, exponent=exponent))
            raw = shares * rate * (price * (1 - price)) ** exponent
            assert abs(fee - raw) <= UNIT + 1e-9 * max(1.0, raw)
            assert Decimal(repr(fee)) % Decimal("0.00001") == 0, fee

    def test_a_fee_does_not_fall_when_shares_or_rate_grow(self) -> None:
        for shares, price, rate, exponent in self._draws(72, 1000):
            a = taker_fee(shares, price, FeeSchedule(rate=rate, exponent=exponent))
            b = taker_fee(shares * 1.5, price, FeeSchedule(rate=rate, exponent=exponent))
            c = taker_fee(
                shares, price, FeeSchedule(rate=min(rate * 1.5, 0.999), exponent=exponent)
            )
            assert b >= a and c >= a

    def test_symmetry_around_one_half(self) -> None:
        for shares, price, rate, exponent in self._draws(73, 1000):
            schedule = FeeSchedule(rate=rate, exponent=exponent)
            mirrored = round(1 - price, 4)
            assert taker_fee(shares, price, schedule) == taker_fee(shares, mirrored, schedule)

    def test_it_peaks_at_one_half(self) -> None:
        for _, _, rate, exponent in self._draws(74, 200):
            schedule = FeeSchedule(rate=rate, exponent=exponent)
            peak = taker_fee(1000, 0.5, schedule)
            assert all(taker_fee(1000, p / 100, schedule) <= peak for p in range(101))

    def test_the_stake_fraction_is_the_fee_per_unit_of_stake(self) -> None:
        for _, price, rate, exponent in self._draws(75, 500):
            if price == 0.0:
                continue
            schedule = FeeSchedule(rate=rate, exponent=exponent)
            stake = 1000.0
            raw = stake * stake_fee_fraction(price, schedule)
            assert abs(taker_fee_on_stake(stake, price, schedule) - raw) <= UNIT


class TestPublicSurface:
    NAMES = (
        "FeeSchedule",
        "taker_fee",
        "maker_fee",
        "taker_fee_for_fills",
        "stake_fee_fraction",
        "taker_fee_on_stake",
    )
    # Polymarket's table and the lookup over it are module-internal in 1.2.0; they are re-exported
    # when the lookup from a market's tags is settled (#71), so that the keys can still change.
    INTERNAL = (
        "category_fee_schedule",
        "POLYMARKET_CATEGORY_FEE_RATES",
        "POLYMARKET_FEE_RATES_AS_OF",
        "POLYMARKET_FEE_RATES_SOURCE_URL",
    )

    def test_the_names_are_public_and_the_same_objects(self) -> None:
        import pmlab.core.fee_schedule as module

        assert set(module.__all__) == set(self.NAMES)
        for name in self.NAMES:
            assert name in pmlab.__all__, name
            assert getattr(pmlab, name) is getattr(module, name)

    def test_the_category_table_and_its_lookup_are_not_public_yet(self) -> None:
        import pmlab.core.fee_schedule as module

        for name in self.INTERNAL:
            assert name not in pmlab.__all__, name
            assert not hasattr(pmlab, name), name
            assert name not in module.__all__, name
            assert hasattr(module, name), name  # still there, for the tests and for #71
