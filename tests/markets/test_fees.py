"""What a market's fee fields say (#71), read from replies recorded from live Gamma and CLOB.

``fixtures/fees/`` holds what ``https://gamma-api.polymarket.com`` and
``https://clob.polymarket.com/fee-rate`` answered on 2026-11-26, each Gamma market trimmed to the
fields the code reads (a key the market does not carry is absent, as received):

* ``gamma_geopolitics_fee_free.json``: ``putin-out-before-2027-346``, ``feesEnabled: false``, no
  ``feeSchedule`` key at all, ``feeType: null``; its ``/fee-rate`` is ``{"base_fee": 0}``;
* ``gamma_sports_v3.json``: ``sports_fees_v3``, rate 0.05; ``/fee-rate`` is ``{"base_fee": 1000}``;
* ``gamma_weather.json``: a "highest temperature" bin, ``weather_fees``, rate 0.01;
  ``/fee-rate`` is ``{"base_fee": 1000}``;
* ``gamma_missing_fields.json``: a stale Jinan weather bin: ``conditionId`` is ``""``, no
  ``clobTokenIds``, no ``outcomePrices`` (so there is no token to ask CLOB about);
* ``gamma_by_slug_unknown.json``: ``/markets?slug=`` for a slug that does not exist: ``[]``, HTTP 200.

Also recorded that day: the 222 fee-charging markets of one sample (six fee types, rates 0.01 and
0.05) all had ``takerBaseFee`` and ``makerBaseFee`` 1000, CLOB's ``base_fee`` was 1000 for one market
of each fee type, and the 8 fee-free markets had ``base_fee`` 0.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from pmlab.core.fee_schedule import FeeSchedule, category_fee_schedule
from pmlab.markets.fees import FeeSnapshot, parse_fee_rate, resolve_fee_snapshot

FIXTURES = Path(__file__).parent / "fixtures" / "fees"
SEEN = datetime(2026, 11, 26, 18, 0, tzinfo=UTC)
NAMES = ("geopolitics_fee_free", "sports_v3", "weather")


def gamma(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((FIXTURES / f"gamma_{name}.json").read_text())
    return data


def clob(name: str) -> Any:
    return json.loads((FIXTURES / f"clob_fee_rate_{name}.json").read_text())


def without(market: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {k: v for k, v in market.items() if k not in keys}


def warnings_of(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


class TestParseFeeRate:
    def test_reads_the_recorded_replies(self) -> None:
        assert parse_fee_rate(clob("sports_v3")) == 1000
        assert parse_fee_rate(clob("weather")) == 1000
        assert parse_fee_rate(clob("geopolitics_fee_free")) == 0

    @pytest.mark.parametrize("value", ["1000", 1000.0])
    def test_a_whole_number_in_another_json_type_is_read(self, value: object) -> None:
        assert parse_fee_rate({"base_fee": value}) == 1000

    @pytest.mark.parametrize(
        "reply",
        [
            None,
            [],
            "1000",
            {},
            {"base_fee": None},
            {"base_fee": "ten percent"},
            {"base_fee": -1},
            {"base_fee": 12.5},
            {"base_fee": True},
            {"base_fee": float("nan")},
            {"base_fee": float("inf")},
        ],
    )
    def test_anything_else_is_unknown_not_zero(self, reply: object) -> None:
        # the official clients read a missing base_fee as 0 (`result.get("base_fee") or 0`); here a
        # reply that says nothing must not read as "this market is fee-free"
        assert parse_fee_rate(reply) is None


class TestWhatBaseFeeMeans:
    """``base_fee`` is the legacy v1 order's ``feeRateBps``, in basis points. It is 1000 on every
    fee-charging market and 0 on a fee-free one, whatever the rate: a flag, never the rate.

    The official clients (py-clob-client ``client.py`` lines 450-458 and 476-490, and the v2 TS
    client's ``getFeeRateBps``) pass it into the signed v1 order; the v2 clients and the v2 exchange
    take the rate from the market's ``fd`` / ``feeSchedule`` instead (``py_clob_client_v2/fees.py``
    line 31). The citation is in the pull request of #71 and in ``resolve_fee_snapshot``'s docs.
    """

    def test_it_is_the_same_1000_on_markets_whose_rates_differ_fivefold(self) -> None:
        weather = resolve_fee_snapshot(gamma("weather"), base_fee=1000, seen_at=SEEN)
        sports = resolve_fee_snapshot(gamma("sports_v3"), base_fee=1000, seen_at=SEEN)
        assert (weather.schedule.rate, sports.schedule.rate) == (0.01, 0.05)
        assert weather.base_fee == sports.base_fee == 1000

    def test_the_rate_is_gammas_and_not_base_fee_over_ten_thousand(self) -> None:
        # 1000 bps is 0.10: reading it as the rate would double sports (0.05) and make weather (0.01)
        # ten times too dear
        sports = resolve_fee_snapshot(gamma("sports_v3"), base_fee=1000, seen_at=SEEN)
        assert sports.base_fee / 10_000 == 0.10
        assert sports.schedule.rate == 0.05
        assert sports.source == "gamma"
        weather = resolve_fee_snapshot(gamma("weather"), base_fee=1000, seen_at=SEEN)
        assert weather.schedule.rate == 0.01

    def test_gammas_own_taker_base_fee_is_the_same_flag(self) -> None:
        # the Gamma object carries the number too, equal to CLOB's reply on every recorded market
        assert gamma("sports_v3")["takerBaseFee"] == clob("sports_v3")["base_fee"]
        assert gamma("weather")["makerBaseFee"] == clob("weather")["base_fee"]
        assert "takerBaseFee" not in gamma("geopolitics_fee_free")  # and it is absent when free
        assert clob("geopolitics_fee_free")["base_fee"] == 0

    @pytest.mark.parametrize("name", NAMES)
    def test_both_sources_agree_on_every_recorded_market(
        self, name: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING", logger="pmlab"):
            snap = resolve_fee_snapshot(gamma(name), base_fee=clob(name)["base_fee"], seen_at=SEEN)
        assert warnings_of(caplog) == []
        assert snap.source == "gamma" and not snap.fallback
        assert (snap.schedule.enabled and snap.schedule.rate > 0) == (snap.base_fee > 0)


class TestWhenTheSourcesDisagreeGammaWins:
    def test_clob_says_free_but_gamma_has_a_rate(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING", logger="pmlab"):
            snap = resolve_fee_snapshot(gamma("sports_v3"), base_fee=0, seen_at=SEEN)
        assert snap.schedule == FeeSchedule(rate=0.05, rebate_rate=0.15)
        assert snap.source == "gamma" and snap.base_fee == 0
        (warning,) = warnings_of(caplog)
        assert "cs2-aim1-kamui-2026-11-26-map-handicap-home-1pt5" in warning
        assert "base_fee 0" in warning and "Gamma" in warning

    def test_clob_charges_but_gamma_says_free(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("WARNING", logger="pmlab"):
            snap = resolve_fee_snapshot(gamma("geopolitics_fee_free"), base_fee=1000, seen_at=SEEN)
        assert snap.schedule.enabled is False and snap.schedule.rate == 0.0
        assert snap.source == "gamma" and snap.base_fee == 1000
        (warning,) = warnings_of(caplog)
        assert "putin-out-before-2027-346" in warning and "base_fee 1000" in warning

    def test_a_base_fee_nobody_fetched_is_not_a_disagreement(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING", logger="pmlab"):
            snap = resolve_fee_snapshot(gamma("sports_v3"), seen_at=SEEN)
        assert warnings_of(caplog) == [] and snap.base_fee is None


class TestFallbackToTheCategoryDefault:
    def test_a_market_with_no_fee_fields_gets_the_default_and_says_so(self) -> None:
        market = without(gamma("sports_v3"), "feesEnabled", "feeSchedule", "feeType")
        snap = resolve_fee_snapshot(market, category="sports", seen_at=SEEN)
        assert snap.fallback is True
        assert snap.source == "category_default" and snap.category == "sports"
        assert snap.schedule == category_fee_schedule("sports")
        assert snap.fee_type is None

    def test_a_market_with_a_schedule_never_falls_back(self) -> None:
        # the category is only the answer to "what if Gamma says nothing"
        snap = resolve_fee_snapshot(gamma("weather"), category="sports", seen_at=SEEN)
        assert snap.fallback is False and snap.source == "gamma" and snap.category is None
        assert snap.schedule == FeeSchedule(rate=0.01, rebate_rate=0.25)
        assert snap.fee_type == "weather_fees"

    def test_fees_disabled_is_an_answer_not_a_missing_schedule(self) -> None:
        # no feeSchedule key, but feesEnabled false: the market is fee-free whatever its category
        # would charge (sports is 0.05)
        snap = resolve_fee_snapshot(gamma("geopolitics_fee_free"), category="sports", seen_at=SEEN)
        assert snap.fallback is False and snap.source == "gamma"
        assert snap.schedule.enabled is False and snap.schedule.rate == 0.0

    def test_fees_enabled_without_a_rate_falls_back_and_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        market = without(gamma("sports_v3"), "feeSchedule")
        with caplog.at_level("WARNING", logger="pmlab"):
            snap = resolve_fee_snapshot(market, category="sports", seen_at=SEEN)
        assert snap.fallback and snap.schedule == category_fee_schedule("sports")
        (warning,) = warnings_of(caplog)
        assert "cs2-aim1-kamui-2026-11-26-map-handicap-home-1pt5" in warning

    @pytest.mark.parametrize(
        "broken",
        [
            "oops",
            [],
            {},
            {"exponent": 1, "takerOnly": True},
            {"rate": "abc"},
            {"rate": None},
            {"rate": 7},
            {"rate": -0.01},
            {"rate": 0.05, "exponent": 0},
            {"rate": 0.05, "takerOnly": "yes"},
        ],
    )
    def test_an_unreadable_schedule_is_a_missing_one(
        self, broken: object, caplog: pytest.LogCaptureFixture
    ) -> None:
        market = {**gamma("sports_v3"), "feeSchedule": broken}
        with caplog.at_level("WARNING", logger="pmlab"):
            snap = resolve_fee_snapshot(market, category="sports", seen_at=SEEN)
        assert snap.fallback and snap.schedule == category_fee_schedule("sports")
        assert len(warnings_of(caplog)) == 1

    def test_a_missing_field_inside_the_schedule_takes_the_schedule_default(self) -> None:
        market = {**gamma("sports_v3"), "feeSchedule": {"rate": 0.05}}
        snap = resolve_fee_snapshot(market, seen_at=SEEN)
        assert snap.schedule == FeeSchedule(rate=0.05) and not snap.fallback

    def test_with_nothing_on_gamma_a_clob_zero_alone_means_fee_free(self) -> None:
        market = without(gamma("geopolitics_fee_free"), "feesEnabled")
        snap = resolve_fee_snapshot(market, base_fee=0, category="sports", seen_at=SEEN)
        assert snap.source == "clob" and not snap.fallback
        assert snap.schedule.enabled is False and snap.base_fee == 0

    def test_with_nothing_on_gamma_a_clob_charge_cannot_give_a_rate(self) -> None:
        market = without(gamma("sports_v3"), "feesEnabled", "feeSchedule", "feeType")
        snap = resolve_fee_snapshot(market, base_fee=1000, category="sports", seen_at=SEEN)
        assert snap.fallback and snap.base_fee == 1000
        assert snap.schedule.rate == 0.05  # the category's, not 1000 / 10_000

    def test_a_fallback_needs_a_category_and_the_error_names_the_market(self) -> None:
        market = without(gamma("sports_v3"), "feesEnabled", "feeSchedule", "feeType")
        with pytest.raises(ValueError, match="cs2-aim1-kamui-2026-11-26-map-handicap-home-1pt5"):
            resolve_fee_snapshot(market, seen_at=SEEN)

    def test_an_unknown_category_is_refused(self) -> None:
        market = without(gamma("sports_v3"), "feesEnabled", "feeSchedule", "feeType")
        with pytest.raises(ValueError, match="category"):
            resolve_fee_snapshot(market, category="astrology", seen_at=SEEN)


class TestAMarketWithMissingFields:
    def test_the_recorded_one_reads_through_its_gaps(self) -> None:
        market = gamma("missing_fields")
        assert market["conditionId"] == "" and "clobTokenIds" not in market
        snap = resolve_fee_snapshot(market, seen_at=SEEN)
        assert snap.market_id == "2290078"
        assert snap.slug == "highest-temperature-in-jinan-on-may-20-2026-15corbelow"
        assert snap.schedule == FeeSchedule(rate=0.01, rebate_rate=0.25)
        assert snap.source == "gamma" and snap.base_fee is None

    def test_the_market_id_is_the_id_then_the_condition_id_then_the_slug(self) -> None:
        market = {"id": "7", "conditionId": "0xabc", "slug": "s", "feesEnabled": False}
        assert resolve_fee_snapshot(market, seen_at=SEEN).market_id == "7"
        market["id"] = ""
        assert resolve_fee_snapshot(market, seen_at=SEEN).market_id == "0xabc"
        market["conditionId"] = ""
        assert resolve_fee_snapshot(market, seen_at=SEEN).market_id == "s"
        market["slug"] = ""
        with pytest.raises(ValueError, match="id"):
            resolve_fee_snapshot(market, seen_at=SEEN)

    def test_fees_enabled_that_is_not_a_boolean_is_not_an_answer(self) -> None:
        market = without(gamma("geopolitics_fee_free"), "feeSchedule") | {"feesEnabled": "false"}
        snap = resolve_fee_snapshot(market, category="sports", seen_at=SEEN)
        assert snap.fallback and snap.schedule.rate == 0.05

    def test_a_disabled_flag_beats_a_leftover_schedule(self) -> None:
        market = {**gamma("sports_v3"), "feesEnabled": False}
        snap = resolve_fee_snapshot(market, seen_at=SEEN)
        assert snap.schedule.enabled is False and snap.source == "gamma"

    def test_an_argument_that_is_not_a_market_is_refused(self) -> None:
        with pytest.raises(TypeError, match="market"):
            resolve_fee_snapshot("sports", seen_at=SEEN)  # type: ignore[arg-type]


class TestSeenAt:
    def test_it_defaults_to_now_in_utc(self) -> None:
        before = datetime.now(UTC)
        snap = resolve_fee_snapshot(gamma("weather"))
        assert before <= snap.seen_at <= datetime.now(UTC)

    def test_a_clock_without_a_zone_is_refused(self) -> None:
        with pytest.raises(ValueError, match="timezone"):
            resolve_fee_snapshot(gamma("weather"), seen_at=datetime(2026, 11, 26, 18, 0))

    def test_another_zone_is_converted_to_utc(self) -> None:
        from datetime import timedelta, timezone

        local = datetime(2026, 11, 27, 1, 0, tzinfo=timezone(timedelta(hours=7)))
        snap = resolve_fee_snapshot(gamma("weather"), seen_at=local)
        assert snap.seen_at == SEEN.replace(hour=18) and snap.seen_at.utcoffset() == timedelta(0)
        assert snap.seen_on == date(2026, 11, 26)


class TestFeeSnapshotValue:
    @pytest.mark.parametrize("name", NAMES)
    def test_it_round_trips_through_json(self, name: str) -> None:
        snap = resolve_fee_snapshot(gamma(name), base_fee=clob(name)["base_fee"], seen_at=SEEN)
        wire = json.loads(json.dumps(snap.to_dict()))
        assert FeeSnapshot.from_dict(wire) == snap

    def test_a_fallback_round_trips_with_its_category(self) -> None:
        market = without(gamma("sports_v3"), "feesEnabled", "feeSchedule", "feeType")
        snap = resolve_fee_snapshot(market, category="Sports", seen_at=SEEN)
        assert FeeSnapshot.from_dict(json.loads(json.dumps(snap.to_dict()))) == snap

    def test_the_dict_is_flat_and_names_the_schedule_fields(self) -> None:
        d = resolve_fee_snapshot(gamma("sports_v3"), base_fee=1000, seen_at=SEEN).to_dict()
        assert d["market_id"] and d["source"] == "gamma" and d["fee_type"] == "sports_fees_v3"
        assert d["seen_at"] == "2026-11-26T18:00:00+00:00"
        assert (d["rate"], d["exponent"], d["taker_only"], d["rebate_rate"]) == (
            0.05,
            1.0,
            True,
            0.15,
        )
        assert all(not isinstance(v, dict | list) for v in d.values())

    def test_seen_on_is_the_date_the_schedule_was_seen(self) -> None:
        assert resolve_fee_snapshot(gamma("weather"), seen_at=SEEN).seen_on == date(2026, 11, 26)

    def test_it_is_immutable(self) -> None:
        snap = resolve_fee_snapshot(gamma("weather"), seen_at=SEEN)
        with pytest.raises(AttributeError):
            snap.source = "clob"  # type: ignore[misc]
