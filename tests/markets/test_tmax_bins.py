"""The per-bin Yes/No temperature markets live Gamma lists, read into one spec per city and day (#87).

Live Gamma has no market with temperature bins as its outcomes. It lists one Yes/No market per bin
(``Will the highest temperature in Warsaw be -4°C or below on November 26?``) and ties the eleven
of a city and day together as an *event*. ``pmlab.markets.tmax_bins`` reads the question of each
into a bin, and groups the bins of one city and day.

``fixtures/per_bin_markets.json`` is 32 markets recorded from
``/markets/keyset?tag_id=103040&active=true&closed=false`` on 2026-11-25, trimmed to the fields read
(submitter and resolver addresses, images and all but the first line of each description are
dropped; ids, questions, token ids and prices are as received): Atlanta on November 26 in °F, Warsaw
on November 26 in °C with negative bins, Panama City on November 23 with 3 of its 11 bins still
active, Atlanta on May 15 with 1, Jinan on May 20 with 4 bins that have no token ids (a stale
event), and 2 *lowest* temperature markets, which are not this family.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import pytest

from pmlab.core.market_spec import OutcomeBin
from pmlab.markets.tmax_bins import (
    LOWER_INCLUSIVE,
    UPPER_INCLUSIVE,
    PerBinGrouper,
    is_per_bin_question,
    parse_bin_question,
)

FIXTURE = Path(__file__).parent / "fixtures" / "per_bin_markets.json"


def recorded() -> list[dict[str, Any]]:
    return json.loads(FIXTURE.read_text())["markets"]


def market(
    i: int,
    city: str = "Austin",
    bin_text: str = "30°C",
    day: str = "November 27",
    *,
    event: str | None = "900",
    end: str | None = "2026-11-27T12:00:00Z",
    outcomes: str = '["Yes", "No"]',
    tokens: str | None = None,
    prices: str | None = '["0.25", "0.75"]',
    slug: str | None = None,
) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "id": str(i),
        "question": f"Will the highest temperature in {city} be {bin_text} on {day}?",
        "outcomes": outcomes,
        "clobTokenIds": tokens if tokens is not None else f'["yes{i}", "no{i}"]',
        "outcomePrices": prices,
        "active": True,
    }
    if end is not None:
        raw["endDate"] = end
    if event is not None:
        raw["events"] = [{"id": event, "slug": slug or f"highest-temperature-event-{event}"}]
    return raw


def warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]


def group_of(markets: list[dict[str, Any]]):
    grouper = PerBinGrouper()
    for raw in markets:
        grouper.offer(raw)
    return grouper


# ---------------------------------------------------------------------------------------------
# One question, one bin
# ---------------------------------------------------------------------------------------------
class TestReadingAQuestion:
    @pytest.mark.parametrize(
        ("bin_text", "label", "lower", "upper"),
        [
            ("33°C", "33°C", 32.5, 33.5),
            ("-3°C", "-3°C", -3.5, -2.5),
            ("0°C", "0°C", -0.5, 0.5),
            ("-4°C or below", "-4°C or below", None, -3.5),
            ("43°F or below", "43°F or below", None, 43.5),
            ("40°C or higher", "40°C or higher", 39.5, None),
            ("62°F or higher", "62°F or higher", 61.5, None),
            ("between 88-89°F", "88-89°F", 87.5, 89.5),
            ("between -5--4°F", "-5--4°F", -5.5, -3.5),
            ("between -1-2°C", "-1-2°C", -1.5, 2.5),
            ("between 5 to 6°C", "5-6°C", 4.5, 6.5),
            ("7°C or above", "7°C or higher", 6.5, None),
            ("7°C or lower", "7°C or below", None, 7.5),
        ],
    )
    def test_the_bin_the_question_asks_about(self, bin_text, label, lower, upper) -> None:
        q = parse_bin_question(
            f"Will the highest temperature in Panama City be {bin_text} on May 3?"
        )
        assert (q.label, q.lower, q.upper) == (label, lower, upper)
        assert q.unit == ("C" if "°C" in bin_text else "F")

    def test_every_bin_is_half_open_lower_end_in_upper_end_out(self) -> None:
        assert (LOWER_INCLUSIVE, UPPER_INCLUSIVE) == (True, False)

    def test_a_bin_covers_the_whole_degrees_that_round_into_it(self) -> None:
        """Gamma resolves on whole degrees, so a bin is half a degree wider than its text on each
        side, and is half-open: 33.5 belongs to the 34 bin, once."""
        q = parse_bin_question("Will the highest temperature in Rome be 33°C on May 3?")
        bin_ = OutcomeBin(
            label=q.label,
            lower=q.lower,
            upper=q.upper,
            lower_inclusive=LOWER_INCLUSIVE,
            upper_inclusive=UPPER_INCLUSIVE,
        )
        assert [bin_.contains(t) for t in (32.4, 32.5, 33.0, 33.49, 33.5)] == [
            False,
            True,
            True,
            True,
            False,
        ]

    def test_city_date_and_unit_come_from_the_question(self) -> None:
        q = parse_bin_question(
            "Will the highest temperature in Seoul (Incheon) be 7°C on November 27?"
        )
        assert (q.city, q.month, q.day, q.year, q.unit) == ("Seoul (Incheon)", 11, 27, None, "C")

    def test_an_abbreviated_month_and_an_explicit_year(self) -> None:
        q = parse_bin_question("Will the highest temperature in Rome be 7°F on Nov 3, 2027?")
        assert (q.month, q.day, q.year) == (11, 3, 2027)

    def test_a_unicode_minus_is_a_minus(self) -> None:
        q = parse_bin_question("Will the highest temperature in Oslo be −3°C on Jan 3?")
        assert q.label == "-3°C" and (q.lower, q.upper) == (-3.5, -2.5)

    @pytest.mark.parametrize(
        "question",
        [
            "Will the highest temperature in Rome be hot on May 3?",
            "Will the highest temperature in Rome exceed 30°C on May 3?",
            "Will the highest temperature in Rome be 30.5°C on May 3?",
            "Will the highest temperature in Rome be 30 on May 3?",
            "Will the highest temperature in Rome be between 31-30°F on May 3?",
            "Will the highest temperature in Rome be between 30°C-31°F on May 3?",
            "Will the highest temperature in Rome be 30°C or higher or below on May 3?",
            "Will the highest temperature in Rome be 30°K on May 3?",
            "Will the highest temperature in Rome be 30°C on Smarch 3?",
            "Will the highest temperature in Rome be 30°C on May?",
            "Will the highest temperature in Rome be 30°C on May 3",
            "Will the highest temperature in Rome be 30°C on May 3? (updated)",
            "Will the highest temperature in  be 30°C on May 3?",
        ],
    )
    def test_what_it_cannot_read_is_refused(self, question) -> None:
        with pytest.raises(ValueError, match="bin|date|city|question"):
            parse_bin_question(question)

    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("Will the highest temperature in Rome be 30°C on May 3?", True),
            ("will the highest temperature in Rome exceed 30°C on May 3?", True),
            ("Will the lowest temperature in Rome be 30°C on May 3?", False),
            ("Highest temperature in Rome on May 3?", False),
            ("Will it rain in Rome on May 3?", False),
            ("", False),
            (None, False),
            (42, False),
        ],
    )
    def test_which_questions_are_this_family_at_all(self, question, expected) -> None:
        assert is_per_bin_question(question) is expected


# ---------------------------------------------------------------------------------------------
# The bins of one city and day
# ---------------------------------------------------------------------------------------------
class TestRecordedEvents:
    def test_the_lowest_temperature_markets_are_not_this_family(self) -> None:
        grouper = PerBinGrouper()
        taken = [grouper.offer(m) for m in recorded()]
        lowest = [m["id"] for m in recorded() if "lowest temperature" in m["question"]]
        assert len(lowest) == 2
        assert [m["id"] for m, t in zip(recorded(), taken, strict=True) if not t] == lowest

    def test_every_market_that_has_what_it_needs_is_in_a_group(self) -> None:
        grouper = group_of(recorded())
        groups = grouper.groups()
        assert [g.market_id for g in groups] == ["479718", "1436284", "1451212", "1451693"]
        assert [len(g.bins) for g in groups] == [1, 3, 11, 11]
        assert grouper.refused and {reason for _, reason in grouper.refused} == {"no Yes token id"}
        assert [i for i, _ in grouper.refused] == ["2290078", "2290079", "2290080", "2290081"]

    def test_a_full_fahrenheit_event(self) -> None:
        by_id = {g.market_id: g for g in group_of(recorded()).groups()}
        atlanta = by_id["1451693"]
        assert atlanta.question == "Highest temperature in Atlanta on November 26?"
        assert (atlanta.city, atlanta.target_date, atlanta.unit) == (
            "Atlanta",
            "2026-11-26",
            "F",
        )
        assert atlanta.slug == "highest-temperature-in-atlanta-on-november-26-2026"
        assert atlanta.close_time == "2026-11-26T12:00:00Z" and atlanta.active
        assert [b.label for b in atlanta.bins] == [
            "43°F or below",
            *[f"{n}-{n + 1}°F" for n in range(44, 62, 2)],
            "62°F or higher",
        ]
        assert (atlanta.bins[0].lower, atlanta.bins[0].upper) == (None, 43.5)
        assert (atlanta.bins[1].lower, atlanta.bins[1].upper) == (43.5, 45.5)
        assert (atlanta.bins[-1].lower, atlanta.bins[-1].upper) == (61.5, None)
        assert atlanta.complete

    def test_each_bin_carries_its_own_yes_token_market_and_price(self) -> None:
        by_id = {g.market_id: g for g in group_of(recorded()).groups()}
        raw = {m["id"]: m for m in recorded()}
        for bin_ in by_id["1451693"].bins:
            m = raw[bin_.market_id]
            assert bin_.token_id == json.loads(m["clobTokenIds"])[0]  # outcomes[0] is "Yes"
            assert bin_.price == float(json.loads(m["outcomePrices"])[0])
            assert bin_.label == m["groupItemTitle"]  # the label is the one the page shows

    def test_negative_celsius_bins_run_from_an_open_bottom_to_an_open_top(self) -> None:
        warsaw = {g.market_id: g for g in group_of(recorded()).groups()}["1451212"]
        assert warsaw.unit == "C" and warsaw.target_date == "2026-11-26" and warsaw.complete
        assert warsaw.bins[0].label == "-6°C or below" and warsaw.bins[0].lower is None
        assert warsaw.bins[1].label == "-5°C" and (warsaw.bins[1].lower, warsaw.bins[1].upper) == (
            -5.5,
            -4.5,
        )
        assert warsaw.bins[-1].upper is None and warsaw.bins[-1].label.endswith("°C or higher")

    def test_the_bins_of_a_full_event_tile_the_line(self) -> None:
        """Whatever the day's maximum, exactly one bin holds it (no gap, no overlap)."""
        for g in group_of(recorded()).groups():
            if not g.complete:
                continue
            bins = [
                OutcomeBin(
                    label=b.label,
                    lower=b.lower,
                    upper=b.upper,
                    lower_inclusive=LOWER_INCLUSIVE,
                    upper_inclusive=UPPER_INCLUSIVE,
                )
                for b in g.bins
            ]
            for tenths in range(-300, 1300):
                t = tenths / 10
                assert sum(b.contains(t) for b in bins) == 1, (g.market_id, t)

    def test_a_partial_event_is_returned_and_says_it_is_partial(self) -> None:
        by_id = {g.market_id: g for g in group_of(recorded()).groups()}
        panama, lone = by_id["1436284"], by_id["479718"]
        assert [b.label for b in panama.bins] == ["32°C", "34°C", "35°C"] and not panama.complete
        assert panama.target_date == "2026-11-23"
        assert [b.label for b in lone.bins] == ["88-89°F"] and not lone.complete
        assert lone.target_date == "2026-05-15"

    def test_a_stale_event_without_tokens_is_refused_with_one_warning(self, caplog) -> None:
        grouper = group_of(recorded())
        with caplog.at_level(logging.WARNING, logger="pmlab"):
            grouper.warn_refused("discover_tmax_markets")
        (said,) = warnings(caplog)
        assert said.startswith("discover_tmax_markets:") and "4" in said
        assert "no Yes token id" in said and "2290078" in said

    def test_nothing_refused_is_nothing_said(self, caplog) -> None:
        grouper = group_of([m for m in recorded() if m["events"][0]["id"] == "1451693"])
        with caplog.at_level(logging.WARNING, logger="pmlab"):
            grouper.warn_refused("x")
        assert warnings(caplog) == [] and grouper.refused == []

    def test_what_the_page_calls_a_bin_is_what_the_question_says(self) -> None:
        """The label comes from the question; it must be the one Gamma shows (``groupItemTitle``)."""
        grouper = PerBinGrouper()
        for m in recorded():
            grouper.offer(m)
        shown = {m["id"]: m["groupItemTitle"] for m in recorded() if "groupItemTitle" in m}
        taken = {b.market_id: b.label for g in grouper.groups() for b in g.bins}
        assert taken and all(shown[i] == label for i, label in taken.items())


class TestGrouping:
    def test_bins_come_back_in_order_whatever_order_they_arrive_in(self) -> None:
        texts = ["30°C or below", "31°C", "32°C", "33°C or higher"]
        markets = [market(i, bin_text=t) for i, t in enumerate(texts)]
        random.Random(7).shuffle(markets)
        (group,) = group_of(markets).groups()
        assert [b.label for b in group.bins] == texts
        assert group.complete

    def test_a_hole_in_the_middle_makes_it_partial_even_with_both_open_ends(self) -> None:
        texts = ["30°C or below", "32°C", "33°C or higher"]  # no 31 bin: it closed early
        (group,) = group_of([market(i, bin_text=t) for i, t in enumerate(texts)]).groups()
        assert not group.complete
        full = ["30°C or below", "31°C", "32°C", "33°C or higher"]
        (group,) = group_of([market(i, bin_text=t) for i, t in enumerate(full)]).groups()
        assert group.complete

    def test_the_yes_token_and_price_are_found_by_name_not_by_position(self) -> None:
        raw = market(1, outcomes='["No", "Yes"]', tokens='["n1", "y1"]', prices='["0.9", "0.1"]')
        (group,) = group_of([raw]).groups()
        assert (group.bins[0].token_id, group.bins[0].price) == ("y1", 0.1)

    def test_lists_may_be_lists_as_well_as_json_strings(self) -> None:
        raw = market(1)
        raw.update(outcomes=["Yes", "No"], clobTokenIds=["y1", "n1"], outcomePrices=["0.3", "0.7"])
        (group,) = group_of([raw]).groups()
        assert (group.bins[0].token_id, group.bins[0].price) == ("y1", 0.3)

    def test_an_unreadable_price_keeps_the_bin_and_has_no_price(self) -> None:
        for prices in (None, '["", "1"]', '["nan", "1"]', '["1.5", "0"]', "[]", "not json"):
            (group,) = group_of([market(1, prices=prices)]).groups()
            assert group.bins[0].price is None, prices

    def test_two_events_of_one_city_and_day_are_two_groups(self) -> None:
        markets = [market(1, event="900"), market(2, bin_text="31°C", event="901")]
        assert [g.market_id for g in group_of(markets).groups()] == ["900", "901"]

    def test_without_an_event_the_bins_of_a_city_and_day_are_one_group(self) -> None:
        markets = [market(i, bin_text=f"{30 + i}°C", event=None) for i in range(3)]
        (group,) = group_of(markets).groups()
        assert len(group.bins) == 3 and group.market_id == group.slug
        assert group.slug == "highest-temperature-in-austin-on-november-27-2026"
        assert group.question == "Highest temperature in Austin on November 27?"

    def test_the_group_closes_when_its_first_bin_does(self) -> None:
        markets = [
            market(1, end="2026-11-27T12:00:00Z"),
            market(2, bin_text="31°C", end="2026-11-26T09:00:00Z"),
        ]
        (group,) = group_of(markets).groups()
        assert group.close_time == "2026-11-26T09:00:00Z"

    def test_closes_may_be_dates_or_timestamps_or_junk_and_the_earliest_still_wins(self) -> None:
        markets = [
            market(1, end="2026-11-27"),  # a date alone: no zone, taken as UTC
            market(2, bin_text="31°C", end="2026-11-26T23:00:00Z"),
            market(3, bin_text="32°C", end="soon"),  # unreadable: never the earliest
        ]
        (group,) = group_of(markets).groups()
        assert group.close_time == "2026-11-26T23:00:00Z"

    def test_an_event_whose_only_close_is_unreadable_has_a_blank_one(self) -> None:
        """Blank is what a MarketSpec accepts for "unknown"; ``"soon"`` it would refuse. The year
        comes from the question here, since a close that cannot be read cannot give one."""
        (group,) = group_of([market(1, day="November 27, 2026", end="soon")]).groups()
        assert group.close_time == "" and group.target_date == "2026-11-27"

    def test_a_group_is_active_only_if_every_bin_is(self) -> None:
        a, b = market(1), market(2, bin_text="31°C")
        b["active"] = False
        (group,) = group_of([a, b]).groups()
        assert not group.active

    @pytest.mark.parametrize(
        ("day", "end", "expected"),
        [
            ("November 27", "2026-11-27T12:00:00Z", "2026-11-27"),
            ("December 31", "2027-01-01T00:30:00Z", "2026-12-31"),  # closes just after the day
            ("January 1", "2026-12-31T23:00:00Z", "2027-01-01"),  # closes just before it
            ("February 29", "2028-02-29T12:00:00Z", "2028-02-29"),
        ],
    )
    def test_the_year_is_the_one_that_puts_the_day_next_to_the_close(
        self, day, end, expected
    ) -> None:
        (group,) = group_of([market(1, day=day, end=end)]).groups()
        assert group.target_date == expected

    def test_a_year_in_the_question_wins(self) -> None:
        raw = market(1, day="November 27, 2025", end="2026-11-27T12:00:00Z")
        (group,) = group_of([raw]).groups()
        assert group.target_date == "2025-11-27"

    def test_without_a_close_the_year_is_read_from_the_slug_or_not_at_all(self) -> None:
        ok = market(1, end=None, event=None)
        ok["slug"] = "highest-temperature-in-austin-on-november-27-2026-30c"
        assert [g.target_date for g in group_of([ok]).groups()] == ["2026-11-27"]
        grouper = group_of([market(2, end=None, event=None)])
        assert grouper.groups() == []
        assert grouper.refused == [("2", "cannot tell the year of the date")]

    def test_a_day_that_does_not_exist_is_refused(self) -> None:
        grouper = group_of([market(1, day="February 30")])
        assert grouper.groups() == [] and len(grouper.refused) == 1


class TestWhatIsRefused:
    def test_outcomes_that_are_not_yes_no(self) -> None:
        grouper = group_of([market(1, outcomes='["30°C", "31°C"]')])
        assert grouper.groups() == []
        assert grouper.refused == [("1", "outcomes are not Yes/No")]

    @pytest.mark.parametrize("tokens", ['["only-one"]', '["", "n"]', "[]", "not json", "null"])
    def test_a_missing_yes_token(self, tokens) -> None:
        grouper = group_of([market(1, tokens=tokens)])
        assert grouper.groups() == [] and grouper.refused == [("1", "no Yes token id")]

    def test_a_market_with_no_token_field_at_all(self) -> None:
        raw = market(1)
        del raw["clobTokenIds"]
        grouper = group_of([raw])
        assert grouper.refused == [("1", "no Yes token id")]

    def test_a_question_it_cannot_read_is_taken_and_refused_not_passed_on(self) -> None:
        raw = market(1)
        raw["question"] = "Will the highest temperature in Austin exceed 30°C on November 27?"
        grouper = PerBinGrouper()
        assert grouper.offer(raw) is True  # this is the family's wording, so it is not legacy's
        assert grouper.groups() == [] and grouper.refused[0][0] == "1"
        assert "bin" in grouper.refused[0][1]

    def test_a_market_without_an_id(self) -> None:
        raw = market(1)
        del raw["id"]
        grouper = group_of([raw])
        assert grouper.groups() == [] and "market id" in grouper.refused[0][1]

    def test_the_condition_id_will_do_for_an_id(self) -> None:
        raw = market(1)
        del raw["id"]
        raw["conditionId"] = "0xabc"
        (group,) = group_of([raw]).groups()
        assert group.bins[0].market_id == "0xabc"

    def test_an_event_that_mixes_units_is_refused_whole(self) -> None:
        markets = [market(1), market(2, bin_text="between 88-89°F")]
        grouper = group_of(markets)
        assert grouper.groups() == []
        assert sorted(i for i, _ in grouper.refused) == ["1", "2"]
        assert all("mix" in reason for _, reason in grouper.refused)

    def test_an_event_that_names_two_cities_or_days_is_refused_whole(self) -> None:
        for other in (market(2, city="Rome"), market(2, day="November 28")):
            grouper = group_of([market(1), other])
            assert grouper.groups() == [], other["question"]
            assert all("cities or days" in reason for _, reason in grouper.refused)

    def test_an_event_with_two_bins_of_one_label_is_refused_whole(self) -> None:
        grouper = group_of([market(1), market(2)])
        assert grouper.groups() == []
        assert all("same label" in reason for _, reason in grouper.refused)

    def test_the_good_events_survive_a_bad_one(self) -> None:
        markets = [market(1, event="900"), market(2, event="901"), market(3, event="901")]
        assert [g.market_id for g in group_of(markets).groups()] == ["900"]

    def test_one_warning_lists_the_reasons_with_counts_and_a_few_ids(self, caplog) -> None:
        markets = [market(i, tokens="[]") for i in range(11, 18)] + [
            market(20, outcomes='["a", "b"]', event="901")
        ]
        grouper = group_of(markets)
        with caplog.at_level(logging.WARNING, logger="pmlab"):
            grouper.warn_refused("plugin")
        (said,) = warnings(caplog)
        assert said.startswith("plugin: 8 per-bin temperature markets were not read")
        assert "no Yes token id (7: 11, 12, 13, ...)" in said  # a few ids, not all seven
        assert "17" not in said
        assert "outcomes are not Yes/No (1: 20)" in said
