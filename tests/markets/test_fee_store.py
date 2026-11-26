"""``FeeScheduleStore``: the schedule a backtest could have known (#71), never a later one."""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from pmlab.core.fee_schedule import FeeSchedule
from pmlab.markets.fees import FeeScheduleStore, FeeSnapshot


def at(day: int, hour: int = 12) -> datetime:
    return datetime(2026, 11, day, hour, tzinfo=UTC)


def snap(
    market: str = "m1", seen: datetime | None = None, rate: float = 0.05, **kw: object
) -> FeeSnapshot:
    kw.setdefault("source", "gamma")
    return FeeSnapshot(
        market_id=market,
        seen_at=seen or at(1),
        schedule=FeeSchedule(rate=rate, rebate_rate=0.15),
        **kw,  # type: ignore[arg-type]
    )


@pytest.fixture
def store(tmp_path: Path) -> FeeScheduleStore:
    return FeeScheduleStore(tmp_path / "fees")


def test_a_backtest_sees_the_schedule_known_at_the_time(store: FeeScheduleStore) -> None:
    store.record([snap(seen=at(1), rate=0.03), snap(seen=at(20), rate=0.05)])
    assert store.as_of(at(10))["m1"].schedule.rate == 0.03
    assert store.as_of(at(25))["m1"].schedule.rate == 0.05


def test_before_the_first_sighting_there_is_no_schedule_not_a_later_one(
    store: FeeScheduleStore,
) -> None:
    store.record([snap(seen=at(20), rate=0.05)])
    assert store.as_of(at(10)) == {}


def test_the_cutoff_is_strict_as_in_the_feature_snapshot_store(store: FeeScheduleStore) -> None:
    store.record([snap(seen=at(10, 12))])
    assert store.as_of(at(10, 12)) == {}
    assert "m1" in store.as_of(at(10, 12).replace(minute=1))


def test_a_date_cutoff_is_midnight_utc_so_a_schedule_seen_that_day_is_not_yet_known(
    store: FeeScheduleStore,
) -> None:
    store.record([snap(seen=at(20, 18))])
    assert store.as_of(date(2026, 11, 20)) == {}
    assert "m1" in store.as_of(date(2026, 11, 21))
    assert "m1" in store.as_of("2026-11-21")
    assert store.as_of("2026-11-20T19:00:00Z")["m1"].seen_at == at(20, 18)


def test_a_cutoff_without_a_zone_is_refused(store: FeeScheduleStore) -> None:
    store.record([snap()])
    with pytest.raises(ValueError, match="timezone"):
        store.as_of(datetime(2026, 11, 20, 12))


def test_a_fee_change_shows_in_the_history_in_time_order(store: FeeScheduleStore) -> None:
    store.record([snap(seen=at(20), rate=0.05)])
    store.record([snap(seen=at(1), rate=0.03)])
    rates = [(s.seen_at, s.schedule.rate) for s in store.history("m1")]
    assert rates == [(at(1), 0.03), (at(20), 0.05)]


def test_recording_the_same_sighting_twice_changes_nothing(store: FeeScheduleStore) -> None:
    assert store.record([snap(), snap(market="m2")]) == 2
    assert store.record([snap()]) == 2
    assert len(store.history("m1")) == 1


def warnings_of(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


class TestTwoSchedulesForOneMarketAndMoment:
    """A snapshot is its market and the moment it was seen, and the first one stays. A second one
    with another schedule used to vanish without a word: a fee that changed between two sightings
    with the same ``seen_at`` was lost."""

    def test_the_first_stays_and_the_second_is_warned_about(
        self, store: FeeScheduleStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        store.record([snap(seen=at(1), rate=0.03, slug="a-weather-bin")])
        with caplog.at_level("WARNING", logger="pmlab"):
            assert store.record([snap(seen=at(1), rate=0.05, slug="a-weather-bin")]) == 1
        assert [s.schedule.rate for s in store.history("m1")] == [0.03]
        assert store.as_of(at(2))["m1"].schedule.rate == 0.03
        (warning,) = warnings_of(caplog)
        assert "a-weather-bin" in warning and "0.03" in warning and "0.05" in warning

    def test_within_one_call_too(
        self, store: FeeScheduleStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING", logger="pmlab"):
            store.record([snap(seen=at(1), rate=0.03), snap(seen=at(1), rate=0.05)])
        assert [s.schedule.rate for s in store.history("m1")] == [0.03]
        (warning,) = warnings_of(caplog)
        assert "m1" in warning

    def test_a_seen_at_that_is_a_date_loses_a_later_change_of_the_day_loudly(
        self, store: FeeScheduleStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        day = date(2026, 11, 26)
        store.record([snap(seen=day, rate=0.03)])  # type: ignore[arg-type]
        with caplog.at_level("WARNING", logger="pmlab"):
            store.record([snap(seen=day, rate=0.05)])  # type: ignore[arg-type]
        assert [s.schedule.rate for s in store.history("m1")] == [0.03]
        assert len(warnings_of(caplog)) == 1

    def test_another_source_with_the_same_numbers_is_a_difference_too(
        self, store: FeeScheduleStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        guess = snap(seen=at(1), rate=0.05, source="category_default", category="sports")
        store.record([snap(seen=at(1), rate=0.05)])
        with caplog.at_level("WARNING", logger="pmlab"):
            store.record([guess])
        assert len(warnings_of(caplog)) == 1 and not store.history("m1")[0].fallback

    def test_the_same_sighting_again_is_silent(
        self, store: FeeScheduleStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING", logger="pmlab"):
            store.record([snap(seen=at(1)), snap(seen=at(1))])
            store.record([snap(seen=at(1))])
            # only what the schedule is counts: a base_fee that came along the second time does not
            store.record([snap(seen=at(1), base_fee=1000)])
        assert warnings_of(caplog) == [] and len(store.history("m1")) == 1

    def test_other_moments_and_other_markets_are_not_conflicts(
        self, store: FeeScheduleStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING", logger="pmlab"):
            store.record(
                [snap("m1", at(1), 0.03), snap("m1", at(2), 0.05), snap("m2", at(1), 0.07)]
            )
            store.record([snap("m3", at(1), 0.01), snap("m1", at(3), 0.02)])
        assert warnings_of(caplog) == [] and len(store.history("m1")) == 3


def test_markets_are_independent_and_can_be_picked(store: FeeScheduleStore) -> None:
    store.record([snap("m1", at(1), 0.03), snap("m2", at(2), 0.05), snap("m1", at(5), 0.04)])
    both = store.as_of(at(10))
    assert {k: v.schedule.rate for k, v in both.items()} == {"m1": 0.04, "m2": 0.05}
    assert list(store.as_of(at(10), market_ids=["m2", "nope"])) == ["m2"]
    assert store.history("m2")[0].seen_at == at(2) and store.history("nope") == []


def test_a_snapshot_comes_back_whole_with_its_gaps(store: FeeScheduleStore) -> None:
    full = FeeSnapshot(
        market_id="m1",
        seen_at=at(1),
        schedule=FeeSchedule(rate=0.01, exponent=1, taker_only=True, rebate_rate=0.25),
        source="gamma",
        slug="a-weather-bin",
        fee_type="weather_fees",
        base_fee=1000,
    )
    fallback = FeeSnapshot(
        market_id="m2",
        seen_at=at(1),
        schedule=FeeSchedule(rate=0.05),
        source="category_default",
        category="sports",
    )
    free = FeeSnapshot(
        market_id="m3",
        seen_at=at(1),
        schedule=FeeSchedule(rate=0.0, enabled=False),
        source="clob",
        base_fee=0,
    )
    store.record([full, fallback, free])
    back = store.as_of(at(2))
    assert back == {"m1": full, "m2": fallback, "m3": free}
    assert back["m1"].base_fee == 1000 and isinstance(back["m1"].base_fee, int)
    assert back["m2"].base_fee is None and back["m2"].fallback
    assert back["m3"].base_fee == 0 and back["m3"].schedule.enabled is False


def test_it_survives_a_new_store_object(tmp_path: Path) -> None:
    FeeScheduleStore(tmp_path / "fees").record([snap(seen=at(3))])
    again = FeeScheduleStore(tmp_path / "fees")
    assert again.history("m1")[0].seen_at == at(3)


def test_an_empty_store_has_nothing(store: FeeScheduleStore) -> None:
    assert store.as_of(at(10)) == {} and store.history("m1") == []
    assert store.record([]) == 0


def test_it_keeps_its_rows_in_the_feature_snapshot_store_family(tmp_path: Path) -> None:
    # reuse, not a second format: one parquet file in the store root, read by FeatureSnapshotStore
    from pmlab.data.snapshot_store import FeatureSnapshotStore

    FeeScheduleStore(tmp_path / "fees").record([snap(seen=at(3))])
    rows = FeatureSnapshotStore(tmp_path / "fees").load("fee_schedules")
    assert list(rows["market_id"]) == ["m1"] and list(rows["rate"]) == [0.05]


def test_what_is_not_a_snapshot_is_refused(store: FeeScheduleStore) -> None:
    with pytest.raises(TypeError, match="FeeSnapshot"):
        store.record([{"market_id": "m1"}])  # type: ignore[list-item]


@pytest.mark.parametrize("bad", ["last tuesday", "2026-13-45", ""])
def test_a_cutoff_that_is_not_a_date_is_refused(store: FeeScheduleStore, bad: str) -> None:
    with pytest.raises(ValueError, match="ISO"):
        store.as_of(bad)


def test_a_cutoff_of_another_type_is_refused(store: FeeScheduleStore) -> None:
    with pytest.raises(TypeError, match="cutoff"):
        store.as_of(5)  # type: ignore[arg-type]


def test_the_root_is_where_it_was_asked_to_live(tmp_path: Path) -> None:
    assert FeeScheduleStore(tmp_path / "fees").root == tmp_path / "fees"
