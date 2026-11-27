"""A fee schedule is known from the moment it was seen, and a `CostModel` says so (#72, review r1).

#89 keeps every `FeeSnapshot` with the moment it was seen, because Polymarket changes its rates
without notice and "a backtest at date T must price a trade with the schedule that was known before
T, not with today's". A bare `FeeSchedule` in a `CostModel` carries no such moment, so the first
version of #72 could price a January trade with a schedule read in November and label it
``fee_source="gamma"``. `CostModel.with_fee_snapshot` carries the moment (``fee_seen_at``);
`rolling_origin_eval` refuses a fold that decided before it (see
``tests/backtest/test_rolling_origin_fees.py``).

The rest of the file holds the other half of the same review: a `CostModel` whose label or fee lies,
and a pickle of 1.4.0 that must still be the flat model.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import pickle
import warnings
from datetime import UTC, date, datetime, timedelta, timezone

import pytest

from pmlab.core.costs import CostModel
from pmlab.core.fee_schedule import FeeSchedule, category_fee_schedule
from pmlab.markets.fees import FeeSnapshot

SEEN = datetime(2026, 3, 1, 9, 30, tzinfo=UTC)


def _snapshot(
    rate: float = 0.01, source: str = "gamma", seen_at: datetime = SEEN, **kwargs: object
) -> FeeSnapshot:
    return FeeSnapshot(
        market_id="m1",
        seen_at=seen_at,
        schedule=FeeSchedule(rate=rate),
        source=source,  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


class TestWithFeeSnapshot:
    def test_it_sets_the_schedule_the_source_and_the_moment_together(self) -> None:
        costs = CostModel().with_fee_snapshot(_snapshot())
        assert costs.fee_model == "polymarket"
        assert costs.fee_schedule == FeeSchedule(rate=0.01)
        assert costs.fee_source == "gamma"
        assert costs.fee_seen_at == SEEN

    def test_it_keeps_the_slippage_and_depth_of_the_model(self) -> None:
        base = CostModel(slippage_bps=50.0, slippage_fixed=0.01, depth_bps_per_unit=20.0)
        costs = base.with_fee_snapshot(_snapshot())
        assert (costs.slippage_bps, costs.slippage_fixed, costs.depth_bps_per_unit) == (
            50.0,
            0.01,
            20.0,
        )

    def test_it_replaces_the_schedule_of_a_model_that_already_has_one(self) -> None:
        later = _snapshot(rate=0.05, source="clob", seen_at=SEEN + timedelta(days=1))
        costs = CostModel().with_fee_snapshot(_snapshot()).with_fee_snapshot(later)
        assert (costs.fee_schedule, costs.fee_source) == (later.schedule, "clob")
        assert costs.fee_seen_at == later.seen_at

    def test_the_fee_it_charges_is_the_schedules(self) -> None:
        costs = CostModel().with_fee_snapshot(_snapshot(rate=0.01))
        assert costs.fee(1.0, 0.50) == pytest.approx(0.005, abs=1e-12)

    def test_a_fallback_snapshot_keeps_its_category_default_label(self) -> None:
        snap = FeeSnapshot(
            market_id="m1",
            seen_at=SEEN,
            schedule=category_fee_schedule("weather"),
            source="category_default",
            category="weather",
        )
        assert CostModel().with_fee_snapshot(snap).fee_source == "category_default"

    def test_a_flat_model_is_not_turned_into_a_polymarket_one_in_silence(self) -> None:
        with pytest.raises(ValueError, match="flat-bps"):
            CostModel(fee_model="flat-bps").with_fee_snapshot(_snapshot())

    def test_only_a_snapshot_will_do(self) -> None:
        with pytest.raises(TypeError, match="FeeSnapshot"):
            CostModel().with_fee_snapshot(FeeSchedule(rate=0.01))  # type: ignore[arg-type]


class TestTheMomentTheScheduleWasSeen:
    def test_a_model_without_one_has_none_and_does_not_write_it(self) -> None:
        costs = CostModel(fee_schedule=FeeSchedule(rate=0.01), fee_source="user")
        assert costs.fee_seen_at is None
        assert "fee_seen_at" not in costs.to_dict()

    def test_it_is_written_as_text_and_round_trips(self) -> None:
        costs = CostModel(slippage_bps=25.0).with_fee_snapshot(_snapshot())
        payload = costs.to_dict()
        assert payload["fee_seen_at"] == "2026-03-01T09:30:00+00:00"
        assert CostModel.from_dict(json.loads(json.dumps(payload))) == costs
        assert CostModel(**json.loads(json.dumps(payload))) == costs

    def test_a_date_is_its_midnight_in_utc(self) -> None:
        costs = CostModel(fee_seen_at=date(2026, 3, 1))  # type: ignore[arg-type]
        assert costs.fee_seen_at == datetime(2026, 3, 1, tzinfo=UTC)

    def test_another_timezone_is_converted_to_utc(self) -> None:
        plus_two = datetime(2026, 3, 1, 11, 30, tzinfo=timezone(timedelta(hours=2)))
        assert CostModel(fee_seen_at=plus_two).fee_seen_at == SEEN

    @pytest.mark.parametrize("naive", [datetime(2026, 3, 1, 9, 30), "2026-03-01T09:30:00"])
    def test_a_moment_without_a_timezone_is_refused(self, naive: object) -> None:
        with pytest.raises(ValueError, match="timezone"):
            CostModel(fee_seen_at=naive)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", ["yesterday", 20260301, True])
    def test_a_value_that_is_no_moment_is_refused(self, bad: object) -> None:
        with pytest.raises((TypeError, ValueError)):
            CostModel(fee_seen_at=bad)  # type: ignore[arg-type]

    def test_a_flat_model_has_no_schedule_to_date(self) -> None:
        with pytest.raises(ValueError, match="flat-bps"):
            CostModel(fee_model="flat-bps", fee_seen_at=SEEN)

    def test_it_is_part_of_what_two_models_are_equal_on(self) -> None:
        a = CostModel().with_fee_snapshot(_snapshot())
        b = CostModel().with_fee_snapshot(_snapshot(seen_at=SEEN + timedelta(seconds=1)))
        assert a != b

    def test_copy_and_pickle_keep_it(self) -> None:
        costs = CostModel().with_fee_snapshot(_snapshot())
        assert copy.deepcopy(costs) == costs
        assert pickle.loads(pickle.dumps(costs)) == costs  # noqa: S301


class TestAModelWhoseLabelOrFeeLies:
    """`dataclasses.replace` re-runs `__post_init__` on fields a caller changed one at a time."""

    def test_a_schedule_swapped_in_keeps_no_label_that_describes_the_old_one(self) -> None:
        with pytest.raises(ValueError, match="fee_source"):
            dataclasses.replace(CostModel(), fee_schedule=FeeSchedule(rate=0.0123))

    def test_naming_the_source_with_the_schedule_is_fine(self) -> None:
        costs = dataclasses.replace(
            CostModel(), fee_schedule=FeeSchedule(rate=0.0123), fee_source="user"
        )
        assert costs.fee_source == "user"

    def test_a_published_category_rate_may_keep_the_category_label(self) -> None:
        # a weather schedule is one of the category defaults, so the label is true of it
        costs = dataclasses.replace(CostModel(), fee_schedule=category_fee_schedule("weather"))
        assert costs.fee_source == "category_default"
        assert costs.fee_schedule == category_fee_schedule("weather")

    def test_a_flat_label_on_a_polymarket_fee_is_refused(self) -> None:
        with pytest.raises(ValueError, match="flat_bps"):
            dataclasses.replace(CostModel(fee_model="flat-bps"), fee_model="polymarket")

    def test_a_flat_label_given_by_hand_to_a_polymarket_model_is_refused(self) -> None:
        with pytest.raises(ValueError, match="flat_bps"):
            CostModel(fee_model="polymarket", fee_source="flat_bps")

    def test_a_taker_bps_the_polymarket_model_ignores_is_called_out(self) -> None:
        with pytest.warns(UserWarning, match="taker_bps"):
            costs = dataclasses.replace(CostModel(), taker_bps=50.0)
        assert costs.fee(100.0, 0.5) == pytest.approx(2.5)  # still the polymarket fee

    def test_the_same_when_the_model_is_named_and_the_warning_names_the_caller(self) -> None:
        with pytest.warns(UserWarning, match="taker_bps") as record:
            CostModel(fee_model="polymarket", taker_bps=50.0)
        assert record[0].filename == __file__

    @pytest.mark.parametrize(
        "build",
        [
            lambda: CostModel(),
            lambda: CostModel(fee_model="polymarket"),
            lambda: CostModel(fee_model="flat-bps", taker_bps=50.0),
            lambda: CostModel.from_dict(CostModel().to_dict()),
            lambda: CostModel.from_dict({"taker_bps": 50.0}),
            lambda: CostModel(fee_model="polymarket", taker_bps=30.0),
        ],
        ids=["default", "named", "flat", "round-trip", "legacy-dict", "taker-at-the-default"],
    )
    def test_nothing_else_warns(self, build: object) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            build()  # type: ignore[operator]


class TestAPickleOfBefore1_5_0:
    """pmlab never pickles a `CostModel`, but a user's object may hold one (a notebook, a cache)."""

    @staticmethod
    def _pickled_by_1_4_0(**fields: float) -> bytes:
        """What 1.4.0 wrote: an instance whose state is the five numbers and nothing else."""
        old = object.__new__(CostModel)
        old.__dict__.update(
            taker_bps=45.0,
            slippage_bps=0.0,
            slippage_fixed=0.0,
            depth_bps_per_unit=0.0,
            depth_reference_stake=1.0,
        )
        old.__dict__.update(fields)
        return pickle.dumps(old)

    def test_it_loads_as_the_flat_model_it_was(self) -> None:
        costs = pickle.loads(self._pickled_by_1_4_0())  # noqa: S301
        assert costs == CostModel(fee_model="flat-bps", taker_bps=45.0)
        assert costs.fee_model == "flat-bps"
        assert costs.fee_schedule is None
        assert costs.fee_source == "flat_bps"

    def test_it_charges_what_it_charged(self) -> None:
        costs = pickle.loads(self._pickled_by_1_4_0(slippage_bps=50.0))  # noqa: S301
        assert costs.fee(1.0) == pytest.approx(0.0045)
        assert costs.fee(1.0, 0.5) == pytest.approx(0.0045)
        assert costs.fill_price(0.40, stake=1.0) == pytest.approx(0.402)

    def test_it_round_trips_through_to_dict_like_any_other(self) -> None:
        costs = pickle.loads(self._pickled_by_1_4_0())  # noqa: S301
        assert CostModel.from_dict(costs.to_dict()) == costs

    def test_a_pickle_of_this_version_is_unchanged(self) -> None:
        for costs in (
            CostModel(),
            CostModel(fee_model="flat-bps", taker_bps=45.0),
            CostModel(slippage_bps=50.0).with_fee_snapshot(_snapshot()),
        ):
            assert pickle.loads(pickle.dumps(costs)) == costs  # noqa: S301
