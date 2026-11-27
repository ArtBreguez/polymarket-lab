"""`CostModel` charges Polymarket's price-dependent taker fee, and keeps the flat-bps model reachable.

Until 1.5.0 `CostModel` charged ``stake x taker_bps / 10_000``: 0.30% of the stake at any price. A
Polymarket taker pays ``shares x rate x p x (1 - p)``, which for a 0.05-rate market is 3.5% of the
stake at 0.30 and 2.5% at 0.50 (#72, epic #69). So the default model is now the Polymarket one, the
old arithmetic is `fee_model="flat-bps"`, and a record, a champion or a tuner written before 1.5.0
(whose cost dict has no `fee_model`) still means flat-bps, so it reproduces with the model it ran with.
"""

from __future__ import annotations

import inspect
import json
import warnings

import pytest

from pmlab import PmlabDeprecationWarning
from pmlab.core.costs import CostModel
from pmlab.core.fee_schedule import FeeSchedule, category_fee_schedule, taker_fee_on_stake

OLD_DICT = {
    "taker_bps": 30.0,
    "slippage_bps": 0.0,
    "slippage_fixed": 0.0,
    "depth_bps_per_unit": 0.0,
    "depth_reference_stake": 1.0,
}


class TestDefaultModelIsPolymarket:
    def test_the_default_model_is_polymarket(self) -> None:
        assert CostModel().fee_model == "polymarket"

    def test_the_default_schedule_is_the_other_category_and_says_it_is_a_guess(self) -> None:
        costs = CostModel()
        assert costs.fee_schedule == category_fee_schedule("other")
        assert costs.fee_source == "category_default"

    @pytest.mark.parametrize(
        ("price", "fee"), [(0.20, 0.04), (0.30, 0.035), (0.50, 0.025), (0.80, 0.01)]
    )
    def test_the_fee_on_a_unit_stake_is_rate_times_one_minus_price(
        self, price: float, fee: float
    ) -> None:
        assert CostModel().fee(1.0, price) == pytest.approx(fee, abs=1e-12)

    def test_the_fee_scales_with_the_stake(self) -> None:
        assert CostModel().fee(10.0, 0.50) == pytest.approx(0.25, abs=1e-12)

    def test_the_fee_is_the_schedule_fee_to_the_last_digit(self) -> None:
        costs = CostModel()
        assert costs.fee(7.0, 0.37) == taker_fee_on_stake(7.0, 0.37, costs.fee_schedule)  # type: ignore[arg-type]

    @pytest.mark.parametrize("price", [0.0, 1.0])
    def test_a_token_at_either_end_pays_no_fee(self, price: float) -> None:
        assert CostModel().fee(1.0, price) == 0.0

    def test_the_polymarket_fee_needs_the_price_it_depends_on(self) -> None:
        with pytest.raises(ValueError, match="price"):
            CostModel().fee(1.0)

    @pytest.mark.parametrize("price", [-0.1, 1.5, float("nan")])
    def test_a_price_that_is_no_probability_is_refused(self, price: float) -> None:
        with pytest.raises(ValueError, match="price"):
            CostModel().fee(1.0, price)

    def test_a_stake_of_zero_is_still_refused(self) -> None:
        with pytest.raises(ValueError, match="stake"):
            CostModel().fee(0.0, 0.5)

    def test_a_given_schedule_is_charged_and_labelled_as_given(self) -> None:
        costs = CostModel(fee_schedule=FeeSchedule(rate=0.01))
        assert costs.fee_model == "polymarket"
        assert costs.fee(1.0, 0.50) == pytest.approx(0.005, abs=1e-12)
        assert costs.fee_source == "user"

    def test_a_given_source_is_kept(self) -> None:
        costs = CostModel(fee_schedule=FeeSchedule(rate=0.01), fee_source="gamma")
        assert costs.fee_source == "gamma"

    def test_a_disabled_schedule_charges_nothing(self) -> None:
        costs = CostModel(fee_schedule=FeeSchedule(rate=0.05, enabled=False))
        assert costs.fee(1.0, 0.30) == 0.0

    def test_slippage_is_untouched_by_the_fee_model(self) -> None:
        costs = CostModel(slippage_bps=100.0)
        assert costs.fill_price(0.50, stake=1.0) == pytest.approx(0.505)

    def test_a_schedule_may_be_given_as_the_mapping_to_dict_wrote(self) -> None:
        costs = CostModel(fee_schedule={"rate": 0.02})  # type: ignore[arg-type]
        assert costs.fee_schedule == FeeSchedule(rate=0.02)


class TestFlatBpsIsStillReachable:
    def test_the_old_arithmetic_is_fee_model_flat_bps(self) -> None:
        costs = CostModel(fee_model="flat-bps")
        assert costs.fee(1.0) == pytest.approx(0.003)
        assert costs.fee(10.0, 0.30) == pytest.approx(0.03)  # the price changes nothing

    def test_flat_bps_takes_its_rate_from_taker_bps(self) -> None:
        assert CostModel(fee_model="flat-bps", taker_bps=50.0).fee(2.0) == pytest.approx(0.01)

    def test_flat_bps_has_no_schedule_and_says_so(self) -> None:
        costs = CostModel(fee_model="flat-bps")
        assert costs.fee_schedule is None
        assert costs.fee_source == "flat_bps"

    def test_a_flat_bps_model_takes_no_schedule(self) -> None:
        with pytest.raises(ValueError, match="flat-bps"):
            CostModel(fee_model="flat-bps", fee_schedule=FeeSchedule(rate=0.05))

    def test_frictionless_is_still_free_at_any_price(self) -> None:
        free = CostModel.frictionless()
        assert free.fee(100.0, 0.5) == 0.0
        assert free.fee(100.0) == 0.0


class TestNamingTheModel:
    def test_taker_bps_alone_still_means_flat_bps_and_warns(self) -> None:
        with pytest.warns(PmlabDeprecationWarning, match="taker_bps"):
            costs = CostModel(taker_bps=50.0)
        assert costs.fee_model == "flat-bps"
        assert costs.fee(1.0, 0.30) == pytest.approx(0.005)

    def test_the_warning_names_the_callers_line_not_pmlab(self) -> None:
        with pytest.warns(PmlabDeprecationWarning) as caught:
            CostModel(taker_bps=50.0)
        assert caught[0].filename.endswith("test_costs_polymarket.py")

    def test_an_explicit_default_taker_bps_is_not_mistaken_for_the_default(self) -> None:
        # `CostModel(taker_bps=30.0)` was the flat 30 bps before 1.5.0 and still is: the signature
        # shows 30.0 as the default, but passing it is a choice.
        with pytest.warns(PmlabDeprecationWarning, match="taker_bps"):
            costs = CostModel(taker_bps=30.0)
        assert costs.fee_model == "flat-bps"

    def test_a_schedule_with_a_taker_bps_is_a_contradiction(self) -> None:
        with pytest.raises(ValueError, match="taker_bps"):
            CostModel(taker_bps=50.0, fee_schedule=FeeSchedule(rate=0.05))

    def test_naming_the_model_is_not_deprecated(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", PmlabDeprecationWarning)
            CostModel(fee_model="flat-bps", taker_bps=50.0)
            CostModel(fee_model="polymarket")
            CostModel()

    def test_an_unknown_model_is_refused_and_names_the_choices(self) -> None:
        with pytest.raises(ValueError, match="polymarket"):
            CostModel(fee_model="kalshi")  # type: ignore[arg-type]

    def test_a_schedule_that_is_not_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="fee_schedule"):
            CostModel(fee_schedule=0.05)  # type: ignore[arg-type]

    def test_an_empty_source_is_refused(self) -> None:
        with pytest.raises(ValueError, match="fee_source"):
            CostModel(fee_source="")

    def test_replace_keeps_the_model(self) -> None:
        import dataclasses

        flat = CostModel(fee_model="flat-bps", taker_bps=40.0)
        assert dataclasses.replace(flat, slippage_bps=5.0).fee_model == "flat-bps"
        poly = CostModel(fee_schedule=FeeSchedule(rate=0.01))
        assert dataclasses.replace(poly, slippage_bps=5.0).fee_schedule == FeeSchedule(rate=0.01)


class TestWhatAnOldFileMeans:
    """A cost dict written by 1.4.0 and before has no `fee_model`: it is the flat-bps model."""

    def test_from_dict_reads_an_old_dict_as_flat_bps_without_a_warning(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", PmlabDeprecationWarning)
            costs = CostModel.from_dict(OLD_DICT)
        assert costs.fee_model == "flat-bps"
        assert costs.fee(1.0, 0.30) == pytest.approx(0.003)

    def test_from_dict_keeps_an_old_dicts_other_fields(self) -> None:
        costs = CostModel.from_dict({**OLD_DICT, "taker_bps": 12.0, "slippage_bps": 7.0})
        assert (costs.taker_bps, costs.slippage_bps) == (12.0, 7.0)

    def test_the_constructor_reads_an_old_dict_as_flat_bps_too_but_warns(self) -> None:
        with pytest.warns(PmlabDeprecationWarning):
            costs = CostModel(**OLD_DICT)  # type: ignore[arg-type]
        assert costs.fee_model == "flat-bps"
        assert costs.fee(1.0, 0.30) == pytest.approx(0.003)

    def test_from_dict_still_refuses_a_key_it_does_not_know(self) -> None:
        with pytest.raises(TypeError, match="surprise"):
            CostModel.from_dict({**OLD_DICT, "surprise": 1})

    def test_from_dict_refuses_what_is_not_a_mapping(self) -> None:
        with pytest.raises(TypeError, match="mapping"):
            CostModel.from_dict([1, 2])  # type: ignore[arg-type]


class TestToDict:
    @pytest.mark.parametrize(
        "costs",
        [
            CostModel(),
            CostModel(fee_schedule=FeeSchedule(rate=0.02, rebate_rate=0.25), fee_source="gamma"),
            CostModel(fee_model="flat-bps", taker_bps=45.0, slippage_bps=10.0),
            CostModel(slippage_fixed=0.01, depth_bps_per_unit=50.0, depth_reference_stake=5.0),
            CostModel.frictionless(),
        ],
    )
    def test_a_model_round_trips_through_json(self, costs: CostModel) -> None:
        written = json.loads(json.dumps(costs.to_dict()))
        assert CostModel.from_dict(written) == costs
        assert CostModel(**written) == costs

    def test_the_dict_says_which_model_and_schedule_it_is(self) -> None:
        written = CostModel().to_dict()
        assert written["fee_model"] == "polymarket"
        assert written["fee_source"] == "category_default"
        assert written["fee_schedule"]["rate"] == 0.05

    def test_a_flat_bps_dict_names_its_model_and_has_the_old_keys(self) -> None:
        written = CostModel(fee_model="flat-bps").to_dict()
        assert written["fee_model"] == "flat-bps"
        assert written["fee_schedule"] is None
        assert {k: written[k] for k in OLD_DICT} == OLD_DICT


class TestFeeFields:
    """What a trade carries about its fee: fee_paid, fee_rate, liquidity and the schedule source."""

    def test_polymarket_fields(self) -> None:
        fields = CostModel().fee_fields(1.0, 0.30)
        assert fields["fee_paid"] == pytest.approx(0.035, abs=1e-12)
        assert fields["fee_rate"] == 0.05
        assert fields["liquidity"] == "taker"
        assert fields["fee_source"] == "category_default"
        assert set(fields) == {"fee_paid", "fee_rate", "liquidity", "fee_source"}

    def test_flat_bps_fields(self) -> None:
        fields = CostModel(fee_model="flat-bps", taker_bps=50.0).fee_fields(2.0, 0.30)
        assert fields["fee_paid"] == pytest.approx(0.01)
        assert fields["fee_rate"] == pytest.approx(0.005)
        assert fields["fee_source"] == "flat_bps"
        assert fields["liquidity"] == "taker"

    def test_a_disabled_schedule_reports_the_rate_it_charged_which_is_zero(self) -> None:
        costs = CostModel(fee_schedule=FeeSchedule(rate=0.05, enabled=False), fee_source="gamma")
        fields = costs.fee_fields(1.0, 0.30)
        assert fields["fee_paid"] == 0.0
        assert fields["fee_rate"] == 0.0
        assert fields["fee_source"] == "gamma"


class TestTheSignatureIsFrozen:
    """1.5.0 appends fields; it changes no existing one (docs/api-stability.md)."""

    def test_the_old_fields_keep_their_names_order_and_defaults(self) -> None:
        params = inspect.signature(CostModel).parameters
        assert list(params)[:5] == [
            "taker_bps",
            "slippage_bps",
            "slippage_fixed",
            "depth_bps_per_unit",
            "depth_reference_stake",
        ]
        assert [repr(params[n].default) for n in list(params)[:5]] == [
            "30.0",
            "0.0",
            "0.0",
            "0.0",
            "1.0",
        ]

    def test_the_new_fields_are_appended_with_defaults(self) -> None:
        params = list(inspect.signature(CostModel).parameters.values())
        assert [p.name for p in params[5:]] == ["fee_model", "fee_schedule", "fee_source"]
        assert all(p.default is not inspect.Parameter.empty for p in params[5:])

    def test_the_taker_bps_attribute_is_a_plain_float(self) -> None:
        assert type(CostModel().taker_bps) is float
        assert CostModel().taker_bps == 30.0
