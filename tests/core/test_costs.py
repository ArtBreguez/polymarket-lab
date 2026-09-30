"""Tests for core.costs — the backtest cost model.

`rolling_origin_eval` fills frictionlessly apart from a flat taker fee: it buys at
`market_price` no matter how large the stake or how thin the book. Real fills pay
the spread, walk the book, and cost a fee. A backtest that ignores that reports an
edge the live broker cannot reproduce.

`CostModel` makes those costs explicit and **opt-in**: the default instance must
reproduce today's numbers exactly, so every published result stays valid.
"""

import pytest

from pmlab.core.costs import CostModel


class TestDefaultIsTodaysBehaviour:
    """The default must be a no-op beyond the existing 30bps fee.

    This is the compatibility contract: adding `costs=` to the backtest must not
    silently move any previously published PnL.
    """

    def test_default_adds_no_slippage(self) -> None:
        assert CostModel().fill_price(0.50, stake=1.0) == 0.50

    def test_default_fee_matches_the_existing_constant(self) -> None:
        # core.fees.DEFAULT_TAKER_BPS is 30.0 and rolling_origin charged
        # flat_stake * taker_bps / 10_000 — the default must agree exactly.
        assert CostModel().fee(stake=1.0) == pytest.approx(1.0 * 30.0 / 10_000.0)

    def test_frictionless_classmethod_is_truly_free(self) -> None:
        free = CostModel.frictionless()
        assert free.fee(stake=100.0) == 0.0
        assert free.fill_price(0.42, stake=100.0) == 0.42


class TestSlippage:
    """Slippage must always move the fill against the buyer."""

    def test_bps_slippage_raises_the_buy_price(self) -> None:
        # 100 bps on a 0.50 price = 0.005 worse.
        model = CostModel(slippage_bps=100.0)
        assert model.fill_price(0.50, stake=1.0) == pytest.approx(0.505)

    def test_fixed_slippage_adds_absolute_price(self) -> None:
        model = CostModel(slippage_fixed=0.01)
        assert model.fill_price(0.50, stake=1.0) == pytest.approx(0.51)

    def test_bps_and_fixed_compose(self) -> None:
        model = CostModel(slippage_bps=100.0, slippage_fixed=0.01)
        assert model.fill_price(0.50, stake=1.0) == pytest.approx(0.515)

    def test_slippage_is_proportional_to_price_not_flat(self) -> None:
        # bps is a fraction OF THE PRICE: a cheap outcome slips less in absolute terms.
        model = CostModel(slippage_bps=200.0)
        cheap = model.fill_price(0.10, stake=1.0) - 0.10
        rich = model.fill_price(0.90, stake=1.0) - 0.90
        assert rich > cheap


class TestDepthPenalty:
    """Larger orders walk the book, so the penalty must grow with stake."""

    def test_no_penalty_at_or_below_the_depth_reference(self) -> None:
        model = CostModel(depth_bps_per_unit=50.0, depth_reference_stake=10.0)
        assert model.fill_price(0.50, stake=10.0) == pytest.approx(0.50)
        assert model.fill_price(0.50, stake=5.0) == pytest.approx(0.50)

    def test_penalty_grows_with_stake_beyond_the_reference(self) -> None:
        model = CostModel(depth_bps_per_unit=50.0, depth_reference_stake=10.0)
        small = model.fill_price(0.50, stake=20.0)
        large = model.fill_price(0.50, stake=40.0)
        assert large > small > 0.50

    def test_depth_penalty_math_is_documented_and_exact(self) -> None:
        # excess = (stake - reference) / reference = (20-10)/10 = 1.0
        # penalty_bps = depth_bps_per_unit * excess = 50.0
        # fill = 0.50 * (1 + 50/10_000) = 0.5025
        model = CostModel(depth_bps_per_unit=50.0, depth_reference_stake=10.0)
        assert model.fill_price(0.50, stake=20.0) == pytest.approx(0.5025)

    def test_depth_disabled_by_default_ignores_stake(self) -> None:
        model = CostModel()
        assert model.fill_price(0.50, stake=1_000_000.0) == 0.50


class TestFillPriceIsClamped:
    """A probability price cannot exceed 1.0 — costs must not create arbitrage."""

    def test_fill_never_exceeds_one(self) -> None:
        model = CostModel(slippage_fixed=0.5)
        assert model.fill_price(0.99, stake=1.0) == 1.0

    def test_fill_stays_positive(self) -> None:
        # Defensive: a negative price would invert position sizing downstream.
        assert CostModel().fill_price(0.0001, stake=1.0) > 0.0


class TestValidation:
    """Reject configurations that would silently produce nonsense."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"taker_bps": -1.0},
            {"slippage_bps": -1.0},
            {"slippage_fixed": -0.01},
            {"depth_bps_per_unit": -1.0},
            {"depth_reference_stake": 0.0},
            {"depth_reference_stake": -5.0},
        ],
    )
    def test_negative_or_zero_parameters_raise(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            CostModel(**kwargs)  # type: ignore[arg-type]

    def test_non_positive_stake_raises(self) -> None:
        with pytest.raises(ValueError):
            CostModel().fee(stake=0.0)
        with pytest.raises(ValueError):
            CostModel().fill_price(0.5, stake=-1.0)

    def test_price_outside_zero_one_raises(self) -> None:
        with pytest.raises(ValueError):
            CostModel().fill_price(1.5, stake=1.0)
        with pytest.raises(ValueError):
            CostModel().fill_price(-0.1, stake=1.0)


class TestDeterminismAndImmutability:
    """Reproducible over clever: same inputs, same costs, forever."""

    def test_repeated_calls_are_identical(self) -> None:
        model = CostModel(slippage_bps=37.0, depth_bps_per_unit=11.0)
        first = [model.fill_price(0.3, stake=s) for s in (1, 10, 100)]
        second = [model.fill_price(0.3, stake=s) for s in (1, 10, 100)]
        assert first == second

    def test_model_is_frozen(self) -> None:
        model = CostModel()
        with pytest.raises(AttributeError):
            model.slippage_bps = 99.0  # type: ignore[misc]

    def test_to_dict_round_trips_for_the_tracker(self) -> None:
        model = CostModel(taker_bps=25.0, slippage_bps=10.0, slippage_fixed=0.002)
        payload = model.to_dict()
        assert payload["slippage_bps"] == 10.0
        assert CostModel(**payload) == model
