"""The trade policy: one decision rule for the backtest and for live trading.

A market offers a YES token at `yes_price` and its complement, the NO token, at
`1 - yes_price`. The model's `p` is the probability the *label* occurs, so the NO
token is worth `1 - p`. The edge of a side is what is left of its probability
advantage after the executable price (the cost model's fill) and the fee:

    yes_edge = p       - fill(yes_price)     - fee_rate
    no_edge  = (1 - p) - fill(1 - yes_price) - fee_rate

`EdgeSignal.best_edge` is the larger of the two and `record-trades --min-edge` is
a threshold on it. The backtest has to take the same decision, or the champion is
promoted on a strategy that is not the one that trades.
"""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest

from pmlab.core.costs import CostModel
from pmlab.core.edge import DEFAULT_MIN_EDGE, edge_clears, quote_edges
from pmlab.core.policy import TradePolicy
from pmlab.execution.edge_signal import EdgeSignal


class TestQuoteEdges:
    def test_yes_edge_is_probability_minus_price_minus_fee(self) -> None:
        # The worked number in tests/core/test_edge.py: 0.6 - 0.3 - 0.003.
        q = quote_edges(0.6, 0.3, CostModel(), stake=1.0)
        assert q.yes_edge == pytest.approx(0.297, abs=1e-12)

    def test_no_edge_prices_the_complementary_token(self) -> None:
        # NO token costs 1 - 0.3 = 0.7 and is worth 1 - 0.6 = 0.4.
        q = quote_edges(0.6, 0.3, CostModel(), stake=1.0)
        assert q.no_edge == pytest.approx(0.4 - 0.7 - 0.003, abs=1e-12)

    def test_the_model_is_bearish_so_no_is_the_better_side(self) -> None:
        q = quote_edges(0.1, 0.6, CostModel(), stake=1.0)
        assert q.no_edge == pytest.approx(0.9 - 0.4 - 0.003, abs=1e-12)
        assert q.direction == "no"
        assert q.best_edge == q.no_edge
        assert q.best_edge > q.yes_edge

    def test_a_bullish_model_takes_yes(self) -> None:
        q = quote_edges(0.9, 0.4, CostModel(), stake=1.0)
        assert q.direction == "yes"
        assert q.best_edge == q.yes_edge

    def test_slippage_is_charged_on_the_side_bought(self) -> None:
        costs = CostModel(taker_bps=0.0, slippage_bps=500.0)  # 5% of the price
        q = quote_edges(0.5, 0.4, costs, stake=1.0)
        assert q.yes_fill == pytest.approx(0.4 * 1.05, abs=1e-12)
        assert q.no_fill == pytest.approx(0.6 * 1.05, abs=1e-12)
        assert q.yes_edge == pytest.approx(0.5 - 0.42, abs=1e-12)
        assert q.no_edge == pytest.approx(0.5 - 0.63, abs=1e-12)

    def test_a_tie_goes_to_yes(self) -> None:
        # p = 0.5 at a 0.5 price: both sides have the same edge. YES is the default.
        q = quote_edges(0.5, 0.5, CostModel(taker_bps=0.0), stake=1.0)
        assert q.yes_edge == q.no_edge
        assert q.direction == "yes"

    @pytest.mark.parametrize("yes_price", [0.0, 1.0])
    def test_a_side_priced_at_zero_or_certainty_cannot_be_bought(self, yes_price: float) -> None:
        # Price 0: stake / 0 shares. Price 1: pays at most what it costs. Neither is a
        # trade, and a free token must not look like an infinite edge.
        q = quote_edges(0.9, yes_price, CostModel(taker_bps=0.0), stake=1.0)
        untradable = q.no_edge if yes_price == 1.0 else q.yes_edge
        assert untradable == -math.inf

    def test_a_fill_pushed_to_certainty_by_slippage_cannot_be_bought(self) -> None:
        costs = CostModel(taker_bps=0.0, slippage_fixed=0.5)
        q = quote_edges(0.99, 0.7, costs, stake=1.0)  # 0.7 + 0.5 caps at 1.0
        assert q.yes_fill == 1.0
        assert q.yes_edge == -math.inf

    @pytest.mark.parametrize("bad", [-0.1, 1.1, math.nan, math.inf])
    def test_a_model_probability_outside_0_1_is_refused(self, bad: float) -> None:
        with pytest.raises(ValueError, match="model_prob"):
            quote_edges(bad, 0.5, CostModel(), stake=1.0)

    @pytest.mark.parametrize("bad", [-0.1, 1.1, math.nan])
    def test_a_price_outside_0_1_is_refused(self, bad: float) -> None:
        with pytest.raises(ValueError, match="yes_price"):
            quote_edges(0.5, bad, CostModel(), stake=1.0)

    @pytest.mark.parametrize("directions", [(), ("maybe",), ("yes", "up")])
    def test_directions_must_name_yes_or_no(self, directions: tuple[str, ...]) -> None:
        with pytest.raises(ValueError, match="directions"):
            quote_edges(0.5, 0.5, CostModel(), stake=1.0, directions=directions)

    def test_the_default_threshold_is_the_one_record_trades_uses(self) -> None:
        assert DEFAULT_MIN_EDGE == 0.20


class TestEdgeClears:
    """`best_edge >= min_edge`, with float round-off forgiven: 0.6 - 0.5 is 0.0999...98."""

    def test_an_edge_equal_to_the_threshold_clears(self) -> None:
        assert edge_clears(0.2, 0.2)

    def test_round_off_below_the_threshold_still_clears(self) -> None:
        assert 0.6 - 0.5 < 0.1  # the float trap this guards against
        assert edge_clears(0.6 - 0.5, 0.1)

    def test_a_real_shortfall_does_not_clear(self) -> None:
        assert not edge_clears(0.0999, 0.1)
        assert not edge_clears(-math.inf, -1.0)

    def test_a_nan_edge_never_clears(self) -> None:
        assert not edge_clears(math.nan, -1.0)


class TestTradePolicy:
    def test_the_legacy_policy_is_yes_only_with_no_threshold(self) -> None:
        legacy = TradePolicy.top_bin_yes()
        assert legacy.selection == "top_prob"
        assert legacy.directions == ("yes",)
        assert legacy.min_edge is None

    def test_the_live_policy_buys_both_sides_above_the_default_threshold(self) -> None:
        live = TradePolicy.live()
        assert live.selection == "best_edge"
        assert live.directions == ("yes", "no")
        assert live.min_edge == DEFAULT_MIN_EDGE

    def test_round_trips_through_a_dict(self) -> None:
        for policy in (TradePolicy.top_bin_yes(), TradePolicy.live(min_edge=0.05)):
            assert TradePolicy.from_dict(policy.to_dict()) == policy

    def test_a_yes_only_edge_policy_is_allowed(self) -> None:
        policy = TradePolicy.live(min_edge=0.1, directions=("yes",))
        assert policy.directions == ("yes",)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"selection": "top_prob", "directions": ("yes", "no"), "min_edge": None},
            {"selection": "top_prob", "directions": ("yes",), "min_edge": 0.1},
            {"selection": "best_edge", "directions": (), "min_edge": 0.1},
            {"selection": "best_edge", "directions": ("yes", "maybe"), "min_edge": 0.1},
            {"selection": "best_edge", "directions": ("yes", "yes"), "min_edge": 0.1},
            {"selection": "best_edge", "directions": ("yes",), "min_edge": None},
            {"selection": "best_edge", "directions": ("yes",), "min_edge": math.nan},
            {"selection": "random", "directions": ("yes",), "min_edge": 0.1},
        ],
    )
    def test_an_incoherent_policy_is_refused(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            TradePolicy(**kwargs)  # type: ignore[arg-type]

    def test_from_dict_refuses_unknown_keys(self) -> None:
        with pytest.raises(ValueError, match="unknown"):
            TradePolicy.from_dict(
                {"selection": "best_edge", "directions": ["yes"], "min_edge": 0.1, "x": 1}
            )


class TestEdgeSignalFromQuote:
    """A live signal is built by the same arithmetic the backtest uses."""

    def test_fields_match_quote_edges(self) -> None:
        costs = CostModel(taker_bps=30.0, slippage_bps=50.0)
        signal = EdgeSignal.from_quote(
            market_id="m1",
            city_or_segment="politics",
            target_date="2099-01-01",
            horizon="previous_evening",
            outcome_label="warm",
            yes_price=0.55,
            model_prob=0.15,
            costs=costs,
            stake=1.0,
        )
        assert signal is not None
        q = quote_edges(0.15, 0.55, costs, stake=1.0)
        assert signal.direction == "no" == q.direction
        assert signal.yes_edge == q.yes_edge
        assert signal.no_edge == q.no_edge
        assert signal.best_edge == q.best_edge
        assert signal.gamma_price == 0.55  # always the YES price, as before
        assert signal.model_prob == 0.15  # always the probability of the label

    def test_restricting_directions_to_yes_ignores_a_better_no(self) -> None:
        signal = EdgeSignal.from_quote(
            market_id="m1",
            city_or_segment="politics",
            target_date="2099-01-01",
            horizon="previous_evening",
            outcome_label="warm",
            yes_price=0.55,
            model_prob=0.15,
            costs=CostModel(),
            stake=1.0,
            directions=("yes",),
        )
        assert signal is not None
        assert signal.direction == "yes"
        assert signal.best_edge == signal.yes_edge  # negative: the caller's threshold rejects it

    def test_no_signal_when_neither_allowed_side_can_be_bought(self) -> None:
        signal = EdgeSignal.from_quote(
            market_id="m1",
            city_or_segment="politics",
            target_date="2099-01-01",
            horizon="previous_evening",
            outcome_label="warm",
            yes_price=1.0,
            model_prob=0.9,
            costs=CostModel(taker_bps=0.0),
            stake=1.0,
            directions=("no",),
        )
        assert signal is None


class TestNumpyScalarsAreNumbers:
    """Model outputs and DataFrame cells arrive as numpy scalars, not Python floats."""

    @pytest.mark.parametrize("make", [np.float32, np.float64, np.longdouble])
    def test_a_numpy_float_probability_and_price(self, make: type) -> None:
        q = quote_edges(make(0.6), make(0.3), CostModel(), stake=1.0)  # type: ignore[arg-type]
        assert q.yes_edge == pytest.approx(0.297, abs=1e-6)

    def test_a_numpy_integer_endpoint(self) -> None:
        q = quote_edges(np.int64(1), np.int64(0), CostModel(taker_bps=0.0), stake=1.0)  # type: ignore[arg-type]
        assert q.yes_edge == -math.inf  # priced at 0: not a trade

    def test_a_numpy_bool_is_still_not_a_probability(self) -> None:
        with pytest.raises(ValueError, match="model_prob"):
            quote_edges(np.bool_(True), 0.5, CostModel(), stake=1.0)  # type: ignore[arg-type]

    def test_a_signal_built_from_numpy_scalars_holds_plain_floats(self) -> None:
        signal = EdgeSignal.from_quote(
            market_id="m",
            city_or_segment="s",
            target_date="2099-01-01",
            horizon="morning_of",
            outcome_label="x",
            yes_price=np.float32(0.55),  # type: ignore[arg-type]
            model_prob=np.float32(0.15),  # type: ignore[arg-type]
            costs=CostModel(),
            stake=1.0,
        )
        assert signal is not None
        assert type(signal.gamma_price) is float
        assert type(signal.model_prob) is float
        json.dumps(dataclasses.asdict(signal), allow_nan=False)


class TestASignalIsStrictJson:
    """A side that cannot be bought has an edge of -inf inside the policy. On a signal that
    would be written as `-Infinity`, which strict JSON parsers refuse."""

    def test_the_unbuyable_side_reads_as_losing_the_whole_stake(self) -> None:
        signal = EdgeSignal.from_quote(
            market_id="m",
            city_or_segment="s",
            target_date="2099-01-01",
            horizon="morning_of",
            outcome_label="x",
            yes_price=0.7,
            model_prob=0.2,
            costs=CostModel(taker_bps=0.0, slippage_fixed=0.5),  # YES fills at 1.0: no trade
            stake=1.0,
        )
        assert signal is not None
        assert signal.direction == "no"
        assert signal.yes_edge == -1.0
        json.dumps(dataclasses.asdict(signal), allow_nan=False)

    def test_the_policy_itself_still_sees_minus_infinity(self) -> None:
        q = quote_edges(0.2, 0.7, CostModel(taker_bps=0.0, slippage_fixed=0.5), stake=1.0)
        assert q.yes_edge == -math.inf
