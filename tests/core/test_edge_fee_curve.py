"""The edge subtracts the price-dependent fee, and Kelly sizes on the post-fee payoff (#72).

``edge = p - fill(price) - fee_per_stake(fill)`` where, for a Polymarket taker, the fee on one unit
of stake is ``rate x (1 - fill)``. The flat 30 bps it replaced was the same 0.003 at every price.
"""

from __future__ import annotations

import pytest

from pmlab.core.costs import CostModel
from pmlab.core.edge import quote_edges
from pmlab.core.fee_schedule import FeeSchedule
from pmlab.core.sizing import kelly_fraction, kelly_stake_size

FLAT = CostModel(fee_model="flat-bps")


class TestQuoteEdgesUseThePriceDependentFee:
    def test_the_default_edge_pays_rate_times_one_minus_the_fill(self) -> None:
        edges = quote_edges(0.6, 0.4, CostModel(), stake=1.0)
        # YES: fill 0.40, fee 0.05 x 0.60 = 0.03.   NO: the NO token costs 0.60, fee 0.05 x 0.40.
        assert edges.yes_edge == pytest.approx(0.6 - 0.4 - 0.03, abs=1e-12)
        assert edges.no_edge == pytest.approx(0.4 - 0.6 - 0.02, abs=1e-12)

    def test_the_flat_model_gives_the_numbers_it_always_gave(self) -> None:
        edges = quote_edges(0.6, 0.4, FLAT, stake=1.0)
        assert edges.yes_edge == pytest.approx(0.6 - 0.4 - 0.003, abs=1e-12)
        assert edges.no_edge == pytest.approx(0.4 - 0.6 - 0.003, abs=1e-12)

    def test_the_fee_is_taken_at_the_fill_not_at_the_quote(self) -> None:
        costs = CostModel(slippage_bps=100.0)  # fill = 0.404
        edges = quote_edges(0.6, 0.4, costs, stake=1.0)
        assert edges.yes_fill == pytest.approx(0.404)
        assert edges.yes_edge == pytest.approx(0.6 - 0.404 - 0.05 * (1 - 0.404), abs=1e-12)

    def test_a_cheap_token_pays_more_of_its_stake_than_a_dear_one(self) -> None:
        cheap = quote_edges(0.5, 0.10, CostModel(), stake=1.0)
        dear = quote_edges(0.5, 0.90, CostModel(), stake=1.0)
        assert 0.5 - 0.10 - cheap.yes_edge == pytest.approx(0.05 * 0.90, abs=1e-12)
        assert 0.5 - 0.90 - dear.yes_edge == pytest.approx(0.05 * 0.10, abs=1e-12)

    def test_the_edge_follows_the_schedule_it_is_given(self) -> None:
        costs = CostModel(fee_schedule=FeeSchedule(rate=0.01))
        assert quote_edges(0.6, 0.4, costs, stake=1.0).yes_edge == pytest.approx(
            0.6 - 0.4 - 0.006, abs=1e-12
        )

    def test_a_fee_free_market_has_the_gross_edge(self) -> None:
        costs = CostModel(fee_schedule=FeeSchedule(rate=0.05, enabled=False))
        assert quote_edges(0.6, 0.4, costs, stake=1.0).yes_edge == pytest.approx(0.2, abs=1e-12)

    def test_the_edge_is_the_after_fee_edge_of_the_trade_that_is_paid(self) -> None:
        # the fee in the edge is the fee the trade pays (rounded to 5 decimals), not an
        # unrounded curve value: a small stake must not see a different edge than PnL does
        costs = CostModel()
        stake = 0.013
        edges = quote_edges(0.6, 0.37, costs, stake=stake)
        assert edges.yes_edge == pytest.approx(
            0.6 - 0.37 - costs.fee(stake, 0.37) / stake, abs=1e-12
        )


class TestKellyUsesThePostFeePayoff:
    def test_without_costs_kelly_is_what_it_was(self) -> None:
        assert kelly_fraction(0.6, 0.4) == pytest.approx(0.25 * (0.6 - 0.4) / (1 - 0.4))

    def test_a_polymarket_fee_lowers_the_fraction(self) -> None:
        # one unit of stake at 0.40 costs 1.03 with the fee: 0.412 per share-worth of payoff
        cost = 0.4 * (1 + 0.05 * 0.6)
        fraction = kelly_fraction(0.6, 0.4, costs=CostModel())
        assert fraction == pytest.approx(0.25 * (0.6 - cost) / (1 - cost), abs=1e-9)
        assert fraction < kelly_fraction(0.6, 0.4)

    def test_a_flat_bps_model_lowers_it_by_its_rate(self) -> None:
        cost = 0.4 * (1 + 0.003)
        assert kelly_fraction(0.6, 0.4, costs=FLAT) == pytest.approx(
            0.25 * (0.6 - cost) / (1 - cost), abs=1e-9
        )

    def test_an_edge_the_fee_eats_is_not_sized(self) -> None:
        assert kelly_fraction(0.41, 0.40) > 0.0
        assert kelly_fraction(0.41, 0.40, costs=CostModel()) == 0.0

    def test_the_stake_leaves_room_for_the_fee_under_the_exposure_cap(self) -> None:
        costs = CostModel()
        stake = kelly_stake_size(0.6, 0.4, bankroll=1000.0, max_exposure=0.05, costs=costs)
        assert stake < 50.0
        # the cap is on what leaves the bankroll: stake plus fee
        assert stake + costs.fee(stake, 0.4) == pytest.approx(50.0, abs=1e-4)

    def test_the_stake_without_costs_is_the_capped_fraction_as_before(self) -> None:
        assert kelly_stake_size(0.6, 0.4, bankroll=1000.0, max_exposure=0.05) == pytest.approx(50.0)

    def test_an_uncapped_stake_plus_its_fee_is_the_fraction_of_the_bankroll(self) -> None:
        costs = CostModel()
        stake = kelly_stake_size(0.6, 0.4, bankroll=1000.0, max_exposure=1.0, costs=costs)
        fraction = kelly_fraction(0.6, 0.4, costs=costs)
        assert stake + costs.fee(stake, 0.4) == pytest.approx(1000.0 * fraction, abs=1e-4)

    def test_costs_must_be_a_cost_model(self) -> None:
        with pytest.raises(TypeError, match="CostModel"):
            kelly_fraction(0.6, 0.4, costs=0.05)  # type: ignore[arg-type]
