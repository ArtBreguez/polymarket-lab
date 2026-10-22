"""Issue #28: the weather panel must show the backtest every bin, at its own price.

The world (``synthetic_world``) prices every bin at its true probability, so a backtest
that sees all of them has an expected PnL of zero under any trade policy. ``pnl_z_score``
turns that into a tolerance that follows from the sample: the PnL of N independent
mean-zero trades is within 4 standard deviations of zero except with probability ~6e-5.

* the per-bin panel the plugin builds must land inside that band;
* the one-winning-row-per-market panel (the legacy ``build_training_row``) must land far
  outside it, which also shows the band is tight enough to see the bias at all.
"""

from __future__ import annotations

from collections import Counter
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from pmlab.backtest.rolling_origin import rolling_origin_eval
from pmlab.core.costs import CostModel
from pmlab.core.policy import TradePolicy
from pmlab.modeling.lgbm_baseline import LGBMForecaster
from tests.plugins.weather_tmax import synthetic_world as sw

# |z| above this is not noise: P(|N(0,1)| > 4) ~ 6e-5, and the world is seeded anyway.
Z_BAND = 4.0
SEEDS = [7, 11, 23]
# stride=2 evaluates ~29 dates x 8 markets, which is the sample the band is sized for.
STRIDE = 2
MIN_TRADES = 150

POLICIES = {
    "live": TradePolicy.live(min_edge=0.05),
    "top-bin-yes": TradePolicy.top_bin_yes(),
}


class _BaseRate:
    """Predicts the share of positive labels it was trained on, for every row."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._p = float(y.mean())

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([np.full(len(X), 1 - self._p), np.full(len(X), self._p)])


def _backtest(
    panel: pd.DataFrame, policy: TradePolicy, model: object | None = None
) -> pd.DataFrame:
    result = rolling_origin_eval(
        panel,
        model or LGBMForecaster(),  # type: ignore[arg-type]
        min_train_rows=20,
        stride=STRIDE,
        costs=CostModel.frictionless(),
        policy=policy,
    )
    return result.trades


@pytest.fixture(scope="module", params=SEEDS)
def world(request: pytest.FixtureRequest) -> sw.World:
    return sw.make_world(seed=request.param)


class TestPerBinPanelIsUnbiased:
    def test_panel_has_every_bin_of_every_market(self, world: sw.World) -> None:
        panel = sw.plugin_panel(world)
        assert len(panel) == len(world.markets) * len(sw.LABELS)
        per_market = panel.groupby("market_id")["outcome_label"].apply(sorted)
        assert (per_market.map(tuple) == tuple(sorted(sw.LABELS))).all()

    def test_panel_contains_the_losing_bins(self, world: sw.World) -> None:
        panel = sw.plugin_panel(world)
        won = panel["outcome_label"] == panel["winning_label"]
        assert won.sum() == len(world.markets)  # one winner per market...
        assert (~won).sum() == len(world.markets) * (len(sw.LABELS) - 1)  # ...and the losers

    def test_each_bin_is_priced_at_its_own_decision_time_price(self, world: sw.World) -> None:
        panel = sw.plugin_panel(world)
        expected = pd.DataFrame(sw.reference_panel_rows(world))
        keys = ["market_id", "outcome_label"]
        got = panel.sort_values(keys).reset_index(drop=True)
        want = expected.sort_values(keys).reset_index(drop=True)
        for col in ("market_price", "winning_label", "decision_date", "feature_fcst_max"):
            assert got[col].tolist() == want[col].tolist(), col

    @pytest.mark.parametrize("policy_name", list(POLICIES))
    def test_backtest_pnl_is_within_noise_of_zero(self, world: sw.World, policy_name: str) -> None:
        trades = _backtest(sw.plugin_panel(world), POLICIES[policy_name])
        total, sd, z = sw.pnl_z_score(trades)
        assert len(trades) >= MIN_TRADES, "too few trades for the band to mean anything"
        assert abs(z) < Z_BAND, f"PnL {total:+.2f} is {z:+.1f} sd from 0 (sd={sd:.2f})"

    def test_a_market_is_one_trade_however_many_bins_it_has(self, world: sw.World) -> None:
        trades = _backtest(sw.plugin_panel(world), POLICIES["live"])
        assert trades.groupby(["market_id", "eval_date"]).size().max() == 1


class TestLegacyWinnerOnlyPanelIsBiased:
    """Control: the instrument must be able to see the bias the issue describes."""

    def test_legacy_panel_has_one_row_per_market_and_only_winners(self, world: sw.World) -> None:
        panel = sw.legacy_panel(world)
        assert len(panel) == len(world.markets)
        assert (panel["outcome_label"] == panel["winning_label"]).all()

    def test_top_bin_yes_always_wins(self, world: sw.World) -> None:
        trades = _backtest(sw.legacy_panel(world), POLICIES["top-bin-yes"])
        total, _, z = sw.pnl_z_score(trades)
        assert len(trades) >= MIN_TRADES
        assert (trades["realized_pnl"] > 0).all()  # a 100% hit rate on a fair-price world
        assert total > 0
        assert z > 2 * Z_BAND

    @pytest.mark.parametrize("model_name", ["lgbm", "base-rate"])
    def test_live_policy_is_far_from_zero_whichever_way_the_model_leans(
        self, world: sw.World, model_name: str
    ) -> None:
        # The sign depends on the model, the size does not. A base-rate model learns p=1
        # from all-positive labels and buys YES on winners; LightGBM, fitted on one class,
        # reports p~1e-15 and buys NO on winners. Either way every trade is a foregone
        # conclusion, and the PnL sits ~18 sd from where fair prices put it.
        model = _BaseRate() if model_name == "base-rate" else None
        trades = _backtest(sw.legacy_panel(world), POLICIES["live"], model)
        _, _, z = sw.pnl_z_score(trades)
        assert len(trades) >= MIN_TRADES
        assert abs(z) > 2 * Z_BAND


class TestADecisionAfterTheMarketClosedIsNotATrainingRow:
    """The review's case on the 5-bin world: prices seen the day the market resolved (it was
    discovered then), a decision made at or after its close."""

    def test_no_row_is_built_and_each_market_is_counted_once(self) -> None:
        world = sw.make_world(n_days=2, seed=7)
        day = world.decision_dates[0]
        plugin = sw._plugin_for_day(world, day)  # discovered 07:30Z on `day`
        market = world.by_day[day][0]
        spec = next(s for s in plugin.discover_markets() if s.market_id == market.market_id)
        closing = f"{market.target_date_iso}T23:59:00Z"  # its endDate
        next_day = (date.fromisoformat(market.target_date_iso) + timedelta(days=1)).isoformat()
        counts: Counter[str] = Counter()
        for decision in (closing, f"{next_day}T00:00:00Z"):
            rows = plugin.build_training_rows(
                spec, "morning_of", decision_time=decision, exclusions=counts
            )
            assert rows == []
        assert counts == {"decision_after_close": 2}

    def test_the_target_day_itself_still_builds_all_five_bins(self) -> None:
        world = sw.make_world(n_days=2, seed=7)
        day = world.decision_dates[0]
        plugin = sw._plugin_for_day(world, day)
        market = world.by_day[day][0]
        spec = next(s for s in plugin.discover_markets() if s.market_id == market.market_id)
        morning_of = f"{market.target_date_iso}T08:00:00Z"
        # discovered 07:30Z on `day`, decided on the morning of the target day: in time, open
        rows = plugin.build_training_rows(spec, "morning_of", decision_time=morning_of)
        assert len(rows) == len(sw.LABELS)
