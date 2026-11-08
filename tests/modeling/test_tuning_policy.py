"""TunedForecaster scores trials with the trade policy it is told to optimise (issue #36).

The historical objective (YES on the top-probability bin, every market) and the live one
(`TradePolicy.live`: best after-fee edge, YES or NO, if it clears `min_edge`) rank
hyper-parameters differently. These tests pin that the tuner can use either, that the
default is still the historical one, and that the objective used is recorded.
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pmlab.backtest.rolling_origin import rolling_origin_eval
from pmlab.core.costs import CostModel
from pmlab.core.policy import TradePolicy
from pmlab.modeling import tuning
from pmlab.modeling.tuning import TunedForecaster
from tests.modeling.tuning_helpers import (
    COL_A,
    COL_B,
    PickFeature,
    build_pick,
    crafted_panel,
    pick_space,
)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("optuna") is None, reason="optuna not installed"
)

LIVE_20 = TradePolicy.live(min_edge=0.20)
LIVE_05 = TradePolicy.live(min_edge=0.05)


def _tuner(**kwargs: Any) -> TunedForecaster:
    base: dict[str, Any] = {
        "n_trials": 12,
        "metric": "total_pnl",
        "stride": 1,
        "min_train_rows": 20,
        "random_state": 0,
    }
    return TunedForecaster(build_pick, pick_space, **{**base, **kwargs})


def _explored(tuner: TunedForecaster) -> set[str]:
    assert tuner.study_ is not None
    return {t.params["col"] for t in tuner.study_.trials}


class TestTheCraftedPanelIsWhatTheDocstringSays:
    """The premise of the tests below, checked against the backtest itself (not the tuner)."""

    @staticmethod
    def _total(col: str, policy: TradePolicy | None) -> float | None:
        trades = rolling_origin_eval(
            crafted_panel(), PickFeature(col), min_train_rows=20, stride=1, policy=policy
        ).trades
        return None if trades.empty else float(trades["realized_pnl"].sum())

    def test_top_prob_ranks_a_over_b(self) -> None:
        a, b = self._total(COL_A, None), self._total(COL_B, None)
        assert a is not None and b is not None and a > 0 > b

    def test_live_at_the_default_threshold_ranks_b_over_a(self) -> None:
        # A never believes it beats a price by 0.20: no trade at all, so nothing to score.
        assert self._total(COL_A, LIVE_20) is None
        b = self._total(COL_B, LIVE_20)
        assert b is not None and b > 0

    def test_live_at_a_low_threshold_ranks_a_over_b(self) -> None:
        a, b = self._total(COL_A, LIVE_05), self._total(COL_B, LIVE_05)
        assert a is not None and b is not None and a > b


class TestObjectiveChangesTheChosenParams:
    def test_default_objective_picks_a(self) -> None:
        tuner = _tuner().tune(crafted_panel())
        assert _explored(tuner) == {COL_A, COL_B}
        assert tuner.best_params_ == {"col": COL_A}

    def test_live_policy_picks_b(self) -> None:
        tuner = _tuner(policy=LIVE_20).tune(crafted_panel())
        assert _explored(tuner) == {COL_A, COL_B}
        assert tuner.best_params_ == {"col": COL_B}

    def test_the_threshold_alone_flips_the_choice(self) -> None:
        low = _tuner(policy=LIVE_05).tune(crafted_panel())
        high = _tuner(policy=LIVE_20).tune(crafted_panel())
        assert low.best_params_ == {"col": COL_A}
        assert high.best_params_ == {"col": COL_B}

    def test_best_score_is_the_score_under_that_policy(self) -> None:
        panel = crafted_panel()
        tuner = _tuner(policy=LIVE_20).tune(panel)
        expected = rolling_origin_eval(
            panel, PickFeature(COL_B), min_train_rows=20, stride=1, policy=LIVE_20
        ).trades["realized_pnl"]
        assert tuner.best_score_ == pytest.approx(float(expected.sum()))

    @pytest.mark.parametrize("metric", ["total_pnl", "mean_pnl", "hit_rate"])
    def test_every_metric_scores_the_trades_the_policy_took(self, metric: str) -> None:
        panel = crafted_panel()
        tuner = _tuner(policy=LIVE_20, metric=metric).tune(panel)
        assert tuner.best_params_ == {"col": COL_B}  # A takes no trade: -inf under any metric
        pnl = rolling_origin_eval(
            panel, PickFeature(COL_B), min_train_rows=20, stride=1, policy=LIVE_20
        ).trades["realized_pnl"]
        expected = {
            "total_pnl": float(pnl.sum()),
            "mean_pnl": float(pnl.mean()),
            "hit_rate": float((pnl > 0).mean()),
        }[metric]
        assert tuner.best_score_ == pytest.approx(expected)

    def test_the_refit_model_is_the_winner_of_that_objective(self) -> None:
        panel = crafted_panel()
        tuner = _tuner(policy=LIVE_20).tune(panel)
        feats = panel[[COL_A, COL_B]]
        np.testing.assert_allclose(tuner.predict_proba(feats)[:, 1], panel[COL_B])


class TestTheObjectiveIsPassedThrough:
    """Spy on the scoring backtest: it must see the policy and the costs of the tuner."""

    @staticmethod
    def _spy(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        real = tuning.rolling_origin_eval

        def spy(panel: pd.DataFrame, model: Any, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return real(panel, model, **kwargs)

        monkeypatch.setattr(tuning, "rolling_origin_eval", spy)
        return calls

    def test_policy_and_costs_reach_every_trial(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self._spy(monkeypatch)
        costs = CostModel(taker_bps=30.0, slippage_bps=50.0)
        _tuner(n_trials=3, policy=LIVE_05, costs=costs).tune(crafted_panel())
        assert len(calls) == 3
        for kw in calls:
            assert kw["policy"] is LIVE_05
            assert kw["costs"] == costs
            assert kw["stride"] == 1 and kw["min_train_rows"] == 20

    def test_default_hands_the_backtest_no_policy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self._spy(monkeypatch)
        _tuner(n_trials=2).tune(crafted_panel())
        assert [kw.get("policy") for kw in calls] == [None, None]
        assert all(kw["costs"] == CostModel() for kw in calls)

    def test_taker_bps_still_sets_the_fee(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = self._spy(monkeypatch)
        _tuner(n_trials=2, taker_bps=75.0).tune(crafted_panel())
        assert all(kw["costs"] == CostModel(taker_bps=75.0) for kw in calls)

    def test_costs_change_the_live_score(self) -> None:
        """The live rule prices the fill: a slippage that eats B's edge must show in the score."""
        panel = crafted_panel()
        free = _tuner(policy=LIVE_20).tune(panel)
        costly = _tuner(policy=LIVE_20, costs=CostModel(slippage_bps=3000.0)).tune(panel)
        assert costly.best_score_ < free.best_score_


class TestDefaultIsUnchanged:
    """Without `policy` the search is the one that shipped before: the same score per trial."""

    def test_trial_scores_are_the_historical_backtest_scores(self) -> None:
        panel = crafted_panel()
        tuner = _tuner(taker_bps=45.0).tune(panel)
        assert tuner.study_ is not None
        for trial in tuner.study_.trials:
            trades = rolling_origin_eval(
                panel,
                PickFeature(trial.params["col"]),
                min_train_rows=20,
                stride=1,
                flat_stake=1.0,
                taker_bps=45.0,
            ).trades
            assert trial.value == pytest.approx(float(trades["realized_pnl"].sum()))

    def test_explicit_top_bin_yes_is_the_default(self) -> None:
        implicit = _tuner().tune(crafted_panel())
        explicit = _tuner(policy=TradePolicy.top_bin_yes()).tune(crafted_panel())
        assert explicit.best_params_ == implicit.best_params_
        assert explicit.best_score_ == implicit.best_score_

    def test_signature_stays_backward_compatible(self) -> None:
        tuner = TunedForecaster(build_pick, pick_space)
        assert tuner.policy is None
        assert tuner.taker_bps == 30.0
        assert tuner.costs == CostModel()
        assert (tuner.stride, tuner.min_train_rows, tuner.flat_stake) == (10, 20, 1.0)


class TestValidation:
    def test_policy_must_be_a_trade_policy(self) -> None:
        with pytest.raises(TypeError, match="TradePolicy"):
            _tuner(policy="live")

    def test_costs_must_be_a_cost_model(self) -> None:
        with pytest.raises(TypeError, match="CostModel"):
            _tuner(costs={"taker_bps": 30.0})

    def test_costs_and_taker_bps_are_two_sources_of_the_fee(self) -> None:
        with pytest.raises(ValueError, match="taker_bps or costs"):
            _tuner(costs=CostModel(), taker_bps=30.0)

    def test_a_search_that_never_trades_says_so(self) -> None:
        # Nothing clears a 0.99 edge: every trial is -inf and the "best" params are noise.
        with pytest.warns(UserWarning, match="no trial produced a trade"):
            tuner = _tuner(n_trials=3, policy=TradePolicy.live(min_edge=0.99)).tune(crafted_panel())
        assert tuner.best_score_ == float("-inf")

    def test_a_search_that_trades_does_not_warn(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _tuner(n_trials=3, policy=LIVE_05).tune(crafted_panel())


class TestNoLookahead:
    """Scoring under the live policy trains on strictly earlier dates, like the default."""

    def test_every_fit_precedes_the_rows_it_predicts(self) -> None:
        log: list[tuple[str, float]] = []

        class Spy:
            def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
                log.append(("fit", float(X["feature_day"].max())))

            def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
                log.append(("predict", float(X["feature_day"].min())))
                p = X[COL_B].to_numpy(float)
                return np.column_stack([1.0 - p, p])

        panel = crafted_panel()
        panel["feature_day"] = pd.to_datetime(panel["decision_date"]).map(pd.Timestamp.toordinal)
        TunedForecaster(
            lambda params: Spy(),
            lambda trial: {"k": trial.suggest_int("k", 0, 3)},
            n_trials=2,
            stride=1,
            min_train_rows=20,
            policy=LIVE_05,
        ).tune(panel)
        pairs = [
            (log[i][1], log[i + 1][1])
            for i in range(len(log) - 1)
            if log[i][0] == "fit" and log[i + 1][0] == "predict"
        ]
        assert pairs
        assert all(trained_to < predicted_from for trained_to, predicted_from in pairs)


class TestTheObjectiveIsRecorded:
    def test_tuning_config_spells_out_every_default(self) -> None:
        cfg = _tuner(n_trials=5).tune(crafted_panel()).tuning_config()
        assert cfg["metric"] == "total_pnl"
        assert cfg["n_trials"] == 5
        assert cfg["policy"] == TradePolicy.top_bin_yes().to_dict()
        assert cfg["costs"] == CostModel().to_dict()
        assert (cfg["stride"], cfg["min_train_rows"], cfg["flat_stake"]) == (1, 20, 1.0)

    def test_tuning_config_carries_the_live_policy_and_costs(self) -> None:
        costs = CostModel(taker_bps=30.0, slippage_bps=50.0)
        cfg = _tuner(policy=LIVE_20, costs=costs).tune(crafted_panel()).tuning_config()
        assert cfg["policy"] == LIVE_20.to_dict()
        assert cfg["costs"] == costs.to_dict()

    def test_tuning_config_is_strict_json_even_for_a_blind_search(self) -> None:
        import json

        with pytest.warns(UserWarning):
            tuner = _tuner(n_trials=2, policy=TradePolicy.live(min_edge=0.99)).tune(crafted_panel())
        cfg = tuner.tuning_config()
        assert cfg["best_score"] is None  # -inf is not JSON
        json.dumps(cfg, allow_nan=False)

    def test_tuning_config_best_score_is_the_search_result(self) -> None:
        tuner = _tuner(policy=LIVE_20).tune(crafted_panel())
        assert tuner.tuning_config()["best_score"] == pytest.approx(tuner.best_score_)

    def test_same_config_same_data_same_params(self) -> None:
        first = _tuner(policy=LIVE_20, random_state=3).tune(crafted_panel())
        again = _tuner(policy=LIVE_20, random_state=3).tune(crafted_panel())
        assert first.best_params_ == again.best_params_
        assert first.best_score_ == again.best_score_

    def test_save_load_keeps_the_objective(self, tmp_path: Path) -> None:
        costs = CostModel(taker_bps=30.0, slippage_bps=50.0)
        tuner = _tuner(policy=LIVE_20, costs=costs).tune(crafted_panel())
        tuner.save(tmp_path / "t.pkl")
        loaded = TunedForecaster.load(tmp_path / "t.pkl")
        assert loaded.policy == LIVE_20
        assert loaded.costs == costs
        assert loaded.tuning_config() == tuner.tuning_config()

    def test_a_tuner_pickled_before_the_policy_existed_still_loads(self, tmp_path: Path) -> None:
        import pickle

        tuner = _tuner().tune(crafted_panel())
        tuner.save(tmp_path / "new.pkl")
        with open(tmp_path / "new.pkl", "rb") as f:
            data = pickle.load(f)
        for key in ("policy", "costs", "tuning"):
            data.pop(key, None)
        with open(tmp_path / "old.pkl", "wb") as f:
            pickle.dump(data, f)
        loaded = TunedForecaster.load(tmp_path / "old.pkl")
        assert loaded.policy is None  # the historical objective, which is what produced it
        assert loaded.costs == CostModel()
        assert loaded.best_params_ == tuner.best_params_
