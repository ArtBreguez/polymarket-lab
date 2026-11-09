"""Costs, fees, Kelly sizing and the backtest's own arguments are checked before they are used (#59)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pmlab import (
    CostModel,
    TradePolicy,
    estimate_fee,
    kelly_fraction,
    kelly_stake_size,
    rolling_origin_eval,
)

NAN, INF = float("nan"), float("inf")


class TestCostModel:
    @pytest.mark.parametrize(
        "field", ["taker_bps", "slippage_bps", "slippage_fixed", "depth_bps_per_unit"]
    )
    @pytest.mark.parametrize("bad", [NAN, INF, -INF, -1.0])
    def test_non_finite_or_negative_is_refused(self, field, bad):
        with pytest.raises(ValueError, match=field):
            CostModel(**{field: bad})

    @pytest.mark.parametrize("bad", [NAN, INF, 0.0, -1.0])
    def test_reference_stake(self, bad):
        with pytest.raises(ValueError, match="depth_reference_stake"):
            CostModel(depth_reference_stake=bad)

    def test_a_string_is_a_value_error_not_a_type_error(self):
        with pytest.raises(ValueError, match="taker_bps"):
            CostModel(taker_bps="30")  # type: ignore[arg-type]

    def test_ordinary_models_still_build(self):
        CostModel()
        CostModel(taker_bps=30, slippage_bps=50, depth_bps_per_unit=20, depth_reference_stake=10)
        CostModel.frictionless()


class TestEstimateFee:
    def test_values(self):
        assert estimate_fee(100, 30) == pytest.approx(0.3)
        assert estimate_fee(0, 30) == 0.0
        assert estimate_fee(5.0) == pytest.approx(0.015)

    @pytest.mark.parametrize(
        "stake,bps", [(-5, 30), (5, -30), (NAN, 30), (5, NAN), (INF, 30), (5, INF)]
    )
    def test_a_fee_is_never_negative_or_non_finite(self, stake, bps):
        with pytest.raises(ValueError):
            estimate_fee(stake, bps)


class TestKelly:
    def test_values_unchanged(self):
        assert kelly_fraction(0.6, 0.5) == pytest.approx(0.05)
        assert kelly_fraction(0.4, 0.5) == 0.0
        assert kelly_fraction(0.6, 0.0) == 0.0  # no valid price: no bet, as before
        assert kelly_stake_size(0.6, 0.5, 100) == pytest.approx(5.0)

    @pytest.mark.parametrize("p", [1.5, -0.1, NAN])
    def test_win_prob(self, p):
        with pytest.raises(ValueError, match="win_prob"):
            kelly_fraction(p, 0.5)

    @pytest.mark.parametrize("f", [-1.0, NAN, INF])
    def test_fraction(self, f):
        with pytest.raises(ValueError, match="fraction"):
            kelly_fraction(0.6, 0.5, fraction=f)

    @pytest.mark.parametrize("bankroll", [-100.0, NAN, INF])
    def test_bankroll(self, bankroll):
        with pytest.raises(ValueError, match="bankroll"):
            kelly_stake_size(0.6, 0.5, bankroll)

    @pytest.mark.parametrize("cap", [-1.0, 2.0, NAN])
    def test_max_exposure(self, cap):
        with pytest.raises(ValueError, match="max_exposure"):
            kelly_stake_size(0.6, 0.5, 100, max_exposure=cap)


def _panel() -> pd.DataFrame:
    rows = []
    for d in range(30):
        for lab in ("a", "b"):
            rows.append(
                {
                    "market_id": f"m{d}",
                    "decision_date": f"2026-01-{d + 1:02d}",
                    "outcome_label": lab,
                    "winning_label": "a",
                    "market_price": 0.4 if lab == "a" else 0.6,
                    "feature_x": float(d),
                }
            )
    return pd.DataFrame(rows)


class Spy:
    def __init__(self) -> None:
        self.fits = 0

    def fit(self, X, y):
        self.fits += 1

    def predict_proba(self, X):
        p = np.full(len(X), 0.5)
        return np.c_[1 - p, p]


class TestRollingOriginArguments:
    def _run(self, **kw):
        spy = Spy()
        args = dict(min_train_rows=4, stride=3)
        args.update(kw)
        model = args.pop("model", spy)
        panel = args.pop("panel", _panel())
        try:
            return rolling_origin_eval(panel, model, **args)
        finally:
            assert spy.fits == 0, "arguments must be refused before the first fit"

    def test_a_good_call_runs(self):
        res = rolling_origin_eval(_panel(), Spy(), min_train_rows=4, stride=3)
        assert len(res.steps) > 0

    @pytest.mark.parametrize("stride", [0, -1])
    def test_stride_must_be_positive(self, stride):
        with pytest.raises(ValueError, match="stride"):
            self._run(stride=stride)

    @pytest.mark.parametrize("stride", [None, 1.5, True, "3"])
    def test_stride_must_be_an_int(self, stride):
        with pytest.raises(TypeError, match="stride"):
            self._run(stride=stride)

    def test_min_train_rows(self):
        with pytest.raises(ValueError, match="min_train_rows"):
            self._run(min_train_rows=-1)
        for bad in (None, 2.5, True):
            with pytest.raises(TypeError, match="min_train_rows"):
                self._run(min_train_rows=bad)

    def test_zero_min_train_rows_never_fits_on_nothing(self):
        spy = Spy()
        res = rolling_origin_eval(_panel(), spy, min_train_rows=0, stride=3)
        assert all(s["train_rows"] > 0 for s in res.steps) and spy.fits == len(res.steps)

    @pytest.mark.parametrize("stake", [0, -1.0, NAN, INF])
    def test_flat_stake_value(self, stake):
        with pytest.raises(ValueError, match="flat_stake"):
            self._run(flat_stake=stake)

    @pytest.mark.parametrize("stake", [None, "1"])
    def test_flat_stake_type(self, stake):
        with pytest.raises(TypeError, match="flat_stake"):
            self._run(flat_stake=stake)

    def test_taker_bps_nan(self):
        with pytest.raises(ValueError, match="taker_bps"):
            self._run(taker_bps=NAN)

    def test_wrong_types(self):
        with pytest.raises(TypeError, match="policy"):
            self._run(policy="live")
        with pytest.raises(TypeError, match="costs"):
            self._run(costs={"taker_bps": 30})
        with pytest.raises(TypeError, match="model"):
            self._run(model=None)
        with pytest.raises(TypeError, match="DataFrame"):
            self._run(panel=_panel().to_dict("records"))

    def test_missing_panel_columns_are_named(self):
        with pytest.raises(ValueError, match="winning_label"):
            self._run(panel=_panel().drop(columns=["winning_label"]))
        with pytest.raises(ValueError, match="market_price"):
            self._run(panel=_panel().drop(columns=["market_price", "winning_label"]))

    def test_policy_and_costs_objects_still_accepted(self):
        res = rolling_origin_eval(
            _panel(),
            Spy(),
            min_train_rows=4,
            stride=3,
            costs=CostModel(taker_bps=10),
            policy=TradePolicy.live(min_edge=0.0),
        )
        assert math.isfinite(float(res.trades["realized_pnl"].sum()))
