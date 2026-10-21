"""A NO trade is a buy of the NO token, settled at the NO price.

`PaperBroker` records a NO trade's entry as the NO price (`1 - gamma_price`, plus
the cost model's friction). `SettlementEngine` used to hand that price to the
short-YES formula `(price - payout) * size`, which expects the *YES* price. The
result was a NO bet that lost its whole stake booked a loss of a third of it, and
a NO bet that won booked a stake-sized win instead of the (1 - p) / p it earned:
wrong in both directions, and biased toward profit exactly when the NO side won.

The worked numbers below use a YES quote of 0.30, so the NO token costs 0.70 and
one unit of stake buys 1 / 0.70 = 1.428571 shares:

    label occurs        NO token pays 0  ->  -1.000000   (the whole stake)
    label does not occur NO token pays 1  ->  1 / 0.70 - 1 = +0.428571

A trade's per-stake PnL must be the same arithmetic the backtest applies to a buy
of the same token, so the last class pins the live NO trade to the backtest.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pmlab.backtest.rolling_origin import rolling_origin_eval
from pmlab.core.costs import CostModel
from pmlab.core.market_spec import MarketSpec, OutcomeBin
from pmlab.execution.edge_signal import EdgeSignal
from pmlab.execution.paper_broker import PaperBroker
from pmlab.execution.settlement import SettlementEngine
from pmlab.plugins.base import MarketPlugin

_NOW = datetime(2026, 6, 15, 5, 0, tzinfo=UTC)


class _Truth(MarketPlugin):
    family = "dummy"

    def __init__(self, truth: str) -> None:
        self._truth = truth

    def discover_markets(self, **kwargs: Any) -> list[MarketSpec]:
        return []

    def fetch_features(self, spec: MarketSpec, horizon: str, **kwargs: Any) -> dict[str, float]:
        return {}

    def fetch_truth(self, spec: MarketSpec, **kwargs: Any) -> str | None:
        return self._truth

    def build_training_row(
        self, spec: MarketSpec, horizon: str, **kwargs: Any
    ) -> dict[str, Any] | None:
        return None


def _spec() -> MarketSpec:
    return MarketSpec(
        market_id="mkt1",
        slug="s",
        question="q",
        outcome_bins=[OutcomeBin(label="YES"), OutcomeBin(label="NO")],
        close_time="2026-06-15T20:00:00Z",
        market_family="binary",
        metadata={"city": "politics", "target_date": "2026-06-15"},
    )


def _trade(
    *,
    direction: str,
    gamma: float = 0.30,
    fill: float | None = None,
    stake: float = 1.0,
    fee: float = 0.0,
) -> dict[str, Any]:
    """A trade as the broker wrote it, minus anything the test does not need."""
    trade: dict[str, Any] = {
        "recorded_at": "2026-06-14T10:00:00+00:00",
        "city_or_segment": "politics",
        "target_date": "2026-06-15",
        "outcome_label": "YES",
        "direction": direction,
        "gamma_price": gamma,
        "flat_stake": stake,
        "fee_paid": fee,
        "outcome": None,
        "realized_pnl": None,
    }
    if fill is not None:
        trade["fill_price"] = fill
    return trade


def _settle(tmp_path: Path, trade: dict[str, Any], truth: str) -> dict[str, Any]:
    """Settle one trade; `truth` is "YES" (the label occurs) or "NO" (it does not)."""
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"trades": [trade]}))
    SettlementEngine(plugin=_Truth(truth), trades_path=path).settle_all(
        specs=[_spec()], today_str="2026-06-16"
    )
    settled: dict[str, Any] = json.loads(path.read_text())["trades"][0]
    return settled


class TestNoTradeAtTheQuote:
    """No `fill_price` recorded: the entry is the NO quote, 1 - 0.30 = 0.70."""

    def test_label_occurs_the_no_bet_loses_the_whole_stake(self, tmp_path: Path) -> None:
        settled = _settle(tmp_path, _trade(direction="no"), truth="YES")
        assert settled["realized_pnl"] == pytest.approx(-1.0, abs=1e-9)

    def test_label_does_not_occur_the_no_bet_wins_one_over_p_minus_one(
        self, tmp_path: Path
    ) -> None:
        settled = _settle(tmp_path, _trade(direction="no"), truth="NO")
        assert settled["realized_pnl"] == pytest.approx(1.0 / 0.70 - 1.0, abs=1e-6)
        assert settled["realized_pnl"] == pytest.approx(0.428571, abs=1e-6)


class TestNoTradeAtTheRecordedFill:
    """`fill_price` is the NO price actually paid, slippage included."""

    def test_label_occurs_the_whole_stake_is_lost_whatever_the_fill(self, tmp_path: Path) -> None:
        settled = _settle(tmp_path, _trade(direction="no", fill=0.735), truth="YES")
        assert settled["realized_pnl"] == pytest.approx(-1.0, abs=1e-9)

    def test_label_does_not_occur_the_win_is_priced_at_the_fill(self, tmp_path: Path) -> None:
        settled = _settle(tmp_path, _trade(direction="no", fill=0.735), truth="NO")
        assert settled["realized_pnl"] == pytest.approx(1.0 / 0.735 - 1.0, abs=1e-6)
        assert settled["realized_pnl"] == pytest.approx(0.360544, abs=1e-6)

    def test_slippage_makes_a_winning_no_trade_earn_less(self, tmp_path: Path) -> None:
        at_quote = _settle(tmp_path, _trade(direction="no"), truth="NO")
        at_fill = _settle(tmp_path, _trade(direction="no", fill=0.735), truth="NO")
        assert at_fill["realized_pnl"] < at_quote["realized_pnl"]


class TestFees:
    def test_a_winning_no_trade_pays_the_fee_once(self, tmp_path: Path) -> None:
        settled = _settle(tmp_path, _trade(direction="no", stake=10.0, fee=0.03), truth="NO")
        # 10 / 0.70 shares pay 1 each; the stake and the fee come off.
        assert settled["realized_pnl"] == pytest.approx(10.0 / 0.70 - 10.0 - 0.03, abs=1e-6)
        assert settled["realized_pnl"] == pytest.approx(4.255714, abs=1e-6)

    def test_a_losing_no_trade_loses_the_stake_and_the_fee(self, tmp_path: Path) -> None:
        settled = _settle(tmp_path, _trade(direction="no", stake=10.0, fee=0.03), truth="YES")
        assert settled["realized_pnl"] == pytest.approx(-10.03, abs=1e-9)


class TestYesTradesAreUnchanged:
    def test_yes_wins_one_over_p_minus_one(self, tmp_path: Path) -> None:
        settled = _settle(tmp_path, _trade(direction="yes"), truth="YES")
        assert settled["realized_pnl"] == pytest.approx(1.0 / 0.30 - 1.0, abs=1e-6)
        assert settled["realized_pnl"] == pytest.approx(2.333333, abs=1e-6)

    def test_yes_loses_the_stake(self, tmp_path: Path) -> None:
        settled = _settle(tmp_path, _trade(direction="yes"), truth="NO")
        assert settled["realized_pnl"] == pytest.approx(-1.0, abs=1e-9)

    def test_yes_at_a_recorded_fill(self, tmp_path: Path) -> None:
        settled = _settle(tmp_path, _trade(direction="yes", fill=0.315, fee=0.003), truth="YES")
        assert settled["realized_pnl"] == pytest.approx(1.0 / 0.315 - 1.0 - 0.003, abs=1e-6)

    def test_a_trade_with_no_direction_is_a_yes_trade(self, tmp_path: Path) -> None:
        trade = _trade(direction="yes")
        del trade["direction"]
        assert _settle(tmp_path, trade, truth="YES")["realized_pnl"] == pytest.approx(2.333333)


class TestOutcomeStaysAboutTheLabel:
    """`outcome` records whether the *label* occurred, not whether the trade paid.

    Calibration reads it that way (see `calibration_tracker`), so a NO trade that
    lost money on a label that occurred is still `"won"`. Correcting the PnL must
    not flip it.
    """

    def test_a_losing_no_trade_on_an_occurred_label_is_won(self, tmp_path: Path) -> None:
        settled = _settle(tmp_path, _trade(direction="no"), truth="YES")
        assert settled["outcome"] == "won"
        assert settled["realized_pnl"] < 0

    def test_a_winning_no_trade_on_a_label_that_did_not_occur_is_lost(self, tmp_path: Path) -> None:
        settled = _settle(tmp_path, _trade(direction="no"), truth="NO")
        assert settled["outcome"] == "lost"
        assert settled["realized_pnl"] > 0


class TestTheSettlementIsVersioned:
    """Logs settled by the old arithmetic must be told apart from these."""

    @pytest.mark.parametrize("direction", ["yes", "no"])
    def test_a_settled_trade_carries_the_settlement_version(
        self, tmp_path: Path, direction: str
    ) -> None:
        settled = _settle(tmp_path, _trade(direction=direction), truth="YES")
        assert settled["settlement_version"] == 1

    def test_an_already_settled_trade_keeps_its_number_and_gains_no_marker(
        self, tmp_path: Path
    ) -> None:
        """Settlement is idempotent: a wrong legacy number is not silently rewritten."""
        legacy = {**_trade(direction="no"), "outcome": "won", "realized_pnl": -0.428571}
        settled = _settle(tmp_path, legacy, truth="YES")
        assert settled["realized_pnl"] == -0.428571
        assert "settlement_version" not in settled


# --------------------------------------------------------------------------
# Live NO == backtest buy of the same token
# --------------------------------------------------------------------------

_COSTS = CostModel(
    taker_bps=30.0,
    slippage_bps=75.0,
    slippage_fixed=0.002,
    depth_bps_per_unit=40.0,
    depth_reference_stake=10.0,
)


class _Flat:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:  # noqa: N803
        return None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        n = len(X)
        return np.column_stack([np.full(n, 0.2), np.full(n, 0.8)])


def _no_token_panel(no_price: float, winning: str) -> pd.DataFrame:
    """One row per date: the 'NO' bin at its own price, which is what a NO trade buys."""
    return pd.DataFrame(
        [
            {
                "market_id": "mkt1",
                "decision_date": f"2026-06-{d + 1:02d}",
                "outcome_label": "NO",
                "winning_label": winning,
                "market_price": no_price,
                "feature_x": 1.0,
                "segment": "politics",
            }
            for d in range(10)
        ]
    )


def _no_signal(gamma: float) -> EdgeSignal:
    return EdgeSignal(
        market_id="mkt1",
        city_or_segment="politics",
        target_date="2026-06-15",
        horizon="morning_of",
        outcome_label="YES",
        direction="no",
        gamma_price=gamma,
        model_prob=0.2,
        best_edge=0.2,
        yes_edge=-0.2,
        no_edge=0.2,
    )


class TestLiveNoTradeEqualsBacktestBuyOfTheNoToken:
    @pytest.mark.parametrize("truth", ["YES", "NO"])
    @pytest.mark.parametrize("gamma", [0.10, 0.40, 0.90])
    @pytest.mark.parametrize("stake", [1.0, 10.0])
    def test_per_stake_pnl_matches(
        self, tmp_path: Path, gamma: float, truth: str, stake: float
    ) -> None:
        # The backtest buys the NO bin at its price, 1 - gamma; the live trade is a
        # NO-direction trade on the YES label. Same token, same cost model, same
        # market truth: the NO bin wins exactly when the truth is "NO".
        backtest = rolling_origin_eval(
            _no_token_panel(1.0 - gamma, winning=truth),
            _Flat(),
            min_train_rows=4,
            stride=2,
            flat_stake=stake,
            costs=_COSTS,
        )
        path = tmp_path / "t.json"
        PaperBroker(trades_path=path, flat_stake=stake, costs=_COSTS).record(
            [_no_signal(gamma)], city_timezones={"politics": "UTC"}, now_utc=_NOW
        )
        live = _settle(tmp_path, json.loads(path.read_text())["trades"][0], truth=truth)

        assert live["realized_pnl"] / stake == pytest.approx(
            float(backtest.trades.iloc[0]["realized_pnl"]) / stake, abs=1e-5
        )


class TestTheSettlementRecordsItsBasis:
    """Which price the PnL was computed from, written on the trade.

    Releases up to 0.9.0 settled at the quote even when the broker had recorded a
    `fill_price`. Such a trade's `realized_pnl` carries none of the slippage its
    `fill_price` says it paid, and nothing on the trade said so.
    """

    def test_a_trade_settled_at_its_recorded_fill_says_so(self, tmp_path: Path) -> None:
        settled = _settle(tmp_path, _trade(direction="yes", fill=0.315), truth="YES")
        assert settled["settled_basis"] == "fill_price"

    def test_a_no_trade_settled_at_its_recorded_fill_says_so(self, tmp_path: Path) -> None:
        settled = _settle(tmp_path, _trade(direction="no", fill=0.735), truth="NO")
        assert settled["settled_basis"] == "fill_price"

    @pytest.mark.parametrize("direction", ["yes", "no"])
    def test_a_trade_with_no_recorded_fill_is_settled_at_the_quote(
        self, tmp_path: Path, direction: str
    ) -> None:
        settled = _settle(tmp_path, _trade(direction=direction), truth="YES")
        assert settled["settled_basis"] == "quote"

    def test_an_explicit_null_fill_is_the_quote(self, tmp_path: Path) -> None:
        trade = {**_trade(direction="yes"), "fill_price": None}
        assert _settle(tmp_path, trade, truth="YES")["settled_basis"] == "quote"

    def test_the_basis_matches_the_arithmetic_actually_used(self, tmp_path: Path) -> None:
        at_fill = _settle(tmp_path, _trade(direction="yes", fill=0.315), truth="YES")
        assert at_fill["realized_pnl"] == pytest.approx(1.0 / 0.315 - 1.0, abs=1e-6)

    def test_an_already_settled_trade_gains_no_basis(self, tmp_path: Path) -> None:
        legacy = {**_trade(direction="yes", fill=0.315), "outcome": "won", "realized_pnl": 2.333333}
        settled = _settle(tmp_path, legacy, truth="YES")
        assert "settled_basis" not in settled
