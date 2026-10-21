"""The regression guard: the backtest and live trading make the same trades, to the cent.

The whole point of a champion's backtest expectation is that live trades are draws from
it. That holds only if the backtest simulates the strategy that runs live. This test
takes the trades of a live-policy backtest, replays each one as the `EdgeSignal` a live
scan would have produced, runs the signals through `PaperBroker`, settles them with
`SettlementEngine` against the same truth, and compares trade by trade: the side, the
fill, the fee and the realized PnL, YES and NO alike.

If someone changes how the backtest prices a NO trade, how the broker fills it, or how
settlement books it, one of these assertions moves.
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
from pmlab.core.edge import edge_clears
from pmlab.core.market_spec import MarketSpec, OutcomeBin
from pmlab.core.policy import TradePolicy
from pmlab.execution.edge_signal import EdgeSignal
from pmlab.execution.paper_broker import PaperBroker
from pmlab.execution.settlement import SETTLEMENT_VERSION, SettlementEngine
from pmlab.plugins.base import MarketPlugin

_LABELS = ("a", "b", "c")
_RECORDED_AT = datetime(2020, 1, 1, tzinfo=UTC)  # long before every target date
_SETTLED_AT = datetime(2100, 1, 1, tzinfo=UTC)


def _pnl_tolerance(fill: float, stake: float) -> float:
    """How far a booked PnL may sit from the backtest's: the broker's rounding, no more.

    Settlement rounds the PnL to 6 places (5e-7) and the broker rounds the fill to 8
    (5e-9). The fill error matters most for a cheap token, where shares = stake / fill
    and d(pnl)/d(fill) = stake / fill**2.
    """
    return 1e-6 + 5e-9 * stake / fill**2


class _FeatureProb:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = X["feature_p"].to_numpy(dtype=float)
        return np.column_stack([1.0 - p, p])


def _panel(n_dates: int = 45, n_markets: int = 4, seed: int = 11) -> pd.DataFrame:
    """Markets whose forecast and price are independent, so both sides get traded."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for day in range(n_dates):
        date = (datetime(2024, 1, 1) + pd.Timedelta(days=day)).date().isoformat()
        for market in range(n_markets):
            winner = _LABELS[int(rng.integers(0, len(_LABELS)))]
            prices = rng.dirichlet(np.full(len(_LABELS), 4.0))
            for label, price in zip(_LABELS, prices, strict=True):
                rows.append(
                    {
                        "market_id": f"m{market}",
                        # One segment per market, so (segment, target_date) is unique and the
                        # broker's dedup cannot merge two backtest trades into one.
                        "segment": f"seg{market}",
                        "decision_date": date,
                        "outcome_label": label,
                        "winning_label": winner,
                        "market_price": float(price),
                        "feature_p": float(rng.uniform(0.02, 0.98)),
                    }
                )
    return pd.DataFrame(rows)


class _Truth(MarketPlugin):
    """Resolves each market to the winning label the panel recorded for it."""

    family = "parity"

    def __init__(self, winners: dict[str, str]) -> None:
        self._winners = winners

    def discover_markets(self, **kwargs: Any) -> list[MarketSpec]:
        return []

    def fetch_features(self, spec: MarketSpec, horizon: str, **kwargs: Any) -> dict[str, float]:
        return {}

    def fetch_truth(self, spec: MarketSpec, **kwargs: Any) -> str | None:
        return self._winners[spec.market_id]

    def build_training_row(
        self, spec: MarketSpec, horizon: str, **kwargs: Any
    ) -> dict[str, Any] | None:
        return None


def _replay(
    tmp_path: Path,
    trades: pd.DataFrame,
    panel: pd.DataFrame,
    costs: CostModel,
    stake: float,
    policy: TradePolicy,
) -> list[dict[str, Any]]:
    """Run the backtest's trades through signal -> broker -> settlement; return the log."""
    assert policy.min_edge is not None
    signals: list[EdgeSignal] = []
    winners: dict[str, str] = {}
    specs: list[MarketSpec] = []
    for trade in trades.itertuples():
        signal = EdgeSignal.from_quote(
            market_id=f"{trade.market_id}@{trade.eval_date}",
            city_or_segment=str(trade.segment),
            target_date=str(trade.eval_date),
            horizon="previous_evening",
            outcome_label=str(trade.outcome_label),
            yes_price=float(trade.market_price),
            model_prob=float(trade.predicted_prob),
            costs=costs,
            stake=stake,
            directions=policy.directions,
        )
        assert signal is not None
        # `record-trades` keeps the signals whose best_edge clears --min-edge.
        if not edge_clears(signal.best_edge, policy.min_edge):
            continue
        signals.append(signal)
        block = panel[
            (panel["market_id"] == trade.market_id) & (panel["decision_date"] == trade.eval_date)
        ]
        winners[signal.market_id] = str(block["winning_label"].iloc[0])
        specs.append(
            MarketSpec(
                market_id=signal.market_id,
                slug="s",
                question="q",
                outcome_bins=[OutcomeBin(label=label) for label in _LABELS],
                close_time="2100-01-01T00:00:00Z",
                market_family="parity",
                metadata={"city": signal.city_or_segment, "target_date": signal.target_date},
            )
        )
    log = tmp_path / "trades.json"
    broker = PaperBroker(trades_path=log, flat_stake=stake, costs=costs)
    recorded = broker.record(signals, now_utc=_RECORDED_AT)
    assert len(recorded) == len(signals)  # none stale, none deduped
    summary = SettlementEngine(_Truth(winners), log).settle_all(
        specs, today_str="2100-01-02", now_utc=_SETTLED_AT
    )
    assert summary["pending"] == 0
    live: list[dict[str, Any]] = json.loads(log.read_text())["trades"]
    return live


@pytest.mark.parametrize(
    ("costs", "stake"),
    [
        pytest.param(CostModel(), 1.0, id="default-fee"),
        pytest.param(CostModel(taker_bps=30.0, slippage_bps=75.0), 1.0, id="slippage"),
        pytest.param(
            CostModel(taker_bps=40.0, slippage_bps=50.0, depth_bps_per_unit=100.0),
            3.0,
            id="depth-at-3x-reference",
        ),
    ],
)
class TestLiveReplayMatchesTheBacktest:
    def test_every_trade_is_the_same_trade(
        self, tmp_path: Path, costs: CostModel, stake: float
    ) -> None:
        policy = TradePolicy.live(min_edge=0.05)
        panel = _panel()
        result = rolling_origin_eval(
            panel,
            _FeatureProb(),
            min_train_rows=20,
            stride=1,
            flat_stake=stake,
            costs=costs,
            policy=policy,
        )
        trades = result.trades
        live = _replay(tmp_path, trades, panel, costs, stake, policy)

        # Not vacuous: both sides are traded, and both win and lose.
        assert set(trades["direction"]) == {"yes", "no"}
        for side in ("yes", "no"):
            won = trades[trades["direction"] == side]["realized_pnl"] > 0
            assert won.any(), f"no winning {side} trade in the sample"
            assert (~won).any(), f"no losing {side} trade in the sample"
        assert len(live) == len(trades) > 40

        by_key = {(t["city_or_segment"], t["target_date"]): t for t in live}
        for trade in trades.itertuples():
            booked = by_key[(str(trade.segment), str(trade.eval_date))]
            assert booked["direction"] == trade.direction
            assert booked["outcome_label"] == trade.outcome_label
            assert booked["fill_price"] == pytest.approx(trade.fill_price, abs=1e-8)
            assert booked["fee_paid"] == pytest.approx(costs.fee(stake), abs=1e-6)
            assert booked["edge_after_fee"] == pytest.approx(trade.best_edge, abs=1e-6)
            assert booked["model_prob"] == trade.predicted_prob
            assert booked["gamma_price"] == trade.market_price
            # The key assertion: the PnL the live book booked is the PnL the backtest
            # promised for the same trade, up to the broker's rounding of fill and fee.
            assert booked["realized_pnl"] == pytest.approx(
                trade.realized_pnl, abs=_pnl_tolerance(trade.fill_price, stake)
            )
            # `outcome` of a live trade says whether the LABEL occurred, even for a NO.
            assert (booked["outcome"] == "won") == bool(trade.label_occurred)
            assert booked["settlement_version"] == SETTLEMENT_VERSION

    def test_the_mean_pnl_per_stake_agrees_in_total(
        self, tmp_path: Path, costs: CostModel, stake: float
    ) -> None:
        policy = TradePolicy.live(min_edge=0.05)
        panel = _panel(seed=5)
        result = rolling_origin_eval(
            panel, _FeatureProb(), min_train_rows=20, stride=1,
            flat_stake=stake, costs=costs, policy=policy,
        )  # fmt: skip
        live = _replay(tmp_path, result.trades, panel, costs, stake, policy)
        for side in ("yes", "no"):
            backtest = result.trades[result.trades["direction"] == side]["realized_pnl"]
            booked = [t["realized_pnl"] for t in live if t["direction"] == side]
            assert len(booked) == len(backtest) > 0
            assert np.mean(booked) / stake == pytest.approx(backtest.mean() / stake, abs=1e-4)


class TestTheNoSideSettlesLikeTheBacktest:
    """A lost NO is the whole stake; the old short-YES arithmetic booked a fraction of it."""

    def test_a_lost_no_trade_booked_by_the_replay_loses_stake_plus_fee(
        self, tmp_path: Path
    ) -> None:
        costs = CostModel()
        policy = TradePolicy.live(min_edge=0.05, directions=("no",))
        panel = _panel(seed=2)
        result = rolling_origin_eval(
            panel, _FeatureProb(), min_train_rows=20, stride=1, costs=costs, policy=policy
        )
        live = _replay(tmp_path, result.trades, panel, costs, 1.0, policy)
        lost = [t for t in live if t["outcome"] == "won"]  # the label occurred: NO loses
        assert lost
        for trade in lost:
            assert trade["direction"] == "no"
            assert trade["realized_pnl"] == pytest.approx(-1.0 - costs.fee(1.0), abs=1e-6)
