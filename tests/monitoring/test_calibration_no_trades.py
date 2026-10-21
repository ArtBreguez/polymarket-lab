"""Calibration scores the model's probability *of the label*, whichever side was bought.

A NO trade is a bet that the label will not occur, but the forecast the model made is still
the probability that it will. The checks must score `model_prob` against whether the label
occurred, not `1 - model_prob` against whether the bet paid. Driven through the real chain:
signal -> broker -> settlement -> tracker -> watchdog.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from pmlab.backtest.expectation import BacktestExpectation, GroupExpectation
from pmlab.backtest.holdout_gate import HoldoutGateResult, SegmentGateResult
from pmlab.core.market_spec import MarketSpec, OutcomeBin
from pmlab.execution.edge_signal import EdgeSignal
from pmlab.execution.paper_broker import PaperBroker
from pmlab.execution.settlement import SettlementEngine
from pmlab.modeling.champion import ChampionManifest
from pmlab.monitoring.calibration_tracker import CalibrationTracker
from pmlab.monitoring.watchdog import health_report
from pmlab.plugins.base import MarketPlugin

_P = 0.15  # the model's probability that the label occurs
_PRICE = 0.55  # the label's YES price: the NO token is worth 0.85 to the model, costs 0.45


class _Truth(MarketPlugin):
    family = "dummy"

    def __init__(self, occurs: bool) -> None:
        self._occurs = occurs

    def discover_markets(self, **kwargs: Any) -> list[MarketSpec]:
        return []

    def fetch_features(self, spec: MarketSpec, horizon: str, **kwargs: Any) -> dict[str, float]:
        return {}

    def fetch_truth(self, spec: MarketSpec, **kwargs: Any) -> str | None:
        return "YES" if self._occurs else "NO"

    def build_training_row(
        self, spec: MarketSpec, horizon: str, **kwargs: Any
    ) -> dict[str, Any] | None:
        return None


def _day(i: int) -> str:
    return (date(2099, 1, 1) + timedelta(days=i)).isoformat()


def _book(tmp_path: Path, n: int, *, occurs: bool) -> list[dict[str, Any]]:
    """`n` NO trades recorded by the broker and settled by the engine; the label occurs or not."""
    log = tmp_path / f"trades-{occurs}.json"
    signals = [
        EdgeSignal(
            market_id=f"m{i}",
            city_or_segment="A",
            target_date=_day(i),
            horizon="morning_of",
            outcome_label="YES",
            direction="no",
            gamma_price=_PRICE,
            model_prob=_P,
            best_edge=0.4,
            yes_edge=-0.4,
            no_edge=0.4,
        )
        for i in range(n)
    ]
    PaperBroker(trades_path=log).record(signals, now_utc=datetime(2020, 1, 1, tzinfo=UTC))
    specs = [
        MarketSpec(
            market_id=f"m{i}",
            slug="s",
            question="q",
            outcome_bins=[OutcomeBin(label="YES"), OutcomeBin(label="NO")],
            close_time="2099-12-31T00:00:00Z",
            market_family="binary",
            metadata={"city": "A", "target_date": _day(i)},
        )
        for i in range(n)
    ]
    SettlementEngine(_Truth(occurs), log).settle_all(
        specs, today_str="2099-12-31", now_utc=datetime(2099, 12, 31, 12, tzinfo=UTC)
    )
    trades: list[dict[str, Any]] = json.loads(log.read_text())["trades"]
    return trades


class TestTheTrackerScoresTheLabel:
    def test_a_no_trade_on_a_label_that_occurred_is_a_miss(self, tmp_path: Path) -> None:
        trades = _book(tmp_path, 5, occurs=True)
        assert all(t["direction"] == "no" for t in trades)
        # The NO bet lost its stake, and the forecast (0.15 that it occurs) was wrong.
        assert all(t["realized_pnl"] < 0 for t in trades)
        terms = CalibrationTracker(trades).squared_errors()
        assert terms == [pytest.approx((_P - 1.0) ** 2)] * 5

    def test_a_no_trade_on_a_label_that_did_not_occur_is_a_hit(self, tmp_path: Path) -> None:
        trades = _book(tmp_path, 5, occurs=False)
        # The NO bet won, and the forecast (0.15) was good.
        assert all(t["realized_pnl"] > 0 for t in trades)
        terms = CalibrationTracker(trades).squared_errors()
        assert terms == [pytest.approx(_P**2)] * 5

    def test_the_forecast_is_not_flipped_to_the_no_token(self, tmp_path: Path) -> None:
        trades = _book(tmp_path, 5, occurs=False)
        assert all(t["model_prob"] == _P for t in trades)  # recorded as the label's probability
        flipped = (1 - _P) ** 2
        assert CalibrationTracker(trades).squared_errors()[0] != pytest.approx(flipped)


class TestTheWatchdogScoresThemTheSameWay:
    def _champion(self) -> ChampionManifest:
        no = GroupExpectation(100, 0.1, 0.5, 0.05, 0.05)
        return ChampionManifest(
            model_name="c",
            model_path=Path("c.pkl"),
            calibrator_path=None,
            gate=HoldoutGateResult("GO", [SegmentGateResult("A", 100, 5.0, True, "ok")], 5.0, 100),
            published_at="2020-01-01T00:00:00+00:00",
            plugin_family="x",
            expectation=BacktestExpectation(
                100, 1.0, 0.1, 0.5, 0.05, 0.05, by_direction={"no": no}
            ),
        )

    def _brier(self, trades: list[dict[str, Any]]) -> float:
        report = health_report(
            champion=self._champion(),
            trades=trades,
            drift=None,
            as_of="2100-01-01T00:00:00Z",
            checks=("calibration",),
        )
        value = report.checks[0].metrics["brier_live"]
        assert isinstance(value, float)
        return value

    def test_no_trades_that_hit_score_the_label_probability(self, tmp_path: Path) -> None:
        assert self._brier(_book(tmp_path, 40, occurs=False)) == pytest.approx(_P**2)

    def test_no_trades_that_missed_score_the_label_probability(self, tmp_path: Path) -> None:
        assert self._brier(_book(tmp_path, 40, occurs=True)) == pytest.approx((1 - _P) ** 2)

    def test_a_mixed_book_averages_both(self, tmp_path: Path) -> None:
        trades = _book(tmp_path, 30, occurs=False) + [
            {**t, "target_date": "2099-02-01"} for t in _book(tmp_path, 10, occurs=True)
        ]
        expected = (30 * _P**2 + 10 * (1 - _P) ** 2) / 40
        assert self._brier(trades) == pytest.approx(expected)
