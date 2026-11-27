"""A trade carries its fee from the broker to the settled book, charged the Polymarket way (#72).

`PaperBroker` records ``fee_paid``, ``fee_rate``, ``liquidity`` and ``fee_source`` on each trade;
`SettlementEngine` books ``fee_paid`` as recorded, so the PnL a settled trade shows is the gross
result less the fee the model charged at the fill. The fee of a YES at 0.40 is
``0.05 x (1 - 0.40) = 0.03`` of a unit stake, the fee of the NO token beside it (0.60) is
``0.05 x 0.40 = 0.02``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from pmlab import PmlabDeprecationWarning
from pmlab.core.costs import CostModel
from pmlab.core.fee_schedule import FeeSchedule
from pmlab.core.market_spec import MarketSpec, OutcomeBin
from pmlab.execution.edge_signal import EdgeSignal
from pmlab.execution.paper_broker import PaperBroker
from pmlab.execution.resettle import resettle_trades
from pmlab.execution.settlement import SettlementEngine
from pmlab.plugins.base import MarketPlugin

NOW = datetime(2026, 6, 15, 5, 0, tzinfo=UTC)
FLAT = CostModel(fee_model="flat-bps")
FEE_KEYS = {"fee_paid", "fee_rate", "liquidity", "fee_source", "fee_model"}


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


def _signal(price: float = 0.40, direction: str = "yes") -> EdgeSignal:
    return EdgeSignal(
        market_id="mkt1",
        city_or_segment="politics",
        target_date="2026-06-15",
        horizon="morning_of",
        outcome_label="YES",
        direction=direction,
        gamma_price=price,
        model_prob=0.8,
        best_edge=0.2,
        yes_edge=0.2,
        no_edge=-0.2,
    )


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


def _record(path: Path, signal: EdgeSignal, **broker_kwargs: Any) -> dict[str, Any]:
    broker = PaperBroker(trades_path=path, **broker_kwargs)
    (trade,) = broker.record([signal], city_timezones={"politics": "UTC"}, now_utc=NOW)
    return trade


def _settle(path: Path, winning: str) -> dict[str, Any]:
    SettlementEngine(plugin=_Truth(winning), trades_path=path).settle_all(
        specs=[_spec()], today_str="2026-06-16"
    )
    settled: dict[str, Any] = json.loads(path.read_text())["trades"][0]
    return settled


class TestThePaperBrokerChargesAndRecordsTheFee:
    def test_a_yes_trade_pays_rate_times_one_minus_the_fill(self, tmp_path: Path) -> None:
        trade = _record(tmp_path / "t.json", _signal(0.40))
        assert trade["fee_paid"] == pytest.approx(0.03, abs=1e-9)
        assert trade["fee_rate"] == 0.05
        assert trade["liquidity"] == "taker"
        assert trade["fee_source"] == "category_default"
        assert trade["fee_model"] == "polymarket"

    def test_a_no_trade_pays_on_the_no_token_it_buys(self, tmp_path: Path) -> None:
        trade = _record(tmp_path / "t.json", _signal(0.40, direction="no"))
        assert trade["fill_price"] == pytest.approx(0.60)
        assert trade["fee_paid"] == pytest.approx(0.02, abs=1e-9)

    def test_the_fee_scales_with_the_stake(self, tmp_path: Path) -> None:
        trade = _record(tmp_path / "t.json", _signal(0.40), flat_stake=10.0)
        assert trade["fee_paid"] == pytest.approx(0.30, abs=1e-9)

    def test_the_fee_is_taken_at_the_fill_price(self, tmp_path: Path) -> None:
        costs = CostModel(slippage_bps=100.0)
        trade = _record(tmp_path / "t.json", _signal(0.40), costs=costs)
        assert trade["fill_price"] == pytest.approx(0.404)
        assert trade["fee_paid"] == pytest.approx(0.05 * (1 - 0.404), abs=1e-9)

    def test_the_fee_is_the_one_the_cost_model_gives(self, tmp_path: Path) -> None:
        costs = CostModel(fee_schedule=FeeSchedule(rate=0.01), fee_source="gamma")
        trade = _record(tmp_path / "t.json", _signal(0.40), costs=costs)
        assert trade["fee_paid"] == pytest.approx(costs.fee(1.0, 0.40), abs=1e-9)
        assert trade["fee_rate"] == 0.01
        assert trade["fee_source"] == "gamma"

    def test_the_flat_model_records_the_flat_fee(self, tmp_path: Path) -> None:
        trade = _record(tmp_path / "t.json", _signal(0.40), costs=FLAT)
        assert trade["fee_paid"] == pytest.approx(0.003)
        assert trade["fee_rate"] == pytest.approx(0.003)
        assert trade["fee_source"] == "flat_bps"
        assert trade["fee_model"] == "flat-bps"  # which formula that 0.003 is the rate of
        assert trade["liquidity"] == "taker"

    def test_taker_bps_is_the_flat_model_and_warns(self, tmp_path: Path) -> None:
        with pytest.warns(PmlabDeprecationWarning, match="taker_bps"):
            broker = PaperBroker(trades_path=tmp_path / "t.json", taker_bps=50.0)
        assert broker.costs == CostModel(fee_model="flat-bps", taker_bps=50.0)
        assert broker.taker_bps == 50.0

    def test_the_default_broker_holds_the_default_model(self, tmp_path: Path) -> None:
        broker = PaperBroker(trades_path=tmp_path / "t.json")
        assert broker.costs == CostModel()

    def test_the_trade_log_keeps_the_fee_fields_across_a_reload(self, tmp_path: Path) -> None:
        path = tmp_path / "t.json"
        _record(path, _signal(0.40))
        (loaded,) = PaperBroker(trades_path=path).load_trades()
        assert set(loaded) >= FEE_KEYS


class TestSettlementBooksTheFeeTheBrokerRecorded:
    def test_a_won_trade_nets_the_curve_fee(self, tmp_path: Path) -> None:
        path = tmp_path / "t.json"
        _record(path, _signal(0.40))
        settled = _settle(path, "YES")
        # 2.5 shares at 0.40 win 1.00 each: +1.50 gross, less the 0.03 fee
        assert settled["realized_pnl"] == pytest.approx(1.5 - 0.03, abs=1e-6)

    def test_a_lost_trade_loses_the_stake_and_the_fee(self, tmp_path: Path) -> None:
        path = tmp_path / "t.json"
        _record(path, _signal(0.40))
        settled = _settle(path, "NO")
        assert settled["realized_pnl"] == pytest.approx(-1.0 - 0.03, abs=1e-6)

    def test_a_no_trade_nets_the_fee_on_the_no_token(self, tmp_path: Path) -> None:
        path = tmp_path / "t.json"
        _record(path, _signal(0.40, direction="no"))
        settled = _settle(path, "NO")  # the label did not occur: the NO token pays
        # 1 / 0.60 shares win 1.00 each: +0.6667 gross, less 0.02
        assert settled["realized_pnl"] == pytest.approx(1 / 0.6 - 1.0 - 0.02, abs=1e-6)

    def test_the_settled_trade_still_says_what_it_paid_and_how(self, tmp_path: Path) -> None:
        path = tmp_path / "t.json"
        _record(path, _signal(0.40))
        settled = _settle(path, "YES")
        assert set(settled) >= FEE_KEYS
        assert settled["fee_paid"] == pytest.approx(0.03, abs=1e-9)
        assert settled["fee_source"] == "category_default"

    def test_re_settling_a_trade_books_the_same_fee(self, tmp_path: Path) -> None:
        path = tmp_path / "t.json"
        _record(path, _signal(0.40))
        settled = _settle(path, "YES")
        stale = {k: v for k, v in settled.items() if k != "settlement_version"}
        result = resettle_trades([stale], now_utc=NOW)
        assert result.trades[0]["realized_pnl"] == pytest.approx(settled["realized_pnl"], abs=1e-9)

    def test_a_trade_with_no_fee_on_record_is_still_the_legacy_flat_fee(
        self, tmp_path: Path
    ) -> None:
        # a log from before `fee_paid` was written: it was priced at the flat 30 bps
        path = tmp_path / "t.json"
        trade = _record(path, _signal(0.40))
        for key in FEE_KEYS:
            trade.pop(key)
        path.write_text(json.dumps({"trades": [trade]}))
        settled = _settle(path, "YES")
        assert settled["realized_pnl"] == pytest.approx(1.5 - 0.003, abs=1e-6)
