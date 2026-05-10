"""Tests for PaperBroker."""

import json
from datetime import UTC, datetime

from pmlab.execution.edge_signal import EdgeSignal
from pmlab.execution.paper_broker import PaperBroker


def make_signal(
    city="Chicago",
    target_date="2099-12-31",
    horizon="morning_of",
    direction="yes",
    gamma_price=0.6,
) -> EdgeSignal:
    return EdgeSignal(
        market_id="mkt-001",
        city_or_segment=city,
        target_date=target_date,
        horizon=horizon,
        outcome_label="YES",
        direction=direction,
        gamma_price=gamma_price,
        model_prob=0.72,
        best_edge=0.12,
        yes_edge=0.12,
        no_edge=-0.02,
    )


def test_record_creates_trades_file(tmp_path):
    trades_file = tmp_path / "trades.json"
    broker = PaperBroker(trades_path=trades_file)
    sig = make_signal(target_date="2099-12-31", horizon="morning_of")
    new_trades = broker.record([sig])
    assert len(new_trades) == 1
    assert trades_file.exists()
    data = json.loads(trades_file.read_text())
    assert len(data["trades"]) == 1


def test_record_deduplicates(tmp_path):
    trades_file = tmp_path / "trades.json"
    broker = PaperBroker(trades_path=trades_file)
    sig = make_signal(target_date="2099-12-31", horizon="morning_of")
    first = broker.record([sig])
    assert len(first) == 1
    second = broker.record([sig])
    assert len(second) == 0
    data = json.loads(trades_file.read_text())
    assert len(data["trades"]) == 1


def test_record_segment_gate_blocks_non_allowed(tmp_path):
    trades_file = tmp_path / "trades.json"
    broker = PaperBroker(trades_path=trades_file, allowed_segments={"Dallas", "Miami"})
    sig = make_signal(city="Chicago", target_date="2099-12-31")
    new_trades = broker.record([sig])
    assert len(new_trades) == 0


def test_record_stale_signal_skipped(tmp_path):
    """A past date (2020-01-01) should be stale for all horizons with now=datetime.now(UTC)."""
    trades_file = tmp_path / "trades.json"
    broker = PaperBroker(trades_path=trades_file)
    sig = make_signal(target_date="2020-01-01", horizon="morning_of")
    now = datetime.now(UTC)
    new_trades = broker.record([sig], now_utc=now)
    assert len(new_trades) == 0


def test_record_valid_future_not_stale(tmp_path):
    """A future date (2099-12-31) should NOT be stale with now=datetime.now(UTC)."""
    trades_file = tmp_path / "trades.json"
    broker = PaperBroker(trades_path=trades_file)
    sig = make_signal(target_date="2099-12-31", horizon="morning_of")
    now = datetime.now(UTC)
    new_trades = broker.record([sig], now_utc=now)
    assert len(new_trades) == 1


def test_load_trades_empty_if_no_file(tmp_path):
    trades_file = tmp_path / "nonexistent.json"
    broker = PaperBroker(trades_path=trades_file)
    assert broker.load_trades() == []


def test_trade_dict_has_required_fields(tmp_path):
    trades_file = tmp_path / "trades.json"
    broker = PaperBroker(trades_path=trades_file, flat_stake=2.0, taker_bps=30.0)
    sig = make_signal(target_date="2099-12-31", horizon="morning_of", gamma_price=0.7)
    new_trades = broker.record([sig])
    trade = new_trades[0]
    assert trade["outcome"] is None
    assert trade["realized_pnl"] is None
    assert "recorded_at" in trade
    assert "city_or_segment" in trade
    assert "target_date" in trade
    assert "horizon" in trade
    assert "flat_stake" in trade
    assert "size" in trade
    assert "fee_paid" in trade
    assert "edge_after_fee" in trade
    assert trade["flat_stake"] == 2.0
