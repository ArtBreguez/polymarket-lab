"""Tests for SettlementEngine."""

import json
from pathlib import Path
from typing import Any

from pmlab.core.market_spec import MarketSpec, OutcomeBin
from pmlab.execution.settlement import SettlementEngine
from pmlab.plugins.base import MarketPlugin

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class DummyPlugin(MarketPlugin):
    """Test plugin that returns a fixed truth string."""

    family = "dummy"

    def __init__(self, truth: str | None = "YES"):
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


def make_spec(city="Chicago", target_date="2024-01-15") -> MarketSpec:
    return MarketSpec(
        market_id="mkt-001",
        slug="will-it-be-hot",
        question="Will max temp exceed 90F?",
        outcome_bins=[
            OutcomeBin(label="YES"),
            OutcomeBin(label="NO"),
        ],
        close_time="2024-01-15T20:00:00Z",
        market_family="binary",
        metadata={"city": city, "target_date": target_date},
    )


def write_trades(path: Path, trades: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"trades": trades}))


def base_trade(
    city="Chicago",
    target_date="2024-01-15",
    outcome_label="YES",
    direction="yes",
    gamma_price=0.6,
    outcome=None,
    realized_pnl=None,
) -> dict:
    return {
        "recorded_at": "2024-01-14T10:00:00+00:00",
        "city_or_segment": city,
        "target_date": target_date,
        "outcome_label": outcome_label,
        "direction": direction,
        "gamma_price": gamma_price,
        "edge_after_fee": 0.12,
        "horizon": "morning_of",
        "flat_stake": 1.0,
        "size": round(1.0 / max(gamma_price, 1e-9), 6),
        "fee_paid": 0.003,
        "outcome": outcome,
        "realized_pnl": realized_pnl,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_settles_won_trade(tmp_path):
    trades_file = tmp_path / "trades.json"
    trade = base_trade(outcome_label="YES", direction="yes", gamma_price=0.6)
    write_trades(trades_file, [trade])

    plugin = DummyPlugin(truth="YES")
    engine = SettlementEngine(plugin=plugin, trades_path=trades_file)
    result = engine.settle_all(
        specs=[make_spec(city="Chicago", target_date="2024-01-15")],
        today_str="2024-01-16",
    )

    assert result["settled"] == 1
    data = json.loads(trades_file.read_text())
    settled = data["trades"][0]
    assert settled["outcome"] == "won"
    assert settled["realized_pnl"] is not None


def test_settles_lost_trade(tmp_path):
    trades_file = tmp_path / "trades.json"
    # Our bet is on "YES" but truth is "NO"
    trade = base_trade(outcome_label="YES", direction="yes", gamma_price=0.6)
    write_trades(trades_file, [trade])

    plugin = DummyPlugin(truth="NO")
    engine = SettlementEngine(plugin=plugin, trades_path=trades_file)
    result = engine.settle_all(
        specs=[make_spec(city="Chicago", target_date="2024-01-15")],
        today_str="2024-01-16",
    )

    assert result["settled"] == 1
    data = json.loads(trades_file.read_text())
    settled = data["trades"][0]
    assert settled["outcome"] == "lost"
    assert settled["realized_pnl"] < 0


def test_future_trade_not_settled(tmp_path):
    trades_file = tmp_path / "trades.json"
    trade = base_trade(target_date="2099-12-31")
    write_trades(trades_file, [trade])

    plugin = DummyPlugin(truth="YES")
    engine = SettlementEngine(plugin=plugin, trades_path=trades_file)
    result = engine.settle_all(
        specs=[make_spec(city="Chicago", target_date="2099-12-31")],
        today_str="2024-01-16",
    )

    assert result["settled"] == 0
    assert result["pending"] == 1


def test_summary_returns_correct_counts(tmp_path):
    trades_file = tmp_path / "trades.json"
    # 2 past trades (will be settled), 1 future (pending)
    trades = [
        base_trade(city="Chicago", target_date="2024-01-15"),
        base_trade(city="Dallas", target_date="2024-01-15"),
        base_trade(city="Miami", target_date="2099-12-31"),
    ]
    write_trades(trades_file, trades)

    specs = [
        make_spec(city="Chicago", target_date="2024-01-15"),
        make_spec(city="Dallas", target_date="2024-01-15"),
    ]

    plugin = DummyPlugin(truth="YES")
    engine = SettlementEngine(plugin=plugin, trades_path=trades_file)
    result = engine.settle_all(specs=specs, today_str="2024-01-16")

    assert result["settled"] == 2
    assert result["pending"] == 1


def test_no_trades_file_returns_zeros(tmp_path: Path) -> None:
    """Line 40: trades_path doesn't exist."""
    engine = SettlementEngine(
        plugin=DummyPlugin(),
        trades_path=tmp_path / "nonexistent.json",
    )
    result = engine.settle_all(specs=[], today_str="2024-01-16")
    assert result == {"settled": 0, "pending": 0, "total_pnl": 0.0}


def test_today_str_defaults_to_today(tmp_path: Path) -> None:
    """Line 36: today_str=None uses date.today()."""
    trades_file = tmp_path / "trades.json"
    # Write a future trade so it stays pending regardless of today
    trade = base_trade(target_date="2099-01-01")
    write_trades(trades_file, [trade])
    engine = SettlementEngine(plugin=DummyPlugin(), trades_path=trades_file)
    # today_str=None — should not crash
    result = engine.settle_all(specs=[], today_str=None)
    assert result["pending"] == 1


def test_already_settled_trade_counted_in_pnl(tmp_path: Path) -> None:
    """Lines 60-63: trade with outcome already set contributes to total_pnl."""
    trades_file = tmp_path / "trades.json"
    already_settled = {
        **base_trade(),
        "outcome": "won",
        "realized_pnl": 0.697,
    }
    write_trades(trades_file, [already_settled])
    engine = SettlementEngine(plugin=DummyPlugin(), trades_path=trades_file)
    result = engine.settle_all(specs=[], today_str="2024-01-16")
    assert result["settled"] == 0  # not newly settled
    assert abs(result["total_pnl"] - 0.697) < 1e-6


def test_trade_with_no_matching_spec_stays_pending(tmp_path: Path) -> None:
    """Lines 77-79: spec not found → pending."""
    trades_file = tmp_path / "trades.json"
    write_trades(trades_file, [base_trade(city="UnknownCity")])
    # specs list is empty — no match possible
    engine = SettlementEngine(plugin=DummyPlugin(), trades_path=trades_file)
    result = engine.settle_all(specs=[], today_str="2024-01-16")
    assert result["pending"] == 1
    assert result["settled"] == 0


def test_truth_not_final_stays_pending(tmp_path: Path) -> None:
    """Lines 83-85: is_truth_final=False → pending."""

    class NotFinalPlugin(DummyPlugin):
        def is_truth_final(self, spec: MarketSpec, **kwargs: Any) -> bool:
            return False

    trades_file = tmp_path / "trades.json"
    write_trades(trades_file, [base_trade()])
    engine = SettlementEngine(plugin=NotFinalPlugin(), trades_path=trades_file)
    result = engine.settle_all(specs=[make_spec()], today_str="2024-01-16")
    assert result["pending"] == 1


def test_truth_none_stays_pending(tmp_path: Path) -> None:
    """Lines 90-92: fetch_truth returns None → pending."""
    trades_file = tmp_path / "trades.json"
    write_trades(trades_file, [base_trade()])
    engine = SettlementEngine(plugin=DummyPlugin(truth=None), trades_path=trades_file)
    result = engine.settle_all(specs=[make_spec()], today_str="2024-01-16")
    assert result["pending"] == 1


def test_numeric_truth_resolves_via_winning_bin(tmp_path: Path) -> None:
    """Line 98: truth is float → spec.resolve_winning_bin() used."""

    class NumericPlugin(DummyPlugin):
        def fetch_truth(self, spec: MarketSpec, **kwargs: Any) -> float:
            return 32.5  # celsius — should fall in 'hot' bin

    trades_file = tmp_path / "trades.json"
    trade = base_trade(outcome_label="hot")
    write_trades(trades_file, [trade])

    spec = MarketSpec(
        market_id="mkt-001",
        slug="s",
        question="q",
        outcome_bins=[
            OutcomeBin(label="cold", upper=20.0, upper_inclusive=False),
            OutcomeBin(label="hot", lower=20.0),
        ],
        close_time="2024-01-15T20:00:00Z",
        market_family="range",
        metadata={"city": "Chicago", "target_date": "2024-01-15"},
    )
    engine = SettlementEngine(plugin=NumericPlugin(), trades_path=trades_file)
    result = engine.settle_all(specs=[spec], today_str="2024-01-16")
    assert result["settled"] == 1
    assert result["total_pnl"] > 0


def test_winning_bin_none_stays_pending(tmp_path: Path) -> None:
    """Lines 101-103: resolve_winning_bin returns None (value out of range)."""

    class OutOfRangePlugin(DummyPlugin):
        def fetch_truth(self, spec: MarketSpec, **kwargs: Any) -> float:
            return 999.0  # outside all bins

    trades_file = tmp_path / "trades.json"
    write_trades(trades_file, [base_trade(outcome_label="hot")])

    spec = MarketSpec(
        market_id="mkt-001",
        slug="s",
        question="q",
        outcome_bins=[OutcomeBin(label="hot", lower=20.0, upper=50.0)],
        close_time="2024-01-15T20:00:00Z",
        market_family="range",
        metadata={"city": "Chicago", "target_date": "2024-01-15"},
    )
    engine = SettlementEngine(plugin=OutOfRangePlugin(), trades_path=trades_file)
    result = engine.settle_all(specs=[spec], today_str="2024-01-16")
    assert result["pending"] == 1


def test_realized_pnl_is_float_after_settlement(tmp_path: Path) -> None:
    trades_file = tmp_path / "trades.json"
    trade = base_trade(outcome_label="YES", direction="yes", gamma_price=0.6)
    write_trades(trades_file, [trade])

    plugin = DummyPlugin(truth="YES")
    engine = SettlementEngine(plugin=plugin, trades_path=trades_file)
    engine.settle_all(
        specs=[make_spec(city="Chicago", target_date="2024-01-15")],
        today_str="2024-01-16",
    )

    data = json.loads(trades_file.read_text())
    pnl = data["trades"][0]["realized_pnl"]
    assert isinstance(pnl, float)
