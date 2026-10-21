"""Re-settling a log written by a release that booked NO trades wrongly.

Before 0.9.1 a NO trade was settled as a short YES at the NO price: a NO bet that lost
its stake booked a fraction of it, and one that won booked a stake-sized win. Settlement
is idempotent, so those numbers stay in the log, and the watchdog (rightly) refuses to
judge a NO book by them. `resettle_trades` recomputes the PnL of every settled trade that
lacks the current `settlement_version`, from the fields the trade recorded: its side, its
fill (or quote), its stake and fee, and whether its label occurred. No network, no truth
lookup, nothing about an open trade changes, and a second pass changes nothing.

The arithmetic is the settlement engine's own (`book_trade`), so this file also holds the
two to each other trade by trade.
"""

from __future__ import annotations

import copy
import itertools
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from pmlab.core.market_spec import MarketSpec, OutcomeBin
from pmlab.execution.resettle import resettle_trades
from pmlab.execution.settlement import SETTLEMENT_VERSION, SettlementEngine
from pmlab.plugins.base import MarketPlugin

_NOW = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)


def _legacy(
    *,
    direction: str = "no",
    outcome: str = "won",
    gamma: float = 0.30,
    fill: float | None = None,
    stake: float = 1.0,
    fee: float = 0.0,
    pnl: float = -0.428571,
    **extra: Any,
) -> dict[str, Any]:
    """A settled trade as 0.9.0 left it: no settlement_version, no settled_basis."""
    trade: dict[str, Any] = {
        "recorded_at": "2026-06-14T10:00:00+00:00",
        "city_or_segment": "politics",
        "target_date": "2026-06-15",
        "outcome_label": "YES",
        "direction": direction,
        "gamma_price": gamma,
        "flat_stake": stake,
        "fee_paid": fee,
        "horizon": "morning_of",
        "model_prob": 0.4,
        "outcome": outcome,
        "realized_pnl": pnl,
    }
    if fill is not None:
        trade["fill_price"] = fill
    trade.update(extra)
    return trade


def _one(trade: dict[str, Any]) -> dict[str, Any]:
    result = resettle_trades([trade], now_utc=_NOW)
    return result.trades[0]


class TestNoTrades:
    def test_a_lost_no_bet_loses_the_whole_stake(self) -> None:
        # The label occurred ("won"), the NO token paid nothing. The old number was -0.4286.
        assert _one(_legacy(outcome="won"))["realized_pnl"] == pytest.approx(-1.0)

    def test_a_won_no_bet_pays_one_over_the_no_price_minus_one(self) -> None:
        # The label did not occur ("lost"), the NO token paid 1: 1/0.70 - 1. The old number was 1.0.
        fixed = _one(_legacy(outcome="lost", pnl=1.0))
        assert fixed["realized_pnl"] == pytest.approx(1.0 / 0.70 - 1.0, abs=1e-6)

    def test_the_recorded_fill_is_the_entry_price(self) -> None:
        fixed = _one(_legacy(outcome="lost", fill=0.735, pnl=1.0))
        assert fixed["realized_pnl"] == pytest.approx(1.0 / 0.735 - 1.0, abs=1e-6)
        assert fixed["settled_basis"] == "fill_price"

    def test_the_fee_comes_off_once(self) -> None:
        fixed = _one(_legacy(outcome="lost", stake=10.0, fee=0.03, pnl=10.0))
        assert fixed["realized_pnl"] == pytest.approx(10.0 / 0.70 - 10.0 - 0.03, abs=1e-6)

    def test_outcome_keeps_meaning_the_label_occurred(self) -> None:
        assert _one(_legacy(outcome="won"))["outcome"] == "won"
        assert _one(_legacy(outcome="lost", pnl=1.0))["outcome"] == "lost"

    def test_the_basis_is_quote_when_no_fill_was_recorded(self) -> None:
        assert _one(_legacy())["settled_basis"] == "quote"


class TestYesTrades:
    def test_a_yes_trade_settled_at_the_quote_is_unchanged_in_value(self) -> None:
        trade = _legacy(direction="yes", outcome="won", pnl=round(1 / 0.30 - 1, 6))
        assert _one(trade)["realized_pnl"] == pytest.approx(trade["realized_pnl"], abs=1e-6)

    def test_a_yes_trade_that_paid_a_slipped_fill_but_was_settled_at_the_quote_is_corrected(
        self,
    ) -> None:
        # 0.9.0 entered at the quote 0.30 although the broker filled at 0.315.
        trade = _legacy(
            direction="yes",
            outcome="won",
            fill=0.315,
            fee=0.003,
            pnl=round(1 / 0.30 - 1 - 0.003, 6),
        )
        fixed = _one(trade)
        assert fixed["realized_pnl"] == pytest.approx(1 / 0.315 - 1 - 0.003, abs=1e-6)
        assert fixed["realized_pnl"] != trade["realized_pnl"]

    def test_a_trade_with_no_direction_is_a_yes_trade(self) -> None:
        trade = _legacy(outcome="won", pnl=round(1 / 0.30 - 1, 6))
        del trade["direction"]
        assert _one(trade)["realized_pnl"] == pytest.approx(1 / 0.30 - 1, abs=1e-6)


class TestWhatChangesAndWhatDoesNot:
    def test_it_stamps_the_current_version_and_the_time(self) -> None:
        fixed = _one(_legacy())
        assert fixed["settlement_version"] == SETTLEMENT_VERSION
        assert fixed["resettled_at"] == _NOW.isoformat()

    def test_every_other_field_is_kept(self) -> None:
        trade = _legacy(settled_at="2026-06-15T20:00:00+00:00", note="hand-written")
        fixed = _one(trade)
        for key, value in trade.items():
            if key != "realized_pnl":
                assert fixed[key] == value

    def test_settled_at_is_not_invented_and_not_moved(self) -> None:
        assert "settled_at" not in _one(_legacy())
        assert _one(_legacy(settled_at="2026-06-15T20:00:00+00:00"))["settled_at"] == (
            "2026-06-15T20:00:00+00:00"
        )

    def test_the_input_is_not_mutated(self) -> None:
        trades = [_legacy(), _legacy(direction="yes", pnl=2.333333)]
        before = copy.deepcopy(trades)
        resettle_trades(trades, now_utc=_NOW)
        assert trades == before

    def test_an_open_trade_is_never_touched(self) -> None:
        open_trade = _legacy(outcome=None, pnl=None)
        result = resettle_trades([open_trade], now_utc=_NOW)
        assert result.trades == [open_trade]
        assert result.n_open == 1
        assert result.changes == []

    def test_a_current_trade_is_left_alone(self) -> None:
        current = _legacy(settlement_version=SETTLEMENT_VERSION, settled_basis="quote")
        result = resettle_trades([current], now_utc=_NOW)
        assert result.trades == [current]
        assert result.n_current == 1

    def test_a_trade_from_a_newer_release_is_left_alone(self) -> None:
        newer = _legacy(settlement_version=SETTLEMENT_VERSION + 1, pnl=123.0)
        result = resettle_trades([newer], now_utc=_NOW)
        assert result.trades == [newer]
        assert result.n_current == 1

    def test_an_older_version_is_resettled(self) -> None:
        old = _legacy(settlement_version=0)
        assert _one(old)["settlement_version"] == SETTLEMENT_VERSION

    def test_a_second_pass_changes_nothing(self) -> None:
        trades = [_legacy(), _legacy(outcome="lost", pnl=1.0), _legacy(outcome=None, pnl=None)]
        first = resettle_trades(trades, now_utc=_NOW)
        second = resettle_trades(
            first.trades, now_utc=_NOW + (datetime(2027, 1, 1, tzinfo=UTC) - _NOW)
        )
        assert second.trades == first.trades
        assert second.changes == []
        assert second.n_resettled == 0

    def test_the_summary_adds_up(self) -> None:
        trades = [
            _legacy(outcome="won", pnl=-0.428571),  # wrong -> changes
            _legacy(direction="yes", outcome="won", pnl=round(1 / 0.30 - 1, 6)),  # right -> same
            _legacy(outcome=None, pnl=None),  # open
            _legacy(settlement_version=1),  # current
            _legacy(gamma=None),  # cannot be recomputed
        ]
        result = resettle_trades(trades, now_utc=_NOW)
        assert (result.n_resettled, result.n_changed) == (2, 1)
        assert (result.n_open, result.n_current, result.n_unrecomputable) == (1, 1, 1)
        assert len(result.changes) == 1
        change = result.changes[0]
        assert change.old == pytest.approx(-0.428571)
        assert change.new == pytest.approx(-1.0)
        # The totals cover the trades that were recomputed, as they were and as they are now.
        yes_win = round(1 / 0.30 - 1, 6)
        assert result.total_before == pytest.approx(-0.428571 + yes_win)
        assert result.total_after == pytest.approx(-1.0 + yes_win)

    def test_the_pnl_delta_is_the_sum_of_the_changes(self) -> None:
        trades = [_legacy(outcome="won"), _legacy(outcome="lost", pnl=1.0)]
        result = resettle_trades(trades, now_utc=_NOW)
        assert result.total_after - result.total_before == pytest.approx(
            sum(c.new - c.old for c in result.changes)
        )


class TestWhatCannotBeRecomputedIsLeftAsItWas:
    @pytest.mark.parametrize(
        "damage",
        [
            {"gamma_price": None},
            {"gamma_price": float("nan")},
            {"gamma_price": 1.5},
            {"gamma_price": 1.5, "fill_price": 0.4},
            {"gamma_price": -0.1, "fill_price": 0.4},
            {"outcome_label": None},
            {"flat_stake": 0.0},
            {"flat_stake": -1.0},
            {"flat_stake": "ten"},
            {"fee_paid": float("inf")},
            {"fill_price": 0.0},
            {"fill_price": 1.2},
            {"outcome": "void"},
        ],
    )
    def test_a_trade_with_unusable_fields_is_not_guessed_at(self, damage: dict[str, Any]) -> None:
        trade = _legacy(**damage)
        result = resettle_trades([trade], now_utc=_NOW)
        assert result.trades == [trade]
        assert result.n_unrecomputable == 1
        assert result.changes == []
        assert result.unrecomputable and "politics" in result.unrecomputable[0]

    def test_the_rest_of_the_log_is_still_fixed(self) -> None:
        result = resettle_trades([_legacy(gamma_price=None), _legacy()], now_utc=_NOW)
        assert result.trades[1]["realized_pnl"] == pytest.approx(-1.0)

    def test_a_missing_fee_defaults_as_settlement_does(self) -> None:
        trade = _legacy(outcome="lost", pnl=1.0)
        del trade["fee_paid"]
        # SettlementEngine reads a missing fee_paid as 0.003.
        assert _one(trade)["realized_pnl"] == pytest.approx(1 / 0.70 - 1 - 0.003, abs=1e-6)

    def test_a_missing_stake_defaults_as_settlement_does(self) -> None:
        trade = _legacy(outcome="won")
        del trade["flat_stake"]
        assert _one(trade)["realized_pnl"] == pytest.approx(-1.0)


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


def _engine_settles(trade: dict[str, Any], label_occurred: bool, tmp_path: Path) -> dict[str, Any]:
    """What SettlementEngine books for this trade, opened fresh, when the label does/doesn't occur."""
    import json

    open_trade = {**trade, "outcome": None, "realized_pnl": None}
    for key in ("settlement_version", "settled_basis", "settled_at", "resettled_at"):
        open_trade.pop(key, None)
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"trades": [open_trade]}))
    spec = MarketSpec(
        market_id="m",
        slug="s",
        question="q",
        outcome_bins=[OutcomeBin(label="YES"), OutcomeBin(label="NO")],
        close_time="2026-06-15T20:00:00Z",
        market_family="binary",
        metadata={"city": trade["city_or_segment"], "target_date": trade["target_date"]},
    )
    SettlementEngine(plugin=_Truth("YES" if label_occurred else "NO"), trades_path=path).settle_all(
        specs=[spec], today_str="2026-06-16", now_utc=_NOW
    )
    settled: dict[str, Any] = json.loads(path.read_text())["trades"][0]
    return settled


class TestItIsTheSettlementEnginesArithmetic:
    """The key regression guard: resettling a legacy trade books what the engine books."""

    @pytest.mark.parametrize(
        ("direction", "fill", "stake", "fee", "occurred"),
        list(
            itertools.product(
                ["yes", "no"], [None, 0.31, 0.74], [1.0, 7.5], [0.0, 0.003, 0.03], [True, False]
            )
        ),
    )
    def test_the_same_pnl_the_same_basis_the_same_outcome(
        self,
        tmp_path: Path,
        direction: str,
        fill: float | None,
        stake: float,
        fee: float,
        occurred: bool,
    ) -> None:
        legacy = _legacy(
            direction=direction,
            fill=fill,
            stake=stake,
            fee=fee,
            outcome="won" if occurred else "lost",
            pnl=123.456,  # whatever the old release wrote
        )
        engine = _engine_settles(legacy, occurred, tmp_path)
        fixed = _one(legacy)
        assert fixed["realized_pnl"] == engine["realized_pnl"]
        assert fixed["settled_basis"] == engine["settled_basis"]
        assert fixed["outcome"] == engine["outcome"]
        assert fixed["settlement_version"] == engine["settlement_version"]


class TestMoreCornersOfTheLibrary:
    def test_a_naive_time_is_refused(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            resettle_trades([_legacy()], now_utc=datetime(2026, 7, 1))

    def test_a_fill_price_that_is_not_a_number_is_not_guessed_at(self) -> None:
        trade = _legacy(fill_price="cheap")
        result = resettle_trades([trade], now_utc=_NOW)
        assert result.trades == [trade]
        assert "fill_price" in result.unrecomputable[0]

    def test_a_failed_write_leaves_the_log_and_no_temp_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import json
        import os

        from pmlab.execution.resettle import resettle_file

        log = tmp_path / "trades.json"
        log.write_text(json.dumps({"trades": [_legacy()]}))
        original = log.read_bytes()

        def boom(src: object, dst: object) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(os, "replace", boom)
        with pytest.raises(OSError, match="disk full"):
            resettle_file(log, dry_run=False, now_utc=_NOW)
        assert log.read_bytes() == original
        assert [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"] == []
        # The backup was written before the attempt, and holds the original.
        backups = list(tmp_path.glob("trades.json.bak-*"))
        assert len(backups) == 1 and backups[0].read_bytes() == original


class TestASymlinkedLog:
    def test_the_real_file_is_fixed_and_the_link_stays_a_link(self, tmp_path: Path) -> None:
        import json

        from pmlab.execution.resettle import resettle_file

        real = tmp_path / "real"
        real.mkdir()
        target = real / "trades.json"
        target.write_text(json.dumps({"trades": [_legacy()]}))
        original = target.read_bytes()
        link = tmp_path / "forward_paper_trades.json"
        link.symlink_to(target)

        result, backup = resettle_file(link, dry_run=False, now_utc=_NOW)

        assert link.is_symlink()  # not swapped for a regular file
        assert json.loads(target.read_text())["trades"][0]["settlement_version"] == 1
        assert result.n_changed == 1
        assert backup is not None and backup.read_bytes() == original
        assert backup.parent == real  # next to the file that was changed
