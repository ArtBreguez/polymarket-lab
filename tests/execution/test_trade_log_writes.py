"""The trade log is replaced atomically, and "today" is a UTC date (#68)."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from pmlab.execution import settlement
from pmlab.execution.paper_broker import PaperBroker
from pmlab.execution.settlement import SettlementEngine
from tests.execution.test_paper_broker import make_signal
from tests.execution.test_settlement import DummyPlugin, base_trade, make_spec, write_trades


def _boom(*args, **kwargs):
    raise OSError(28, "No space left on device")


class TestPaperBrokerRecord:
    def test_a_failed_write_leaves_the_log_it_had(self, tmp_path: Path, monkeypatch) -> None:
        log = tmp_path / "trades.json"
        broker = PaperBroker(trades_path=log)
        broker.record([make_signal(city="Chicago")])
        before = log.read_text()

        monkeypatch.setattr(os, "replace", _boom)
        with pytest.raises(OSError, match="No space"):
            broker.record([make_signal(city="Dallas")])
        monkeypatch.undo()

        assert log.read_text() == before
        assert len(json.loads(before)["trades"]) == 1
        assert sorted(p.name for p in tmp_path.iterdir()) == ["trades.json"]

    def test_a_successful_record_leaves_only_the_log(self, tmp_path: Path) -> None:
        broker = PaperBroker(trades_path=tmp_path / "sub" / "trades.json")
        broker.record([make_signal()])
        assert sorted(p.name for p in (tmp_path / "sub").iterdir()) == ["trades.json"]

    def test_the_log_keeps_its_shape(self, tmp_path: Path) -> None:
        log = tmp_path / "trades.json"
        PaperBroker(trades_path=log).record([make_signal()])
        text = log.read_text()
        assert json.loads(text)["trades"][0]["city_or_segment"] == "Chicago"
        assert text.startswith('{\n  "trades": [')  # indent=2, as before


class TestSettleAll:
    def _engine(self, tmp_path: Path, *trades: dict) -> tuple[SettlementEngine, Path]:
        log = tmp_path / "trades.json"
        write_trades(log, list(trades))
        return SettlementEngine(plugin=DummyPlugin(truth="YES"), trades_path=log), log

    def test_a_failed_write_leaves_the_log_it_had(self, tmp_path: Path, monkeypatch) -> None:
        engine, log = self._engine(tmp_path, base_trade(target_date="2024-01-15"))
        before = log.read_text()
        monkeypatch.setattr(os, "replace", _boom)
        with pytest.raises(OSError, match="No space"):
            engine.settle_all(specs=[make_spec()], today_str="2024-01-16")
        monkeypatch.undo()
        assert log.read_text() == before  # the trade is still open: nothing half-settled
        assert sorted(p.name for p in tmp_path.iterdir()) == ["trades.json"]

    def test_a_successful_settlement_leaves_only_the_log(self, tmp_path: Path) -> None:
        engine, log = self._engine(tmp_path, base_trade(target_date="2024-01-15"))
        assert engine.settle_all(specs=[make_spec()], today_str="2024-01-16")["settled"] == 1
        assert sorted(p.name for p in tmp_path.iterdir()) == ["trades.json"]
        assert json.loads(log.read_text())["trades"][0]["outcome"] == "won"

    def test_a_missing_log_is_still_nothing_to_do(self, tmp_path: Path) -> None:
        engine = SettlementEngine(plugin=DummyPlugin(), trades_path=tmp_path / "none.json")
        assert engine.settle_all(specs=[], today_str="2024-01-16")["settled"] == 0
        assert not (tmp_path / "none.json").exists()


class TestTodayIsAUtcDate:
    """``today_str`` defaulted to ``date.today()``: the machine's local date, whatever the zone.

    A trade due on a UTC day was held back (or released) for the hours the local date and the
    UTC date disagree, and an injected ``now_utc`` did not move it at all.
    """

    FAR = "2099-06-02"

    def _run(self, tmp_path: Path, **kwargs) -> dict:
        log = tmp_path / "trades.json"
        write_trades(log, [base_trade(target_date=self.FAR)])
        engine = SettlementEngine(plugin=DummyPlugin(truth="YES"), trades_path=log)
        return engine.settle_all(specs=[make_spec(target_date=self.FAR)], **kwargs)

    def test_the_injected_instant_decides_what_day_it_is(self, tmp_path: Path) -> None:
        result = self._run(tmp_path, now_utc=datetime(2099, 6, 2, 0, 30, tzinfo=UTC))
        assert result["settled"] == 1

    def test_the_day_before_in_utc_is_not_yet_due(self, tmp_path: Path) -> None:
        result = self._run(tmp_path, now_utc=datetime(2099, 6, 1, 23, 59, tzinfo=UTC))
        assert (result["settled"], result["pending"]) == (0, 1)

    def test_an_aware_instant_in_another_zone_is_read_as_the_utc_date(self, tmp_path: Path) -> None:
        # 2099-06-01 20:00 at UTC-5 is 2099-06-02 01:00 UTC: the 2nd has begun
        five_behind = timezone(timedelta(hours=-5))
        result = self._run(tmp_path, now_utc=datetime(2099, 6, 1, 20, 0, tzinfo=five_behind))
        assert result["settled"] == 1

    def test_an_instant_east_of_utc_that_is_still_yesterday_in_utc_is_not_due(
        self, tmp_path: Path
    ) -> None:
        # 2099-06-02 03:00 at UTC+5 is 2099-06-01 22:00 UTC: the 2nd has not begun
        five_ahead = timezone(timedelta(hours=5))
        result = self._run(tmp_path, now_utc=datetime(2099, 6, 2, 3, 0, tzinfo=five_ahead))
        assert (result["settled"], result["pending"]) == (0, 1)

    def test_without_an_instant_the_utc_clock_is_read(self, tmp_path: Path, monkeypatch) -> None:
        class FrozenDatetime(datetime):
            @classmethod
            def now(cls, tz=None):  # type: ignore[override]
                instant = datetime(2099, 6, 2, 0, 5, tzinfo=UTC)
                return instant if tz is None else instant.astimezone(tz)

        monkeypatch.setattr(settlement, "datetime", FrozenDatetime)
        assert self._run(tmp_path)["settled"] == 1

    def test_an_explicit_today_str_still_wins(self, tmp_path: Path) -> None:
        result = self._run(
            tmp_path, today_str="2099-06-01", now_utc=datetime(2099, 6, 5, tzinfo=UTC)
        )
        assert (result["settled"], result["pending"]) == (0, 1)

    def test_a_naive_instant_is_still_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            self._run(tmp_path, now_utc=datetime(2099, 6, 2))
