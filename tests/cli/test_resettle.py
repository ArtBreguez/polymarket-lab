"""`pmlab resettle --trades PATH`: fix a log booked by a release that settled NO trades wrongly.

A real user's flow: a log with NO trades from 0.9.0 sits next to a champion; `pmlab monitor`
leaves them out and says so; `pmlab resettle --dry-run` shows what would change; `pmlab
resettle` backs the file up, fixes it, and a second run does nothing; `pmlab monitor` now
counts them.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from pmlab.cli.main import app
from tests.cli.test_cli_policy import _backtest, _publish

runner = CliRunner()


def _legacy_pnl(direction: str, occurred: bool) -> float:
    """What 0.9.0 booked at an entry of 0.40, a stake of 1 and a fee of 0.003.

    A YES trade was a buy: right. A NO trade was a short YES priced at the NO token's 0.40:
    (price - payout) * shares - fee, which is the wrong formula for that price.
    """
    shares = 1.0 / 0.40
    if direction == "yes":
        return round(((1.0 if occurred else 0.0) - 0.40) * shares - 0.003, 6)
    return round((0.40 - (1.0 if occurred else 0.0)) * shares - 0.003, 6)


def _legacy(i: int, *, direction: str, occurred: bool, open_: bool = False) -> dict[str, Any]:
    return {
        "recorded_at": "2026-12-01T10:00:00+00:00",
        "city_or_segment": "A",
        "target_date": f"2026-12-{i + 1:02d}",
        "horizon": "morning_of",
        "outcome_label": "YES",
        "direction": direction,
        "gamma_price": 0.60 if direction == "no" else 0.40,
        "fill_price": 0.40,
        "model_prob": 0.30 if direction == "no" else 0.70,
        "flat_stake": 1.0,
        "fee_paid": 0.003,
        "edge_after_fee": 0.3,
        "outcome": None if open_ else ("won" if occurred else "lost"),
        "realized_pnl": None if open_ else _legacy_pnl(direction, occurred),
    }


def _log(path: Path, trades: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"trades": trades}, indent=2))
    return path


def _mixed(n: int = 12) -> list[dict[str, Any]]:
    trades = []
    for i in range(n):
        trades.append(_legacy(i, direction="no" if i % 2 else "yes", occurred=i % 3 != 0))
    trades.append(_legacy(20, direction="no", occurred=True, open_=True))
    return trades


def _run(path: Path, *extra: str) -> Any:
    return runner.invoke(app, ["resettle", "--trades", str(path), *extra])


def _backups(path: Path) -> list[Path]:
    return sorted(path.parent.glob(f"{path.name}.bak-*"))


class TestDryRun:
    def test_it_changes_nothing_and_makes_no_backup(self, tmp_path: Path) -> None:
        log = _log(tmp_path / "trades.json", _mixed())
        before = log.read_bytes()
        result = _run(log, "--dry-run")
        assert result.exit_code == 0, result.output
        assert log.read_bytes() == before
        assert _backups(log) == []

    def test_it_reports_the_count_the_trades_and_the_total_delta(self, tmp_path: Path) -> None:
        trades = _mixed()
        log = _log(tmp_path / "trades.json", trades)
        out = _run(log, "--dry-run").output
        assert "dry run" in out
        assert "12 settled trades" in out
        assert "1 open trade" in out
        # Six NO trades in the mix are wrong; every YES trade at its quote is already right.
        assert "6 of them change" in out
        assert "total realized PnL" in out
        assert "delta" in out
        assert "A 2026-12-02 morning_of no" in out
        assert "->" in out

    def test_the_total_delta_is_exact(self, tmp_path: Path) -> None:
        trades = _mixed()
        log = _log(tmp_path / "trades.json", trades)
        from datetime import UTC, datetime

        from pmlab.execution.resettle import resettle_trades

        result = resettle_trades(trades, now_utc=datetime.now(UTC))
        out = _run(log, "--dry-run").output
        delta = result.total_after - result.total_before
        assert f"{delta:+.6f}" in out

    def test_show_limits_the_per_trade_lines(self, tmp_path: Path) -> None:
        log = _log(tmp_path / "trades.json", _mixed())
        few = _run(log, "--dry-run", "--show", "2").output
        none = _run(log, "--dry-run", "--show", "0").output
        assert few.count(" -> ") == 2
        assert "and 4 more" in few
        assert " -> " not in none
        assert "6 of them change" in none  # the count is still reported

    def test_it_never_touches_the_network(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*args: object, **kwargs: object) -> None:
            raise AssertionError("resettle opened a socket")

        monkeypatch.setattr(socket, "socket", boom)
        monkeypatch.setattr(socket, "create_connection", boom)
        log = _log(tmp_path / "trades.json", _mixed())
        assert _run(log).exit_code == 0


class TestTheRealRun:
    def test_it_backs_the_original_up_first(self, tmp_path: Path) -> None:
        log = _log(tmp_path / "trades.json", _mixed())
        original = log.read_bytes()
        result = _run(log)
        assert result.exit_code == 0, result.output
        backups = _backups(log)
        assert len(backups) == 1
        assert backups[0].read_bytes() == original
        assert backups[0].name.startswith("trades.json.bak-")
        assert backups[0].name.endswith("Z")
        assert str(backups[0]) in result.output.replace("\n", "")

    def test_it_fixes_the_no_trades_and_stamps_everything_it_recomputed(
        self, tmp_path: Path
    ) -> None:
        log = _log(tmp_path / "trades.json", _mixed())
        _run(log)
        trades = json.loads(log.read_text())["trades"]
        settled = [t for t in trades if t["outcome"] is not None]
        assert all(t["settlement_version"] == 1 for t in settled)
        no_lost_money = [t for t in settled if t["direction"] == "no" and t["outcome"] == "won"]
        assert no_lost_money
        assert all(
            t["realized_pnl"] == pytest.approx(-1.0 - 0.003 + 0.003 * 0, abs=0.01)
            for t in no_lost_money
        )
        no_won_money = [t for t in settled if t["direction"] == "no" and t["outcome"] == "lost"]
        assert all(
            t["realized_pnl"] == pytest.approx(1 / 0.40 - 1 - 0.003, abs=1e-6) for t in no_won_money
        )

    def test_open_trades_are_byte_for_byte_what_they_were(self, tmp_path: Path) -> None:
        trades = _mixed()
        log = _log(tmp_path / "trades.json", trades)
        _run(log)
        after = json.loads(log.read_text())["trades"]
        assert after[-1] == trades[-1]
        assert "settlement_version" not in after[-1]

    def test_a_second_run_is_a_no_op(self, tmp_path: Path) -> None:
        log = _log(tmp_path / "trades.json", _mixed())
        _run(log)
        once = log.read_bytes()
        second = _run(log)
        assert second.exit_code == 0
        assert log.read_bytes() == once
        assert len(_backups(log)) == 1  # no second backup
        assert "nothing to re-settle" in second.output.lower()

    def test_other_top_level_keys_survive(self, tmp_path: Path) -> None:
        log = tmp_path / "trades.json"
        log.write_text(json.dumps({"schema": "v2", "trades": _mixed()}))
        _run(log)
        assert json.loads(log.read_text())["schema"] == "v2"

    def test_no_temp_files_are_left_behind(self, tmp_path: Path) -> None:
        log = _log(tmp_path / "trades.json", _mixed())
        _run(log)
        assert sorted(
            p.name for p in tmp_path.iterdir() if not p.name.startswith("trades.json.bak")
        ) == ["trades.json"]

    def test_the_file_mode_is_kept(self, tmp_path: Path) -> None:
        log = _log(tmp_path / "trades.json", _mixed())
        log.chmod(0o640)
        _run(log)
        assert log.stat().st_mode & 0o777 == 0o640

    def test_two_runs_in_the_same_second_do_not_overwrite_a_backup(self, tmp_path: Path) -> None:
        from datetime import UTC, datetime

        from pmlab.execution.resettle import resettle_file

        log = _log(tmp_path / "trades.json", _mixed())
        now = datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC)
        _, first = resettle_file(log, dry_run=False, now_utc=now)
        # Make it need re-settling again, and re-settle at the same second.
        _log(log, _mixed())
        _, second = resettle_file(log, dry_run=False, now_utc=now)
        assert first is not None and second is not None and first != second
        assert first.exists() and second.exists()

    def test_a_trade_it_cannot_recompute_is_reported_and_left_alone(self, tmp_path: Path) -> None:
        trades = _mixed()
        trades.append({**_legacy(25, direction="no", occurred=True), "gamma_price": None})
        log = _log(tmp_path / "trades.json", trades)
        result = _run(log)
        assert result.exit_code == 0
        assert "1 trade could not be recomputed" in result.output
        assert "gamma_price" in result.output
        assert json.loads(log.read_text())["trades"][-1] == trades[-1]


class TestBadInput:
    def test_a_missing_file(self, tmp_path: Path) -> None:
        result = _run(tmp_path / "nope.json")
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_a_corrupt_file_is_left_alone(self, tmp_path: Path) -> None:
        log = tmp_path / "trades.json"
        log.write_text("{not json")
        result = _run(log)
        assert result.exit_code == 1
        assert log.read_text() == "{not json"
        assert _backups(log) == []

    @pytest.mark.parametrize("payload", ["[]", '{"trades": 3}', '{"other": []}', '{"trades": [1]}'])
    def test_something_that_is_not_a_trade_log(self, tmp_path: Path, payload: str) -> None:
        log = tmp_path / "trades.json"
        log.write_text(payload)
        result = _run(log)
        assert result.exit_code == 1
        assert log.read_text() == payload

    def test_the_trades_option_is_required(self) -> None:
        assert runner.invoke(app, ["resettle"]).exit_code != 0


class TestTheWatchdogCountsThemAfterwards:
    """The point of it all: monitor leaves unfixed NO trades out, and counts them once fixed."""

    def _champion_and_log(self, tmp_path: Path) -> tuple[Path, Path]:
        _backtest(tmp_path, "--min-edge", "0.1")
        artifacts = tmp_path / "art"
        _publish(tmp_path, artifacts, tmp_path / "metrics.json")
        trades = []
        for i in range(60):
            trade = _legacy(i % 28, direction="no" if i % 2 else "yes", occurred=i % 3 != 0)
            trade["target_date"] = f"2099-01-{i % 28 + 1:02d}"
            trade["horizon"] = ["market_open", "previous_evening", "morning_of"][i // 28]
            trade["recorded_at"] = "2099-01-01T00:00:00+00:00"
            trade["settled_at"] = "2099-02-01T00:00:00+00:00"
            trades.append(trade)
        return artifacts, _log(tmp_path / "trades.json", trades)

    def _pnl(self, artifacts: Path, log: Path, out: Path) -> dict[str, Any]:
        result = runner.invoke(
            app,
            [
                "monitor",
                "--champion",
                str(artifacts / "public_models" / "champion.json"),
                "--trades",
                str(log),
                "--skip",
                "drift",
                "--min-trades",
                "5",
                "--as-of",
                "2100-01-01T00:00:00Z",
                "--json",
                str(out),
            ],  # fmt: skip
        )
        assert result.exit_code in (0, 10, 20, 30), result.output
        report = json.loads(out.read_text())
        return next(c for c in report["checks"] if c["name"] == "pnl")["metrics"]

    def test_the_legacy_no_trades_are_excluded_then_included(self, tmp_path: Path) -> None:
        artifacts, log = self._champion_and_log(tmp_path)
        before = self._pnl(artifacts, log, tmp_path / "before.json")
        assert before["n_excluded_unfixed_no"] == 30
        assert before["n_by_direction"] == {"yes": 30}

        assert _run(log).exit_code == 0

        after = self._pnl(artifacts, log, tmp_path / "after.json")
        assert after["n_excluded_unfixed_no"] == 0
        assert after["n_by_direction"] == {"yes": 30, "no": 30}
        assert after["n_trades"] == 60
