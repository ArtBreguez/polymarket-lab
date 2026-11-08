"""``embargo_days`` through the reproducibility layer and ``pmlab backtest`` / ``pmlab reproduce``.

The CLI already exposes the walk-forward's shape (``--stride``, ``--min-train-rows``), so the
hold-back is one more option, recorded in the run's config so that ``pmlab reproduce`` replays
the same walk-forward. A run that does not use it records exactly what 1.0.3 recorded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console
from typer.testing import CliRunner

from pmlab import BacktestConfig, resolve_seeds, run_backtest
from pmlab.backtest import rolling_origin
from pmlab.cli.main import app
from pmlab.reproducibility import runner as repro_runner
from tests.reproducibility.helpers import STOCHASTIC_FLAGS, make_panel, write_panel

cli = CliRunner()

_OLD_KEYS = {
    "plugin",
    "model",
    "model_params",
    "stride",
    "min_train_rows",
    "flat_stake",
    "costs",
    "policy",
}


# --------------------------------------------------------------------------- BacktestConfig


def _resolve(**kwargs: Any) -> BacktestConfig:
    return BacktestConfig.resolve(
        plugin="demo", stride=10, min_train_rows=40, policy="top-bin-yes", **kwargs
    )


def test_a_config_without_an_embargo_is_what_1_0_3_wrote() -> None:
    config = _resolve()
    assert config.embargo_days is None
    assert set(config.to_dict()) == _OLD_KEYS  # no new key: old and new default records match


def test_the_embargo_is_recorded_in_the_config_when_set() -> None:
    config = _resolve(embargo_days=7)
    assert config.to_dict()["embargo_days"] == 7
    assert BacktestConfig.from_dict(config.to_dict()) == config


def test_a_config_recorded_before_1_1_0_reads_back_without_an_embargo() -> None:
    old = _resolve().to_dict()
    assert "embargo_days" not in old
    assert BacktestConfig.from_dict(old).embargo_days is None


@pytest.mark.parametrize("bad", [-1, -5])
def test_a_negative_embargo_is_a_value_error(bad: int) -> None:
    with pytest.raises(ValueError, match="embargo_days"):
        _resolve(embargo_days=bad)


@pytest.mark.parametrize("bad", [1.5, "3", True])
def test_a_non_int_embargo_is_a_type_error(bad: object) -> None:
    with pytest.raises(TypeError, match="embargo_days"):
        _resolve(embargo_days=bad)


def test_a_record_with_a_bad_embargo_is_refused_when_it_is_read() -> None:
    config = _resolve(embargo_days=3).to_dict()
    config["embargo_days"] = -2
    with pytest.raises(ValueError, match="embargo_days"):
        BacktestConfig.from_dict(config)


# --------------------------------------------------------------------------- run_backtest


def _seeds() -> dict[str, int]:
    return resolve_seeds(123)


def test_run_backtest_passes_the_embargo_to_the_walk_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Any] = []
    real = rolling_origin.rolling_origin_eval

    def spy(*args: Any, **kwargs: Any):
        seen.append(kwargs.get("embargo_days", "not passed"))
        return real(*args, **kwargs)

    monkeypatch.setattr(repro_runner, "rolling_origin_eval", spy)
    panel = make_panel()
    run_backtest(panel, _resolve(), _seeds())
    run_backtest(panel, _resolve(embargo_days=9), _seeds())
    assert seen == [None, 9]


def test_a_backtest_with_an_embargo_trains_on_no_more_than_one_without() -> None:
    panel = make_panel()
    base = run_backtest(panel, _resolve(), _seeds())
    held = run_backtest(panel, _resolve(embargo_days=20), _seeds())
    assert len(held.steps) < len(base.steps)
    by_date = {s["eval_date"]: s["train_rows"] for s in base.steps}
    for step in held.steps:
        assert step["train_rows"] <= by_date[step["eval_date"]]


def test_the_default_backtest_is_unchanged_by_the_option_existing() -> None:
    panel = make_panel()
    a = run_backtest(panel, _resolve(), _seeds())
    b = run_backtest(panel, _resolve(embargo_days=None), _seeds())
    assert a.metrics == b.metrics and a.steps == b.steps


# --------------------------------------------------------------------------- the CLI


def _backtest(panel: Path, *extra: str):
    return cli.invoke(
        app,
        ["backtest", "--plugin", "demo", "--panel", str(panel), "--stride", "10",
         "--min-train-rows", "40", "--policy", "top-bin-yes", *STOCHASTIC_FLAGS,
         "--seed", "123", *extra],
    )  # fmt: skip


@pytest.fixture
def panel(tmp_path: Path) -> Path:
    return write_panel(tmp_path / "panel.parquet")


def _record(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())["reproducibility"]


def test_the_option_is_listed_in_help() -> None:
    out = " ".join(cli.invoke(app, ["backtest", "--help"], terminal_width=200).output.split())
    assert "--embargo-days" in out


def test_without_the_option_the_record_has_no_embargo(tmp_path: Path, panel: Path) -> None:
    out = tmp_path / "m.json"
    assert _backtest(panel, "--output", str(out)).exit_code == 0
    assert "embargo_days" not in _record(out)["config"]


def test_with_the_option_the_run_records_and_uses_it(tmp_path: Path, panel: Path) -> None:
    plain, held = tmp_path / "plain.json", tmp_path / "held.json"
    assert _backtest(panel, "--output", str(plain)).exit_code == 0
    result = _backtest(panel, "--embargo-days", "20", "--output", str(held))
    assert result.exit_code == 0, result.output
    assert _record(held)["config"]["embargo_days"] == 20
    assert _record(held)["metrics"]["steps"] < _record(plain)["metrics"]["steps"]


def test_reproduce_replays_the_recorded_embargo(
    tmp_path: Path, panel: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("pmlab.cli.main.console", Console(width=120))
    out = tmp_path / "held.json"
    assert _backtest(panel, "--embargo-days", "20", "--output", str(out)).exit_code == 0
    ok = cli.invoke(app, ["reproduce", str(out)])
    assert ok.exit_code == 0, ok.output
    assert "REPRODUCED" in ok.output and "NOT REPRODUCED" not in ok.output

    # the embargo is part of what is replayed: change it in the record and the numbers differ
    record = json.loads(out.read_text())
    record["reproducibility"]["config"]["embargo_days"] = 0
    out.write_text(json.dumps(record))
    bad = cli.invoke(app, ["reproduce", str(out)])
    assert bad.exit_code == 1 and "NOT REPRODUCED" in bad.output


def test_a_record_without_an_embargo_still_reproduces(
    tmp_path: Path, panel: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("pmlab.cli.main.console", Console(width=120))
    out = tmp_path / "m.json"
    assert _backtest(panel, "--output", str(out)).exit_code == 0
    assert cli.invoke(app, ["reproduce", str(out)]).exit_code == 0


def test_the_tracker_logs_the_embargo_as_a_param(tmp_path: Path, panel: Path) -> None:
    runs = tmp_path / "runs"
    result = _backtest(panel, "--embargo-days", "20", "--runs-dir", str(runs))
    assert result.exit_code == 0, result.output
    stored = json.loads(next((runs / "demo").glob("*.json")).read_text())
    assert stored["params"]["embargo_days"] == 20


def test_a_negative_embargo_exits_1_and_says_why(panel: Path) -> None:
    result = _backtest(panel, "--embargo-days", "-3")
    assert result.exit_code == 1
    assert "embargo_days" in result.output.replace("-", "_") or "embargo" in result.output


def test_a_fractional_embargo_is_a_usage_error(panel: Path) -> None:
    assert _backtest(panel, "--embargo-days", "2.5").exit_code == 2
