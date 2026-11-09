"""Bad option values are one line and exit 1, not a traceback (#61)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from pmlab.cli.main import app

runner = CliRunner()


@pytest.fixture
def panel_path(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for d in range(60):
        winner = "YES" if rng.random() < 0.5 else "NO"
        for lab in ("YES", "NO"):
            rows.append(
                {
                    "market_id": f"m{d}",
                    "decision_date": f"2026-{1 + d // 28:02d}-{d % 28 + 1:02d}",
                    "outcome_label": lab,
                    "winning_label": winner,
                    "market_price": 0.5,
                    "feature_a": (1.0 if lab == winner else -1.0) * 0.4 + rng.normal(),
                    "feature_b": rng.normal(),
                }
            )
    path = tmp_path / "panel.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return path


def _backtest(panel_path, *extra):
    return runner.invoke(
        app,
        ["backtest", "--plugin", "weather_tmax", "--panel", str(panel_path), "--stride", "10",
         "--min-train-rows", "20", *extra],
    )  # fmt: skip


@pytest.mark.parametrize(
    "extra",
    [
        ["--flat-stake", "0"],
        ["--flat-stake", "-2"],
        ["--flat-stake", "nan"],
        ["--min-train-rows", "-1"],
        ["--seed", "-1"],
        ["--lgbm-param", "n_estimators=abc"],
        ["--lgbm-param", "n_estimators=-5"],
    ],
)
def test_backtest_bad_values_are_a_one_line_error(panel_path, extra):
    result = _backtest(panel_path, *extra)
    assert result.exit_code == 1, result.output
    assert "Traceback" not in result.output and "│" not in result.output.split("Panel")[-1]
    assert "could not run" in result.output


def test_a_good_backtest_still_runs(panel_path):
    result = _backtest(panel_path, "--min-edge", "0.0")
    assert result.exit_code == 0, result.output


def test_no_trades_says_that_output_was_not_written(panel_path, tmp_path):
    out = tmp_path / "m.json"
    result = _backtest(panel_path, "--min-edge", "0.99", "--output", str(out))
    assert result.exit_code == 0 and "No trades produced" in result.output
    assert "was not written" in " ".join(result.output.split()) and not out.exists()


def test_help_states_the_stride_minimum():
    result = runner.invoke(app, ["backtest", "--help"], env={"COLUMNS": "200", "NO_COLOR": "1"})
    assert "minimum" in result.output.replace("\n", " ") or "below 10" in result.output.replace(
        "\n", " "
    )


@pytest.mark.parametrize("body", ["not json", "{}", '{"decision": "GO"}', "[]"])
def test_promote_champion_bad_gate_file(tmp_path, body):
    model = tmp_path / "model.pkl"
    model.write_bytes(b"x")
    gate = tmp_path / "gate.json"
    gate.write_text(body)
    result = runner.invoke(
        app,
        ["promote-champion", str(model), "--gate-path", str(gate), "--plugin", "weather_tmax",
         "--output-dir", str(tmp_path / "out")],
    )  # fmt: skip
    assert result.exit_code == 1, result.output
    assert "Not a valid gate file" in result.output and "Traceback" not in result.output
    assert not (tmp_path / "out").exists()


def test_promote_champion_still_refuses_no_go(tmp_path):
    model = tmp_path / "model.pkl"
    model.write_bytes(b"x")
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "decision": "NO_GO",
                "aggregate_pnl": -1.0,
                "aggregate_trades": 50,
                "segment_results": [],
            }
        )
    )
    result = runner.invoke(
        app,
        ["promote-champion", str(model), "--gate-path", str(gate), "--plugin", "weather_tmax"],
    )
    assert result.exit_code == 1 and "NO_GO" in result.output
