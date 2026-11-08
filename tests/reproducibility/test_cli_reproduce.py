"""``pmlab backtest`` records, ``pmlab reproduce`` re-runs: the user-facing loop, through the CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rich.console import Console
from typer.testing import CliRunner

from pmlab.backtest.holdout_gate import HoldoutGateResult
from pmlab.cli.main import app
from pmlab.modeling.lgbm_baseline import LGBMForecaster
from tests.reproducibility.helpers import STOCHASTIC_FLAGS, make_panel, write_panel

runner = CliRunner()
SEED = "123"


def _backtest(panel: Path, *extra: str):
    return runner.invoke(
        app,
        ["backtest", "--plugin", "demo", "--panel", str(panel), "--stride", "10",
         "--min-train-rows", "40", "--policy", "top-bin-yes", *STOCHASTIC_FLAGS, *extra],
    )  # fmt: skip


@pytest.fixture
def run(tmp_path):
    """One recorded backtest: its panel, its metrics file and its tracker run id."""
    panel = write_panel(tmp_path / "panel.parquet")
    metrics = tmp_path / "metrics.json"
    runs = tmp_path / "runs"
    result = _backtest(
        panel, "--seed", SEED, "--output", str(metrics), "--runs-dir", str(runs), "--run-name", "r1"
    )
    assert result.exit_code == 0, result.output
    run_id = next((runs / "demo").glob("*.json")).stem
    return {"panel": panel, "metrics": metrics, "runs": runs, "run_id": run_id, "dir": tmp_path}


def test_backtest_output_carries_the_seeds_versions_config_and_data(run):
    rec = json.loads(run["metrics"].read_text())["reproducibility"]
    assert rec["seeds"] == {"python_random": 123, "numpy": 123, "lightgbm": 123}
    assert rec["environment"]["lightgbm"] and rec["environment"]["pandas"]
    assert (
        rec["config"]["stride"] == 10 and rec["config"]["model_params"]["bagging_fraction"] == 0.6
    )
    assert rec["config"]["policy"]["selection"] == "top_prob"
    assert rec["data"]["n_rows"] == len(make_panel()) and rec["data"]["columns"][0] == "market_id"
    assert rec["data"]["source"] == str(run["panel"].resolve())  # absolute: reproduce from anywhere
    assert rec["metrics"]["total_trades"] == json.loads(run["metrics"].read_text())["total_trades"]


def test_backtest_logs_a_tracker_run_with_the_record(run):
    stored = json.loads((run["runs"] / "demo" / f"{run['run_id']}.json").read_text())
    assert stored["run_name"] == "r1" and stored["status"] == "finished"
    assert stored["params"]["stride"] == 10 and stored["params"]["seed"] == 123
    assert stored["metrics"]["total_trades"] == stored["reproducibility"]["metrics"]["total_trades"]


def test_reproduce_a_metrics_file(run, monkeypatch):
    # a table cell is ellipsised in a very narrow terminal: give the module's console room
    monkeypatch.setattr("pmlab.cli.main.console", Console(width=120))
    result = runner.invoke(app, ["reproduce", str(run["metrics"])])
    assert result.exit_code == 0, result.output
    assert "REPRODUCED" in result.output and "NOT REPRODUCED" not in result.output
    assert "total_pnl" in result.output and "avg_edge" in result.output


def test_reproduce_a_tracker_run_by_id_and_by_prefix(run):
    for ref in (run["run_id"], run["run_id"][:8]):
        result = runner.invoke(app, ["reproduce", ref, "--runs-dir", str(run["runs"])])
        assert result.exit_code == 0, result.output
        assert "REPRODUCED" in result.output


def test_reproduce_works_from_another_working_directory(run, monkeypatch, tmp_path_factory):
    monkeypatch.chdir(tmp_path_factory.mktemp("elsewhere"))
    assert runner.invoke(app, ["reproduce", str(run["metrics"])]).exit_code == 0


def test_strict_tolerance_still_reproduces_on_the_same_machine(run):
    result = runner.invoke(app, ["reproduce", str(run["metrics"]), "--rtol", "0", "--atol", "0"])
    assert result.exit_code == 0, result.output


def test_a_different_dataset_is_refused_with_exit_2(run):
    make_panel(seed=99).to_parquet(run["panel"])
    result = runner.invoke(app, ["reproduce", str(run["metrics"])])
    assert result.exit_code == 2
    assert "data" in result.output and "differs" in result.output
    assert "--allow-data-mismatch" in result.output
    assert "REPRODUCED" not in result.output


def test_allowing_the_data_mismatch_runs_and_reports_the_result_honestly(run):
    make_panel(seed=99).to_parquet(run["panel"])
    result = runner.invoke(app, ["reproduce", str(run["metrics"]), "--allow-data-mismatch"])
    assert result.exit_code == 1
    assert "NOT REPRODUCED" in result.output and "different data" in result.output


def test_version_drift_warns_and_shows_the_diff_but_a_reproduced_run_still_exits_0(run):
    rec = json.loads(run["metrics"].read_text())
    rec["reproducibility"]["environment"]["lightgbm"] = "0.0.1"
    run["metrics"].write_text(json.dumps(rec))
    result = runner.invoke(app, ["reproduce", str(run["metrics"])])
    assert result.exit_code == 0, result.output
    assert "Version drift" in result.output and "lightgbm" in result.output
    assert "0.0.1" in result.output
    assert "REPRODUCED" in result.output


def test_a_miss_with_drift_points_at_the_drift(run):
    rec = json.loads(run["metrics"].read_text())
    rec["reproducibility"]["environment"]["lightgbm"] = "0.0.1"
    rec["reproducibility"]["metrics"]["total_pnl"] += 1.0
    run["metrics"].write_text(json.dumps(rec))
    result = runner.invoke(app, ["reproduce", str(run["metrics"])])
    assert result.exit_code == 1
    assert "NOT REPRODUCED" in result.output and "lightgbm" in result.output


def test_a_missing_run_is_a_clear_error(run):
    result = runner.invoke(app, ["reproduce", "deadbeef", "--runs-dir", str(run["runs"])])
    assert result.exit_code == 2
    assert "No run" in result.output and "deadbeef" in result.output


def test_a_run_without_a_record_says_how_to_get_one(run):
    legacy = run["dir"] / "legacy.json"
    legacy.write_text(json.dumps({"total_pnl": 1.0}))
    result = runner.invoke(app, ["reproduce", str(legacy)])
    assert result.exit_code == 2
    assert "no reproducibility record" in result.output and "0.10.0" in result.output


def test_a_moved_panel_is_found_with_panel_and_checked_against_the_hash(run):
    moved = run["dir"] / "moved.parquet"
    run["panel"].rename(moved)
    assert runner.invoke(app, ["reproduce", str(run["metrics"])]).exit_code == 2  # not found
    ok = runner.invoke(app, ["reproduce", str(run["metrics"]), "--panel", str(moved)])
    assert ok.exit_code == 0, ok.output


def test_model_params_cannot_smuggle_in_a_second_seed(tmp_path):
    panel = write_panel(tmp_path / "p.parquet")
    result = _backtest(panel, "--lgbm-param", "random_state=9")
    assert result.exit_code == 1 and "--seed" in result.output


def test_a_malformed_lgbm_param_is_a_clear_error(tmp_path):
    panel = write_panel(tmp_path / "p.parquet")
    result = _backtest(panel, "--lgbm-param", "no_equals_sign")
    assert result.exit_code == 1 and "KEY=VALUE" in result.output


def test_backtest_without_the_new_options_behaves_as_before(tmp_path):
    panel = write_panel(tmp_path / "p.parquet")
    out = tmp_path / "m.json"
    result = runner.invoke(
        app,
        ["backtest", "--plugin", "demo", "--panel", str(panel), "--stride", "10",
         "--min-train-rows", "40", "--policy", "top-bin-yes", "--output", str(out)],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text())
    assert data["reproducibility"]["seeds"]["lightgbm"] == 0  # seeded by default
    assert not (tmp_path / "runs").exists()  # nothing is tracked unless asked


# ── the champion / registry loop ─────────────────────────────────────────
def _promote(run, out: Path, *extra: str):
    gate_trades = pd.DataFrame([{"realized_pnl": 0.1, "outcome": "won", "segment": "A"}] * 50)
    gate = HoldoutGateResult.evaluate(
        gate_trades, required_segments=["A"], min_trades_per_segment=40
    )
    gate_path = run["dir"] / "gate.json"
    gate_path.write_text(json.dumps(gate.to_dict()))
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"feature_a": rng.normal(size=60)})  # noqa: N806
    model = LGBMForecaster(n_estimators=3, min_child_samples=2)
    model.fit(X, pd.Series((X["feature_a"] > 0).astype(int)))
    model_path = run["dir"] / "model.pkl"
    model.save(model_path)
    return runner.invoke(
        app,
        ["promote-champion", str(model_path), "--gate-path", str(gate_path), "--plugin", "demo",
         "--output-dir", str(out), "--expectation", str(run["metrics"]), *extra],
    )  # fmt: skip


def test_promote_champion_records_the_backtests_reproducibility_and_reproduce_reads_it(run):
    out = run["dir"] / "champion"
    promoted = _promote(run, out)
    assert promoted.exit_code == 0, promoted.output
    assert "Reproducibility: recorded" in promoted.output
    champion = json.loads((out / "champion.json").read_text())
    assert champion["reproducibility"]["seeds"]["lightgbm"] == 123
    for ref in (out / "champion.json", out):
        result = runner.invoke(app, ["reproduce", str(ref)])
        assert result.exit_code == 0, result.output
        assert "REPRODUCED" in result.output


def test_promoting_from_a_file_without_a_record_says_not_recorded(run):
    legacy = json.loads(run["metrics"].read_text())
    del legacy["reproducibility"]
    run["metrics"].write_text(json.dumps(legacy))
    promoted = _promote(run, run["dir"] / "champion")
    assert promoted.exit_code == 0, promoted.output
    assert "Reproducibility: not recorded" in promoted.output


def test_reproduce_a_registry_version(run):
    from pmlab.modeling.champion import ChampionManifest
    from pmlab.modeling.registry import ModelRegistry

    out = run["dir"] / "champion"
    assert _promote(run, out).exit_code == 0
    registry = ModelRegistry(run["dir"] / "registry")
    version = registry.record(ChampionManifest.load(out / "champion.json"))
    result = runner.invoke(
        app, ["reproduce", version, "--registry-dir", str(run["dir"] / "registry")]
    )
    assert result.exit_code == 0, result.output
    assert "REPRODUCED" in result.output


def test_a_promoted_champion_survives_the_registry_intact(run):
    """`promote-champion` records costs, the expectation and the record; the registry keeps them."""
    from pmlab.modeling.champion import ChampionManifest
    from pmlab.modeling.registry import ModelRegistry
    from tests.modeling.manifest_helpers import assert_manifest_survived, assert_same_champion_json

    out = run["dir"] / "champion"
    assert _promote(run, out).exit_code == 0
    promoted = ChampionManifest.load(out / "champion.json")
    assert promoted.costs is not None and promoted.reproducibility is not None
    assert promoted.expectation is not None and promoted.expectation.policy is not None

    registry = ModelRegistry(run["dir"] / "registry")
    version = registry.record(promoted)
    restored = registry.rollback(version, run["dir"] / "restored")

    assert_manifest_survived(promoted, registry.get(version))
    assert_manifest_survived(promoted, restored)
    assert_same_champion_json(out / "champion.json", run["dir"] / "restored" / "champion.json")


def test_promote_champion_takes_the_record_from_its_own_option(run):
    bare = run["dir"] / "record.json"
    bare.write_text(json.dumps(json.loads(run["metrics"].read_text())["reproducibility"]))
    stripped = json.loads(run["metrics"].read_text())
    del stripped["reproducibility"]  # the expectation file has none: the option supplies it
    run["metrics"].write_text(json.dumps(stripped))
    out = run["dir"] / "champion"
    promoted = _promote(run, out, "--reproducibility", str(bare))
    assert promoted.exit_code == 0, promoted.output
    assert "Reproducibility: recorded" in promoted.output
    assert runner.invoke(app, ["reproduce", str(out / "champion.json")]).exit_code == 0


def test_promote_champion_rejects_a_reproducibility_file_that_holds_no_record(run):
    empty = run["dir"] / "empty.json"
    empty.write_text(json.dumps({"total_pnl": 1.0}))
    promoted = _promote(run, run["dir"] / "champion", "--reproducibility", str(empty))
    assert promoted.exit_code == 1
    assert "no reproducibility record" in promoted.output
    assert not (run["dir"] / "champion" / "champion.json").exists()  # nothing half-published


def test_reproduce_reports_an_unusable_run_ref_without_a_traceback(run):
    result = runner.invoke(app, ["reproduce", str(run["dir"])])  # a directory, no champion.json
    assert result.exit_code == 2 and "without a champion.json" in result.output
    assert "Traceback" not in result.output


def test_the_cli_seed_default_is_the_librarys_default_seed():
    from pmlab.reproducibility import DEFAULT_SEED

    assert DEFAULT_SEED == 0  # the literal in the `--seed` option
