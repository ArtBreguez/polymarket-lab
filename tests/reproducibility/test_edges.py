"""Edge cases of the capture / run / reproduce pieces, and the unseeded code paths made seeded."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from pmlab.cli.main import app
from pmlab.modeling.lgbm_baseline import LGBMForecaster
from pmlab.modeling.sklearn_forecaster import SklearnForecaster
from pmlab.reproducibility import (
    BacktestConfig,
    ReproduceError,
    ReproducibilityRecord,
    backtest_metrics,
    capture_backtest,
    compare_metrics,
    fingerprint_panel,
    load_panel,
    load_record,
    reproduce,
    resolve_seeds,
    run_backtest,
)
from pmlab.reproducibility.runner import BacktestOutcome
from pmlab.tracking.mlflow_tracker import _MLflowActiveRun
from tests.reproducibility.helpers import make_panel, write_panel

runner = CliRunner()


# ── fingerprint of unusual columns ───────────────────────────────────────
def test_fingerprint_handles_timestamps_booleans_and_nullable_integers():
    df = pd.DataFrame(
        {
            "when": pd.to_datetime(["2026-01-01", "2026-01-02"]).tz_localize(
                timezone(timedelta(hours=-5))
            ),
            "flag": [True, False],
            "n": pd.array([1, None], dtype="Int64"),
            "s": ["a", None],
        }
    )
    a = fingerprint_panel(df).content_sha256
    assert fingerprint_panel(df.copy()).content_sha256 == a
    same_instants_in_utc = df.assign(when=df["when"].dt.tz_convert("UTC"))
    assert fingerprint_panel(same_instants_in_utc).content_sha256 == a  # the instant, not the zone
    assert fingerprint_panel(df.assign(flag=[False, True])).content_sha256 != a


def test_a_malformed_data_entry_is_reported_as_such():
    rec = capture_backtest(
        make_panel(n_dates=4), BacktestConfig.resolve(plugin="d"), resolve_seeds(0), {}
    ).to_dict()
    rec["data"] = {"n_rows": 3}
    with pytest.raises(ValueError, match="'data'"):
        ReproducibilityRecord.from_dict(rec)


# ── the config ───────────────────────────────────────────────────────────
def test_config_validation_names_the_problem():
    with pytest.raises(ValueError, match="unknown model 'xgb'"):
        BacktestConfig.resolve(plugin="d", model="xgb")
    with pytest.raises(ValueError, match="unknown policy"):
        BacktestConfig.resolve(plugin="d", policy="martingale")
    with pytest.raises(ValueError, match="min_edge"):
        BacktestConfig.resolve(plugin="d", policy="top-bin-yes", min_edge=0.1)
    live = BacktestConfig.resolve(plugin="d", policy="live", min_edge=0.35)
    assert live.policy["min_edge"] == 0.35


def test_config_from_dict_names_what_is_missing():
    with pytest.raises(ValueError, match="stride"):
        BacktestConfig.from_dict({"plugin": "d"})


def test_a_seed_alias_is_refused_too():
    for alias in ("seed", "random_seed"):
        with pytest.raises(ValueError, match="seed"):
            BacktestConfig.resolve(plugin="d", model_params={alias: 1})


def test_load_panel_reads_parquet_and_csv_and_refuses_the_rest(tmp_path):
    panel = make_panel(n_dates=3)
    panel.to_parquet(tmp_path / "p.parquet")
    panel.to_csv(tmp_path / "p.csv", index=False)
    assert len(load_panel(tmp_path / "p.parquet")) == len(load_panel(str(tmp_path / "p.csv")))
    with pytest.raises(ValueError, match=r"\.parquet or \.csv"):
        load_panel(tmp_path / "p.xlsx")


def test_metrics_of_a_backtest_without_trades_are_explicit_nulls():
    outcome = run_backtest(
        make_panel(n_dates=3),
        BacktestConfig.resolve(plugin="d", min_train_rows=10_000),
        resolve_seeds(0),
    )
    assert isinstance(outcome, BacktestOutcome) and outcome.trades.empty
    assert outcome.metrics == {
        "total_trades": 0,
        "steps": 0,
        "total_pnl": 0.0,
        "hit_rate": None,
        "avg_edge": None,
    }
    assert backtest_metrics(SimpleNamespace(trades=outcome.trades, steps=[])) == outcome.metrics


# ── compare_metrics ──────────────────────────────────────────────────────
def test_compare_metrics_handles_nulls_missing_and_non_numbers():
    got = compare_metrics(
        {"a": None, "b": None, "c": 1.0, "d": "text", "e": 5, "f": True},
        {"a": None, "b": 0.5, "d": "text", "e": 5.0, "f": True},
    )
    by_name = {c.name: c for c in got}
    assert by_name["a"].ok  # null came back null
    assert not by_name["b"].ok  # null became a number
    assert not by_name["c"].ok  # the re-run no longer produces it
    assert "d" not in by_name  # a string is not something a re-run is judged on
    assert by_name["f"].ok
    assert by_name["e"].ok  # an int recorded, a whole float re-run: a float comparison


# ── reproduce errors ─────────────────────────────────────────────────────
def _record(tmp_path):
    panel = make_panel()
    path = tmp_path / "p.parquet"
    panel.to_parquet(path)
    config = BacktestConfig.resolve(plugin="d", stride=10, min_train_rows=40, policy="top-bin-yes")
    seeds = resolve_seeds(1)
    outcome = run_backtest(panel, config, seeds)
    return capture_backtest(panel, config, seeds, outcome.metrics, source=str(path)), path


def test_an_unusable_recorded_config_is_a_reproduce_error(tmp_path):
    record, _ = _record(tmp_path)
    with pytest.raises(ReproduceError, match="config is not usable"):
        reproduce(replace(record, config={"plugin": "d"}))


def test_an_unreadable_panel_format_is_a_reproduce_error(tmp_path):
    record, _ = _record(tmp_path)
    odd = tmp_path / "p.xlsx"
    odd.write_bytes(b"")
    with pytest.raises(ReproduceError, match=r"\.parquet or \.csv"):
        reproduce(record, panel_path=odd)


def test_a_directory_without_champion_json_is_a_clear_error(tmp_path):
    with pytest.raises(ReproduceError, match="without a champion.json"):
        load_record(str(tmp_path))


def test_a_record_from_a_newer_pmlab_is_a_reproduce_error(tmp_path):
    record, _ = _record(tmp_path)
    future = tmp_path / "future.json"
    future.write_text(json.dumps({"reproducibility": {**record.to_dict(), "schema_version": 99}}))
    with pytest.raises(ReproduceError, match="newer pmlab"):
        load_record(str(future))


def test_registry_versions_are_found_by_id(tmp_path):
    record, _ = _record(tmp_path)
    version = tmp_path / "registry" / "v1"
    version.mkdir(parents=True)
    (version / "champion.json").write_text(json.dumps({"reproducibility": record.to_dict()}))
    assert load_record("v1", registry_dir=tmp_path / "registry").seeds == record.seeds
    with pytest.raises(ReproduceError, match="registry"):
        load_record("v2", registry_dir=tmp_path / "registry")


def test_a_run_id_that_is_not_a_plain_token_is_never_globbed(tmp_path):
    with pytest.raises(ReproduceError, match="No run"):
        load_record("../*", runs_dir=tmp_path)


# ── unseeded code paths, now seeded ──────────────────────────────────────
def _xy():
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(200, 6)), columns=list("abcdef"))  # noqa: N806
    return X, pd.Series((X["a"] + rng.normal(size=200) > 0).astype(int))


@pytest.mark.parametrize("preset", ["random_forest", "gradient_boosting"])
def test_sklearn_ensemble_presets_are_deterministic_whatever_the_global_state(preset):
    X, y = _xy()  # noqa: N806
    probas = []
    for global_state in (1, 2):
        np.random.seed(global_state)  # the state an unseeded estimator would have read
        f = SklearnForecaster(estimator=preset, n_estimators=10, max_depth=3)
        f.fit(X, y)
        probas.append(f.predict_proba(X))
    assert np.array_equal(probas[0], probas[1])


def test_sklearn_preset_seed_can_still_be_chosen_or_unset():
    X, y = _xy()  # noqa: N806

    def proba(seed):
        f = SklearnForecaster(estimator="random_forest", n_estimators=10, random_state=seed)
        f.fit(X, y)
        return f.predict_proba(X)

    assert np.array_equal(proba(7), proba(7))
    assert not np.array_equal(proba(7), proba(8))


def test_lgbm_resolved_params_show_the_defaults_and_the_overrides():
    assert LGBMForecaster().resolved_params() == {"objective": "binary", "verbose": -1}
    f = LGBMForecaster(objective="multiclass", num_leaves=4, verbose=0)
    assert f.resolved_params() == {"objective": "multiclass", "verbose": 0, "num_leaves": 4}


# ── MLflow: no mlflow needed, the logging calls are what matters ─────────
def test_mlflow_active_run_logs_the_record_as_an_artifact():
    from pmlab.tracking import RunRecord

    mlflow = MagicMock()
    run = _MLflowActiveRun(SimpleNamespace(_mlflow=mlflow), RunRecord("id", "exp", "r"))
    run.log_reproducibility({"seeds": {"numpy": 1}, "bad": float("nan")})
    name = mlflow.log_dict.call_args.args[1]
    assert name == "reproducibility.json"
    assert mlflow.log_dict.call_args.args[0] == {"seeds": {"numpy": 1}, "bad": None}
    assert run._record.reproducibility["seeds"] == {"numpy": 1}


# ── CLI edges ────────────────────────────────────────────────────────────
def test_an_unknown_model_is_refused_instead_of_silently_running_lightgbm(tmp_path):
    panel = write_panel(tmp_path / "p.parquet")
    result = runner.invoke(
        app, ["backtest", "--plugin", "d", "--panel", str(panel), "--model", "xgb"]
    )
    assert result.exit_code == 1 and "unknown model" in result.output


def test_a_panel_of_an_unknown_format_is_refused(tmp_path):
    odd = tmp_path / "p.xlsx"
    odd.write_bytes(b"")
    result = runner.invoke(app, ["backtest", "--plugin", "d", "--panel", str(odd)])
    assert result.exit_code == 1 and "Panel must be .parquet or .csv" in result.output


# ── found by review ──────────────────────────────────────────────────────
def test_a_record_with_no_numeric_metrics_verifies_nothing_and_says_so(tmp_path):
    record, _ = _record(tmp_path)
    for metrics in ({}, {"note": "text only"}):
        with pytest.raises(ReproduceError, match="no numeric metrics"):
            reproduce(replace(record, metrics=metrics))


def test_a_panel_with_list_or_dict_cells_can_be_fingerprinted():
    df = pd.DataFrame({"a": [[1, 2], [3]], "b": [{"x": 1}, {"y": 2}], "n": [1, 2]})
    a = fingerprint_panel(df).content_sha256
    assert fingerprint_panel(df.copy()).content_sha256 == a
    changed = df.assign(a=[[1, 2], [4]])
    assert fingerprint_panel(changed).content_sha256 != a


def test_running_a_backtest_leaves_the_callers_random_streams_alone():
    import random

    np.random.seed(5)
    random.seed(6)
    want = (np.random.random(), random.random())
    np.random.seed(5)
    random.seed(6)
    run_backtest(
        make_panel(n_dates=40), BacktestConfig.resolve(plugin="d", stride=10), resolve_seeds(99)
    )
    assert (np.random.random(), random.random()) == want


def test_seeded_applies_the_seeds_inside_and_restores_the_state_after():
    import random

    from pmlab.reproducibility import seeded

    random.seed(1)
    before = random.getstate()
    with seeded({"python_random": 9, "numpy": 9}):
        inside = (random.random(), np.random.random())
    assert random.getstate() == before
    with seeded({"python_random": 9, "numpy": 9}):
        assert (random.random(), np.random.random()) == inside


@pytest.mark.parametrize("junk", [[], "x", 3])
def test_a_champion_whose_record_is_not_an_object_still_loads(tmp_path, junk):
    from pmlab.modeling.champion import ChampionManifest
    from tests.reproducibility.test_stores import _gate, _model

    out = tmp_path / "out"
    ChampionManifest.publish(_model(), _gate(), out, plugin_family="demo")
    path = out / "champion.json"
    raw = json.loads(path.read_text())
    raw["reproducibility"] = junk
    path.write_text(json.dumps(raw))
    with pytest.warns(UserWarning, match="ignoring the reproducibility record"):
        assert ChampionManifest.load(path).reproducibility is None


def test_a_run_id_is_not_shadowed_by_an_unrelated_directory(tmp_path, monkeypatch):
    record, _ = _record(tmp_path)
    runs = tmp_path / "runs" / "demo"
    runs.mkdir(parents=True)
    (runs / "ab12cd.json").write_text(json.dumps({"reproducibility": record.to_dict()}))
    elsewhere = tmp_path / "cwd"
    (elsewhere / "ab12").mkdir(parents=True)  # a folder that happens to be named like the prefix
    monkeypatch.chdir(elsewhere)
    assert load_record("ab12", runs_dir=tmp_path / "runs").seeds == record.seeds


def test_a_policy_object_and_a_separate_min_edge_are_not_silently_reconciled():
    from pmlab.core.policy import TradePolicy

    with pytest.raises(ValueError, match="min_edge is ignored"):
        BacktestConfig.resolve(plugin="d", policy=TradePolicy.live(min_edge=0.05), min_edge=0.1)
    ok = BacktestConfig.resolve(plugin="d", policy=TradePolicy.live(min_edge=0.05))
    assert ok.policy["min_edge"] == 0.05


def test_cpu_count_is_recorded_but_never_counted_as_drift():
    from pmlab.reproducibility import capture_environment, environment_drift

    env = capture_environment()
    assert env["cpu_count"].isdigit()
    assert environment_drift({**env, "cpu_count": "9999"}) == []
