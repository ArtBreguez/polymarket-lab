"""``reproduce``: re-execute a recorded backtest and say whether the result came back.

These are the core guarantees: same record -> same metrics; a different dataset is refused;
a different library version is warned about, never silently accepted.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest

from pmlab.reproducibility import (
    BacktestConfig,
    DataMismatchError,
    ReproduceError,
    ReproducibilityRecord,
    capture_backtest,
    load_record,
    reproduce,
    resolve_seeds,
    run_backtest,
)
from pmlab.tracking import LocalJSONTracker
from tests.reproducibility.helpers import STOCHASTIC_PARAMS, make_panel

SEED = 123  # deliberately not the default: dropping the seed on re-execution must show


def _config(**kw) -> BacktestConfig:
    base = {
        "plugin": "demo",
        "model_params": dict(STOCHASTIC_PARAMS),
        "stride": 10,
        "min_train_rows": 40,
        "policy": "top-bin-yes",
    }
    return BacktestConfig.resolve(**{**base, **kw})


def _recorded(tmp_path, seed=SEED, **config_kw):
    panel = make_panel()
    path = tmp_path / "panel.parquet"
    panel.to_parquet(path)
    config, seeds = _config(**config_kw), resolve_seeds(seed)
    outcome = run_backtest(panel, config, seeds)
    return capture_backtest(panel, config, seeds, outcome.metrics, source=str(path)), path


# ── the run itself ───────────────────────────────────────────────────────
def test_the_config_is_resolved_with_every_default_spelled_out():
    config = BacktestConfig.resolve(plugin="demo")
    d = config.to_dict()
    assert d["model"] == "lgbm_baseline"
    assert d["model_params"] == {"objective": "binary", "verbose": -1}
    assert d["stride"] == 30 and d["min_train_rows"] == 20 and d["flat_stake"] == 1.0
    assert d["costs"]["taker_bps"] == 30.0  # the CostModel default, not "None"
    assert d["policy"]["selection"] == "best_edge" and d["policy"]["min_edge"] == 0.2
    assert BacktestConfig.from_dict(json.loads(json.dumps(d))) == config


def test_seeds_are_explicit_and_cover_every_generator_the_run_uses():
    assert resolve_seeds(5) == {"python_random": 5, "numpy": 5, "lightgbm": 5}
    assert resolve_seeds() == resolve_seeds(0)


def test_a_seed_in_the_model_params_is_refused_there_is_one_place_for_seeds():
    with pytest.raises(ValueError, match="seed"):
        BacktestConfig.resolve(plugin="demo", model_params={"random_state": 3})


def test_the_same_seed_gives_the_same_metrics_and_another_seed_gives_other_ones():
    panel, config = make_panel(), _config()
    a = run_backtest(panel, config, resolve_seeds(1)).metrics
    assert run_backtest(panel, config, resolve_seeds(1)).metrics == a
    assert run_backtest(panel, config, resolve_seeds(2)).metrics["avg_edge"] != a["avg_edge"]


def test_running_the_backtest_does_not_mutate_the_callers_panel():
    panel = make_panel()
    before = panel.copy()
    run_backtest(panel, _config(), resolve_seeds(1))
    assert panel.equals(before)


# ── reproduce ────────────────────────────────────────────────────────────
def test_reproduce_gives_identical_metrics(tmp_path):
    record, path = _recorded(tmp_path)
    report = reproduce(record)
    assert report.status == "reproduced" and report.reproduced
    assert report.drift == [] and report.data_match
    assert {c.name for c in report.comparisons} >= {"total_trades", "total_pnl", "avg_edge"}
    assert all(c.ok and c.abs_diff == 0.0 for c in report.comparisons)  # bit for bit here


def test_reproduce_is_not_fooled_by_a_stale_record_it_reports_the_metric_that_moved(tmp_path):
    record, _ = _recorded(tmp_path)
    forged = copy.deepcopy(record.to_dict())
    forged["metrics"]["total_pnl"] += 0.5
    report = reproduce(ReproducibilityRecord.from_dict(forged))
    assert report.status == "not_reproduced" and not report.reproduced
    bad = [c for c in report.comparisons if not c.ok]
    assert [c.name for c in bad] == ["total_pnl"]
    assert bad[0].abs_diff == pytest.approx(0.5)


def test_a_different_seed_in_the_record_does_not_reproduce(tmp_path):
    record, _ = _recorded(tmp_path)
    other = replace(record, seeds={**record.seeds, "lightgbm": SEED + 1})
    report = reproduce(other)
    assert report.status == "not_reproduced"
    assert any(c.name == "avg_edge" and not c.ok for c in report.comparisons)


def test_float_metrics_within_the_tolerance_pass_and_beyond_it_fail(tmp_path):
    record, _ = _recorded(tmp_path)
    nudged = replace(
        record, metrics={**record.metrics, "avg_edge": record.metrics["avg_edge"] * (1 + 1e-12)}
    )
    assert reproduce(nudged).reproduced  # default rtol 1e-9 absorbs a last-digit difference
    assert not reproduce(nudged, rtol=0.0, atol=0.0).reproduced  # strict mode is exact


def test_counts_must_match_exactly_whatever_the_tolerance(tmp_path):
    record, _ = _recorded(tmp_path)
    off = replace(
        record, metrics={**record.metrics, "total_trades": record.metrics["total_trades"] + 1}
    )
    assert not reproduce(off, rtol=1.0).reproduced


def test_a_different_dataset_is_refused_unless_allowed(tmp_path):
    record, path = _recorded(tmp_path)
    other = make_panel(seed=99)
    other.to_parquet(path)  # same path, other content
    with pytest.raises(DataMismatchError) as err:
        reproduce(record)
    assert "content" in str(err.value) and "--allow-data-mismatch" in str(err.value)
    assert err.value.recorded.content_sha256 == record.data.content_sha256
    assert err.value.current.content_sha256 != record.data.content_sha256

    report = reproduce(record, allow_data_mismatch=True)
    assert not report.data_match and report.status == "not_reproduced"
    assert any("different data" in n for n in report.notes)


def test_the_row_count_and_columns_are_named_in_the_mismatch(tmp_path):
    record, path = _recorded(tmp_path)
    make_panel(n_dates=70).drop(columns=["feature_d"]).to_parquet(path)
    with pytest.raises(DataMismatchError) as err:
        reproduce(record)
    msg = str(err.value)
    assert f"{record.data.n_rows} rows" in msg and "feature_d" in msg


def test_a_panel_can_be_handed_in_instead_of_read_from_the_recorded_path(tmp_path):
    record, path = _recorded(tmp_path)
    path.unlink()  # the recorded path is gone
    assert reproduce(record, panel=make_panel()).reproduced
    with pytest.raises(ReproduceError, match="not found"):
        reproduce(record)  # without a panel the missing file is a clear error


def test_a_record_without_a_source_needs_a_panel(tmp_path):
    record, _ = _recorded(tmp_path)
    anon = replace(record, data=replace(record.data, source=None))
    with pytest.raises(ReproduceError, match="--panel"):
        reproduce(anon)


def test_library_version_drift_is_warned_about_and_listed(tmp_path):
    record, _ = _recorded(tmp_path)
    drifted = replace(
        record, environment={**record.environment, "lightgbm": "0.0.1", "numpy": "0.0.2"}
    )
    report = reproduce(drifted)
    assert report.reproduced  # the metrics came back: drift is a warning, not a verdict
    assert {d.name for d in report.drift} == {"lightgbm", "numpy"}


def test_drift_and_a_miss_are_reported_together_so_the_cause_is_visible(tmp_path):
    record, _ = _recorded(tmp_path)
    forged = replace(
        record,
        environment={**record.environment, "lightgbm": "0.0.1"},
        metrics={**record.metrics, "total_pnl": record.metrics["total_pnl"] + 1},
    )
    report = reproduce(forged)
    assert report.status == "not_reproduced"
    assert [d.name for d in report.drift] == ["lightgbm"]


def test_unknown_run_kinds_are_a_clear_error(tmp_path):
    record, _ = _recorded(tmp_path)
    with pytest.raises(ReproduceError, match="kind"):
        reproduce(replace(record, kind="tune"))


# ── finding the record ───────────────────────────────────────────────────
def test_load_record_from_a_tracker_run_id_and_from_a_prefix(tmp_path):
    record, _ = _recorded(tmp_path)
    tracker = LocalJSONTracker(tmp_path / "runs")
    with tracker.run("demo", "r") as run:
        run.log_reproducibility(record)
    assert load_record(run.run_id, runs_dir=tmp_path / "runs").seeds == record.seeds
    assert load_record(run.run_id[:8], runs_dir=tmp_path / "runs").seeds == record.seeds


def test_load_record_from_files_and_directories(tmp_path):
    record, _ = _recorded(tmp_path)
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps(record.to_dict()))
    nested = tmp_path / "metrics.json"
    nested.write_text(json.dumps({"total_pnl": 1.0, "reproducibility": record.to_dict()}))
    champion_dir = tmp_path / "champion"
    champion_dir.mkdir()
    (champion_dir / "champion.json").write_text(json.dumps({"reproducibility": record.to_dict()}))
    for ref in (bare, nested, champion_dir, champion_dir / "champion.json"):
        assert load_record(str(ref)).data == record.data


def test_a_missing_run_is_a_clear_error_that_says_where_it_looked(tmp_path):
    with pytest.raises(ReproduceError, match="No run") as err:
        load_record("deadbeef", runs_dir=tmp_path / "runs")
    assert str(tmp_path / "runs") in str(err.value)


def test_an_ambiguous_run_prefix_lists_the_candidates(tmp_path):
    record, _ = _recorded(tmp_path)
    runs = tmp_path / "runs" / "demo"
    runs.mkdir(parents=True)
    for rid in ("ab11", "ab22"):
        (runs / f"{rid}.json").write_text(
            json.dumps(
                {
                    "run_id": rid,
                    "experiment": "demo",
                    "run_name": "r",
                    "reproducibility": record.to_dict(),
                }
            )
        )
    with pytest.raises(ReproduceError, match="ambiguous") as err:
        load_record("ab", runs_dir=tmp_path / "runs")
    assert "ab11" in str(err.value) and "ab22" in str(err.value)


def test_a_run_from_before_the_record_existed_says_so(tmp_path):
    legacy = tmp_path / "champion.json"
    legacy.write_text(json.dumps({"model_name": "champion", "reproducibility": None}))
    with pytest.raises(ReproduceError, match="no reproducibility record"):
        load_record(str(legacy))
    no_key = tmp_path / "old.json"
    no_key.write_text(json.dumps({"model_name": "champion"}))
    with pytest.raises(ReproduceError, match="0.10.0"):
        load_record(str(no_key))


def test_a_file_that_is_not_json_is_a_clear_error(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ReproduceError, match="not valid JSON"):
        load_record(str(bad))
