"""What a run captures: seeds, library versions, resolved config, a fingerprint of the data."""

from __future__ import annotations

import json
import platform
import random
import sys

import lightgbm
import numpy as np
import pandas as pd
import pytest
import sklearn

import pmlab
from pmlab.reproducibility import (
    SCHEMA_VERSION,
    DataFingerprint,
    ReproducibilityRecord,
    capture_environment,
    environment_drift,
    fingerprint_panel,
    seed_everything,
)
from tests.reproducibility.helpers import make_panel


def test_environment_records_every_library_and_the_platform():
    env = capture_environment()
    assert env["pmlab"] == pmlab.__version__
    assert env["python"] == platform.python_version() == sys.version.split()[0]
    assert env["numpy"] == np.__version__
    assert env["pandas"] == pd.__version__
    assert env["lightgbm"] == lightgbm.__version__
    assert env["scikit-learn"] == sklearn.__version__
    assert env["system"] == platform.system() and env["machine"] == platform.machine()
    assert env["platform"] == platform.platform()


def test_optuna_is_recorded_when_installed_and_absent_otherwise(monkeypatch):
    optuna = pytest.importorskip("optuna")
    assert capture_environment()["optuna"] == optuna.__version__
    monkeypatch.setitem(sys.modules, "optuna", None)  # `import optuna` now raises ImportError
    assert capture_environment()["optuna"] is None


def test_environment_is_json_serialisable():
    json.dumps(capture_environment())


def test_drift_lists_only_what_changed_and_ignores_the_kernel_string():
    recorded = {**capture_environment(), "pandas": "0.0.1", "platform": "Linux-1.2.3-other-kernel"}
    drift = environment_drift(recorded)
    assert [(d.name, d.recorded, d.current) for d in drift] == [
        ("pandas", "0.0.1", pd.__version__)
    ]  # `platform` is informational: a kernel upgrade is not a reason to distrust a result


def test_drift_reports_a_library_that_is_gone_or_new():
    recorded = {**capture_environment(), "optuna": "9.9.9", "lightgbm": None}
    names = {d.name: (d.recorded, d.current) for d in environment_drift(recorded)}
    assert names["optuna"][0] == "9.9.9"
    assert names["lightgbm"] == (None, lightgbm.__version__)


def test_no_drift_against_oneself():
    assert environment_drift(capture_environment()) == []


def test_fingerprint_records_rows_columns_and_a_content_hash():
    panel = make_panel()
    fp = fingerprint_panel(panel, source="/data/panel.parquet")
    assert fp.n_rows == len(panel)
    assert fp.columns == list(panel.columns)
    assert fp.source == "/data/panel.parquet"
    assert len(fp.content_sha256) == 64
    assert fingerprint_panel(panel.copy()).content_sha256 == fp.content_sha256


@pytest.mark.parametrize(
    "mutate",
    [
        lambda df: df.assign(feature_a=df["feature_a"] + 1e-12),  # a value
        lambda df: df.rename(columns={"feature_d": "feature_z"}),  # a column name
        lambda df: df.iloc[::-1].reset_index(drop=True),  # the row order
        lambda df: df.iloc[:-1],  # a row
        lambda df: df.assign(market_id=df["market_id"].str.upper()),  # a string
    ],
    ids=["value", "column", "row-order", "row-count", "string"],
)
def test_fingerprint_changes_when_the_data_does(mutate):
    panel = make_panel()
    assert (
        fingerprint_panel(mutate(panel)).content_sha256 != fingerprint_panel(panel).content_sha256
    )


def test_fingerprint_survives_a_file_round_trip_and_the_index(tmp_path):
    panel = make_panel()
    panel.to_parquet(tmp_path / "p.parquet")
    want = fingerprint_panel(panel).content_sha256
    assert fingerprint_panel(pd.read_parquet(tmp_path / "p.parquet")).content_sha256 == want
    assert fingerprint_panel(panel.set_axis(range(10, 10 + len(panel)))).content_sha256 == want


def test_fingerprint_does_not_depend_on_the_integer_width():
    a = pd.DataFrame({"n": np.array([1, -2, 3], dtype="int32")})
    b = pd.DataFrame({"n": np.array([1, -2, 3], dtype="int64")})
    assert fingerprint_panel(a).content_sha256 == fingerprint_panel(b).content_sha256


def test_seed_everything_pins_python_and_numpy_global_generators():
    seed_everything({"python_random": 7, "numpy": 11, "lightgbm": 3})
    a = (random.random(), np.random.random())
    seed_everything({"python_random": 7, "numpy": 11, "lightgbm": 3})
    assert (random.random(), np.random.random()) == a
    seed_everything({"python_random": 8, "numpy": 11, "lightgbm": 3})
    assert random.random() != a[0]


def test_seed_everything_leaves_a_library_without_a_seed_alone():
    random.seed(1)
    expected = random.random()
    random.seed(1)
    seed_everything({"python_random": None, "numpy": None})  # nothing to apply
    assert random.random() == expected


def _record(**kw) -> ReproducibilityRecord:
    base = {
        "kind": "backtest",
        "seeds": {"python_random": 1, "numpy": 1, "lightgbm": 1},
        "config": {"stride": 10, "costs": {"taker_bps": 30.0}},
        "data": fingerprint_panel(make_panel(n_dates=5), source="p.parquet"),
        "metrics": {"total_trades": 3, "total_pnl": 0.25},
    }
    return ReproducibilityRecord.capture(**{**base, **kw})


def test_record_captures_environment_seeds_config_data_and_metrics():
    rec = _record()
    assert rec.schema_version == SCHEMA_VERSION
    assert rec.environment == capture_environment()
    assert rec.seeds == {"python_random": 1, "numpy": 1, "lightgbm": 1}
    assert rec.config["costs"] == {"taker_bps": 30.0}
    assert rec.data.n_rows == 30  # 5 dates x 3 markets x 2 labels
    assert rec.metrics["total_trades"] == 3
    assert rec.created_at


def test_record_round_trips_through_json():
    rec = _record()
    again = ReproducibilityRecord.from_dict(json.loads(json.dumps(rec.to_dict())))
    assert again == rec
    assert isinstance(again.data, DataFingerprint)


def test_record_from_dict_rejects_a_newer_schema_with_a_clear_message():
    d = _record().to_dict()
    d["schema_version"] = SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="newer pmlab"):
        ReproducibilityRecord.from_dict(d)


def test_record_from_dict_rejects_garbage_with_the_field_named():
    d = _record().to_dict()
    del d["seeds"]
    with pytest.raises(ValueError, match="seeds"):
        ReproducibilityRecord.from_dict(d)
