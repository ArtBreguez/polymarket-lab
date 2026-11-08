"""``TunedForecaster(embargo_days=N)``: every trial is scored under the same hold-back (1.1.0).

The tuner scores each trial with `rolling_origin_eval` and already forwards the options that
shape its folds (``stride``, ``min_train_rows``). ``embargo_days`` is one more of them: tuning
under a hold-back the backtest will not use would pick parameters for the wrong walk-forward.
"""

from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path
from typing import Any

import pytest

from pmlab.backtest import rolling_origin
from pmlab.modeling import tuning
from pmlab.modeling.tuning import TunedForecaster
from tests.modeling.tuning_helpers import build_pick, crafted_panel, pick_space

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("optuna") is None, reason="optuna not installed"
)


def _tuner(**kwargs: Any) -> TunedForecaster:
    base: dict[str, Any] = {
        "n_trials": 3,
        "stride": 1,
        "min_train_rows": 20,
        "random_state": 0,
    }
    return TunedForecaster(build_pick, pick_space, **{**base, **kwargs})


def test_the_default_is_no_embargo_and_the_record_is_unchanged() -> None:
    tuner = _tuner()
    assert tuner.embargo_days is None
    # the keys a 1.0.3 record has, and no new one: an untuned-embargo record does not change
    assert set(tuner.tuning_config()) == {
        "metric",
        "n_trials",
        "policy",
        "costs",
        "stride",
        "min_train_rows",
        "flat_stake",
        "best_score",
    }


def test_the_embargo_is_recorded_when_set() -> None:
    assert _tuner(embargo_days=4).tuning_config()["embargo_days"] == 4


def test_it_is_keyword_only_and_after_the_existing_parameters() -> None:
    import inspect

    params = inspect.signature(TunedForecaster).parameters
    assert params["embargo_days"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["embargo_days"].default is None
    assert list(params)[-1] == "embargo_days"


@pytest.mark.parametrize("value", [None, 0, 3])
def test_every_trial_is_scored_with_the_embargo(
    monkeypatch: pytest.MonkeyPatch, value: int | None
) -> None:
    seen: list[dict[str, Any]] = []
    real = rolling_origin.rolling_origin_eval

    def spy(*args: Any, **kwargs: Any):
        seen.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(tuning, "rolling_origin_eval", spy)
    _tuner(embargo_days=value).tune(crafted_panel())
    assert len(seen) == 3
    assert all(kw["embargo_days"] == value for kw in seen)


def test_a_search_under_a_long_embargo_sees_fewer_training_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows: dict[int | None, int] = {}
    real = rolling_origin.rolling_origin_eval

    def spy(*args: Any, **kwargs: Any):
        result = real(*args, **kwargs)
        rows[kwargs["embargo_days"]] = sum(s["train_rows"] for s in result.steps)
        return result

    monkeypatch.setattr(tuning, "rolling_origin_eval", spy)
    for n in (None, 10):
        _tuner(n_trials=1, embargo_days=n).tune(crafted_panel())
    assert rows[10] < rows[None]


@pytest.mark.parametrize("bad", [-1, -7])
def test_a_negative_embargo_is_refused_at_construction(bad: int) -> None:
    with pytest.raises(ValueError, match="embargo_days"):
        _tuner(embargo_days=bad)


@pytest.mark.parametrize("bad", [2.0, "2", True])
def test_a_non_int_embargo_is_refused_at_construction(bad: object) -> None:
    with pytest.raises(TypeError, match="embargo_days"):
        _tuner(embargo_days=bad)


def test_the_embargo_survives_save_and_load(tmp_path: Path) -> None:
    tuner = _tuner(embargo_days=5).tune(crafted_panel())
    tuner.save(tmp_path / "t.pkl")
    loaded = TunedForecaster.load(tmp_path / "t.pkl")
    assert loaded.embargo_days == 5
    assert loaded.tuning_config()["embargo_days"] == 5


def test_a_tuner_saved_before_1_1_0_loads_without_an_embargo(tmp_path: Path) -> None:
    tuner = _tuner().tune(crafted_panel())
    tuner.save(tmp_path / "t.pkl")
    data = pickle.loads((tmp_path / "t.pkl").read_bytes())  # noqa: S301
    assert "embargo_days" not in data["tuning"]  # exactly what 1.0.3 wrote
    loaded = TunedForecaster.load(tmp_path / "t.pkl")
    assert loaded.embargo_days is None
    assert "embargo_days" not in loaded.tuning_config()
