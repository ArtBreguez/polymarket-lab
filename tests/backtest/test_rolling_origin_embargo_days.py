"""``rolling_origin_eval(..., embargo_days=N)``: a stricter hold-back, never a looser one.

1.0.3 trains a fold only on rows whose label was known at the fold's decision (``market_close_ts``,
else the day after the decision). ``embargo_days`` adds a second condition on top of that one:
the row's ``decision_date`` must be at least ``N`` days before the fold's. The two are AND-ed, so
the embargo can only remove rows from a fold, and ``None`` (the default) is the 1.0.3 rule exactly.
"""

from __future__ import annotations

import inspect
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pmlab.backtest.rolling_origin import rolling_origin_eval
from pmlab.data.panel_builder import build_panel
from pmlab.datasets import load_sample

REPO = Path(__file__).resolve().parents[2]


class _Recorder:
    """Predicts 0.5 and remembers the panel rows each fold trained and tested on."""

    def __init__(self) -> None:
        self.folds: list[tuple[pd.Index, pd.Index]] = []
        self._train: pd.Index = pd.Index([])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._train = X.index

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.folds.append((self._train, X.index))
        return np.full((len(X), 2), 0.5)


def _panel(
    n_days: int = 14, lag_days: int = 1, *, timed: bool = True, decision_hour: int = 7
) -> pd.DataFrame:
    """Two markets a day, two bins each, decided at ``decision_hour`` UTC on day D.

    The market asks about day ``D + lag_days`` and ends with it (``market_close_ts``), as the
    weather plugin's does; ``timed=False`` drops both timestamp columns.
    """
    start = date(2025, 4, 1)
    rows: list[dict[str, object]] = []
    for i in range(n_days):
        day = start + timedelta(days=i)
        decided = datetime(day.year, day.month, day.day, decision_hour, 30, tzinfo=UTC)
        ends = datetime.combine(day + timedelta(days=lag_days + 1), datetime.min.time(), UTC)
        for market in range(2):
            for label in ("A", "B"):
                row: dict[str, object] = {
                    "market_id": f"m{i}-{market}",
                    "decision_date": day.isoformat(),
                    "outcome_label": label,
                    "winning_label": "A",
                    "market_price": 0.5,
                    "feature_x": float(i),
                }
                if timed:
                    row["decision_ts"] = decided.isoformat()
                    row["market_close_ts"] = ends.isoformat()
                rows.append(row)
    return pd.DataFrame(rows)


def _folds(panel: pd.DataFrame, **kwargs: object) -> dict[str, pd.Index]:
    """eval_date -> the panel rows its fold trained on."""
    model = _Recorder()
    rolling_origin_eval(panel, model, min_train_rows=1, stride=1, **kwargs)  # type: ignore[arg-type]
    return {
        panel.loc[eval_idx, "decision_date"].iloc[0]: train_idx
        for train_idx, eval_idx in model.folds
    }


def _day(value: str) -> date:
    return date.fromisoformat(value)


# --------------------------------------------------------------------------- the signature


def test_embargo_days_is_a_keyword_only_parameter_defaulting_to_none() -> None:
    param = inspect.signature(rolling_origin_eval).parameters["embargo_days"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is None


def test_the_existing_parameters_keep_their_order_and_kind() -> None:
    names = list(inspect.signature(rolling_origin_eval).parameters)
    assert names[:8] == [
        "panel",
        "model",
        "min_train_rows",
        "stride",
        "flat_stake",
        "taker_bps",
        "costs",
        "policy",
    ]
    assert names[-1] == "embargo_days"


# --------------------------------------------------------------------------- the default is 1.0.3


@pytest.mark.parametrize("lag_days", [0, 1, 3])
@pytest.mark.parametrize("timed", [True, False])
def test_none_is_the_same_as_leaving_it_out(lag_days: int, timed: bool) -> None:
    panel = _panel(lag_days=lag_days, timed=timed)
    left_out = _folds(panel)
    none = _folds(panel, embargo_days=None)
    assert left_out.keys() == none.keys()
    for eval_date in left_out:
        assert left_out[eval_date].equals(none[eval_date])


@pytest.fixture(scope="module")
def sample_panel() -> pd.DataFrame:
    """The sample's backtest panel as the tutorial and the home page build it: 120 days."""
    sample = load_sample()
    resolver = sample.plugin()
    rows = [
        resolver.build_training_rows(
            spec, "previous_evening", decision_time=f"{day}T12:00:00+00:00", exclusions={}
        )
        for day in sample.days
        for spec in sample.plugin(day).discover_markets()
    ]
    panel = build_panel(rows, check_leakage=True)
    dates = sorted(panel["decision_date"].unique())
    return panel[panel["decision_date"] < dates[120]].reset_index(drop=True)


def _the_1_0_3_site_folds() -> list[dict[str, object]]:
    """The folds 1.0.3 produced on the sample, as recorded for the docs home page."""
    data = json.loads((REPO / "docs" / "assets" / "data" / "sample-run.json").read_text())
    return list(data["folds"])


def test_the_default_reproduces_the_1_0_3_folds_of_the_sample(sample_panel: pd.DataFrame) -> None:
    """59 folds, each trained on exactly the rows 1.0.3 trained it on (the home page's numbers)."""
    expected = _the_1_0_3_site_folds()
    for kwargs in ({}, {"embargo_days": None}):
        model = _Recorder()
        result = rolling_origin_eval(sample_panel, model, min_train_rows=2700, stride=1, **kwargs)  # type: ignore[arg-type]
        got = [(s["eval_date"], s["train_rows"]) for s in result.steps]
        assert len(got) == 59
        assert got == [(f["test"], f["train_rows"]) for f in expected]


def test_the_default_results_are_identical_with_a_real_model(sample_panel: pd.DataFrame) -> None:
    """Same trades, to the last bit, with the parameter left out and with ``None``."""
    from pmlab import LGBMForecaster

    def run(**kwargs: object) -> pd.DataFrame:
        model = LGBMForecaster(n_estimators=30, num_leaves=6, random_state=7, n_jobs=1)
        return rolling_origin_eval(
            sample_panel,
            model,
            min_train_rows=2700,
            stride=10,
            **kwargs,  # type: ignore[arg-type]
        ).trades

    left_out = run()
    assert len(left_out) > 0
    pd.testing.assert_frame_equal(left_out, run(embargo_days=None), check_exact=True)


def test_zero_and_one_day_change_nothing_because_the_fold_rule_already_needs_a_day(
    sample_panel: pd.DataFrame,
) -> None:
    """Rows are strictly before the fold (``decision_date < eval_date``): N <= 1 asks no more."""
    base = _folds(sample_panel)
    for n in (0, 1):
        got = _folds(sample_panel, embargo_days=n)
        assert got.keys() == base.keys()
        assert all(got[d].equals(base[d]) for d in base), n


# --------------------------------------------------------------------------- the embargo itself


@pytest.mark.parametrize("n", [2, 3, 5, 9])
def test_no_fold_trains_on_a_row_inside_the_embargo(n: int) -> None:
    panel = _panel(n_days=20)
    got = _folds(panel, embargo_days=n)
    assert got, "the check must cover folds, not pass vacuously"
    for eval_date, train_idx in got.items():
        latest = _day(panel.loc[train_idx, "decision_date"].max())
        assert (_day(eval_date) - latest).days >= n, (eval_date, n)
        gaps = [(_day(eval_date) - _day(d)).days for d in panel.loc[train_idx, "decision_date"]]
        assert min(gaps) >= n


def test_the_embargo_is_exact_not_just_safe() -> None:
    """With labels known a day later, N = 5 trains on the days up to eval - 5, all of them."""
    panel = _panel(n_days=20)
    got = _folds(panel, embargo_days=5)
    for eval_date, train_idx in got.items():
        allowed = {
            d for d in panel["decision_date"].unique() if (_day(eval_date) - _day(d)).days >= 5
        }
        assert set(panel.loc[train_idx, "decision_date"]) == allowed


def test_embargo_two_gives_fewer_or_equal_rows_in_every_fold_of_the_sample(
    sample_panel: pd.DataFrame,
) -> None:
    base = _folds(sample_panel)
    got = _folds(sample_panel, embargo_days=2)
    assert got.keys() <= base.keys()
    for eval_date, train_idx in got.items():
        assert len(train_idx) <= len(base[eval_date])
        assert set(train_idx) <= set(base[eval_date])


def test_a_longer_embargo_on_the_sample_trains_on_strictly_fewer_rows(
    sample_panel: pd.DataFrame,
) -> None:
    base = _folds(sample_panel)
    got = _folds(sample_panel, embargo_days=7)
    assert any(len(got[d]) < len(base[d]) for d in got)
    for eval_date, train_idx in got.items():
        latest = _day(sample_panel.loc[train_idx, "decision_date"].max())
        assert (_day(eval_date) - latest).days >= 7


def test_the_embargo_bites_where_labels_are_known_sooner_than_the_default_assumes() -> None:
    """Same-day markets: their label is known by the next morning, so 1.0.3 trains on E-1."""
    panel = _panel(n_days=12, lag_days=0)
    base = _folds(panel)
    one_day_old = [
        d
        for d, idx in base.items()
        if (_day(d) - _day(panel.loc[idx, "decision_date"].max())).days == 1
    ]
    assert one_day_old, "premise: the default trains on the day before the fold"
    got = _folds(panel, embargo_days=3)
    for eval_date, train_idx in got.items():
        assert (_day(eval_date) - _day(panel.loc[train_idx, "decision_date"].max())).days >= 3
    assert any(len(got[d]) < len(base[d]) for d in got)


def test_min_train_rows_counts_the_rows_left_after_the_embargo() -> None:
    panel = _panel(n_days=14)
    steps = rolling_origin_eval(
        panel, _Recorder(), min_train_rows=8, stride=1, embargo_days=4
    ).steps
    assert steps, "premise: some fold has enough rows"
    assert all(s["train_rows"] >= 8 for s in steps)
    base = rolling_origin_eval(panel, _Recorder(), min_train_rows=8, stride=1).steps
    assert len(steps) < len(base)  # the embargo pushed the first fold later


# --------------------------------------------------------------------------- never looser


@pytest.mark.parametrize("lag_days", [0, 1, 2, 4])
@pytest.mark.parametrize("timed", [True, False])
@pytest.mark.parametrize("n", [0, 1, 2, 3, 6])
def test_the_embargo_is_the_intersection_with_the_label_known_rule(
    lag_days: int, timed: bool, n: int
) -> None:
    """For every lag and every N, a fold trains on (the 1.0.3 rows) and (>= N days old): no more."""
    panel = _panel(n_days=16, lag_days=lag_days, timed=timed)
    base = _folds(panel)
    got = _folds(panel, embargo_days=n)
    for eval_date, train_idx in got.items():
        old_enough = {
            i
            for i in base[eval_date]
            if (_day(eval_date) - _day(panel.loc[i, "decision_date"])).days >= n
        }
        assert set(train_idx) == old_enough


def test_a_small_embargo_cannot_pull_in_a_row_whose_label_is_not_known_yet() -> None:
    """Labels that arrive 4 days late stay out under N = 0, 1, 2, as under the default."""
    panel = _panel(n_days=16, lag_days=4)
    base = _folds(panel)
    for n in (0, 1, 2):
        got = _folds(panel, embargo_days=n)
        assert got.keys() == base.keys()
        assert all(got[d].equals(base[d]) for d in base)
    for eval_date, train_idx in base.items():
        cutoff = pd.Timestamp(f"{eval_date}T07:30:00+00:00")
        known = pd.to_datetime(panel.loc[train_idx, "market_close_ts"], utc=True) <= cutoff
        assert known.all()


def test_one_late_label_stays_out_whatever_the_embargo() -> None:
    panel = _panel(n_days=12, lag_days=1)
    late_row = panel.index[panel["decision_date"] == "2025-04-03"][0]
    panel.loc[late_row, "market_close_ts"] = "2025-04-30T00:00:00+00:00"  # known only in May
    for n in (None, 0, 1, 2, 5):
        for train_idx in _folds(panel, embargo_days=n).values():
            assert late_row not in train_idx


def test_without_market_close_ts_the_fallback_still_holds_under_a_zero_embargo() -> None:
    """A panel that does not say when labels arrive is assumed to resolve a day later."""
    panel = _panel(n_days=10, timed=False)
    got = _folds(panel, embargo_days=0)
    for eval_date, train_idx in got.items():
        latest = _day(panel.loc[train_idx, "decision_date"].max())
        assert (_day(eval_date) - latest).days >= 2


def test_an_unreadable_market_close_ts_falls_back_and_is_still_embargoed() -> None:
    panel = _panel(n_days=10)
    panel["market_close_ts"] = "not a time"
    for n in (None, 0, 4):
        for eval_date, train_idx in _folds(panel, embargo_days=n).items():
            latest = _day(panel.loc[train_idx, "decision_date"].max())
            assert (_day(eval_date) - latest).days >= max(2, n or 0)


# --------------------------------------------------------------------------- validation


@pytest.mark.parametrize("bad", [-1, -30])
def test_a_negative_embargo_is_a_value_error(bad: int) -> None:
    with pytest.raises(ValueError, match="embargo_days"):
        rolling_origin_eval(_panel(), _Recorder(), embargo_days=bad)


@pytest.mark.parametrize("bad", [2.0, 1.5, "2", True, False, [2], float("nan")])
def test_a_value_that_is_not_an_int_is_a_type_error(bad: object) -> None:
    with pytest.raises(TypeError, match="embargo_days"):
        rolling_origin_eval(_panel(), _Recorder(), embargo_days=bad)  # type: ignore[arg-type]


def test_a_numpy_integer_is_accepted_as_an_int() -> None:
    panel = _panel(n_days=14)
    assert _folds(panel, embargo_days=np.int64(4)).keys() == _folds(panel, embargo_days=4).keys()


def test_an_invalid_value_is_refused_before_anything_is_fitted() -> None:
    model = _Recorder()
    with pytest.raises(ValueError):
        rolling_origin_eval(_panel(), model, embargo_days=-1)
    assert model.folds == [] and len(model._train) == 0


def test_an_invalid_value_is_refused_even_when_no_fold_could_run() -> None:
    with pytest.raises(ValueError, match="embargo_days"):
        rolling_origin_eval(_panel(n_days=2), _Recorder(), min_train_rows=10_000, embargo_days=-1)


def test_a_huge_embargo_trains_nothing_and_returns_an_empty_result() -> None:
    result = rolling_origin_eval(_panel(n_days=10), _Recorder(), min_train_rows=1, embargo_days=999)
    assert result.steps == [] and result.trades.empty
