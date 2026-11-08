"""Issue #47: a fold never trains on a label that was not yet known at its decision.

``rolling_origin_eval`` used to train on every row with ``decision_date < eval_date``. A
``weather_tmax`` decision taken on day D asks about day D+1, so the row decided on E-1 is labelled
with day E's temperature: a value that exists only after the decision at E was taken. These tests
pin the repaired rule: a row trains a fold only if its label was known at the fold's decision
(``market_close_ts <= the fold's decision_ts``), whatever the lag between a decision and its label.
"""

from __future__ import annotations

import io
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from pmlab.backtest.rolling_origin import rolling_origin_eval
from pmlab.data.panel_builder import build_panel
from pmlab.datasets import load_sample


class _Recorder:
    """A model that predicts 0.5 and remembers which panel rows each fold trained and tested on."""

    def __init__(self) -> None:
        self.folds: list[tuple[pd.Index, pd.Index]] = []
        self._train: pd.Index = pd.Index([])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._train = X.index

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.folds.append((self._train, X.index))
        return np.full((len(X), 2), 0.5)


def _utc(value: str | pd.Series) -> pd.Series:
    return pd.to_datetime(value, utc=True)


# --------------------------------------------------------------------------- the bundled sample


@pytest.fixture(scope="module")
def sample_panel() -> pd.DataFrame:
    """The sample's panel, built the way the tutorial builds it (one 12:00 UTC decision a day)."""
    sample = load_sample()
    resolver = sample.plugin()
    rows = [
        resolver.build_training_rows(
            spec, "previous_evening", decision_time=f"{day}T12:00:00+00:00", exclusions={}
        )
        for day in sample.days
        for spec in sample.plugin(day).discover_markets()
    ]
    return build_panel(rows, check_leakage=True).reset_index(drop=True)


def test_the_sample_labels_resolve_after_the_next_decision(sample_panel: pd.DataFrame) -> None:
    """The premise of the bug, on the real panel: the row decided on E-1 resolves after E decides."""
    decided = _utc(sample_panel["decision_ts"])
    resolved = _utc(sample_panel["market_close_ts"])
    one_day_later = resolved - decided
    assert (one_day_later > pd.Timedelta(hours=12)).all()  # the label is the *next* day's


def test_no_fold_of_the_sample_trains_on_a_label_resolved_after_its_decision(
    sample_panel: pd.DataFrame,
) -> None:
    model = _Recorder()
    result = rolling_origin_eval(sample_panel, model, min_train_rows=500, stride=5)
    assert len(result.steps) >= 20, "the check must cover many folds, not vacuously pass"
    assert len(model.folds) == len(result.steps)

    for train_idx, eval_idx in model.folds:
        decision = _utc(sample_panel.loc[eval_idx, "decision_ts"]).min()
        resolved = _utc(sample_panel.loc[train_idx, "market_close_ts"])
        late = sample_panel.loc[train_idx][resolved > decision]
        assert late.empty, (
            f"fold deciding at {decision} trained on {len(late)} rows whose label resolves "
            f"after it (latest {resolved.max()})"
        )


def test_the_last_day_a_sample_fold_trains_on_is_two_before_its_eval_date(
    sample_panel: pd.DataFrame,
) -> None:
    model = _Recorder()
    rolling_origin_eval(sample_panel, model, min_train_rows=500, stride=5)
    for train_idx, eval_idx in model.folds:
        eval_day = date.fromisoformat(sample_panel.loc[eval_idx, "decision_date"].iloc[0])
        last_train = date.fromisoformat(sample_panel.loc[train_idx, "decision_date"].max())
        assert last_train == eval_day - timedelta(days=2)


# --------------------------------------------------------------------------- the exact off-by-one


def _panel(
    n_days: int = 8, lag_days: int = 1, *, timed: bool = True, decision_hour: int = 7
) -> pd.DataFrame:
    """Two markets a day, two bins each, decided at ``decision_hour`` UTC on day D.

    The market asks about day ``D + lag_days`` and ends with it, as the weather plugin's does.
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


def _trained_days(panel: pd.DataFrame, model: _Recorder) -> dict[str, list[str]]:
    """eval_date -> the distinct decision dates its fold trained on."""
    return {
        panel.loc[eval_idx, "decision_date"].iloc[0]: sorted(
            panel.loc[train_idx, "decision_date"].unique()
        )
        for train_idx, eval_idx in model.folds
    }


def _train_frame(panel: pd.DataFrame, model: _Recorder, eval_day: str) -> pd.DataFrame:
    """The rows the fold evaluated on ``eval_day`` trained on."""
    for train_idx, eval_idx in model.folds:
        if panel.loc[eval_idx, "decision_date"].iloc[0] == eval_day:
            return panel.loc[train_idx]
    raise AssertionError(f"no fold evaluated {eval_day}")


def test_the_row_decided_the_day_before_is_held_back() -> None:
    """The off-by-one itself: eval on Apr 4 must not train on Apr 3 (its label is Apr 4's)."""
    panel = _panel()
    model = _Recorder()
    rolling_origin_eval(panel, model, min_train_rows=1, stride=1)

    trained = _trained_days(panel, model)
    assert trained["2025-04-04"] == ["2025-04-01", "2025-04-02"]  # not 2025-04-03
    assert trained["2025-04-08"] == [f"2025-04-0{d}" for d in range(1, 7)]  # up to the 6th


def test_the_first_fold_waits_for_a_label_to_exist() -> None:
    panel = _panel()
    model = _Recorder()
    result = rolling_origin_eval(panel, model, min_train_rows=1, stride=1)
    assert [s["eval_date"] for s in result.steps][0] == "2025-04-03"  # Apr 2's label is Apr 3's
    assert result.steps[0]["train_rows"] == 4  # Apr 1 only: two markets x two bins


def test_the_hold_back_follows_the_data_not_a_constant() -> None:
    """Markets that resolve three days after the decision keep three more days out."""
    panel = _panel(n_days=10, lag_days=3)
    model = _Recorder()
    rolling_origin_eval(panel, model, min_train_rows=1, stride=1)

    trained = _trained_days(panel, model)
    assert trained["2025-04-08"] == [f"2025-04-0{d}" for d in range(1, 5)]  # up to the 4th
    for eval_day, days in trained.items():
        assert max(days) <= (date.fromisoformat(eval_day) - timedelta(days=4)).isoformat()


def test_a_label_known_exactly_at_the_decision_is_usable_a_second_later_is_not() -> None:
    panel = _panel(n_days=3, lag_days=0)  # decided on D, resolves at the end of D
    decision = datetime(2025, 4, 3, 7, 30, tzinfo=UTC)
    panel.loc[panel["market_id"] == "m0-0", "market_close_ts"] = decision.isoformat()
    panel.loc[panel["market_id"] == "m0-1", "market_close_ts"] = (
        decision + timedelta(seconds=1)
    ).isoformat()
    model = _Recorder()
    rolling_origin_eval(panel, model, min_train_rows=1, stride=1)

    markets = set(_train_frame(panel, model, "2025-04-03")["market_id"])
    assert "m0-0" in markets  # known AT the decision: usable
    assert "m0-1" not in markets  # known one second after it: not yet


def test_a_label_closing_between_midnight_and_the_decision_is_usable() -> None:
    """The cut is the fold's decision_ts, not the start of its day: decide 07:30, close 06:00."""
    panel = _panel(n_days=3, lag_days=0)  # m1-* are decided on Apr 2
    panel.loc[panel["market_id"] == "m1-0", "market_close_ts"] = datetime(
        2025, 4, 3, 6, 0, tzinfo=UTC
    ).isoformat()
    panel.loc[panel["market_id"] == "m1-1", "market_close_ts"] = datetime(
        2025, 4, 3, 8, 0, tzinfo=UTC
    ).isoformat()
    model = _Recorder()
    rolling_origin_eval(panel, model, min_train_rows=1, stride=1)

    markets = set(_train_frame(panel, model, "2025-04-03")["market_id"])
    assert "m1-0" in markets  # closed at 06:00, before the 07:30 decision of Apr 3
    assert "m1-1" not in markets  # closes at 08:00, after it


# --------------------------------------------------------------------------- panels without times


def test_a_panel_without_timestamps_holds_back_one_decision_day() -> None:
    """Nothing says when a label was known: assume the next day's, as every shipped plugin's is."""
    panel = _panel(timed=False)
    model = _Recorder()
    rolling_origin_eval(panel, model, min_train_rows=1, stride=1)

    trained = _trained_days(panel, model)
    assert trained["2025-04-04"] == ["2025-04-01", "2025-04-02"]
    assert min(trained) == "2025-04-03"


def test_a_row_with_no_close_time_in_a_timed_panel_gets_the_same_hold_back() -> None:
    panel = _panel(lag_days=0)  # same-day labels: a timed row trains the fold one day later...
    panel["market_close_ts"] = panel["market_close_ts"].astype(object)
    panel.loc[panel["market_id"] == "m4-0", "market_close_ts"] = None  # ...this one has no time
    model = _Recorder()
    rolling_origin_eval(panel, model, min_train_rows=1, stride=1)

    on_the_6th = set(_train_frame(panel, model, "2025-04-06")["market_id"])
    assert "m4-1" in on_the_6th  # decided Apr 5, resolved by the end of Apr 5
    assert "m4-0" not in on_the_6th  # untimed: the one-day fallback keeps Apr 5 out
    assert "m4-0" in set(_train_frame(panel, model, "2025-04-07")["market_id"])


def test_without_a_decision_ts_a_fold_cuts_at_the_start_of_its_day() -> None:
    panel = _panel(n_days=3, lag_days=0).drop(columns=["decision_ts"])  # m1-* decided on Apr 2
    panel.loc[panel["market_id"] == "m1-0", "market_close_ts"] = datetime(
        2025, 4, 3, 0, 0, tzinfo=UTC
    ).isoformat()
    panel.loc[panel["market_id"] == "m1-1", "market_close_ts"] = datetime(
        2025, 4, 3, 6, 0, tzinfo=UTC
    ).isoformat()
    model = _Recorder()
    rolling_origin_eval(panel, model, min_train_rows=1, stride=1)

    markets = set(_train_frame(panel, model, "2025-04-03")["market_id"])
    assert "m1-0" in markets  # known at the very start of the eval day
    assert "m1-1" not in markets  # known six hours into it: with no decision time, too late


# --------------------------------------------------------------------------- dates that are not dates


@pytest.mark.parametrize("bad", ["2025-04-32", "2025-13-01", "yesterday", ""])
def test_a_decision_date_that_is_not_a_calendar_date_is_refused(bad: str) -> None:
    """Skipping the unreadable rows would shrink the training set silently: say so instead."""
    panel = _panel()
    panel.loc[panel.index[-1], "decision_date"] = bad
    with pytest.raises(ValueError, match=r"decision_date must be a 'YYYY-MM-DD' calendar date"):
        rolling_origin_eval(panel, _Recorder(), min_train_rows=1, stride=1)


def test_a_timestamp_decision_date_still_works() -> None:
    """A datetime column was always accepted (it is cast to text first)."""
    panel = _panel(timed=False)
    panel["decision_date"] = pd.to_datetime(panel["decision_date"])
    model = _Recorder()
    rolling_origin_eval(panel, model, min_train_rows=1, stride=1)
    assert model.folds


# --------------------------------------------------------------------------- how times arrive


def _as_tz_aware(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.assign(
        decision_ts=pd.to_datetime(panel["decision_ts"]),
        market_close_ts=pd.to_datetime(panel["market_close_ts"]),
    )


def _as_naive_utc(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.assign(
        decision_ts=pd.to_datetime(panel["decision_ts"]).dt.tz_localize(None),
        market_close_ts=pd.to_datetime(panel["market_close_ts"]).dt.tz_localize(None),
    )


def _with_a_space(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.assign(
        decision_ts=panel["decision_ts"].str.replace("T", " "),
        market_close_ts=panel["market_close_ts"].str.replace("T", " "),
    )


def _through_csv(panel: pd.DataFrame) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(panel.to_csv(index=False)))


@pytest.mark.parametrize(
    "convert", [_as_tz_aware, _as_naive_utc, _with_a_space, _through_csv], ids=lambda f: f.__name__
)
def test_the_times_may_arrive_as_a_parquet_or_csv_panel_has_them(convert) -> None:
    """``pmlab backtest`` reads panels from files: the same folds whatever the time's dtype."""
    panel = _panel(lag_days=2)
    reference, model = _Recorder(), _Recorder()
    rolling_origin_eval(panel, reference, min_train_rows=1, stride=1)
    converted = convert(panel)
    rolling_origin_eval(converted, model, min_train_rows=1, stride=1)

    assert _trained_days(converted, model) == _trained_days(panel, reference)
    assert min(_trained_days(panel, reference)) == "2025-04-04"  # a 2-day lag: first fold is Apr 4
