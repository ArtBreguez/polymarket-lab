"""The backtest expectation per segment: a segment can bleed while the aggregate hides it.

The gate promotes a champion per segment, so a segment's live health has to be held to
that segment's own backtest. A segment the backtest traded too rarely to promise anything
about is recorded as *absent* (a null), never silently dropped: "we could not check this
segment" must stay distinguishable from "nobody ever looked".
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pmlab.backtest.expectation import BacktestExpectation, SegmentExpectation
from pmlab.backtest.holdout_gate import HoldoutGateResult
from pmlab.modeling.champion import ChampionManifest
from pmlab.modeling.lgbm_baseline import LGBMForecaster


def _trades(spec: dict[str, tuple[int, int]], seed: int = 0) -> pd.DataFrame:
    """Trades per segment: {segment: (n_yes, n_no)}."""
    rng = np.random.default_rng(seed)
    rows = []
    for segment, (n_yes, n_no) in spec.items():
        for side, n, mean in (("yes", n_yes, 0.25), ("no", n_no, 0.05)):
            for _ in range(n):
                rows.append(
                    {
                        "segment": segment,
                        "direction": side,
                        "realized_pnl": float(rng.normal(mean, 0.8)),
                        "predicted_prob": float(rng.uniform(0.1, 0.9)),
                        "label_occurred": bool(rng.random() < 0.5),
                    }
                )
    return pd.DataFrame(rows)


class TestPerSegment:
    def test_each_segment_gets_its_own_numbers(self) -> None:
        trades = _trades({"A": (60, 0), "B": (45, 0)})
        exp = BacktestExpectation.from_trades(trades, flat_stake=2.0)
        assert exp.by_segment is not None
        assert set(exp.by_segment) == {"A", "B"}
        for name in ("A", "B"):
            rows = trades[trades["segment"] == name]
            per_stake = rows["realized_pnl"].to_numpy() / 2.0
            got = exp.by_segment[name]
            assert got is not None
            assert got.n_trades == len(rows)
            assert got.mean_pnl_per_stake == pytest.approx(per_stake.mean())
            assert got.std_pnl_per_stake == pytest.approx(per_stake.std(ddof=1))
            terms = (rows["predicted_prob"] - rows["label_occurred"].astype(float)) ** 2
            assert got.brier_score == pytest.approx(terms.mean())
            assert got.brier_std == pytest.approx(terms.std(ddof=1))

    def test_a_segment_keeps_its_own_split_by_side(self) -> None:
        trades = _trades({"A": (70, 40), "B": (70, 5)})
        exp = BacktestExpectation.from_trades(trades)
        assert exp.by_segment is not None
        a, b = exp.by_segment["A"], exp.by_segment["B"]
        assert a is not None
        assert a.by_direction is not None
        assert set(a.by_direction) == {"yes", "no"}
        rows = trades[(trades["segment"] == "A") & (trades["direction"] == "no")]
        assert a.by_direction["no"].n_trades == len(rows) == 40
        assert a.by_direction["no"].mean_pnl_per_stake == pytest.approx(rows["realized_pnl"].mean())
        # B's NO trades are too thin to be promised on their own: only YES is recorded.
        assert b is not None
        assert b.by_direction is not None
        assert set(b.by_direction) == {"yes"}

    def test_a_segment_without_enough_trades_is_recorded_as_absent(self) -> None:
        exp = BacktestExpectation.from_trades(_trades({"A": (60, 0), "THIN": (12, 0)}))
        assert exp.by_segment is not None
        assert "THIN" in exp.by_segment  # present as a key ...
        assert exp.by_segment["THIN"] is None  # ... with no expectation behind it
        assert exp.by_segment["A"] is not None

    def test_a_segment_at_exactly_the_minimum_is_kept(self) -> None:
        exp = BacktestExpectation.from_trades(_trades({"A": (30, 0), "B": (29, 0)}))
        assert exp.by_segment is not None
        assert exp.by_segment["A"] is not None
        assert exp.by_segment["B"] is None

    def test_the_minimum_follows_min_group_trades(self) -> None:
        exp = BacktestExpectation.from_trades(
            _trades({"A": (60, 0), "THIN": (12, 0)}), min_group_trades=10
        )
        assert exp.by_segment is not None
        assert exp.by_segment["THIN"] is not None

    def test_without_a_segment_column_nothing_is_recorded(self) -> None:
        exp = BacktestExpectation.from_trades(_trades({"A": (60, 0)}).drop(columns=["segment"]))
        assert exp.by_segment is None

    def test_the_top_level_numbers_are_unchanged(self) -> None:
        trades = _trades({"A": (60, 10), "B": (45, 0)})
        with_segments = BacktestExpectation.from_trades(trades)
        without = BacktestExpectation.from_trades(trades.drop(columns=["segment"]))
        assert with_segments.mean_pnl_per_stake == without.mean_pnl_per_stake
        assert with_segments.std_pnl_per_stake == without.std_pnl_per_stake
        assert with_segments.by_direction == without.by_direction

    def test_a_frame_with_no_side_column_has_segments_without_sides(self) -> None:
        trades = _trades({"A": (60, 0)}).drop(columns=["direction"])
        exp = BacktestExpectation.from_trades(trades)
        assert exp.by_segment is not None
        a = exp.by_segment["A"]
        assert a is not None
        assert a.by_direction is None

    def test_segment_names_are_text(self) -> None:
        trades = _trades({"A": (40, 0)})
        trades["segment"] = 7
        exp = BacktestExpectation.from_trades(trades)
        assert exp.by_segment is not None
        assert set(exp.by_segment) == {"7"}


class TestSerialisation:
    def test_round_trips_through_json_including_absent_segments(self) -> None:
        exp = BacktestExpectation.from_trades(_trades({"A": (70, 40), "THIN": (5, 0)}))
        data = json.loads(json.dumps(exp.to_dict()))
        assert data["by_segment"]["THIN"] is None
        assert BacktestExpectation.from_dict(data) == exp

    def test_a_record_without_segments_loads_as_not_recorded(self) -> None:
        old = BacktestExpectation.from_trades(_trades({"A": (60, 0)})).to_dict()
        del old["by_segment"]
        assert BacktestExpectation.from_dict(old).by_segment is None

    def test_not_recorded_and_recorded_empty_are_different(self) -> None:
        exp = BacktestExpectation(100, 1.0, 0.1, 0.9, by_segment={})
        assert exp.to_dict()["by_segment"] == {}
        assert BacktestExpectation.from_dict(exp.to_dict()).by_segment == {}
        assert BacktestExpectation(100, 1.0, 0.1, 0.9).to_dict()["by_segment"] is None


class TestAnUnusableSegmentIsRefused:
    def _base(self) -> dict[str, object]:
        return BacktestExpectation.from_trades(_trades({"A": (70, 40), "B": (50, 0)})).to_dict()

    def _segment(self, data: dict[str, object], name: str) -> dict[str, object]:
        segments = data["by_segment"]
        assert isinstance(segments, dict)
        segment = segments[name]
        assert isinstance(segment, dict)
        return segment

    @pytest.mark.parametrize(
        "changes",
        [
            {"mean_pnl_per_stake": math.nan},
            {"std_pnl_per_stake": -1.0},
            {"n_trades": 1},
            {"n_trades": 10_000},  # more than the whole backtest
            {"brier_score": 2.0},
        ],
    )
    def test_a_bad_number_in_a_segment_is_refused(self, changes: dict[str, object]) -> None:
        data = self._base()
        self._segment(data, "B").update(changes)
        with pytest.raises(ValueError):
            BacktestExpectation.from_dict(data)

    def test_a_bad_number_in_a_segments_side_is_refused(self) -> None:
        data = self._base()
        sides = self._segment(data, "A")["by_direction"]
        assert isinstance(sides, dict)
        sides["no"]["std_pnl_per_stake"] = math.inf
        with pytest.raises(ValueError):
            BacktestExpectation.from_dict(data)

    def test_a_side_with_more_trades_than_its_segment_is_refused(self) -> None:
        data = self._base()
        sides = self._segment(data, "A")["by_direction"]
        assert isinstance(sides, dict)
        sides["no"]["n_trades"] = 500
        with pytest.raises(ValueError, match="segment"):
            BacktestExpectation.from_dict(data)

    def test_segments_that_are_not_an_object_are_refused(self) -> None:
        data = self._base()
        data["by_segment"] = ["A"]
        with pytest.raises(ValueError, match="by_segment"):
            BacktestExpectation.from_dict(data)

    def test_a_segment_that_is_not_an_object_is_refused(self) -> None:
        data = self._base()
        segments = data["by_segment"]
        assert isinstance(segments, dict)
        segments["A"] = "fine"
        with pytest.raises(ValueError, match="segment"):
            BacktestExpectation.from_dict(data)

    def test_a_segment_with_more_trades_than_the_whole_backtest_is_refused(self) -> None:
        with pytest.raises(ValueError, match="more than the whole"):
            BacktestExpectation(
                100, 1.0, 0.1, 0.9, by_segment={"A": SegmentExpectation(200, 0.1, 0.9)}
            )


def _gate(segments: tuple[str, ...]) -> HoldoutGateResult:
    rows = [
        {"realized_pnl": 0.1, "outcome": "won", "segment": s} for s in segments for _ in range(50)
    ]
    return HoldoutGateResult.evaluate(
        pd.DataFrame(rows), required_segments=list(segments), min_trades_per_segment=40
    )


def _model() -> LGBMForecaster:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f1": rng.random(50), "f2": rng.random(50)})  # noqa: N806
    model = LGBMForecaster()
    model.fit(X, pd.Series((X["f1"] > 0.5).astype(int)))
    return model


class TestThePublishedChampionMarksEveryAllowedSegment:
    def test_an_allowed_segment_the_backtest_never_traded_is_marked_absent(
        self, tmp_path: Path
    ) -> None:
        exp = BacktestExpectation.from_trades(_trades({"A": (60, 0)}))
        manifest = ChampionManifest.publish(
            _model(), _gate(("A", "B")), tmp_path, plugin_family="t", expectation=exp
        )
        assert manifest.expectation is not None
        assert manifest.expectation.by_segment is not None
        assert manifest.expectation.by_segment["B"] is None
        on_disk = json.loads((tmp_path / "champion.json").read_text())
        assert on_disk["backtest_expectation"]["by_segment"]["B"] is None
        assert ChampionManifest.load(tmp_path / "champion.json").expectation == manifest.expectation

    def test_a_present_segment_is_left_as_it_was(self, tmp_path: Path) -> None:
        exp = BacktestExpectation.from_trades(_trades({"A": (60, 0), "B": (60, 0)}))
        manifest = ChampionManifest.publish(
            _model(), _gate(("A", "B")), tmp_path, plugin_family="t", expectation=exp
        )
        assert manifest.expectation == exp

    def test_an_expectation_with_no_segment_record_is_left_alone(self, tmp_path: Path) -> None:
        exp = BacktestExpectation(100, 1.0, 0.1, 0.9)
        manifest = ChampionManifest.publish(
            _model(), _gate(("A",)), tmp_path, plugin_family="t", expectation=exp
        )
        assert manifest.expectation == exp
        assert manifest.expectation.by_segment is None

    def test_a_segment_that_is_not_allowed_is_not_added(self, tmp_path: Path) -> None:
        exp = BacktestExpectation.from_trades(_trades({"A": (60, 0)}))
        manifest = ChampionManifest.publish(
            _model(), _gate(("A",)), tmp_path, plugin_family="t", expectation=exp
        )
        assert manifest.expectation is not None
        assert manifest.expectation.by_segment is not None
        assert set(manifest.expectation.by_segment) == {"A"}


class TestTheSegmentRecordRefusesWhatItCannotUse:
    def test_segments_that_are_not_a_mapping(self) -> None:
        with pytest.raises(ValueError, match="by_segment"):
            BacktestExpectation(100, 1.0, 0.1, 0.9, by_segment=["A"])  # type: ignore[arg-type]

    def test_a_segment_that_is_not_a_segment_expectation(self) -> None:
        with pytest.raises(ValueError, match="SegmentExpectation"):
            BacktestExpectation(100, 1.0, 0.1, 0.9, by_segment={"A": "fine"})  # type: ignore[dict-item]
