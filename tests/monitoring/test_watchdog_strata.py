"""The book is held to the backtest trade by trade, in the cell the trade came from.

A champion that recorded its backtest per segment (and per side within a segment) has a
promise for each (segment, side) cell. The reference for the book is the average of the
cells the live trades fell in, weighted by how many fell in each: not the pooled backtest,
which mixes in segments the gate rejected and assumes the live segments trade in the
proportions the backtest did. Without that a book that earns exactly what every cell
promised can alert (the segment mix moved) or stay green (the pooled mean is diluted).

Fixtures use two-point samples (`mean +/- sd`, alternating) so the arithmetic is exact.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pmlab.backtest.expectation import BacktestExpectation, GroupExpectation, SegmentExpectation
from pmlab.backtest.holdout_gate import HoldoutGateResult, SegmentGateResult
from pmlab.modeling.champion import ChampionManifest
from pmlab.monitoring.watchdog import HealthReport, health_report

_AS_OF = "2026-03-01T00:00:00+00:00"


def _two_point(n: int, mean: float, sd: float) -> list[float]:
    return [mean + sd * (1 if i % 2 == 0 else -1) for i in range(n)]


def _trades(
    segment: str,
    n: int,
    *,
    pnl: float,
    sd: float = 0.6,
    brier: float = 0.2,
    brier_sd: float = 0.1,
    side: str = "yes",
    day0: int = 0,
) -> list[dict[str, Any]]:
    rows = []
    for i, (ret, term) in enumerate(
        zip(_two_point(n, pnl, sd), _two_point(n, brier, brier_sd), strict=True)
    ):
        hour = day0 + i
        rows.append(
            {
                "recorded_at": "2026-02-01T10:00:00+00:00",
                "city_or_segment": segment,
                "target_date": f"2026-02-{1 + (hour // 24) % 27:02d}",
                "horizon": f"h{hour}",
                "outcome_label": "YES",
                "direction": side,
                "gamma_price": 0.4,
                "model_prob": math.sqrt(term),
                "flat_stake": 1.0,
                "outcome": "lost",
                "realized_pnl": ret,
                "settlement_version": 1,
                "settled_basis": "quote",
            }
        )
    return rows


def _seg(
    n: int,
    mean: float,
    std: float = 0.6,
    brier: float = 0.2,
    sides: dict[str, GroupExpectation] | None = None,
) -> SegmentExpectation:
    return SegmentExpectation(n, mean, std, brier, 0.1, by_direction=sides)


def _report(
    expectation: BacktestExpectation, trades: list[dict[str, Any]], passing: tuple[str, ...]
) -> HealthReport:
    gate = HoldoutGateResult(
        "GO", [SegmentGateResult(s, 100, 5.0, True, "ok") for s in passing], 5.0, 100
    )
    champion = ChampionManifest(
        "c", Path("c.pkl"), None, gate, "2026-01-01T00:00:00+00:00", "x", expectation=expectation
    )
    return health_report(
        champion=champion,
        trades=trades,
        drift=None,
        as_of=_AS_OF,
        checks=("calibration", "pnl"),
    )


def _check(report: HealthReport, name: str):
    return next(c for c in report.checks if c.name == name)


class TestAGateRejectedSegmentDoesNotDiluteThePromise:
    """Segment A was promised +0.40; rejected segment B lost (-0.10), so the pooled mean is
    +0.15. A book of A trades earning +0.20 has delivered half of what A promised."""

    def _expectation(self) -> BacktestExpectation:
        return BacktestExpectation(
            800, 1.0, 0.15, 0.6, 0.2, 0.1,
            by_segment={"A": _seg(400, 0.40), "B": _seg(400, -0.10)},
        )  # fmt: skip

    def test_the_book_is_held_to_the_segment_it_traded_in(self) -> None:
        pnl = _check(_report(self._expectation(), _trades("A", 800, pnl=0.20), ("A",)), "pnl")
        assert pnl.metrics["mean_backtest"] == pytest.approx(0.40)
        assert pnl.metrics["shortfall_rel"] == pytest.approx(0.5)
        assert pnl.status == "warn"

    def test_the_pooled_expectation_would_have_said_ok(self) -> None:
        pooled = BacktestExpectation(800, 1.0, 0.15, 0.6, 0.2, 0.1)
        pnl = _check(_report(pooled, _trades("A", 800, pnl=0.20), ("A",)), "pnl")
        assert pnl.status == "ok"  # the false green the cells prevent

    def test_the_sequential_test_standardizes_by_the_same_cell(self) -> None:
        expectation = self._expectation()
        trades = _trades("A", 300, pnl=0.20)
        seq = _check(_report(expectation, trades, ("A",)), "pnl").metrics["sequential"]
        # x = (0.40 - r) / 0.6 with r = 0.2 +/- 0.6 has mean 1/3; allowance 0.5 * 0.4 / 0.6
        # + 0.5 / sqrt(400) = 0.3583: the statistic does not climb. Against the pooled +0.15
        # the same trades looked like an outperformance.
        assert seq["n"] == 300


class TestTheSegmentMixOfTheBookIsNotAShortfall:
    """A (+0.35) and B (+0.05) have the same number of backtest trades, so the pooled mean is
    +0.20. Live trades mostly in B, each segment exactly as promised, average +0.08."""

    def _expectation(self) -> BacktestExpectation:
        return BacktestExpectation(
            400, 1.0, 0.20, 0.6, 0.2, 0.1,
            by_segment={"A": _seg(200, 0.35), "B": _seg(200, 0.05)},
        )  # fmt: skip

    def _book(self) -> list[dict[str, Any]]:
        return _trades("A", 100, pnl=0.35) + _trades("B", 900, pnl=0.05, day0=100)

    def test_the_book_is_quiet(self) -> None:
        pnl = _check(_report(self._expectation(), self._book(), ("A", "B")), "pnl")
        assert pnl.metrics["mean_backtest"] == pytest.approx((100 * 0.35 + 900 * 0.05) / 1000)
        assert pnl.status == "ok"
        assert pnl.metrics["sequential"]["signalled"] is False

    def test_the_pooled_expectation_would_have_alerted(self) -> None:
        pooled = BacktestExpectation(400, 1.0, 0.20, 0.6, 0.2, 0.1)
        pnl = _check(_report(pooled, self._book(), ("A", "B")), "pnl")
        assert pnl.status in ("warn", "alert")  # the false alarm the cells prevent

    def test_the_variance_of_the_reference_is_the_weighted_sum_of_the_cells(self) -> None:
        pnl = _check(_report(self._expectation(), self._book(), ("A", "B")), "pnl")
        live = np.array([t["realized_pnl"] for t in self._book()])
        ref_var = (0.1**2) * 0.6**2 / 200 + (0.9**2) * 0.6**2 / 200
        expected = (0.08 - live.mean()) / math.sqrt(live.var(ddof=1) / 1000 + ref_var)
        assert pnl.metrics["z"] == pytest.approx(expected)

    def test_calibration_has_the_same_property(self) -> None:
        expectation = BacktestExpectation(
            400, 1.0, 0.20, 0.6, 0.20, 0.1,
            by_segment={"A": _seg(200, 0.35, brier=0.10), "B": _seg(200, 0.05, brier=0.30)},
        )  # fmt: skip
        trades = _trades("A", 100, pnl=0.35, brier=0.10) + _trades(
            "B", 900, pnl=0.05, brier=0.30, day0=100
        )
        calibration = _check(_report(expectation, trades, ("A", "B")), "calibration")
        assert calibration.metrics["brier_backtest"] == pytest.approx(
            (100 * 0.10 + 900 * 0.30) / 1000
        )
        assert calibration.status == "ok"


class TestCellsInsideASegment:
    def _expectation(self) -> BacktestExpectation:
        a_sides = {
            "yes": GroupExpectation(100, 0.40, 0.6, 0.2, 0.1),
            "no": GroupExpectation(100, 0.00 + 0.05, 0.6, 0.2, 0.1),
        }
        b_sides = {
            "yes": GroupExpectation(100, 0.20, 0.6, 0.2, 0.1),
            "no": GroupExpectation(100, 0.20, 0.6, 0.2, 0.1),
        }
        return BacktestExpectation(
            400, 1.0, 0.2, 0.6, 0.2, 0.1,
            by_direction={
                "yes": GroupExpectation(200, 0.30, 0.6, 0.2, 0.1),
                "no": GroupExpectation(200, 0.12, 0.6, 0.2, 0.1),
            },
            by_segment={"A": _seg(200, 0.225, sides=a_sides), "B": _seg(200, 0.20, sides=b_sides)},
        )  # fmt: skip

    def test_each_side_of_each_segment_is_its_own_promise(self) -> None:
        trades = (
            _trades("A", 20, pnl=0.40, side="yes")
            + _trades("A", 80, pnl=0.05, side="no", day0=50)
            + _trades("B", 100, pnl=0.20, side="yes", day0=200)
        )
        pnl = _check(_report(self._expectation(), trades, ("A", "B")), "pnl")
        assert pnl.metrics["mean_backtest"] == pytest.approx(
            (20 * 0.40 + 80 * 0.05 + 100 * 0.20) / 200
        )
        assert pnl.status == "ok"

    def test_a_side_is_held_to_the_cells_of_its_trades(self) -> None:
        trades = (
            _trades("A", 60, pnl=0.05, side="no")
            + _trades("B", 40, pnl=0.20, side="no", day0=100)
            + _trades("A", 60, pnl=0.40, side="yes", day0=200)
        )
        no = _check(_report(self._expectation(), trades, ("A", "B")), "pnl").metrics["directions"][
            "no"
        ]
        assert no["mean_backtest"] == pytest.approx((60 * 0.05 + 40 * 0.20) / 100)


class TestFallingBackToTheBookLevelRecord:
    def _expectation(self) -> BacktestExpectation:
        # Segment A recorded only its YES cell; its NO trades are held to the book's NO numbers.
        a_sides = {"yes": GroupExpectation(100, 0.40, 0.6, 0.2, 0.1)}
        return BacktestExpectation(
            400, 1.0, 0.2, 0.6, 0.2, 0.1,
            by_direction={
                "yes": GroupExpectation(200, 0.30, 0.6, 0.2, 0.1),
                "no": GroupExpectation(200, 0.10, 0.6, 0.2, 0.1),
            },
            by_segment={"A": _seg(200, 0.25, sides=a_sides), "B": None},
        )  # fmt: skip

    def test_a_missing_cell_uses_the_books_side(self) -> None:
        trades = _trades("A", 50, pnl=0.40, side="yes") + _trades(
            "A", 50, pnl=0.10, side="no", day0=100
        )
        pnl = _check(_report(self._expectation(), trades, ("A", "B")), "pnl")
        assert pnl.metrics["mean_backtest"] == pytest.approx((50 * 0.40 + 50 * 0.10) / 100)
        assert pnl.metrics["n_excluded_no_side"] == 0

    def test_a_segment_recorded_as_absent_uses_the_books_numbers(self) -> None:
        trades = _trades("B", 50, pnl=0.30, side="yes") + _trades(
            "B", 50, pnl=0.10, side="no", day0=100
        )
        pnl = _check(_report(self._expectation(), trades, ("A", "B")), "pnl")
        assert pnl.metrics["mean_backtest"] == pytest.approx((50 * 0.30 + 50 * 0.10) / 100)

    def test_a_champion_promoted_on_an_aggregate_gate_uses_the_cells_for_the_segments_it_knows(
        self,
    ) -> None:
        expectation = BacktestExpectation(
            400, 1.0, 0.2, 0.6, 0.2, 0.1,
            by_direction={"yes": GroupExpectation(400, 0.20, 0.6, 0.2, 0.1)},
            by_segment={"known": _seg(200, 0.50)},
        )  # fmt: skip
        trades = _trades("known", 50, pnl=0.50) + _trades("stranger", 50, pnl=0.20, day0=100)
        gate = HoldoutGateResult("GO", [SegmentGateResult("all", 100, 5.0, True, "ok")], 5.0, 100)
        champion = ChampionManifest(
            "c",
            Path("c.pkl"),
            None,
            gate,
            "2026-01-01T00:00:00+00:00",
            "x",
            expectation=expectation,
        )
        report = health_report(
            champion=champion, trades=trades, drift=None, as_of=_AS_OF, checks=("pnl",)
        )
        # known -> its cell (+0.50), stranger -> the book's YES (+0.20)
        assert report.checks[0].metrics["mean_backtest"] == pytest.approx(0.35)


class TestNothingChangesWithoutASegmentRecord:
    def test_the_book_level_reference_is_used_as_before(self) -> None:
        expectation = BacktestExpectation(400, 1.0, 0.2, 0.6, 0.2, 0.1)
        pnl = _check(_report(expectation, _trades("A", 100, pnl=0.2), ("A", "B")), "pnl")
        assert pnl.metrics["mean_backtest"] == pytest.approx(0.2)
        assert pnl.status == "ok"


class TestASegmentNamedLikeASide:
    def test_it_does_not_collide_with_the_books_side_record(self) -> None:
        expectation = BacktestExpectation(
            400, 1.0, 0.2, 0.6, 0.2, 0.1,
            by_direction={"yes": GroupExpectation(400, 0.20, 0.6, 0.2, 0.1)},
            by_segment={"yes": _seg(200, 0.50), "": _seg(200, 0.10)},
        )  # fmt: skip
        trades = _trades("yes", 50, pnl=0.50) + _trades("", 50, pnl=0.10, day0=100)
        pnl = _check(_report(expectation, trades, ("yes", "")), "pnl")
        assert pnl.metrics["mean_backtest"] == pytest.approx(0.30)


class TestTheSequentialTestUsesTheFinestRecordThatIsLargeEnough:
    """A (segment, side) cell is often promised on fewer than 100 trades. Below the floor the
    sequential test falls back to the book's record for that side, and skips the trade only if
    that is below the floor too; otherwise it would never run for a champion with many segments."""

    def _expectation(self) -> BacktestExpectation:
        return BacktestExpectation(
            400, 1.0, 0.20, 0.6, 0.2, 0.1,
            by_direction={"yes": GroupExpectation(400, 0.20, 0.6, 0.2, 0.1)},
            by_segment={
                "A": _seg(60, 0.50, sides={"yes": GroupExpectation(60, 0.50, 0.6, 0.2, 0.1)}),
                "B": _seg(300, 0.10, sides={"yes": GroupExpectation(300, 0.10, 0.6, 0.2, 0.1)}),
            },
        )  # fmt: skip

    def _book(self) -> list[dict[str, Any]]:
        return _trades("A", 50, pnl=0.50) + _trades("B", 50, pnl=0.10, day0=100)

    def test_a_thin_cell_falls_back_to_the_books_record(self) -> None:
        seq = _check(_report(self._expectation(), self._book(), ("A", "B")), "pnl").metrics[
            "sequential"
        ]
        assert seq["n"] == 100
        assert seq["n_skipped"] == 0
        # A (cell n=60, below the floor) is standardized by the book's +0.20 / 0.6 over 400
        # trades, B by its own cell (n=300): the mean allowance is the average of the two.
        k_book = 0.5 * 0.20 / 0.6 + 0.5 / math.sqrt(400)
        k_b = 0.5 * 0.10 / 0.6 + 0.5 / math.sqrt(300)
        assert seq["allowance"] == pytest.approx((50 * k_book + 50 * k_b) / 100)

    def test_a_cell_at_the_floor_is_used_as_it_is(self) -> None:
        expectation = BacktestExpectation(
            400, 1.0, 0.20, 0.6, 0.2, 0.1,
            by_direction={"yes": GroupExpectation(400, 0.20, 0.6, 0.2, 0.1)},
            by_segment={"A": _seg(100, 0.50, sides={"yes": GroupExpectation(100, 0.50, 0.6, 0.2, 0.1)})},
        )  # fmt: skip
        seq = _check(_report(expectation, _trades("A", 50, pnl=0.50), ("A", "B")), "pnl").metrics[
            "sequential"
        ]
        assert seq["allowance"] == pytest.approx(0.5 * 0.50 / 0.6 + 0.5 / 10)

    def _thin_no(self) -> BacktestExpectation:
        return BacktestExpectation(
            480, 1.0, 0.20, 0.6, 0.2, 0.1,
            by_direction={
                "yes": GroupExpectation(400, 0.20, 0.6, 0.2, 0.1),
                "no": GroupExpectation(80, 0.20, 0.6, 0.2, 0.1),
            },
            by_segment={
                "A": _seg(
                    140, 0.20,
                    sides={
                        "yes": GroupExpectation(80, 0.20, 0.6, 0.2, 0.1),
                        "no": GroupExpectation(60, 0.20, 0.6, 0.2, 0.1),
                    },
                )
            },
        )  # fmt: skip

    def test_a_trade_with_nothing_large_enough_is_skipped_and_counted(self) -> None:
        trades = _trades("A", 50, pnl=0.20, side="yes") + _trades(
            "A", 50, pnl=0.20, side="no", day0=100
        )
        seq = _check(_report(self._thin_no(), trades, ("A", "B")), "pnl").metrics["sequential"]
        # YES: its cell (80) is below the floor, the book's YES (400) is not. NO: its cell (60)
        # and the book's NO (80) are both below it.
        assert seq["n"] == 50
        assert seq["n_skipped"] == 50
        assert "50 trades left out" in seq["reason"]

    def test_the_fixed_test_is_not_held_to_the_floor(self) -> None:
        trades = _trades("A", 50, pnl=0.20, side="yes") + _trades(
            "A", 50, pnl=0.20, side="no", day0=100
        )
        pnl = _check(_report(self._thin_no(), trades, ("A", "B")), "pnl")
        assert pnl.metrics["n_trades"] == 100
        assert pnl.metrics["mean_backtest"] == pytest.approx(0.20)
