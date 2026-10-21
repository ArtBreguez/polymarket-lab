"""The watchdog's second lines of defence for the per-side and per-segment results.

Real inputs cannot make one side's or one segment's statistic non-finite while the book's
is finite, so these force it. The rule they pin: a result that cannot be measured is *not
assessed* (never `ok`, never a verdict), it is named, and it neither hides nor replaces a
result that can be measured.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any

import pytest

from pmlab.backtest.expectation import BacktestExpectation, GroupExpectation, SegmentExpectation
from pmlab.backtest.holdout_gate import HoldoutGateResult, SegmentGateResult
from pmlab.modeling.champion import ChampionManifest
from pmlab.monitoring import watchdog as wd
from pmlab.monitoring.watchdog import _Scope, health_report

_AS_OF = "2026-03-01T00:00:00+00:00"


def _two_point(n: int, mean: float, sd: float) -> list[float]:
    return [mean + sd * (1 if i % 2 == 0 else -1) for i in range(n)]


def _trades(
    n: int,
    *,
    side: str = "yes",
    segment: str = "A",
    pnl: float = 0.2,
    sd: float = 0.6,
    **extra: Any,
) -> list[dict[str, Any]]:
    rows = []
    for ret, term in zip(_two_point(n, pnl, sd), _two_point(n, 0.2, 0.1), strict=True):
        row = {
            "recorded_at": "2026-02-01T10:00:00+00:00",
            "city_or_segment": segment,
            "target_date": "2026-02-05",
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
        row.update(extra)
        rows.append(row)
    return rows


def _champion(
    expectation: BacktestExpectation, passing: tuple[str, ...] = ("A", "B")
) -> ChampionManifest:
    gate = HoldoutGateResult(
        "GO", [SegmentGateResult(s, 100, 5.0, True, "ok") for s in passing], 5.0, 100
    )
    return ChampionManifest(
        "c", Path("c.pkl"), None, gate, "2026-01-01T00:00:00+00:00", "x", expectation=expectation
    )


def _two_sided() -> BacktestExpectation:
    return BacktestExpectation(
        700, 1.0, 0.2, 0.6, 0.2, 0.1,
        by_direction={
            "yes": GroupExpectation(400, 0.2, 0.6, 0.2, 0.1),
            "no": GroupExpectation(300, 0.2, 0.6, 0.2, 0.1),
        },
    )  # fmt: skip


def _report(
    expectation: BacktestExpectation, trades: list[dict[str, Any]], checks: tuple[str, ...]
) -> Any:
    return health_report(
        champion=_champion(expectation), trades=trades, drift=None, as_of=_AS_OF, checks=checks
    )


def _poison_when(monkeypatch: pytest.MonkeyPatch, predicate: Any) -> None:
    original = wd._welch

    def poisoned(
        values: Any, *, counts: Any, groups: Any, pooled: Any, worse_when_higher: bool
    ) -> Any:
        result = original(
            values, counts=counts, groups=groups, pooled=pooled, worse_when_higher=worse_when_higher
        )
        return dataclasses.replace(result, live=math.nan) if predicate(values, counts) else result

    monkeypatch.setattr(wd, "_welch", poisoned)


class TestANonFiniteSideIsNotAssessed:
    def _trades(self) -> list[dict[str, Any]]:
        return _trades(60, side="yes") + _trades(60, side="no")

    @pytest.mark.parametrize("check", ["pnl", "calibration"])
    def test_the_side_is_named_and_the_book_still_stands(
        self, monkeypatch: pytest.MonkeyPatch, check: str
    ) -> None:
        _poison_when(monkeypatch, lambda values, counts: len(counts) == 1)
        report = _report(_two_sided(), self._trades(), (check,))
        result = report.checks[0]
        assert result.status == "ok"  # the book is measured and fine
        for side in ("yes", "no"):
            entry = result.metrics["directions"][side]
            assert entry["status"] == "insufficient"
            assert entry["cause"] == "invalid_statistic"
        assert any("non-finite" in r and "not assessed" in r for r in result.reasons)


class TestANonFiniteSegmentIsNotAssessed:
    @pytest.mark.parametrize("check", ["pnl", "calibration"])
    def test_the_segment_is_named_and_the_book_still_stands(
        self, monkeypatch: pytest.MonkeyPatch, check: str
    ) -> None:
        expectation = BacktestExpectation(
            400, 1.0, 0.2, 0.6, 0.2, 0.1,
            by_segment={
                "A": SegmentExpectation(200, 0.2, 0.6, 0.2, 0.1),
                "B": SegmentExpectation(200, 0.2, 0.6, 0.2, 0.1),
            },
        )  # fmt: skip
        _poison_when(monkeypatch, lambda values, counts: len(values) == 80)
        trades = (
            _trades(80, segment="A") + _trades(80, segment="B") + _trades(20, segment="A")
        )  # the book has 180 trades, each segment's own 100 or 80
        report = _report(expectation, trades, (check,))
        result = report.checks[0]
        assert result.metrics["segments"]["B"]["status"] == "insufficient"
        assert result.metrics["segments"]["B"]["cause"] == "invalid_statistic"
        assert result.metrics["segments"]["A"]["status"] != "insufficient"
        assert result.status == "ok"


class TestTheVerdictRule:
    """Insufficient only when nothing in the check could be measured."""

    def _scope(self, status: str, kind: str = "overall", cause: str | None = None) -> _Scope:
        return _Scope(kind, kind, 10, status, cause, f"{kind} is {status}", {}, kind)

    def test_nothing_measured_is_insufficient_with_the_books_cause(self) -> None:
        overall = self._scope("insufficient", cause="too_few_samples")
        result = wd._assemble(
            "pnl",
            overall,
            {"segments": [self._scope("insufficient", "segment", "no_expectation")]},
            {},
        )
        assert result.status == "insufficient"
        assert result.cause == "too_few_samples"

    def test_one_measured_scope_is_enough_for_a_verdict(self) -> None:
        overall = self._scope("insufficient", cause="invalid_expectation")
        side = self._scope("alert", "side")
        result = wd._assemble("pnl", overall, {"directions": [side]}, {})
        assert result.status == "alert"
        assert result.cause is None

    def test_a_measured_ok_beats_a_thin_scope(self) -> None:
        overall = self._scope("ok")
        thin = self._scope("insufficient", "segment", "too_few_samples")
        assert wd._assemble("pnl", overall, {"segments": [thin]}, {}).status == "ok"

    def test_the_worst_measured_scope_decides(self) -> None:
        overall = self._scope("warn")
        result = wd._assemble("pnl", overall, {"segments": [self._scope("alert", "segment")]}, {})
        assert result.status == "alert"


class TestZeroSpreadIsAGap_NotANan:
    def test_a_side_with_no_spread_at_all_is_judged_on_the_gap(self) -> None:
        """The backtest promised +0.2 with no spread and a NO side earns exactly -1.0 every
        time: the gap is certain. Holm must not trip over a z that does not exist."""
        expectation = BacktestExpectation(
            700, 1.0, 0.2, 0.0, 0.2, 0.1,
            by_direction={
                "yes": GroupExpectation(400, 0.2, 0.0, 0.2, 0.1),
                "no": GroupExpectation(300, 0.2, 0.0, 0.2, 0.1),
            },
        )  # fmt: skip
        trades = _trades(60, side="yes", pnl=0.2, sd=0.0) + _trades(60, side="no", pnl=-1.0, sd=0.0)
        result = _report(expectation, trades, ("pnl",)).checks[0]
        assert result.metrics["directions"]["no"]["status"] == "alert"
        assert result.metrics["directions"]["no"]["z_adjusted"] is None  # "inf", not NaN
        assert result.metrics["directions"]["yes"]["status"] == "ok"
        assert result.status == "alert"


class TestSmallerCorners:
    def test_a_trade_whose_fill_price_is_not_a_number_is_left_out_as_legacy(self) -> None:
        expectation = BacktestExpectation(400, 1.0, 0.2, 0.6, 0.2, 0.1)
        trades = [{k: v for k, v in t.items() if k != "settled_basis"} for t in _trades(60)]
        for t in trades:
            t["fill_price"] = "cheap"
        result = _report(expectation, trades, ("pnl",)).checks[0]
        assert result.metrics["n_excluded_legacy_settlement"] == 60

    def test_two_sides_that_both_promised_nothing_say_so_per_side(self) -> None:
        expectation = BacktestExpectation(
            700, 1.0, -0.1, 0.6, 0.2, 0.1,
            by_direction={
                "yes": GroupExpectation(400, -0.1, 0.6, 0.2, 0.1),
                "no": GroupExpectation(300, 0.0, 0.6, 0.2, 0.1),
            },
        )  # fmt: skip
        result = _report(
            expectation, _trades(40, side="yes") + _trades(40, side="no"), ("pnl",)
        ).checks[0]
        assert result.status == "insufficient"
        assert result.cause == "invalid_expectation"
        assert (
            "YES side -0.100 over 400 trades, NO side +0.000 over 300 trades" in result.reasons[0]
        )
