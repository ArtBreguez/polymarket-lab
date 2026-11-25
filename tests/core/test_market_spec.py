"""Tests for generic MarketSpec and OutcomeBin."""

from __future__ import annotations

import math
from typing import Any

import pytest
from pydantic import ValidationError

from pmlab.core.market_spec import MarketSpec, OutcomeBin


class TestOutcomeBin:
    def test_unbounded_contains_any(self) -> None:
        b = OutcomeBin(label="YES")
        assert b.contains(0.0) is True
        assert b.contains(999.9) is True
        assert b.contains(-5.0) is True

    def test_lower_bound_inclusive(self) -> None:
        b = OutcomeBin(label="hot", lower=30.0, lower_inclusive=True)
        assert b.contains(30.0) is True
        assert b.contains(29.99) is False

    def test_lower_bound_exclusive(self) -> None:
        b = OutcomeBin(label="hot", lower=30.0, lower_inclusive=False)
        assert b.contains(30.0) is False
        assert b.contains(30.01) is True

    def test_upper_bound_inclusive(self) -> None:
        b = OutcomeBin(label="warm", upper=25.0, upper_inclusive=True)
        assert b.contains(25.0) is True
        assert b.contains(25.01) is False

    def test_upper_bound_exclusive(self) -> None:
        b = OutcomeBin(label="warm", upper=25.0, upper_inclusive=False)
        assert b.contains(25.0) is False
        assert b.contains(24.99) is True

    def test_bounded_range(self) -> None:
        b = OutcomeBin(
            label="30°C",
            lower=29.5,
            upper=30.5,
            lower_inclusive=True,
            upper_inclusive=False,
        )
        assert b.contains(29.5) is True
        assert b.contains(30.0) is True
        assert b.contains(30.5) is False
        assert b.contains(29.4) is False


class TestMarketSpec:
    def _make_spec(self) -> MarketSpec:
        return MarketSpec(
            market_id="abc123",
            slug="will-it-rain",
            question="Will it rain in NYC on May 10?",
            outcome_bins=[OutcomeBin(label="YES"), OutcomeBin(label="NO")],
            close_time="2026-05-10T20:00:00Z",
            market_family="binary",
        )

    def test_minimal_creation(self) -> None:
        spec = self._make_spec()
        assert spec.market_id == "abc123"
        assert spec.slug == "will-it-rain"
        assert len(spec.outcome_bins) == 2

    def test_tags_default_empty(self) -> None:
        spec = self._make_spec()
        assert spec.tags == []

    def test_metadata_default_empty(self) -> None:
        spec = self._make_spec()
        assert spec.metadata == {}

    def test_resolve_winning_bin_returns_matching_label(self) -> None:
        bins = [
            OutcomeBin(label="cold", upper=15.0, upper_inclusive=False),
            OutcomeBin(
                label="warm", lower=15.0, upper=25.0, lower_inclusive=True, upper_inclusive=False
            ),
            OutcomeBin(label="hot", lower=25.0, lower_inclusive=True),
        ]
        spec = MarketSpec(
            market_id="t1",
            slug="s",
            question="q",
            outcome_bins=bins,
            close_time="2026-05-10T20:00:00Z",
            market_family="range",
        )
        assert spec.resolve_winning_bin(10.0) == "cold"
        assert spec.resolve_winning_bin(20.0) == "warm"
        assert spec.resolve_winning_bin(30.0) == "hot"

    def test_resolve_winning_bin_returns_none_if_no_match(self) -> None:
        bins = [OutcomeBin(label="hot", lower=50.0)]
        spec = MarketSpec(
            market_id="t2",
            slug="s",
            question="q",
            outcome_bins=bins,
            close_time="2026-05-10T20:00:00Z",
            market_family="range",
        )
        assert spec.resolve_winning_bin(10.0) is None

    def test_resolve_categorical_label(self) -> None:
        """For categorical outcomes, bins have no numeric bounds — match by label."""
        bins = [
            OutcomeBin(label="Verstappen"),
            OutcomeBin(label="Hamilton"),
            OutcomeBin(label="Norris"),
        ]
        spec = MarketSpec(
            market_id="f1_winner",
            slug="f1-winner",
            question="Who wins the race?",
            outcome_bins=bins,
            close_time="2026-05-25T16:00:00Z",
            market_family="categorical",
        )
        # Categorical resolution done externally — just verify spec stores bins
        labels = [b.label for b in spec.outcome_bins]
        assert "Verstappen" in labels

    def test_market_family_stored(self) -> None:
        spec = self._make_spec()
        assert spec.market_family == "binary"

    def test_extra_metadata(self) -> None:
        spec = MarketSpec(
            market_id="x",
            slug="x",
            question="q",
            outcome_bins=[],
            close_time="2026-01-01T00:00:00Z",
            market_family="binary",
            tags=["weather", "temperature"],
            metadata={"city": "NYC", "country": "US"},
        )
        assert "weather" in spec.tags
        assert spec.metadata["city"] == "NYC"


class TestNonFiniteTruth:
    """A NaN observation belongs to no bin (#52); before 1.1.1 it fell into the first one."""

    def test_nan_is_in_no_bin(self) -> None:
        nan = float("nan")
        for b in (
            OutcomeBin(label="lo", upper=27.0),
            OutcomeBin(label="mid", lower=28.0, upper=30.0),
            OutcomeBin(label="hi", lower=31.0),
            OutcomeBin(label="any"),
        ):
            assert b.contains(nan) is False

    def test_nan_resolves_to_no_winner(self) -> None:
        spec = MarketSpec(
            market_id="m",
            slug="s",
            question="q",
            outcome_bins=[OutcomeBin(label="lo", upper=27.0), OutcomeBin(label="hi", lower=31.0)],
            close_time="2026-01-01T00:00:00Z",
            market_family="f",
        )
        assert spec.resolve_winning_bin(float("nan")) is None
        assert spec.resolve_winning_bin(20.0) == "lo"


class TestOutcomeBinValidation:
    """A bin that can never contain anything, or whose bounds are not numbers (#68)."""

    def test_a_lower_bound_above_the_upper_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="lower"):
            OutcomeBin(label="backwards", lower=30.0, upper=20.0)

    @pytest.mark.parametrize("bound", ["lower", "upper"])
    def test_a_nan_bound_is_refused(self, bound: str) -> None:
        # nan compares False with everything, so a nan bound used to accept every value
        with pytest.raises(ValidationError, match=bound):
            OutcomeBin(label="nan", **{bound: float("nan")})

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"lower": 20.0, "upper": 20.0},  # a single point
            {"lower": 20.0, "upper": 30.0},
            {"lower": -math.inf, "upper": 5.0},
            {"lower": 5.0, "upper": math.inf},
            {"lower": None, "upper": 5.0},
            {"lower": 5.0, "upper": None},
            {},
        ],
    )
    def test_every_bin_that_worked_still_does(self, kwargs: dict[str, float | None]) -> None:
        assert OutcomeBin(label="ok", **kwargs).label == "ok"  # type: ignore[arg-type]

    def test_a_point_bin_still_contains_its_point(self) -> None:
        assert OutcomeBin(label="30", lower=30.0, upper=30.0).contains(30.0) is True


class TestMarketSpecValidation:
    def _spec(self, **overrides: Any) -> MarketSpec:
        fields: dict[str, Any] = {
            "market_id": "m",
            "slug": "s",
            "question": "q",
            "outcome_bins": [OutcomeBin(label="YES"), OutcomeBin(label="NO")],
            "close_time": "2026-05-10T20:00:00Z",
            "market_family": "binary",
        }
        fields.update(overrides)
        return MarketSpec(**fields)

    def test_duplicate_bin_labels_are_refused_and_named(self) -> None:
        bins = [OutcomeBin(label="30°C"), OutcomeBin(label="31°C"), OutcomeBin(label="30°C")]
        with pytest.raises(ValidationError, match="duplicate") as err:
            self._spec(outcome_bins=bins)
        assert "30°C" in str(err.value)

    def test_labels_that_differ_only_by_case_are_two_labels(self) -> None:
        assert (
            len(
                self._spec(
                    outcome_bins=[OutcomeBin(label="Yes"), OutcomeBin(label="YES")]
                ).outcome_bins
            )
            == 2
        )

    @pytest.mark.parametrize(
        "close_time",
        [
            "2026-05-10T20:00:00Z",
            "2026-05-10T20:00:00+00:00",
            "2026-05-10T20:00:00",
            "2026-05-10 20:00:00+00",
            "2026-05-10",
            " 2026-05-10T20:00:00Z ",
        ],
    )
    def test_the_forms_gamma_and_the_plugins_use_are_accepted(self, close_time: str) -> None:
        assert self._spec(close_time=close_time).close_time == close_time

    @pytest.mark.parametrize(
        "close_time", ["tomorrow", "2026-13-45", "June 5", "2026-05-10T25:00:00Z", "n/a"]
    )
    def test_an_unparsable_close_time_is_refused(self, close_time: str) -> None:
        # the backtest's "decided after the close" guard reads it, and skips what it cannot read
        with pytest.raises(ValidationError, match="close_time"):
            self._spec(close_time=close_time)

    def test_a_blank_close_time_means_unknown_and_is_accepted(self) -> None:
        # a Gamma payload without `endDate` gives "" (the plugins' own default), and the
        # backtest then has no close to compare a decision with
        assert self._spec(close_time="").close_time == ""
        assert self._spec(close_time="   ").close_time == "   "

    def test_an_empty_bin_list_is_still_accepted(self) -> None:
        """Kept on purpose: `WeatherTmaxPlugin` documents that a sparse Gamma payload is not
        discarded during discovery, and a spec without bins resolves to no winner."""
        spec = self._spec(outcome_bins=[])
        assert spec.outcome_bins == []
        assert spec.resolve_winning_bin(1.0) is None

    def test_a_pydantic_validation_error_is_a_value_error(self) -> None:
        with pytest.raises(ValueError):
            self._spec(close_time="soon")
