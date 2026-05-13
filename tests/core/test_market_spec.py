"""Tests for generic MarketSpec and OutcomeBin."""

from __future__ import annotations

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
