"""Tests for SportsF1Plugin."""

from __future__ import annotations

from unittest.mock import MagicMock

from pmlab.core.market_spec import MarketSpec
from pmlab.plugins.base import MarketPlugin
from pmlab.plugins.sports_f1.plugin import SportsF1Plugin


class TestSportsF1Family:
    def test_family_name(self) -> None:
        assert SportsF1Plugin.family == "sports_f1"

    def test_implements_plugin_abc(self) -> None:
        assert issubclass(SportsF1Plugin, MarketPlugin)


class TestF1DiscoverMarkets:
    def test_requires_gamma_client(self) -> None:
        import pytest

        plugin = SportsF1Plugin()
        with pytest.raises(RuntimeError, match="gamma_client"):
            plugin.discover_markets()

    def test_returns_categorical_market_spec(self) -> None:
        mock_gamma = MagicMock()
        mock_gamma.fetch_markets.return_value = [
            {
                "id": "f1_001",
                "slug": "f1-monaco-winner",
                "question": "F1 Monaco GP winner 2026?",
                "endDate": "2026-05-25T16:00:00Z",
                "tokens": [
                    {"outcome": "Verstappen"},
                    {"outcome": "Norris"},
                    {"outcome": "Hamilton"},
                ],
            }
        ]
        plugin = SportsF1Plugin(gamma_client=mock_gamma)
        markets = plugin.discover_markets()
        assert len(markets) == 1
        spec = markets[0]
        assert spec.market_family == "categorical"
        labels = [b.label for b in spec.outcome_bins]
        assert "Verstappen" in labels
        assert "Norris" in labels

    def test_categorical_bins_have_no_numeric_bounds(self) -> None:
        mock_gamma = MagicMock()
        mock_gamma.fetch_markets.return_value = [
            {
                "id": "f1_002",
                "slug": "f1-test",
                "question": "F1 test GP winner?",
                "endDate": "2026-05-25T16:00:00Z",
                "tokens": [{"outcome": "Verstappen"}, {"outcome": "Norris"}],
            }
        ]
        plugin = SportsF1Plugin(gamma_client=mock_gamma)
        markets = plugin.discover_markets()
        for bin_ in markets[0].outcome_bins:
            assert bin_.lower is None
            assert bin_.upper is None


class TestF1FetchTruth:
    def _spec(self) -> MarketSpec:
        from pmlab.core.market_spec import OutcomeBin

        return MarketSpec(
            market_id="f1_001",
            slug="s",
            question="q",
            outcome_bins=[OutcomeBin(label="Verstappen"), OutcomeBin(label="Norris")],
            close_time="2026-05-25T16:00:00Z",
            market_family="categorical",
            metadata={"gp": "Monaco", "market_type": "race_winner"},
        )

    def test_returns_none_without_results_client(self) -> None:
        plugin = SportsF1Plugin()
        assert plugin.fetch_truth(self._spec()) is None

    def test_returns_winner_string(self) -> None:
        mock_results = MagicMock()
        mock_results.get_winner.return_value = "Verstappen"
        plugin = SportsF1Plugin(results_client=mock_results)
        assert plugin.fetch_truth(self._spec()) == "Verstappen"


class TestF1BuildTrainingRow:
    def _spec(self) -> MarketSpec:
        from pmlab.core.market_spec import OutcomeBin

        return MarketSpec(
            market_id="f1_001",
            slug="s",
            question="q",
            outcome_bins=[OutcomeBin(label="Verstappen"), OutcomeBin(label="Norris")],
            close_time="2026-05-25T16:00:00Z",
            market_family="categorical",
            metadata={"gp": "Monaco", "market_type": "race_winner"},
        )

    def test_returns_none_if_no_truth(self) -> None:
        plugin = SportsF1Plugin()
        assert plugin.build_training_row(self._spec(), "pre_quali") is None

    def test_winning_label_is_string(self) -> None:
        mock_results = MagicMock()
        mock_results.get_winner.return_value = "Norris"
        plugin = SportsF1Plugin(results_client=mock_results)
        row = plugin.build_training_row(self._spec(), "pre_quali")
        assert row is not None
        assert row["winning_label"] == "Norris"
        assert isinstance(row["winning_label"], str)
