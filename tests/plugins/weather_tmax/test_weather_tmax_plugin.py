"""Tests for WeatherTmaxPlugin."""

from __future__ import annotations

from unittest.mock import MagicMock

from pmlab.core.market_spec import MarketSpec, OutcomeBin
from pmlab.plugins.weather_tmax._spec_builder import _parse_bounds, build_tmax_spec
from pmlab.plugins.weather_tmax.plugin import WeatherTmaxPlugin


class TestWeatherTmaxFamily:
    def test_family_name(self) -> None:
        assert WeatherTmaxPlugin.family == "weather_tmax"

    def test_implements_plugin_abc(self) -> None:
        from pmlab.plugins.base import MarketPlugin
        assert issubclass(WeatherTmaxPlugin, MarketPlugin)


class TestDiscoverMarkets:
    def test_requires_gamma_client(self) -> None:
        import pytest
        plugin = WeatherTmaxPlugin()
        with pytest.raises(RuntimeError, match="gamma_client"):
            plugin.discover_markets()

    def test_filters_non_tmax_markets(self) -> None:
        mock_gamma = MagicMock()
        mock_gamma.fetch_markets.return_value = [
            {"id": "a", "question": "Highest temperature in NYC on May 10?",
             "slug": "tmax-nyc", "endDate": "2026-05-10T20:00:00Z", "tokens": []},
            {"id": "b", "question": "Will it rain in London?",
             "slug": "rain-london", "endDate": "2026-05-10T20:00:00Z", "tokens": []},
        ]
        plugin = WeatherTmaxPlugin(gamma_client=mock_gamma)
        markets = plugin.discover_markets()
        assert len(markets) == 1
        assert markets[0].market_id == "a"

    def test_returns_market_spec_objects(self) -> None:
        mock_gamma = MagicMock()
        mock_gamma.fetch_markets.return_value = [
            {"id": "t1", "question": "Highest temperature in Paris on May 12?",
             "slug": "tmax-paris", "endDate": "2026-05-12T20:00:00Z",
             "tokens": [{"outcome": "30°C"}, {"outcome": "31°C"}]},
        ]
        plugin = WeatherTmaxPlugin(gamma_client=mock_gamma)
        markets = plugin.discover_markets()
        assert all(isinstance(m, MarketSpec) for m in markets)
        assert markets[0].market_family == "range"
        assert "weather" in markets[0].tags


class TestFetchFeatures:
    def _spec(self) -> MarketSpec:
        return MarketSpec(
            market_id="t1", slug="s", question="q",
            outcome_bins=[OutcomeBin(label="30°C")],
            close_time="2026-05-10T20:00:00Z", market_family="range",
            metadata={"city": "NYC", "target_date": "2026-05-10"},
        )

    def test_returns_float_dict(self) -> None:
        plugin = WeatherTmaxPlugin()
        features = plugin.fetch_features(self._spec(), "previous_evening")
        assert isinstance(features, dict)
        assert all(isinstance(v, float) for v in features.values())

    def test_delegates_to_forecast_client(self) -> None:
        mock_fc = MagicMock()
        mock_fc.get_features.return_value = {"lead_time_days": 1.0, "tmax_forecast": 29.5}
        plugin = WeatherTmaxPlugin(forecast_client=mock_fc)
        features = plugin.fetch_features(self._spec(), "previous_evening")
        assert features["tmax_forecast"] == 29.5
        mock_fc.get_features.assert_called_once_with(
            city="NYC", target_date="2026-05-10", horizon="previous_evening"
        )


class TestFetchTruth:
    def _spec(self) -> MarketSpec:
        return MarketSpec(
            market_id="t1", slug="s", question="q",
            outcome_bins=[OutcomeBin(label="30°C", lower=29.5, upper=30.5)],
            close_time="2026-05-10T20:00:00Z", market_family="range",
            metadata={"city": "NYC", "target_date": "2026-05-10"},
        )

    def test_returns_none_without_truth_client(self) -> None:
        plugin = WeatherTmaxPlugin()
        assert plugin.fetch_truth(self._spec()) is None

    def test_returns_float_from_truth_client(self) -> None:
        mock_truth = MagicMock()
        mock_truth.get_daily_max.return_value = 31.2
        plugin = WeatherTmaxPlugin(truth_client=mock_truth)
        result = plugin.fetch_truth(self._spec())
        assert result == 31.2

    def test_is_truth_final_false_when_unresolved(self) -> None:
        plugin = WeatherTmaxPlugin()
        assert plugin.is_truth_final(self._spec()) is False

    def test_is_truth_final_true_when_resolved(self) -> None:
        mock_truth = MagicMock()
        mock_truth.get_daily_max.return_value = 31.2
        plugin = WeatherTmaxPlugin(truth_client=mock_truth)
        assert plugin.is_truth_final(self._spec()) is True


class TestBuildTrainingRow:
    def _spec(self) -> MarketSpec:
        return MarketSpec(
            market_id="t1", slug="s", question="q",
            outcome_bins=[
                OutcomeBin(label="cold", upper=20.0, upper_inclusive=False),
                OutcomeBin(label="warm", lower=20.0, upper=30.0, lower_inclusive=True, upper_inclusive=False),
                OutcomeBin(label="hot", lower=30.0),
            ],
            close_time="2026-05-10T20:00:00Z", market_family="range",
            metadata={"city": "NYC", "target_date": "2026-05-10", "market_price": 0.3},
        )

    def test_returns_none_if_no_truth(self) -> None:
        plugin = WeatherTmaxPlugin()
        assert plugin.build_training_row(self._spec(), "previous_evening") is None

    def test_returns_row_with_winning_label(self) -> None:
        mock_truth = MagicMock()
        mock_truth.get_daily_max.return_value = 32.0  # falls in "hot" bin
        plugin = WeatherTmaxPlugin(truth_client=mock_truth)
        row = plugin.build_training_row(self._spec(), "previous_evening")
        assert row is not None
        assert row["winning_label"] == "hot"
        assert row["market_id"] == "t1"
        assert "decision_horizon" in row


class TestSpecBuilder:
    def test_parses_city_from_question(self) -> None:
        raw = {
            "id": "abc", "slug": "tmax-seoul",
            "question": "Highest temperature in Seoul on May 11?",
            "endDate": "2026-05-11T20:00:00Z",
            "tokens": [],
        }
        spec = build_tmax_spec(raw)
        assert spec.metadata["city"] == "Seoul"

    def test_parse_bounds_range(self) -> None:
        lower, upper = _parse_bounds("28-30°C")
        assert lower == 28.0
        assert upper == 30.0

    def test_parse_bounds_single(self) -> None:
        lower, upper = _parse_bounds("30°C")
        assert lower == 29.5
        assert upper == 30.5

    def test_parse_bounds_gt(self) -> None:
        lower, upper = _parse_bounds(">35°C")
        assert lower == 35.0
        assert upper is None

    def test_parse_bounds_lt(self) -> None:
        lower, upper = _parse_bounds("<20°C")
        assert lower is None
        assert upper == 20.0

    def test_parse_bounds_unknown_returns_none_none(self) -> None:
        lower, upper = _parse_bounds("Unknown bin")
        assert lower is None
        assert upper is None
