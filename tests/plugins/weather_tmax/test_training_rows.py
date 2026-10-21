"""``WeatherTmaxPlugin.build_training_rows``: one row per bin, each at its own point-in-time price."""

from __future__ import annotations

import logging
import math
import random
from collections import Counter
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from pmlab.core.market_spec import MarketSpec, OutcomeBin
from pmlab.plugins.weather_tmax.plugin import WeatherTmaxPlugin

DECISION = "2026-05-10T08:00:00Z"
BEFORE = "2026-05-10T07:30:00+00:00"
AFTER = "2026-05-10T09:00:00+00:00"
LABELS = ("cold", "warm", "hot")


def _spec(
    prices: tuple[Any, ...] = (0.2, 0.5, 0.3),
    observed: str | None = BEFORE,
    source: str | None = "discovery",
    schema_extra: dict[int, dict[str, Any]] | None = None,
) -> MarketSpec:
    schema = [{"label": lbl, "price": p} for lbl, p in zip(LABELS, prices, strict=True)]
    for i, extra in (schema_extra or {}).items():
        schema[i].update(extra)
    metadata: dict[str, Any] = {
        "city": "NYC",
        "target_date": "2026-05-11",
        "outcome_schema": schema,
    }
    if observed is not None:
        metadata["price_observed_at"] = observed
    if source is not None:
        metadata["price_source"] = source
    return MarketSpec(
        market_id="t1",
        slug="s",
        question="q",
        outcome_bins=[
            OutcomeBin(label="cold", upper=20.0, upper_inclusive=False),
            OutcomeBin(label="warm", lower=20.0, upper=30.0, upper_inclusive=False),
            OutcomeBin(label="hot", lower=30.0),
        ],
        close_time="2026-05-11T20:00:00Z",
        market_family="range",
        metadata=metadata,
    )


def _plugin(truth: float | None = 32.0, features: dict[str, float] | None = None) -> Any:
    truth_client = MagicMock()
    truth_client.get_daily_max.return_value = truth
    forecast = MagicMock()
    forecast.get_features.return_value = features or {"feature_fcst": 31.0, "feature_lead": 1.0}
    return WeatherTmaxPlugin(truth_client=truth_client, forecast_client=forecast)


def _rows(
    spec: MarketSpec | None = None, plugin: Any = None, **kwargs: Any
) -> list[dict[str, Any]]:
    kwargs.setdefault("decision_time", DECISION)
    return (plugin or _plugin()).build_training_rows(spec or _spec(), "morning_of", **kwargs)


class TestOneRowPerBin:
    def test_every_bin_gets_a_row_at_its_own_price(self) -> None:
        rows = _rows()
        assert [r["outcome_label"] for r in rows] == list(LABELS)
        assert [r["market_price"] for r in rows] == [0.2, 0.5, 0.3]

    def test_the_market_maximum_is_not_every_rows_price(self) -> None:
        # The bug: every row carried max(prices) = 0.5, the favourite's price.
        assert len({r["market_price"] for r in _rows()}) == 3

    def test_winning_label_is_shared_and_exactly_one_row_is_the_winner(self) -> None:
        rows = _rows()
        assert {r["winning_label"] for r in rows} == {"hot"}
        assert [r["outcome_label"] for r in rows if r["outcome_label"] == r["winning_label"]] == [
            "hot"
        ]

    def test_rows_carry_the_market_decision_and_features(self) -> None:
        for row in _rows():
            assert row["market_id"] == "t1"
            assert row["decision_horizon"] == "morning_of"
            assert row["feature_fcst"] == 31.0
            assert row["feature_lead"] == 1.0

    def test_a_numeric_string_price_is_accepted(self) -> None:
        rows = _rows(_spec(prices=("0.25", 0.5, 0.25)))
        assert [r["market_price"] for r in rows] == [0.25, 0.5, 0.25]


class TestPriceProvenance:
    def test_rows_record_where_and_when_the_price_was_observed(self) -> None:
        for row in _rows():
            assert row["price_source"] == "discovery"
            assert row["price_ts"] == BEFORE

    def test_the_decision_is_recorded_as_utc_date_and_timestamp(self) -> None:
        for row in _rows():
            assert row["decision_date"] == "2026-05-10"
            assert row["decision_ts"] == "2026-05-10T08:00:00+00:00"

    def test_a_non_utc_decision_time_is_converted_before_its_date_is_taken(self) -> None:
        rows = _rows(decision_time="2026-05-10T23:30:00-05:00")  # 04:30Z the next day
        assert {r["decision_date"] for r in rows} == {"2026-05-11"}
        assert {r["decision_ts"] for r in rows} == {"2026-05-11T04:30:00+00:00"}

    def test_a_naive_datetime_is_taken_as_utc(self) -> None:
        rows = _rows(decision_time=datetime(2026, 5, 10, 8, 0))
        assert {r["decision_ts"] for r in rows} == {"2026-05-10T08:00:00+00:00"}

    def test_a_bin_level_quote_overrides_the_market_level_one(self) -> None:
        spec = _spec(
            schema_extra={1: {"observed_at": "2026-05-10T06:00:00Z", "source": "snapshot"}}
        )
        rows = {r["outcome_label"]: r for r in _rows(spec)}
        assert (rows["warm"]["price_source"], rows["warm"]["price_ts"]) == (
            "snapshot",
            "2026-05-10T06:00:00+00:00",
        )
        assert (rows["cold"]["price_source"], rows["cold"]["price_ts"]) == ("discovery", BEFORE)

    def test_a_spec_that_names_no_source_is_marked_unknown(self) -> None:
        assert {r["price_source"] for r in _rows(_spec(source=None))} == {"unknown"}


class TestDecisionTime:
    def test_it_is_required(self) -> None:
        with pytest.raises(ValueError, match="decision_time"):
            _plugin().build_training_rows(_spec(), "morning_of")

    def test_an_unparseable_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="decision_time"):
            _rows(decision_time="next tuesday-ish")


class TestExclusionsAreCounted:
    def test_a_missing_price_is_excluded_not_imputed(self) -> None:
        counts: Counter[str] = Counter()
        rows = _rows(_spec(prices=(0.2, None, 0.3)), exclusions=counts)
        assert [r["outcome_label"] for r in rows] == ["cold", "hot"]
        assert 0.5 not in [r["market_price"] for r in rows]
        assert counts == {"price_missing": 1}

    def test_a_bin_with_no_price_key_is_missing_too(self) -> None:
        spec = _spec()
        del spec.metadata["outcome_schema"][0]["price"]
        counts: Counter[str] = Counter()
        assert len(_rows(spec, exclusions=counts)) == 2
        assert counts == {"price_missing": 1}

    @pytest.mark.parametrize("bad", ["abc", float("nan"), float("inf"), -0.1, 1.5, True])
    def test_a_price_that_is_not_a_probability_is_invalid(self, bad: Any) -> None:
        counts: Counter[str] = Counter()
        rows = _rows(_spec(prices=(bad, 0.5, 0.3)), exclusions=counts)
        assert [r["outcome_label"] for r in rows] == ["warm", "hot"]
        assert counts == {"price_invalid": 1}

    @pytest.mark.parametrize("settled", [0.0, 1.0])
    def test_a_price_of_exactly_zero_or_one_is_a_settlement_not_a_quote(
        self, settled: float
    ) -> None:
        counts: Counter[str] = Counter()
        rows = _rows(_spec(prices=(settled, 0.5, 0.3)), exclusions=counts)
        assert [r["outcome_label"] for r in rows] == ["warm", "hot"]
        assert counts == {"price_degenerate": 1}

    def test_a_price_observed_after_the_decision_is_excluded(self) -> None:
        spec = _spec(schema_extra={0: {"observed_at": AFTER}})
        counts: Counter[str] = Counter()
        rows = _rows(spec, exclusions=counts)
        assert [r["outcome_label"] for r in rows] == ["warm", "hot"]
        assert counts == {"price_after_cutoff": 1}

    def test_a_price_observed_at_the_decision_instant_is_excluded(self) -> None:
        # Strictly before, as FeatureSnapshotStore.as_of reads: `captured_at < cutoff`.
        spec = _spec(schema_extra={0: {"observed_at": "2026-05-10T08:00:00Z"}})
        counts: Counter[str] = Counter()
        assert len(_rows(spec, exclusions=counts)) == 2
        assert counts == {"price_after_cutoff": 1}

    def test_a_price_observed_just_before_the_decision_is_kept(self) -> None:
        spec = _spec(observed="2026-05-10T07:59:59.999+00:00")
        counts: Counter[str] = Counter()
        assert len(_rows(spec, exclusions=counts)) == 3
        assert counts == {}

    def test_every_bin_of_a_market_observed_late_leaves_no_rows(self) -> None:
        counts: Counter[str] = Counter()
        assert _rows(_spec(observed=AFTER), exclusions=counts) == []
        assert counts == {"price_after_cutoff": 3, "market_no_usable_price": 1}

    def test_a_price_with_no_observation_time_cannot_be_shown_to_be_in_time(self) -> None:
        counts: Counter[str] = Counter()
        assert _rows(_spec(observed=None), exclusions=counts) == []
        assert counts == {"price_untimed": 3, "market_no_usable_price": 1}

    def test_an_unparseable_observation_time_is_untimed(self) -> None:
        counts: Counter[str] = Counter()
        assert _rows(_spec(observed="yesterday"), exclusions=counts) == []
        assert counts["price_untimed"] == 3

    def test_an_unresolved_market_is_counted_not_silently_dropped(self) -> None:
        counts: Counter[str] = Counter()
        assert _rows(plugin=_plugin(truth=None), exclusions=counts) == []
        assert counts == {"market_unresolved": 1}

    def test_a_truth_outside_every_bin_is_counted(self) -> None:
        spec = _spec()
        spec.outcome_bins.pop()  # no bin takes 32
        counts: Counter[str] = Counter()
        assert _rows(spec, exclusions=counts) == []
        assert counts == {"market_truth_unbinned": 1}

    def test_the_counter_accumulates_across_markets(self) -> None:
        counts: Counter[str] = Counter()
        _rows(_spec(prices=(None, 0.5, 0.3)), exclusions=counts)
        _rows(_spec(prices=(None, 0.5, 0.3)), exclusions=counts)
        assert counts == {"price_missing": 2}

    def test_without_a_counter_the_exclusions_are_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            _rows(_spec(prices=(None, 0.5, 0.3)))
        assert "price_missing" in caplog.text
        assert "t1" in caplog.text

    def test_a_clean_market_logs_nothing(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            _rows()
        assert caplog.text == ""


class TestNothingUnusableEverReachesARow:
    def test_over_messy_prices_every_emitted_row_is_a_timed_open_interval_quote(self) -> None:
        rng = random.Random(0)
        pool: list[Any] = [None, "x", float("nan"), -1, 2, 0, 1, 0.0, 1.0, 0.1, 0.4, 0.9, "0.3"]
        times = [None, BEFORE, AFTER, "2026-05-10T08:00:00Z", "junk"]
        for _ in range(300):
            spec = _spec(
                prices=tuple(rng.choice(pool) for _ in LABELS),
                observed=rng.choice(times),
                schema_extra={i: {"observed_at": rng.choice(times)} for i in range(3)},
            )
            for row in _rows(spec, exclusions=Counter()):
                assert 0.0 < row["market_price"] < 1.0 and math.isfinite(row["market_price"])
                assert row["price_ts"] < row["decision_ts"]


class TestLegacySingularRow:
    def test_it_is_deprecated(self) -> None:
        with pytest.warns(DeprecationWarning, match="build_training_rows"):
            _plugin().build_training_row(_spec(), "morning_of")

    def test_it_returns_the_winners_row_at_the_winners_own_price(self) -> None:
        with pytest.warns(DeprecationWarning):
            row = _plugin().build_training_row(_spec(), "morning_of")
        assert row is not None
        assert row["outcome_label"] == row["winning_label"] == "hot"
        assert row["market_price"] == 0.3  # not max(prices) == 0.5

    def test_it_needs_no_decision_time(self) -> None:
        with pytest.warns(DeprecationWarning):
            row = _plugin().build_training_row(_spec(), "morning_of")
        assert row is not None
        assert "decision_ts" not in row and "decision_date" not in row

    def test_it_returns_none_when_the_winners_price_is_unusable(self) -> None:
        with pytest.warns(DeprecationWarning):
            row = _plugin().build_training_row(_spec(prices=(0.2, 0.5, None)), "morning_of")
        assert row is None

    def test_it_returns_none_without_truth(self) -> None:
        with pytest.warns(DeprecationWarning):
            assert _plugin(truth=None).build_training_row(_spec(), "morning_of") is None


class TestDiscoveryStampsThePriceTime:
    @staticmethod
    def _raw(**over: Any) -> dict[str, Any]:
        raw = {
            "id": "g1",
            "question": "Highest temperature in Paris on June 5?",
            "endDate": "2026-06-05T23:00:00Z",
            "outcomes": '["<20°C", "20-22°C", ">22°C"]',
            "outcomePrices": '["0.2", "0.5", "0.3"]',
        }
        raw.update(over)
        return raw

    def _discover(self, raw: dict[str, Any], clock: Any = None) -> MarketSpec:
        gamma = MagicMock()
        gamma.fetch_markets_all.return_value = [raw]
        return WeatherTmaxPlugin(gamma_client=gamma, clock=clock).discover_markets()[0]

    def test_the_clock_gives_the_observation_time_and_the_source_is_discovery(self) -> None:
        now = datetime(2026, 6, 4, 7, 30, tzinfo=UTC)
        spec = self._discover(self._raw(), clock=lambda: now)
        assert spec.metadata["price_observed_at"] == "2026-06-04T07:30:00+00:00"
        assert spec.metadata["price_source"] == "discovery"

    def test_the_default_clock_is_real_utc_time(self) -> None:
        before = datetime.now(UTC)
        spec = self._discover(self._raw())
        stamped = datetime.fromisoformat(spec.metadata["price_observed_at"])
        assert before <= stamped <= datetime.now(UTC)

    def test_a_naive_clock_is_taken_as_utc(self) -> None:
        spec = self._discover(self._raw(), clock=lambda: datetime(2026, 6, 4, 7, 30))
        assert spec.metadata["price_observed_at"] == "2026-06-04T07:30:00+00:00"

    def test_each_bin_keeps_its_own_price(self) -> None:
        spec = self._discover(self._raw())
        assert [(b["label"], b["price"]) for b in spec.metadata["outcome_schema"]] == [
            ("<20°C", 0.2),
            ("20-22°C", 0.5),
            (">22°C", 0.3),
        ]


class TestSpecNeverFabricatesAPrice:
    def _schema(self, **over: Any) -> list[dict[str, Any]]:
        spec = WeatherTmaxPlugin()._build_spec(TestDiscoveryStampsThePriceTime._raw(**over))
        return list(spec.metadata["outcome_schema"])

    def test_unparseable_prices_are_unknown_not_one_half(self) -> None:
        assert [b["price"] for b in self._schema(outcomePrices="not json")] == [None] * 3

    def test_non_numeric_prices_are_unknown_not_one_half(self) -> None:
        assert [b["price"] for b in self._schema(outcomePrices='["a", "b", "c"]')] == [None] * 3

    def test_absent_prices_are_unknown_not_one_half(self) -> None:
        raw = TestDiscoveryStampsThePriceTime._raw()
        del raw["outcomePrices"]
        spec = WeatherTmaxPlugin()._build_spec(raw)
        assert [b["price"] for b in spec.metadata["outcome_schema"]] == [None] * 3

    def test_a_short_price_list_leaves_the_rest_unknown(self) -> None:
        assert [b["price"] for b in self._schema(outcomePrices='["0.2", "0.5"]')] == [
            0.2,
            0.5,
            None,
        ]

    def test_legacy_tokens_without_a_price_are_unknown(self) -> None:
        raw = {
            "id": "g2",
            "question": "Highest temperature in Paris on June 5?",
            "tokens": [{"outcome": "<20°C"}, {"outcome": ">20°C", "price": 0.4}, "plain"],
        }
        spec = WeatherTmaxPlugin()._build_spec(raw)
        assert [b["price"] for b in spec.metadata["outcome_schema"]] == [None, 0.4, None]

    def test_one_bad_entry_does_not_discard_the_others(self) -> None:
        assert [b["price"] for b in self._schema(outcomePrices='["0.2", "x", "0.3"]')] == [
            0.2,
            None,
            0.3,
        ]

    def test_the_display_maximum_ignores_unknown_prices_and_is_none_when_all_are(self) -> None:
        spec = WeatherTmaxPlugin()._build_spec(
            TestDiscoveryStampsThePriceTime._raw(outcomePrices='["0.2", "x", "0.3"]')
        )
        assert spec.metadata["market_price"] == 0.3
        spec = WeatherTmaxPlugin()._build_spec(
            TestDiscoveryStampsThePriceTime._raw(outcomePrices="not json")
        )
        assert spec.metadata["market_price"] is None
