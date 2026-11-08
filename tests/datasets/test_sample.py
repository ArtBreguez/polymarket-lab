"""The bundled sample dataset and the offline plugin built on it (issue #15).

Nothing here touches the network: the plugin under test is the real ``WeatherTmaxPlugin``
with the three clients it takes by injection replaced by readers of the bundled file.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import socket
from datetime import datetime, timedelta
from importlib import resources
from typing import Any

import pytest

from pmlab import datasets
from pmlab.data.leakage import check_no_leakage
from pmlab.data.panel_builder import build_panel
from pmlab.datasets import SampleDataError, SampleDataset, load_sample
from pmlab.plugins.weather_tmax.plugin import WeatherTmaxPlugin

DATA_NAME = "weather_tmax_sample.json.gz"
SUM_NAME = "weather_tmax_sample.sha256"
MAX_BYTES = 200_000  # the wheel stays lean: the whole dataset fits in a fifth of a megabyte


@pytest.fixture(scope="module")
def sample() -> SampleDataset:
    return load_sample()


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the offline path opened a socket")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


def _raw(name: str) -> bytes:
    return resources.files("pmlab.datasets").joinpath("data", name).read_bytes()


# --------------------------------------------------------------------------- the file


def test_the_data_ships_inside_the_package_and_is_small() -> None:
    assert len(_raw(DATA_NAME)) < MAX_BYTES
    assert len(_raw(DATA_NAME)) + len(_raw(SUM_NAME)) < MAX_BYTES


def test_the_checksum_file_matches_the_data() -> None:
    assert _raw(SUM_NAME).decode().strip() == hashlib.sha256(_raw(DATA_NAME)).hexdigest()


def test_the_payload_is_valid_json_with_a_schema_version() -> None:
    payload = json.loads(gzip.decompress(_raw(DATA_NAME)))
    assert payload["schema_version"] == 1
    assert set(payload) >= {"schema_version", "meta", "markets"}


# --------------------------------------------------------------------------- schema


def test_meta_says_what_the_data_is(sample: SampleDataset) -> None:
    meta = sample.meta
    assert meta["synthetic"] is True  # never mistaken for real Polymarket history
    assert meta["family"] == "weather_tmax"
    assert meta["unit"] == "C"
    assert meta["horizon"] == "previous_evening"
    assert {"seed", "generator", "description"} <= set(meta)


def test_the_sample_has_enough_markets_to_backtest(sample: SampleDataset) -> None:
    assert sample.n_markets >= 600
    assert len(sample.cities) >= 5
    assert len(sample.days) >= 120
    assert sample.first_target_date < sample.last_target_date


def test_every_market_record_has_the_documented_schema(sample: SampleDataset) -> None:
    ids = set()
    for m in sample.markets:
        assert set(m) == {"id", "city", "target_date", "center", "prices", "truth_c", "features"}
        assert m["id"] not in ids
        ids.add(m["id"])
        assert m["city"] in sample.cities
        assert len(m["prices"]) == 9
        assert all(0.0 < p < 1.0 for p in m["prices"])
        assert 1.0 < sum(m["prices"]) < 1.2  # an overround, as a real book has
        assert isinstance(m["truth_c"], int)
        assert m["features"] and all(k.startswith("feature_") for k in m["features"])
        assert all(isinstance(v, float) for v in m["features"].values())
    assert len({tuple(sorted(m["features"])) for m in sample.markets}) == 1


def test_one_market_per_city_per_day(sample: SampleDataset) -> None:
    keys = [(m["city"], m["target_date"]) for m in sample.markets]
    assert len(keys) == len(set(keys))
    assert sample.n_markets == len(sample.cities) * len(sample.days)


def test_a_sample_with_a_city_missing_a_day_is_refused(sample: SampleDataset) -> None:
    broken = {**sample.payload, "markets": sample.payload["markets"][1:]}
    with pytest.raises(SampleDataError, match="one market per city per day"):
        SampleDataset.from_payload(broken)


# --------------------------------------------------------------------------- loader


def test_loading_is_deterministic(sample: SampleDataset) -> None:
    again = load_sample()
    assert again.markets == sample.markets
    assert again.meta == sample.meta


def test_load_sample_exports_are_public() -> None:
    for name in ("load_sample", "SampleDataset", "SampleDataError", "SampleGammaClient"):
        assert name in datasets.__all__


def test_a_corrupted_file_is_refused_not_half_read(monkeypatch: pytest.MonkeyPatch) -> None:
    good = _raw(DATA_NAME)
    bad = good[:-20] + bytes(b ^ 0xFF for b in good[-20:])
    monkeypatch.setattr(
        "pmlab.datasets.sample._read", lambda name: bad if name == DATA_NAME else _raw(name)
    )
    with pytest.raises(SampleDataError, match="checksum"):
        load_sample()


def test_a_missing_file_is_refused_with_a_clear_message(monkeypatch: pytest.MonkeyPatch) -> None:
    def gone(name: str) -> bytes:
        raise FileNotFoundError(name)

    monkeypatch.setattr("pmlab.datasets.sample._read", gone)
    with pytest.raises(SampleDataError, match="not found"):
        load_sample()


# --------------------------------------------------------------------------- offline discover


def test_discover_works_offline_and_is_the_real_plugin(
    sample: SampleDataset, no_network: None
) -> None:
    day = sample.days[10]
    plugin = sample.plugin(day)
    assert isinstance(plugin, WeatherTmaxPlugin)
    specs = plugin.discover_markets()
    assert len(specs) == len(sample.cities)
    nxt = (datetime.fromisoformat(day) + timedelta(days=1)).date().isoformat()
    assert {s.metadata["city"] for s in specs} == set(sample.cities)
    for s in specs:
        assert s.market_family == "range"
        assert s.metadata["target_date"] == nxt
        assert s.metadata["price_source"] == "discovery"
        assert s.metadata["price_observed_at"] == f"{day}T07:30:00+00:00"
        assert len(s.outcome_bins) == 9
        assert s.metadata["unit"] == "C"


def test_the_real_plugin_is_not_weakened() -> None:
    with pytest.raises(RuntimeError, match="gamma_client"):
        WeatherTmaxPlugin().discover_markets()
    with pytest.raises(RuntimeError, match="gamma_client"):
        load_sample().plugin().discover_markets()  # the resolver has nothing "open" to list


def test_every_truth_falls_in_exactly_one_bin(sample: SampleDataset) -> None:
    resolver = sample.plugin()
    for day in (sample.days[0], sample.days[len(sample.days) // 2], sample.days[-1]):
        for spec in sample.plugin(day).discover_markets():
            truth = resolver.fetch_truth(spec)
            assert truth is not None
            hits = [b.label for b in spec.outcome_bins if b.contains(truth)]
            assert len(hits) == 1, (spec.market_id, truth, hits)
            assert spec.resolve_winning_bin(truth) == hits[0]


def test_truth_is_final_only_after_the_day_has_ended(sample: SampleDataset) -> None:
    day = sample.days[20]
    morning = sample.plugin(day)
    spec = morning.discover_markets()[0]
    assert morning.is_truth_final(spec) is False  # tomorrow has not happened
    assert sample.plugin(sample.days[21]).is_truth_final(spec) is False  # nor has the day itself
    assert sample.plugin(sample.days[22]).is_truth_final(spec) is True
    assert sample.plugin().is_truth_final(spec) is True


def test_the_resolver_knows_yesterday_but_the_morning_plugin_does_not_know_today(
    sample: SampleDataset,
) -> None:
    day = sample.days[30]
    morning = sample.plugin(day)
    today_spec = sample.plugin(sample.days[29]).discover_markets()[0]  # target == day
    assert today_spec.metadata["target_date"] == day
    assert morning.fetch_truth(today_spec) is None
    assert sample.plugin().fetch_truth(today_spec) is not None


def test_features_come_through_the_forecast_client(sample: SampleDataset) -> None:
    spec = sample.plugin(sample.days[3]).discover_markets()[0]
    feats = sample.plugin().fetch_features(spec, "previous_evening")
    assert feats and all(k.startswith("feature_") for k in feats)


# --------------------------------------------------------------------------- the panel


def _decision(day: str) -> str:
    return f"{day}T12:00:00+00:00"


def test_the_whole_sample_builds_a_leak_free_panel(sample: SampleDataset, no_network: None) -> None:
    resolver = sample.plugin()
    rows = []
    for day in sample.days:
        for spec in sample.plugin(day).discover_markets():
            rows.append(
                resolver.build_training_rows(spec, "previous_evening", decision_time=_decision(day))
            )
    panel = build_panel(rows)  # always runs the price-timing guard (LeakageError if late)
    won = (panel["outcome_label"] == panel["winning_label"]).astype(int)
    assert not check_no_leakage(panel.assign(won=won), label_col="won").leaked
    assert len(panel) == sample.n_markets * 9
    assert panel.groupby("market_id")["outcome_label"].count().eq(9).all()
    assert (
        won.groupby(panel["market_id"]).sum().eq(1).all()
    )  # exactly one winning bin per market, never a winners-only panel
    assert panel["decision_date"].min() == sample.days[0]
    assert panel["market_price"].between(0.0, 1.0).all()


def test_a_decision_at_the_discovery_time_uses_no_price(sample: SampleDataset) -> None:
    day = sample.days[5]
    spec = sample.plugin(day).discover_markets()[0]
    rows = sample.plugin().build_training_rows(
        spec, "previous_evening", decision_time=f"{day}T07:30:00+00:00"
    )
    assert rows == []  # the price was seen AT the decision, not before it


def test_the_edge_is_real_but_not_free(sample: SampleDataset) -> None:
    """The market under-reacts to the forecast (so a model can beat it) but is not a giveaway:
    the favourite is not always the winner and the prices carry an overround."""
    favourite_wins = 0
    for m in sample.markets:
        top = max(range(9), key=lambda i: m["prices"][i])
        winner = sample.winning_index(m)
        favourite_wins += top == winner
    rate = favourite_wins / sample.n_markets
    assert 0.15 < rate < 0.45


# --------------------------------------------------------------------------- refusals


def test_a_payload_from_another_schema_version_is_refused(sample: SampleDataset) -> None:
    with pytest.raises(SampleDataError, match="schema_version"):
        SampleDataset.from_payload({**sample.payload, "schema_version": 2})


def test_a_payload_with_no_markets_is_refused(sample: SampleDataset) -> None:
    with pytest.raises(SampleDataError, match="no markets"):
        SampleDataset.from_payload({**sample.payload, "markets": []})


def test_a_market_without_nine_bins_is_refused(sample: SampleDataset) -> None:
    markets = [dict(m) for m in sample.payload["markets"]]
    markets[0]["prices"] = markets[0]["prices"][:-1]
    with pytest.raises(SampleDataError, match="9 bins"):
        SampleDataset.from_payload({**sample.payload, "markets": markets})


def test_the_forecast_client_refuses_a_horizon_the_sample_does_not_have(
    sample: SampleDataset,
) -> None:
    spec = sample.plugin(sample.days[0]).discover_markets()[0]
    with pytest.raises(SampleDataError, match="previous_evening"):
        sample.plugin().fetch_features(spec, "morning_of")


def test_a_market_outside_the_sample_has_no_features_and_no_truth(sample: SampleDataset) -> None:
    spec = sample.plugin(sample.days[0]).discover_markets()[0]
    spec.metadata["target_date"] = "2031-01-01"
    resolver = sample.plugin()
    assert resolver.fetch_truth(spec) is None
    with pytest.raises(SampleDataError, match="no market for"):
        resolver.fetch_features(spec, "previous_evening")


def test_the_winning_index_clamps_to_the_tails_and_labels_match_the_bins(
    sample: SampleDataset,
) -> None:
    m = sample.markets[0]
    labels = sample.labels(m)
    assert labels[0].startswith("<=") and labels[-1].startswith(">=") and len(set(labels)) == 9
    assert sample.winning_index({**m, "truth_c": m["center"] - 20}) == 0
    assert sample.winning_index({**m, "truth_c": m["center"] + 20}) == 8
    assert sample.winning_index({**m, "truth_c": m["center"]}) == 4
    assert "750 markets" in repr(sample) and "synthetic" in repr(sample)


def test_the_gamma_client_serves_one_morning_and_honours_max_results(sample: SampleDataset) -> None:
    client = sample.plugin(sample.days[0])._gamma  # the injected client, as the plugin holds it
    assert len(client.fetch_markets_all()) == 5
    assert len(client.fetch_markets_all(max_results=2)) == 2
