"""TDD (RED first): point-in-time feature snapshot store.

The real-data dogfood proved pmlab cannot honor its own no-lookahead principle
during ingestion: open markets have live features but no label; resolved markets
have the label but their features are all post-fact/zeroed. The fix is to
persist point-in-time snapshots of open-market features over time, so that when
a market later resolves the training row uses the features *as seen at decision
time*, not the post-resolution corpse.
"""

from __future__ import annotations

import pytest

from pmlab.data import FeatureSnapshotStore


@pytest.fixture
def store(tmp_path):
    return FeatureSnapshotStore(tmp_path / "snapshots")


def _snap(market_id: str, captured_at: str, yes_price: float) -> dict:
    return {
        "market_id": market_id,
        "captured_at": captured_at,
        "feature_yes_price": yes_price,
        "feature_spread": 0.02,
    }


class TestSnapshotRoundtrip:
    def test_append_and_load_single(self, store):
        store.append("politics", [_snap("m1", "2025-01-01T00:00:00Z", 0.4)])
        df = store.load("politics")
        assert len(df) == 1
        assert df.iloc[0]["market_id"] == "m1"
        assert df.iloc[0]["feature_yes_price"] == 0.4

    def test_append_is_additive_across_calls(self, store):
        store.append("politics", [_snap("m1", "2025-01-01T00:00:00Z", 0.4)])
        store.append("politics", [_snap("m1", "2025-01-02T00:00:00Z", 0.5)])
        df = store.load("politics")
        assert len(df) == 2  # two point-in-time snapshots of the same market

    def test_families_are_isolated(self, store):
        store.append("politics", [_snap("m1", "2025-01-01T00:00:00Z", 0.4)])
        store.append("sports", [_snap("g1", "2025-01-01T00:00:00Z", 0.6)])
        assert len(store.load("politics")) == 1
        assert len(store.load("sports")) == 1

    def test_load_missing_family_returns_empty(self, store):
        df = store.load("nonexistent")
        assert df.empty

    def test_append_empty_list_is_noop(self, store):
        store.append("politics", [_snap("m1", "2025-01-01T00:00:00Z", 0.4)])
        n = store.append("politics", [])  # empty append returns current count
        assert n == 1

    def test_as_of_empty_family_returns_empty(self, store):
        got = store.as_of("nonexistent", "2025-01-06T00:00:00Z")
        assert got.empty

    def test_as_of_accepts_tz_naive_cutoff(self, store):
        store.append("politics", [_snap("m1", "2025-01-01T00:00:00Z", 0.4)])
        got = store.as_of("politics", "2025-01-06")  # no timezone → assumed UTC
        assert len(got) == 1


class TestSnapshotImmutability:
    def test_duplicate_snapshot_is_deduped(self, store):
        # same (market_id, captured_at) must not double-count — snapshots are
        # point-in-time facts, re-capturing the same instant is idempotent.
        store.append("politics", [_snap("m1", "2025-01-01T00:00:00Z", 0.4)])
        store.append("politics", [_snap("m1", "2025-01-01T00:00:00Z", 0.4)])
        assert len(store.load("politics")) == 1

    def test_captured_at_required(self, store):
        with pytest.raises((KeyError, ValueError)):
            store.append("politics", [{"market_id": "m1", "feature_x": 1.0}])

    def test_market_id_required(self, store):
        with pytest.raises((KeyError, ValueError)):
            store.append("politics", [{"captured_at": "2025-01-01T00:00:00Z", "feature_x": 1.0}])


class TestAsOf:
    def test_as_of_returns_latest_snapshot_before_cutoff(self, store):
        store.append(
            "politics",
            [
                _snap("m1", "2025-01-01T00:00:00Z", 0.30),
                _snap("m1", "2025-01-05T00:00:00Z", 0.55),
                _snap("m1", "2025-01-10T00:00:00Z", 0.80),
            ],
        )
        # as of Jan 6, the latest known snapshot is Jan 5 (0.55) — never Jan 10.
        got = store.as_of("politics", "2025-01-06T00:00:00Z")
        assert len(got) == 1
        assert got.iloc[0]["feature_yes_price"] == 0.55

    def test_as_of_excludes_future_snapshots(self, store):
        store.append("politics", [_snap("m1", "2025-01-10T00:00:00Z", 0.80)])
        got = store.as_of("politics", "2025-01-05T00:00:00Z")
        assert got.empty  # the only snapshot is in the future → no leakage

    def test_as_of_one_row_per_market(self, store):
        store.append(
            "politics",
            [
                _snap("m1", "2025-01-01T00:00:00Z", 0.30),
                _snap("m1", "2025-01-05T00:00:00Z", 0.55),
                _snap("m2", "2025-01-03T00:00:00Z", 0.20),
            ],
        )
        got = store.as_of("politics", "2025-01-06T00:00:00Z")
        assert set(got["market_id"]) == {"m1", "m2"}
        assert len(got) == 2
