"""TDD (RED first): assemble a no-lookahead training panel by joining
point-in-time snapshots with realized truth.

This is the piece that closes the loop: capture open-market features over time
(FeatureSnapshotStore), and when a market resolves, build a training row from
the snapshot taken strictly BEFORE the decision date joined to the realized
outcome. No feature can come from after the decision — no lookahead by
construction.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pmlab.data import FeatureSnapshotStore, build_panel_from_snapshots


@pytest.fixture
def store(tmp_path):
    s = FeatureSnapshotStore(tmp_path / "snaps")
    # two markets, each captured on several days
    s.append(
        "politics",
        [
            {"market_id": "m1", "captured_at": "2025-01-01T00:00:00Z", "feature_p": 0.30},
            {"market_id": "m1", "captured_at": "2025-01-05T00:00:00Z", "feature_p": 0.55},
            {"market_id": "m1", "captured_at": "2025-01-20T00:00:00Z", "feature_p": 0.95},
            {"market_id": "m2", "captured_at": "2025-01-02T00:00:00Z", "feature_p": 0.10},
            {"market_id": "m2", "captured_at": "2025-01-06T00:00:00Z", "feature_p": 0.08},
        ],
    )
    return s


def _resolutions() -> pd.DataFrame:
    # market truth known only after each market's decision date
    return pd.DataFrame(
        [
            {"market_id": "m1", "decision_date": "2025-01-10", "winning_label": "Yes"},
            {"market_id": "m2", "decision_date": "2025-01-10", "winning_label": "No"},
        ]
    )


class TestBuildFromSnapshots:
    def test_uses_snapshot_before_decision_date(self, store):
        panel = build_panel_from_snapshots(store, "politics", _resolutions())
        m1 = panel[panel["market_id"] == "m1"].iloc[0]
        # decision 2025-01-10 → latest snapshot before it is 2025-01-05 (0.55),
        # NEVER the 2025-01-20 snapshot (0.95) which is post-decision.
        assert m1["feature_p"] == 0.55

    def test_attaches_truth(self, store):
        panel = build_panel_from_snapshots(store, "politics", _resolutions())
        assert set(panel["winning_label"]) == {"Yes", "No"}
        assert panel[panel["market_id"] == "m2"].iloc[0]["winning_label"] == "No"

    def test_one_row_per_resolved_market(self, store):
        panel = build_panel_from_snapshots(store, "politics", _resolutions())
        assert len(panel) == 2

    def test_output_is_valid_training_panel(self, store):
        panel = build_panel_from_snapshots(store, "politics", _resolutions())
        for col in ["market_id", "decision_date", "winning_label", "market_price"]:
            assert col in panel.columns
        assert any(c.startswith("feature_") for c in panel.columns)

    def test_market_price_defaults_to_yes_feature_when_present(self, store):
        # market_price should be populated so the panel is backtestable; when a
        # feature_yes-style price exists it is used, else it must still be numeric.
        panel = build_panel_from_snapshots(store, "politics", _resolutions(), price_col="feature_p")
        assert panel[panel["market_id"] == "m1"].iloc[0]["market_price"] == 0.55


class TestBuildFromSnapshotsEdgeCases:
    def test_market_with_no_prior_snapshot_is_dropped(self, store):
        # a market whose only snapshots are AFTER the decision contributes nothing
        res = pd.DataFrame(
            [{"market_id": "m1", "decision_date": "2024-12-01", "winning_label": "Yes"}]
        )
        panel = build_panel_from_snapshots(store, "politics", res)
        assert panel.empty

    def test_unknown_market_is_ignored(self, store):
        res = pd.DataFrame(
            [{"market_id": "ghost", "decision_date": "2025-06-01", "winning_label": "Yes"}]
        )
        panel = build_panel_from_snapshots(store, "politics", res)
        assert panel.empty

    def test_empty_resolutions_returns_empty(self, store):
        panel = build_panel_from_snapshots(
            store, "politics", pd.DataFrame(columns=["market_id", "decision_date", "winning_label"])
        )
        assert panel.empty

    def test_missing_resolution_columns_raises(self, store):
        bad = pd.DataFrame([{"market_id": "m1"}])  # no decision_date / winning_label
        with pytest.raises(ValueError):
            build_panel_from_snapshots(store, "politics", bad)

    def test_snapshot_without_feature_columns_is_dropped(self, tmp_path):
        # a store whose snapshots carry no feature_* columns yields nothing
        s = FeatureSnapshotStore(tmp_path / "nofeat")
        s.append("politics", [{"market_id": "m1", "captured_at": "2025-01-01T00:00:00Z"}])
        res = pd.DataFrame(
            [{"market_id": "m1", "decision_date": "2025-01-10", "winning_label": "Yes"}]
        )
        panel = build_panel_from_snapshots(s, "politics", res)
        assert panel.empty
