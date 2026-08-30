"""Point-in-time feature snapshot store.

Persists snapshots of open-market features as they were *observed at a moment in
time*, so a later retrain uses the features a decision could actually have seen —
not the post-resolution state. This is what lets pmlab honor its no-lookahead
principle during ingestion: capture open markets repeatedly, and when one
resolves, join its truth to the snapshot taken before the decision date.

Storage: one append-only parquet file per market family. Snapshots are keyed by
``(market_id, captured_at)`` and de-duplicated on that key, so re-capturing the
same instant is idempotent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

__all__ = ["FeatureSnapshotStore"]

_MARKET_ID = "market_id"
_CAPTURED_AT = "captured_at"
_KEY = [_MARKET_ID, _CAPTURED_AT]


class FeatureSnapshotStore:
    """Append-only, point-in-time store of per-market feature snapshots."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, family: str) -> Path:
        return self.root / f"{family}.parquet"

    def append(self, family: str, snapshots: list[dict[str, Any]]) -> int:
        """Append point-in-time snapshots for a market family.

        Each snapshot dict must contain ``market_id`` and ``captured_at`` (an
        ISO-8601 timestamp) plus any ``feature_*`` values. Existing
        ``(market_id, captured_at)`` rows are preserved and duplicates dropped,
        so the store is append-only and idempotent.

        Returns:
            The number of rows in the family store after the append.

        Raises:
            ValueError: if a snapshot is missing ``market_id`` or ``captured_at``.
        """
        if not snapshots:
            return len(self.load(family))

        for snap in snapshots:
            if _MARKET_ID not in snap:
                raise ValueError(f"Snapshot missing required '{_MARKET_ID}': {snap!r}")
            if _CAPTURED_AT not in snap:
                raise ValueError(f"Snapshot missing required '{_CAPTURED_AT}': {snap!r}")

        incoming = pd.DataFrame(snapshots)
        incoming[_CAPTURED_AT] = pd.to_datetime(incoming[_CAPTURED_AT], utc=True, format="mixed")

        existing = self.load(family)
        combined = (
            pd.concat([existing, incoming], ignore_index=True) if not existing.empty else incoming
        )
        combined = (
            combined.drop_duplicates(subset=_KEY, keep="first")
            .sort_values(_KEY)
            .reset_index(drop=True)
        )
        combined.to_parquet(self._path(family), index=False)
        return len(combined)

    def load(self, family: str) -> pd.DataFrame:
        """Load all snapshots for a family (empty DataFrame if none)."""
        path = self._path(family)
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def as_of(self, family: str, cutoff: str) -> pd.DataFrame:
        """Return the latest snapshot per market captured strictly before ``cutoff``.

        This is the no-lookahead read: for each market, the most recent snapshot
        whose ``captured_at < cutoff``. Markets whose only snapshots are at or
        after ``cutoff`` are excluded (their features were not yet observable).

        Args:
            family: Market family.
            cutoff: ISO-8601 timestamp; snapshots at or after this are excluded.

        Returns:
            One row per market, or an empty DataFrame if nothing qualifies.
        """
        df = self.load(family)
        if df.empty:
            return df
        cutoff_ts = pd.Timestamp(cutoff)
        if cutoff_ts.tzinfo is None:
            cutoff_ts = cutoff_ts.tz_localize("UTC")
        past = df[df[_CAPTURED_AT] < cutoff_ts]
        if past.empty:
            return past
        latest_idx = past.groupby(_MARKET_ID)[_CAPTURED_AT].idxmax()
        return past.loc[latest_idx].sort_values(_MARKET_ID).reset_index(drop=True)
