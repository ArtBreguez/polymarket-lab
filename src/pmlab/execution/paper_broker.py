"""PaperBroker — record paper trades from EdgeSignals with staleness and dedup checks."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from pmlab.core.fees import estimate_fee
from pmlab.execution.edge_signal import EdgeSignal

# Stale cutoffs per horizon: (day_offset, cutoff_hour_local)
# A signal for target_date is stale if now_utc >= cutoff_utc, where:
#   cutoff_local = datetime(target_date + day_offset, cutoff_hour_local, 0, 0, tzinfo=city_tz)
HORIZON_CUTOFFS = {
    "market_open": (-2, 12),
    "previous_evening": (-1, 18),
    "morning_of": (0, 6),
}


class PaperBroker:
    """Records paper trades from EdgeSignals with dedup, staleness, and segment filtering."""

    def __init__(
        self,
        trades_path: Path,
        allowed_segments: set[str] | None = None,
        flat_stake: float = 1.0,
        taker_bps: float = 30.0,
    ) -> None:
        self.trades_path = trades_path
        self.allowed_segments = allowed_segments
        self.flat_stake = flat_stake
        self.taker_bps = taker_bps

    def record(
        self,
        signals: list[EdgeSignal],
        now_utc: datetime | None = None,
        city_timezones: dict[str, str] | None = None,
    ) -> list[dict]:
        """Process signals and append non-stale, non-duplicate trades.

        Args:
            signals: List of EdgeSignal objects to process.
            now_utc: Current time in UTC (injected for testing; defaults to now).
            city_timezones: Mapping of city_or_segment -> tz string (defaults to UTC).

        Returns:
            List of newly added trade dicts.
        """
        if now_utc is None:
            now_utc = datetime.now(UTC)
        if city_timezones is None:
            city_timezones = {}

        existing_trades = self.load_trades()
        # Build set of existing keys for dedup
        existing_keys = {
            (t["city_or_segment"], t["target_date"], t["horizon"])
            for t in existing_trades
        }

        new_trades: list[dict] = []
        for signal in signals:
            # 1. Segment filter
            if self.allowed_segments is not None and signal.city_or_segment not in self.allowed_segments:
                continue

            # 2. Staleness check
            city_tz = city_timezones.get(signal.city_or_segment, "UTC")
            if self._is_stale(signal, now_utc, city_tz):
                continue

            # 3. Dedup check
            key = (signal.city_or_segment, signal.target_date, signal.horizon)
            if key in existing_keys:
                continue

            # 4. Build and append
            trade = self._build_trade(signal, now_utc)
            new_trades.append(trade)
            existing_keys.add(key)

        all_trades = existing_trades + new_trades
        self.trades_path.parent.mkdir(parents=True, exist_ok=True)
        self.trades_path.write_text(json.dumps({"trades": all_trades}, indent=2))
        return new_trades

    def load_trades(self) -> list[dict]:
        """Load existing trades from trades_path, or return empty list."""
        if not self.trades_path.exists():
            return []
        data = json.loads(self.trades_path.read_text())
        return data.get("trades", [])

    def _is_stale(
        self, signal: EdgeSignal, now_utc: datetime, city_tz: str
    ) -> bool:
        """Return True if the signal's horizon cutoff has already passed."""
        day_offset, cutoff_hour = HORIZON_CUTOFFS.get(signal.horizon, (0, 0))

        # Parse target_date
        target = date.fromisoformat(signal.target_date)
        cutoff_date = target + timedelta(days=day_offset)

        tz = ZoneInfo(city_tz)
        cutoff_local = datetime(
            cutoff_date.year,
            cutoff_date.month,
            cutoff_date.day,
            cutoff_hour,
            0,
            0,
            tzinfo=tz,
        )
        cutoff_utc = cutoff_local.astimezone(UTC)
        return now_utc >= cutoff_utc

    def _build_trade(self, signal: EdgeSignal, now_utc: datetime) -> dict:
        """Build a trade dict from a signal."""
        entry_price = signal.gamma_price if signal.direction == "yes" else (1.0 - signal.gamma_price)
        size = self.flat_stake / max(entry_price, 1e-9)
        fee = estimate_fee(self.flat_stake, self.taker_bps)
        return {
            "recorded_at": now_utc.isoformat(),
            "city_or_segment": signal.city_or_segment,
            "target_date": signal.target_date,
            "outcome_label": signal.outcome_label,
            "direction": signal.direction,
            "gamma_price": signal.gamma_price,
            "edge_after_fee": round(signal.best_edge, 6),
            "horizon": signal.horizon,
            "flat_stake": self.flat_stake,
            "size": round(size, 6),
            "fee_paid": round(fee, 6),
            "outcome": None,
            "realized_pnl": None,
        }
