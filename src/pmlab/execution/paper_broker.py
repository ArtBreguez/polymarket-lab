"""PaperBroker — record paper trades from EdgeSignals with staleness and dedup checks."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from pmlab.core.costs import CostModel
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
        taker_bps: float | None = None,
        costs: CostModel | None = None,
    ) -> None:
        """Record paper trades from EdgeSignals.

        Args:
            trades_path: JSON file the trade log is appended to.
            allowed_segments: Only record signals in these segments (from the
                champion's passing gate segments). None means no filtering.
            flat_stake: Notional USDC per trade.
            taker_bps: Legacy fee-only knob. Mutually exclusive with ``costs``.
            costs: Full :class:`~pmlab.core.costs.CostModel`. Use the SAME model the
                backtest was run with, otherwise live and backtest PnL are not
                comparable and a drift alarm cannot tell the market from our own
                arithmetic. Omit both for the historical default (30bps, fill at
                the quote).

        Raises:
            ValueError: If both ``taker_bps`` and ``costs`` are supplied.
        """
        if taker_bps is not None and costs is not None:
            raise ValueError(
                "pass either taker_bps or costs, not both — costs.taker_bps already carries the fee"
            )
        self.trades_path = trades_path
        self.allowed_segments = allowed_segments
        self.flat_stake = flat_stake
        self.costs = costs if costs is not None else CostModel(taker_bps=taker_bps or 30.0)
        # Kept so existing callers reading broker.taker_bps keep working.
        self.taker_bps = self.costs.taker_bps

    def record(
        self,
        signals: list[EdgeSignal],
        now_utc: datetime | None = None,
        city_timezones: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
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
            (t["city_or_segment"], t["target_date"], t["horizon"]) for t in existing_trades
        }

        new_trades: list[dict[str, Any]] = []
        for signal in signals:
            # 1. Segment filter
            if (
                self.allowed_segments is not None
                and signal.city_or_segment not in self.allowed_segments
            ):
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

    def load_trades(self) -> list[dict[str, Any]]:
        """Load existing trades from trades_path, or return empty list."""
        if not self.trades_path.exists():
            return []
        data = json.loads(self.trades_path.read_text())
        result: list[dict[str, Any]] = data.get("trades", [])
        return result

    def _is_stale(self, signal: EdgeSignal, now_utc: datetime, city_tz: str) -> bool:
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

    def _build_trade(self, signal: EdgeSignal, now_utc: datetime) -> dict[str, Any]:
        """Build a trade dict from a signal.

        Fills at the costed price, not the quote — the same arithmetic
        ``rolling_origin_eval`` applies, so a trade recorded here is comparable to
        the backtest that promoted the champion.
        """
        quote = signal.gamma_price if signal.direction == "yes" else (1.0 - signal.gamma_price)
        fill = self.costs.fill_price(quote, stake=self.flat_stake)
        size = self.flat_stake / max(fill, 1e-9)
        fee = self.costs.fee(self.flat_stake)
        return {
            "recorded_at": now_utc.isoformat(),
            "city_or_segment": signal.city_or_segment,
            "target_date": signal.target_date,
            "outcome_label": signal.outcome_label,
            "direction": signal.direction,
            "gamma_price": signal.gamma_price,
            "fill_price": round(fill, 8),
            "edge_after_fee": round(signal.best_edge, 6),
            "horizon": signal.horizon,
            "flat_stake": self.flat_stake,
            "size": round(size, 6),
            "fee_paid": round(fee, 6),
            "outcome": None,
            "realized_pnl": None,
        }
