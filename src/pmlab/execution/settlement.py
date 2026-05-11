"""SettlementEngine — settles paper trades against plugin truth values."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from pmlab.core.market_spec import MarketSpec
from pmlab.core.pnl import Position, settle_position
from pmlab.plugins.base import MarketPlugin


class SettlementEngine:
    """Settles open paper trades using plugin truth resolution."""

    def __init__(self, plugin: MarketPlugin, trades_path: Path) -> None:
        self.plugin = plugin
        self.trades_path = trades_path

    def settle_all(
        self,
        specs: list[MarketSpec],
        today_str: str | None = None,
    ) -> dict[str, Any]:
        """Settle all unsettled trades that have a final truth available."""
        if today_str is None:
            today_str = date.today().isoformat()

        if not self.trades_path.exists():
            return {"settled": 0, "pending": 0, "total_pnl": 0.0}

        data = json.loads(self.trades_path.read_text())
        trades = data.get("trades", [])

        spec_map: dict[tuple[str, str], MarketSpec] = {}
        for spec in specs:
            city = spec.metadata.get("city", "")
            target_date = spec.metadata.get("target_date", "")
            spec_map[(city, target_date)] = spec

        settled_count = 0
        pending_count = 0
        total_pnl = 0.0
        updated_trades: list[dict[str, Any]] = []

        for trade in trades:
            if trade.get("outcome") is not None:
                updated_trades.append(trade)
                if trade.get("realized_pnl") is not None:
                    total_pnl += float(trade["realized_pnl"])
                continue

            target_date = trade.get("target_date", "")
            city_or_segment = trade.get("city_or_segment", "")

            if target_date > today_str:
                pending_count += 1
                updated_trades.append(trade)
                continue

            maybe_spec: MarketSpec | None = spec_map.get((city_or_segment, target_date))
            if maybe_spec is None:
                pending_count += 1
                updated_trades.append(trade)
                continue

            if not self.plugin.is_truth_final(maybe_spec):
                pending_count += 1
                updated_trades.append(trade)
                continue

            truth = self.plugin.fetch_truth(maybe_spec)
            if truth is None:
                pending_count += 1
                updated_trades.append(trade)
                continue

            if isinstance(truth, str):
                winning_label: str | None = truth
            else:
                winning_label = maybe_spec.resolve_winning_bin(float(truth))

            if winning_label is None:
                pending_count += 1
                updated_trades.append(trade)
                continue

            settled_trade = self._settle_trade(trade, winning_label)
            settled_count += 1
            total_pnl += float(settled_trade["realized_pnl"])
            updated_trades.append(settled_trade)

        self.trades_path.write_text(json.dumps({"trades": updated_trades}, indent=2))

        return {
            "settled": settled_count,
            "pending": pending_count,
            "total_pnl": round(total_pnl, 6),
        }

    def _settle_trade(self, trade: dict[str, Any], winning_label: str) -> dict[str, Any]:
        """Compute PnL and outcome for a single trade."""
        direction = trade.get("direction", "yes")
        gamma_price = float(trade["gamma_price"])
        flat_stake = float(trade.get("flat_stake", 1.0))
        fee_paid = float(trade.get("fee_paid", 0.003))
        entry_price = gamma_price if direction == "yes" else (1.0 - gamma_price)
        size = flat_stake / max(entry_price, 1e-9)
        side = "buy" if direction == "yes" else "sell"
        pos = Position(outcome_label=trade["outcome_label"], price=entry_price, size=size, side=side)
        pnl = settle_position(pos, winning_label=winning_label, fee_paid=fee_paid)
        outcome = "won" if trade["outcome_label"] == winning_label else "lost"
        return {**trade, "outcome": outcome, "realized_pnl": round(pnl, 6)}
