"""SettlementEngine — settles paper trades against plugin truth values."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from pmlab.core.market_spec import MarketSpec
from pmlab.core.pnl import Position, settle_position
from pmlab.plugins.base import MarketPlugin

# Stamped on every trade this engine settles. Logs settled by 0.9.0 or earlier have
# no such key, and a NO trade among them carries the old, wrong realized_pnl: settling
# is idempotent, so it is never recomputed. 1 = NO settled as a buy of the NO token.
SETTLEMENT_VERSION = 1

# The first version whose NO trades carry a correct realized_pnl. A NO trade stamped with
# a lower version (or none) was booked by the short-YES arithmetic and must not be used
# as evidence about the NO strategy; `pmlab resettle` recomputes it.
MIN_TRUSTED_NO_SETTLEMENT_VERSION = 1


class SettlementEngine:
    """Settles open paper trades using plugin truth resolution."""

    def __init__(self, plugin: MarketPlugin, trades_path: Path) -> None:
        self.plugin = plugin
        self.trades_path = trades_path

    def settle_all(
        self,
        specs: list[MarketSpec],
        today_str: str | None = None,
        now_utc: datetime | None = None,
    ) -> dict[str, Any]:
        """Settle all unsettled trades that have a final truth available.

        Each trade it settles is stamped with ``settled_at`` (UTC): when its PnL
        became known, which is what lets the watchdog replay an earlier ``--as-of``
        without seeing truth that had not arrived yet.

        Args:
            specs: The markets to resolve against.
            today_str: ``YYYY-MM-DD`` of "today"; trades dated after it stay open.
            now_utc: The settlement time to stamp (injected for testing; defaults
                to now). Must be timezone-aware.

        Raises:
            ValueError: If ``now_utc`` is naive.
        """
        if today_str is None:
            today_str = date.today().isoformat()
        if now_utc is None:
            now_utc = datetime.now(UTC)
        elif now_utc.tzinfo is None:
            raise ValueError("now_utc must be timezone-aware (a naive time has no zone to stamp)")
        settled_at = now_utc.astimezone(UTC).isoformat()

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

            settled_trade = self._settle_trade(trade, winning_label, settled_at)
            settled_count += 1
            total_pnl += float(settled_trade["realized_pnl"])
            updated_trades.append(settled_trade)

        self.trades_path.write_text(json.dumps({"trades": updated_trades}, indent=2))

        return {
            "settled": settled_count,
            "pending": pending_count,
            "total_pnl": round(total_pnl, 6),
        }

    def _settle_trade(
        self, trade: dict[str, Any], winning_label: str, settled_at: str
    ) -> dict[str, Any]:
        """Compute PnL and outcome for a single trade."""
        direction = trade.get("direction", "yes")
        gamma_price = float(trade["gamma_price"])
        flat_stake = float(trade.get("flat_stake", 1.0))
        fee_paid = float(trade.get("fee_paid", 0.003))
        # The price of the token this trade buys: the label's YES price, or for a NO
        # trade the NO token's price. `gamma_price` is always the YES price.
        quote = gamma_price if direction == "yes" else (1.0 - gamma_price)
        # Enter at what the broker actually paid. `fill_price` carries the cost
        # model's slippage and depth; falling back to the quote only for logs
        # written before it was recorded. Settling at the quote would make live
        # PnL look better than the backtest by exactly the friction it modelled.
        fill_price = trade.get("fill_price")
        entry_price = float(fill_price) if fill_price is not None else quote
        # Said on the trade, so a reader can tell a PnL that carries the slippage
        # from one that does not. Releases up to 0.9.0 settled at the quote even
        # when a fill_price was recorded, and wrote no such key.
        basis = "fill_price" if fill_price is not None else "quote"
        size = flat_stake / max(entry_price, 1e-9)
        # Both directions are a BUY of a token at `entry_price`, exactly as the
        # backtest prices a trade. A NO trade used to be settled as a short YES at
        # the NO price, which is the wrong price for that formula: it booked a lost
        # NO bet as a fraction of its stake and a won one as a full stake.
        side = "buy" if direction == "yes" else "buy_no"
        pos = Position(
            outcome_label=trade["outcome_label"], price=entry_price, size=size, side=side
        )
        pnl = settle_position(pos, winning_label=winning_label, fee_paid=fee_paid)
        # About the label, not the money: a NO trade on a label that occurred is
        # "won" while its PnL is negative. Calibration reads it that way.
        outcome = "won" if trade["outcome_label"] == winning_label else "lost"
        return {
            **trade,
            "outcome": outcome,
            "realized_pnl": round(pnl, 6),
            "settled_at": settled_at,
            "settled_basis": basis,
            "settlement_version": SETTLEMENT_VERSION,
        }
