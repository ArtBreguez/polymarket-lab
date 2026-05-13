"""LiveBroker — real-money order execution via Polymarket CLOB API.

Uses py-clob-client for proper L1 (ECDSA private key) + L2 (API key/secret/passphrase)
authentication, matching the Polymarket CLOB API requirements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, OrderArgs

CLOB_API_BASE = "https://clob.polymarket.com"


@dataclass
class OrderReceipt:
    order_id: str
    token_id: str
    side: str
    price: float
    size: float
    status: str
    placed_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class LiveBrokerError(Exception):
    pass


class LiveBroker:
    """Real-money order execution via the Polymarket CLOB API.

    Requires:
      - private_key: L1 ECDSA private key (0x-prefixed hex string)
      - api_key / api_secret / api_passphrase: L2 API credentials

    In dry_run mode, no requests are made to the API.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        api_passphrase: str,
        private_key: str = "",
        chain_id: int = 137,
        signature_type: int = 0,
        funder_address: str | None = None,
        base_url: str = CLOB_API_BASE,
        dry_run: bool = False,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase
        self.private_key = private_key
        self.chain_id = chain_id
        self.signature_type = signature_type
        self.funder_address = funder_address
        self.base_url = base_url
        self.dry_run = dry_run

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def place_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
        order_type: str = "GTC",
    ) -> OrderReceipt:
        if not 0 < price < 1:
            raise ValueError(f"price must be in (0, 1), got {price}")
        if size <= 0:
            raise ValueError(f"size must be > 0, got {size}")
        if side not in ("BUY", "SELL"):
            raise ValueError(f"side must be BUY or SELL, got {side}")

        if self.dry_run:
            return OrderReceipt(
                order_id=f"dry-run-{token_id[:8]}",
                token_id=token_id,
                side=side,
                price=price,
                size=size,
                status="dry_run",
            )

        try:
            client = self._build_client()
            order = client.create_order(
                OrderArgs(token_id=token_id, price=price, size=size, side=side)
            )
            resp: dict[str, Any] = client.post_order(order)
        except Exception as exc:
            raise LiveBrokerError(f"place_order failed: {exc}") from exc

        return OrderReceipt(
            order_id=str(resp.get("orderID", resp.get("order_id", ""))),
            token_id=token_id,
            side=side,
            price=price,
            size=size,
            status=str(resp.get("status", "unknown")),
        )

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        if self.dry_run:
            return {"cancelled": order_id, "dry_run": True}
        try:
            client = self._build_client()
            result: dict[str, Any] = client.cancel(order_id)
            return result
        except Exception as exc:
            raise LiveBrokerError(f"cancel_order failed: {exc}") from exc

    def cancel_all_orders(self) -> dict[str, Any]:
        if self.dry_run:
            return {"cancelled_all": True, "dry_run": True}
        try:
            client = self._build_client()
            result: dict[str, Any] = client.cancel_all()
            return result
        except Exception as exc:
            raise LiveBrokerError(f"cancel_all_orders failed: {exc}") from exc

    def get_open_orders(self) -> list[dict[str, Any]]:
        try:
            client = self._build_client()
            result: list[dict[str, Any]] = client.get_orders()
            return result
        except Exception as exc:
            raise LiveBrokerError(f"get_open_orders failed: {exc}") from exc

    def get_balance(self) -> float:
        try:
            client = self._build_client()
            data = client.get_balance_allowance()
            if isinstance(data, dict):
                return float(data.get("balance", 0.0))
            return float(data)
        except Exception as exc:
            raise LiveBrokerError(f"get_balance failed: {exc}") from exc

    def preflight(self) -> dict[str, Any]:
        """Run a health check against the CLOB API. Returns status dict."""
        messages: list[str] = []
        ok = True

        if not self.private_key:
            ok = False
            messages.append("Missing private_key (L1 ECDSA key required).")

        if not all([self.api_key, self.api_secret, self.api_passphrase]):
            ok = False
            messages.append("Missing L2 API credentials (api_key/api_secret/api_passphrase).")

        try:
            client = self._build_client()
            client.get_ok()
            messages.append("CLOB health check succeeded.")
        except Exception as exc:
            ok = False
            messages.append(f"CLOB health check failed: {exc}")

        return {"ok": ok, "messages": messages}

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> LiveBroker:
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_client(self) -> ClobClient:
        creds = ApiCreds(
            api_key=self.api_key,
            api_secret=self.api_secret,
            api_passphrase=self.api_passphrase,
        )
        return ClobClient(
            self.base_url,
            chain_id=self.chain_id,
            key=self.private_key or None,
            creds=creds,
            signature_type=self.signature_type,
            funder=self.funder_address or None,
        )
