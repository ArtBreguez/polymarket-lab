"""LiveBroker - place and manage real orders via the Polymarket CLOB API."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx

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
    """Real-money order execution via the Polymarket CLOB API."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        api_passphrase: str,
        base_url: str = CLOB_API_BASE,
        timeout: float = 30.0,
        dry_run: bool = False,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase
        self.base_url = base_url
        self.dry_run = dry_run
        self._client = httpx.Client(timeout=timeout)

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

        payload: dict[str, Any] = {
            "token_id": token_id,
            "side": side,
            "price": str(round(price, 4)),
            "size": str(round(size, 4)),
            "order_type": order_type,
        }

        if self.dry_run:
            return OrderReceipt(
                order_id="dry-run-" + token_id[:8],
                token_id=token_id,
                side=side,
                price=price,
                size=size,
                status="dry_run",
            )

        try:
            resp = self._client.post(
                f"{self.base_url}/order",
                json=payload,
                headers=self._auth_headers("POST", "/order", payload),
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LiveBrokerError(f"place_order failed: {exc.response.text}") from exc
        except httpx.HTTPError as exc:
            raise LiveBrokerError(f"place_order network error: {exc}") from exc

        data: dict[str, Any] = resp.json()
        return OrderReceipt(
            order_id=str(data.get("orderID", data.get("order_id", ""))),
            token_id=token_id,
            side=side,
            price=price,
            size=size,
            status=str(data.get("status", "unknown")),
        )

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        if self.dry_run:
            return {"cancelled": order_id, "dry_run": True}
        try:
            resp = self._client.delete(
                f"{self.base_url}/order/{order_id}",
                headers=self._auth_headers("DELETE", f"/order/{order_id}", {}),
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LiveBrokerError(f"cancel_order failed: {exc.response.text}") from exc
        except httpx.HTTPError as exc:
            raise LiveBrokerError(f"cancel_order network error: {exc}") from exc
        result: dict[str, Any] = resp.json()
        return result

    def cancel_all_orders(self) -> dict[str, Any]:
        if self.dry_run:
            return {"cancelled_all": True, "dry_run": True}
        try:
            resp = self._client.delete(
                f"{self.base_url}/orders",
                headers=self._auth_headers("DELETE", "/orders", {}),
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LiveBrokerError(f"cancel_all_orders failed: {exc.response.text}") from exc
        except httpx.HTTPError as exc:
            raise LiveBrokerError(f"cancel_all_orders network error: {exc}") from exc
        result: dict[str, Any] = resp.json()
        return result

    def get_open_orders(self) -> list[dict[str, Any]]:
        try:
            resp = self._client.get(
                f"{self.base_url}/orders",
                params={"status": "OPEN"},
                headers=self._auth_headers("GET", "/orders", {}),
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LiveBrokerError(f"get_open_orders failed: {exc.response.text}") from exc
        except httpx.HTTPError as exc:
            raise LiveBrokerError(f"get_open_orders network error: {exc}") from exc
        result: list[dict[str, Any]] = resp.json()
        return result

    def get_balance(self) -> float:
        try:
            resp = self._client.get(
                f"{self.base_url}/balance",
                headers=self._auth_headers("GET", "/balance", {}),
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise LiveBrokerError(f"get_balance failed: {exc.response.text}") from exc
        except httpx.HTTPError as exc:
            raise LiveBrokerError(f"get_balance network error: {exc}") from exc
        data: dict[str, Any] = resp.json()
        return float(data.get("balance", 0.0))

    def _auth_headers(self, method: str, path: str, body: dict[str, Any]) -> dict[str, str]:
        import hashlib
        import hmac
        import json

        ts = str(int(datetime.now(UTC).timestamp()))
        body_str = json.dumps(body, separators=(",", ":")) if body else ""
        message = ts + method.upper() + path + body_str
        sig = hmac.new(
            self.api_secret.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()
        return {
            "POLY-API-KEY": self.api_key,
            "POLY-SIGNATURE": sig,
            "POLY-TIMESTAMP": ts,
            "POLY-PASSPHRASE": self.api_passphrase,
            "Content-Type": "application/json",
        }

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> LiveBroker:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
