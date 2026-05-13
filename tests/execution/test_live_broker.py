"""Tests for LiveBroker."""

from __future__ import annotations

import httpx
import pytest
import respx

from pmlab.execution.live_broker import LiveBroker, LiveBrokerError


@pytest.fixture
def broker():
    return LiveBroker(
        api_key="test-key",
        api_secret="test-secret",
        api_passphrase="test-pass",
        base_url="https://mock-clob.test",
    )


@pytest.fixture
def dry_broker():
    return LiveBroker(
        api_key="test-key",
        api_secret="test-secret",
        api_passphrase="test-pass",
        base_url="https://mock-clob.test",
        dry_run=True,
    )


class TestDryRun:
    def test_place_order_dry_run(self, dry_broker):
        receipt = dry_broker.place_order("tok123", "BUY", 0.55, 10.0)
        assert receipt.status == "dry_run"
        assert receipt.token_id == "tok123"

    def test_cancel_order_dry_run(self, dry_broker):
        result = dry_broker.cancel_order("ord-abc")
        assert result["dry_run"] is True

    def test_cancel_all_dry_run(self, dry_broker):
        result = dry_broker.cancel_all_orders()
        assert result["cancelled_all"] is True


class TestValidation:
    def test_invalid_price_zero(self, broker):
        with pytest.raises(ValueError, match="price"):
            broker.place_order("tok", "BUY", 0.0, 10.0)

    def test_invalid_price_one(self, broker):
        with pytest.raises(ValueError, match="price"):
            broker.place_order("tok", "BUY", 1.0, 10.0)

    def test_invalid_size(self, broker):
        with pytest.raises(ValueError, match="size"):
            broker.place_order("tok", "BUY", 0.5, 0.0)

    def test_invalid_side(self, broker):
        with pytest.raises(ValueError, match="side"):
            broker.place_order("tok", "LONG", 0.5, 10.0)


class TestPlaceOrder:
    @respx.mock
    def test_success(self, broker):
        respx.post("https://mock-clob.test/order").mock(
            return_value=httpx.Response(200, json={"orderID": "abc123", "status": "LIVE"})
        )
        receipt = broker.place_order("tok", "BUY", 0.55, 10.0)
        assert receipt.order_id == "abc123"
        assert receipt.status == "LIVE"

    @respx.mock
    def test_http_error(self, broker):
        respx.post("https://mock-clob.test/order").mock(
            return_value=httpx.Response(400, text="Bad Request")
        )
        with pytest.raises(LiveBrokerError, match="place_order failed"):
            broker.place_order("tok", "BUY", 0.55, 10.0)


class TestGetBalance:
    @respx.mock
    def test_success(self, broker):
        respx.get("https://mock-clob.test/balance").mock(
            return_value=httpx.Response(200, json={"balance": "42.5"})
        )
        balance = broker.get_balance()
        assert balance == pytest.approx(42.5)


class TestGetOpenOrders:
    @respx.mock
    def test_success(self, broker):
        respx.get("https://mock-clob.test/orders").mock(
            return_value=httpx.Response(200, json=[{"orderID": "x"}])
        )
        orders = broker.get_open_orders()
        assert len(orders) == 1


class TestCancelOrder:
    @respx.mock
    def test_success(self, broker):
        respx.delete("https://mock-clob.test/order/abc").mock(
            return_value=httpx.Response(200, json={"cancelled": "abc"})
        )
        result = broker.cancel_order("abc")
        assert result["cancelled"] == "abc"

    @respx.mock
    def test_cancel_all(self, broker):
        respx.delete("https://mock-clob.test/orders").mock(
            return_value=httpx.Response(200, json={"count": 3})
        )
        result = broker.cancel_all_orders()
        assert result["count"] == 3


class TestContextManager:
    def test_context_manager(self):
        with LiveBroker("k", "s", "p", dry_run=True) as b:
            assert b.dry_run is True
