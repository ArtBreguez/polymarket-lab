"""Tests for LiveBroker (py-clob-client implementation)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pmlab.execution.live_broker import LiveBroker, LiveBrokerError

MOCK_CREDS = {
    "api_key": "test-key",
    "api_secret": "test-secret",
    "api_passphrase": "test-pass",
    "private_key": "0xdeadbeef",
    "base_url": "https://mock-clob.test",
}


@pytest.fixture
def broker():
    return LiveBroker(**MOCK_CREDS)


@pytest.fixture
def dry_broker():
    return LiveBroker(**MOCK_CREDS, dry_run=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_clob(broker_fixture, **method_returns):
    """Return a context manager that patches ClobClient on the broker's module."""
    mock_client = MagicMock()
    for method, retval in method_returns.items():
        getattr(mock_client, method).return_value = retval
    return patch(
        "pmlab.execution.live_broker.ClobClient",
        return_value=mock_client,
    )


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# place_order
# ---------------------------------------------------------------------------


class TestPlaceOrder:
    def test_success(self, broker):
        mock_order = MagicMock()
        post_resp = {"orderID": "abc123", "status": "LIVE"}

        with _patch_clob(broker, create_order=mock_order, post_order=post_resp):
            mock_client = MagicMock()
            mock_client.create_order.return_value = mock_order
            mock_client.post_order.return_value = post_resp
            with patch(
                "pmlab.execution.live_broker.ClobClient",
                return_value=mock_client,
            ):
                receipt = broker.place_order("tok", "BUY", 0.55, 10.0)

        assert receipt.order_id == "abc123"
        assert receipt.status == "LIVE"

    def test_api_error_raises_live_broker_error(self, broker):
        with (
            patch(
                "pmlab.execution.live_broker.ClobClient",
                return_value=MagicMock(
                    create_order=MagicMock(side_effect=Exception("Bad Request"))
                ),
            ),
            pytest.raises(LiveBrokerError, match="place_order failed"),
        ):
            broker.place_order("tok", "BUY", 0.55, 10.0)


# ---------------------------------------------------------------------------
# get_balance
# ---------------------------------------------------------------------------


class TestGetBalance:
    def test_success(self, broker):
        mock_client = MagicMock()
        mock_client.get_balance_allowance.return_value = {"balance": "42.5"}
        with patch("pmlab.execution.live_broker.ClobClient", return_value=mock_client):
            balance = broker.get_balance()
        assert balance == pytest.approx(42.5)


# ---------------------------------------------------------------------------
# get_open_orders
# ---------------------------------------------------------------------------


class TestGetOpenOrders:
    def test_success(self, broker):
        mock_client = MagicMock()
        mock_client.get_orders.return_value = [{"orderID": "x"}]
        with patch("pmlab.execution.live_broker.ClobClient", return_value=mock_client):
            orders = broker.get_open_orders()
        assert len(orders) == 1


# ---------------------------------------------------------------------------
# cancel_order / cancel_all_orders
# ---------------------------------------------------------------------------


class TestCancelOrder:
    def test_success(self, broker):
        mock_client = MagicMock()
        mock_client.cancel.return_value = {"cancelled": "abc"}
        with patch("pmlab.execution.live_broker.ClobClient", return_value=mock_client):
            result = broker.cancel_order("abc")
        assert result["cancelled"] == "abc"

    def test_cancel_all(self, broker):
        mock_client = MagicMock()
        mock_client.cancel_all.return_value = {"count": 3}
        with patch("pmlab.execution.live_broker.ClobClient", return_value=mock_client):
            result = broker.cancel_all_orders()
        assert result["count"] == 3


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class TestContextManager:
    def test_context_manager(self):
        with LiveBroker("k", "s", "p", dry_run=True) as b:
            assert b.dry_run is True
