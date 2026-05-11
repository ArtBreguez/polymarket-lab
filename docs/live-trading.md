# Live Trading with LiveBroker

This guide covers the paper → live transition using `LiveBroker`.

---

## Overview

`pmlab` has a deliberate two-stage execution path:

```
PaperBroker  →  (validators pass)  →  LiveBroker
   ↓                                      ↓
JSON trades                         real CLOB orders
no capital at risk                  real USDC at risk
```

You **must** pass your validators with `PaperBroker` before switching to `LiveBroker`.

---

## Getting a Polymarket API Key

1. Go to [polymarket.com](https://polymarket.com) and log in
2. Navigate to Settings → API keys
3. Create a new key — you'll receive `api_key`, `api_secret`, `api_passphrase`
4. Store them in environment variables — **never in code**:
   ```bash
   export POLY_API_KEY="..."
   export POLY_API_SECRET="..."
   export POLY_API_PASSPHRASE="..."
   ```

---

## Dry-Run Mode (Always First)

Before sending real orders, always validate with `dry_run=True`:

```python
import os
from pmlab import LiveBroker

with LiveBroker(
    api_key=os.environ["POLY_API_KEY"],
    api_secret=os.environ["POLY_API_SECRET"],
    api_passphrase=os.environ["POLY_API_PASSPHRASE"],
    dry_run=True,  # no real orders sent
) as broker:
    receipt = broker.place_order(
        token_id="0xabc...",
        side="BUY",
        price=0.35,
        size=14.28,
    )
    print(receipt.status)   # "dry_run"
    print(receipt.order_id) # "dry-run-0xabc..."
```

All methods return realistic responses in dry-run mode without touching the API.

---

## Checking Your Balance

```python
with LiveBroker(...) as broker:
    balance = broker.get_balance()
    print(f"Available: ${balance:.2f} USDC")
```

---

## Placing a Real Order

```python
from pmlab import LiveBroker, kelly_stake_size

win_prob = 0.65
entry_price = 0.35
bankroll = 50.0  # USDC

stake = kelly_stake_size(
    win_prob=win_prob,
    entry_price=entry_price,
    bankroll=bankroll,
    fraction=0.25,
    max_exposure=0.05,
)
shares = stake / entry_price

with LiveBroker(
    api_key=os.environ["POLY_API_KEY"],
    api_secret=os.environ["POLY_API_SECRET"],
    api_passphrase=os.environ["POLY_API_PASSPHRASE"],
    dry_run=False,  # REAL ORDER
) as broker:
    receipt = broker.place_order(
        token_id="0xabc...",
        side="BUY",
        price=entry_price,
        size=round(shares, 2),
        order_type="GTC",
    )
    print(f"Order ID: {receipt.order_id}")
    print(f"Status:   {receipt.status}")
```

---

## Managing Orders

```python
with LiveBroker(...) as broker:
    # List open orders
    orders = broker.get_open_orders()
    for o in orders:
        print(o["orderID"], o["price"], o["size"])

    # Cancel a specific order
    result = broker.cancel_order("order-id-here")

    # Cancel ALL open orders (use with care!)
    result = broker.cancel_all_orders()
```

---

## Kelly Sizing Reference

| Parameter | Recommended value |
|---|---|
| `fraction` | 0.25 (quarter-Kelly) — conservative, reduces variance |
| `max_exposure` | 0.05 (5% of bankroll max per trade) |
| `bankroll` | Available USDC balance — call `get_balance()` before sizing |

Quarter-Kelly is standard for real markets. Full Kelly (`fraction=1.0`) maximizes long-run growth but produces large drawdowns in practice.

---

## Validator Checklist (Before Real Capital)

Run these before flipping `dry_run=False`:

- [ ] `PaperBroker` has 80+ settled trades
- [ ] ROI > 20% on paper trades
- [ ] Holdout gate passes with `HoldoutGateResult.decision == "GO"`
- [ ] Brier skill score > 0 (model better than climatology)
- [ ] No open positions from a failed/stale paper run
- [ ] `LiveBroker(dry_run=True)` tested with target token IDs
- [ ] Wallet funded with USDC you are willing to lose

---

## Error Handling

`LiveBroker` raises `LiveBrokerError` on any API failure:

```python
from pmlab import LiveBroker
from pmlab.execution.live_broker import LiveBrokerError

try:
    receipt = broker.place_order(...)
except LiveBrokerError as e:
    print(f"Order failed: {e}")
    # log, alert, do NOT retry blindly
```

Never retry a failed `place_order` without first checking `get_open_orders()` — the order may have been accepted before the network error.
