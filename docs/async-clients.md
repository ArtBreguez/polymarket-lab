# Async Clients & Disk Cache

`pmlab.markets` provides both synchronous and asynchronous API clients, plus a TTL disk cache to avoid redundant network calls.

---

## Sync Clients (Default)

### GammaClient

```python
from pmlab import GammaClient

with GammaClient() as client:
    markets = client.fetch_markets(tag="temperature", keyword="highest", limit=100)

for m in markets:
    print(m["question"], m["conditionId"])
```

### ClobClient

```python
from pmlab import ClobClient

with ClobClient() as client:
    prices = client.fetch_prices(token_ids=["0xabc...", "0xdef..."])

for token_id, price in prices.items():
    print(f"{token_id}: {price:.3f}")
```

---

## Async Clients

Use async clients when scanning many markets or tokens in parallel.

### AsyncGammaClient

```python
import asyncio
from pmlab import AsyncGammaClient

async def main():
    async with AsyncGammaClient() as client:
        markets = await client.fetch_markets(tag="temperature", keyword="highest")
    return markets

markets = asyncio.run(main())
```

### AsyncClobClient

```python
import asyncio
from pmlab import AsyncClobClient

async def fetch_all_prices(token_ids: list[str]) -> dict[str, float]:
    async with AsyncClobClient(concurrency=20) as client:
        return await client.fetch_prices(token_ids)

prices = asyncio.run(fetch_all_prices(["0xabc...", "0xdef...", ...]))
```

The `concurrency` parameter controls the `asyncio.Semaphore` — how many tokens are fetched simultaneously. Default is 10. Failed tokens are silently skipped (not in the result dict).

### Parallel scan pattern

```python
import asyncio
from pmlab import AsyncGammaClient, AsyncClobClient

async def full_scan(tag: str):
    async with AsyncGammaClient() as gamma:
        markets = await gamma.fetch_markets(tag=tag)

    token_ids = [m["conditionId"] for m in markets if "conditionId" in m]

    async with AsyncClobClient(concurrency=15) as clob:
        prices = await clob.fetch_prices(token_ids)

    return [
        {**m, "current_price": prices.get(m.get("conditionId", ""))}
        for m in markets
    ]

results = asyncio.run(full_scan("temperature"))
```

---

## DiskCache

Avoid hitting the API on every scan by caching responses to disk.

### Basic usage

```python
from pmlab import DiskCache, GammaClient

cache = DiskCache(cache_dir=".pmlab_cache", ttl_seconds=3600)  # 1-hour TTL
client = GammaClient(cache=cache)

markets = client.fetch_markets(tag="temperature")  # API call
markets = client.fetch_markets(tag="temperature")  # served from disk cache
```

### Manual cache operations

```python
from pmlab import DiskCache

cache = DiskCache(".pmlab_cache", ttl_seconds=1800)

# Set a value
cache.set("my_key", {"data": [1, 2, 3]})

# Get a value (returns None if missing or expired)
value = cache.get("my_key")
value = cache.get("missing", default=[])  # custom default

# Check membership
if "my_key" in cache:
    print("cached!")

# Delete one entry
cache.delete("my_key")

# Remove all entries
cache.clear()

# Remove only expired entries
purged = cache.purge_expired()
print(f"Purged {purged} expired entries")
```

### Cache key design

The cache uses MD5 hashes of the key string. Keys can be any string — use descriptive prefixes:

```python
cache.set(f"gamma:markets:{tag}:{limit}", markets)
cache.set(f"clob:price:{token_id}", price)
cache.set(f"weather:forecast:{city}:{date}", forecast)
```

### TTL recommendations

| Data type | Recommended TTL |
|---|---|
| Polymarket market list | 1–4 hours (`3600–14400`) |
| CLOB prices | 5–15 minutes (`300–900`) |
| Weather forecasts | 3–6 hours (`10800–21600`) |
| Resolved outcomes | 24 hours+ (`86400`) |

---

## Choosing Sync vs Async

| Use sync when | Use async when |
|---|---|
| Fetching a single tag/set of markets | Fetching 50+ token prices |
| Running in a cron script | Building a real-time scanner |
| Simplicity matters | Latency matters |
| < 10 tokens | 10+ tokens |
