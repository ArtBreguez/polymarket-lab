# Market data clients

The Gamma and CLOB clients, their async twins, and the caches in front of them.

## Fee schedules

A market's fee is read from Gamma (`feesEnabled`, `feeType`, `feeSchedule`) and checked against CLOB's
`/fee-rate`. That endpoint's `base_fee` is **a flag, not the rate**: 1000 on every market that charges
a fee, 0 on one that does not, so the rate is always Gamma's (or, only when Gamma says nothing, the
category default, recorded as a fallback). Snapshots keep the moment they were seen, so a backtest
prices a trade with the schedule that was known before it. The sources are in the docs of
`pmlab.markets.fees`.

```python
import tempfile
from datetime import UTC, datetime

from pmlab import FeeScheduleStore, resolve_fee_snapshot

market = {  # what Gamma returns for a market (trimmed)
    "id": "2247194",
    "slug": "highest-temperature-in-atlanta-on-may-15-2026-88-89f",
    "feesEnabled": True,
    "feeType": "weather_fees",
    "feeSchedule": {"rate": 0.01, "exponent": 1, "takerOnly": True, "rebateRate": 0.25},
}
seen = datetime(2026, 11, 26, 18, 0, tzinfo=UTC)
snapshot = resolve_fee_snapshot(market, base_fee=1000, category="weather", seen_at=seen)
print(snapshot.schedule.rate, snapshot.source, snapshot.fallback)

with tempfile.TemporaryDirectory() as root:
    store = FeeScheduleStore(root)
    store.record([snapshot])
    print(sorted(store.as_of("2026-11-27")), store.as_of("2026-11-26"))
```

```text
0.01 gamma False
['2247194'] {}
```


::: pmlab.GammaClient

::: pmlab.ClobClient

::: pmlab.AsyncGammaClient

::: pmlab.AsyncClobClient

::: pmlab.DiskCache

::: pmlab.TypedCache

::: pmlab.FeeSnapshot

::: pmlab.FeeScheduleStore

::: pmlab.resolve_fee_snapshot
