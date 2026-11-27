# Market data clients

The Gamma and CLOB clients, their async twins, and the caches in front of them.

## Fee schedules

A market's fee is read from Gamma (`feesEnabled`, `feeType`, `feeSchedule`) and checked against CLOB's
`/fee-rate`. That endpoint's `base_fee` is **a flag, not the rate**: 1000 on every market that charges
a fee, 0 on one that does not, so the rate is always Gamma's (or, only when Gamma says nothing, the
category default, recorded as a fallback). Snapshots keep the moment they were seen, so a backtest
prices a trade with the schedule that was known before it. The sources are listed under
[Where `base_fee` comes from](#where-base_fee-comes-from).

`fetch_fee_snapshot(slug)` reads one market from `GET /markets/slug/{slug}`, which serves a market
whatever its state: `GET /markets?slug=` lists open markets only and answers `[]` for a closed one.
For a closed market the snapshot is the schedule Gamma still carries today, seen now; it says nothing
about what the market charged when it traded, so a backtest needs a snapshot taken then.

`cache=` (a `DiskCache`) on the clients keeps a market's fee fields and a token's `base_fee` for the
cache's own TTL (`DiskCache` defaults to one hour; the clients do not read `PMLAB_CACHE_TTL`). Its
keys carry the slug or token id and not the host, so give a staging and a production client each
their own cache directory. A failed request is never cached.

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

### Where `base_fee` comes from

CLOB's `base_fee` is the `feeRateBps` of a legacy v1 order, in basis points. It is 1000 on every market
that charges a fee, whatever its `rate`, and 0 on one that does not: reading it as `1000 / 10_000` would
make a 0.05 market twice and a 0.01 market ten times too dear. Read on 2026-11-26:

- [`py-clob-client` b076b04 `client.py` L450-L458](https://github.com/Polymarket/py-clob-client/blob/b076b04d61135657e25dccc1bbd6866a96bd8c6e/py_clob_client/client.py#L450-L458):
  `get_fee_rate_bps` returns `result.get("base_fee") or 0`; [L476-L490](https://github.com/Polymarket/py-clob-client/blob/b076b04d61135657e25dccc1bbd6866a96bd8c6e/py_clob_client/client.py#L476-L490):
  it must equal the `fee_rate_bps` the user signs; `order_builder/builder.py` L139 and L178:
  `feeRateBps=str(order_args.fee_rate_bps)`.
- [`clob-client-v2` 2a3f0b0 `src/types/ordersV1.ts` L86-L88](https://github.com/Polymarket/clob-client-v2/blob/2a3f0b02b46de77afee2364881e0256eab4d75cb/src/types/ordersV1.ts#L86-L88):
  "Fee rate, in basis points, charged to the order maker, charged on proceeds"; `src/client.ts`
  L505-L518: `getFeeRateBps` reads `base_fee`.
- [`py-clob-client-v2` 9c7c9e8 `client.py` L773-L774 and L842-L843](https://github.com/Polymarket/py-clob-client-v2/blob/9c7c9e89cbbc11528daae90dc66ef65c9380cbdf/py_clob_client_v2/client.py#L773-L774):
  `fee_rate_bps` is asked for only `if version == 1`; a v2 order's fee comes from the market's rate
  and exponent, [`fees.py` L31](https://github.com/Polymarket/py-clob-client-v2/blob/9c7c9e89cbbc11528daae90dc66ef65c9380cbdf/py_clob_client_v2/fees.py#L31):
  `fee_rate * (price * (1 - price)) ** fee_exponent`.
- [`ctf-exchange-v2` ccc0596 `Fees.sol` L17, L36-L51](https://github.com/Polymarket/ctf-exchange-v2/blob/ccc0596074f4dfd62c944fbca4de252893b82b4b/src/exchange/mixins/Fees.sol#L36-L51)
  and `Trading.sol` L563-L575: a v2 order has no fee-rate field; the operator passes a fee amount and
  the contract only checks it is at most `maxFeeRateBps` (500 by default) of the trade's cash value.
- [`ctf-exchange` ed5c770 `CalculatorHelper.sol` L27-L48](https://github.com/Polymarket/ctf-exchange/blob/ed5c7708b7be3aa98bf5f0c6602b57cc498e2ef4/src/exchange/libraries/CalculatorHelper.sol#L27-L48)
  (v1, archived): `feeRateBps x min(price, 1 - price)`, a different curve from Gamma's
  `rate x (p(1-p))^exponent`, so the two numbers cannot be compared.
- Live Gamma and CLOB, 2026-11-26: every fee-charging market of one sample (six `feeType` values,
  `rate` 0.01 and 0.05) carried `takerBaseFee` and `makerBaseFee` 1000, and CLOB's `base_fee` was 1000
  for the Yes and for the No token of a sports, a weather and a crypto market, and 0 for both tokens
  of a fee-free one. The "flag" reading is an inference from that constancy; the code does not depend
  on it: any positive `base_fee` reads as "a fee is charged", and the rate is Gamma's.


::: pmlab.GammaClient

::: pmlab.ClobClient

::: pmlab.AsyncGammaClient

::: pmlab.AsyncClobClient

::: pmlab.DiskCache

::: pmlab.TypedCache

::: pmlab.FeeSnapshot

::: pmlab.FeeScheduleStore

::: pmlab.resolve_fee_snapshot
