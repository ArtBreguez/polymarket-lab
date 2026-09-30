# Backtest Costs — `pmlab.core.costs`

A backtest that fills at the quoted price reports an edge the broker cannot
reproduce. `CostModel` makes execution friction explicit, and it is **opt-in**:
omit it and `rolling_origin_eval` behaves exactly as it did before this module
existed, so previously published results stay reproducible.

---

## Why this is not a detail

On Polymarket's mutually-exclusive (negRisk) events the gap between the mid and
the executable price is structural, not incidental. Read at the mid, a field of
outcomes sums to ~1.0 and looks coherent; read at the ask, it sums above 1.0, and
in large fields half the outcomes have no book at all. Any edge measured against a
mid price is a spread you have not paid yet.

The same applies inside a single market: `market_price` in the training panel is a
quote, and your fill is worse than a quote by the half-spread plus whatever the
book charges for your size.

---

## Basic Usage

```python
from pmlab import CostModel
from pmlab.backtest.rolling_origin import rolling_origin_eval

# Default — identical to pre-0.8.3 behaviour: 30bps taker fee, fill at the quote.
result = rolling_origin_eval(panel, model)

# Realistic: pay the fee, cross the half-spread, and walk the book on size.
costs = CostModel(
    taker_bps=30.0,            # Polymarket CLOB taker fee
    slippage_bps=50.0,         # half-spread, as a fraction of price
    depth_bps_per_unit=20.0,   # extra bps per multiple of the reference stake
    depth_reference_stake=10.0,  # USDC that fills without moving the book
)
result = rolling_origin_eval(panel, model, flat_stake=25.0, costs=costs)

# Upper bound — what the strategy would earn in a frictionless world.
best_case = rolling_origin_eval(panel, model, costs=CostModel.frictionless())
```

Comparing the two runs is the point: the distance between them is how much of
your edge is paid to the market.

---

## The arithmetic

```
depth_penalty_bps = depth_bps_per_unit * max(0, stake - ref) / ref
total_bps         = slippage_bps + depth_penalty_bps
fill              = price * (1 + total_bps / 10_000) + slippage_fixed
fill              = min(fill, 1.0)
fee               = stake * taker_bps / 10_000
```

Notes on each term:

| Parameter | Unit | Models |
|---|---|---|
| `taker_bps` | bps of notional stake | exchange fee |
| `slippage_bps` | bps of price | half-spread crossed on entry |
| `slippage_fixed` | probability units | a floor cost (0.01 = one cent) |
| `depth_bps_per_unit` | bps per multiple of reference | walking the book on size |
| `depth_reference_stake` | USDC | size that fills at top of book |

`slippage_bps` is proportional to price, so a $0.10 outcome slips less in absolute
terms than a $0.90 one — which matches how a percentage spread behaves. Use
`slippage_fixed` when the cost is a floor rather than a percentage.

The fill is clamped at 1.0. A probability price above certainty would create a
position paying more than it can settle for.

---

## What the depth term is, and is not

It is a **documented linear approximation**, not an order-book walk. pmlab's
training panel carries `market_price`, not L2 depth, so the model cannot know
where the book thins out. The penalty grows linearly in *excess* stake above a
reference you choose.

If you have depth data, measure it and set `depth_bps_per_unit` from the
measurement rather than guessing. If you don't, prefer a pessimistic value: a
backtest that overstates cost fails safe, one that understates it promotes a
champion that loses money live.

---

## Where costs propagate

Costs are not cosmetic — they flow into the promotion decision:

1. `rolling_origin_eval` fills at `fill_price` instead of `market_price`, and
   records that column in the trade log.
2. `edge` is computed as `predicted_prob - fill_price`. Measuring it against the
   quote would overstate every trade by the friction.
3. `compute_metrics` averages `edge` into `avg_edge`.
4. `HoldoutGateResult` reads those metrics. **A book whose edge exists only at the
   mid can no longer pass the gate.**
5. `stability_report` bootstraps the post-cost PnL, so the confidence interval is
   around what you would actually have earned.

---

## Compatibility

- No `costs=` argument → 30bps fee, fill at the quote. Byte-identical to the old
  behaviour; there is a test asserting frame equality.
- `taker_bps=` still works on its own for fee-only adjustments.
- Passing **both** `taker_bps=` and `costs=` raises `ValueError`. Two sources of
  truth for the fee would silently double-charge it.

## Reproducibility

`CostModel.to_dict()` returns a flat mapping that round-trips through
`CostModel(**payload)`, so log it with the run:

```python
with tracker.run(experiment="weather_tmax", run_name="with_costs") as run:
    run.log_params(costs.to_dict())
```

A PnL number without its cost model is not reproducible — the same panel and the
same model give a different answer under different friction.
