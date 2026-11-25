> ## Documentation Index
> Fetch the complete documentation index at: https://docs.polymarket.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Fees

> Understanding trading fees on Polymarket

Polymarket charges a small taker fee on certain markets. Fees are set by the protocol and applied at match time — you don't include fee information in your orders. These fees fund the [Maker Rebates Program](https://docs.polymarket.com/programs/maker-rebates.md), which redistributes fees daily to market makers to incentivize deeper liquidity and tighter spreads. Takers can also earn a portion of fees back through the tiered [Taker Rebate Program](https://docs.polymarket.com/programs/taker-rebates.md).

**Geopolitics markets are fee-free.** Polymarket does not charge fees or profit from trading activity on these markets. There are also no Polymarket fees to deposit or withdraw USDC (though intermediaries like Coinbase or MoonPay may charge their own fees).

**Combos have their own fee rate.** Every combo is charged the Combos taker fee rate in the table below, whatever categories its legs come from. The fee rates of the underlying markets don't apply.

***

## Fee Structure

Fees are calculated using the following formula:

```text theme={null} theme={null}
fee = C × feeRate × p × (1 - p)
```

Where **C** = number of shares traded and **p** = price of the shares. See
[Market Details](https://docs.polymarket.com/market-data/market-details.md#trading-fees) to read the fee
parameters for a market.

> [!NOTE]
>
> **Makers are never charged fees.** Only takers pay fees.

The fee parameters differ by market category:

| Category | Taker Fee Rate | Maker Fee Rate |
| - | - | - |
| Crypto — BTC 5-minute and 15-minute up/down | 0.05 | 0 |
| Crypto — all other markets | 0.01 | 0 |
| Sports, including NFL and college football | 0.05 | 0 |
| Finance | 0.05 | 0 |
| Tech | 0.05 | 0 |
| Politics — elections | 0.05 | 0 |
| Politics — excluding elections and geopolitics | 0.01 | 0 |
| Economics | 0.01 | 0 |
| Culture | 0.01 | 0 |
| Weather | 0.01 | 0 |
| Mentions | 0.01 | 0 |
| Geopolitics | 0 | 0 |
| Combos | 0.05 | 0 |
| Other / General | 0.05 | 0 |

Maker rebates depend on your [maker tier](https://docs.polymarket.com/programs/maker-rebates.md#funding). Combos and fee-free trades earn no maker rebates.

Taker fees are calculated in USDC and vary based on the share price. The fee amount in USDC is symmetric around 50% probability — a trade at 30¢ incurs the same dollar fee as a trade at 70¢.

## Fee Tables (100 Shares)

> [tab] Crypto
>
> BTC 5-minute and 15-minute up/down markets use 0.05; all other Crypto markets use 0.01.
>
> | Price | Trade Value | Taker Fee (0.05, USDC) | Taker Fee (0.01, USDC) |
> | - | - | - | - |
> | \$0.01 | \$1 | \$0.05 | \$0.01 |
> | \$0.05 | \$5 | \$0.24 | \$0.05 |
> | \$0.10 | \$10 | \$0.45 | \$0.09 |
> | \$0.15 | \$15 | \$0.64 | \$0.13 |
> | \$0.20 | \$20 | \$0.80 | \$0.16 |
> | \$0.25 | \$25 | \$0.94 | \$0.19 |
> | \$0.30 | \$30 | \$1.05 | \$0.21 |
> | \$0.35 | \$35 | \$1.14 | \$0.23 |
> | \$0.40 | \$40 | \$1.20 | \$0.24 |
> | \$0.45 | \$45 | \$1.24 | \$0.25 |
> | \$0.50 | \$50 | \$1.25 | \$0.25 |
> | \$0.55 | \$55 | \$1.24 | \$0.25 |
> | \$0.60 | \$60 | \$1.20 | \$0.24 |
> | \$0.65 | \$65 | \$1.14 | \$0.23 |
> | \$0.70 | \$70 | \$1.05 | \$0.21 |
> | \$0.75 | \$75 | \$0.94 | \$0.19 |
> | \$0.80 | \$80 | \$0.80 | \$0.16 |
> | \$0.85 | \$85 | \$0.64 | \$0.13 |
> | \$0.90 | \$90 | \$0.45 | \$0.09 |
> | \$0.95 | \$95 | \$0.24 | \$0.05 |
> | \$0.99 | \$99 | \$0.05 | \$0.01 |
>
> The fee in USDC **peaks at 50%** probability (\$1.25 for 0.05 and \$0.25 for 0.01) and decreases symmetrically toward both extremes.

> [tab] Sports
>
> | Price | Trade Value | Taker Fee (USDC) |
> | - | - | - |
> | \$0.01 | \$1 | \$0.05 |
> | \$0.05 | \$5 | \$0.24 |
> | \$0.10 | \$10 | \$0.45 |
> | \$0.15 | \$15 | \$0.64 |
> | \$0.20 | \$20 | \$0.80 |
> | \$0.25 | \$25 | \$0.94 |
> | \$0.30 | \$30 | \$1.05 |
> | \$0.35 | \$35 | \$1.14 |
> | \$0.40 | \$40 | \$1.20 |
> | \$0.45 | \$45 | \$1.24 |
> | \$0.50 | \$50 | \$1.25 |
> | \$0.55 | \$55 | \$1.24 |
> | \$0.60 | \$60 | \$1.20 |
> | \$0.65 | \$65 | \$1.14 |
> | \$0.70 | \$70 | \$1.05 |
> | \$0.75 | \$75 | \$0.94 |
> | \$0.80 | \$80 | \$0.80 |
> | \$0.85 | \$85 | \$0.64 |
> | \$0.90 | \$90 | \$0.45 |
> | \$0.95 | \$95 | \$0.24 |
> | \$0.99 | \$99 | \$0.05 |
>
> The fee in USDC **peaks at 50%** probability (\$1.25) and decreases symmetrically toward both extremes.

> [tab] Finance / Politics / Mentions / Tech
>
> Finance, Tech, and elections use 0.05. Other Politics and Mentions use 0.01. Geopolitics is fee-free.
>
> | Price | Trade Value | Taker Fee (0.05, USDC) | Taker Fee (0.01, USDC) |
> | - | - | - | - |
> | \$0.01 | \$1 | \$0.05 | \$0.01 |
> | \$0.05 | \$5 | \$0.24 | \$0.05 |
> | \$0.10 | \$10 | \$0.45 | \$0.09 |
> | \$0.15 | \$15 | \$0.64 | \$0.13 |
> | \$0.20 | \$20 | \$0.80 | \$0.16 |
> | \$0.25 | \$25 | \$0.94 | \$0.19 |
> | \$0.30 | \$30 | \$1.05 | \$0.21 |
> | \$0.35 | \$35 | \$1.14 | \$0.23 |
> | \$0.40 | \$40 | \$1.20 | \$0.24 |
> | \$0.45 | \$45 | \$1.24 | \$0.25 |
> | \$0.50 | \$50 | \$1.25 | \$0.25 |
> | \$0.55 | \$55 | \$1.24 | \$0.25 |
> | \$0.60 | \$60 | \$1.20 | \$0.24 |
> | \$0.65 | \$65 | \$1.14 | \$0.23 |
> | \$0.70 | \$70 | \$1.05 | \$0.21 |
> | \$0.75 | \$75 | \$0.94 | \$0.19 |
> | \$0.80 | \$80 | \$0.80 | \$0.16 |
> | \$0.85 | \$85 | \$0.64 | \$0.13 |
> | \$0.90 | \$90 | \$0.45 | \$0.09 |
> | \$0.95 | \$95 | \$0.24 | \$0.05 |
> | \$0.99 | \$99 | \$0.05 | \$0.01 |
>
> The fee in USDC **peaks at 50%** probability (\$1.25 for 0.05 and \$0.25 for 0.01) and decreases symmetrically toward both extremes.

> [tab] Economics / Culture / Weather / Other
>
> Other / General uses 0.05. Economics, Culture, and Weather use 0.01.
>
> | Price | Trade Value | Taker Fee (0.05, USDC) | Taker Fee (0.01, USDC) |
> | - | - | - | - |
> | \$0.01 | \$1 | \$0.05 | \$0.01 |
> | \$0.05 | \$5 | \$0.24 | \$0.05 |
> | \$0.10 | \$10 | \$0.45 | \$0.09 |
> | \$0.15 | \$15 | \$0.64 | \$0.13 |
> | \$0.20 | \$20 | \$0.80 | \$0.16 |
> | \$0.25 | \$25 | \$0.94 | \$0.19 |
> | \$0.30 | \$30 | \$1.05 | \$0.21 |
> | \$0.35 | \$35 | \$1.14 | \$0.23 |
> | \$0.40 | \$40 | \$1.20 | \$0.24 |
> | \$0.45 | \$45 | \$1.24 | \$0.25 |
> | \$0.50 | \$50 | \$1.25 | \$0.25 |
> | \$0.55 | \$55 | \$1.24 | \$0.25 |
> | \$0.60 | \$60 | \$1.20 | \$0.24 |
> | \$0.65 | \$65 | \$1.14 | \$0.23 |
> | \$0.70 | \$70 | \$1.05 | \$0.21 |
> | \$0.75 | \$75 | \$0.94 | \$0.19 |
> | \$0.80 | \$80 | \$0.80 | \$0.16 |
> | \$0.85 | \$85 | \$0.64 | \$0.13 |
> | \$0.90 | \$90 | \$0.45 | \$0.09 |
> | \$0.95 | \$95 | \$0.24 | \$0.05 |
> | \$0.99 | \$99 | \$0.05 | \$0.01 |
>
> The fee in USDC **peaks at 50%** probability (\$1.25 for 0.05 and \$0.25 for 0.01) and decreases symmetrically toward both extremes.

## Fee Precision

Fees are rounded to 5 decimal places. The smallest fee charged is **0.00001 USDC**. Anything smaller rounds to zero, so very small trades near the extremes may incur no fee at all.


This documentation is built and hosted on [Mintlify](https://mintlify.com), a developer documentation platform.