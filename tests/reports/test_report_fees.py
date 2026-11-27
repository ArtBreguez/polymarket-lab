"""The report shows what the fees cost: their total, and what share of the stake they are (#72).

A fee that is 2.5% of a stake is a different strategy from one that is 0.3%, and a PnL figure alone
does not say which one produced it. The totals are over the settled trades, like the PnL and the ROI
beside them, and only over trades that recorded a ``fee_paid`` (a log from before 1.5.0 may not).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pmlab.reports.html_report import generate_report


def _trade(pnl: float | None, fee: float | None, stake: float = 5.0) -> dict[str, Any]:
    trade: dict[str, Any] = {
        "recorded_at": "2026-05-01T10:00:00+00:00",
        "city_or_segment": "Atlanta",
        "target_date": "2026-05-02",
        "outcome_label": "above_30",
        "direction": "yes",
        "gamma_price": 0.55,
        "edge_after_fee": 0.08,
        "flat_stake": stake,
        "realized_pnl": pnl,
    }
    if fee is not None:
        trade["fee_paid"] = fee
    return trade


def _cards(html: str) -> dict[str, str]:
    return dict(re.findall(r"card-label'>([^<]+)</div><div class='card-value[^']*'>([^<]+)<", html))


def _report(tmp_path: Path, trades: list[dict[str, Any]]) -> str:
    return generate_report(trades, tmp_path / "r.html").read_text(encoding="utf-8")


class TestFeeCards:
    def test_total_fees_and_their_share_of_the_stake(self, tmp_path: Path) -> None:
        html = _report(tmp_path, [_trade(3.0, 0.25), _trade(-5.0, 0.05)])
        cards = _cards(html)
        assert cards["Total Fees"] == "$0.30"
        assert cards["Fees % of Stake"] == "3.00%"

    def test_an_open_trades_fee_is_not_in_the_settled_totals(self, tmp_path: Path) -> None:
        html = _report(tmp_path, [_trade(3.0, 0.25), _trade(None, 9.0)])
        cards = _cards(html)
        assert cards["Total Fees"] == "$0.25"
        assert cards["Fees % of Stake"] == "5.00%"

    def test_stakes_of_different_sizes_are_weighed_by_their_stake(self, tmp_path: Path) -> None:
        html = _report(tmp_path, [_trade(1.0, 0.10, stake=1.0), _trade(1.0, 0.50, stake=9.0)])
        assert _cards(html)["Fees % of Stake"] == "6.00%"

    def test_a_log_with_no_recorded_fee_says_so_instead_of_a_zero(self, tmp_path: Path) -> None:
        html = _report(tmp_path, [_trade(3.0, None), _trade(-5.0, None)])
        cards = _cards(html)
        assert cards["Total Fees"] == "n/a"
        assert cards["Fees % of Stake"] == "n/a"

    def test_only_the_trades_that_recorded_a_fee_count(self, tmp_path: Path) -> None:
        # the stake of the trade without a fee is not in the denominator
        html = _report(tmp_path, [_trade(3.0, 0.25), _trade(-5.0, None, stake=100.0)])
        cards = _cards(html)
        assert cards["Total Fees"] == "$0.25"
        assert cards["Fees % of Stake"] == "5.00%"

    def test_an_empty_report_has_no_fee_figures_and_does_not_fail(self, tmp_path: Path) -> None:
        cards = _cards(_report(tmp_path, []))
        assert cards["Total Fees"] == "n/a"

    def test_a_malformed_fee_is_left_out_not_a_crash(self, tmp_path: Path) -> None:
        trades = [_trade(3.0, 0.25), _trade(1.0, None)]
        trades[1]["fee_paid"] = "oops"
        assert _cards(_report(tmp_path, trades))["Total Fees"] == "$0.25"


class TestTradeLogShowsTheFee:
    def test_the_log_has_a_fee_column(self, tmp_path: Path) -> None:
        html = _report(tmp_path, [_trade(3.0, 0.25)])
        assert "<th>Fee</th>" in html
        assert "$0.25" in html.split("Trade Log")[1]

    def test_a_trade_with_no_fee_on_record_shows_a_dash(self, tmp_path: Path) -> None:
        html = _report(tmp_path, [_trade(3.0, None)])
        assert "<td>&mdash;</td>" in html.split("Trade Log")[1]
