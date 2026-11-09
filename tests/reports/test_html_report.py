"""Tests for HTML report generator."""

from __future__ import annotations

from pathlib import Path

import pytest

from pmlab.reports.html_report import generate_report


@pytest.fixture
def sample_trades():
    return [
        {
            "recorded_at": "2025-05-01T10:00:00+00:00",
            "city_or_segment": "Buenos Aires",
            "target_date": "2025-05-02",
            "outcome_label": "above_30",
            "direction": "yes",
            "gamma_price": 0.55,
            "edge_after_fee": 0.08,
            "flat_stake": 5.0,
            "realized_pnl": 3.64,
        },
        {
            "recorded_at": "2025-05-02T10:00:00+00:00",
            "city_or_segment": "Buenos Aires",
            "target_date": "2025-05-03",
            "outcome_label": "above_30",
            "direction": "yes",
            "gamma_price": 0.60,
            "edge_after_fee": 0.05,
            "flat_stake": 5.0,
            "realized_pnl": -3.0,
        },
        {
            "recorded_at": "2025-05-03T10:00:00+00:00",
            "city_or_segment": "Atlanta",
            "target_date": "2025-05-04",
            "outcome_label": "above_35",
            "direction": "yes",
            "gamma_price": 0.45,
            "edge_after_fee": 0.10,
            "flat_stake": 5.0,
            "realized_pnl": None,
        },
    ]


class TestGenerateReport:
    def test_creates_file(self, tmp_path, sample_trades):
        out = generate_report(sample_trades, output_path=tmp_path / "report.html")
        assert out.exists() and out.suffix == ".html"

    def test_html_contains_title(self, tmp_path, sample_trades):
        out = generate_report(sample_trades, output_path=tmp_path / "r.html", title="Test Report")
        assert "Test Report" in out.read_text()

    def test_html_contains_segments(self, tmp_path, sample_trades):
        out = generate_report(sample_trades, output_path=tmp_path / "r.html")
        content = out.read_text()
        assert "Buenos Aires" in content and "Atlanta" in content

    def test_with_brier_score(self, tmp_path, sample_trades):
        out = generate_report(sample_trades, output_path=tmp_path / "r.html", brier_score=0.23)
        assert "0.2300" in out.read_text()

    def test_empty_trades(self, tmp_path):
        out = generate_report([], output_path=tmp_path / "empty.html")
        assert out.exists()

    def test_returns_path(self, tmp_path, sample_trades):
        result = generate_report(sample_trades, output_path=tmp_path / "r.html")
        assert isinstance(result, Path)


class TestEscapingAndOutputDir:
    """Values from market data are text, never markup (#55)."""

    HOSTILE = "<img src=x onerror=alert(1)>"

    def _trade(self, **over):
        t = {
            "recorded_at": "2026-05-01T10:00:00+00:00",
            "city_or_segment": "Madrid",
            "target_date": "2026-05-02",
            "outcome_label": "above_30",
            "direction": "yes",
            "gamma_price": 0.35,
            "edge_after_fee": 0.08,
            "flat_stake": 5.0,
            "realized_pnl": 9.28,
        }
        t.update(over)
        return t

    def test_title_is_escaped(self, tmp_path):
        out = generate_report(
            [self._trade()], tmp_path / "r.html", title="<script>alert(1)</script>"
        )
        html = out.read_text()
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html

    @pytest.mark.parametrize(
        "key", ["city_or_segment", "outcome_label", "direction", "target_date"]
    )
    def test_trade_fields_are_escaped(self, tmp_path, key):
        out = generate_report([self._trade(**{key: self.HOSTILE})], tmp_path / "r.html")
        html = out.read_text()
        assert self.HOSTILE not in html
        assert "&lt;img src=x onerror=alert(1)&gt;" in html

    def test_open_trades_are_escaped_too(self, tmp_path):
        out = generate_report(
            [self._trade(city_or_segment=self.HOSTILE, realized_pnl=None)], tmp_path / "r.html"
        )
        assert self.HOSTILE not in out.read_text()

    def test_a_plain_report_is_unchanged_by_escaping(self, tmp_path):
        out = generate_report([self._trade()], tmp_path / "r.html", title="May 2026 — Buenos Aires")
        html = out.read_text()
        assert "May 2026 — Buenos Aires" in html and "<td>Madrid</td>" in html

    def test_missing_output_directories_are_created(self, tmp_path):
        """docs/reports.md writes to reports/may_2026.html in a fresh directory."""
        out = generate_report([self._trade()], tmp_path / "reports" / "nested" / "may.html")
        assert out.exists() and out.parent.name == "nested"
