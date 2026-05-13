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
