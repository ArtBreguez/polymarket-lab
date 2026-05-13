"""TDD tests for `pmlab report` CLI command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pmlab.cli.main import app

runner = CliRunner()


@pytest.fixture
def trades_json(tmp_path) -> Path:
    trades = [
        {
            "recorded_at": "2026-05-01T10:00:00+00:00",
            "city_or_segment": "Buenos Aires",
            "target_date": "2026-05-02",
            "outcome_label": "above_30",
            "direction": "yes",
            "gamma_price": 0.55,
            "edge_after_fee": 0.08,
            "flat_stake": 5.0,
            "realized_pnl": 3.64,
        },
        {
            "recorded_at": "2026-05-02T10:00:00+00:00",
            "city_or_segment": "Atlanta",
            "target_date": "2026-05-03",
            "outcome_label": "above_35",
            "direction": "yes",
            "gamma_price": 0.45,
            "edge_after_fee": 0.10,
            "flat_stake": 5.0,
            "realized_pnl": None,
        },
    ]
    path = tmp_path / "trades.json"
    path.write_text(json.dumps({"trades": trades}))
    return path


class TestReportCommand:
    def test_report_creates_html(self, tmp_path, trades_json):
        """pmlab report creates an HTML file."""
        out = tmp_path / "report.html"
        result = runner.invoke(
            app,
            [
                "report",
                "--trades",
                str(trades_json),
                "--output",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()
        assert out.suffix == ".html"

    def test_report_html_contains_segment(self, tmp_path, trades_json):
        """HTML report mentions trade segments."""
        out = tmp_path / "report.html"
        runner.invoke(app, ["report", "--trades", str(trades_json), "--output", str(out)])
        content = out.read_text()
        assert "Buenos Aires" in content

    def test_report_with_title(self, tmp_path, trades_json):
        """--title is reflected in the HTML."""
        out = tmp_path / "report.html"
        runner.invoke(
            app,
            [
                "report",
                "--trades",
                str(trades_json),
                "--output",
                str(out),
                "--title",
                "My Custom Title",
            ],
        )
        assert "My Custom Title" in out.read_text()

    def test_report_missing_trades_file(self, tmp_path):
        """Exits 1 when trades file does not exist."""
        result = runner.invoke(
            app,
            [
                "report",
                "--trades",
                str(tmp_path / "nope.json"),
                "--output",
                str(tmp_path / "r.html"),
            ],
        )
        assert result.exit_code == 1

    def test_report_default_output_path(self, tmp_path, trades_json, monkeypatch):
        """Without --output, writes pmlab_report.html in cwd."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["report", "--trades", str(trades_json)])
        assert result.exit_code == 0, result.output
        assert (tmp_path / "pmlab_report.html").exists()

    def test_report_prints_output_path(self, tmp_path, trades_json):
        """CLI prints where the report was saved."""
        out = tmp_path / "report.html"
        result = runner.invoke(
            app,
            [
                "report",
                "--trades",
                str(trades_json),
                "--output",
                str(out),
            ],
        )
        assert str(out) in result.output or "report" in result.output.lower()
