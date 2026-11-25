"""Tests for CLI commands."""

from __future__ import annotations

import re

from typer.testing import CliRunner

from pmlab.cli.main import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "pmlab" in result.output


def test_status_no_champion(tmp_path):
    result = runner.invoke(app, ["status", "--artifacts-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "champion" in result.output.lower() or "No champion" in result.output


def test_backtest_missing_panel(tmp_path):
    result = runner.invoke(
        app,
        [
            "backtest",
            "--plugin",
            "test",
            "--panel",
            str(tmp_path / "missing.parquet"),
        ],
    )
    assert result.exit_code == 1


def test_backtest_low_stride():
    result = runner.invoke(
        app,
        [
            "backtest",
            "--plugin",
            "test",
            "--panel",
            "panel.parquet",
            "--stride",
            "3",
        ],
    )
    assert result.exit_code == 1
    assert "10" in result.output


def test_promote_champion_missing_files(tmp_path):
    result = runner.invoke(
        app,
        [
            "promote-champion",
            str(tmp_path / "model.pkl"),
            "--gate-path",
            str(tmp_path / "gate.json"),
            "--plugin",
            "test",
        ],
    )
    assert result.exit_code == 1


def test_scan_markets_unknown_plugin_errors():
    """An unknown --plugin exits 1 and lists available plugins (no raw dump)."""
    result = runner.invoke(app, ["scan-markets", "--plugin", "banana"])
    assert result.exit_code == 1
    assert "banana" in result.output
    assert "weather_tmax" in result.output  # available list shown


def test_scan_markets_delegates_to_plugin(monkeypatch):
    """Regression: scan-markets must route through the plugin's discover_markets
    (family-filtered), not dump the raw Gamma feed. Guards the bug where a
    weather scan showed unrelated political markets.
    """
    from pmlab.core.market_spec import MarketSpec, OutcomeBin
    from pmlab.plugins import registry as registry_mod

    spec = MarketSpec(
        market_id="tmax-1",
        slug="tmax-nyc",
        question="Highest temperature in NYC on May 10?",
        outcome_bins=[OutcomeBin(label="30°C")],
        close_time="2026-05-10T20:00:00Z",
        market_family="range",
        tags=["weather"],
    )

    class _FakePlugin:
        def __init__(self) -> None:
            self.called_with: dict | None = None

        def discover_markets(self, **kwargs):
            self.called_with = kwargs
            return [spec]

    fake = _FakePlugin()
    monkeypatch.setattr(registry_mod, "build_plugin", lambda family, **kw: fake)

    result = runner.invoke(app, ["scan-markets", "--plugin", "weather_tmax", "--limit", "50"])
    assert result.exit_code == 0
    assert "Highest temperature in NYC" in result.output
    assert fake.called_with is not None
    assert fake.called_with.get("limit") == 50


def test_scan_markets_reads_the_weather_tag_and_the_per_bin_markets(monkeypatch):
    """The real plugin through the command, against markets recorded from live Gamma: one row per
    city and day, read from the Daily Temperature tag (#86, #87). ``--limit`` is the page size."""
    import json
    from pathlib import Path

    import httpx
    import respx

    recorded = json.loads(
        (Path(__file__).parent / "markets" / "fixtures" / "per_bin_markets.json").read_text()
    )["markets"]
    seen: list[dict[str, str]] = []

    def keyset(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.url.params))
        return httpx.Response(200, json={"markets": recorded})

    monkeypatch.setenv("PMLAB_GAMMA_API_BASE", "https://mock-gamma.test")
    with respx.mock:
        respx.get("https://mock-gamma.test/markets/keyset").mock(side_effect=keyset)
        result = runner.invoke(app, ["scan-markets", "--plugin", "weather_tmax"])
    assert result.exit_code == 0, result.output
    assert "4 'weather_tmax' markets found" in result.output
    assert "Highest temperature in Atlanta on November 26?" in result.output
    assert seen and all(r["tag_id"] == "103040" for r in seen)


def test_scan_markets_limit_is_a_page_size_and_never_multiplies_a_whole_tag_read(monkeypatch):
    """Round 1 of #85 (MINOR-2): ``--limit 1`` on the weather tag (2,633 markets on live Gamma)
    was 1000 requests and an error; the tag is read in full pages whatever the limit is."""
    import json
    from pathlib import Path

    import httpx
    import respx

    recorded = json.loads(
        (Path(__file__).parent / "markets" / "fixtures" / "per_bin_markets.json").read_text()
    )["markets"]
    seen: list[dict[str, str]] = []

    def keyset(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        seen.append(params)
        start = int(params["after_cursor"].removeprefix("c")) if "after_cursor" in params else 0
        size = int(params["limit"])
        body: dict[str, object] = {"markets": recorded[start : start + size]}
        if start + size < len(recorded):
            body["next_cursor"] = f"c{start + size}"
        return httpx.Response(200, json=body)

    monkeypatch.setenv("PMLAB_GAMMA_API_BASE", "https://mock-gamma.test")
    with respx.mock:
        respx.get("https://mock-gamma.test/markets/keyset").mock(side_effect=keyset)
        result = runner.invoke(app, ["scan-markets", "--plugin", "weather_tmax", "--limit", "1"])
    assert result.exit_code == 0, result.output
    assert "4 'weather_tmax' markets found" in result.output
    assert len(seen) == 1 and seen[0]["limit"] == "100", seen  # 32 markets: one full page


def test_scan_markets_limit_help_says_it_is_a_page_size():
    """The help said "Max markets to fetch", which it never was (it is the page size)."""
    result = runner.invoke(app, ["scan-markets", "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    text = " ".join(re.sub(r"[│╭╮╰╯─]", " ", result.output).split())  # the help table's frame
    assert "Max markets to fetch" not in text
    assert "page size" in text.lower() and "not a number of markets" in text
