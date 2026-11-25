"""`promote-champion --plugin` names a family; a name no plugin registers is flagged (#68).

It used to be recorded in ``champion.json`` without a word, so a typo (``weather_tmx``) produced a
champion for a family nothing resolves. It is **not refused**: the docs and several flows use a
family label for a plugin that is installed on another machine (``--plugin my_family``,
``politics``), and refusing would change documented behaviour of a frozen command. The command
says so on stderr and records the name as given.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pmlab.cli.main import app
from tests.cli.test_cli_costs import _gate, _model

runner = CliRunner()


def _promote(tmp_path: Path, family: str):
    return runner.invoke(
        app,
        [
            "promote-champion",
            str(_model(tmp_path)),
            "--gate-path",
            str(_gate(tmp_path)),
            "--plugin",
            family,
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )


def _recorded_family(tmp_path: Path) -> str:
    return json.loads((tmp_path / "out" / "champion.json").read_text())["plugin_family"]


class TestAnUnregisteredFamilyIsFlagged:
    def test_a_typo_warns_on_stderr_names_the_registered_families_and_still_promotes(
        self, tmp_path
    ) -> None:
        result = _promote(tmp_path, "weather_tmx")
        assert result.exit_code == 0, result.output
        assert "warning:" in result.stderr and "'weather_tmx'" in result.stderr
        assert "weather_tmax" in result.stderr and "sports_f1" in result.stderr
        assert "warning" not in result.stdout  # stdout stays what it was
        assert _recorded_family(tmp_path) == "weather_tmx"  # recorded as given

    @pytest.mark.parametrize("family", ["weather_tmax", "sports_f1"])
    def test_a_built_in_family_does_not_warn(self, tmp_path, family) -> None:
        result = _promote(tmp_path, family)
        assert result.exit_code == 0, result.output
        assert "warning" not in result.output.lower()
        assert _recorded_family(tmp_path) == family

    def test_a_family_an_installed_plugin_registers_does_not_warn(self, tmp_path, monkeypatch):
        import pmlab.plugins.registry as registry

        monkeypatch.setattr(
            registry, "available_families", lambda: ["my_family", "sports_f1", "weather_tmax"]
        )
        result = _promote(tmp_path, "my_family")
        assert result.exit_code == 0, result.output
        assert "warning" not in result.output.lower()

    def test_the_warning_does_not_hide_a_refusal(self, tmp_path) -> None:
        # a NO_GO gate is still the error, and it is the only thing said about the run
        gate = tmp_path / "gate.json"
        gate.write_text(
            json.dumps(
                {
                    "decision": "NO_GO",
                    "segment_results": [],
                    "aggregate_pnl": -1.0,
                    "aggregate_trades": 1,
                }
            )
        )
        result = runner.invoke(
            app,
            ["promote-champion", str(_model(tmp_path)), "--gate-path", str(gate), "--plugin", "nope",
             "--output-dir", str(tmp_path / "out")],
        )  # fmt: skip
        assert result.exit_code == 1 and "NO_GO" in result.output
        assert not (tmp_path / "out").exists()
