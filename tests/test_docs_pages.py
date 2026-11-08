"""The reference pages that restate the code by hand: the command table and the package map.

A page generated from docstrings cannot drift from the code; these two are written by hand, so a test
compares them with it. A command renamed, a package split or a name made private fails here.
"""

from __future__ import annotations

import importlib
import json
import re
from pathlib import Path

import pmlab

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
SNAPSHOT = REPO / "tests" / "api" / "public_api.json"


def _commands() -> set[str]:
    return set(json.loads(SNAPSHOT.read_text())["cli"]["commands"])


class TestCommandLinePage:
    TEXT = (DOCS / "cli.md").read_text()

    def test_every_command_has_a_row_in_the_table(self) -> None:
        rows = set(re.findall(r"^\| `([a-z-]+)` \|", self.TEXT, re.MULTILINE))
        assert rows == _commands()

    def test_no_command_that_does_not_exist_is_named(self) -> None:
        named = set(re.findall(r"`?pmlab ([a-z][a-z-]+)", self.TEXT))
        named -= {"is", "has", "puts", "command", "commands"}  # prose around the word
        assert named <= _commands(), sorted(named - _commands())

    def test_the_placeholder_is_called_one(self) -> None:
        """settle-trades settles nothing and exits 1: the page must say both (issue #43)."""
        row = next(r for r in self.TEXT.splitlines() if r.startswith("| `settle-trades`"))
        assert "placeholder" in row.lower() and "settles nothing" in row
        assert "exits 1" in row and "stderr" in row and "SettlementEngine" in row


class TestArchitecturePage:
    TEXT = (DOCS / "architecture.md").read_text()

    def _rows(self) -> list[tuple[str, list[str]]]:
        rows = []
        for line in self.TEXT.splitlines():
            match = re.match(r"^\| `(pmlab\.[a-z_.]+)` \| (.*) \|$", line)
            if match:
                rows.append((match.group(1), re.findall(r"`([^`]+)`", match.group(2))))
        return rows

    def test_every_package_of_the_library_has_a_row(self) -> None:
        packages = {
            f"pmlab.{p.name}"
            for p in (REPO / "src" / "pmlab").iterdir()
            if p.is_dir() and (p / "__init__.py").exists()
        }
        listed = {name for name, _ in self._rows()}
        assert packages - listed == set(), f"packages with no row: {sorted(packages - listed)}"
        assert listed <= packages, (
            f"rows for packages that do not exist: {sorted(listed - packages)}"
        )

    def test_every_name_in_the_map_is_public(self) -> None:
        """Each name is in `pmlab.__all__` or in the `__all__` of the package of its row."""
        assert len(self._rows()) >= 10
        not_names = {"pmlab", "track"}  # the command, and the name of an extra
        problems = []
        for package, tokens in self._rows():
            exported = set(getattr(importlib.import_module(package), "__all__", ())) | set(
                pmlab.__all__
            )
            for token in tokens:
                name = token.removesuffix("()")
                if name.startswith("pmlab.") or name in not_names or not name.isidentifier():
                    continue  # a module path or a phrase, not an exported name
                if name not in exported:
                    problems.append(f"{package}: {name}")
        assert problems == []
