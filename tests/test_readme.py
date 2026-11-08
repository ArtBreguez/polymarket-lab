"""The README is the first thing a user reads, and (as the long description) the PyPI page.

So it must not rot. These tests hold the properties a README loses quietly: a counter that was true
once, a link that PyPI cannot resolve, a command that was renamed, a code block that stopped
running. The same blocks are run again, against the built wheel, by ``pytest -m e2e``.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
SNAPSHOT = REPO / "tests" / "api" / "public_api.json"
WALKTHROUGH = REPO / "tests" / "e2e" / "run_doc_walkthrough.py"
REPO_URL = "https://github.com/ArtBreguez/polymarket-lab"
SITE_URL = "https://artbreguez.github.io/polymarket-lab/"

TEXT = README.read_text()
FENCE = re.compile(r"^```(\w*)[^\n]*\n(.*?)^```", re.DOTALL | re.MULTILINE)


def _prose() -> str:
    """The README without its fenced code blocks."""
    return FENCE.sub("", TEXT)


def _links() -> list[str]:
    markdown = re.findall(r"\]\(([^)\s]+)\)", _prose())
    html = re.findall(r'(?:href|src)="([^"]+)"', _prose())
    return markdown + html


def _commands() -> dict[str, set[str]]:
    """Command name -> every flag it accepts, from the public API snapshot."""
    cli = json.loads(SNAPSHOT.read_text())["cli"]["commands"]
    return {
        name: {flag for param in spec["params"] for flag in param["flags"] if flag.startswith("-")}
        for name, spec in cli.items()
    }


def _required_options() -> dict[str, list[set[str]]]:
    """Command name -> for each required option, the flags that can spell it."""
    cli = json.loads(SNAPSHOT.read_text())["cli"]["commands"]
    return {
        name: [
            set(param["flags"])
            for param in spec["params"]
            if param["required"] and param["kind"] == "option"
        ]
        for name, spec in cli.items()
    }


class TestNothingThatCanGoStale:
    def test_no_static_badge_carries_a_number(self) -> None:
        """`tests-3109 passing` and `coverage-96%` were typed in once and are false today."""
        static = re.findall(r"img\.shields\.io/badge/([^)\s\"]+)", TEXT)
        decoded = [re.sub(r"%[0-9A-Fa-f]{2}", " ", b) for b in static]  # %2F is not a number
        assert [b for b in decoded if re.search(r"\d", b)] == []

    def test_the_badges_that_remain_are_computed_from_the_project(self) -> None:
        assert "img.shields.io/pypi/v/pmlab" in TEXT  # the released version
        assert (
            "img.shields.io/pypi/pyversions/pmlab" in TEXT
        )  # the Python versions the wheel declares
        assert "actions/workflows/ci.yml/badge.svg" in TEXT  # whether CI is green

    def test_no_test_count_or_coverage_figure_in_the_prose(self) -> None:
        prose = _prose()
        assert not re.search(r"\b\d[\d,]*\s+(?:tests?|passing)\b", prose, re.IGNORECASE)
        assert not re.search(r"coverage[^\n]{0,40}\d+(?:\.\d+)?\s*%", prose, re.IGNORECASE)
        assert not re.search(r"\d+(?:\.\d+)?\s*%[^\n]{0,20}coverage", prose, re.IGNORECASE)

    def test_it_does_not_name_a_release_it_would_have_to_update(self) -> None:
        version = re.search(r'^version = "([^"]+)"', (REPO / "pyproject.toml").read_text(), re.M)
        assert version is not None
        assert not re.search(rf"(?<![\w.-]){re.escape(version.group(1))}(?![\w.-])", TEXT)


class TestLinksSurviveOnPyPI:
    """PyPI renders the README away from the repository: a relative link goes nowhere."""

    def test_every_link_and_image_is_absolute_or_an_in_page_anchor(self) -> None:
        relative = [link for link in _links() if not link.startswith(("https://", "#"))]
        assert relative == [], f"relative links break on PyPI: {relative}"

    def test_in_page_anchors_name_a_heading(self) -> None:
        slugs = {
            re.sub(r"[^\w\- ]", "", h.lower()).strip().replace(" ", "-")
            for h in re.findall(r"^#+\s+(.+)$", _prose(), re.MULTILINE)
        }
        missing = [a for a in _links() if a.startswith("#") and a[1:] not in slugs]
        assert missing == []

    def test_links_into_the_repository_name_files_that_exist(self) -> None:
        paths = re.findall(rf"{re.escape(REPO_URL)}/(?:blob|tree)/main/([^)\s#]+)", TEXT)
        assert paths, "the README links to nothing in the repository: is the pattern broken?"
        assert [p for p in paths if not (REPO / p).exists()] == []

    def test_links_into_the_docs_site_name_pages_that_exist(self) -> None:
        pages = re.findall(rf"{re.escape(SITE_URL)}([\w/-]*)", TEXT)
        assert len(pages) > 10, "the README links to few docs pages: is the pattern broken?"
        missing = []
        for page in pages:
            slug = page.strip("/")
            candidates = [REPO / "docs" / f"{slug}.md", REPO / "docs" / slug / "index.md"]
            if slug and not any(c.exists() for c in candidates):
                missing.append(page)
        assert missing == []

    def test_the_docs_site_is_linked(self) -> None:
        assert SITE_URL in TEXT  # also asserted from the package metadata in test_docs_site


class TestTheCommandLineItShowsExists:
    def test_every_pmlab_command_in_a_code_span_or_block_is_real(self) -> None:
        commands = _commands()
        shown = re.findall(r"(?:^|[`$] ?)pmlab ([a-z][a-z-]+)", TEXT, re.MULTILINE)
        assert shown, "no pmlab command found in the README: is the pattern broken?"
        unknown = sorted({c for c in shown if c not in commands})
        assert unknown == [], f"not commands of this release: {unknown}"

    def test_every_command_is_in_the_README(self) -> None:
        missing = [c for c in _commands() if f"pmlab {c}" not in TEXT and f"`{c}`" not in TEXT]
        assert missing == [], f"commands the README does not mention: {missing}"

    def test_every_option_shown_exists_and_every_required_one_is_shown(self) -> None:
        commands = _commands()
        required = _required_options()
        checked = 0
        for kind, block in FENCE.findall(TEXT):
            if kind not in {"bash", "shell"}:
                continue
            for line in block.replace("\\\n", " ").splitlines():
                match = re.match(r"\s*pmlab ([a-z][a-z-]+)(.*)", line)
                if not match:
                    continue
                name, rest = match.groups()
                given = set(re.findall(r"(?<!\S)(--?[a-z][\w-]*)", rest.split("#")[0]))
                for flag in sorted(given):
                    assert flag in commands[name], f"`pmlab {name}` has no option {flag}"
                    checked += 1
                for spellings in required[name]:
                    assert spellings & given, f"`pmlab {name}` is shown without {sorted(spellings)}"
        assert checked >= 5


class TestTheCodeRuns:
    def test_the_python_and_bash_blocks_run_as_written(self, tmp_path: Path) -> None:
        """Every ``python`` and ``bash`` block, in order, from an empty directory.

        The quickstart writes ``panel.parquet``; the command-line block reads it. A neutered block
        (one that stops printing what the text promises) fails below, not only a crashing one.
        """
        proc = subprocess.run(
            [sys.executable, "-I", str(WALKTHROUGH), str(README), "--cwd", str(tmp_path / "run"),
             "--session"],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )  # fmt: skip
        assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
        out = proc.stdout
        assert "refused: Cannot publish champion with NO_GO gate" in out
        assert "REPRODUCED: 5 metrics match" in out
        assert re.search(r"\d+ trades \| PnL -?[\d.]+ \| (GO|NO_GO)", out)
        assert (tmp_path / "run" / "panel.parquet").exists()
        assert (tmp_path / "run" / "metrics.json").exists()

    def test_every_executable_block_is_python_or_bash(self) -> None:
        """A `python` fence is run by the test above; anything illustrative uses another tag."""
        kinds = {kind for kind, _ in FENCE.findall(TEXT)}
        assert kinds <= {"python", "bash", "shell", "text"}, kinds


@pytest.mark.parametrize("name", ["SECURITY.md", "CONTRIBUTING.md", "ROADMAP.md", "CHANGELOG.md"])
def test_the_files_the_project_row_links_to_exist(name: str) -> None:
    assert (REPO / name).exists()
