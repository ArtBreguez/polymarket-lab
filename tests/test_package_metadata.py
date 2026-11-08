"""What a user sees before importing anything: the PyPI metadata, the changelog, the repo files.

Each of these was wrong in the 1.0.0 repository without any test noticing (a changelog missing three
published releases, a release dated before the one it follows, a CI workflow with its own lower
coverage gate). They are cheap to hold.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
BASE = "https://github.com/ArtBreguez/polymarket-lab"
# Merged but never tagged or published on PyPI: they carry a note and no compare link.
NEVER_RELEASED = {"0.5.0", "0.6.1"}
NO_TAG = {"0.1.0"}  # published to PyPI before tags were made


def _project() -> dict[str, Any]:
    data: dict[str, Any] = tomllib.loads((REPO / "pyproject.toml").read_text())
    project: dict[str, Any] = data["project"]
    return project


def _versions(text: str) -> list[tuple[str, str]]:
    return re.findall(r"^## \[(\d+\.\d+\.\d+)\] — (\S+)$", text, re.MULTILINE)


def _key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


class TestPyPIMetadata:
    def test_a_released_typed_library(self) -> None:
        classifiers = _project()["classifiers"]
        assert "Development Status :: 5 - Production/Stable" in classifiers
        assert "Typing :: Typed" in classifiers
        assert (REPO / "src" / "pmlab" / "py.typed").exists()  # the marker the classifier promises

    def test_the_license_classifier_matches_the_license_file(self) -> None:
        assert "License :: OSI Approved :: MIT License" in _project()["classifiers"]
        assert "MIT License" in (REPO / "LICENSE").read_text()

    def test_the_python_versions_declared_are_the_ones_ci_runs(self) -> None:
        declared = {
            c.rsplit(" :: ", 1)[1]
            for c in _project()["classifiers"]
            if re.fullmatch(r"Programming Language :: Python :: 3\.\d+", c)
        }
        ci = yaml.safe_load((REPO / ".github" / "workflows" / "ci.yml").read_text())
        tested = {
            str(step["with"]["python-version"])
            for job in ci["jobs"].values()
            for step in job["steps"]
            if "python-version" in step.get("with", {})
        }
        assert declared == tested, "list a Python version only when CI runs it, and vice versa"
        minimum = re.search(r">=\s*(3\.\d+)", _project()["requires-python"])
        assert minimum is not None and minimum.group(1) in declared

    def test_the_description_and_keywords_say_what_it_is(self) -> None:
        project = _project()
        assert "Polymarket" in project["description"] and len(project["description"]) < 200
        assert len(set(project["keywords"])) == len(project["keywords"])
        assert {"polymarket", "prediction-markets"} <= set(project["keywords"])
        assert project["readme"] == "README.md"

    def test_the_security_policy_is_linked_and_exists(self) -> None:
        assert _project()["urls"]["Security"].endswith("/SECURITY.md")
        assert (REPO / "SECURITY.md").exists()


class TestChangelog:
    TEXT = (REPO / "CHANGELOG.md").read_text()

    def test_versions_are_listed_newest_first_without_a_gap_in_dates(self) -> None:
        versions = _versions(self.TEXT)
        assert len(versions) > 20
        assert [v for v, _ in versions] == sorted((v for v, _ in versions), key=_key, reverse=True)
        dates = [d for _, d in versions]
        assert all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", d) for d in dates)
        assert dates == sorted(dates, reverse=True), "a release dated before the one it follows"

    def test_every_tagged_release_is_in_the_changelog(self) -> None:
        """0.8.0, 0.8.1 and 0.8.2 were published for months without an entry."""
        listed = {v for v, _ in _versions(self.TEXT)}
        released = {
            n
            for n in re.findall(
                rf"^\[(\d+\.\d+\.\d+)\]: {re.escape(BASE)}", self.TEXT, re.MULTILINE
            )
        }
        assert released <= listed
        assert {"0.8.0", "0.8.1", "0.8.2"} <= listed

    def test_every_heading_links_to_its_tag_or_says_it_was_never_released(self) -> None:
        defined = dict(re.findall(r"^\[([^\]]+)\]: (\S+)$", self.TEXT, re.MULTILINE))
        sections = re.split(r"^## \[", self.TEXT, flags=re.MULTILINE)[1:]
        for section in sections:
            name = section.split("]", 1)[0]
            if name == "Unreleased":
                assert defined[name].endswith("...HEAD")
            elif name in NEVER_RELEASED:
                assert "not released on its own" in section and name not in defined
            elif name in NO_TAG:
                assert name not in defined
            else:
                assert name in defined, f"[{name}] has no link definition at the end of the file"
                assert defined[name].endswith(f"v{name}")

    def test_an_unreleased_section_sits_above_the_newest_release(self) -> None:
        headings = re.findall(r"^## \[([^\]]+)\]", self.TEXT, re.MULTILINE)
        assert headings[0] == "Unreleased"


class TestRepositoryFiles:
    def test_dependabot_covers_the_lockfile_and_the_actions(self) -> None:
        config = yaml.safe_load((REPO / ".github" / "dependabot.yml").read_text())
        ecosystems = {u["package-ecosystem"] for u in config["updates"]}
        assert ecosystems == {"uv", "github-actions"}
        assert (REPO / "uv.lock").exists()

    def test_the_issue_and_pr_templates_parse(self) -> None:
        for name in ("bug_report.yml", "feature_request.yml"):
            form = yaml.safe_load((REPO / ".github" / "ISSUE_TEMPLATE" / name).read_text())
            assert form["name"] and form["body"]
        assert (REPO / ".github" / "pull_request_template.md").read_text().strip()

    def test_nothing_generated_is_tracked(self) -> None:
        ignored = (REPO / ".gitignore").read_text()
        for pattern in ("__pycache__/", "dist/", ".venv/", "coverage.xml", "artifacts/", "site/"):
            assert pattern in ignored, pattern
