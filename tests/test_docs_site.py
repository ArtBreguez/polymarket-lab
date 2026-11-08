"""The documentation site: what is built, from what, and how it is published.

``mkdocs build --strict`` is the real check (CI runs it, and the e2e test runs it from the sdist);
these are the fast guards that run in the default suite and fail before a build would.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

import pmlab
from pmlab import _deprecation

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
SITE_URL = "https://pmlab.hiddenterps.si/"
SNAPSHOT = REPO / "tests" / "api" / "public_api.json"


def _config() -> dict[str, Any]:
    config: dict[str, Any] = yaml.safe_load((REPO / "mkdocs.yml").read_text())
    return config


def _nav_targets(nav: list[Any]) -> list[str]:
    targets: list[str] = []
    for item in nav:
        if isinstance(item, str):
            targets.append(item)
        elif isinstance(item, dict):
            for value in item.values():
                targets.extend(_nav_targets(value) if isinstance(value, list) else [value])
    return targets


def _workflow() -> dict[str, Any]:
    path = REPO / ".github" / "workflows" / "docs.yml"
    workflow: dict[str, Any] = yaml.safe_load(path.read_text())
    return workflow


def _pyproject() -> dict[str, Any]:
    return tomllib.loads((REPO / "pyproject.toml").read_text())


class TestMkdocsConfig:
    def test_it_builds_strictly_with_material_and_mkdocstrings(self) -> None:
        config = _config()
        assert config["strict"] is True
        assert config["theme"]["name"] == "material"
        plugins = [p if isinstance(p, str) else next(iter(p)) for p in config["plugins"]]
        assert {"search", "mkdocstrings"} <= set(plugins)
        assert config["site_url"] == SITE_URL
        assert config["repo_url"] == "https://github.com/ArtBreguez/polymarket-lab"

    def test_a_broken_link_or_anchor_fails_the_build_instead_of_logging_it(self) -> None:
        validation = _config()["validation"]
        for kind in ("omitted_files", "not_found", "absolute_links"):
            assert validation["nav"][kind] == "warn", kind  # `warn` + strict = a failed build
        for kind in ("not_found", "absolute_links", "unrecognized_links", "anchors"):
            assert validation["links"][kind] == "warn", kind

    def test_the_nav_lists_every_page_and_only_pages_that_exist(self) -> None:
        targets = _nav_targets(_config()["nav"])
        pages = [t for t in targets if not t.startswith("http")]
        on_disk = {str(p.relative_to(DOCS)) for p in DOCS.rglob("*.md")}
        assert set(pages) == on_disk, (
            f"in nav but missing: {sorted(set(pages) - on_disk)}; "
            f"on disk but not in nav: {sorted(on_disk - set(pages))}"
        )
        assert len(pages) == len(set(pages)), "a page is listed twice"

    def test_the_pages_the_issue_names_are_in_the_nav(self) -> None:
        pages = set(_nav_targets(_config()["nav"]))
        for name in (
            "features",
            "costs",
            "calibration",
            "drift-monitoring",
            "live-trading",
            "plugin-authoring",
            "reports",
            "tracking",
            "watchdog",
            "async-clients",
            "performance",
            "reproducibility",
            "api-stability",
            "deprecation",
        ):
            assert f"{name}.md" in pages, name

    def test_the_tutorial_notebook_is_linked_from_the_nav(self) -> None:
        # Not rendered: mkdocs-jupyter would add a Jupyter kernel stack (47 packages, ~75 MB) to
        # show the code cells of a notebook committed without outputs.
        targets = _nav_targets(_config()["nav"])
        assert any(t.endswith("notebooks/tutorial.ipynb") for t in targets)
        assert (REPO / "notebooks" / "tutorial.ipynb").exists()

    def test_the_home_page_exists(self) -> None:
        assert (DOCS / "index.md").exists()
        assert _config()["nav"][0] == {"Home": "index.md"}


class TestApiReference:
    """Generated from docstrings, for the public surface only."""

    DIRECTIVE = re.compile(r"^::: ([\w.]+)\s*$", re.MULTILINE)

    def _directives(self) -> list[str]:
        found: list[str] = []
        for page in sorted((DOCS / "reference").glob("*.md")):
            found.extend(self.DIRECTIVE.findall(page.read_text()))
        return found

    def test_every_public_name_has_an_entry(self) -> None:
        snapshot = json.loads(SNAPSHOT.read_text())
        expected = {
            f"{module}.{name}"
            for module, entries in snapshot["modules"].items()
            for name in entries
            if name not in {"__version__", "datasets"}
        }
        # `pmlab.features.add_lags` is the same object as `pmlab.add_lags`: documenting one path
        # per object is enough, so a name re-exported by a public subpackage counts via either.
        documented = set(self._directives())
        missing = {
            path
            for path in expected
            if path not in documented
            and not any(
                f"{other}.{path.rsplit('.', 1)[1]}" in documented
                for other in snapshot["modules"]
                if other != path.rsplit(".", 1)[0]
            )
        }
        assert not missing, f"public names with no docs/reference entry: {sorted(missing)}"

    def test_only_public_names_are_documented(self) -> None:
        snapshot = json.loads(SNAPSHOT.read_text())
        for path in self._directives():
            module, _, name = path.rpartition(".")
            assert not any(part.startswith("_") for part in path.split(".")), path
            assert module in snapshot["modules"], f"{path}: {module} is not a public module"
            assert name in snapshot["modules"][module], f"{path}: not in {module}.__all__"

    def test_there_is_one_entry_per_object(self) -> None:
        directives = self._directives()
        assert len(directives) == len(set(directives)), "an entry appears twice"

    def test_each_entry_has_a_docstring_to_show(self) -> None:
        import importlib

        empty = []
        for path in self._directives():
            module, _, name = path.rpartition(".")
            obj = getattr(importlib.import_module(module), name)
            if callable(obj) and not (obj.__doc__ or "").strip():
                empty.append(path)
        assert not empty, (
            f"public callables without a docstring (the reference would be bare): {empty}"
        )


class TestStabilityPages:
    def test_api_stability_names_every_public_module_and_every_command(self) -> None:
        text = (DOCS / "api-stability.md").read_text()
        snapshot = json.loads(SNAPSHOT.read_text())
        for module in snapshot["modules"]:
            assert f"`{module}`" in text, f"{module} is not described in docs/api-stability.md"
        for command in snapshot["cli"]["commands"]:
            assert f"`{command}`" in text, f"command {command} is not in docs/api-stability.md"
        for variable in snapshot["formats"]["environment_variables"]:
            assert variable in text, f"{variable} is not in docs/api-stability.md"
        assert "scripts/api_surface.py" in text
        assert "tests/api/public_api.json" in text

    def test_api_stability_lists_the_champion_and_record_keys(self) -> None:
        text = (DOCS / "api-stability.md").read_text()
        snapshot = json.loads(SNAPSHOT.read_text())
        for key in snapshot["formats"]["champion.json"]:
            assert f"`{key}`" in text, f"champion.json key {key} is not documented"
        for key in snapshot["formats"]["run_record.json"]:
            assert f"`{key}`" in text, f"run record key {key} is not documented"

    def test_the_deprecation_policy_states_the_timeline_and_the_warning(self) -> None:
        text = (DOCS / "deprecation.md").read_text()
        for needle in (
            "PmlabDeprecationWarning",
            "2.0.0",
            "error::pmlab.PmlabDeprecationWarning",
            "FutureWarning",  # the page justifies not using it
            "deprecated(",
            "warn_deprecated(",
        ):
            assert needle in text, needle

    @staticmethod
    def _table_rows() -> list[list[str]]:
        """The cells of each row of the "Currently deprecated" table, header and rule dropped."""
        text = (DOCS / "deprecation.md").read_text()
        section = text.split("## Currently deprecated", 1)[1]
        rows = [
            [cell.strip() for cell in line.strip().strip("|").split("|")]
            for line in section.splitlines()
            if line.startswith("|")
        ]
        return [row for row in rows[2:] if row]  # [name, since, removed_in, use instead]

    def test_every_deprecation_in_the_code_is_a_row_of_the_policy_table(self) -> None:
        rows = self._table_rows()
        registered = _deprecation.registered()
        assert registered, "the registry is empty: is anything deprecated?"
        for entry in registered:
            matching = [r for r in rows if r[0] == f"`{entry.name}`"]
            assert matching, f"{entry.name} is deprecated but has no row in docs/deprecation.md"
            row = matching[0]
            assert (row[1], row[2]) == (entry.since, entry.removed_in), (
                f"{entry.name}: the table says since {row[1]}, removed in {row[2]}; the code "
                f"says since {entry.since}, removed in {entry.removed_in}"
            )

    def test_the_table_lists_nothing_that_is_no_longer_deprecated(self) -> None:
        # When a shim is deleted in the next major release its row goes with it.
        in_code = {f"`{entry.name}`" for entry in _deprecation.registered()}
        stale = [row[0] for row in self._table_rows() if row[0] not in in_code]
        assert not stale, (
            f"docs/deprecation.md lists deprecations that are not in the code: {stale}"
        )


class TestDocsExtra:
    def test_the_docs_dependencies_are_an_optional_extra_not_runtime(self) -> None:
        project = _pyproject()["project"]
        extra = " ".join(project["optional-dependencies"]["docs"]).lower()
        for needed in ("mkdocs>=", "mkdocs-material", "mkdocstrings"):
            assert needed in extra, needed
        runtime = " ".join(project["dependencies"]).lower()
        assert "mkdocs" not in runtime
        assert "mkdocs" not in " ".join(project["optional-dependencies"]["dev"]).lower()

    def test_mkdocs_is_held_below_2_0(self) -> None:
        # MkDocs 2.0 drops the plugin system Material and mkdocstrings are built on.
        extra = _pyproject()["project"]["optional-dependencies"]["docs"]
        assert any(re.fullmatch(r"mkdocs>=1\.\d+,\s*<2", d.replace(" ", "")) for d in extra), extra

    def test_the_docs_site_is_linked_from_the_package_metadata_and_the_readme(self) -> None:
        urls = _pyproject()["project"]["urls"]
        assert urls["Documentation"] == SITE_URL
        assert SITE_URL in (REPO / "README.md").read_text()


class TestPagesWorkflow:
    def test_it_builds_on_push_to_main_for_docs_and_src_and_on_demand(self) -> None:
        on = _workflow()["on"]
        assert on["push"]["branches"] == ["main"]
        for path in ("docs/**", "src/**", "mkdocs.yml"):
            assert path in on["push"]["paths"], path
        assert "workflow_dispatch" in on

    def test_pull_requests_build_but_never_deploy(self) -> None:
        workflow = _workflow()
        assert "pull_request" in workflow["on"]
        jobs = workflow["jobs"]
        assert "build" in jobs and "deploy" in jobs
        assert "mkdocs build --strict" in json.dumps(jobs["build"])
        deploy = jobs["deploy"]
        assert deploy["needs"] == "build"
        assert "pull_request" in deploy["if"] and "!=" in deploy["if"]
        assert deploy["environment"]["name"] == "github-pages"

    def test_only_the_deploy_job_can_write_pages(self) -> None:
        workflow = _workflow()
        assert workflow["permissions"] == {"contents": "read"}
        assert workflow["jobs"]["deploy"]["permissions"] == {"pages": "write", "id-token": "write"}
        assert "pages" not in workflow["jobs"]["build"].get("permissions", {})

    def test_every_action_is_pinned_to_a_full_commit_sha(self) -> None:
        text = (REPO / ".github" / "workflows" / "docs.yml").read_text()
        uses = re.findall(r"uses:\s*(\S+)", text)
        assert uses, "no actions used?"
        for ref in uses:
            assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", ref), f"not pinned by SHA: {ref}"
        names = {ref.split("@")[0] for ref in uses}
        assert {"actions/upload-pages-artifact", "actions/deploy-pages"} <= names

    def test_deploys_queue_instead_of_cancelling_each_other(self) -> None:
        # On the deploy job, not the workflow: a pull-request build must not queue behind a deploy.
        workflow = _workflow()
        concurrency = workflow["jobs"]["deploy"]["concurrency"]
        assert concurrency["group"] == "pages"
        assert concurrency["cancel-in-progress"] is False
        assert "concurrency" not in workflow and "concurrency" not in workflow["jobs"]["build"]


class TestVersionIsOneNumber:
    def test_package_pyproject_and_changelog_agree(self) -> None:
        version = _pyproject()["project"]["version"]
        assert pmlab.__version__ == version
        first = re.search(r"^## \[(\d+\.\d+\.\d+)\]", (REPO / "CHANGELOG.md").read_text(), re.M)
        assert first is not None and first.group(1) == version

    @pytest.mark.parametrize("marker", ["Documentation", "Homepage", "Repository", "Issues"])
    def test_the_project_urls_are_there(self, marker: str) -> None:
        assert marker in _pyproject()["project"]["urls"]


class TestDeprecationsShowInTheReference:
    """The API reference is built from source; ``@deprecated`` adds its notice at run time.

    ``scripts/griffe_deprecated.py`` closes that gap by reading the decorator's arguments. These
    need Griffe (the ``docs`` extra); CI installs it, a bare ``dev`` checkout skips them.
    """

    @pytest.fixture(scope="class")
    def extension(self) -> Any:
        pytest.importorskip("griffe")
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location(
            "griffe_deprecated", REPO / "scripts" / "griffe_deprecated.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules["griffe_deprecated"] = module
        spec.loader.exec_module(module)
        return module

    @pytest.mark.parametrize(
        ("alternative", "reason"),
        [(None, None), ("new_thing", None), (None, "it is slow"), ("`b`", "biased, and slow")],
    )
    @pytest.mark.parametrize(
        "doc",
        [None, "", "  ", "Summary.", "Summary.\n\nBody.", "    Indented.\n\n    Body.\n    More."],
    )
    def test_the_notice_is_the_one_the_runtime_writes(
        self, extension: Any, doc: str | None, alternative: str | None, reason: str | None
    ) -> None:
        entry = _deprecation.Deprecation("x", "1.2.0", "2.0.0", alternative, reason)
        text = extension.notice("1.2.0", "2.0.0", alternative, reason)
        assert text == entry.notice()
        assert extension.with_notice(doc, text) == _deprecation._with_notice(doc, entry)

    def test_every_registered_deprecation_is_in_the_static_docstring(self, extension: Any) -> None:
        import griffe

        registered = _deprecation.registered()
        assert registered
        extensions = griffe.Extensions(extension.DeprecatedNotice())
        package = griffe.load(
            "pmlab", search_paths=[REPO / "src"], extensions=extensions, allow_inspection=False
        )
        for entry in registered:
            obj = package[f"{entry.module.removeprefix('pmlab.')}.{entry.name}"]
            assert obj.docstring is not None
            assert f"Deprecated since {entry.since}" in obj.docstring.value, entry.name
            assert f"removed in {entry.removed_in}" in obj.docstring.value
