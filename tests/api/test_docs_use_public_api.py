"""The documentation teaches the stable import paths and nothing else.

A page that says ``from pmlab.core.market_spec import MarketSpec`` makes an internal module path a
de-facto API. After 1.0 a user who copies an example must be copying a name that is frozen: every
``from pmlab... import ...`` in the docs, the README, the examples and the tutorial notebook has
to resolve to a public name (``pmlab.__all__`` or the ``__all__`` of a public subpackage).
"""

from __future__ import annotations

import ast
import importlib
import json
import re
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PUBLIC_MODULES = (
    "pmlab",
    "pmlab.data",
    "pmlab.datasets",
    "pmlab.features",
    "pmlab.monitoring",
    "pmlab.reports",
    "pmlab.reproducibility",
    "pmlab.tracking",
)
_FENCE = re.compile(r"```(?:python|py)\n(.*?)```", re.DOTALL)
_FROM_IMPORT = re.compile(r"^\s*from\s+(pmlab[\w.]*)\s+import\s+([^#\n]+)", re.MULTILINE)


def _sources() -> Iterator[tuple[str, str]]:
    """(where, python source) for every place the project shows pmlab code."""
    pages = [*sorted((REPO / "docs").glob("*.md")), REPO / "README.md", REPO / "CONTRIBUTING.md"]
    for page in pages:
        text = page.read_text()
        for match in _FENCE.finditer(text):
            line = text[: match.start()].count("\n") + 2
            yield f"{page.relative_to(REPO)}:{line}", match.group(1)
    for script in sorted((REPO / "examples").rglob("*.py")):
        yield str(script.relative_to(REPO)), script.read_text()
    notebook = json.loads((REPO / "notebooks" / "tutorial.ipynb").read_text())
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            yield f"notebooks/tutorial.ipynb[cell {index}]", "".join(cell["source"])


def _imports(source: str) -> Iterator[tuple[str, str]]:
    """(module, name) for each ``from pmlab... import name`` and ``import pmlab...`` (name '')."""
    try:
        tree = ast.parse(source)
    except SyntaxError:  # an illustrative snippet: fall back to the import lines themselves
        for module, names in _FROM_IMPORT.findall(source):
            for name in re.split(r"[,\s()]+", names):
                if name and name != "as":
                    yield module, name
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("pmlab"):
            for alias in node.names:
                yield node.module, alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("pmlab"):
                    yield alias.name, ""


def _all_documented_imports() -> list[tuple[str, str, str]]:
    return [(where, mod, name) for where, src in _sources() for mod, name in _imports(src)]


def test_the_scan_finds_the_examples_it_is_meant_to_check() -> None:
    imports = _all_documented_imports()
    assert len(imports) > 150, "the docs scan found almost nothing: is the fence pattern broken?"
    assert {w.split(":")[0] for w, _, _ in imports} >= {
        "README.md",
        "docs/plugin-authoring.md",
        "docs/registry.md",
        "docs/tracking.md",
        "examples/perf_budget.py",
    }


@pytest.mark.parametrize(
    ("where", "module", "name"),
    sorted(set(_all_documented_imports())),
    ids=lambda v: str(v),
)
def test_every_documented_import_is_a_public_name(where: str, module: str, name: str) -> None:
    if module == "pmlab._deprecation" and where.startswith("docs/deprecation.md"):
        return  # the contributor-facing page shows the internal helper, on purpose
    assert module in PUBLIC_MODULES, (
        f"{where}: `{module}` is not a public module; import from `pmlab` (or one of "
        f"{', '.join(PUBLIC_MODULES[1:])}) instead"
    )
    if name:  # `import pmlab.datasets` has no name
        exported = importlib.import_module(module).__all__
        assert name in exported, f"{where}: `{name}` is not in {module}.__all__"


class TestNamesThatWereOnlyReachableThroughAnInternalPath:
    """Documented in a page, so public: they had no stable path until 1.0."""

    def test_the_model_registry_is_public(self) -> None:
        import pmlab

        assert "ModelRegistry" in pmlab.__all__
        assert "RegistryEntry" in pmlab.__all__
        from pmlab.modeling.registry import ModelRegistry, RegistryEntry

        assert pmlab.ModelRegistry is ModelRegistry
        assert pmlab.RegistryEntry is RegistryEntry

    def test_mlflow_tracker_is_public_without_importing_mlflow(self) -> None:
        # `pmlab.tracking.MLflowTracker` is the documented path; it must not make the core
        # install pull MLflow, and importing `pmlab.tracking` must not import the module either.
        code = textwrap.dedent(
            """
            import sys
            import pmlab.tracking as tracking
            assert "MLflowTracker" in tracking.__all__
            assert "pmlab.tracking.mlflow_tracker" not in sys.modules, "imported eagerly"
            tracker_class = tracking.MLflowTracker
            assert "pmlab.tracking.mlflow_tracker" in sys.modules
            assert tracker_class.__module__ == "pmlab.tracking.mlflow_tracker"
            assert "mlflow" not in sys.modules, "importing the class must not import mlflow"
            from pmlab.tracking import MLflowTracker
            assert MLflowTracker is tracker_class
            assert "MLflowTracker" in dir(tracking)
            try:
                tracking.NotAThing
            except AttributeError:
                pass
            else:
                raise SystemExit("unknown attribute did not raise")
            print("ok")
            """
        )
        proc = subprocess.run(  # noqa: S603 - our own interpreter and code
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "ok"
