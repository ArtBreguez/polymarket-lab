"""``scripts/api_surface.py`` reads the CLI the same way whichever Typer is installed.

Typer 0.29 dropped Click: an option no longer has ``.type`` (a Click ``ParamType``), it has a
``type_descriptor``. The reader reads both object models and says the same thing about either.
The locked Typer (the one these tests run under) is checked on a real toy app; the Click-free model
is checked here on a stand-in for it, and for real by ``pytest -m e2e``, which reads the CLI under
the newest Typer pip resolves into a fresh venv (``tests/e2e/audit_cli_surface.py``).
"""

from __future__ import annotations

import datetime
import enum
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Literal

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def reader() -> ModuleType:
    return _load("api_surface", REPO / "scripts" / "api_surface.py")


@pytest.fixture(scope="module")
def audit() -> ModuleType:
    return _load("audit_cli_surface", REPO / "tests" / "e2e" / "audit_cli_surface.py")


def test_the_toy_app_is_read_as_the_snapshot_would_record_it(
    reader: ModuleType, audit: ModuleType
) -> None:
    audit.check(reader)  # the same check the e2e job runs under the newest Typer


class _Mode(enum.Enum):
    fast = "fast"
    slow = "slow"


def _click_free(name: str, annotation: Any, **extra: Any) -> SimpleNamespace:
    """What Typer >= 0.29 hands over for an option: no ``type``, a ``type_descriptor`` instead."""
    option = {
        "name": name,
        "opts": [f"--{name}"],
        "secondary_opts": [],
        "required": False,
        "multiple": False,
        "is_flag": False,
        "default": None,
        "envvar": None,
        "min": None,
        "max": None,
        "type_descriptor": SimpleNamespace(annotation=annotation),
    }
    return SimpleNamespace(**{**option, **extra})


@pytest.mark.parametrize(
    ("annotation", "extra", "label"),
    [
        (str, {}, "str"),
        (int, {}, "int"),
        (float, {}, "float"),
        (bool, {"is_flag": True}, "bool"),
        (Path, {}, "path"),
        (str | None, {}, "str"),
        (list[str], {"multiple": True}, "str"),
        (list[str] | None, {"multiple": True}, "str"),
        (int, {"min": 0}, "int[0..]"),
        (int, {"min": 1, "max": 9}, "int[1..9]"),
        (float, {"min": 0.0}, "float[0.0..]"),
        (float, {"max": 1.0}, "float[..1.0]"),
        (Literal["a", "b"], {}, "choice[a,b]"),
        (_Mode, {}, "choice[fast,slow]"),
    ],
)
def test_a_click_free_option_gets_the_label_the_click_based_one_gets(
    reader: ModuleType, annotation: Any, extra: dict[str, Any], label: str
) -> None:
    described = reader._describe_param(_click_free("x", annotation, **extra))
    assert described["type"] == label
    assert described["flags"] == ["--x"] and described["kind"] == "option"


def test_a_click_free_option_with_an_environment_variable_and_a_default(reader: ModuleType) -> None:
    option = _click_free("workspace", str, envvar=["A", "B"], default="main", required=True)
    described = reader._describe_param(option)
    assert described["envvar"] == ["A", "B"] and described["default"] == "main"
    assert described["required"] is True


def test_an_object_model_the_reader_does_not_know_fails_loudly_not_with_an_attribute_error(
    reader: ModuleType,
) -> None:
    option = _click_free("mystery", str)
    del option.type_descriptor
    with pytest.raises(RuntimeError, match=r"mystery.*Typer|Typer.*mystery"):
        reader._describe_param(option)


def test_a_type_the_click_free_reader_has_no_label_for_is_refused_by_name(
    reader: ModuleType,
) -> None:
    with pytest.raises(RuntimeError, match=r"when.*datetime|datetime.*when"):
        reader._describe_param(_click_free("when", datetime.datetime))
