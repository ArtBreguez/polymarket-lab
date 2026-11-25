"""The command-line surface reader, against whatever Typer is installed (Typer 0.29 audit).

``scripts/api_surface.py`` reads every command, option and type off the Typer app, and Typer's
object model changed under it: up to 0.28 an option carries a Click ``ParamType`` (``param.type``);
from 0.29 Typer no longer builds on Click and an option carries a ``type_descriptor`` instead. A
toy app with one option of every kind pmlab's CLI may grow is read, and what the reader says must
be the same text under both, so a user's fresh install does not see the API "change" because a
dependency was upgraded.

Standalone, from the installed wheel:  python audit_cli_surface.py <scripts/api_surface.py>
Prints ``audit cli surface: OK <typer version>``. The default suite imports ``check`` and runs it
under the locked Typer; the e2e job runs this file under the newest one.
"""

from __future__ import annotations

import enum
import importlib.metadata
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Annotated, Any, Literal

import typer
import typer.main


class Mode(enum.Enum):
    fast = "fast"
    slow = "slow"


def build_app() -> typer.Typer:
    app = typer.Typer()

    @app.command()
    def run(
        name: Annotated[str, typer.Option("--name", "-n")],
        count: Annotated[int, typer.Option()] = 3,
        ratio: Annotated[float, typer.Option(min=0.0)] = 0.5,
        limit: Annotated[int, typer.Option(min=1, max=9)] = 4,
        verbose: Annotated[bool, typer.Option("--verbose/--quiet")] = False,
        dry: Annotated[bool, typer.Option("--dry-run")] = False,
        out: Annotated[Path, typer.Option()] = Path("x.json"),
        mode: Annotated[Mode, typer.Option()] = Mode.fast,
        kind: Annotated[Literal["a", "b"], typer.Option()] = "a",
        maybe: Annotated[str | None, typer.Option()] = None,
        tags: Annotated[list[str] | None, typer.Option()] = None,
        token: Annotated[str, typer.Option(envvar=["T_ONE", "T_TWO"])] = "",
        src: Annotated[str, typer.Argument()] = "-",
    ) -> None: ...

    @app.command()
    def other(x: Annotated[int, typer.Option()] = 1) -> None: ...

    return app


# name -> kind, flags, required, multiple, flag, type (what the snapshot records for each)
EXPECTED: dict[str, tuple[str, list[str], bool, bool, bool, str]] = {
    "name": ("option", ["--name", "-n"], True, False, False, "str"),
    "count": ("option", ["--count"], False, False, False, "int"),
    "ratio": ("option", ["--ratio"], False, False, False, "float[0.0..]"),
    "limit": ("option", ["--limit"], False, False, False, "int[1..9]"),
    "verbose": ("option", ["--verbose", "--quiet"], False, False, True, "bool"),
    "dry": ("option", ["--dry-run"], False, False, True, "bool"),
    "out": ("option", ["--out"], False, False, False, "path"),
    "mode": ("option", ["--mode"], False, False, False, "choice[fast,slow]"),
    "kind": ("option", ["--kind"], False, False, False, "choice[a,b]"),
    "maybe": ("option", ["--maybe"], False, False, False, "str"),
    "tags": ("option", ["--tags"], False, True, False, "str"),
    "token": ("option", ["--token"], False, False, False, "str"),
    "src": ("argument", ["src"], False, False, False, "str"),
}
DEFAULTS = {"count": "3", "ratio": "0.5", "limit": "4", "dry": "False", "out": "x.json"}
DEFAULTS |= {"kind": "a", "token": "", "src": "-", "name": None, "maybe": None}


def load_reader(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("api_surface", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["api_surface"] = module
    spec.loader.exec_module(module)
    return module


def check(reader: Any) -> None:
    """Assert what ``reader`` (the ``api_surface`` module) says about the toy app."""
    described = reader._describe_command(typer.main.get_command(build_app()))
    assert sorted(described["commands"]) == ["other", "run"], described["commands"].keys()
    params = {p["name"]: p for p in described["commands"]["run"]["params"]}
    assert list(params) == list(EXPECTED), list(params)  # in declaration order, none added
    for name, (kind, flags, required, multiple, flag, type_name) in EXPECTED.items():
        got = params[name]
        assert (got["kind"], got["flags"], got["required"], got["multiple"], got["flag"]) == (
            kind,
            flags,
            required,
            multiple,
            flag,
        ), (name, got)
        assert got["type"] == type_name, (name, got["type"], type_name)
    for name, default in DEFAULTS.items():
        assert params[name]["default"] == default, (name, params[name]["default"])
    assert params["token"]["envvar"] == ["T_ONE", "T_TWO"] and params["count"]["envvar"] is None
    assert described["commands"]["other"]["params"][0]["type"] == "int"


def main(argv: list[str]) -> int:
    check(load_reader(Path(argv[1])))
    print("audit cli surface: OK", importlib.metadata.version("typer"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
