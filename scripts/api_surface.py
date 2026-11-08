#!/usr/bin/env python
"""Read pmlab's public API off the installed package; check it against, or write, the snapshot.

What counts as public is defined in ``docs/api-stability.md``. This script is the machine-readable
form of that page:

* every name in ``pmlab.__all__`` and in the ``__all__`` of the documented subpackages
  (:data:`PUBLIC_MODULES`): kind, signature, and for a class its public members;
* the ``pmlab`` command line: commands, options, their flags, defaults and environment variables;
* the file formats and environment variables pmlab reads and writes.

Usage::

    python scripts/api_surface.py --check    # exit 1 and say what differs (the default)
    python scripts/api_surface.py --write    # rewrite tests/api/public_api.json, on purpose
    python scripts/api_surface.py --print    # show the current surface

It imports every public name, so it doubles as "does the installed package still import?": the
end-to-end test runs it with the interpreter of a venv holding only the built wheel.
Standard library plus pmlab's own dependencies; it never touches the network.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import importlib
import inspect
import json
import re
import sys
import types
from pathlib import Path
from typing import Any

SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / "tests" / "api" / "public_api.json"
SNAPSHOT_VERSION = 1

# `pmlab` and every subpackage that declares an `__all__`. The other subpackages (core, backtest,
# execution, markets, modeling, plugins, workspace) are internal paths: their public names are
# re-exported by `pmlab` itself.
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

_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")
# `pmlab.core.market_spec.OutcomeBin` -> `OutcomeBin`: an internal module path is not API.
_INTERNAL_PATH = re.compile(r"\bpmlab(?:\.[a-z_0-9]+)+\.([A-Za-z_]\w*)")
_PRIMITIVES = (bool, int, float, str, type(None))
# Typer adds these to the root command; they are Typer's, not pmlab's.
_TYPER_OWN_PARAMS = {"help", "install_completion", "show_completion"}


# ── signatures ────────────────────────────────────────────────────────────────────────────────
class _Text:
    """Prints as the text it was given: how an annotation is rendered in the snapshot."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __repr__(self) -> str:
        return self.text


def _annotation_text(annotation: Any) -> str:
    """An annotation as written in the source.

    Under ``from __future__ import annotations`` it already is a string; otherwise it is formatted,
    so adding or dropping that import in a module does not change the snapshot.
    """
    text = annotation if isinstance(annotation, str) else inspect.formatannotation(annotation)
    return _INTERNAL_PATH.sub(r"\1", text.replace("typing.", ""))


def _signature(obj: Any) -> str | None:
    """``inspect.signature`` minus the private parameters, annotations as text, no addresses.

    A parameter whose name starts with an underscore is not part of the API (``PmlabSettings``
    inherits dozens from pydantic-settings, whose set and defaults change with that library).
    """
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    parameters = []
    for param in signature.parameters.values():
        if param.name.startswith("_"):
            continue
        if param.annotation is not inspect.Parameter.empty:
            param = param.replace(annotation=_Text(_annotation_text(param.annotation)))
        parameters.append(param)
    returns = signature.return_annotation
    if returns is not inspect.Signature.empty:
        returns = _Text(_annotation_text(returns))
    return _ADDRESS.sub(
        "", str(signature.replace(parameters=parameters, return_annotation=returns))
    )


# ── names ─────────────────────────────────────────────────────────────────────────────────────
def _constant_value(value: Any) -> Any:
    if isinstance(value, _PRIMITIVES):
        return value
    if isinstance(value, (set, frozenset)):
        return sorted(_constant_value(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [_constant_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _constant_value(v) for k, v in sorted(value.items(), key=str)}
    return _ADDRESS.sub("", repr(value))


def _describe_member(name: str, raw: Any) -> dict[str, Any] | None:
    """One public member of a class, from its raw ``vars()`` entry; ``None`` to skip."""
    if isinstance(raw, (classmethod, staticmethod)):
        func = raw.__func__
        entry: dict[str, Any] = {
            "kind": "classmethod" if isinstance(raw, classmethod) else "staticmethod",
            "signature": _signature(func),
        }
    elif isinstance(raw, property) or type(raw).__name__ == "cached_property":
        entry = {"kind": "property"}
        func = raw.fget if isinstance(raw, property) else getattr(raw, "func", None)
    elif inspect.isfunction(raw):
        func = raw
        entry = {"kind": "method", "signature": _signature(raw)}
    elif inspect.isclass(raw):
        return {"kind": "class"}
    else:
        value = raw if isinstance(raw, _PRIMITIVES) else None
        entry = {"kind": "attribute", "type": type(raw).__name__}
        if value is not None:
            entry["value"] = value
        return entry
    if func is not None:
        if getattr(func, "__isabstractmethod__", False):
            entry["abstract"] = True
        if inspect.iscoroutinefunction(func):
            entry["async"] = True
    return entry


def _describe_class(cls: type) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "kind": "class",
        "bases": [base.__qualname__ for base in cls.__bases__ if base is not object],
    }
    signature = _signature(cls)
    if signature is not None:  # an exception class has none
        entry["signature"] = signature
    skip: set[str] = set()
    if dataclasses.is_dataclass(cls):  # a dataclass field is in the signature already
        skip |= {f.name for f in dataclasses.fields(cls)}
    fields = getattr(cls, "model_fields", None)  # so is a pydantic field
    if isinstance(fields, dict):
        skip |= set(fields)
    members: dict[str, Any] = {}
    # Base to derived, so an override replaces what it overrides. Only pmlab's own classes:
    # what is inherited from the standard library or a dependency is not pmlab's contract.
    for klass in reversed(cls.__mro__):
        if not klass.__module__.startswith("pmlab"):
            continue
        for name, raw in vars(klass).items():
            if name.startswith("_") or name in skip:
                continue
            described = _describe_member(name, raw)
            if described is None:
                continue
            members[name] = described
    if issubclass(cls, enum.Enum):
        entry["enum_members"] = {m.name: _constant_value(m.value) for m in cls}
        members = {k: v for k, v in members.items() if k not in cls.__members__}
    if members:
        entry["members"] = dict(sorted(members.items()))
    return entry


def _describe(name: str, obj: Any) -> dict[str, Any]:
    if isinstance(obj, types.ModuleType):
        return {"kind": "module"}
    if inspect.isclass(obj):
        return _describe_class(obj)
    if callable(obj) and (inspect.isfunction(obj) or inspect.isbuiltin(obj)):
        entry: dict[str, Any] = {"kind": "function", "signature": _signature(obj)}
        if inspect.iscoroutinefunction(obj):
            entry["async"] = True
        return entry
    if name == "__version__":  # changes every release; the type is the contract
        return {"kind": "constant", "type": type(obj).__name__}
    if isinstance(obj, _PRIMITIVES + (tuple, list, set, frozenset, dict)):
        return {"kind": "constant", "type": type(obj).__name__, "value": _constant_value(obj)}
    return {"kind": "other", "type": type(obj).__name__}


def _modules() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for module_name in PUBLIC_MODULES:
        module = importlib.import_module(module_name)
        names = list(module.__all__)
        out[module_name] = {name: _describe(name, getattr(module, name)) for name in sorted(names)}
    return out


# ── the command line ──────────────────────────────────────────────────────────────────────────
def _default_text(default: Any) -> str | None:
    """A click default as text; ``None`` when there is none (also click's "unset" sentinel)."""
    if default is None or type(default).__name__ == "Sentinel":
        return None
    if callable(default):  # a callable default is resolved at run time: not frozen
        return "<callable>"
    return _ADDRESS.sub("", str(default))


# By class name, not `isinstance`: newer Typer no longer depends on `click` (it carries its own
# copy of these classes), so the classes themselves cannot be imported by a stable name.
_TYPE_LABELS = {
    "BoolParamType": "bool",
    "IntParamType": "int",
    "IntRange": "int",
    "FloatParamType": "float",
    "FloatRange": "float",
    "StringParamType": "str",
    "Path": "path",
}


def _type_name(param_type: Any) -> str:
    """A parameter's type from the class of its type object, not the library's own label for it.

    The label (``text``, ``integer``, ``integer range``) changed between click releases, so a
    user's fresh install would see the API "change" because a dependency was upgraded.
    """
    cls_name = type(param_type).__name__
    if cls_name == "Choice":
        return "choice[" + ",".join(str(c) for c in param_type.choices) + "]"
    name = _TYPE_LABELS.get(cls_name, cls_name)
    low, high = getattr(param_type, "min", None), getattr(param_type, "max", None)
    if low is not None or high is not None:  # IntRange / FloatRange
        name += f"[{'' if low is None else low}..{'' if high is None else high}]"
    return name


def _describe_param(param: Any) -> dict[str, Any]:
    envvar = param.envvar
    return {
        "name": param.name,
        "kind": "argument" if type(param).__name__.endswith("Argument") else "option",
        "flags": [*param.opts, *param.secondary_opts],
        "required": bool(param.required),
        "multiple": bool(param.multiple),
        "flag": bool(getattr(param, "is_flag", False)),
        "type": _type_name(param.type),
        "default": _default_text(param.default),
        "envvar": [envvar] if isinstance(envvar, str) else (list(envvar) if envvar else None),
    }


def _describe_command(command: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "params": [_describe_param(p) for p in command.params if p.name not in _TYPER_OWN_PARAMS],
    }
    if getattr(command, "deprecated", False):
        entry["deprecated"] = True
    subcommands = getattr(command, "commands", None)
    if subcommands:
        entry["commands"] = {
            name: _describe_command(sub) for name, sub in sorted(subcommands.items())
        }
    return entry


def _cli() -> dict[str, Any]:
    import typer.main

    from pmlab.cli.main import app

    root = typer.main.get_command(app)
    described = _describe_command(root)
    return {
        "name": "pmlab",
        "commands": described.get("commands", {}),
        "params": described["params"],
    }


# ── file formats and environment ──────────────────────────────────────────────────────────────
def _key_tree(value: Any, free: frozenset[str] = frozenset()) -> Any:
    """The key structure of a JSON payload. Maps named in ``free`` hold data, not schema: ``*``."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, inner in value.items():
            if key in free and isinstance(inner, dict):
                out[key] = {"*": next((_key_tree(v, free) for v in inner.values()), None)}
            else:
                out[key] = _key_tree(inner, free)
        return out
    if isinstance(value, list):
        return [_key_tree(value[0], free)] if value else []
    return None


def _env_variables(pmlab: Any) -> list[str]:
    """Every ``PMLAB_*`` variable pmlab reads: the settings' own, plus any named in the source."""
    names: set[str] = set()
    prefix = pmlab.PmlabSettings.model_config.get("env_prefix", "")
    for field_name, field in pmlab.PmlabSettings.model_fields.items():
        alias = field.validation_alias
        if alias is None:
            names.add(f"{prefix}{field_name}".upper())
        elif isinstance(alias, str):
            names.add(alias)
        else:
            names.update(getattr(alias, "choices", []))
    literal = re.compile(r"\bPMLAB_[A-Z][A-Z_]*[A-Z]\b")
    for source in Path(pmlab.__file__).parent.rglob("*.py"):
        names.update(literal.findall(source.read_text(encoding="utf-8")))
    return sorted(n for n in names if n.startswith("PMLAB_"))


def _formats() -> dict[str, Any]:
    import pmlab
    from pmlab.plugins.discovery import ENTRY_POINT_GROUP

    gate = pmlab.HoldoutGateResult(
        decision="GO",
        segment_results=[pmlab.SegmentGateResult("segment", 1, 0.0, True, "reason")],
        aggregate_pnl=0.0,
        aggregate_trades=1,
    )
    group = pmlab.GroupExpectation(2, 0.0, 0.0, 0.0, 0.0)
    expectation = pmlab.BacktestExpectation(
        n_trades=2,
        flat_stake=1.0,
        mean_pnl_per_stake=0.0,
        std_pnl_per_stake=0.0,
        brier_score=0.0,
        brier_std=0.0,
        by_direction={"yes": group},
        policy=pmlab.TradePolicy(),
        by_segment={"segment": pmlab.SegmentExpectation(2, 0.0, 0.0, 0.0, 0.0, {"yes": group})},
    )
    repro = pmlab.ReproducibilityRecord(
        kind="backtest",
        seeds={"python_random": 0},
        environment={"python": "3"},
        config={"k": 1},
        data=pmlab.DataFingerprint("sha", 1, ["c"], "source"),
        metrics={"m": 0.0},
    )
    manifest = pmlab.ChampionManifest(
        model_name="m",
        model_path=Path("m.pkl"),
        calibrator_path=Path("c.pkl"),
        gate=gate,
        published_at="2026-01-01T00:00:00+00:00",
        plugin_family="family",
        costs=pmlab.CostModel(),
        expectation=expectation,
        reproducibility=repro,
    )
    free = frozenset({"by_segment", "by_direction", "seeds", "environment", "config", "metrics"})
    run = pmlab.RunRecord(run_id="id", experiment="e", run_name="n", ended_at="t", error="e")
    free_env = _env_variables(pmlab)
    return {
        "champion.json": _key_tree(manifest.to_dict(), free),
        "run_record.json": _key_tree(run.to_dict(), free),
        "reproducibility_record": {
            "schema_version": pmlab.reproducibility.SCHEMA_VERSION,
            "keys": _key_tree(repro.to_dict(), free),
        },
        "plugin_entry_point_group": ENTRY_POINT_GROUP,
        "environment_variables": free_env,
    }


# ── snapshot ──────────────────────────────────────────────────────────────────────────────────
def surface() -> dict[str, Any]:
    """The public API, as it is right now in the importable ``pmlab``."""
    return {
        "snapshot_version": SNAPSHOT_VERSION,
        "modules": _modules(),
        "cli": _cli(),
        "formats": _formats(),
    }


def render(data: dict[str, Any]) -> str:
    """The exact text of the snapshot file (stable: sorted keys, one trailing newline)."""
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _named(items: Any) -> bool:
    """A non-empty list of dicts that each have a ``name``: a command's parameters."""
    return (
        isinstance(items, list)
        and bool(items)
        and all(isinstance(i, dict) and isinstance(i.get("name"), str) for i in items)
    )


def _join(path: str, key: Any, *, entry: bool = False) -> str:
    """``a.b``; a list entry or a dotted module name is bracketed: ``params[stride]``."""
    text = f"[{key}]" if entry or "." in str(key) else f".{key}"
    return f"{path}{text}" if path else str(key)


def diff(before: Any, after: Any, path: str = "") -> list[str]:
    """What differs between two surfaces, one readable line per name, option or field."""
    lines: list[str] = []
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(set(before) | set(after)):
            here = _join(path, key)
            if key not in after:
                lines.append(f"removed: {here}")
            elif key not in before:
                lines.append(f"added:   {here}")
            else:
                lines.extend(diff(before[key], after[key], here))
        return lines
    if _named(before) and _named(after):  # a command's parameters: by name, then by order
        old = {item["name"]: item for item in before}
        new = {item["name"]: item for item in after}
        for name in sorted(set(old) | set(new)):
            here = _join(path, name, entry=True)
            if name not in new:
                lines.append(f"removed: {here}")
            elif name not in old:
                lines.append(f"added:   {here}")
            else:
                lines.extend(diff(old[name], new[name], here))
        order_before = [n for n in old if n in new]
        order_after = [n for n in new if n in old]
        if order_before != order_after:  # positional arguments depend on it
            lines.append(f"changed: {path}: order {order_before} -> {order_after}")
        return lines
    if before != after:
        return [f"changed: {path}: {before!r} -> {after!r}"]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="compare with the snapshot (default)")
    mode.add_argument("--write", action="store_true", help="rewrite the snapshot")
    mode.add_argument("--print", action="store_true", dest="show", help="print the surface")
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT_PATH, help="snapshot file")
    args = parser.parse_args(argv)

    current = surface()
    if args.show:
        sys.stdout.write(render(current))
        return 0
    if args.write:
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        args.snapshot.write_text(render(current))
        names = sum(len(v) for v in current["modules"].values())
        print(
            f"wrote {args.snapshot}: {names} public names, {len(current['cli']['commands'])} commands"
        )
        return 0

    expected = json.loads(args.snapshot.read_text())
    changes = diff(expected, current)
    if changes:
        print(f"the public API differs from {args.snapshot}:", file=sys.stderr)
        for line in changes:
            print(f"  {line}", file=sys.stderr)
        print("deliberate? rerun with --write and see docs/deprecation.md", file=sys.stderr)
        return 1
    names = sum(len(v) for v in current["modules"].values())
    print(
        f"public API matches {args.snapshot.name}: {names} public names imported, "
        f"{len(current['cli']['commands'])} commands"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
