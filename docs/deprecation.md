# Deprecation policy

A public name (see [Public API and stability](api-stability.md)) is never removed without notice.
This page is the notice procedure: what a deprecation looks like to a user, how long it lasts, how
to find the calls that need updating, and how a contributor deprecates something.

## The timeline

```text
 1.N.0                      1.N+1 … 1.last                       2.0.0
   │                                │                               │
   ▼                                ▼                               ▼
 deprecated:  works, warns   ── still works, still warns ──►   removed
 (changelog: "Deprecated")                                  (changelog: "Removed")
```

* A name is deprecated in a **minor** release (`since`). From then on it works exactly as before and
  emits a `PmlabDeprecationWarning` naming the replacement.
* It is removed in the **next major** release at the earliest (`removed_in`), always an `X.0.0`.
  A 1.x release never removes anything, and nothing is removed in the same release that deprecates
  it.
* Nothing deprecated before 1.0.0 is removed before **2.0.0** either: the promise starts with 1.0.0.
* Every deprecation is announced in four places: the warning, a notice in the object's docstring (so
  in `help()` and in the [API reference](reference/index.md)), the changelog under *Deprecated*, and
  the table [below](#currently-deprecated).

The helper that implements this refuses a timeline that breaks the rule: `removed_in` must be a
later major release and never earlier than `2.0.0`, or the decorator raises when the module is
imported. A test fails when the package reaches the `removed_in` of any deprecation still in the
code, as a reminder to delete the shim.

## The warning

Python code gets a `PmlabDeprecationWarning`, a subclass of `DeprecationWarning`. For example:

```text
script.py:12: PmlabDeprecationWarning: WeatherTmaxPlugin.build_training_row is deprecated since
0.9.2 and will be removed in 2.0.0; use build_training_rows (one row per bin) instead (it emits
only the winning bin, which biases any backtest built from it)
```

The line it names is **your** call, not a line inside pmlab.

**Why `DeprecationWarning` and not `FutureWarning`.** Python shows a `DeprecationWarning` by
default when the code that triggers it is `__main__` (a script, a notebook cell) and under pytest,
and hides it when another library's code triggers it. Because pmlab points the warning at the
caller, that is exactly the audience that can act on it: the author of the call sees it, and the
users of a package that merely depends on pmlab are not nagged about something they cannot change.
`FutureWarning` is for the end users of an application; it would show the message to everyone.

**The command line is different.** Someone typing `pmlab backtest` is not a Python caller, and a
warning attributed to pmlab's own code is hidden from them. So a deprecated *command* is flagged
with Typer's `deprecated=True`, which prints a notice (`DeprecationWarning: The command 'x' is
deprecated.`) and runs it; a deprecated *option*, which Typer cannot flag, prints one plain line to
stderr and still works:

```text
warning: --min-edge-pct is deprecated since 1.1.0 and will be removed in 2.0.0; use --min-edge instead
```

## Finding what to update

Turn pmlab's deprecations into errors, and only pmlab's, so the first one fails loudly with a
traceback to your call:

=== "pytest"

    ```text
    pytest -W error::pmlab.PmlabDeprecationWarning
    ```

    or, permanently, in `pyproject.toml`:

    ```toml
    [tool.pytest.ini_options]
    filterwarnings = ["error::pmlab.PmlabDeprecationWarning"]
    ```

=== "Python"

    ```python
    import warnings

    import pmlab

    warnings.simplefilter("error", pmlab.PmlabDeprecationWarning)
    ```

Both leave other libraries' deprecations alone. To *see* every pmlab deprecation without failing,
use `warnings.simplefilter("always", pmlab.PmlabDeprecationWarning)` or run the script with
`python -W default::DeprecationWarning script.py`.

!!! note
    The interpreter's own `-W` option cannot name a category from a third-party package: at
    start-up it ignores `-W error::pmlab.PmlabDeprecationWarning` with a message about an invalid
    module name. Use one of the recipes above, or `-W error::DeprecationWarning` for *all*
    deprecations.

## Deprecating something (for contributors)

The helper is internal (`pmlab._deprecation`); only its warning class is public.

```python
from pmlab._deprecation import deprecated, warn_deprecated


@deprecated(
    since="1.3.0",
    removed_in="2.0.0",
    alternative="new_function",
    reason="it ignores the cost model",       # optional: one short clause
)
def old_function(...): ...
```

`deprecated(...)` wraps a function, method, `classmethod`, `staticmethod`, coroutine function or
class (a class warns when it is instantiated). It keeps the signature, `__name__` and `__wrapped__`
(so the public-API snapshot does not change), adds the notice to the docstring, and registers the
deprecation. For what a decorator cannot wrap, call `warn_deprecated(...)`:

```python
def run(min_edge_pct=None, min_edge=0.05):
    if min_edge_pct is not None:
        warn_deprecated(
            "the `min_edge_pct` argument",
            since="1.3.0",
            removed_in="2.0.0",
            alternative="`min_edge`",
            stacklevel=3,          # name the caller of `run`, not `run` itself
        )
        min_edge = min_edge_pct / 100
```

and pass `cli=True` for a command-line option. The checklist:

1. Apply `deprecated(...)` / call `warn_deprecated(...)` with `since`, `removed_in` and, always,
   an `alternative`.
2. Add a row to the table below. A test fails if a registered deprecation is missing from it.
3. Add the changelog line under *Deprecated*.
4. Keep a test that the old path still works *and* warns (`pytest.warns(PmlabDeprecationWarning)`).
5. Stop using the old path inside pmlab itself: pmlab must not trigger its own warning.
6. In the major release that removes it: delete the shim, regenerate the snapshot
   (`uv run python scripts/api_surface.py --write`), list it under *Removed*.

## Currently deprecated

| Name | Since | Removed in | Use instead |
|------|-------|------------|-------------|
| `WeatherTmaxPlugin.build_training_row` | 0.9.2 | 2.0.0 | `build_training_rows`: one row per bin; the single winning-bin row biases a backtest (issue #28) |
