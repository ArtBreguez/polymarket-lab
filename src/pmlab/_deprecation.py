"""How pmlab deprecates a public name (the policy is in ``docs/deprecation.md``).

Internal module: contributors use it, users only ever meet its warning,
`PmlabDeprecationWarning`, which is public (``pmlab.PmlabDeprecationWarning``).

Two entry points:

* `deprecated` — a decorator for a function, method, coroutine function or class.
* `warn_deprecated` — for what a decorator cannot wrap: an argument, a keyword value, a
  module attribute, a key in a file.

Both emit a `PmlabDeprecationWarning`, validate the timeline (see below), and record the
deprecation in a registry that a test reads, so a shim cannot silently outlive its removal.

Why ``DeprecationWarning`` and not ``FutureWarning``
----------------------------------------------------
Python shows a ``DeprecationWarning`` by default when the code that *triggers* it is
``__main__`` (a script, a notebook cell) or a test run, and hides it when another library's code
triggers it. With ``stacklevel`` pointing at the caller, that is exactly the audience that can act
on the message: the author of the call. ``FutureWarning`` is shown to everyone, which suits an
application warning *its* end users; for a library it would also nag the users of a package that
merely depends on pmlab, who can do nothing about it.

The command line is the one surface where the audience is a person at a terminal, not a Python
caller. There a deprecated command uses Typer's own ``deprecated=True`` (it prints a notice), and
a deprecated option, which Typer cannot flag, calls ``warn_deprecated(..., cli=True)``: one plain
``warning: ...`` line on stderr. (Not ``FutureWarning`` either: its ``file:line: Category``
format means nothing to someone typing ``pmlab backtest``.)

The timeline
------------
A name deprecated in release ``since`` is removed in ``removed_in``, which must be a later *major*
release (``X.0.0``) and never earlier than ``2.0.0``: a 1.x release never removes anything. Both
are checked when the decorator is applied, so a wrong timeline fails at import, not in review.
"""

from __future__ import annotations

import functools
import inspect
import re
import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar, cast

__all__ = [
    "Deprecation",
    "PmlabDeprecationWarning",
    "deprecated",
    "parse_version",
    "registered",
    "warn_deprecated",
]

_T = TypeVar("_T")
_VERSION = re.compile(r"(\d+)\.(\d+)\.(\d+)")


class PmlabDeprecationWarning(DeprecationWarning):
    """A pmlab name, argument or format is deprecated and will be removed in a later major release.

    It is a `DeprecationWarning`, so the usual filters apply, and it has its own name so a
    project can single out pmlab's: ``-W error::pmlab.PmlabDeprecationWarning`` turns only these
    into errors (for instance in CI, to find every call to update before the next major release).
    """


# Shown as `pmlab.PmlabDeprecationWarning` in tracebacks and test reports: the public path.
PmlabDeprecationWarning.__module__ = "pmlab"


@dataclass(frozen=True)
class Deprecation:
    """One deprecation: what, since when, until when, and what to use instead."""

    name: str
    since: str
    removed_in: str
    alternative: str | None
    reason: str | None
    module: str = ""

    def message(self) -> str:
        """The text of the warning."""
        text = (
            f"{self.name} is deprecated since {self.since} and will be removed in {self.removed_in}"
        )
        if self.alternative:
            text += f"; use {self.alternative} instead"
        if self.reason:
            text += f" ({self.reason})"
        return text

    def notice(self) -> str:
        """The paragraph added to a docstring (plain text: it must read well in ``help()``)."""
        text = f"Deprecated since {self.since} and will be removed in {self.removed_in}"
        text += f": use {self.alternative} instead." if self.alternative else "."
        if self.reason:
            text += f" {self.reason[0].upper()}{self.reason[1:]}."
        return text


_REGISTRY: list[Deprecation] = []


def parse_version(text: str, what: str = "version") -> tuple[int, int, int]:
    """Parse a plain ``X.Y.Z`` release number (no pre-release or local part)."""
    match = _VERSION.fullmatch(text)
    if match is None:
        raise ValueError(f"{what} must be a plain X.Y.Z release number, got {text!r}")
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


def registered() -> tuple[Deprecation, ...]:
    """Every deprecation declared so far, as a read-only snapshot."""
    return tuple(_REGISTRY)


def _check_timeline(since: str, removed_in: str) -> None:
    first = parse_version(since, "since version")
    last = parse_version(removed_in, "removed_in version")
    if last[1] != 0 or last[2] != 0 or last[0] < 2 or last[0] <= first[0]:
        raise ValueError(
            f"removed_in={removed_in!r} breaks the deprecation timeline: a name deprecated in "
            f"{since} may only be removed in a later major release (X.0.0), and never before "
            "2.0.0 (see docs/deprecation.md)"
        )


def _record(entry: Deprecation) -> None:
    if entry not in _REGISTRY:
        _REGISTRY.append(entry)


def _short_qualname(obj: Any) -> str:
    qualname: str = getattr(obj, "__qualname__", getattr(obj, "__name__", repr(obj)))
    return qualname.rsplit("<locals>.", 1)[-1]


def _with_notice(doc: str | None, entry: Deprecation) -> str:
    """Put the deprecation notice right after the summary paragraph of ``doc``."""
    notice = entry.notice()
    if not doc or not doc.strip():
        return notice
    head, sep, rest = inspect.cleandoc(doc).partition("\n\n")
    return f"{head}\n\n{notice}{sep}{rest}" if sep else f"{head}\n\n{notice}"


def _wrap_callable[T](func: T, entry: Deprecation) -> T:
    message = entry.message()

    @functools.wraps(func)  # type: ignore[arg-type]
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        warnings.warn(message, PmlabDeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)  # type: ignore[operator]

    if inspect.iscoroutinefunction(func):
        # The wrapper warns at the call and hands back the coroutine; tell `inspect` it is one.
        inspect.markcoroutinefunction(wrapper)
    wrapper.__doc__ = _with_notice(getattr(func, "__doc__", None), entry)
    return cast(T, wrapper)


def _wrap_class(cls: type, entry: Deprecation) -> type:
    original_init = cls.__init__  # type: ignore[misc]
    message = entry.message()

    @functools.wraps(original_init)
    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        warnings.warn(message, PmlabDeprecationWarning, stacklevel=2)
        original_init(self, *args, **kwargs)

    cls.__init__ = __init__  # type: ignore[misc]
    cls.__doc__ = _with_notice(cls.__doc__, entry)
    return cls


def deprecated(
    *,
    since: str,
    removed_in: str,
    alternative: str | None = None,
    reason: str | None = None,
) -> Callable[[_T], _T]:
    """Mark a function, method, coroutine function or class as deprecated.

    Calling it (instantiating, for a class) emits a `PmlabDeprecationWarning` that names
    the caller's line, then runs the original unchanged. The signature, ``__name__`` and
    ``__wrapped__`` are preserved (the public-API snapshot reads them), and a notice is added to
    the docstring.

    Args:
        since: The release that deprecates it, as ``X.Y.Z``.
        removed_in: The release that removes it: a later major (``X.0.0``), never before 2.0.0.
        alternative: What to use instead, as it should read in "use ... instead".
        reason: One short clause on why, appended to the message.

    Raises:
        ValueError: A version is not ``X.Y.Z``, or ``removed_in`` breaks the timeline.
    """
    _check_timeline(since, removed_in)

    def decorate(obj: _T) -> _T:
        if isinstance(obj, classmethod):
            return cast(_T, classmethod(decorate(obj.__func__)))
        if isinstance(obj, staticmethod):
            return cast(_T, staticmethod(decorate(obj.__func__)))
        entry = Deprecation(
            name=_short_qualname(obj),
            since=since,
            removed_in=removed_in,
            alternative=alternative,
            reason=reason,
            module=getattr(obj, "__module__", ""),
        )
        _record(entry)
        if inspect.isclass(obj):
            return cast(_T, _wrap_class(obj, entry))
        return _wrap_callable(obj, entry)

    return decorate


def warn_deprecated(
    what: str,
    *,
    since: str,
    removed_in: str,
    alternative: str | None = None,
    reason: str | None = None,
    stacklevel: int = 2,
    cli: bool = False,
) -> None:
    """Warn that something a decorator cannot wrap is deprecated: an argument, a value, a key.

    ``stacklevel=2`` (the default) names the line that calls this function; ``3`` names the
    caller of the function that called it, which is what an API function wants (``run`` warns
    about its own argument, the warning should name whoever called ``run``).

    The deprecation is registered the first time it is emitted, so the removal reminder test only
    sees it once something has triggered it: prefer `deprecated` where a callable fits.

    Args:
        what: The subject of the sentence, e.g. ``"the `min_edge_pct` argument"``.
        since: The release that deprecates it, as ``X.Y.Z``.
        removed_in: The release that removes it: a later major (``X.0.0``), never before 2.0.0.
        alternative: What to use instead, as it should read in "use ... instead".
        reason: One short clause on why, appended to the message.
        stacklevel: As for `warnings.warn`: 2 names the line that calls this function,
            3 names the caller of the function that called it.
        cli: The deprecated thing is a command-line option: print one plain
            ``warning: ...`` line to stderr instead of raising a Python warning (a person at a
            terminal does not see, and cannot act on, a warning attributed to pmlab's own code).

    Raises:
        ValueError: A version is not ``X.Y.Z``, or ``removed_in`` breaks the timeline.
    """
    _check_timeline(since, removed_in)
    entry = Deprecation(
        name=what, since=since, removed_in=removed_in, alternative=alternative, reason=reason
    )
    _record(entry)
    if cli:
        print(f"warning: {entry.message()}", file=sys.stderr)
        return
    warnings.warn(entry.message(), PmlabDeprecationWarning, stacklevel=stacklevel)
