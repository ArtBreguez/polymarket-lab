"""Show a deprecation in the API reference, from the ``@deprecated(...)`` call alone.

``pmlab._deprecation.deprecated`` adds a notice to the docstring *at run time* (so ``help()`` shows
it), but the API reference is built by Griffe from the source, without running anything: it would
never see that notice. This extension reads the decorator's arguments from the source and adds the
same notice, so ``since``, ``removed_in`` and ``alternative`` stay written in exactly one place.

Loaded by ``mkdocs.yml`` (``plugins: mkdocstrings: handlers: python: options: extensions``). The two
helpers below mirror ``Deprecation.notice`` and ``_with_notice`` in ``pmlab._deprecation``;
``tests/test_docs_site.py`` fails if they ever disagree.
"""

from __future__ import annotations

import ast
import inspect
from typing import Any

import griffe


def notice(
    since: str, removed_in: str, alternative: str | None = None, reason: str | None = None
) -> str:
    """The paragraph ``Deprecation.notice`` writes."""
    text = f"Deprecated since {since} and will be removed in {removed_in}"
    text += f": use {alternative} instead." if alternative else "."
    if reason:
        text += f" {reason[0].upper()}{reason[1:]}."
    return text


def with_notice(doc: str | None, text: str) -> str:
    """Put ``text`` right after the summary paragraph, as ``_with_notice`` does."""
    if not doc or not doc.strip():
        return text
    head, sep, rest = inspect.cleandoc(doc).partition("\n\n")
    return f"{head}\n\n{text}{sep}{rest}" if sep else f"{head}\n\n{text}"


def _literal(expression: Any) -> Any:
    try:
        return ast.literal_eval(str(expression))
    except (ValueError, SyntaxError):
        return None


def deprecation_of(obj: griffe.Object) -> str | None:
    """The notice for an object decorated with ``@deprecated(...)``, or ``None``."""
    for decorator in getattr(obj, "decorators", []):
        call = decorator.value
        if not isinstance(call, griffe.ExprCall) or not str(call.function).endswith("deprecated"):
            continue
        args = {
            argument.name: _literal(argument.value)
            for argument in call.arguments
            if isinstance(argument, griffe.ExprKeyword)
        }
        if isinstance(args.get("since"), str) and isinstance(args.get("removed_in"), str):
            return notice(
                args["since"], args["removed_in"], args.get("alternative"), args.get("reason")
            )
    return None


class DeprecatedNotice(griffe.Extension):
    """Add the deprecation notice to the docstring of every ``@deprecated`` object."""

    def on_instance(self, *, node: Any, obj: griffe.Object, agent: Any, **kwargs: Any) -> None:
        text = deprecation_of(obj)
        if text is None:
            return
        if obj.docstring is None:
            obj.docstring = griffe.Docstring(text, parent=obj)
        else:
            obj.docstring.value = with_notice(obj.docstring.value, text)
