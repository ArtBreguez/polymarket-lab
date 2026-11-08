"""The deprecation helper: what it emits, where it points, what it refuses (docs/deprecation.md)."""

from __future__ import annotations

import inspect
import subprocess
import sys
import textwrap
import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest

import pmlab
from pmlab import _deprecation
from pmlab._deprecation import (
    Deprecation,
    PmlabDeprecationWarning,
    deprecated,
    registered,
    warn_deprecated,
)


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Each test starts from an empty registry: the helper's own tests must not leak entries."""
    monkeypatch.setattr(_deprecation, "_REGISTRY", [])
    yield


def _kwargs(**over: str) -> dict[str, str]:
    base = {"since": "1.2.0", "removed_in": "2.0.0", "alternative": "new_thing"}
    base.update(over)
    return base


class TestWarning:
    def test_it_is_a_deprecation_warning_with_its_own_name(self) -> None:
        # Subclass: default filters and `-W error::DeprecationWarning` still apply, and a user
        # can single out pmlab's with `error::pmlab.PmlabDeprecationWarning`.
        assert issubclass(PmlabDeprecationWarning, DeprecationWarning)
        assert pmlab.PmlabDeprecationWarning is PmlabDeprecationWarning

    def test_it_presents_itself_under_its_public_path(self) -> None:
        # A traceback or a pytest report shows `pmlab.PmlabDeprecationWarning`, not the internal
        # module it happens to live in.
        assert PmlabDeprecationWarning.__module__ == "pmlab"
        assert PmlabDeprecationWarning.__qualname__ == "PmlabDeprecationWarning"

    def test_function_warns_and_still_works(self) -> None:
        @deprecated(**_kwargs())
        def old(a: int, b: int = 2) -> int:
            return a + b

        with pytest.warns(PmlabDeprecationWarning) as caught:
            assert old(1, b=5) == 6
        assert len(caught) == 1

    def test_message_states_what_since_when_removed_and_the_way_out(self) -> None:
        @deprecated(**_kwargs(), reason="it biases the backtest")
        def old() -> None:
            return None

        with pytest.warns(PmlabDeprecationWarning) as caught:
            old()
        msg = str(caught[0].message)
        assert "old" in msg
        assert "1.2.0" in msg
        assert "2.0.0" in msg
        assert "use new_thing instead" in msg
        assert "it biases the backtest" in msg

    def test_alternative_is_optional(self) -> None:
        @deprecated(since="1.2.0", removed_in="2.0.0")
        def old() -> None:
            return None

        with pytest.warns(PmlabDeprecationWarning) as caught:
            old()
        assert "instead" not in str(caught[0].message)
        assert "2.0.0" in str(caught[0].message)

    def test_warning_points_at_the_caller_not_at_the_helper(self) -> None:
        @deprecated(**_kwargs())
        def old() -> None:
            return None

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            old()  # this line is the one the warning must name
        assert caught[0].filename == __file__
        assert caught[0].lineno == inspect.currentframe().f_lineno - 2  # type: ignore[union-attr]

    def test_instance_method_warns_at_the_caller(self) -> None:
        class Thing:
            @deprecated(**_kwargs())
            def old(self) -> str:
                return "ok"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert Thing().old() == "ok"
        assert caught[0].filename == __file__
        assert "Thing.old" in str(caught[0].message)

    def test_classmethod_and_staticmethod_keep_working(self) -> None:
        class Thing:
            @classmethod
            @deprecated(**_kwargs())
            def make(cls) -> str:
                return cls.__name__

            @staticmethod
            @deprecated(**_kwargs())
            def util(x: int) -> int:
                return x * 2

        with pytest.warns(PmlabDeprecationWarning):
            assert Thing.make() == "Thing"
        with pytest.warns(PmlabDeprecationWarning):
            assert Thing.util(4) == 8

    def test_decorator_above_classmethod_also_works(self) -> None:
        class Thing:
            @deprecated(**_kwargs())
            @classmethod
            def make(cls) -> str:
                return cls.__name__

        with pytest.warns(PmlabDeprecationWarning):
            assert Thing.make() == "Thing"

    def test_class_warns_on_construction_and_stays_the_same_class(self) -> None:
        @deprecated(**_kwargs())
        class Old:
            def __init__(self, x: int) -> None:
                self.x = x

        with pytest.warns(PmlabDeprecationWarning, match="Old"):
            obj = Old(3)
        assert obj.x == 3
        assert isinstance(obj, Old)
        assert inspect.isclass(Old)

    def test_dataclass_is_supported(self) -> None:
        import dataclasses

        @deprecated(**_kwargs())
        @dataclasses.dataclass
        class Old:
            x: int = 1

        with pytest.warns(PmlabDeprecationWarning):
            assert Old(2).x == 2

    def test_async_function_warns_at_call_and_stays_a_coroutine_function(self) -> None:
        @deprecated(**_kwargs())
        async def old(x: int) -> int:
            return x + 1

        assert inspect.iscoroutinefunction(old)
        with pytest.warns(PmlabDeprecationWarning):
            coro = old(1)
        import asyncio

        assert asyncio.run(coro) == 2

    def test_generator_function_keeps_yielding(self) -> None:
        @deprecated(**_kwargs())
        def old() -> Iterator[int]:
            yield 1
            yield 2

        with pytest.warns(PmlabDeprecationWarning):
            assert list(old()) == [1, 2]


class TestMetadataIsPreserved:
    def test_signature_name_and_wrapped_are_untouched(self) -> None:
        # The public-API snapshot reads signatures: decorating must not change them.
        def old(a: int, *, b: str = "x") -> bool:
            """Do the old thing."""
            return True

        before = inspect.signature(old)
        wrapped = deprecated(**_kwargs())(old)
        assert inspect.signature(wrapped) == before
        assert wrapped.__name__ == "old"
        assert wrapped.__wrapped__ is old  # type: ignore[attr-defined]

    def test_docstring_gains_a_notice_after_the_summary(self) -> None:
        @deprecated(**_kwargs())
        def old() -> None:
            """Do the old thing.

            Longer text that must survive.
            """

        doc = old.__doc__ or ""
        lines = doc.splitlines()
        assert lines[0] == "Do the old thing."
        assert "Deprecated since 1.2.0" in doc
        assert "removed in 2.0.0" in doc
        assert "use new_thing instead" in doc
        assert "Longer text that must survive." in doc
        assert doc.index("Deprecated since") < doc.index("Longer text")

    def test_docstring_notice_is_added_when_there_was_no_docstring(self) -> None:
        @deprecated(**_kwargs())
        def old() -> None:
            return None

        assert "Deprecated since 1.2.0" in (old.__doc__ or "")

    def test_class_docstring_gains_the_notice(self) -> None:
        @deprecated(**_kwargs())
        class Old:
            """An old class."""

        assert (Old.__doc__ or "").startswith("An old class.")
        assert "Deprecated since 1.2.0" in (Old.__doc__ or "")


class TestTimelineIsEnforced:
    @pytest.mark.parametrize(
        ("since", "removed_in", "why"),
        [
            ("1.2.0", "1.9.0", "same major"),
            ("1.2.0", "1.3.0", "next minor"),
            ("1.2.0", "2.1.0", "not a major release"),
            ("1.2.0", "2.0.1", "not a major release"),
            ("1.2.0", "1.0.0", "before since"),
            ("2.0.0", "2.0.0", "same release"),
            ("0.12.0", "1.0.0", "1.0.0 is the release that starts the policy"),
            ("0.12.0", "0.13.0", "pre-1.0 minor"),
        ],
    )
    def test_removal_is_a_later_major_and_never_before_2_0_0(
        self, since: str, removed_in: str, why: str
    ) -> None:
        with pytest.raises(ValueError, match="removed_in"):
            deprecated(since=since, removed_in=removed_in)(lambda: None)
        with pytest.raises(ValueError, match="removed_in"):
            warn_deprecated("x", since=since, removed_in=removed_in)

    @pytest.mark.parametrize("bad", ["", "2", "2.0", "two", "2.0.0rc1", "v2.0.0", "2.0.x"])
    def test_versions_must_be_plain_x_y_z(self, bad: str) -> None:
        with pytest.raises(ValueError, match="version"):
            deprecated(since=bad, removed_in="2.0.0")(lambda: None)
        with pytest.raises(ValueError, match="version"):
            deprecated(since="1.2.0", removed_in=bad)(lambda: None)

    def test_valid_timelines_are_accepted(self) -> None:
        deprecated(since="1.0.0", removed_in="2.0.0")(lambda: None)
        deprecated(since="1.9.3", removed_in="2.0.0")(lambda: None)
        deprecated(since="1.4.0", removed_in="3.0.0")(lambda: None)  # a longer notice is fine
        deprecated(since="0.12.0", removed_in="2.0.0")(lambda: None)  # carried over from 0.x


class TestWarnDeprecated:
    def test_it_covers_what_a_decorator_cannot(self) -> None:
        # A parameter, a keyword value, a module attribute, a file key...
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_deprecated(
                "the `min_edge_pct` argument",
                since="1.1.0",
                removed_in="2.0.0",
                alternative="`min_edge`",
            )
        assert len(caught) == 1
        assert issubclass(caught[0].category, PmlabDeprecationWarning)
        msg = str(caught[0].message)
        assert "the `min_edge_pct` argument is deprecated since 1.1.0" in msg
        assert "removed in 2.0.0" in msg
        assert "use `min_edge` instead" in msg
        assert caught[0].filename == __file__

    def test_stacklevel_lets_an_api_function_name_whoever_called_it(self) -> None:
        def api() -> None:
            warn_deprecated("x", since="1.1.0", removed_in="2.0.0", stacklevel=3)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            api()  # stacklevel 3 from inside api: this is the line to name
        assert caught[0].lineno == inspect.currentframe().f_lineno - 1  # type: ignore[union-attr]

        def api_default() -> None:
            warn_deprecated("x", since="1.1.0", removed_in="2.0.0")  # default: this line

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            api_default()
        assert caught[0].lineno == inspect.getsourcelines(api_default)[1] + 1

    def test_it_registers_too(self) -> None:
        with pytest.warns(PmlabDeprecationWarning):
            warn_deprecated("the `x` flag", since="1.1.0", removed_in="2.0.0")
            warn_deprecated("the `x` flag", since="1.1.0", removed_in="2.0.0")
        # Emitting twice (it is called on every use) must not grow the registry.
        assert [d.name for d in registered()] == ["the `x` flag"]


class TestRegistry:
    def test_decorating_registers_the_qualified_name_and_its_timeline(self) -> None:
        class Thing:
            @deprecated(**_kwargs())
            def old(self) -> None:
                return None

        (entry,) = registered()
        assert isinstance(entry, Deprecation)
        assert entry.name == "Thing.old"
        assert entry.module == __name__
        assert (entry.since, entry.removed_in, entry.alternative) == ("1.2.0", "2.0.0", "new_thing")

    def test_registered_is_a_read_only_snapshot(self) -> None:
        @deprecated(**_kwargs())
        def old() -> None:
            return None

        snapshot = registered()
        assert isinstance(snapshot, tuple)
        deprecated(**_kwargs())(lambda: None)
        assert len(snapshot) == 1  # the earlier snapshot did not grow

    def test_entries_are_immutable(self) -> None:
        import dataclasses

        entry = Deprecation("x", "1.2.0", "2.0.0", None, None)
        with pytest.raises(dataclasses.FrozenInstanceError):
            entry.since = "1.3.0"  # type: ignore[misc]


class TestCommandLine:
    """The terminal is the one surface where a Python warning is the wrong tool.

    A ``DeprecationWarning`` raised from inside pmlab is attributed to pmlab's own module, which
    the default filters hide, and a ``file:line: Category`` line means nothing to a person typing
    ``pmlab backtest``. A deprecated command uses Typer's own flag (it prints a message); a
    deprecated option, which Typer cannot flag, uses ``warn_deprecated(..., cli=True)``.
    """

    def test_cli_mode_prints_one_plain_line_to_stderr_and_no_python_warning(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_deprecated(
                "the --min-edge-pct option",
                since="1.1.0",
                removed_in="2.0.0",
                alternative="--min-edge",
                cli=True,
            )
        assert caught == []
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == (
            "warning: the --min-edge-pct option is deprecated since 1.1.0 and will be removed "
            "in 2.0.0; use --min-edge instead\n"
        )

    def test_cli_mode_still_registers_the_deprecation(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        warn_deprecated("the --x option", since="1.1.0", removed_in="2.0.0", cli=True)
        assert [d.name for d in registered()] == ["the --x option"]
        capsys.readouterr()

    def test_cli_mode_still_checks_the_timeline(self) -> None:
        with pytest.raises(ValueError, match="removed_in"):
            warn_deprecated("x", since="1.1.0", removed_in="1.5.0", cli=True)

    def test_a_deprecated_option_in_a_real_typer_command(self) -> None:
        import typer
        from typer.testing import CliRunner

        app = typer.Typer()

        @app.command()
        def run(
            old: str | None = typer.Option(None, "--old"),
            new: str = typer.Option("n", "--new"),
        ) -> None:
            if old is not None:
                warn_deprecated(
                    "--old", since="1.1.0", removed_in="2.0.0", alternative="--new", cli=True
                )
                new = old
            typer.echo(f"ran with {new}")

        runner = CliRunner()
        quiet = runner.invoke(app, ["--new", "a"])
        assert quiet.exit_code == 0 and "warning" not in quiet.output
        loud = runner.invoke(app, ["--old", "b"])
        assert loud.exit_code == 0  # a deprecated option still works
        assert "ran with b" in loud.output
        assert "warning: --old is deprecated since 1.1.0" in loud.output
        assert "use --new instead" in loud.output

    def test_a_deprecated_command_uses_typers_own_flag(self) -> None:
        # Documented in docs/deprecation.md: Typer prints the notice and runs the command.
        import typer
        from typer.testing import CliRunner

        app = typer.Typer()

        @app.command(deprecated=True)
        def old() -> None:
            typer.echo("old ran")

        @app.command()
        def new() -> None:
            typer.echo("new ran")

        result = CliRunner().invoke(app, ["old"])
        assert result.exit_code == 0
        assert "old ran" in result.output
        assert "deprecated" in result.output


class TestHowAUserTurnsThemIntoErrors:
    """The recipes in docs/deprecation.md, run for real."""

    def test_simplefilter_error_raises_at_the_call(self) -> None:
        @deprecated(**_kwargs())
        def old() -> int:
            return 1

        with warnings.catch_warnings():
            warnings.simplefilter("error", PmlabDeprecationWarning)
            with pytest.raises(PmlabDeprecationWarning, match="use new_thing instead"):
                old()

    def test_a_filter_on_pmlabs_class_leaves_other_deprecations_alone(self) -> None:
        @deprecated(**_kwargs())
        def old() -> int:
            return 1

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.simplefilter("error", PmlabDeprecationWarning)
            warnings.warn("someone else's", DeprecationWarning, stacklevel=1)
            with pytest.raises(PmlabDeprecationWarning):
                old()
        assert [str(w.message) for w in caught] == ["someone else's"]

    def test_pytest_dash_w_with_the_public_path(self, tmp_path: Path) -> None:
        (tmp_path / "test_user.py").write_text(
            "from pmlab._deprecation import deprecated\n"
            "@deprecated(since='1.2.0', removed_in='2.0.0', alternative='new')\n"
            "def old(): return 1\n"
            "def test_calls_old(): old()\n"
        )
        (tmp_path / "pytest.ini").write_text("[pytest]\n")

        def run(*extra: str) -> subprocess.CompletedProcess[str]:
            return subprocess.run(  # noqa: S603 - our own interpreter
                [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *extra],
                cwd=tmp_path,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )

        assert run().returncode == 0  # a warning by default: the test passes
        strict = run("-W", "error::pmlab.PmlabDeprecationWarning")
        assert strict.returncode == 1, strict.stdout
        assert "PmlabDeprecationWarning" in strict.stdout


class TestDefaultVisibility:
    """Why DeprecationWarning and not FutureWarning: who sees it by default.

    Python shows a DeprecationWarning when the code that *triggers* it is `__main__` (a script,
    a notebook cell) and hides it when the trigger is another library's code. With `stacklevel`
    pointing at the caller, that is exactly the audience that can act on it.
    """

    @staticmethod
    def _run(code: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603 - our own interpreter, our own code
            [sys.executable, "-I", "-c", textwrap.dedent(code)],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )

    def test_a_script_sees_it_and_a_library_in_between_does_not(self) -> None:
        proc = self._run(
            """
            import sys, types
            from pmlab._deprecation import deprecated

            @deprecated(since="1.2.0", removed_in="2.0.0", alternative="new_thing")
            def old():
                return 1

            lib = types.ModuleType("some_other_library")
            exec("def call(f):\\n    return f()", lib.__dict__)

            print("script call:", file=sys.stderr)
            old()
            print("library call:", file=sys.stderr)
            lib.call(old)
            """
        )
        assert proc.returncode == 0, proc.stderr
        before_library = proc.stderr.split("library call:")[0]
        after_library = proc.stderr.split("library call:")[1]
        assert "PmlabDeprecationWarning" in before_library
        assert "use new_thing instead" in before_library
        assert "DeprecationWarning" not in after_library


class TestNoDeprecationOutlivesItsRemoval:
    def test_every_registered_deprecation_is_still_before_its_removal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reminder: when the package reaches `removed_in`, the shim must be deleted.

        Uses the real registry (not the isolated one): every deprecation shipped in pmlab.
        """
        monkeypatch.undo()  # drop the empty-registry fixture
        current = _deprecation.parse_version(pmlab.__version__)
        overdue = [
            f"{d.name} (removed_in {d.removed_in})"
            for d in registered()
            if current >= _deprecation.parse_version(d.removed_in)
        ]
        assert not overdue, f"pmlab {pmlab.__version__} has reached removal: delete {overdue}"

    def test_the_overdue_check_would_fire(self) -> None:
        deprecated(since="1.0.0", removed_in="2.0.0")(lambda: None)
        assert _deprecation.parse_version("2.0.0") >= _deprecation.parse_version(
            registered()[0].removed_in
        )
        assert _deprecation.parse_version("1.99.99") < _deprecation.parse_version("2.0.0")
