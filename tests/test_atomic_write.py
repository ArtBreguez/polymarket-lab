"""`atomic_write_text`: a file is replaced whole or not at all (#68).

`PaperBroker.record` and `SettlementEngine.settle_all` rewrote the whole trade log with
``Path.write_text``, which truncates the file first: a crash, a full disk or a second reader in
the middle left an empty or half-written log, and the log is the only record of the paper trades.
"""

from __future__ import annotations

import errno
import os
import stat
from pathlib import Path

import pytest

from pmlab._atomic import atomic_write_text


def _names(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


class TestWriting:
    def test_it_creates_and_replaces(self, tmp_path: Path) -> None:
        target = tmp_path / "trades.json"
        atomic_write_text(target, '{"trades": []}')
        assert target.read_text() == '{"trades": []}'
        atomic_write_text(target, '{"trades": [1]}')
        assert target.read_text() == '{"trades": [1]}'

    def test_it_leaves_nothing_but_the_file(self, tmp_path: Path) -> None:
        atomic_write_text(tmp_path / "trades.json", "x")
        assert _names(tmp_path) == ["trades.json"]

    def test_text_is_written_as_utf8(self, tmp_path: Path) -> None:
        target = tmp_path / "t.json"
        atomic_write_text(target, '{"city": "São Paulo °C"}')
        assert target.read_bytes() == '{"city": "São Paulo °C"}'.encode()

    def test_a_missing_directory_is_an_error_not_a_silent_mkdir(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            atomic_write_text(tmp_path / "nope" / "t.json", "x")


class TestPermissions:
    def test_a_new_file_gets_the_mode_write_text_would_have_given_it(self, tmp_path: Path) -> None:
        old = os.umask(0o022)
        try:
            atomic_write_text(tmp_path / "new.json", "x")
        finally:
            os.umask(old)
        assert stat.S_IMODE((tmp_path / "new.json").stat().st_mode) == 0o644

    def test_an_existing_files_mode_is_kept(self, tmp_path: Path) -> None:
        target = tmp_path / "t.json"
        target.write_text("old")
        target.chmod(0o640)
        atomic_write_text(target, "new")
        assert stat.S_IMODE(target.stat().st_mode) == 0o640

    def test_a_symlink_is_followed_not_replaced(self, tmp_path: Path) -> None:
        real = tmp_path / "real.json"
        real.write_text("old")
        link = tmp_path / "link.json"
        link.symlink_to(real)
        atomic_write_text(link, "new")
        assert link.is_symlink() and real.read_text() == "new" and link.read_text() == "new"


class TestAFailureLeavesTheOldFile:
    @pytest.mark.parametrize("failing", ["fsync", "replace"])
    def test_the_original_survives_and_no_temp_file_is_left(
        self, tmp_path: Path, monkeypatch, failing: str
    ) -> None:
        target = tmp_path / "trades.json"
        target.write_text("ORIGINAL")

        def boom(*args, **kwargs):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(os, failing, boom)
        with pytest.raises(OSError, match="No space"):
            atomic_write_text(target, "NEW CONTENT THAT NEVER LANDS")
        monkeypatch.undo()
        assert target.read_text() == "ORIGINAL"
        assert _names(tmp_path) == ["trades.json"]

    def test_a_failure_on_a_new_file_leaves_no_file_at_all(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        def boom(*args, **kwargs):
            raise OSError("boom")

        monkeypatch.setattr(os, "replace", boom)
        with pytest.raises(OSError):
            atomic_write_text(tmp_path / "t.json", "x")
        monkeypatch.undo()
        assert _names(tmp_path) == []

    def test_a_failing_directory_sync_does_not_fail_the_write(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        real_fsync = os.fsync
        directories: list[int] = []

        def fsync(fd: int) -> None:
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                directories.append(fd)
                raise OSError("this filesystem cannot sync a directory")
            real_fsync(fd)

        monkeypatch.setattr(os, "fsync", fsync)
        atomic_write_text(tmp_path / "t.json", "x")
        monkeypatch.undo()
        assert directories, "the directory was meant to be synced"
        assert (tmp_path / "t.json").read_text() == "x"


class TestWhereARenameIsRefused:
    """Windows will not rename over a file another process has open (``PermissionError``) and a
    Docker single-file bind mount will not be renamed over (``EBUSY``). ``write_text`` worked on
    both, so the write falls back to it, with a warning, instead of failing the caller."""

    REFUSALS = [
        PermissionError(errno.EACCES, "Access is denied"),
        PermissionError(errno.EPERM, "Operation not permitted"),
        OSError(errno.EBUSY, "Device or resource busy"),
    ]

    @staticmethod
    def _refuse_the_rename(monkeypatch, error: OSError) -> None:
        def refuse(*args, **kwargs):
            raise error

        monkeypatch.setattr(os, "replace", refuse)

    @pytest.mark.parametrize("error", REFUSALS, ids=lambda e: type(e).__name__ + str(e.errno))
    def test_the_content_is_written_in_place_and_it_warns(
        self, tmp_path: Path, monkeypatch, caplog, error: OSError
    ) -> None:
        target = tmp_path / "trades.json"
        target.write_text("ORIGINAL")
        self._refuse_the_rename(monkeypatch, error)
        with caplog.at_level("WARNING", logger="pmlab"):
            atomic_write_text(target, "NEW CONTENT")
        monkeypatch.undo()
        assert target.read_text() == "NEW CONTENT"
        assert _names(tmp_path) == ["trades.json"]  # the temp file is gone
        warned = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert len(warned) == 1 and "in place" in warned[0] and "trades.json" in warned[0], warned

    def test_a_new_file_falls_back_too(self, tmp_path: Path, monkeypatch, caplog) -> None:
        self._refuse_the_rename(monkeypatch, OSError(errno.EBUSY, "busy"))
        with caplog.at_level("WARNING", logger="pmlab"):
            atomic_write_text(tmp_path / "t.json", "x")
        monkeypatch.undo()
        assert (tmp_path / "t.json").read_text() == "x" and _names(tmp_path) == ["t.json"]

    def test_the_mode_of_an_existing_file_is_kept(self, tmp_path: Path, monkeypatch) -> None:
        target = tmp_path / "trades.json"
        target.write_text("ORIGINAL")
        os.chmod(target, 0o640)
        self._refuse_the_rename(monkeypatch, PermissionError(errno.EACCES, "denied"))
        atomic_write_text(target, "NEW")
        monkeypatch.undo()
        assert stat.S_IMODE(target.stat().st_mode) == 0o640 and target.read_text() == "NEW"

    def test_a_fallback_that_fails_too_is_raised_and_keeps_the_temp_file_it_names(
        self, tmp_path: Path, monkeypatch, caplog
    ) -> None:
        """The temporary file is the only complete copy of the new content until the write in
        place has succeeded: it is kept, and the error says where it is."""
        target = tmp_path / "trades.json"
        target.write_text("ORIGINAL")
        self._refuse_the_rename(monkeypatch, PermissionError(errno.EACCES, "denied"))

        def locked(self, *args, **kwargs):
            raise PermissionError(errno.EACCES, "the file is locked")

        monkeypatch.setattr(Path, "write_text", locked)
        with (
            caplog.at_level("WARNING", logger="pmlab"),
            pytest.raises(PermissionError, match="locked") as err,
        ):
            atomic_write_text(target, "NEW")
        monkeypatch.undo()
        kept = [tmp_path / n for n in _names(tmp_path) if n != "trades.json"]
        assert len(kept) == 1, _names(tmp_path)
        assert kept[0].name.startswith(".trades.json.") and kept[0].name.endswith(".tmp")
        assert kept[0].read_text() == "NEW" and target.read_text() == "ORIGINAL"
        assert str(kept[0]) in str(err.value), str(err.value)
        assert "wrote it in place" not in caplog.text  # it did not

    def test_a_write_in_place_that_dies_half_way_leaves_the_new_content_in_the_temp_file(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A full disk after the file was truncated: the file is empty, as ``write_text`` would
        leave it, but the new content is not lost with it."""
        target = tmp_path / "trades.json"
        target.write_text("ORIGINAL")
        self._refuse_the_rename(monkeypatch, PermissionError(errno.EACCES, "denied"))

        def full(self, *args, **kwargs):
            self.write_bytes(b"")  # what open(..., "w") does before the first byte is written
            raise OSError(errno.ENOSPC, "No space left on device")

        monkeypatch.setattr(Path, "write_text", full)
        with pytest.raises(OSError, match="No space left") as err:
            atomic_write_text(target, "NEW CONTENT")
        monkeypatch.undo()
        assert err.value.errno == errno.ENOSPC
        assert target.read_text() == ""
        kept = [tmp_path / n for n in _names(tmp_path) if n != "trades.json"]
        assert [k.read_text() for k in kept] == ["NEW CONTENT"]
        assert str(kept[0]) in str(err.value)

    def test_the_warning_comes_after_the_write_it_reports_and_the_temp_file_goes(
        self, tmp_path: Path, monkeypatch, caplog
    ) -> None:
        target = tmp_path / "trades.json"
        target.write_text("ORIGINAL")
        self._refuse_the_rename(monkeypatch, PermissionError(errno.EACCES, "denied"))
        said_before_the_write: list[int] = []
        real_write_text = Path.write_text

        def spy(self, *args, **kwargs):
            said_before_the_write.append(len(caplog.records))
            return real_write_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", spy)
        with caplog.at_level("WARNING", logger="pmlab"):
            atomic_write_text(target, "NEW")
        monkeypatch.undo()
        assert said_before_the_write == [0]  # "wrote it in place" was not said in advance
        assert "wrote it in place" in caplog.text
        assert target.read_text() == "NEW" and _names(tmp_path) == ["trades.json"]

    @pytest.mark.parametrize("code", [errno.ENOSPC, errno.EIO, errno.EXDEV])
    def test_any_other_failure_of_the_rename_still_leaves_the_old_file(
        self, tmp_path: Path, monkeypatch, caplog, code: int
    ) -> None:
        target = tmp_path / "trades.json"
        target.write_text("ORIGINAL")
        self._refuse_the_rename(monkeypatch, OSError(code, "nope"))
        with caplog.at_level("WARNING", logger="pmlab"), pytest.raises(OSError):
            atomic_write_text(target, "NEW")
        monkeypatch.undo()
        assert target.read_text() == "ORIGINAL" and _names(tmp_path) == ["trades.json"]
        assert "in place" not in caplog.text
