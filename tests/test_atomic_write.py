"""`atomic_write_text`: a file is replaced whole or not at all (#68).

`PaperBroker.record` and `SettlementEngine.settle_all` rewrote the whole trade log with
``Path.write_text``, which truncates the file first: a crash, a full disk or a second reader in
the middle left an empty or half-written log, and the log is the only record of the paper trades.
"""

from __future__ import annotations

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
