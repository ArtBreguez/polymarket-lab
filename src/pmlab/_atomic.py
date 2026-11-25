"""Replace a file whole or not at all (internal).

``Path.write_text`` truncates the file and then writes it: a crash, a full disk or a reader in
the middle leaves it empty or half written. For a file that is the only record of something (the
paper-trade log) the safe way is to write the new content beside it, flush it to disk and rename
it over the old one: the rename is atomic, so a reader and a crash see the old file or the new
one, never a mixture.
"""

from __future__ import annotations

import errno
import os
import shutil
import uuid
from pathlib import Path

__all__ = ["atomic_write_text"]


def _sync_directory(directory: Path) -> None:
    """Flush a directory's entries (a rename) to disk, where the platform allows it.

    Not every platform or filesystem can sync a directory; the file's data is safe by then, so a
    refusal is not an error.
    """
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _rename_refused(error: OSError) -> bool:
    """True for a refusal of the rename that a plain write would not have met.

    Windows will not rename over a file another process has open (``PermissionError``), and a
    Docker single-file bind mount cannot be renamed over (``EBUSY``). A full disk or an I/O error
    is not one of these: a plain write would fail too.
    """
    return isinstance(error, PermissionError) or error.errno == errno.EBUSY


def atomic_write_text(path: Path | str, text: str) -> None:
    """Write *text* (UTF-8) to *path*, replacing what is there atomically.

    The content goes to a temporary file in the same directory (so the rename stays on one
    filesystem), is flushed and synced, and replaces *path* with ``os.replace``. If anything
    fails the temporary file is removed and *path* is untouched. A new file gets the mode
    ``write_text`` would have given it (``0o666`` less the umask); an existing file keeps its
    mode. A symlink is followed: the file it points to is the one replaced.

    Where the rename itself is refused (``PermissionError``, or ``EBUSY``: a file another process
    holds open on Windows, a Docker single-file bind mount) the content is written in place with
    ``write_text``, which is what the file got before, and a warning (logged once that write has
    succeeded) says that it was not atomic. The temporary file is removed after that write only: if
    it fails too (a full disk after the file was truncated, say) the temporary file is kept, because
    it holds the complete new content, and the error names it. Any other failure leaves the old
    file as it was.

    Differences from ``write_text`` that stay: the file is replaced, so a hard link to it keeps
    the old content; a directory that is not writable fails even when the file is (the temporary
    file is made beside it); a read-only file in a writable directory is replaced, not refused
    (it keeps its mode); and a process killed in the middle leaves a hidden ``.*.tmp`` file.

    Nothing locks the file: two writers that read, change and write the same file still lose one
    another's changes, they just cannot corrupt it.

    Raises:
        FileNotFoundError: If the directory does not exist.
        OSError: If the content cannot be written or synced, or renamed into place (and the
            rename was not one of the refusals above, or the write in place failed too: that
            error names the temporary file that still holds the new content).
    """
    target = Path(os.path.realpath(path))
    directory = target.parent
    temp = directory / f".{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    refused: OSError | None = None
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if target.exists():
            shutil.copymode(target, temp)
        try:
            os.replace(temp, target)
        except OSError as error:
            if not _rename_refused(error):
                raise
            refused = error
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    if refused is not None:
        # The temporary file is the one complete copy of the new content until the write in place
        # has succeeded: a failure after write_text truncated the file must not lose both.
        try:
            target.write_text(text, encoding="utf-8")
        except OSError as error:
            raise OSError(
                error.errno,
                f"{error.strerror or error} (the rename was refused, and so was the write in "
                f"place; the complete new content is kept in {temp})",
                error.filename,
            ) from error
        temp.unlink(missing_ok=True)
        from pmlab.logging import get_logger

        get_logger(__name__).warning(
            "atomic_write_text: %s could not be replaced by a rename (%s); wrote it in place "
            "instead, so a crash during that write could have left it half written.",
            target,
            refused,
        )
        return
    _sync_directory(directory)
