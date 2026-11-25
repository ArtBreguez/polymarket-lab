"""Replace a file whole or not at all (internal).

``Path.write_text`` truncates the file and then writes it: a crash, a full disk or a reader in
the middle leaves it empty or half written. For a file that is the only record of something (the
paper-trade log) the safe way is to write the new content beside it, flush it to disk and rename
it over the old one: the rename is atomic, so a reader and a crash see the old file or the new
one, never a mixture.
"""

from __future__ import annotations

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


def atomic_write_text(path: Path | str, text: str) -> None:
    """Write *text* (UTF-8) to *path*, replacing what is there atomically.

    The content goes to a temporary file in the same directory (so the rename stays on one
    filesystem), is flushed and synced, and replaces *path* with ``os.replace``. If anything
    fails the temporary file is removed and *path* is untouched. A new file gets the mode
    ``write_text`` would have given it (``0o666`` less the umask); an existing file keeps its
    mode. A symlink is followed: the file it points to is the one replaced.

    Nothing locks the file: two writers that read, change and write the same file still lose one
    another's changes, they just cannot corrupt it.

    Raises:
        FileNotFoundError: If the directory does not exist.
        OSError: If the content cannot be written or synced, or renamed into place.
    """
    target = Path(os.path.realpath(path))
    directory = target.parent
    temp = directory / f".{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if target.exists():
            shutil.copymode(target, temp)
        os.replace(temp, target)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    _sync_directory(directory)
