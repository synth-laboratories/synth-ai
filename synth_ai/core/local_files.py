"""Bounded local regular-file reads. See SDK_0230_IMPLEMENTATION.md."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import BinaryIO


def open_regular_file(path: Path) -> BinaryIO:
    """Open without following the final symlink or blocking on a special file."""
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode):
        raise ValueError("upload source must not be a symlink")
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise ValueError("upload source changed while opening")
        if not stat.S_ISREG(after.st_mode):
            raise ValueError("upload source must be a regular file")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise
