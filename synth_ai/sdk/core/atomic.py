"""Publish verified directories without replacing an existing destination.

Darwin flags: apple-oss-distributions/xnu, bsd/sys/stdio.h.
Linux flags: linux/include/uapi/linux/fs.h and rename(2).
"""

from __future__ import annotations

import ctypes
import errno
import os
import sys
from pathlib import Path

from synth_ai.errors import ContractError


def publish_directory(source: Path, destination: Path) -> None:
    """Move a verified directory on the same filesystem with an exclusive rename."""
    if os.name == "nt":
        os.rename(source, destination)
        return
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        name, flags, directory_fd = "renamex_np", 0x4, None
    elif sys.platform.startswith("linux"):
        name, flags, directory_fd = "renameat2", 0x1, -100
    else:
        raise ContractError(
            "Exclusive directory publication is unavailable on this platform",
            code="directory_publication_unsupported",
            stage="download_publish",
        )
    function = getattr(library, name, None)
    if function is None:
        raise ContractError(
            "Exclusive directory publication is unavailable in this C library",
            code="directory_publication_unsupported",
            stage="download_publish",
        )
    function.restype = ctypes.c_int
    if directory_fd is None:
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        result = function(os.fsencode(source), os.fsencode(destination), flags)
    else:
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        result = function(
            directory_fd, os.fsencode(source), directory_fd, os.fsencode(destination), flags
        )
    if result != 0:
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(error, "Download destination already exists", str(destination))
        raise OSError(error, os.strerror(error), str(destination))
