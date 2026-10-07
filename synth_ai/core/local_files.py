"""Selected local upload bytes, confined by directory descriptors on POSIX.

See sdk/index/README.md: uploads never scan a workspace. Opening each component
without following links closes the validation/read race; the read itself is
bounded even when a writer grows the file after inspection. Unsupported hosts
fail explicitly rather than silently weakening confinement.
"""

import errno
import os
import stat
from pathlib import Path


class SelectedFileReader:
    """Own one explicitly selected root descriptor for a bounded upload batch."""

    def __init__(self, root: str) -> None:
        if os.open not in os.supports_dir_fd or not hasattr(os, "O_NOFOLLOW"):
            raise ValueError("Confined Index uploads require POSIX directory-relative file access")
        base = Path(root).expanduser().resolve(strict=True)
        self._descriptor = os.open(base, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)

    def __enter__(self) -> "SelectedFileReader":
        return self

    def __exit__(self, *_error: object) -> None:
        os.close(self._descriptor)

    def read(self, relative: str, remaining_bytes: int) -> bytes:
        path = Path(relative)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise ValueError("Upload paths must be relative files inside the selected root")
        if len(path.parts) > 128 or remaining_bytes < 0:
            raise ValueError("Selected upload path or byte budget exceeds its bound")
        directory = os.dup(self._descriptor)
        try:
            for component in path.parts[:-1]:
                child = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory,
                )
                os.close(directory)
                directory = child
            descriptor = os.open(
                path.parts[-1],
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=directory,
            )
            try:
                information = os.fstat(descriptor)
                if not stat.S_ISREG(information.st_mode):
                    raise ValueError("Selected upload must be a regular file")
                if information.st_size > remaining_bytes:
                    raise ValueError("Selected files exceed the 64 MiB in-memory upload bound")
                with os.fdopen(descriptor, "rb", closefd=False) as stream:
                    content = stream.read(remaining_bytes + 1)
                if len(content) > remaining_bytes:
                    raise ValueError("Selected files exceed the 64 MiB in-memory upload bound")
                return content
            finally:
                os.close(descriptor)
        except OSError as error:
            # Preserve the upload boundary's refusal type for symlink/non-directory
            # traversal; missing files and infrastructure failures keep their cause.
            if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise ValueError(
                    "Upload paths cannot traverse symlinks or non-directories"
                ) from error
            raise
        finally:
            os.close(directory)
