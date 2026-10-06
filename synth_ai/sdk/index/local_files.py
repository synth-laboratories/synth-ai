"""Explicit, bounded local bytes shared by Index SDK, CLI and MCP uploads.

# See: synth_ai/core/local_files.py (descriptor confinement)
"""

import re
from collections.abc import Mapping

from synth_ai.core.local_files import SelectedFileReader

_UPLOAD_MAX_BYTES = 64 * 1024 * 1024
_CREDENTIAL = re.compile(
    rb"(sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY"
    rb"|ghp_[A-Za-z0-9]{30,}|xox[bpa]-[A-Za-z0-9-]{10,})"
)


def read_selected_files(root: str, files: Mapping[str, str]) -> dict[str, bytes]:
    """Read exactly the listed regular files; reject escapes, symlinks and secrets."""
    content: dict[str, bytes] = {}
    total = 0
    with SelectedFileReader(root) as reader:
        for logical_path, relative in files.items():
            data = reader.read(relative, _UPLOAD_MAX_BYTES - total)
            total += len(data)
            if _CREDENTIAL.search(data):
                raise ValueError(f"{logical_path}: possible credential; remove it before upload")
            content[logical_path] = data
    return content


__all__ = ["read_selected_files"]
