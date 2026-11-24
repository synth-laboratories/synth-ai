"""Explicit local stdio filesystem capability; never part of the hosted catalog."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from synth_ai.errors import (
    DirectoryUploadCancelledError,
    DirectoryUploadError,
    InvalidRequestError,
    local_problem,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from pydantic import JsonValue

    from synth_ai.aclient import AsyncSynthClient

NAME = "swarms_upload_directory"
DESCRIPTION = (
    "Upload the explicitly granted local directory, retaining relative names and retry custody."
)
SCHEMA = {
    "type": "object",
    "properties": {"idempotency_key": {"type": "string", "minLength": 1, "maxLength": 255}},
    "required": ["idempotency_key"],
    "additionalProperties": False,
}


@contextmanager
def grant_root(root: str | None) -> Generator[int | None]:
    """Pin every ancestor without following links; retain only the granted root FD."""
    if root is None:
        yield None
        return
    path = Path(root)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("upload root must be an absolute path without parent traversal")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(path.anchor, flags)
    try:
        for segment in path.parts[1:]:
            child = os.open(segment, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


async def upload(client: AsyncSynthClient, descriptor: int, arguments: dict[str, JsonValue]) -> str:
    key = arguments.get("idempotency_key")
    if (
        set(arguments) != {"idempotency_key"}
        or not isinstance(key, str)
        or not key.strip()
        or len(key.encode("utf-8")) > 255
    ):
        raise InvalidRequestError(local_problem("tool_arguments_invalid", "mcp_input", status=422))
    files = await client.swarms.upload_directory(descriptor, idempotency_key=key)
    return json.dumps({"idempotency_key": key, "files": [f.model_dump(mode="json") for f in files]})


def failure(error: DirectoryUploadError | DirectoryUploadCancelledError) -> str:
    return json.dumps(
        {
            "code": error.code,
            "name": error.name,
            "idempotency_key": error.idempotency_key,
            "file_idempotency_key": error.file_idempotency_key,
            "completed_files": [f.model_dump(mode="json") for f in error.completed_files],
        }
    )
