"""Bounded inputs and private, create-only local receipts.

See notes/specifications/synth-index/research-archive-release.md.
Files contain research evidence, never credential discovery or implicit consent.
"""

import json
import os
import stat
import tempfile
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

Model = TypeVar("Model", bound=BaseModel)
CONTRACT_BYTES_MAX = 1_048_576


def read_contract_file(path: Path, model: type[Model]) -> Model:
    """Validate one explicitly selected, bounded contract before any HTTP call.

    Args:
        path: Explicit regular contract or private receipt path.
        model: Pydantic model used to validate the selected contract.

    Returns:
        Model: Validated instance of the requested Pydantic model.

    Raises:
        ValueError: Input bounds, validation, duplicate fields or private receipt invariants fail.
        OSError: The selected regular input or private destination cannot be accessed.

    Examples:
        result = read_contract_file(path, model)
    """
    try:
        payload = json.loads(read_input_file(path), object_pairs_hook=_unique_fields)
        return model.model_validate(payload)
    except (ValueError, UnicodeDecodeError, ValidationError) as error:
        # ValidationError includes input values: private native evidence must not
        # become terminal/log output when a typed boundary refuses it.
        raise ValueError(f"Contract input does not match {model.__name__}") from error


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate contract field")
        result[key] = value
    return result


def read_input_file(path: Path) -> bytes:
    """Read one regular input with the CLI's 1 MiB contract bound.

    Args:
        path: Explicit regular contract or private receipt path.

    Returns:
        bytes: Selected regular file bytes within the 1 MiB bound.

    Raises:
        ValueError: Input bounds, validation, duplicate fields or private receipt invariants fail.
        OSError: The selected regular input or private destination cannot be accessed.

    Examples:
        result = read_input_file(path)
    """
    if path.is_symlink() or not path.is_file():
        raise ValueError("Contract input must be a regular file, not a link")
    with path.open("rb") as source:
        raw = source.read(CONTRACT_BYTES_MAX + 1)
    if len(raw) > CONTRACT_BYTES_MAX:
        raise ValueError("Contract input exceeds 1 MiB")
    return raw


def write_private_receipt(path: Path, value: object) -> None:
    """Atomically create a 0600 receipt; identical retries preserve the same file.

    Changed results never overwrite earlier evidence. A failed API call must not
    invoke this function. No directory or uploaded file is made public here.

    Args:
        path: Explicit regular contract or private receipt path.
        value: Value serialized into canonical bytes or a private receipt.

    Returns:
        None: Creates a private receipt or retains identical existing evidence; never overwrites changed evidence.

    Raises:
        ValueError: Input bounds, validation, duplicate fields or private receipt invariants fail.
        OSError: The selected regular input or private destination cannot be accessed.

    Examples:
        result = write_private_receipt(path, value)
    """
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    raw = (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode()
    if len(raw) > CONTRACT_BYTES_MAX:
        raise ValueError("Receipt exceeds 1 MiB")
    if path.is_symlink():
        raise ValueError("Receipt destination may not be a link")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=".research-receipt-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(raw)
            destination.flush()
            os.fsync(destination.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            existing = os.open(path, flags)
            with os.fdopen(existing, "rb") as source:
                metadata = os.fstat(source.fileno())
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
                    raise ValueError("Existing receipt is not a private regular file") from None
                if source.read(CONTRACT_BYTES_MAX + 1) != raw:
                    raise ValueError("Receipt destination contains different evidence") from None
    finally:
        os.unlink(temporary)
