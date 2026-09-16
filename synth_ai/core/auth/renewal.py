"""Credential renewal and revocation for long-running CLI and MCP sessions.

A process that runs for hours cannot hold one key forever: keys get rotated, and
revoked. A ``RenewableCredential`` reads its value from an explicit source at
each use, so a rotated key is picked up without a restart. When the backend
rejects a credential (HTTP 401), the caller re-reads the source once:

- a different value means the key was rotated; the rejected request never ran,
  so it is repeated once with the new value;
- the same value means the key was revoked or expired; that is reported as
  ``CredentialRevokedError`` with what to do, never retried in a loop.

An authorization refusal (HTTP 403) is a permission decision, not a credential
problem, and passes through unchanged. Nothing here discovers credentials from
home directories or keychains: the source is exactly what the caller configured.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from synth_ai.core.auth.credentials import ApiCredential
from synth_ai.core.errors import AuthenticationError, SynthError, SynthErrorCategory

T = TypeVar("T")

_MAX_CREDENTIAL_FILE_BYTES = 4096


class CredentialRevokedError(AuthenticationError):
    """The backend rejected a credential and its source offers no replacement."""


def is_authentication_failure(error: BaseException) -> bool:
    """True when the backend refused the credential itself (HTTP 401).

    A locally raised ``AuthenticationError`` (no key configured, unreadable
    file) is not a backend rejection and is not renewable.
    """
    if not isinstance(error, SynthError) or isinstance(error, CredentialRevokedError):
        return False
    failure = getattr(error, "failure", None)
    return getattr(failure, "category", None) == SynthErrorCategory.AUTHENTICATION or (
        getattr(error, "status", None) == 401
    )


def static_source(value: str, description: str) -> Callable[[], str]:
    """A fixed value captured at startup; renewal can only report revocation.

    Environment variables are captured once on purpose: an agent must not be
    able to change a running server's identity. Use ``file_source`` when keys
    rotate during a session.
    """

    def read() -> str:
        return value

    read.description = description  # type: ignore[attr-defined]
    return read


def file_source(path: str | os.PathLike[str]) -> Callable[[], str]:
    """Read a credential file at each use, so rotation needs no restart.

    The file must be a regular file (not a link), readable only by its owner,
    and small; its content is the key, surrounding whitespace ignored.
    """
    location = Path(path).expanduser()

    def read() -> str:
        try:
            info = location.lstat()
        except FileNotFoundError as error:
            raise AuthenticationError(f"Credential file {location} does not exist") from error
        if not stat.S_ISREG(info.st_mode):
            raise AuthenticationError(f"Credential file {location} must be a regular file")
        if os.name == "posix" and info.st_mode & 0o077:
            raise AuthenticationError(
                f"Credential file {location} is readable by other users; chmod 600 it"
            )
        if info.st_size > _MAX_CREDENTIAL_FILE_BYTES:
            raise AuthenticationError(f"Credential file {location} is too large")
        return location.read_text(encoding="utf-8")

    read.description = f"credential file {location}"  # type: ignore[attr-defined]
    return read


class RenewableCredential:
    """A credential re-read from its source at every use.

    ``current()`` records the value it handed out; ``renew()`` re-reads the
    source after a rejection and either returns a different value or raises
    ``CredentialRevokedError``.
    """

    def __init__(self, source: Callable[[], str], *, description: str | None = None) -> None:
        self._source = source
        self._description = description or getattr(source, "description", "configured source")
        self._last: str | None = None

    @property
    def description(self) -> str:
        return self._description

    def _read(self) -> ApiCredential:
        value = (self._source() or "").strip()
        if not value:
            raise AuthenticationError(f"No Synth API credential in {self._description}")
        return ApiCredential(value)

    def current(self) -> ApiCredential:
        credential = self._read()
        self._last = credential.value
        return credential

    def renew(self) -> ApiCredential:
        """Re-read after a rejection; the same value means it was revoked."""
        rejected = self._last
        credential = self._read()
        if credential.value == rejected:
            raise CredentialRevokedError(
                "The Synth backend rejected the API credential from "
                f"{self._description}; it is revoked or expired. Configure a valid "
                "key and retry (a key file named by SYNTH_API_KEY_FILE or "
                "--api-key-file is re-read without a restart)."
            )
        self._last = credential.value
        return credential


def call_with_renewal(
    credential: RenewableCredential, operation: Callable[[ApiCredential], T]
) -> T:
    """Run ``operation`` with the current credential, renewing once on a 401.

    A 401 means the backend did not execute the request, so repeating it once
    with a rotated key cannot duplicate work. Multi-request operations must be
    resumable on their own (the Index intake and upload flows are).
    """
    try:
        return operation(credential.current())
    except Exception as error:
        if not is_authentication_failure(error):
            raise
        renewed = credential.renew()
    try:
        return operation(renewed)
    except Exception as error:
        if not is_authentication_failure(error):
            raise
        raise CredentialRevokedError(
            f"The Synth backend also rejected the replacement credential from "
            f"{credential.description}; it is revoked or expired."
        ) from error


__all__ = [
    "CredentialRevokedError",
    "RenewableCredential",
    "call_with_renewal",
    "file_source",
    "is_authentication_failure",
    "static_source",
]
