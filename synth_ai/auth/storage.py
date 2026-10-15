"""Private credential storage. See backend auth-023.md, CLI login."""

from __future__ import annotations

import errno
import importlib
import json
import os
import stat
import tempfile
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import ContextManager, Protocol, cast


def _lock(descriptor: int, *, release: bool = False) -> None:
    if os.name == "nt":
        module = importlib.import_module("msvcrt")
        os.lseek(descriptor, 0, os.SEEK_SET)
        module.locking(descriptor, module.LK_UNLCK if release else module.LK_NBLCK, 1)
    else:
        module = importlib.import_module("fcntl")
        module.flock(descriptor, module.LOCK_UN if release else module.LOCK_EX | module.LOCK_NB)


@dataclass(frozen=True, slots=True)
class Credentials:
    access_token: str = field(repr=False)
    refresh_token: str | None = field(default=None, repr=False)
    expires_at: float | None = None
    issuer: str | None = None
    client_id: str | None = None
    token_endpoint: str | None = None
    revocation_endpoint: str | None = None
    scopes: tuple[str, ...] = ()
    resource: str | None = None
    token_exchange_required: bool = False
    source_access_token: str | None = field(default=None, repr=False)
    source_scopes: tuple[str, ...] = ()
    api_origin: str | None = None
    resource_exchange_url: str | None = None

    def __post_init__(self) -> None:
        if not self.access_token or any(c.isspace() for c in self.access_token):
            raise ValueError("Malformed stored credential")
        if self.expires_at is not None:
            import math

            if not math.isfinite(self.expires_at) or self.expires_at <= 0:
                raise ValueError("Malformed credential expiry")


class CredentialStore(Protocol):
    def load(self) -> Credentials | None: ...
    def save(self, credentials: Credentials) -> None: ...
    def clear(self) -> None: ...
    def locked(self) -> ContextManager[None]: ...


class FileCredentialStore:
    def __init__(self, path: Path):
        self.path = path

    def _directory(self, *, create: bool = False) -> None:
        if create:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not self.path.parent.exists():
            return
        info = self.path.parent.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o077 or info.st_uid != os.getuid():
            raise PermissionError("Credential directory must be private (0700)")

    @contextmanager
    def locked(self) -> Generator[None]:
        self._directory(create=True)
        descriptor = os.open(
            self.path.with_suffix(".lock"), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
        )
        try:
            info = os.fstat(descriptor)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_mode & 0o077
                or info.st_uid != os.getuid()
                or info.st_nlink != 1
            ):
                raise PermissionError("Credential lock must be private")
            deadline = time.monotonic() + 5
            while True:
                try:
                    _lock(descriptor)
                    break
                except OSError as error:
                    if error.errno not in {errno.EACCES, errno.EAGAIN}:
                        raise
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Credential store is in use") from None
                    time.sleep(0.05)
            try:
                yield
            finally:
                _lock(descriptor, release=True)
        finally:
            os.close(descriptor)

    def load(self) -> Credentials | None:
        self._directory()
        try:
            descriptor = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            return None
        with os.fdopen(descriptor) as stream:
            info = os.fstat(stream.fileno())
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_mode & 0o077
                or info.st_uid != os.getuid()
                or info.st_nlink != 1
            ):
                raise PermissionError("Credentials file must be private (0600)")
            payload = json.load(stream)
        payload["scopes"] = tuple(payload.get("scopes", ()))
        payload["source_scopes"] = tuple(payload.get("source_scopes", ()))
        return Credentials(**payload)

    def save(self, credentials: Credentials) -> None:
        self._directory(create=True)
        if self.path.is_symlink():
            raise PermissionError("Credential file cannot be a symlink")
        descriptor, temporary = tempfile.mkstemp(prefix=".credentials-", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "w") as stream:
                json.dump(asdict(credentials), stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)


class KeyringErrors(Protocol):
    KeyringError: type[Exception]


class KeyringBackend(Protocol):
    errors: KeyringErrors

    def get_password(self, service: str, username: str) -> str | None: ...
    def set_password(self, service: str, username: str, password: str) -> None: ...
    def delete_password(self, service: str, username: str) -> None: ...


class KeyringCredentialStore:
    def __init__(self, keyring: KeyringBackend):
        self.keyring = keyring

    def load(self) -> Credentials | None:
        value = self.keyring.get_password("synth-ai", "oauth")
        if value is None:
            return None
        payload = json.loads(value)
        payload["scopes"] = tuple(payload.get("scopes", ()))
        payload["source_scopes"] = tuple(payload.get("source_scopes", ()))
        return Credentials(**payload)

    def save(self, credentials: Credentials) -> None:
        self.keyring.set_password("synth-ai", "oauth", json.dumps(asdict(credentials)))

    def clear(self) -> None:
        if self.keyring.get_password("synth-ai", "oauth") is not None:
            self.keyring.delete_password("synth-ai", "oauth")


class PreferredCredentialStore:
    """Only credential-backend availability errors select the private fallback."""

    def __init__(self, file: FileCredentialStore, keyring: KeyringBackend | None = None):
        self.file = file
        self.keyring = KeyringCredentialStore(keyring) if keyring is not None else None
        self.errors = (keyring.errors.KeyringError,) if keyring is not None else ()

    def locked(self) -> ContextManager[None]:
        return self.file.locked()

    def load(self) -> Credentials | None:
        # A fallback write must supersede an older keychain credential.
        fallback = self.file.load()
        if fallback is not None:
            return fallback
        if self.keyring is None:
            return None
        try:
            return self.keyring.load()
        except self.errors:
            return None

    def save(self, credentials: Credentials) -> None:
        if self.keyring is not None:
            try:
                self.keyring.save(credentials)
            except self.errors:
                self.file.save(credentials)
                return
            self.file.clear()
            return
        self.file.save(credentials)

    def clear(self) -> None:
        # Do not report a successful logout if a keychain copy cannot be removed.
        if self.keyring is not None:
            self.keyring.clear()
        self.file.clear()


def default_store() -> CredentialStore:
    """Use private files unless the customer explicitly selects keychain storage."""
    root = Path(os.environ.get("SYNTH_CONFIG_DIR", str(Path.home() / ".synth")))
    storage = os.environ.get("SYNTH_AUTH_STORAGE", "file")
    if storage == "file":
        return FileCredentialStore(root / "credentials.json")
    if storage != "keychain":
        raise ValueError("Unsupported credential storage")
    try:
        keyring = cast(KeyringBackend, importlib.import_module("keyring"))
    except ImportError:
        keyring = None
    return PreferredCredentialStore(FileCredentialStore(root / "credentials.json"), keyring)
