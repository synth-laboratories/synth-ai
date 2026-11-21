"""Clerk session credentials that an external refresher keeps alive in 0600 files.

A Clerk session token lives about 60 s, shorter than one hosted driver's run.
`<NAME>_FILE` (for SYNTH_JOURNEY_SESSION_TOKEN and
SYNTH_JOURNEY_FOREIGN_SESSION_TOKEN) names a file that is re-read at the moment
a request needs the token, so a refresher that atomically rewrites it keeps the
credential valid. The fixed-string variables keep working; setting both forms
of one name is a configuration error. A token is never logged or retained.
"""

from __future__ import annotations

# ruff: noqa: UP045 -- Explicit Optional at acceptance contract boundaries.
import os
import stat
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, ExitStack
from pathlib import Path
from typing import Optional

from .surfaces import Call, Outcome, Surface

__all__ = [
    "FILE_SUFFIX",
    "JourneyConfigError",
    "LiveSecrets",
    "RefreshingSurface",
    "SessionRefused",
    "load_secrets",
    "read_token_file",
]

FILE_SUFFIX = "_FILE"


class JourneyConfigError(ValueError):
    """Typed refusal of the driver's credential configuration (names only, never values)."""

    def __init__(self, code: str, name: str) -> None:
        self.code = code
        self.name = name
        super().__init__(f"{code}: {name}")


def read_token_file(name: str, path: Path) -> str:
    try:
        info = os.stat(path)
    except OSError as error:
        raise JourneyConfigError("session_token_file_unreadable", name) from error
    if not stat.S_ISREG(info.st_mode):
        raise JourneyConfigError("session_token_file_not_regular", name)
    if stat.S_IMODE(info.st_mode) != 0o600:
        raise JourneyConfigError("session_token_file_not_0600", name)
    token = path.read_text().strip()
    if not token or any(character.isspace() for character in token):
        raise JourneyConfigError("session_token_file_malformed", name)
    return token


class LiveSecrets(Mapping[str, str]):
    """Fixed values plus file-backed names whose value is read at each lookup."""

    def __init__(self, values: Mapping[str, str], files: Mapping[str, Path]) -> None:
        overlap = set(values) & set(files)
        if overlap:
            raise JourneyConfigError("session_token_file_and_value_both_set", min(overlap))
        self._values = dict(values)
        self._files = dict(files)

    def __getitem__(self, name: str) -> str:
        if name in self._files:
            return read_token_file(name, self._files[name])
        return self._values[name]

    def __iter__(self) -> Iterator[str]:
        yield from self._values
        yield from self._files

    def __len__(self) -> int:
        return len(self._values) + len(self._files)

    def file_backed(self, name: str) -> bool:
        return name in self._files

    def __repr__(self) -> str:  # Never the values.
        return f"LiveSecrets(names={sorted(self)})"


def load_secrets(
    environ: Mapping[str, str], names: tuple[str, ...], file_names: tuple[str, ...]
) -> LiveSecrets:
    """`names` are read as fixed strings; each of `file_names` may instead come from a file."""
    values = {name: environ[name] for name in names if environ.get(name, "") != ""}
    files: dict[str, Path] = {}
    for name in file_names:
        location = environ.get(name + FILE_SUFFIX, "")
        if location == "":
            continue
        if name in values:
            raise JourneyConfigError("session_token_file_and_value_both_set", name)
        files[name] = Path(location)
        read_token_file(name, files[name])  # Refuse a bad file before any request.
    return LiveSecrets(values, files)


class SessionRefused(RuntimeError):
    """A freshly read session failed its principal check; carries the typed cause."""

    def __init__(self, cause: str, account: Outcome) -> None:
        self.cause = cause
        self.account = account
        super().__init__(cause)


class RefreshingSurface:
    """A session-credentialed surface reopened whenever the token source changes.

    Every newly read token is opened as a fresh surface and must pass `verify`
    (on its getAccount outcome) before any other request uses it, so the
    principal checks hold for each token, not only the first.
    """

    def __init__(
        self,
        make: Callable[[str], AbstractContextManager[Surface]],
        token: Callable[[], str],
        account: Callable[[Surface], Outcome],
        verify: Callable[[Outcome], Optional[str]],
    ) -> None:
        self._make, self._token, self._account, self._verify = make, token, account, verify
        self._stack: Optional[ExitStack] = None
        self._current: Optional[Surface] = None
        self._opened_with: Optional[str] = None
        self.opened = 0

    def surface(self) -> Surface:
        token = self._token()
        if self._current is None or token != self._opened_with:
            self.close()
            stack = ExitStack()
            opened = stack.enter_context(self._make(token))
            self._stack, self._current, self._opened_with = stack, opened, token
            self.opened += 1
            account = self._account(opened)
            cause = self._verify(account)
            if cause is not None:
                self.close()
                raise SessionRefused(cause, account)
        return self._current

    def call(self, call: Call) -> Outcome:
        return self.surface().call(call)

    def close(self) -> None:
        stack, self._stack, self._current, self._opened_with = self._stack, None, None, None
        if stack is not None:
            stack.close()

    def __enter__(self) -> RefreshingSurface:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
