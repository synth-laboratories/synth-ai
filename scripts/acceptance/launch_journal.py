"""Write-ahead identity for one acceptance launch; uncertain outcomes never retry.

See SYN-5692. A new attempt needs a new owned directory. Existing intent may be
read/reconciled; its presence never authorizes another POST.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from synth_ai.models import RunSnapshot

__all__ = ["InputIdentity", "LaunchIntent", "LaunchJournal", "LaunchJournalError"]
_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class LaunchJournalError(Exception):
    code: str
    stage: str
    cause: str
    directory: Path


@dataclass(frozen=True, slots=True)
class InputIdentity:
    name: str
    sha256: str


@dataclass(frozen=True, slots=True)
class LaunchIntent:
    driver: str
    idempotency_key: str
    request_sha256: str
    inputs: tuple[InputIdentity, ...]


@dataclass(frozen=True, slots=True)
class LaunchJournal:
    directory: Path

    def begin(self, intent: LaunchIntent) -> None:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._write("launch-intent.json", json.dumps(asdict(intent), sort_keys=True))

    def accepted(self, snapshot: RunSnapshot) -> None:
        self._write("launch-accepted.json", snapshot.model_dump_json())

    def read_accepted(self) -> RunSnapshot:
        return RunSnapshot.model_validate_json(
            (self.directory / "launch-accepted.json").read_text()
        )

    def _write(self, filename: str, payload: str) -> None:
        try:
            descriptor = os.open(
                self.directory / filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
        except FileExistsError as error:
            code = "launch_journal_record_exists"
            cause = "reconcile the retained attempt; another launch is refused"
            _LOGGER.error(
                "code=%s stage=%s directory=%s cause=%s", code, filename, self.directory, cause
            )
            raise LaunchJournalError(code, filename, cause, self.directory) from error
        with os.fdopen(descriptor, "w") as output:
            output.write(payload + "\n")
            output.flush()
            os.fsync(output.fileno())
        descriptor = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def request_digest(payload: str) -> str:
        return hashlib.sha256(payload.encode()).hexdigest()
