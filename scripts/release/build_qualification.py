"""Assemble the synth.sdk.qualified-artifact.v1 ZIP that the synth-ai publisher verifies.

Contract: `.github/verify_qualified_release.py` (same repo, same commit) (`verify_bundle`) and README
"Qualified publication". This tool never runs a check and never builds a wheel: it binds
existing evidence files to the exact wheel bytes, and refuses to emit a bundle unless every
one of the seven required checks has a passed evidence record.

Usage:
  build_qualification.py --checks checks.json --wheel <whl> --source <sha> --version 0.23.0 \
      --channel stable --output qualification.zip

checks.json maps each CheckName value to {"status": "passed"|"pending", "evidence": <path>}
or, for a check the release policy lets this version defer, {"status": "deferred", "reason": ..., "ticket": ...}.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

__all__ = [
    "CheckName",
    "CheckStatus",
    "QualificationCheck",
    "QualificationError",
    "QualificationErrorCode",
    "QualifiedArtifact",
    "build_bundle",
]

SCHEMA = "synth.sdk.qualified-artifact.v1"


class CheckName(StrEnum):
    SDK_CONTRACTS = "sdk-contracts"
    FRESH_INSTALL = "fresh-install"
    NATIVE_INDEX_FAST = "native-index-fast"
    NATIVE_INDEX_DEEP = "native-index-deep"
    NATIVE_MONITOR = "native-monitor"
    NATIVE_SWARMS = "native-swarms"
    STAGING = "staging"


class CheckStatus(StrEnum):
    PASSED = "passed"
    PENDING = "pending"
    DEFERRED = "deferred"


class Channel(StrEnum):
    STABLE = "stable"
    NIGHTLY = "nightly"


class QualificationErrorCode(StrEnum):
    CHECK_NOT_PASSED = "check_not_passed"
    CHECK_MISSING = "check_missing"
    EVIDENCE_MISSING = "evidence_missing"
    WHEEL_VERSION_MISMATCH = "wheel_version_mismatch"
    OUTPUT_EXISTS = "output_exists"
    DEFERRAL_INVALID = "deferral_invalid"


@dataclass(frozen=True, slots=True)
class QualificationError(Exception):
    code: QualificationErrorCode
    stage: str
    cause: str

    def __str__(self) -> str:
        return f"{self.code.value} at {self.stage}: {self.cause}"


@dataclass(frozen=True, slots=True)
class Distribution:
    filename: str
    sha256: str

    @classmethod
    def from_path(cls, path: Path) -> Distribution:
        return cls(filename=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@dataclass(frozen=True, slots=True)
class QualificationCheck:
    name: CheckName
    status: CheckStatus
    source_sha: str
    distributions: tuple[Distribution, ...]
    evidence_path: str
    evidence_sha256: str
    reason: str | None = None
    ticket: str | None = None

    def to_wire(self) -> dict[str, object]:
        wire: dict[str, object] = {
            "name": self.name.value,
            "status": self.status.value,
            "source_sha": self.source_sha,
            "distributions": {item.filename: item.sha256 for item in self.distributions},
            "evidence_path": self.evidence_path,
            "evidence_sha256": self.evidence_sha256,
        }
        if self.status is CheckStatus.DEFERRED:
            wire["reason"] = self.reason
            wire["ticket"] = self.ticket
        return wire


@dataclass(frozen=True, slots=True)
class QualifiedArtifact:
    source_sha: str
    version: str
    channel: Channel
    distributions: tuple[Distribution, ...]
    checks: tuple[QualificationCheck, ...]

    def to_wire(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "status": CheckStatus.PASSED.value,
            "source_sha": self.source_sha,
            "version": self.version,
            "channel": self.channel.value,
            "distributions": [
                {"filename": item.filename, "sha256": item.sha256} for item in self.distributions
            ],
            "checks": [check.to_wire() for check in self.checks],
        }


@dataclass(frozen=True, slots=True)
class CheckInput:
    status: CheckStatus
    evidence: Path | None
    reason: str | None = None
    ticket: str | None = None

    @classmethod
    def from_wire(cls, name: CheckName, raw: Mapping[str, str | None]) -> CheckInput:
        evidence = raw.get("evidence")
        return cls(
            status=CheckStatus(raw["status"]),
            evidence=Path(evidence) if evidence else None,
            reason=raw.get("reason"),
            ticket=raw.get("ticket"),
        )


def build_bundle(
    *,
    checks: Mapping[CheckName, CheckInput],
    wheel: Path,
    source_sha: str,
    version: str,
    channel: Channel,
    output: Path,
    deferrable: frozenset[CheckName] = frozenset(),
) -> QualifiedArtifact:
    if output.exists():
        raise QualificationError(QualificationErrorCode.OUTPUT_EXISTS, "output", str(output))
    if not wheel.name.startswith(f"synth_ai-{version}-"):
        raise QualificationError(
            QualificationErrorCode.WHEEL_VERSION_MISMATCH, "wheel", f"{wheel.name} is not {version}"
        )
    distributions: tuple[Distribution, ...] = (Distribution.from_path(wheel),)
    records: list[QualificationCheck] = []
    evidence_files: dict[str, bytes] = {}
    for name in CheckName:
        if name not in checks:
            raise QualificationError(QualificationErrorCode.CHECK_MISSING, name.value, "no record")
        selected = checks[name]
        if selected.status is CheckStatus.DEFERRED:
            # The verifier enforces the same rules; refuse here so no bad bundle is written.
            if name not in deferrable:
                raise QualificationError(
                    QualificationErrorCode.DEFERRAL_INVALID, name.value, "not deferrable by policy"
                )
            if not (selected.reason or "").strip() or not re.fullmatch(
                r"[A-Z][A-Z0-9]*-[0-9]+", selected.ticket or ""
            ):
                raise QualificationError(
                    QualificationErrorCode.DEFERRAL_INVALID,
                    name.value,
                    "reason and ticket id required",
                )
            raw = json.dumps(
                {"check": name.value, "reason": selected.reason, "ticket": selected.ticket}
            ).encode()
            archived = f"evidence/{name.value}/deferral.json"
        else:
            if selected.status is not CheckStatus.PASSED:
                raise QualificationError(
                    QualificationErrorCode.CHECK_NOT_PASSED, name.value, selected.status.value
                )
            if selected.evidence is None or not selected.evidence.is_file():
                raise QualificationError(
                    QualificationErrorCode.EVIDENCE_MISSING, name.value, str(selected.evidence)
                )
            raw = selected.evidence.read_bytes()
            archived = f"evidence/{name.value}/{selected.evidence.name}"
        evidence_files[archived] = raw
        records.append(
            QualificationCheck(
                name=name,
                status=selected.status,
                reason=selected.reason if selected.status is CheckStatus.DEFERRED else None,
                ticket=selected.ticket if selected.status is CheckStatus.DEFERRED else None,
                source_sha=source_sha,
                distributions=distributions,
                evidence_path=archived,
                evidence_sha256=hashlib.sha256(raw).hexdigest(),
            )
        )
    artifact = QualifiedArtifact(
        source_sha=source_sha,
        version=version,
        channel=channel,
        distributions=distributions,
        checks=tuple(records),
    )
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(
            "qualification.json", json.dumps(artifact.to_wire(), indent=2, sort_keys=True) + "\n"
        )
        bundle.write(wheel, wheel.name)
        for archived, raw in evidence_files.items():
            bundle.writestr(archived, raw)
    return artifact


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checks", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--channel", choices=[item.value for item in Channel], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path(__file__).resolve().parents[2] / ".github" / "release-policy.json",
        help="release policy naming deferrable checks (default: this repo's)",
    )
    args = parser.parse_args(argv)
    policy = json.loads(args.policy.read_text())
    deferrable = frozenset(
        CheckName(item) for item in policy["releases"].get(args.version, {}).get("deferrable", [])
    )
    raw_checks: dict[str, dict[str, str]] = json.loads(args.checks.read_text())
    checks = {
        CheckName(name): CheckInput.from_wire(CheckName(name), value)
        for name, value in raw_checks.items()
    }
    build_bundle(
        checks=checks,
        wheel=args.wheel,
        source_sha=args.source,
        version=args.version,
        channel=Channel(args.channel),
        output=args.output,
        deferrable=deferrable,
    )
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({"archive": str(args.output), "archive_sha256": digest}))


if __name__ == "__main__":
    main()
