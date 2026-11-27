"""Map Git release tags to canonical PEP 440 PyPI versions and verify/stamp package alignment.

Supported release tags:
- Stable:  `vX.Y.Z`                    -> PEP 440 `X.Y.Z`            (channel: stable)
- RC:      `vX.Y.Z-rc.N` / `vX.Y.ZrcN` -> PEP 440 `X.Y.ZrcN`         (channel: rc)
- Nightly: `vX.Y.Z-nightly.YYYYMMDD` or
           `vX.Y.Z.devYYYYMMDD`        -> PEP 440 `X.Y.Z.devYYYYMMDD` (channel: nightly)

PEP 440 `.devYYYYMMDD` pre-releases are ignored by default `pip install synth-ai`
and require `--pre` or an explicit `==` pin.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "ParsedReleaseTag",
    "ReleaseChannel",
    "read_declared_versions",
    "stamp_version",
    "tag_to_pep440",
    "verify_tag_version_alignment",
]

ROOT = Path(__file__).resolve().parents[1]
_TAG_RE = re.compile(
    r"^v(?P<base>\d+\.\d+\.\d+)"
    r"(?:"
    r"-nightly\.(?P<nightly_date>\d{8})"
    r"|\.dev(?P<dev_date>\d{8})"
    r"|-rc\.(?P<semver_rc>[1-9]\d*)"
    r"|rc(?P<pep440_rc>[1-9]\d*)"
    r")?$"
)
_PYPROJECT_VERSION_LINE_RE = re.compile(r'^(version\s*=\s*")([^"]+)(")$', re.MULTILINE)
_VERSION_MODULE_RE = re.compile(r'^__version__\s*=\s*["\']([^"\']+)["\']$', re.MULTILINE)


class ReleaseChannel(StrEnum):
    STABLE = "stable"
    RC = "rc"
    NIGHTLY = "nightly"


@dataclass(frozen=True, slots=True)
class ParsedReleaseTag:
    raw_tag: str
    base_version: str
    pep440_version: str
    semver_version: str
    channel: ReleaseChannel
    date_stamp: str | None = None
    rc_number: int | None = None

    def as_dict(self) -> dict[str, str | int | None]:
        return {
            "raw_tag": self.raw_tag,
            "base_version": self.base_version,
            "pep440_version": self.pep440_version,
            "semver_version": self.semver_version,
            "channel": self.channel.value,
            "date_stamp": self.date_stamp,
            "rc_number": self.rc_number,
        }


def _validate_calendar_stamp(stamp: str, raw_tag: str) -> None:
    try:
        date(int(stamp[0:4]), int(stamp[4:6]), int(stamp[6:8]))
    except ValueError as exc:
        raise ValueError(
            f"release tag {raw_tag!r} contains invalid YYYYMMDD calendar date {stamp!r}"
        ) from exc


def tag_to_pep440(tag: str) -> ParsedReleaseTag:
    """Parse a release Git tag and convert it into its canonical PEP 440 version."""
    cleaned = tag.strip()
    match = _TAG_RE.match(cleaned)
    if match is None:
        raise ValueError(
            f"unsupported release tag {tag!r}; expected vX.Y.Z, vX.Y.Z-rc.N, "
            "vX.Y.ZrcN, vX.Y.Z-nightly.YYYYMMDD, or vX.Y.Z.devYYYYMMDD"
        )
    base = match.group("base")
    nightly_date = match.group("nightly_date") or match.group("dev_date")
    if nightly_date is not None:
        _validate_calendar_stamp(nightly_date, cleaned)
        return ParsedReleaseTag(
            raw_tag=cleaned,
            base_version=base,
            pep440_version=f"{base}.dev{nightly_date}",
            semver_version=f"{base}-nightly.{nightly_date}",
            channel=ReleaseChannel.NIGHTLY,
            date_stamp=nightly_date,
        )
    rc_raw = match.group("semver_rc") or match.group("pep440_rc")
    if rc_raw is not None:
        rc_number = int(rc_raw)
        return ParsedReleaseTag(
            raw_tag=cleaned,
            base_version=base,
            pep440_version=f"{base}rc{rc_number}",
            semver_version=f"{base}-rc.{rc_number}",
            channel=ReleaseChannel.RC,
            rc_number=rc_number,
        )
    return ParsedReleaseTag(
        raw_tag=cleaned,
        base_version=base,
        pep440_version=base,
        semver_version=base,
        channel=ReleaseChannel.STABLE,
    )


def verify_tag_version_alignment(
    tag: str,
    pyproject_version: str,
    module_version: str | None = None,
) -> ParsedReleaseTag:
    """Verify that the declared package version(s) match the tag's canonical PEP 440 version."""
    parsed = tag_to_pep440(tag)
    if pyproject_version != parsed.pep440_version:
        raise ValueError(
            f"tag {tag!r} maps to PEP 440 version {parsed.pep440_version!r}, "
            f"but pyproject.toml declares {pyproject_version!r}"
        )
    if module_version is not None and module_version != parsed.pep440_version:
        raise ValueError(
            f"tag {tag!r} maps to PEP 440 version {parsed.pep440_version!r}, "
            f"but synth_ai/_version.py declares {module_version!r}"
        )
    return parsed


def read_declared_versions(repo_root: Path = ROOT) -> tuple[str, str]:
    """Read `(pyproject_version, module_version)` from the checkout."""
    pyproject_path = repo_root / "pyproject.toml"
    pyproject_data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = pyproject_data.get("project")
    if not isinstance(project, dict) or not isinstance(project.get("version"), str):
        raise ValueError(f"{pyproject_path}: missing [project].version string")
    pyproject_version = str(project["version"])

    version_module_path = repo_root / "synth_ai" / "_version.py"
    module_text = version_module_path.read_text(encoding="utf-8")
    match = _VERSION_MODULE_RE.search(module_text)
    if match is None:
        raise ValueError(f"{version_module_path}: missing __version__ assignment")
    module_version = match.group(1)
    assert isinstance(module_version, str)
    return pyproject_version, module_version


def stamp_version(repo_root: Path, tag: str, *, check_only: bool = False) -> ParsedReleaseTag:
    """Stamp `pyproject.toml` and `synth_ai/_version.py` with the tag's PEP 440 version."""
    parsed = tag_to_pep440(tag)
    if check_only:
        pyproject_version, module_version = read_declared_versions(repo_root)
        return verify_tag_version_alignment(tag, pyproject_version, module_version)

    pyproject_path = repo_root / "pyproject.toml"
    pyproject_text = pyproject_path.read_text(encoding="utf-8")
    updated_pyproject, count = _PYPROJECT_VERSION_LINE_RE.subn(
        rf"\g<1>{parsed.pep440_version}\g<3>",
        pyproject_text,
        count=1,
    )
    if count != 1:
        raise ValueError(f"{pyproject_path}: could not locate [project].version line to stamp")
    pyproject_path.write_text(updated_pyproject, encoding="utf-8")

    version_module_path = repo_root / "synth_ai" / "_version.py"
    version_module_path.write_text(
        f"# Generated from pyproject.toml; do not edit.\n__version__ = {parsed.pep440_version!r}\n",
        encoding="utf-8",
    )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    sub = parser.add_subparsers(dest="command", required=True)

    parse_cmd = sub.add_parser("parse", help="Convert a release tag to PEP 440 JSON.")
    parse_cmd.add_argument("--tag", required=True)

    verify_cmd = sub.add_parser(
        "verify", help="Verify pyproject.toml and synth_ai/_version.py match the tag."
    )
    verify_cmd.add_argument("--tag", required=True)

    stamp_cmd = sub.add_parser(
        "stamp", help="Stamp pyproject.toml and synth_ai/_version.py for a release tag."
    )
    stamp_cmd.add_argument("--tag", required=True)
    stamp_cmd.add_argument("--check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw_root = args.repo_root
    assert isinstance(raw_root, Path)
    repo_root = raw_root.resolve()
    try:
        if args.command == "parse":
            parsed = tag_to_pep440(str(args.tag))
        elif args.command == "verify":
            pyproject_version, module_version = read_declared_versions(repo_root)
            parsed = verify_tag_version_alignment(str(args.tag), pyproject_version, module_version)
        elif args.command == "stamp":
            parsed = stamp_version(repo_root, str(args.tag), check_only=bool(args.check))
        else:
            raise ValueError(f"unknown command: {args.command}")
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(parsed.as_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
