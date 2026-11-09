"""Check pinned contract provenance and exact agreement with a committed Core cohort.

Local use: uv run python scripts/check_cohort.py --core-checkout /path/to/core
--core-revision defaults to HEAD; use the frozen cohort's commit for release work.
This command never rewrites the contract or infers compatibility from versions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tomllib
from pathlib import Path

from pydantic import JsonValue

ROOT = Path(__file__).resolve().parents[1]


def check(
    root: Path, *, core_checkout: Path | None = None, core_revision: str = "HEAD"
) -> dict[str, JsonValue]:
    pin = tomllib.loads((root / "contracts/SOURCE.toml").read_text())
    commit, path, digest = pin["commit"], pin["path"], pin["sha256"]
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("contract provenance requires a full Core commit")
    if (
        not isinstance(path, str)
        or not path
        or path.startswith(("/", ":", "-"))
        or ".." in Path(path).parts
    ):
        raise ValueError("invalid Core contract path")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("contract provenance requires a SHA256")
    actual = hashlib.sha256((root / "contracts/synth-v1.openapi.yaml").read_bytes()).hexdigest()
    if actual != digest:
        raise ValueError("SDK contract does not match SOURCE.toml SHA256")
    report: dict[str, JsonValue] = {"sdk_core_commit": commit, "sdk_sha256": actual}
    if core_checkout is not None:
        revision = subprocess.check_output(
            ["git", "-C", str(core_checkout), "rev-parse", "--verify", core_revision + "^{commit}"],
            text=True,
        ).strip()
        source = subprocess.check_output(
            ["git", "-C", str(core_checkout), "show", revision + ":" + path]
        )
        candidate_digest = hashlib.sha256(source).hexdigest()
        report.update(
            core_commit=revision, core_sha256=candidate_digest, aligned=actual == candidate_digest
        )
        # Also verify the source claim; a matching candidate alone cannot prove provenance.
        pinned = subprocess.check_output(
            ["git", "-C", str(core_checkout), "show", commit + ":" + path]
        )
        report["provenance_verified"] = hashlib.sha256(pinned).hexdigest() == actual
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-checkout", type=Path)
    parser.add_argument("--core-revision", default="HEAD")
    args = parser.parse_args()
    report = check(ROOT, core_checkout=args.core_checkout, core_revision=args.core_revision)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report.get("aligned") is False or report.get("provenance_verified") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
