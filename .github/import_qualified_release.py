"""Materialize validated qualification bytes for an immutable Actions artifact.

See README.md, Qualified publication. This producer never executes distributions
or treats its successful upload as evidence that a missing check passed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import tomllib
import zipfile
from pathlib import Path


def materialize(
    archive: Path,
    output: Path,
    *,
    source: str,
    version: str,
    channel: str,
    archive_sha: str,
    pyproject: Path,
) -> dict:
    """Validate with the publisher's verifier before copying any archive entry."""
    spec = importlib.util.spec_from_file_location(
        "qualified_release", Path(__file__).with_name("verify_qualified_release.py")
    )
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    verifier.source_gate(source, version, channel, "main" if channel == "stable" else "staging")
    project = tomllib.loads(pyproject.read_text())["project"]
    verifier.require(
        project.get("name") == "synth-ai" and project.get("version") == version,
        "source package/version mismatch",
    )
    verifier.require(not output.exists(), "artifact output already exists")
    prepared = output.with_name(output.name + "-verified")
    receipt = verifier.verify_bundle(
        archive, prepared, source=source, version=version, channel=channel, archive_sha=archive_sha
    )
    # verify_bundle has already refused traversal, symlinks, duplicates,
    # undeclared files, invalid evidence and expansion beyond its size bound.
    with zipfile.ZipFile(archive) as zipped:
        for entry in zipped.infolist():
            if entry.is_dir():
                continue
            destination = output / entry.filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(zipped.read(entry))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive-sha", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--channel", choices=("nightly", "stable"), required=True)
    args = parser.parse_args()
    if os.environ.get("GITHUB_SHA") != args.source:
        raise ValueError("producer must run at the exact qualified source")
    receipt = materialize(
        args.archive,
        args.output,
        source=args.source,
        version=args.version,
        channel=args.channel,
        archive_sha=args.archive_sha,
        pyproject=Path(__file__).parents[1] / "pyproject.toml",
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
