"""Verify retained release bytes; never build, stamp, install, or execute them.

See README.md, Qualified publication. Qualification receipts are reviewed
release evidence; this verifier binds them to source and distribution bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import stat
import tarfile
import time
import tomllib
import urllib.error
import urllib.request
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

SCHEMA = "synth.sdk.qualified-artifact.v1"
CHECKS = {
    "sdk-contracts",
    "fresh-install",
    "native-index-fast",
    "native-index-deep",
    "native-monitor",
    "native-swarms",
    "staging",
}
SHA = re.compile(r"[0-9a-f]{64}")
COMMIT = re.compile(r"[0-9a-f]{40}")
VERSION = re.compile(r"\d+\.\d+\.\d+(?:\.dev\d+)?")
MAX_BYTES = 256 * 1024 * 1024


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def document(raw: bytes) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    value = json.loads(raw, object_pairs_hook=unique)
    require(isinstance(value, dict), "expected object")
    return value


def source_gate(source: str, version: str, channel: str, branch: str) -> None:
    require(COMMIT.fullmatch(source) is not None, "invalid source SHA")
    require(VERSION.fullmatch(version) is not None, "invalid release version")
    require(
        (channel, branch) in {("nightly", "staging"), ("stable", "main")},
        "release branch/channel mismatch",
    )
    require((".dev" in version) == (channel == "nightly"), "release version/channel mismatch")


def artifact_gate(
    artifact: dict, run: dict, *, artifact_id: int, source: str, repository: str, archive_sha: str
) -> None:
    require(SHA.fullmatch(archive_sha) is not None, "invalid artifact digest")
    require(
        artifact.get("id") == artifact_id and artifact.get("expired") is False,
        "artifact unavailable",
    )
    require(artifact.get("digest") == "sha256:" + archive_sha, "GitHub artifact digest mismatch")
    producer = artifact.get("workflow_run", {})
    require(
        producer.get("head_sha") == source and producer.get("id") == run.get("id"),
        "artifact source/run mismatch",
    )
    require(
        producer.get("repository_id") == producer.get("head_repository_id")
        and producer.get("repository_id") is not None,
        "fork artifact is not a release candidate",
    )
    require(run.get("repository", {}).get("full_name") == repository, "foreign qualification run")
    require(
        run.get("head_sha") == source
        and run.get("status") == "completed"
        and run.get("conclusion") == "success",
        "qualification run did not succeed at requested source",
    )
    require(
        run.get("event") not in {"pull_request", "pull_request_target"},
        "PR artifacts cannot be published",
    )


def version_of_distribution(name: str, raw: bytes) -> str:
    if name.endswith(".whl"):
        with zipfile.ZipFile(io.BytesIO(raw)) as wheel:
            names = [n for n in wheel.namelist() if n.endswith(".dist-info/METADATA")]
            require(len(names) == 1, "wheel must contain one METADATA")
            require(wheel.getinfo(names[0]).file_size <= 1024 * 1024, "wheel metadata too large")
            metadata = BytesParser().parsebytes(wheel.read(names[0]))
    else:
        metadata = None
        total = 0
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r|gz") as source:
            for count, member in enumerate(source):
                total += member.size
                require(count < 10000 and total <= MAX_BYTES * 2, "sdist expands beyond bounds")
                if (
                    len(PurePosixPath(member.name).parts) == 2
                    and member.name.endswith("/PKG-INFO")
                    and member.isfile()
                ):
                    require(
                        metadata is None and member.size <= 1024 * 1024, "invalid sdist metadata"
                    )
                    stream = source.extractfile(member)
                    require(stream is not None, "sdist metadata missing")
                    metadata = BytesParser().parsebytes(stream.read(1024 * 1024))
        require(metadata is not None, "sdist metadata missing")
    require(metadata["Name"] == "synth-ai", "wrong package name")
    return str(metadata["Version"])


def verify_bundle(
    archive: Path, output: Path, *, source: str, version: str, channel: str, archive_sha: str
) -> dict:
    require(archive.stat().st_size <= MAX_BYTES, "artifact too large")
    require(digest(archive.read_bytes()) == archive_sha, "downloaded artifact digest mismatch")
    with zipfile.ZipFile(archive) as zipped:
        members = zipped.infolist()
        names = [entry.filename for entry in members if not entry.is_dir()]
        require(len(names) == len(set(names)), "duplicate archive paths")
        require(
            sum(entry.file_size for entry in members) <= MAX_BYTES * 2,
            "expanded artifact too large",
        )
        for entry in members:
            path = PurePosixPath(entry.filename)
            require(
                not path.is_absolute() and ".." not in path.parts and "\\" not in entry.filename,
                "unsafe archive path",
            )
            require(not stat.S_ISLNK(entry.external_attr >> 16), "archive symlink refused")
        require("qualification.json" in names, "qualification manifest missing")
        raw_manifest = zipped.read("qualification.json")
        require(len(raw_manifest) <= 1024 * 1024, "qualification manifest too large")
        manifest = document(raw_manifest)
        require(
            manifest.get("schema") == SCHEMA and manifest.get("status") == "passed",
            "artifact is not qualified",
        )
        require(
            (manifest.get("source_sha"), manifest.get("version"), manifest.get("channel"))
            == (source, version, channel),
            "qualification identity mismatch",
        )
        distributions = manifest.get("distributions")
        require(
            isinstance(distributions, list) and 1 <= len(distributions) <= 2,
            "invalid distributions",
        )
        files = {}
        for item in distributions:
            require(isinstance(item, dict), "invalid distribution record")
            name, sha = item.get("filename"), item.get("sha256")
            require(
                isinstance(name, str) and PurePosixPath(name).name == name and name not in files,
                "invalid distribution filename",
            )
            require(
                isinstance(sha, str) and SHA.fullmatch(sha) is not None, "invalid distribution hash"
            )
            require(name.endswith((".whl", ".tar.gz")), "unexpected distribution")
            require(
                name in names and digest(zipped.read(name)) == sha, "distribution digest mismatch"
            )
            require(
                version_of_distribution(name, zipped.read(name)) == version,
                "distribution version mismatch",
            )
            files[name] = sha
        require(
            sum(name.endswith(".whl") for name in files) == 1,
            "exactly one qualified wheel required",
        )
        require(sum(name.endswith(".tar.gz") for name in files) <= 1, "at most one qualified sdist")
        checks = manifest.get("checks")
        require(
            isinstance(checks, list) and len(checks) == len(CHECKS),
            "required qualification checks missing",
        )
        require(
            {check.get("name") for check in checks} == CHECKS, "qualification coverage mismatch"
        )
        allowed = {"qualification.json", *files}
        for check in checks:
            require(
                check.get("status") == "passed" and check.get("source_sha") == source,
                "qualification check incomplete",
            )
            require(
                check.get("distributions") == files, "check qualified different distribution bytes"
            )
            evidence = check.get("evidence_path")
            require(
                isinstance(evidence, str)
                and evidence.startswith("evidence/")
                and evidence in names,
                "qualification evidence missing",
            )
            require(
                digest(zipped.read(evidence)) == check.get("evidence_sha256"),
                "qualification evidence changed",
            )
            allowed.add(evidence)
        require(set(names) == allowed, "undeclared artifact file")
        require(not output.exists(), "prepared output already exists")
        (output / "dist").mkdir(parents=True)
        for name in files:
            (output / "dist" / name).write_bytes(zipped.read(name))
        (output / "qualification.json").write_bytes(raw_manifest)
    receipt = {
        "schema": "synth.sdk.publish-receipt.v1",
        "source_sha": source,
        "version": version,
        "channel": channel,
        "artifact_sha256": archive_sha,
        "qualification_sha256": digest(raw_manifest),
        "distributions": files,
        "checks": sorted(CHECKS),
    }
    (output / "publish-receipt.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n"
    )
    return receipt


def verify_prepared(output: Path, receipt_sha: str) -> dict:
    raw = (output / "publish-receipt.json").read_bytes()
    require(digest(raw) == receipt_sha, "prepared receipt changed")
    receipt = document(raw)
    require(
        digest((output / "qualification.json").read_bytes()) == receipt["qualification_sha256"],
        "prepared qualification changed",
    )
    require(
        {p.name for p in (output / "dist").iterdir()} == set(receipt["distributions"]),
        "prepared distribution set changed",
    )
    for name, sha in receipt["distributions"].items():
        path = output / "dist" / name
        require(
            path.is_file() and not path.is_symlink() and digest(path.read_bytes()) == sha,
            "prepared distribution changed",
        )
    return receipt


def registry_state(receipt: dict, *, require_published: bool = False) -> bool:
    version = receipt["version"]
    require(VERSION.fullmatch(version) is not None, "invalid registry version")
    try:
        with urllib.request.urlopen(
            f"https://pypi.org/pypi/synth-ai/{version}/json", timeout=30
        ) as response:
            value = document(response.read(8 * 1024 * 1024))
    except urllib.error.HTTPError as error:
        if error.code != 404 or require_published:
            raise
        return False
    require(value.get("info", {}).get("version") == version, "registry version mismatch")
    files = {item["filename"]: item["digests"]["sha256"] for item in value["urls"]}
    require(files == receipt["distributions"], "registry has different bytes for this version")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    for key in ("archive", "artifact", "run", "output", "pyproject"):
        prepare.add_argument("--" + key, type=Path, required=True)
    for key in ("source", "version", "channel", "branch", "repository", "archive-sha"):
        prepare.add_argument("--" + key, required=True)
    prepare.add_argument("--artifact-id", type=int, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--receipt-sha", required=True)
    verify.add_argument("--require-published", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        source_gate(args.source, args.version, args.channel, args.branch)
        project = tomllib.loads(args.pyproject.read_text())["project"]
        require(
            project.get("name") == "synth-ai" and project.get("version") == args.version,
            "source package/version mismatch",
        )
        artifact_gate(
            document(args.artifact.read_bytes()),
            document(args.run.read_bytes()),
            artifact_id=args.artifact_id,
            source=args.source,
            repository=args.repository,
            archive_sha=args.archive_sha,
        )
        receipt = verify_bundle(
            args.archive,
            args.output,
            source=args.source,
            version=args.version,
            channel=args.channel,
            archive_sha=args.archive_sha,
        )
    else:
        receipt = verify_prepared(args.output, args.receipt_sha)
    require_published = getattr(args, "require_published", False)
    for attempt in range(6 if require_published else 1):
        try:
            published = registry_state(receipt, require_published=require_published)
            break
        except (urllib.error.HTTPError, urllib.error.URLError):
            if not require_published or attempt == 5:
                raise
            time.sleep(2)
    print(
        json.dumps(
            {
                "already_published": published,
                "receipt_sha256": digest((args.output / "publish-receipt.json").read_bytes()),
                **receipt,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
