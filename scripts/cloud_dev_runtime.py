#!/usr/bin/env python3
"""Finite native Python profile controller; see Backend unity-work-admission.md.

The native owner supplies only its retained source and lease IDs. No network,
provider API, project installation, tests, services or additional allocator.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "synth.unity.environment-bootstrap-receipt.v1"
UV_VERSION = "uv 0.11.33"
SEED = Path("/opt/repository-uv-cache")


class RuntimeRefusedError(Exception):
    def __init__(self, code: str):
        self.code = code


def command(argv: list[str], *, environment: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        argv,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=540,
    )
    if result.returncode:
        # Dependency/auth details and command output are not receipt content.
        raise RuntimeRefusedError("runtime_command_failed")
    return result.stdout.strip()


def private_directory(path: Path) -> Path:
    with contextlib.suppress(FileExistsError):
        path.mkdir(mode=0o700)
    metadata = path.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or metadata.st_mode & 0o077
    ):
        raise RuntimeRefusedError("runtime_private_directory_refused")
    return path


def private_tree(path: Path) -> list[Path]:
    paths = [path, *path.rglob("*")]
    for entry in paths:
        metadata = entry.lstat()
        if (
            metadata.st_uid != os.getuid()
            or not (stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode))
            or (stat.S_ISREG(metadata.st_mode) and metadata.st_nlink != 1)
        ):
            raise RuntimeRefusedError("runtime_private_cache_refused")
    return paths


def seed_cache(destination: Path) -> None:
    # The immutable image seed may be shared for reads; the lease never writes it.
    for path in [SEED, *SEED.rglob("*")]:
        metadata = path.lstat()
        if metadata.st_uid != 0 or (
            not stat.S_ISLNK(metadata.st_mode) and metadata.st_mode & 0o022
        ):
            raise RuntimeRefusedError("runtime_cache_seed_refused")
        if stat.S_ISLNK(metadata.st_mode):
            try:
                path.resolve(strict=True).relative_to(SEED)
            except (OSError, ValueError) as error:
                raise RuntimeRefusedError("runtime_cache_seed_refused") from error
        elif not (stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)):
            raise RuntimeRefusedError("runtime_cache_seed_refused")
    if not SEED.is_dir():
        raise RuntimeRefusedError("runtime_cache_seed_unavailable")
    cache = private_directory(destination)
    private_tree(cache)
    if not (cache / ".seeded").exists():
        shutil.copytree(SEED, cache, dirs_exist_ok=True)
        # Copying preserves read-only image modes. Restore writes in this private copy.
        for path in private_tree(cache):
            path.chmod(0o700 if path.is_dir() else 0o600, follow_symlinks=False)
        (cache / ".seeded").write_text("dependency seed copied\n")


def prepare(operation: str) -> dict:
    source = os.environ.get("SYNTH_ENVIRONMENT_SOURCE_COMMIT", "")
    lease = os.environ.get("SYNTH_ENVIRONMENT_LEASE_ID", "")
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", source) or not re.fullmatch(
        r"[0-9a-f-]{36}", lease
    ):
        raise RuntimeRefusedError("runtime_owner_binding_missing")
    if sys.version_info < (3, 11):  # noqa: UP036 - invoked by the guest image system Python
        raise RuntimeRefusedError("runtime_python_version_refused")
    head = command(["git", "rev-parse", "HEAD"])
    protected = ["pyproject.toml", "uv.lock", "scripts/cloud_dev_runtime.py"]
    command(
        ["git", "diff", "--no-ext-diff", "--no-textconv", "--exit-code", source, "--", *protected]
    )
    if operation == "prepare" and (
        head != source or command(["git", "status", "--porcelain", "--untracked-files=normal"])
    ):
        raise RuntimeRefusedError("runtime_source_mismatch")
    if (ROOT / ".venv").is_symlink():
        raise RuntimeRefusedError("runtime_venv_path_refused")
    uv = shutil.which("uv")
    if uv is None or command([uv, "--version"]).split(" (", 1)[0] != UV_VERSION:
        raise RuntimeRefusedError("runtime_uv_version_refused")
    digest = hashlib.sha256()
    for name in ("pyproject.toml", "uv.lock", "scripts/cloud_dev_runtime.py"):
        path = ROOT / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeRefusedError("runtime_source_file_refused")
        digest.update(name.encode() + b"\0" + path.read_bytes() + b"\0")
    source_digest = digest.hexdigest()
    private_root = private_directory(Path(f"/tmp/synth-cloud-dev-{os.getuid()}"))
    identity = hashlib.sha256(f"{ROOT}\0{lease}\0{source}".encode()).hexdigest()
    cache = private_root / identity
    seed_cache(cache)
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR")
    }
    environment.update(
        {
            "UV_CACHE_DIR": str(cache),
            "UV_PYTHON_DOWNLOADS": "never",
            "UV_OFFLINE": "1",
            "UV_NO_PROGRESS": "1",
        }
    )
    command(
        [
            uv,
            "--no-config",
            "sync",
            "--offline",
            "--frozen",
            "--no-dev",
            "--no-default-groups",
            "--no-install-project",
        ],
        environment=environment,
    )
    python = ROOT / ".venv/bin/python"
    # These are production readiness observations, not a test suite or API call.
    command(
        [
            str(python),
            "-c",
            "from synth_ai import SynthClient; import synth_ai; assert SynthClient is not None",
        ],
        environment=environment,
    )
    command([str(python), "-m", "synth_ai.cli_auth", "--help"], environment=environment)
    return {
        "schema": SCHEMA,
        "profile": "python-uv-v1",
        "environment": "synth-ai-exe-dev",
        "environment_revision": "v1",
        "catalog_revision": "v5",
        "source_commit": source,
        "phase": "ready",
        "build": {"revision": source, "source_sha256": f"sha256:{source_digest}", "dirty": False},
        "runtime": {
            "python_version": sys.version.split()[0],
            "uv_version": UV_VERSION,
            "dependencies_frozen": True,
            "offline": True,
            "project_installed": False,
        },
        "observation": {"sdk_imported": True, "cli_help_exit_code": 0},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("prepare", "resume"))
    parser.add_argument("--json", action="store_true", required=True)
    args = parser.parse_args()
    try:
        receipt = prepare(args.operation)
    except (RuntimeRefusedError, OSError, subprocess.SubprocessError) as error:
        code = (
            error.code
            if isinstance(error, RuntimeRefusedError)
            else "runtime_execution_unavailable"
        )
        receipt = {
            "schema": SCHEMA,
            "profile": "python-uv-v1",
            "environment": "synth-ai-exe-dev",
            "environment_revision": "v1",
            "catalog_revision": "v5",
            "source_commit": os.environ.get("SYNTH_ENVIRONMENT_SOURCE_COMMIT", ""),
            "phase": "refused",
            "code": code,
            "stage": "python_prepare",
            "cause": code,
        }
        print(json.dumps(receipt, sort_keys=True))
        return 1
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
