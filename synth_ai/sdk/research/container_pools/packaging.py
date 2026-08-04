"""Build a Harbor task bundle archive for a container-pool runtime image release.

The backend accepts a Harbor bundle as one base64 tar.gz on
``POST /v1/pools/{pool_id}/runtime_image_releases`` with
``metadata.container_subtype = "harbor"``. Three properties of that path are not
obvious from the request shape and are enforced here, because every one of them
fails late and confusingly otherwise:

1. **The Docker build context is the Dockerfile's own directory, not the bundle
   root.** ``materialize_harbor_runtime_release`` archives ``dockerfile.parent``
   and nothing else, so a Dockerfile that reaches above ``environment/`` builds
   fine on a laptop and fails inside the pool. :func:`build_harbor_bundle_archive`
   rejects those ``COPY``/``ADD`` instructions up front.

2. **``instruction.md`` and ``tests/`` are injected at run time**, not baked by
   the Dockerfile. Copying them in the image is harmless but drifts from what
   actually executes; the verifier that runs is the one in the archive.

3. **Whatever sits in the bundle directory ships.** The archive goes to a
   container image and to S3. A ``codex exec`` run with ``CODEX_HOME`` pointed at
   a bundle leaves ``.codex/auth.json`` — live OAuth access and refresh tokens —
   next to the task source. :data:`EXCLUDED_DIRECTORY_NAMES` drops run residue,
   and :func:`build_harbor_bundle_archive` additionally fails closed on anything
   credential-shaped rather than trusting the exclusion list to be complete.
"""

from __future__ import annotations

import base64
import io
import re
import tarfile
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

# Mirrors backend `services/container_pools/rhodes/harbor_runtime.py`.
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_BASE64_CHARS = 180 * 1024 * 1024
# Mirrors `harbor_subtype.py::_MAX_TASK_CONFIG_BYTES`.
MAX_TASK_CONFIG_BYTES = 1024 * 1024

TASK_CONFIG_PATH = "task.toml"
INSTRUCTION_PATH = "instruction.md"
DOCKERFILE_PATH = "environment/Dockerfile"

#: Run residue that must never reach an image or S3.
EXCLUDED_DIRECTORY_NAMES = frozenset(
    {
        ".codex",  # Codex OAuth bundle, logs, and sqlite state
        ".cache",  # uv archive tree
        "__pycache__",
        ".git",
        ".venv",
        "node_modules",
        "target",  # Rust build output
    }
)

#: Filenames that indicate a credential regardless of directory.
_CREDENTIAL_FILENAMES = frozenset({"auth.json", ".env", "credentials.json", ".netrc"})

_COPY_SOURCE = re.compile(r"^\s*(COPY|ADD)\s+(?P<rest>.+)$", re.IGNORECASE)


class HarborBundleError(ValueError):
    """The bundle cannot be packaged for a container pool."""


@dataclass(frozen=True, slots=True)
class HarborBundle:
    """A packaged Harbor task bundle plus the facts the caller has to record."""

    archive_base64: str
    archive_bytes: int
    task_name: str
    #: Timeouts and resources the backend will derive from ``task.toml``.
    agent_timeout_s: int
    verifier_timeout_s: int
    allow_internet: bool
    file_count: int

    @property
    def rollout_timeout_s(self) -> int:
        """What the backend will set as the rollout ceiling.

        ``materialize_harbor_runtime_release`` computes
        ``agent + verifier + 300``. Platform ceiling is 6h.
        """
        return self.agent_timeout_s + self.verifier_timeout_s + 300


def _is_excluded(relative: PurePosixPath) -> bool:
    return any(part in EXCLUDED_DIRECTORY_NAMES for part in relative.parts)


def _check_dockerfile_context(dockerfile: Path, *, bundle_root: Path) -> None:
    """Reject a Dockerfile whose build context is wider than its own directory."""

    offenders: list[str] = []
    for number, line in enumerate(dockerfile.read_text(encoding="utf-8").splitlines(), 1):
        match = _COPY_SOURCE.match(line)
        if match is None:
            continue
        rest = match.group("rest")
        if "--from=" in rest:
            # A multi-stage copy reads from an earlier stage, not the context.
            continue
        # Everything but the final argument is a source path.
        arguments = rest.split()
        for source in arguments[:-1]:
            if source.startswith("--"):
                continue
            candidate = PurePosixPath(source)
            if candidate.is_absolute() or ".." in candidate.parts:
                offenders.append(f"  line {number}: {line.strip()}")
                continue
            # The context root is environment/; a source naming a sibling of
            # environment/ (tasks/, adapters/, ...) resolves outside it.
            if not (dockerfile.parent / source).exists():
                offenders.append(f"  line {number}: {line.strip()}")
    if offenders:
        relative = dockerfile.relative_to(bundle_root)
        raise HarborBundleError(
            f"{relative} copies from outside its own directory. The container-pool "
            "build context is the Dockerfile's directory, not the bundle root, so "
            "these instructions cannot resolve:\n"
            + "\n".join(offenders)
            + "\n\nVendor the needed files under "
            f"{dockerfile.parent.relative_to(bundle_root)}/ or rewrite the paths."
        )


def _read_task_config(bundle_root: Path) -> dict[str, object]:
    task_config = bundle_root / TASK_CONFIG_PATH
    if not task_config.is_file():
        raise HarborBundleError(f"Harbor bundle requires {TASK_CONFIG_PATH} at its root.")
    raw = task_config.read_bytes()
    if len(raw) > MAX_TASK_CONFIG_BYTES:
        raise HarborBundleError(
            f"{TASK_CONFIG_PATH} is {len(raw)} bytes; the backend rejects anything "
            f"over {MAX_TASK_CONFIG_BYTES}."
        )
    try:
        return tomllib.loads(raw.decode("utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise HarborBundleError(f"{TASK_CONFIG_PATH} is not valid TOML: {error}") from error


def build_harbor_bundle_archive(
    bundle_dir: str | Path,
    *,
    dockerfile_path: str = DOCKERFILE_PATH,
    allow_credential_files: bool = False,
) -> HarborBundle:
    """Package ``bundle_dir`` into a base64 tar.gz for a Harbor runtime release.

    Raises :class:`HarborBundleError` rather than producing an archive that would
    fail at build or bind time, or that would ship a credential.
    """

    bundle_root = Path(bundle_dir).expanduser().resolve()
    if not bundle_root.is_dir():
        raise HarborBundleError(f"{bundle_root} is not a directory.")

    task_payload = _read_task_config(bundle_root)
    if not (bundle_root / INSTRUCTION_PATH).is_file():
        raise HarborBundleError(f"Harbor bundle requires {INSTRUCTION_PATH} at its root.")

    dockerfile = bundle_root / dockerfile_path
    if not dockerfile.is_file():
        raise HarborBundleError(f"Harbor bundle requires {dockerfile_path}.")
    _check_dockerfile_context(dockerfile, bundle_root=bundle_root)

    if not (bundle_root / "tests" / "test.sh").is_file():
        raise HarborBundleError(
            "Harbor bundle requires tests/test.sh; it is injected at /tests and run "
            "as the verifier phase."
        )

    members: list[tuple[Path, str]] = []
    for path in sorted(bundle_root.rglob("*")):
        if not path.is_file():
            continue
        relative = PurePosixPath(path.relative_to(bundle_root).as_posix())
        if _is_excluded(relative):
            continue
        if not allow_credential_files and relative.name in _CREDENTIAL_FILENAMES:
            raise HarborBundleError(
                f"{relative} looks like a credential and would ship inside the "
                "container image and to S3. Remove it, or pass "
                "allow_credential_files=True if it is genuinely task fixture data."
            )
        members.append((path, relative.as_posix()))

    if not members:
        raise HarborBundleError(f"{bundle_root} has no packageable files.")

    with io.BytesIO() as buffer:
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            for path, arcname in members:
                archive.add(path, arcname=arcname)
        raw = buffer.getvalue()

    if len(raw) > MAX_ARCHIVE_BYTES:
        raise HarborBundleError(
            f"Bundle archive is {len(raw)} bytes; the backend rejects anything over "
            f"{MAX_ARCHIVE_BYTES}. Check for build output left in the bundle."
        )
    encoded = base64.b64encode(raw).decode("ascii")
    if len(encoded) > MAX_BASE64_CHARS:
        raise HarborBundleError(
            f"Encoded bundle is {len(encoded)} characters; the backend rejects "
            f"anything over {MAX_BASE64_CHARS}."
        )

    task_section = task_payload.get("task")
    task_section = task_section if isinstance(task_section, dict) else {}
    agent_section = task_payload.get("agent")
    agent_section = agent_section if isinstance(agent_section, dict) else {}
    verifier_section = task_payload.get("verifier")
    verifier_section = verifier_section if isinstance(verifier_section, dict) else {}
    environment_section = task_payload.get("environment")
    environment_section = environment_section if isinstance(environment_section, dict) else {}

    return HarborBundle(
        archive_base64=encoded,
        archive_bytes=len(raw),
        task_name=str(task_section.get("name") or bundle_root.name),
        agent_timeout_s=int(agent_section.get("timeout_sec") or 900),
        verifier_timeout_s=int(verifier_section.get("timeout_sec") or 900),
        allow_internet=environment_section.get("allow_internet") is not False,
        file_count=len(members),
    )


__all__ = [
    "DOCKERFILE_PATH",
    "EXCLUDED_DIRECTORY_NAMES",
    "HarborBundle",
    "HarborBundleError",
    "INSTRUCTION_PATH",
    "MAX_ARCHIVE_BYTES",
    "TASK_CONFIG_PATH",
    "build_harbor_bundle_archive",
]
