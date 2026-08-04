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


def _load_dockerignore(dockerfile: Path) -> list[str]:
    """Read ignore patterns for a build context.

    The backend never sees a ``.dockerignore``: the archive the client uploads
    *is* the context. Applying the rules here is therefore the only thing
    keeping excluded trees out of the image and out of S3 — and for some
    contexts it is the difference between fitting the archive cap and not.
    BuildKit prefers ``<dockerfile>.dockerignore`` over a context-root
    ``.dockerignore``, so this checks the same two locations in the same order.
    """
    for candidate in (
        dockerfile.with_name(dockerfile.name + ".dockerignore"),
        dockerfile.parent / ".dockerignore",
    ):
        if not candidate.is_file():
            continue
        patterns: list[str] = []
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#"):
                patterns.append(line)
        return patterns
    return []


def _matches_dockerignore(relative: PurePosixPath, patterns: list[str]) -> bool:
    """Approximate Docker's ignore matching for the patterns people actually write.

    Negations (``!pattern``) are treated as "do not ignore", matching Docker.
    This is deliberately conservative: an unmatched pattern only means a file is
    included, so the worst case is a larger archive, never a missing file.
    """
    ignored = False
    for pattern in patterns:
        negated = pattern.startswith("!")
        candidate = pattern[1:] if negated else pattern
        candidate = candidate.strip("/")
        if not candidate:
            continue
        matched = relative.match(candidate) or any(
            PurePosixPath(*relative.parts[: index + 1]).match(candidate)
            for index in range(len(relative.parts))
        )
        if matched:
            ignored = not negated
    return ignored


_COPY_SOURCE = re.compile(r"^\s*(COPY|ADD)\s+(?P<rest>.+)$", re.IGNORECASE)
_ARG_DEFAULT = re.compile(
    r"^\s*ARG\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>.*)$", re.IGNORECASE
)
_INTERPOLATION = re.compile(
    r"\$\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|\$(?P<bare>[A-Za-z_][A-Za-z0-9_]*)"
)


def _expand_build_args(source: str, build_args: dict[str, str]) -> str | None:
    """Substitute ``ARG`` defaults into a COPY source path.

    Build args are not plumbed through the container-pool build path — the
    backend's ``do_build`` takes a Dockerfile and a context and nothing else —
    so a Dockerfile's ARG *defaults* are what the image is actually built with.
    Expanding them here validates the paths the build will really resolve, and
    catches a bundle whose vendored contents disagree with its ARG defaults.

    Returns ``None`` when a referenced variable has no default, since the value
    is then genuinely unknown and the path cannot be checked either way.
    """
    unresolved = False

    def _replace(match: re.Match[str]) -> str:
        nonlocal unresolved
        name = match.group("braced") or match.group("bare")
        if name not in build_args:
            unresolved = True
            return ""
        return build_args[name]

    expanded = _INTERPOLATION.sub(_replace, source)
    return None if unresolved else expanded


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
    build_args: dict[str, str] = {}
    for number, line in enumerate(dockerfile.read_text(encoding="utf-8").splitlines(), 1):
        arg_match = _ARG_DEFAULT.match(line)
        if arg_match is not None:
            build_args[arg_match.group("name")] = arg_match.group("value").strip().strip("\"'")
            continue
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
            expanded = _expand_build_args(source, build_args)
            if expanded is None:
                # A variable with no ARG default: the build value is unknown, so
                # neither accepting nor rejecting the path would be honest.
                continue
            candidate = PurePosixPath(expanded)
            if candidate.is_absolute() or ".." in candidate.parts:
                offenders.append(f"  line {number}: {line.strip()}")
                continue
            # The context root is the Dockerfile's own directory; a source
            # naming anything above it cannot resolve during a pool build.
            if not (dockerfile.parent / expanded).exists():
                detail = f"  line {number}: {line.strip()}"
                if expanded != source:
                    detail += f"\n      (resolves to {expanded!r} via ARG defaults)"
                offenders.append(detail)
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

    ignore_patterns = _load_dockerignore(dockerfile)
    members: list[tuple[Path, str]] = []
    for path in sorted(bundle_root.rglob("*")):
        if not path.is_file():
            continue
        relative = PurePosixPath(path.relative_to(bundle_root).as_posix())
        if _is_excluded(relative):
            continue
        if ignore_patterns and _matches_dockerignore(relative, ignore_patterns):
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


@dataclass(frozen=True, slots=True)
class DockerContextArchive:
    """A packaged build context for an arbitrary (non-Harbor) container."""

    archive_base64: str
    archive_bytes: int
    file_count: int
    dockerfile_path: str
    ignored_by_dockerignore: int


def build_docker_context_archive(
    context_dir: str | Path,
    *,
    dockerfile_path: str = "Dockerfile",
    allow_credential_files: bool = False,
) -> DockerContextArchive:
    """Package a plain Docker build context for an ``arbitrary`` pool task.

    Unlike :func:`build_harbor_bundle_archive` this requires no task.toml,
    instruction, or tests — an arbitrary container serves the Synth HTTP
    contract rather than running agent and verifier phases. The context is
    rooted at ``context_dir`` rather than at the Dockerfile's directory, because
    the arbitrary path uploads the context as given.
    """

    root = Path(context_dir).expanduser().resolve()
    if not root.is_dir():
        raise HarborBundleError(f"{root} is not a directory.")
    dockerfile = root / dockerfile_path
    if not dockerfile.is_file():
        raise HarborBundleError(f"Build context is missing {dockerfile_path}.")

    ignore_patterns = _load_dockerignore(dockerfile)
    members: list[tuple[Path, str]] = []
    ignored = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = PurePosixPath(path.relative_to(root).as_posix())
        if _is_excluded(relative):
            ignored += 1
            continue
        if ignore_patterns and _matches_dockerignore(relative, ignore_patterns):
            ignored += 1
            continue
        if not allow_credential_files and relative.name in _CREDENTIAL_FILENAMES:
            raise HarborBundleError(
                f"{relative} looks like a credential and would ship inside the "
                "container image and to S3. Remove it, or pass "
                "allow_credential_files=True if it is genuinely fixture data."
            )
        members.append((path, relative.as_posix()))

    if not members:
        raise HarborBundleError(f"{root} has no packageable files.")

    with io.BytesIO() as buffer:
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            for path, arcname in members:
                archive.add(path, arcname=arcname)
        raw = buffer.getvalue()

    if len(raw) > MAX_ARCHIVE_BYTES:
        raise HarborBundleError(
            f"Build context archive is {len(raw)} bytes; the backend rejects anything "
            f"over {MAX_ARCHIVE_BYTES}. {ignored} files were already excluded — check "
            "for build output that no ignore rule covers."
        )
    encoded = base64.b64encode(raw).decode("ascii")
    if len(encoded) > MAX_BASE64_CHARS:
        raise HarborBundleError(
            f"Encoded context is {len(encoded)} characters; the backend rejects "
            f"anything over {MAX_BASE64_CHARS}."
        )

    return DockerContextArchive(
        archive_base64=encoded,
        archive_bytes=len(raw),
        file_count=len(members),
        dockerfile_path=dockerfile_path,
        ignored_by_dockerignore=ignored,
    )


__all__ = [
    "DOCKERFILE_PATH",
    "EXCLUDED_DIRECTORY_NAMES",
    "DockerContextArchive",
    "HarborBundle",
    "HarborBundleError",
    "INSTRUCTION_PATH",
    "MAX_ARCHIVE_BYTES",
    "TASK_CONFIG_PATH",
    "build_docker_context_archive",
    "build_harbor_bundle_archive",
]
