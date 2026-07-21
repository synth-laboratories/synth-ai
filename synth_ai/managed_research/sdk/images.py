"""Customer actor image namespace: upload OCI archives, receive executable releases.

``client.research.images`` turns a customer-built OCI layout
archive into an immutable, org-scoped, digest-pinned runtime image release.
The returned ``release_id`` binds a run's worker role through
``actor_image_overrides``; the imgrel artifact identity stays available for
audits alongside it.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, List

import httpx

from synth_ai.managed_research.errors import SmrApiError
from synth_ai.managed_research.sdk._base import _ClientNamespace

try:
    import fcntl
except ImportError:  # pragma: no cover - Craftax managed images require Unix Docker hosts.
    fcntl = None

ACTOR_RUNTIME_IMAGE_KIND = "actor_runtime"
ACTOR_RUNTIME_INTERFACE_MODE = "synth_actor_runtime"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_RELEASE_ID = re.compile(r"^imgrel_[0-9a-f]{64}$")
_IMAGE_NAME = re.compile(r"^[a-z0-9]+(?:[._/-][a-z0-9]+)*$")
_IMAGE_TAG = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")
_CAPABILITY = re.compile(r"^[a-z0-9][a-z0-9_]{0,63}$")
_PACKAGE_REQUIREMENT = re.compile(
    r"^(?P<name>[a-z0-9][a-z0-9._-]*)==(?P<version>[A-Za-z0-9][A-Za-z0-9.!+_-]{0,127})$"
)
_PLATFORMS = ("linux/amd64", "linux/arm64")
CUSTOMER_ACTOR_IMAGE_PACKAGE_LABEL = "io.synth.actor-runtime.python-packages"
CRAFTAX_WORKER_BASE_IMAGE = "synth-local-open-research-craftax:latest"
CRAFTAX_WORKER_SCORER_PATH = "/opt/synth/task-assets/craftax_repl"
CRAFTAX_WORKER_CAPABILITIES = (
    "craftax_eval",
    "jax_cpu",
    "managed_research_sdk",
    "mcp_client",
    "synth_sdk",
)
CUSTOMER_ACTOR_IMAGE_PYPI_ALLOWLIST = frozenset(
    {
        "chex", "craftax", "crafter", "distrax", "flax", "gymnasium", "gymnax",
        "httpx", "imageio", "imageio-ffmpeg", "jax", "modal", "nle", "numpy",
        "optax", "orbax-checkpoint", "pillow", "pydantic", "pyyaml", "synth-ai",
    }
)


@dataclass(frozen=True, slots=True)
class ActorImage:
    """An uploaded customer actor image and its executable release identity."""

    release_id: str
    """Org-owned runtime image release ID; bind this in actor_image_overrides."""

    image_release_id: str
    """Content-addressed immutable artifact identity (imgrel_...)."""

    status: str
    image_ref: str
    digest: str
    platform: str
    actor_role: str
    capabilities: tuple[str, ...]
    archive_sha256: str
    python_packages: tuple[str, ...] = ()
    package_release_timestamps: Mapping[str, str] | None = None
    recipe_digest: str | None = None

    image_substrates: tuple[str, ...] = ()
    """Where the executable image lives: 'org_registry' means registry-pulling
    hosts (Daytona) can run it; 'wasabi_artifact' means docker-load hosts can."""

    daytona_pullable: bool = False


@dataclass(frozen=True, slots=True)
class ActorImageRecipe:
    """A constrained, cacheable worker-image recipe built by the public SDK.

    This deliberately exposes no Dockerfile shell commands, arbitrary base
    references, or unchecked package resolver behavior. The resulting OCI
    archive is uploaded through the same immutable release protocol as a
    customer-built image.
    """

    name: str
    base_image: str
    source_repository: str
    source_commit_sha: str
    base_image_id: str
    capabilities: tuple[str, ...]
    python_packages: tuple[str, ...]
    assets: tuple[tuple[Path, str], ...] = ()
    platform: str = "linux/arm64"

    @classmethod
    def craftax_worker(
        cls,
        *,
        craftax_repl_path: str | Path,
        source: Mapping[str, Any],
        python_packages: Sequence[str] = (),
    ) -> "ActorImageRecipe":
        repository, commit_sha = _normalized_source(source)
        scorer = Path(craftax_repl_path).expanduser().resolve()
        if not scorer.is_file():
            raise ValueError(f"Craftax scorer asset is not a file: {scorer}")
        header = scorer.read_bytes()[:20]
        if header[:4] != b"\x7fELF" or header[18:20] != b"\xb7\x00":
            raise ValueError("Craftax scorer asset must be an ELF Linux aarch64 binary")
        return cls(
            name="craftax-worker",
            base_image=CRAFTAX_WORKER_BASE_IMAGE,
            source_repository=repository,
            source_commit_sha=commit_sha,
            base_image_id=_local_image_id(CRAFTAX_WORKER_BASE_IMAGE),
            capabilities=CRAFTAX_WORKER_CAPABILITIES,
            python_packages=tuple(_normalized_python_packages(python_packages)),
            assets=((scorer, CRAFTAX_WORKER_SCORER_PATH),),
        )

    def recipe_digest(self) -> str:
        payload = {
            "name": self.name,
            "base_image": self.base_image,
            "base_image_id": self.base_image_id,
            "source_repository": self.source_repository,
            "source_commit_sha": self.source_commit_sha,
            "capabilities": sorted(self.capabilities),
            "python_packages": sorted(self.python_packages),
            "assets": [
                {
                    "sha256": _sha256_file(path),
                    "destination": destination,
                }
                for path, destination in self.assets
            ],
            "platform": self.platform,
        }
        return "sha256:" + hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def validate(self) -> None:
        if self.name != "craftax-worker":
            raise ValueError("only the craftax-worker managed recipe is currently supported")
        if self.base_image != CRAFTAX_WORKER_BASE_IMAGE:
            raise ValueError("craftax-worker must use the screened Craftax actor base image")
        if not _DIGEST.fullmatch(self.base_image_id):
            raise ValueError("craftax-worker must record a local sha256 base-image identity")
        if self.platform != "linux/arm64":
            raise ValueError("craftax-worker is currently available only for linux/arm64")
        _normalized_source(
            {"repository": self.source_repository, "commit_sha": self.source_commit_sha}
        )
        _normalized_python_packages(self.python_packages)
        if tuple(sorted(self.capabilities)) != CRAFTAX_WORKER_CAPABILITIES:
            raise ValueError("craftax-worker capabilities must match the screened recipe")
        if not self.assets:
            raise ValueError("craftax-worker requires the Craftax scorer asset")
        for path, destination in self.assets:
            if not path.is_file():
                raise ValueError(f"managed image asset is not a file: {path}")
            header = path.read_bytes()[:20]
            if header[:4] != b"\x7fELF" or header[18:20] != b"\xb7\x00":
                raise ValueError("Craftax scorer asset must be an ELF Linux aarch64 binary")
            normalized_destination = PurePosixPath(destination)
            if str(normalized_destination) != CRAFTAX_WORKER_SCORER_PATH:
                raise ValueError(
                    "managed Craftax assets may only target " + CRAFTAX_WORKER_SCORER_PATH
                )


def _text(payload: Mapping[str, Any], key: str, *, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.{key} must be a nonempty string")
    return value.strip()


def _actor_image_from_materialization(
    materialization: Mapping[str, Any],
    *,
    platform: str,
    archive_sha256: str,
) -> ActorImage:
    release_id = _text(materialization, "runtime_image_release_id", label="runtime_image_release")
    digest = _text(materialization, "resolved_digest", label="runtime_image_release")
    if not _DIGEST.fullmatch(digest):
        raise ValueError("runtime_image_release.resolved_digest is not a sha256 digest")
    image_release_id = _text(materialization, "image_release_id", label="runtime_image_release")
    if not _RELEASE_ID.fullmatch(image_release_id):
        raise ValueError("runtime_image_release.image_release_id is invalid")
    capabilities = materialization.get("capabilities")
    if not isinstance(capabilities, Sequence) or isinstance(capabilities, str):
        raise ValueError("runtime_image_release.capabilities must be a list")
    return ActorImage(
        release_id=release_id,
        image_release_id=image_release_id,
        status=_text(materialization, "status", label="runtime_image_release"),
        image_ref=_text(materialization, "image_ref", label="runtime_image_release"),
        digest=digest,
        platform=platform,
        actor_role=_text(materialization, "actor_role", label="runtime_image_release"),
        capabilities=tuple(str(item) for item in capabilities),
        archive_sha256=archive_sha256,
        python_packages=tuple(
            str(item) for item in (materialization.get("python_packages") or ())
        ),
        package_release_timestamps=(
            {
                str(key): str(value)
                for key, value in materialization.get("package_release_timestamps", {}).items()
            }
            if isinstance(materialization.get("package_release_timestamps"), Mapping)
            else None
        ),
        recipe_digest=(
            str(materialization.get("recipe_digest") or "").strip() or None
        ),
        image_substrates=tuple(
            str(item) for item in (materialization.get("image_substrates") or ())
        ),
        daytona_pullable=bool(materialization.get("daytona_pullable")),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_archive_member(archive: tarfile.TarFile, name: str) -> bytes:
    try:
        handle = archive.extractfile(name)
    except KeyError as exc:
        raise ValueError(f"OCI archive is missing {name}") from exc
    if handle is None:
        raise ValueError(f"OCI archive member {name} is unreadable")
    return handle.read(4 * 1024 * 1024)


def inspect_oci_archive(archive_path: Path) -> dict[str, str]:
    """Read the manifest digest and platform out of a single-image OCI archive."""
    with tarfile.open(archive_path, mode="r:*") as archive:
        index = json.loads(_read_archive_member(archive, "index.json"))
        manifests = index.get("manifests")
        if not isinstance(manifests, list) or len(manifests) != 1:
            raise ValueError("OCI archive must contain exactly one image manifest")
        descriptor = manifests[0]
        if (
            isinstance(descriptor, Mapping)
            and descriptor.get("mediaType") == "application/vnd.oci.image.index.v1+json"
        ):
            # containerd-backed `docker save` nests a single-image index one
            # level below the top index; chase exactly one hop.
            nested_digest = str(descriptor.get("digest") or "")
            if not _DIGEST.fullmatch(nested_digest):
                raise ValueError("OCI archive nested index digest is invalid")
            nested = json.loads(
                _read_archive_member(
                    archive,
                    f"blobs/sha256/{nested_digest.removeprefix('sha256:')}",
                )
            )
            nested_manifests = [
                item
                for item in (nested.get("manifests") or [])
                if isinstance(item, Mapping)
                and not (
                    isinstance(item.get("annotations"), Mapping)
                    and item["annotations"].get("vnd.docker.reference.type")
                    == "attestation-manifest"
                )
            ]
            if len(nested_manifests) != 1:
                raise ValueError("OCI archive nested index must contain exactly one image manifest")
            descriptor = nested_manifests[0]
        manifest_digest = str(descriptor.get("digest") or "")
        if not _DIGEST.fullmatch(manifest_digest):
            raise ValueError("OCI archive manifest digest is invalid")
        platform = descriptor.get("platform")
        if isinstance(platform, Mapping) and platform.get("os"):
            platform_os = str(platform.get("os") or "").lower()
            platform_architecture = str(platform.get("architecture") or "").lower()
        else:
            manifest = json.loads(
                _read_archive_member(
                    archive,
                    f"blobs/sha256/{manifest_digest.removeprefix('sha256:')}",
                )
            )
            config_digest = str((manifest.get("config") or {}).get("digest") or "")
            if not _DIGEST.fullmatch(config_digest):
                raise ValueError("OCI archive config digest is invalid")
            config = json.loads(
                _read_archive_member(
                    archive,
                    f"blobs/sha256/{config_digest.removeprefix('sha256:')}",
                )
            )
            platform_os = str(config.get("os") or "").lower()
            platform_architecture = str(config.get("architecture") or "").lower()
    return {
        "image_manifest_digest": manifest_digest,
        "platform_os": platform_os,
        "platform_architecture": platform_architecture,
    }


def _normalized_source(source: Mapping[str, Any]) -> tuple[str, str]:
    if not isinstance(source, Mapping):
        raise ValueError("source must be a mapping with repository and commit_sha")
    unexpected = set(source) - {"repository", "commit_sha"}
    if unexpected:
        raise ValueError(f"source has unexpected fields: {sorted(unexpected)}")
    repository = str(source.get("repository") or "").strip()
    commit_sha = str(source.get("commit_sha") or "").strip().lower()
    if not repository.startswith("https://github.com/"):
        raise ValueError("source.repository must be an HTTPS GitHub repository URL")
    if not _GIT_SHA.fullmatch(commit_sha):
        raise ValueError("source.commit_sha must be a full lowercase Git commit SHA")
    return repository, commit_sha


def _local_image_id(image_ref: str) -> str:
    """Resolve the screened local base and make tag movement cache-visible."""
    result = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image_ref],
        capture_output=True,
        text=True,
        check=False,
    )
    image_id = result.stdout.strip().lower()
    if result.returncode or not _DIGEST.fullmatch(image_id):
        detail = (result.stderr or result.stdout or "image was not found").strip()
        raise RuntimeError(
            f"screened Craftax base image is unavailable: {image_ref} ({detail[-500:]})"
        )
    return image_id


def _pinned_base_image_ref(recipe: ActorImageRecipe) -> str:
    """Return the exact local base reference used by the constrained builder."""
    if recipe.base_image != CRAFTAX_WORKER_BASE_IMAGE:
        raise ValueError("only the screened Craftax base image may be pinned")
    return recipe.base_image.removesuffix(":latest") + "@" + recipe.base_image_id


@contextmanager
def _recipe_cache_lock(archive_path: Path):
    """Serialize one local recipe build/upload across concurrent benchmark lanes."""
    lock_path = archive_path.with_suffix(archive_path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        if fcntl is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _normalized_python_packages(values: Sequence[str]) -> list[str]:
    if len(values) > 32:
        raise ValueError("python_packages must list at most 32 exact-pinned packages")
    normalized: list[str] = []
    for raw_value in values:
        value = str(raw_value or "").strip()
        match = _PACKAGE_REQUIREMENT.fullmatch(value)
        if match is None:
            raise ValueError("python_packages entries must use the exact package==version form")
        package_name = re.sub(r"[-_.]+", "-", match.group("name").lower())
        if package_name not in CUSTOMER_ACTOR_IMAGE_PYPI_ALLOWLIST:
            raise ValueError(f"python package is not allowlisted: {package_name}")
        normalized.append(f"{package_name}=={match.group('version')}")
    if len(set(normalized)) != len(normalized):
        raise ValueError("python_packages must not repeat packages")
    return sorted(normalized)


class ImagesAPI(_ClientNamespace):
    """Customer actor images: upload, read, list, and archive org-owned releases."""

    def upload_archive(
        self,
        *,
        name: str,
        archive_path: str | Path,
        source: Mapping[str, Any],
        capabilities: Sequence[str],
        python_packages: Sequence[str],
        recipe_digest: str | None = None,
        kind: str = ACTOR_RUNTIME_IMAGE_KIND,
        role: str = "worker",
        platform: str | None = None,
        tag: str | None = None,
        expires_in: int = 3600,
        upload_timeout_seconds: float = 1800.0,
    ) -> ActorImage:
        """Upload a declared, allowlisted OCI runtime image and return its release.

        The OCI config must set ``io.synth.actor-runtime.python-packages`` to the
        canonical JSON list of the same exact-pinned requirements. The control
        plane independently rejects versions published less than seven days ago.
        """
        if kind != ACTOR_RUNTIME_IMAGE_KIND:
            raise ValueError(
                "images.upload_archive only uploads actor_runtime images; "
                "scorer releases use client.research.image_releases"
            )
        normalized_name = str(name or "").strip()
        if not _IMAGE_NAME.fullmatch(normalized_name):
            raise ValueError(
                "name must be a lowercase image repository name (for example 'craftax-worker')"
            )
        normalized_capabilities = [str(item or "").strip().lower() for item in capabilities]
        if not normalized_capabilities or any(
            not _CAPABILITY.fullmatch(item) for item in normalized_capabilities
        ):
            raise ValueError("capabilities must be nonempty lowercase snake_case slugs")
        normalized_python_packages = _normalized_python_packages(python_packages)
        normalized_recipe_digest = str(recipe_digest or "").strip() or None
        if normalized_recipe_digest is not None and not _DIGEST.fullmatch(normalized_recipe_digest):
            raise ValueError("recipe_digest must be sha256:<64 lowercase hex>")
        if platform is not None and platform not in _PLATFORMS:
            raise ValueError(f"platform must be one of: {', '.join(_PLATFORMS)}")
        repository, commit_sha = _normalized_source(source)
        path = Path(archive_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"actor image archive is not a file: {path}")
        archive_sha256 = _sha256_file(path)
        archive_size_bytes = path.stat().st_size
        inspection = inspect_oci_archive(path)
        actual_platform = f"{inspection['platform_os']}/{inspection['platform_architecture']}"
        if platform is not None and platform != actual_platform:
            raise ValueError(
                f"platform mismatch: archive is {actual_platform}, declaration requested {platform}"
            )
        normalized_tag = str(tag or archive_sha256[:12]).strip()
        if not _IMAGE_TAG.fullmatch(normalized_tag):
            raise ValueError("tag must be a valid image tag")
        declaration: dict[str, Any] = {
            "kind": ACTOR_RUNTIME_IMAGE_KIND,
            "archive_sha256": archive_sha256,
            "archive_size_bytes": archive_size_bytes,
            "image_manifest_digest": inspection["image_manifest_digest"],
            "image_ref": f"{normalized_name}:{normalized_tag}",
            "platform_os": inspection["platform_os"],
            "platform_architecture": inspection["platform_architecture"],
            "source_repository": repository,
            "source_commit_sha": commit_sha,
            "actor_role": str(role or "").strip().lower(),
            "interface_mode": ACTOR_RUNTIME_INTERFACE_MODE,
            "capabilities": normalized_capabilities,
            "python_packages": normalized_python_packages,
        }
        if normalized_recipe_digest is not None:
            declaration["recipe_digest"] = normalized_recipe_digest
        upload = self._client._request_json(
            "POST",
            "/smr/v1/image-releases/upload-url",
            json_body={"declaration": declaration, "expires_in": expires_in},
        )
        if not isinstance(upload, Mapping):
            raise ValueError("image upload response must be an object")
        upload_id = _text(upload, "upload_id", label="image_upload")
        upload_required = upload.get("upload_required") is not False
        if upload_required:
            upload_url = _text(upload, "upload_url", label="image_upload")
            # Loopback HTTP is the local-stack MinIO shape; anything else must be HTTPS.
            if not upload_url.startswith(("https://", "http://localhost", "http://127.")):
                raise ValueError("image upload_url must use HTTPS")
            self._put_archive(
                upload_url,
                path,
                archive_size_bytes=archive_size_bytes,
                upload_timeout_seconds=upload_timeout_seconds,
                upload_id=upload_id,
            )
        finalized = self._finalize(
            upload_id=upload_id,
            declaration=declaration,
            timeout_seconds=upload_timeout_seconds,
        )
        release = finalized.get("release")
        if not isinstance(release, Mapping):
            raise ValueError("image finalize response must include the release")
        materialization = release.get("runtime_image_release")
        if not isinstance(materialization, Mapping):
            raise ValueError(
                "image finalize did not materialize an executable runtime image "
                "release; the upload is not runnable"
            )
        return _actor_image_from_materialization(
            materialization,
            platform=actual_platform,
            archive_sha256=archive_sha256,
        )

    def ensure_recipe(self, recipe: ActorImageRecipe) -> ActorImage:
        """Build once, reuse the local OCI archive, and reuse an admitted release.

        A release UUID is an implementation detail.  Recipe identity is the
        stable API: the SDK first finds a matching active release, otherwise it
        builds a deterministic OCI archive in its user cache and uploads it.
        """
        recipe.validate()
        recipe_digest = recipe.recipe_digest()
        active_release = self._active_recipe_release(recipe_digest)
        if active_release is not None:
            return active_release
        cache_root = Path.home() / ".cache" / "synth-ai" / "actor-images"
        cache_root.mkdir(parents=True, exist_ok=True)
        archive_path = cache_root / f"{recipe_digest.removeprefix('sha256:')}.oci.tar"
        with _recipe_cache_lock(archive_path):
            active_release = self._active_recipe_release(recipe_digest)
            if active_release is not None:
                return active_release
            if not archive_path.is_file():
                self._build_recipe_oci_archive(recipe, archive_path=archive_path)
            return self.upload_archive(
                name=recipe.name,
                archive_path=archive_path,
                source={
                    "repository": recipe.source_repository,
                    "commit_sha": recipe.source_commit_sha,
                },
                capabilities=recipe.capabilities,
                python_packages=recipe.python_packages,
                platform=recipe.platform,
                tag=f"recipe-{recipe_digest.removeprefix('sha256:')[:12]}",
                recipe_digest=recipe_digest,
            )

    def _active_recipe_release(self, recipe_digest: str) -> ActorImage | None:
        for item in self.list():
            if (
                str(item.get("recipe_digest") or "").strip() == recipe_digest
                and str(item.get("status") or "").strip().lower() != "archived"
            ):
                image_release_id = str(item.get("image_release_id") or "").strip()
                if image_release_id:
                    return self.get(image_release_id=image_release_id)
        return None

    @staticmethod
    def _build_recipe_oci_archive(recipe: ActorImageRecipe, *, archive_path: Path) -> None:
        """Build a constrained managed recipe as a single-image OCI archive."""
        if _local_image_id(recipe.base_image) != recipe.base_image_id:
            raise RuntimeError(
                "screened Craftax base image changed while creating the managed recipe; "
                "retry to derive a new cache key"
            )
        package_label = json.dumps(list(sorted(recipe.python_packages)))
        with tempfile.TemporaryDirectory(prefix="synth-actor-image-") as temporary:
            context = Path(temporary)
            assets = context / "assets"
            assets.mkdir()
            dockerfile_lines = [
                f"FROM {_pinned_base_image_ref(recipe)}",
                f"LABEL {CUSTOMER_ACTOR_IMAGE_PACKAGE_LABEL}={json.dumps(package_label)}",
            ]
            for index, (source, destination) in enumerate(recipe.assets):
                asset_name = f"asset-{index}"
                shutil.copy2(source, assets / asset_name)
                dockerfile_lines.append(
                    f"COPY --chmod=0755 assets/{asset_name} {destination}"
                )
            if recipe.python_packages:
                dockerfile_lines.append(
                    "RUN python -m pip install --no-cache-dir --no-deps "
                    + " ".join(recipe.python_packages)
                )
            (context / "Dockerfile").write_text(
                "\n".join(dockerfile_lines) + "\n", encoding="utf-8"
            )
            temporary_archive = context / "image.oci.tar"
            result = subprocess.run(
                [
                    "docker",
                    "buildx",
                    "build",
                    "--platform",
                    recipe.platform,
                    "--output",
                    f"type=oci,dest={temporary_archive}",
                    str(context),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0 or not temporary_archive.is_file():
                detail = (result.stderr or result.stdout or "Docker buildx produced no archive").strip()
                raise RuntimeError(f"managed actor image build failed: {detail[-2000:]}")
            shutil.move(str(temporary_archive), archive_path)

    def _put_archive(
        self,
        upload_url: str,
        path: Path,
        *,
        archive_size_bytes: int,
        upload_timeout_seconds: float,
        upload_id: str,
    ) -> None:
        upload_error: httpx.TransportError | None = None
        with httpx.Client(
            timeout=upload_timeout_seconds,
            follow_redirects=False,
        ) as upload_client:
            for _attempt in range(2):
                try:
                    with path.open("rb") as handle:
                        response = upload_client.put(
                            upload_url,
                            content=handle,
                            headers={"Content-Length": str(archive_size_bytes)},
                        )
                    upload_error = None
                    break
                except httpx.TransportError as exc:
                    # Exact-key PUT is idempotent; one replay reconciles the only
                    # ambiguous outcome without crossing the storage boundary.
                    upload_error = exc
            else:
                raise RuntimeError(
                    f"actor image archive upload outcome is uncertain (upload_id={upload_id})"
                ) from upload_error
        if response.is_error:
            # Object stores put the actionable SigV4 failure code in the
            # response body (for example, a signed-header mismatch). Keep a
            # bounded, whitespace-normalized diagnostic so launch receipts
            # identify a deterministic storage-contract failure without
            # dumping a potentially large provider error document.
            detail = " ".join(response.text.split())[:600]
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(
                f"actor image archive upload failed with HTTP {response.status_code}{suffix}"
            )

    def _finalize(
        self,
        *,
        upload_id: str,
        declaration: Mapping[str, Any],
        timeout_seconds: float = 1800.0,
    ) -> Mapping[str, Any]:
        try:
            finalized = self._client._request_json(
                "POST",
                "/smr/v1/image-releases/finalize",
                json_body={"upload_id": upload_id, "declaration": dict(declaration)},
                timeout_seconds=timeout_seconds,
            )
        except SmrApiError as exc:
            if exc.status_code is not None:
                raise
            if isinstance(exc.__cause__, httpx.ConnectError):
                # A failed TCP connect proves the request was not admitted.
                finalized = self._client._request_json(
                    "POST",
                    "/smr/v1/image-releases/finalize",
                    json_body={
                        "upload_id": upload_id,
                        "declaration": dict(declaration),
                    },
                    timeout_seconds=timeout_seconds,
                )
            else:
                # A read timeout or disconnect can happen after the backend has
                # begun a long registry publication.  Replaying here would run
                # that materialization twice and multiply storage/network use.
                raise RuntimeError(
                    "actor image finalize outcome is uncertain; retry "
                    "images.ensure_recipe after backend reconciliation "
                    f"(upload_id={upload_id})"
                ) from exc
        if not isinstance(finalized, Mapping):
            raise ValueError("image finalize response must be an object")
        return finalized

    def get(self, *, image_release_id: str) -> ActorImage:
        """Read one uploaded actor image by its immutable artifact identity."""
        normalized = str(image_release_id or "").strip()
        if not _RELEASE_ID.fullmatch(normalized):
            raise ValueError("image_release_id must look like imgrel_<sha256>")
        receipt = self._client._request_json(
            "GET",
            f"/smr/v1/image-releases/{normalized}",
        )
        if not isinstance(receipt, Mapping):
            raise ValueError("image release response must be an object")
        declaration = receipt.get("declaration")
        if not isinstance(declaration, Mapping):
            raise ValueError("image release response must include its declaration")
        if declaration.get("kind") != ACTOR_RUNTIME_IMAGE_KIND:
            raise ValueError(
                "release is not an actor_runtime image; read scorer releases "
                "through client.research.image_releases"
            )
        materialization = receipt.get("runtime_image_release")
        if not isinstance(materialization, Mapping):
            raise ValueError("image release exists but has no executable runtime image release")
        return _actor_image_from_materialization(
            materialization,
            platform=(
                f"{declaration.get('platform_os')}/{declaration.get('platform_architecture')}"
            ),
            archive_sha256=str(declaration.get("archive_sha256") or ""),
        )

    def status(self, *, image_release_id: str) -> str:
        """Return the executable release status for an uploaded actor image."""
        return self.get(image_release_id=image_release_id).status

    def list(self) -> List[dict[str, Any]]:
        """List this org's customer actor runtime image releases."""
        payload = self._client._request_json("GET", "/smr/v1/image-releases")
        if not isinstance(payload, Mapping) or not isinstance(payload.get("releases"), list):
            raise ValueError("image list response must include releases")
        return [dict(item) for item in payload["releases"] if isinstance(item, Mapping)]

    def archive(self, *, release_id: str) -> dict[str, Any]:
        """Archive an org-owned actor runtime image release (blocks new runs)."""
        normalized = str(release_id or "").strip()
        if not normalized:
            raise ValueError("release_id must be a nonempty string")
        payload = self._client._request_json(
            "POST",
            f"/smr/v1/image-releases/{normalized}/archive",
        )
        if not isinstance(payload, Mapping) or not isinstance(
            payload.get("runtime_image_release"), Mapping
        ):
            raise ValueError("image archive response must include the release")
        return dict(payload["runtime_image_release"])


__all__ = [
    "ACTOR_RUNTIME_IMAGE_KIND",
    "ACTOR_RUNTIME_INTERFACE_MODE",
    "ActorImage",
    "ActorImageRecipe",
    "CRAFTAX_WORKER_BASE_IMAGE",
    "CRAFTAX_WORKER_CAPABILITIES",
    "CRAFTAX_WORKER_SCORER_PATH",
    "CUSTOMER_ACTOR_IMAGE_PACKAGE_LABEL",
    "CUSTOMER_ACTOR_IMAGE_PYPI_ALLOWLIST",
    "ImagesAPI",
    "inspect_oci_archive",
]
