"""Customer actor image namespace: upload OCI archives, receive executable releases.

``client.research.images`` turns a customer-built ``linux/amd64`` OCI layout
archive into an immutable, org-scoped, digest-pinned runtime image release.
The returned ``release_id`` binds a run's worker role through
``actor_image_overrides``; the imgrel artifact identity stays available for
audits alongside it.
"""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from synth_ai.managed_research.errors import SmrApiError
from synth_ai.managed_research.sdk._base import _ClientNamespace

ACTOR_RUNTIME_IMAGE_KIND = "actor_runtime"
ACTOR_RUNTIME_INTERFACE_MODE = "synth_actor_runtime"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_RELEASE_ID = re.compile(r"^imgrel_[0-9a-f]{64}$")
_IMAGE_NAME = re.compile(r"^[a-z0-9]+(?:[._/-][a-z0-9]+)*$")
_IMAGE_TAG = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")
_CAPABILITY = re.compile(r"^[a-z0-9][a-z0-9_]{0,63}$")
_PLATFORMS = ("linux/amd64", "linux/arm64")


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

    image_substrates: tuple[str, ...] = ()
    """Where the executable image lives: 'org_registry' means registry-pulling
    hosts (Daytona) can run it; 'wasabi_artifact' means docker-load hosts can."""

    daytona_pullable: bool = False


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
    release_id = _text(
        materialization, "runtime_image_release_id", label="runtime_image_release"
    )
    digest = _text(materialization, "resolved_digest", label="runtime_image_release")
    if not _DIGEST.fullmatch(digest):
        raise ValueError("runtime_image_release.resolved_digest is not a sha256 digest")
    image_release_id = _text(
        materialization, "image_release_id", label="runtime_image_release"
    )
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
            and descriptor.get("mediaType")
            == "application/vnd.oci.image.index.v1+json"
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
                raise ValueError(
                    "OCI archive nested index must contain exactly one image manifest"
                )
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


class ImagesAPI(_ClientNamespace):
    """Customer actor images: upload, read, list, and archive org-owned releases."""

    def upload_archive(
        self,
        *,
        name: str,
        archive_path: str | Path,
        source: Mapping[str, Any],
        capabilities: Sequence[str],
        kind: str = ACTOR_RUNTIME_IMAGE_KIND,
        role: str = "worker",
        platform: str | None = None,
        tag: str | None = None,
        expires_in: int = 3600,
        upload_timeout_seconds: float = 1800.0,
    ) -> ActorImage:
        """Upload an OCI layout archive and return its executable release."""
        if kind != ACTOR_RUNTIME_IMAGE_KIND:
            raise ValueError(
                "images.upload_archive only uploads actor_runtime images; "
                "scorer releases use client.research.image_releases"
            )
        normalized_name = str(name or "").strip()
        if not _IMAGE_NAME.fullmatch(normalized_name):
            raise ValueError(
                "name must be a lowercase image repository name "
                "(for example 'craftax-worker')"
            )
        normalized_capabilities = [
            str(item or "").strip().lower() for item in capabilities
        ]
        if not normalized_capabilities or any(
            not _CAPABILITY.fullmatch(item) for item in normalized_capabilities
        ):
            raise ValueError("capabilities must be nonempty lowercase snake_case slugs")
        if platform is not None and platform not in _PLATFORMS:
            raise ValueError(f"platform must be one of: {', '.join(_PLATFORMS)}")
        repository, commit_sha = _normalized_source(source)
        path = Path(archive_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"actor image archive is not a file: {path}")
        archive_sha256 = _sha256_file(path)
        archive_size_bytes = path.stat().st_size
        inspection = inspect_oci_archive(path)
        actual_platform = (
            f"{inspection['platform_os']}/{inspection['platform_architecture']}"
        )
        if platform is not None and platform != actual_platform:
            raise ValueError(
                f"platform mismatch: archive is {actual_platform}, "
                f"declaration requested {platform}"
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
        }
        upload = self._client._request_json(
            "POST",
            "/smr/v1/image-releases/upload-url",
            json_body={"declaration": declaration, "expires_in": expires_in},
        )
        if not isinstance(upload, Mapping):
            raise ValueError("image upload response must be an object")
        upload_id = _text(upload, "upload_id", label="image_upload")
        upload_url = _text(upload, "upload_url", label="image_upload")
        # Loopback HTTP is the local-stack MinIO shape; anything else must be HTTPS.
        if not upload_url.startswith(
            ("https://", "http://localhost", "http://127.")
        ):
            raise ValueError("image upload_url must use HTTPS")
        self._put_archive(
            upload_url,
            path,
            archive_size_bytes=archive_size_bytes,
            upload_timeout_seconds=upload_timeout_seconds,
            upload_id=upload_id,
        )
        finalized = self._finalize(upload_id=upload_id, declaration=declaration)
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
            raise RuntimeError(
                f"actor image archive upload failed with HTTP {response.status_code}"
            )

    def _finalize(
        self,
        *,
        upload_id: str,
        declaration: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        try:
            finalized = self._client._request_json(
                "POST",
                "/smr/v1/image-releases/finalize",
                json_body={"upload_id": upload_id, "declaration": dict(declaration)},
            )
        except SmrApiError as exc:
            if exc.status_code is not None:
                raise
            # Finalize is idempotent and proves staging disposal; replay exactly
            # once after response loss.
            finalized = self._client._request_json(
                "POST",
                "/smr/v1/image-releases/finalize",
                json_body={"upload_id": upload_id, "declaration": dict(declaration)},
            )
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
            raise ValueError(
                "image release exists but has no executable runtime image release"
            )
        return _actor_image_from_materialization(
            materialization,
            platform=(
                f"{declaration.get('platform_os')}/"
                f"{declaration.get('platform_architecture')}"
            ),
            archive_sha256=str(declaration.get("archive_sha256") or ""),
        )

    def status(self, *, image_release_id: str) -> str:
        """Return the executable release status for an uploaded actor image."""
        return self.get(image_release_id=image_release_id).status

    def list(self) -> list[dict[str, Any]]:
        """List this org's customer actor runtime image releases."""
        payload = self._client._request_json("GET", "/smr/v1/image-releases")
        if not isinstance(payload, Mapping) or not isinstance(
            payload.get("releases"), list
        ):
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
    "ImagesAPI",
    "inspect_oci_archive",
]
