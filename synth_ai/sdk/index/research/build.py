"""Reconstruct approved output bytes from a closed frozen archive, offline.

See notes/specifications/synth-index/research-archive-release.md. This adapter
uses the existing Artifact manifest and Contribution binding validator. Allocation
identities and the exact descriptor are inputs, never regenerated on retry.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from ..artifacts import ArtifactManifest, ArtifactObjectDeclaration
from ..intake import DESCRIPTOR_PATH, validate_package_binding
from ..manifest import encode_manifest
from ..package import ContributionPackage
from .contracts import (
    DerivationBinding,
    canonical_bytes,
    contract_digest,
)

OBJECT_BYTES_MAX = 64 * 1024 * 1024
ARCHIVE_BYTES_MAX = 1024 * 1024 * 1024


class FrozenBuildError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code


def _read_object(root: Path, declaration: ArtifactObjectDeclaration) -> bytes:
    path = root / declaration.logical_path
    if any(parent.is_symlink() for parent in (path, *path.parents) if parent != root.parent):
        raise FrozenBuildError("unsafe_archive_path", "symlink in frozen object path")
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise FrozenBuildError("unsafe_archive_path", "frozen object escapes archive")
        with resolved.open("rb") as source:
            content = source.read(min(declaration.size_bytes + 1, OBJECT_BYTES_MAX + 1))
    except FileNotFoundError as error:
        raise FrozenBuildError(
            "frozen_object_missing", "declared frozen object is absent"
        ) from error
    if declaration.size_bytes > OBJECT_BYTES_MAX or len(content) > OBJECT_BYTES_MAX:
        raise FrozenBuildError("frozen_object_too_large", "object exceeds bounded builder limit")
    if (
        len(content) != declaration.size_bytes
        or hashlib.sha256(content).hexdigest() != declaration.digest_sha256
    ):
        raise FrozenBuildError("frozen_object_corrupt", "object differs from frozen declaration")
    return content


def verify_archive(root: Path, binding: DerivationBinding) -> dict[str, bytes]:
    """Verify every retained input, including unsuccessful attempts and native exports."""
    root = root.resolve(strict=True)
    declarations = binding.snapshot.objects
    if sum(item.object.size_bytes for item in declarations) > ARCHIVE_BYTES_MAX:
        raise FrozenBuildError("archive_too_large", "archive exceeds 1 GiB builder bound")
    expected = {item.object.logical_path for item in declarations}
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual != expected:
        raise FrozenBuildError(
            "archive_object_set_mismatch",
            "archive must contain exactly the frozen object set",
        )
    return {item.object_id: _read_object(root, item.object) for item in declarations}


def validate_release_binding(
    binding: DerivationBinding,
    descriptor: bytes,
    manifest: ArtifactManifest,
) -> ContributionPackage:
    """Require approved exact descriptor and object set with separate archive identity."""
    package = validate_package_binding(descriptor, manifest, binding.disclosure.reference)
    disclosure = binding.disclosure
    if (manifest.collection_id, manifest.manifest_digest) != (
        disclosure.release_collection_id,
        disclosure.release_manifest_digest_sha256,
    ):
        raise FrozenBuildError(
            "release_binding_mismatch",
            "release manifest differs from approved disclosure",
        )
    if package.requested_audience != disclosure.audience:
        raise FrozenBuildError(
            "release_audience_mismatch", "sealed audience differs from disclosure"
        )
    declared = {item.asset_id: item.object.digest_sha256 for item in package.assets}
    disclosed = {item.asset_id: item.digest_sha256 for item in disclosure.deliverable_assets}
    if declared != disclosed:
        raise FrozenBuildError(
            "release_asset_set_mismatch",
            "descriptor must contain exactly approved deliverable assets",
        )
    outputs = {item.release_asset_id for item in binding.recipe.outputs}
    if outputs != set(disclosed):
        raise FrozenBuildError(
            "recipe_output_set_mismatch", "every output must be explicitly disclosed"
        )
    for representation in disclosure.representations:
        asset = next(item for item in package.assets if item.asset_id == representation.asset_id)
        if representation.kind == "utf8_text" and asset.object.media_type not in (
            "text/plain",
            "text/markdown",
        ):
            raise FrozenBuildError(
                "representation_type_mismatch",
                "text parser requires released UTF-8 text",
            )
    return package


def build_release(
    archive_root: Path,
    destination: Path,
    *,
    binding: DerivationBinding,
    descriptor: bytes,
    manifest: ArtifactManifest,
) -> dict:
    """Write only approved bytes atomically; existing destination is verified on retry.

    This proves artifact reconstruction only. Analysis and experimental reruns need
    independently executed receipts; no uploaded commands are run by this builder.
    """
    binding = DerivationBinding.model_validate_json(canonical_bytes(binding))
    package = validate_release_binding(binding, descriptor, manifest)
    archive_root = archive_root.resolve(strict=True)
    destination = destination.absolute()
    if destination.resolve().is_relative_to(archive_root) or archive_root.is_relative_to(
        destination.resolve()
    ):
        raise FrozenBuildError("build_roots_overlap", "release and archive roots must be disjoint")
    inputs = verify_archive(archive_root, binding)
    output_sources = {
        item.release_asset_id: item.source_object_id for item in binding.recipe.outputs
    }
    for representation in binding.disclosure.representations:
        try:
            inputs[output_sources[representation.asset_id]].decode("utf-8")
        except UnicodeDecodeError as error:
            raise FrozenBuildError(
                "representation_encoding_invalid", "approved text is not UTF-8"
            ) from error
    if destination.exists():
        return verify_release(
            destination, binding=binding, descriptor=descriptor, manifest=manifest
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".research-release-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary) / "result"
        released = staging / "package"
        released.mkdir(parents=True)
        outputs = {
            item.release_asset_id: inputs[item.source_object_id] for item in binding.recipe.outputs
        }
        for asset in package.assets:
            target = released / asset.object.logical_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(outputs[asset.asset_id])
        (released / DESCRIPTOR_PATH).write_bytes(descriptor)
        (staging / "artifact-manifest.json").write_bytes(encode_manifest(manifest))
        # Receipt is deliberately outside the uploadable release package.
        receipt = {
            "schema_version": "synth.research.offline-build-receipt.v1",
            "snapshot_digest_sha256": contract_digest(binding.snapshot),
            "recipe_digest_sha256": contract_digest(binding.recipe),
            "disclosure_digest_sha256": contract_digest(binding.disclosure),
            "manifest_digest_sha256": manifest.manifest_digest,
            "descriptor_digest_sha256": hashlib.sha256(descriptor).hexdigest(),
            "scope": "artifact_reconstruction",
            "provider_calls": 0,
            "cost_usd": 0,
            "publication": "not_granted",
        }
        (staging / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        verify_release(staging, binding=binding, descriptor=descriptor, manifest=manifest)
        staging.rename(destination)
    return receipt


def verify_release(
    destination: Path,
    *,
    binding: DerivationBinding,
    descriptor: bytes,
    manifest: ArtifactManifest,
) -> dict:
    package = validate_release_binding(binding, descriptor, manifest)
    if destination.is_symlink():
        raise FrozenBuildError("unsafe_release_path", "release root may not be a symlink")
    released = destination / "package"
    expected = {item.object.logical_path for item in package.assets} | {DESCRIPTOR_PATH}
    actual = {
        path.relative_to(released).as_posix()
        for path in released.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if expected != actual:
        raise FrozenBuildError(
            "release_object_set_mismatch", "release has missing or undeclared objects"
        )
    for asset in package.assets:
        _read_object(released, asset.object)
    if (released / DESCRIPTOR_PATH).is_symlink() or (
        released / DESCRIPTOR_PATH
    ).read_bytes() != descriptor:
        raise FrozenBuildError("release_descriptor_mismatch", "descriptor bytes changed")
    if (destination / "artifact-manifest.json").read_bytes() != encode_manifest(manifest):
        raise FrozenBuildError("release_manifest_mismatch", "manifest bytes changed")
    receipt = json.loads((destination / "receipt.json").read_bytes())
    expected_receipt = {
        "schema_version": "synth.research.offline-build-receipt.v1",
        "snapshot_digest_sha256": contract_digest(binding.snapshot),
        "recipe_digest_sha256": contract_digest(binding.recipe),
        "disclosure_digest_sha256": contract_digest(binding.disclosure),
        "manifest_digest_sha256": manifest.manifest_digest,
        "descriptor_digest_sha256": hashlib.sha256(descriptor).hexdigest(),
        "scope": "artifact_reconstruction",
        "provider_calls": 0,
        "cost_usd": 0,
        "publication": "not_granted",
    }
    if receipt != expected_receipt:
        raise FrozenBuildError("build_receipt_mismatch", "receipt does not bind this exact rebuild")
    return receipt
