"""Validate a descriptor against its committed Artifact manifest, without I/O.

See sibling docs/drafts/synth-index-contribution-format-2026-09-12.md §3 and §6.
The caller must obtain the manifest and descriptor from the authenticated Artifact
authority and verify publication status. This check is not a rights/QA approval.
"""

import hashlib
import json

from .artifacts import ArtifactManifest
from .contracts import ContributionReference
from .manifest import encode_manifest
from .package import ContributionPackage

DESCRIPTOR_PATH = "contribution.json"
DESCRIPTOR_BYTES_MAX = 1_048_576


def validate_package_binding(
    descriptor: bytes,
    manifest: ArtifactManifest,
    expected_reference: ContributionReference,
) -> ContributionPackage:
    """Reject mismatched identities, undeclared bytes and ambiguous JSON before sealing."""
    if len(descriptor) > DESCRIPTOR_BYTES_MAX:
        raise ValueError("Contribution descriptor exceeds 1 MiB")
    encode_manifest(manifest)
    if manifest.revision != 1:
        raise ValueError("Contribution revision collections have one Artifact publication slot")
    if manifest.manifest_schema_version != "synth.contribution.v1":
        raise ValueError("Artifact manifest must identify the Contribution v1 schema")
    declarations = {item.logical_path: item for item in manifest.objects}
    declaration = declarations.get(DESCRIPTOR_PATH)
    if declaration is None or declaration.media_type != "application/json":
        raise ValueError("Artifact manifest must contain contribution.json as application/json")
    if declaration.size_bytes != len(descriptor):
        raise ValueError("Contribution descriptor size does not match Artifact manifest")
    if declaration.digest_sha256 != hashlib.sha256(descriptor).hexdigest():
        raise ValueError("Contribution descriptor digest does not match Artifact manifest")
    payload = json.loads(descriptor, object_pairs_hook=_unique_object)
    package = ContributionPackage.model_validate(payload)
    if (
        package.contribution_id != expected_reference.contribution_id
        or package.revision_id != expected_reference.revision_id
    ):
        raise ValueError("Contribution descriptor does not match server-issued revision identity")
    asset_paths = {asset.object.logical_path for asset in package.assets}
    if DESCRIPTOR_PATH in asset_paths:
        raise ValueError("Contribution descriptor cannot recursively declare itself as an asset")
    if set(declarations) != asset_paths | {DESCRIPTOR_PATH}:
        raise ValueError("Artifact manifest and Contribution asset paths must match exactly")
    for asset in package.assets:
        if declarations[asset.object.logical_path] != asset.object:
            raise ValueError("Contribution asset declaration differs from Artifact manifest")
    return package


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("Contribution descriptor contains a duplicate JSON key")
        result[name] = value
    return result
