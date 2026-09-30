"""Canonical Artifact Platform manifest encoding and integrity verification.

The digest covers the canonical JSON payload with ``manifest_digest`` omitted.
The stored representation adds that digest and remains canonical JSON. This
avoids a recursive hash while making the claimed identity self-verifying.

See: notes/specifications/tanha/current/systems/platform/artifact_platform.md
"""

from __future__ import annotations

import hashlib
import json

from pydantic import ValidationError

from .artifacts import ArtifactManifest


class ArtifactManifestError(ValueError):
    """Base typed error for invalid canonical manifest bytes."""

    error_code = "artifact_manifest_invalid"


class ArtifactManifestDigestMismatchError(ArtifactManifestError):
    """Claimed or expected manifest digest differs from exact canonical contents."""
    error_code = "artifact_manifest_digest_mismatch"


def encode_manifest(manifest: ArtifactManifest) -> bytes:
    """Encode one verified manifest with stable key and object ordering.

    Args:
        manifest: Typed manifest whose embedded digest must match its canonical contents.
    Returns:
        Canonical UTF-8 JSON bytes containing the verified manifest digest.
    Raises:
        ArtifactManifestDigestMismatchError: Embedded digest differs from canonical contents.
    Examples:
        encoded = encode_manifest(manifest)
    """

    expected_digest = _manifest_digest(manifest)
    if manifest.manifest_digest != expected_digest:
        raise ArtifactManifestDigestMismatchError(
            "artifact manifest claimed digest does not match canonical contents"
        )
    return _canonical_json(manifest.model_dump(mode="json"))


def decode_manifest(
    content: bytes,
    *,
    expected_digest: str | None = None,
) -> ArtifactManifest:
    """Strictly parse canonical bytes and verify embedded and expected digests.

    Args:
        content: Exact canonical UTF-8 JSON manifest bytes.
        expected_digest: Optional expected publication SHA-256 digest to compare.
    Returns:
        Validated ArtifactManifest matching canonical bytes and required digests.
    Raises:
        ArtifactManifestError: JSON, schema or byte canonicalization is invalid.
        ArtifactManifestDigestMismatchError: Embedded or expected digest differs.
    Examples:
        manifest = decode_manifest(content, expected_digest=expected_digest)
    """

    try:
        payload = json.loads(content)
        manifest = ArtifactManifest.model_validate(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as error:
        raise ArtifactManifestError("artifact manifest is not valid canonical JSON") from error
    canonical_content = _canonical_json(manifest.model_dump(mode="json"))
    if canonical_content != content:
        raise ArtifactManifestError("artifact manifest bytes are not canonical")
    calculated_digest = _manifest_digest(manifest)
    if manifest.manifest_digest != calculated_digest:
        raise ArtifactManifestDigestMismatchError(
            "artifact manifest embedded digest does not match canonical contents"
        )
    if expected_digest is not None and calculated_digest != expected_digest:
        raise ArtifactManifestDigestMismatchError(
            "artifact manifest digest differs from publication identity"
        )
    return manifest


def _manifest_digest(manifest: ArtifactManifest) -> str:
    payload = manifest.model_dump(mode="json", exclude={"manifest_digest"})
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


__all__ = [
    "ArtifactManifestDigestMismatchError",
    "ArtifactManifestError",
    "decode_manifest",
    "encode_manifest",
]
