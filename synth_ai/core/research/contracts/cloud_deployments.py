"""Shared wire types for durable CloudDeployment authority."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Literal, TypedDict, cast

_BACKEND_REPOSITORY = "https://github.com/synth-laboratories/backend.git"
_FULL_GIT_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOPOLOGY_SOURCE_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "repository",
        "implementation_commit_sha",
        "topology_id",
        "topology_version",
        "topology_digest_sha256",
    }
)


class CloudDeploymentTopologySource(TypedDict):
    """Immutable backend implementation authority for a topology definition."""

    schema_version: Literal["cloud-deployment-topology-source-v1"]
    kind: Literal["backend_git"]
    repository: Literal["https://github.com/synth-laboratories/backend.git"]
    implementation_commit_sha: str
    topology_id: str
    topology_version: str
    topology_digest_sha256: str


def _required_text(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"topology_source.{key} must be a nonempty string")
    return value


def cloud_deployment_topology_source_from_wire(
    payload: Mapping[str, object] | object,
) -> CloudDeploymentTopologySource:
    """Parse the exact, immutable topology-source receipt and reject drift."""

    if not isinstance(payload, Mapping):
        raise ValueError("topology_source must be an object")
    actual_fields = frozenset(payload)
    if actual_fields != _TOPOLOGY_SOURCE_FIELDS:
        missing = sorted(_TOPOLOGY_SOURCE_FIELDS - actual_fields)
        unexpected = sorted(actual_fields - _TOPOLOGY_SOURCE_FIELDS)
        raise ValueError(
            "topology_source fields do not match the v1 contract "
            f"(missing={missing}, unexpected={unexpected})"
        )

    schema_version = _required_text(payload, "schema_version")
    if schema_version != "cloud-deployment-topology-source-v1":
        raise ValueError("topology_source.schema_version is unsupported")
    kind = _required_text(payload, "kind")
    if kind != "backend_git":
        raise ValueError("topology_source.kind must be backend_git")
    repository = _required_text(payload, "repository")
    if repository != _BACKEND_REPOSITORY:
        raise ValueError("topology_source.repository is not the backend authority")
    implementation_commit_sha = _required_text(payload, "implementation_commit_sha")
    if not _FULL_GIT_COMMIT_SHA.fullmatch(implementation_commit_sha):
        raise ValueError(
            "topology_source.implementation_commit_sha must be a lowercase full Git SHA"
        )
    topology_digest_sha256 = _required_text(payload, "topology_digest_sha256")
    if not _SHA256.fullmatch(topology_digest_sha256):
        raise ValueError("topology_source.topology_digest_sha256 must be a lowercase SHA-256")

    return cast(
        CloudDeploymentTopologySource,
        {
            "schema_version": schema_version,
            "kind": kind,
            "repository": repository,
            "implementation_commit_sha": implementation_commit_sha,
            "topology_id": _required_text(payload, "topology_id"),
            "topology_version": _required_text(payload, "topology_version"),
            "topology_digest_sha256": topology_digest_sha256,
        },
    )


__all__ = [
    "CloudDeploymentTopologySource",
    "cloud_deployment_topology_source_from_wire",
]
