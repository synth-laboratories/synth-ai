"""Immutable Environment manifest contracts.

# See: specifications/sdk/core_research_migration.md
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.core.research.contracts._environment_wire import (
    ENVIRONMENT_SCHEMA_VERSION,
    IMAGE_REF_PATTERN,
    SHA256_PATTERN,
    FrozenJson,
    JsonInput,
    boolean,
    exact_object,
    freeze_json,
    input_object,
    integer,
    optional_digest,
    optional_text,
    text,
    thaw_json,
)
from synth_ai.core.research.contracts.common import EnvironmentDigest, EnvironmentName


def _payload(
    value: JsonValue,
    *,
    label: str,
    fields: frozenset[str],
    required: frozenset[str],
    exact: bool,
) -> JsonObject:
    if exact:
        return exact_object(value, label=label, fields=fields)
    return input_object(value, label=label, allowed=fields, required=required)


def _objects(value: object, *, field_name: str) -> tuple[JsonObject, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{field_name} must contain only objects")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class EnvironmentJsonObject:
    """Recursively immutable backend-declared JSON extension island."""

    values: Mapping[str, JsonInput] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frozen = freeze_json(self.values, field="environment JSON object")
        if not isinstance(frozen, Mapping):
            raise ValueError("environment JSON value must be an object")
        object.__setattr__(self, "values", frozen)

    @classmethod
    def from_wire(cls, value: JsonValue) -> EnvironmentJsonObject:
        if not isinstance(value, dict):
            raise ValueError("environment JSON value must be an object")
        return cls(value)

    def to_wire(self) -> JsonObject:
        thawed = thaw_json(freeze_json(self.values, field="environment JSON object"))
        if not isinstance(thawed, dict):
            raise ValueError("environment JSON value must be an object")
        return thawed


class RuntimeImageKind(StrEnum):
    LOCAL_DOCKER_IMAGE = "local_docker_image"
    REGISTRY_IMAGE = "registry_image"
    DAYTONA_SNAPSHOT = "daytona_snapshot"


@dataclass(frozen=True, slots=True)
class RuntimeImageRef:
    ref: str
    kind: RuntimeImageKind = RuntimeImageKind.REGISTRY_IMAGE
    digest: EnvironmentDigest | None = None

    def __post_init__(self) -> None:
        reference = text(self.ref, field="image ref", maximum=4000)
        if (
            SHA256_PATTERN.fullmatch(reference) is None
            and IMAGE_REF_PATTERN.fullmatch(reference) is None
        ):
            raise ValueError("image ref must be a digest or Docker image reference")
        if not isinstance(self.kind, RuntimeImageKind):
            raise ValueError("image kind must be a RuntimeImageKind")
        image_digest = optional_digest(self.digest, field="image digest")
        object.__setattr__(self, "ref", reference)
        object.__setattr__(
            self,
            "digest",
            EnvironmentDigest(image_digest) if image_digest is not None else None,
        )

    @property
    def resolved_ref(self) -> str:
        return self.digest or self.ref

    @classmethod
    def from_wire(cls, value: JsonValue, *, exact: bool = True) -> RuntimeImageRef:
        fields = frozenset({"kind", "ref", "digest"})
        payload = _payload(
            value,
            label="runtime image",
            fields=fields,
            required=frozenset({"ref"}),
            exact=exact,
        )
        return cls(
            ref=text(payload["ref"], field="image ref", maximum=4000),
            kind=RuntimeImageKind(payload.get("kind", RuntimeImageKind.REGISTRY_IMAGE)),
            digest=optional_digest(payload.get("digest"), field="image digest"),
        )

    def to_wire(self) -> JsonObject:
        return {"kind": self.kind.value, "ref": self.ref, "digest": self.digest}


@dataclass(frozen=True, slots=True)
class EnvironmentSecretSpec:
    label: str
    kind: str | None = None
    provider: str | None = None
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", text(self.label, field="secret label", maximum=255))
        object.__setattr__(self, "kind", optional_text(self.kind, field="secret kind", maximum=128))
        object.__setattr__(
            self,
            "provider",
            optional_text(self.provider, field="secret provider", maximum=128),
        )
        boolean(self.required, field="secret required")

    @classmethod
    def from_wire(cls, value: JsonValue, *, exact: bool = True) -> EnvironmentSecretSpec:
        fields = frozenset({"label", "kind", "provider", "required"})
        payload = _payload(
            value,
            label="environment secret",
            fields=fields,
            required=frozenset({"label"}),
            exact=exact,
        )
        return cls(
            label=text(payload["label"], field="secret label", maximum=255),
            kind=optional_text(payload.get("kind"), field="secret kind", maximum=128),
            provider=optional_text(
                payload.get("provider"),
                field="secret provider",
                maximum=128,
            ),
            required=boolean(payload.get("required", True), field="secret required"),
        )

    def to_wire(self) -> JsonObject:
        return {
            "label": self.label,
            "kind": self.kind,
            "provider": self.provider,
            "required": self.required,
        }


@dataclass(frozen=True, slots=True)
class EnvironmentRepositorySpec:
    name: str
    url: str | None = None
    role: str = "dependency"
    branch: str | None = None
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", text(self.name, field="repository name", maximum=255))
        object.__setattr__(
            self, "url", optional_text(self.url, field="repository url", maximum=4000)
        )
        object.__setattr__(self, "role", text(self.role, field="repository role", maximum=128))
        object.__setattr__(
            self,
            "branch",
            optional_text(self.branch, field="repository branch", maximum=255),
        )
        boolean(self.required, field="repository required")

    @classmethod
    def from_wire(
        cls,
        value: JsonValue,
        *,
        exact: bool = True,
    ) -> EnvironmentRepositorySpec:
        fields = frozenset({"name", "url", "role", "branch", "required"})
        payload = _payload(
            value,
            label="environment repository",
            fields=fields,
            required=frozenset({"name"}),
            exact=exact,
        )
        return cls(
            name=text(payload["name"], field="repository name", maximum=255),
            url=optional_text(payload.get("url"), field="repository url", maximum=4000),
            role=text(payload.get("role", "dependency"), field="repository role", maximum=128),
            branch=optional_text(
                payload.get("branch"),
                field="repository branch",
                maximum=255,
            ),
            required=boolean(payload.get("required", True), field="repository required"),
        )

    def to_wire(self) -> JsonObject:
        return {
            "name": self.name,
            "url": self.url,
            "role": self.role,
            "branch": self.branch,
            "required": self.required,
        }


@dataclass(frozen=True, slots=True)
class EnvironmentMountSpec:
    path: str
    source: str | None = None
    required: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", text(self.path, field="mount path", maximum=4000))
        object.__setattr__(
            self,
            "source",
            optional_text(self.source, field="mount source", maximum=4000),
        )
        boolean(self.required, field="mount required")

    @classmethod
    def from_wire(cls, value: JsonValue, *, exact: bool = True) -> EnvironmentMountSpec:
        fields = frozenset({"path", "source", "required"})
        payload = _payload(
            value,
            label="environment mount",
            fields=fields,
            required=frozenset({"path"}),
            exact=exact,
        )
        return cls(
            path=text(payload["path"], field="mount path", maximum=4000),
            source=optional_text(payload.get("source"), field="mount source", maximum=4000),
            required=boolean(payload.get("required", True), field="mount required"),
        )

    def to_wire(self) -> JsonObject:
        return {"path": self.path, "source": self.source, "required": self.required}


@dataclass(frozen=True, slots=True)
class EnvironmentIsolationSpec:
    egress_allowlist: tuple[str, ...] = ()
    network_mode: str | None = None

    def __post_init__(self) -> None:
        allowlist = tuple(
            text(item, field="egress allowlist item", maximum=4000)
            for item in self.egress_allowlist
        )
        object.__setattr__(self, "egress_allowlist", allowlist)
        object.__setattr__(
            self,
            "network_mode",
            optional_text(self.network_mode, field="network mode", maximum=128),
        )

    @classmethod
    def from_wire(
        cls,
        value: JsonValue,
        *,
        exact: bool = True,
    ) -> EnvironmentIsolationSpec:
        fields = frozenset({"egress_allowlist", "network_mode"})
        payload = _payload(
            value,
            label="environment isolation",
            fields=fields,
            required=frozenset(),
            exact=exact,
        )
        raw = payload.get("egress_allowlist", [])
        if not isinstance(raw, list):
            raise ValueError("egress_allowlist must be an array")
        return cls(
            tuple(text(item, field="egress allowlist item", maximum=4000) for item in raw),
            optional_text(payload.get("network_mode"), field="network mode", maximum=128),
        )

    def to_wire(self) -> JsonObject:
        return {"egress_allowlist": list(self.egress_allowlist), "network_mode": self.network_mode}


@dataclass(frozen=True, slots=True)
class EnvironmentPreflightSpec:
    command: str | None = None
    timeout_seconds: int = 60

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "command",
            optional_text(self.command, field="preflight command", maximum=20_000),
        )
        integer(self.timeout_seconds, field="preflight timeout", minimum=1, maximum=1800)

    @classmethod
    def from_wire(
        cls,
        value: JsonValue,
        *,
        exact: bool = True,
    ) -> EnvironmentPreflightSpec:
        fields = frozenset({"cmd", "timeout_seconds"})
        payload = _payload(
            value,
            label="environment preflight",
            fields=fields,
            required=frozenset(),
            exact=exact,
        )
        return cls(
            optional_text(payload.get("cmd"), field="preflight command", maximum=20_000),
            integer(
                payload.get("timeout_seconds", 60),
                field="preflight timeout",
                minimum=1,
                maximum=1800,
            ),
        )

    def to_wire(self) -> JsonObject:
        return {"cmd": self.command, "timeout_seconds": self.timeout_seconds}


@dataclass(frozen=True, slots=True)
class EnvironmentManifest:
    """Versioned, content-addressed runtime declaration."""

    name: EnvironmentName
    images: tuple[RuntimeImageRef, ...]
    schema_version: str = ENVIRONMENT_SCHEMA_VERSION
    digest: EnvironmentDigest | None = None
    secrets: tuple[EnvironmentSecretSpec, ...] = ()
    repositories: tuple[EnvironmentRepositorySpec, ...] = ()
    mounts: tuple[EnvironmentMountSpec, ...] = ()
    isolation: EnvironmentIsolationSpec = field(default_factory=EnvironmentIsolationSpec)
    preflight: EnvironmentPreflightSpec = field(default_factory=EnvironmentPreflightSpec)
    metadata: EnvironmentJsonObject = field(default_factory=EnvironmentJsonObject)

    def __post_init__(self) -> None:
        name = text(self.name, field="environment name", maximum=255)
        if self.schema_version != ENVIRONMENT_SCHEMA_VERSION:
            raise ValueError(f"unsupported environment schema_version: {self.schema_version!r}")
        images, secrets = tuple(self.images), tuple(self.secrets)
        repositories, mounts = tuple(self.repositories), tuple(self.mounts)
        typed_collections = (
            (images, RuntimeImageRef, "images"),
            (secrets, EnvironmentSecretSpec, "secrets"),
            (repositories, EnvironmentRepositorySpec, "repositories"),
            (mounts, EnvironmentMountSpec, "mounts"),
        )
        if not images:
            raise ValueError("environment manifest requires at least one image")
        for values, value_type, label in typed_collections:
            if not all(isinstance(item, value_type) for item in values):
                raise ValueError(f"environment {label} must contain typed declarations")
        if not isinstance(self.isolation, EnvironmentIsolationSpec):
            raise ValueError("environment isolation must be EnvironmentIsolationSpec")
        if not isinstance(self.preflight, EnvironmentPreflightSpec):
            raise ValueError("environment preflight must be EnvironmentPreflightSpec")
        if not isinstance(self.metadata, EnvironmentJsonObject):
            raise ValueError("environment metadata must be EnvironmentJsonObject")
        manifest_digest = optional_digest(self.digest, field="environment digest")
        object.__setattr__(self, "name", EnvironmentName(name))
        object.__setattr__(self, "images", images)
        object.__setattr__(self, "secrets", secrets)
        object.__setattr__(self, "repositories", repositories)
        object.__setattr__(self, "mounts", mounts)
        object.__setattr__(
            self,
            "digest",
            EnvironmentDigest(manifest_digest) if manifest_digest is not None else None,
        )

    @classmethod
    def from_input(cls, value: JsonValue) -> EnvironmentManifest:
        return cls._decode(value, exact=False)

    @classmethod
    def from_wire(cls, value: JsonValue) -> EnvironmentManifest:
        return cls._decode(value, exact=True)

    @classmethod
    def _decode(cls, value: JsonValue, *, exact: bool) -> EnvironmentManifest:
        fields = frozenset(
            {
                "schema_version",
                "name",
                "digest",
                "images",
                "secrets",
                "repos",
                "mounts",
                "isolation",
                "preflight",
                "metadata",
            }
        )
        payload = _payload(
            value,
            label="environment manifest",
            fields=fields,
            required=frozenset({"name", "images"}),
            exact=exact,
        )
        image_values = _objects(payload.get("images", []), field_name="images")
        secret_values = _objects(payload.get("secrets", []), field_name="secrets")
        repository_values = _objects(payload.get("repos", []), field_name="repos")
        mount_values = _objects(payload.get("mounts", []), field_name="mounts")
        manifest_digest = optional_digest(payload.get("digest"), field="environment digest")
        return cls(
            name=EnvironmentName(text(payload["name"], field="environment name", maximum=255)),
            images=tuple(RuntimeImageRef.from_wire(item, exact=exact) for item in image_values),
            schema_version=text(
                payload.get("schema_version", ENVIRONMENT_SCHEMA_VERSION),
                field="environment schema_version",
                maximum=255,
            ),
            digest=EnvironmentDigest(manifest_digest) if manifest_digest is not None else None,
            secrets=tuple(
                EnvironmentSecretSpec.from_wire(item, exact=exact) for item in secret_values
            ),
            repositories=tuple(
                EnvironmentRepositorySpec.from_wire(item, exact=exact) for item in repository_values
            ),
            mounts=tuple(
                EnvironmentMountSpec.from_wire(item, exact=exact) for item in mount_values
            ),
            isolation=EnvironmentIsolationSpec.from_wire(payload.get("isolation", {}), exact=exact),
            preflight=EnvironmentPreflightSpec.from_wire(payload.get("preflight", {}), exact=exact),
            metadata=EnvironmentJsonObject.from_wire(payload.get("metadata", {})),
        )

    def to_wire(self) -> JsonObject:
        payload: JsonObject = {
            "schema_version": self.schema_version,
            "name": self.name,
            "images": [item.to_wire() for item in self.images],
            "secrets": [item.to_wire() for item in self.secrets],
            "repos": [item.to_wire() for item in self.repositories],
            "mounts": [item.to_wire() for item in self.mounts],
            "isolation": self.isolation.to_wire(),
            "preflight": self.preflight.to_wire(),
            "metadata": self.metadata.to_wire(),
        }
        if self.digest is not None:
            payload["digest"] = self.digest
        return payload


__all__ = [
    "ENVIRONMENT_SCHEMA_VERSION",
    "EnvironmentIsolationSpec",
    "EnvironmentJsonObject",
    "EnvironmentManifest",
    "EnvironmentMountSpec",
    "EnvironmentPreflightSpec",
    "EnvironmentRepositorySpec",
    "EnvironmentSecretSpec",
    "FrozenJson",
    "RuntimeImageKind",
    "RuntimeImageRef",
]
