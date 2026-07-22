"""Strict Environment catalog response contracts.

# See: specifications/sdk/core_research_migration.md
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.core.research.contracts._environment_wire import (
    digest,
    exact_object,
    optional_text,
    text,
)
from synth_ai.core.research.contracts._wire import required_bool, required_datetime
from synth_ai.core.research.contracts.common import (
    EnvironmentDigest,
    EnvironmentId,
    EnvironmentName,
    OrganizationId,
    UserId,
)
from synth_ai.core.research.contracts.environment_manifest import (
    EnvironmentJsonObject,
    EnvironmentManifest,
)

_SUMMARY_FIELDS = frozenset(
    {
        "environment_id",
        "name",
        "digest",
        "manifest_digest",
        "org_id",
        "created_by_user_id",
        "spec",
        "created_at",
    }
)


def _optional_identity(value: object, *, field_name: str) -> str | None:
    return optional_text(value, field=field_name, maximum=4000)


@dataclass(frozen=True, slots=True)
class Environment:
    """One immutable Environment catalog summary."""

    environment_id: EnvironmentId
    name: EnvironmentName
    digest: EnvironmentDigest
    manifest_digest: EnvironmentDigest
    organization_id: OrganizationId | None
    created_by_user_id: UserId | None
    spec: EnvironmentJsonObject
    created_at: datetime

    def __post_init__(self) -> None:
        environment_id = text(self.environment_id, field="environment_id", maximum=4000)
        name = text(self.name, field="environment name", maximum=255)
        environment_digest = digest(self.digest, field="environment digest")
        manifest_digest = digest(self.manifest_digest, field="manifest_digest")
        if environment_digest != manifest_digest:
            raise ValueError("environment digest and manifest_digest must match")
        organization_id = _optional_identity(self.organization_id, field_name="org_id")
        user_id = _optional_identity(
            self.created_by_user_id,
            field_name="created_by_user_id",
        )
        if not isinstance(self.spec, EnvironmentJsonObject):
            raise ValueError("environment spec must be EnvironmentJsonObject")
        if self.created_at.tzinfo is None:
            raise ValueError("environment created_at must include a timezone")
        object.__setattr__(self, "environment_id", EnvironmentId(environment_id))
        object.__setattr__(self, "name", EnvironmentName(name))
        object.__setattr__(self, "digest", EnvironmentDigest(environment_digest))
        object.__setattr__(self, "manifest_digest", EnvironmentDigest(manifest_digest))
        object.__setattr__(
            self,
            "organization_id",
            OrganizationId(organization_id) if organization_id is not None else None,
        )
        object.__setattr__(
            self,
            "created_by_user_id",
            UserId(user_id) if user_id is not None else None,
        )

    @classmethod
    def from_wire(cls, value: JsonValue) -> Environment:
        payload = exact_object(value, label="environment", fields=_SUMMARY_FIELDS)
        return _environment_from_payload(payload)

    def to_wire(self) -> JsonObject:
        return {
            "environment_id": self.environment_id,
            "name": self.name,
            "digest": self.digest,
            "manifest_digest": self.manifest_digest,
            "org_id": self.organization_id,
            "created_by_user_id": self.created_by_user_id,
            "spec": self.spec.to_wire(),
            "created_at": self.created_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class EnvironmentDetail(Environment):
    """Catalog summary plus the resolved, digest-bound manifest."""

    manifest: EnvironmentManifest

    def __post_init__(self) -> None:
        Environment.__post_init__(self)
        if not isinstance(self.manifest, EnvironmentManifest):
            raise ValueError("environment manifest must be EnvironmentManifest")
        if self.manifest.name != self.name:
            raise ValueError("environment manifest name drifted from catalog identity")
        if self.manifest.digest != self.manifest_digest:
            raise ValueError("environment manifest body drifted from catalog digest")

    @classmethod
    def from_wire(cls, value: JsonValue) -> EnvironmentDetail:
        payload = exact_object(
            value,
            label="environment detail",
            fields=_SUMMARY_FIELDS | {"manifest"},
        )
        summary = _environment_from_payload(payload)
        return cls(
            environment_id=summary.environment_id,
            name=summary.name,
            digest=summary.digest,
            manifest_digest=summary.manifest_digest,
            organization_id=summary.organization_id,
            created_by_user_id=summary.created_by_user_id,
            spec=summary.spec,
            created_at=summary.created_at,
            manifest=EnvironmentManifest.from_wire(payload["manifest"]),
        )

    def to_wire(self) -> JsonObject:
        payload = Environment.to_wire(self)
        payload["manifest"] = self.manifest.to_wire()
        return payload


def _environment_from_payload(payload: JsonObject) -> Environment:
    organization_id = _optional_identity(payload["org_id"], field_name="org_id")
    created_by = _optional_identity(
        payload["created_by_user_id"],
        field_name="created_by_user_id",
    )
    return Environment(
        environment_id=EnvironmentId(
            text(payload["environment_id"], field="environment_id", maximum=4000)
        ),
        name=EnvironmentName(text(payload["name"], field="environment name", maximum=255)),
        digest=EnvironmentDigest(digest(payload["digest"], field="environment digest")),
        manifest_digest=EnvironmentDigest(
            digest(payload["manifest_digest"], field="manifest_digest")
        ),
        organization_id=(OrganizationId(organization_id) if organization_id is not None else None),
        created_by_user_id=UserId(created_by) if created_by is not None else None,
        spec=EnvironmentJsonObject.from_wire(payload["spec"]),
        created_at=required_datetime(payload, "created_at"),
    )


class EnvironmentPreflightResult(StrEnum):
    FAILED = "failed"
    OK = "ok"
    SKIPPED = "skipped"


class EnvironmentResponsibleParty(StrEnum):
    INFRA = "infra"
    PROJECT = "project"
    USER = "user"


@dataclass(frozen=True, slots=True)
class EnvironmentPreflightError:
    error_code: str
    responsible_party: EnvironmentResponsibleParty
    message: str
    details: EnvironmentJsonObject

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "error_code",
            text(self.error_code, field="preflight error_code", maximum=4000),
        )
        if not isinstance(self.responsible_party, EnvironmentResponsibleParty):
            raise ValueError("responsible_party must be EnvironmentResponsibleParty")
        object.__setattr__(
            self,
            "message",
            text(self.message, field="preflight error message", maximum=20_000),
        )
        if not isinstance(self.details, EnvironmentJsonObject):
            raise ValueError("preflight error details must be EnvironmentJsonObject")

    @classmethod
    def from_wire(cls, value: JsonValue) -> EnvironmentPreflightError:
        payload = exact_object(
            value,
            label="environment preflight error",
            fields=frozenset({"error_code", "responsible_party", "message", "details"}),
        )
        return cls(
            error_code=text(payload["error_code"], field="preflight error_code", maximum=4000),
            responsible_party=EnvironmentResponsibleParty(
                text(payload["responsible_party"], field="responsible_party", maximum=64)
            ),
            message=text(payload["message"], field="preflight error message", maximum=20_000),
            details=EnvironmentJsonObject.from_wire(payload["details"]),
        )

    def to_wire(self) -> JsonObject:
        return {
            "error_code": self.error_code,
            "responsible_party": self.responsible_party.value,
            "message": self.message,
            "details": self.details.to_wire(),
        }


@dataclass(frozen=True, slots=True)
class EnvironmentPreflight:
    name: EnvironmentName
    digest: EnvironmentDigest
    manifest_digest: EnvironmentDigest
    ok: bool
    result: EnvironmentPreflightResult
    details: EnvironmentJsonObject
    error: EnvironmentPreflightError | None

    def __post_init__(self) -> None:
        name = text(self.name, field="environment name", maximum=255)
        environment_digest = digest(self.digest, field="environment digest")
        manifest_digest = digest(self.manifest_digest, field="manifest_digest")
        if environment_digest != manifest_digest:
            raise ValueError("preflight digest and manifest_digest must match")
        if not isinstance(self.ok, bool):
            raise ValueError("preflight ok must be a boolean")
        if not isinstance(self.result, EnvironmentPreflightResult):
            raise ValueError("preflight result must be EnvironmentPreflightResult")
        if self.ok != (self.result is not EnvironmentPreflightResult.FAILED):
            raise ValueError("preflight ok and result disagree")
        if not isinstance(self.details, EnvironmentJsonObject):
            raise ValueError("preflight details must be EnvironmentJsonObject")
        if self.error is not None and not isinstance(self.error, EnvironmentPreflightError):
            raise ValueError("preflight error must be EnvironmentPreflightError")
        if self.error is not None and self.result is not EnvironmentPreflightResult.FAILED:
            raise ValueError("successful preflight cannot carry an error")
        object.__setattr__(self, "name", EnvironmentName(name))
        object.__setattr__(self, "digest", EnvironmentDigest(environment_digest))
        object.__setattr__(self, "manifest_digest", EnvironmentDigest(manifest_digest))

    @classmethod
    def from_wire(cls, value: JsonValue) -> EnvironmentPreflight:
        payload = exact_object(
            value,
            label="environment preflight",
            fields=frozenset(
                {"name", "digest", "manifest_digest", "ok", "result", "details", "error"}
            ),
        )
        error_value = payload["error"]
        return cls(
            name=EnvironmentName(text(payload["name"], field="environment name", maximum=255)),
            digest=EnvironmentDigest(digest(payload["digest"], field="environment digest")),
            manifest_digest=EnvironmentDigest(
                digest(payload["manifest_digest"], field="manifest_digest")
            ),
            ok=required_bool(payload, "ok"),
            result=EnvironmentPreflightResult(
                text(payload["result"], field="preflight result", maximum=64)
            ),
            details=EnvironmentJsonObject.from_wire(payload["details"]),
            error=(
                None if error_value is None else EnvironmentPreflightError.from_wire(error_value)
            ),
        )

    def to_wire(self) -> JsonObject:
        return {
            "name": self.name,
            "digest": self.digest,
            "manifest_digest": self.manifest_digest,
            "ok": self.ok,
            "result": self.result.value,
            "details": self.details.to_wire(),
            "error": self.error.to_wire() if self.error is not None else None,
        }


__all__ = [
    "Environment",
    "EnvironmentDetail",
    "EnvironmentPreflight",
    "EnvironmentPreflightError",
    "EnvironmentPreflightResult",
    "EnvironmentResponsibleParty",
]
