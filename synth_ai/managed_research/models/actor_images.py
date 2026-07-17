"""Typed per-role actor image bindings for run launches.

A binding names an org-owned runtime image release by its stable release ID.
Run launches carry bindings under ``actor_image_overrides`` keyed by actor
role; the backend freezes the resolved release, digest, and selection source
on the run.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

ACTOR_IMAGE_ROLES: tuple[str, ...] = (
    "orchestrator",
    "worker",
    "verifier",
    "evaluator",
    "judge",
)


@dataclass(frozen=True, slots=True)
class ActorImageBinding:
    release_id: str
    reason: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.release_id, str) or not self.release_id.strip():
            raise ValueError("ActorImageBinding.release_id must be a nonempty string")
        for field_name in ("reason", "notes"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(
                    f"ActorImageBinding.{field_name} must be a nonempty string when set"
                )

    def to_wire(self) -> dict[str, object]:
        payload: dict[str, object] = {"release_id": self.release_id.strip()}
        if self.reason is not None:
            payload["reason"] = self.reason.strip()
        if self.notes is not None:
            payload["notes"] = self.notes.strip()
        return payload

    @classmethod
    def from_wire(cls, payload: ActorImageBindingInput) -> ActorImageBinding:
        if isinstance(payload, ActorImageBinding):
            return payload
        if isinstance(payload, str):
            return cls(release_id=payload)
        if isinstance(payload, Mapping):
            unexpected = set(payload) - {"release_id", "reason", "notes"}
            if unexpected:
                raise ValueError(
                    f"actor image binding has unexpected fields: {sorted(unexpected)}"
                )
            release_id = payload.get("release_id")
            if not isinstance(release_id, str):
                raise ValueError("actor image binding release_id must be a string")
            reason = payload.get("reason")
            notes = payload.get("notes")
            return cls(
                release_id=release_id,
                reason=reason if isinstance(reason, str) or reason is None else None,
                notes=notes if isinstance(notes, str) or notes is None else None,
            )
        raise ValueError(
            "actor image binding must be an ActorImageBinding, release id string, "
            "or mapping"
        )


ActorImageBindingInput: TypeAlias = "ActorImageBinding | str | Mapping[str, object]"
ActorImageBindings: TypeAlias = Mapping[str, ActorImageBindingInput]


def actor_image_overrides_payload(
    overrides: ActorImageBindings | None,
) -> dict[str, dict[str, object]] | None:
    if overrides is None:
        return None
    if not isinstance(overrides, Mapping):
        raise ValueError("actor_image_overrides must map actor roles to bindings")
    payload: dict[str, dict[str, object]] = {}
    for raw_role, raw_binding in overrides.items():
        role = str(raw_role or "").strip().lower()
        if role not in ACTOR_IMAGE_ROLES:
            allowed = ", ".join(ACTOR_IMAGE_ROLES)
            raise ValueError(f"actor_image_overrides role must be one of: {allowed}")
        payload[role] = ActorImageBinding.from_wire(raw_binding).to_wire()
    return payload or None


def image_override_payload(
    override: ActorImageBindingInput | None,
) -> dict[str, object] | None:
    if override is None:
        return None
    return ActorImageBinding.from_wire(override).to_wire()


__all__ = [
    "ACTOR_IMAGE_ROLES",
    "ActorImageBinding",
    "ActorImageBindingInput",
    "ActorImageBindings",
    "actor_image_overrides_payload",
    "image_override_payload",
]
