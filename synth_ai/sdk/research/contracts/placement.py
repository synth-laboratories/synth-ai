"""Typed run environment catalog and actor placement configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum


class PlacementMode(StrEnum):
    PER_ACTOR = "per_actor"
    SHARED_GROUP = "shared_group"


MAX_SHARED_GROUP_ACTORS = 64


@dataclass(frozen=True, slots=True)
class SharedPlacementGroup:
    environment_release_id: str
    workspace_mode: str = "separate_worktrees"
    max_actors: int = 4
    max_heavy_commands: int = 2
    trust_policy: str = "same_trust_boundary"

    def __post_init__(self) -> None:
        if not self.environment_release_id.strip():
            raise ValueError("environment_release_id is required")
        if self.workspace_mode != "separate_worktrees":
            raise ValueError("only separate_worktrees is currently supported")
        if self.max_actors < 1 or self.max_heavy_commands < 1:
            raise ValueError("placement limits must be positive")
        if self.max_actors > MAX_SHARED_GROUP_ACTORS:
            raise ValueError(
                f"shared placement groups support at most {MAX_SHARED_GROUP_ACTORS} actors"
            )

    def to_wire(self) -> dict[str, object]:
        return {
            "mode": PlacementMode.SHARED_GROUP.value,
            "environment": self.environment_release_id,
            "workspace_mode": self.workspace_mode,
            "max_actors": self.max_actors,
            "max_heavy_commands": self.max_heavy_commands,
            "trust_policy": self.trust_policy,
        }


@dataclass(frozen=True, slots=True)
class PlacementPolicy:
    default: PlacementMode = PlacementMode.PER_ACTOR
    groups: Mapping[str, SharedPlacementGroup] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "default", PlacementMode(self.default))
        grouped_environments: set[str] = set()
        for name, group in self.groups.items():
            if not str(name).strip() or not isinstance(group, SharedPlacementGroup):
                raise ValueError("placement groups require nonempty names and typed specs")
            if group.environment_release_id in grouped_environments:
                raise ValueError("an environment may belong to only one shared placement group")
            grouped_environments.add(group.environment_release_id)

    def to_wire(self) -> dict[str, object]:
        return {
            "default": self.default.value,
            "groups": {str(name): group.to_wire() for name, group in self.groups.items()},
        }


@dataclass(frozen=True, slots=True)
class RunEnvironmentCatalog:
    allowed: tuple[str, ...]
    role_defaults: Mapping[str, str] = field(default_factory=dict)
    task_bindings: Mapping[str, str] = field(default_factory=dict)
    actor_selections: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_allowed = tuple(
            dict.fromkeys(str(item).strip() for item in self.allowed if str(item).strip())
        )
        if not normalized_allowed:
            raise ValueError("environment catalog requires at least one allowed release")
        object.__setattr__(self, "allowed", normalized_allowed)
        allowed = set(normalized_allowed)
        for mapping_name, values in (
            ("role_defaults", self.role_defaults),
            ("task_bindings", self.task_bindings),
            ("actor_selections", self.actor_selections),
        ):
            for key, release_id in values.items():
                if not str(key).strip() or str(release_id).strip() not in allowed:
                    raise ValueError(
                        f"{mapping_name} must reference an allowed environment release"
                    )

    def to_wire(self) -> dict[str, object]:
        return {
            "allowed": list(self.allowed),
            "role_defaults": dict(self.role_defaults),
            "task_bindings": dict(self.task_bindings),
            "actor_selections": dict(self.actor_selections),
        }


__all__ = [
    "MAX_SHARED_GROUP_ACTORS",
    "PlacementMode",
    "PlacementPolicy",
    "RunEnvironmentCatalog",
    "SharedPlacementGroup",
]
