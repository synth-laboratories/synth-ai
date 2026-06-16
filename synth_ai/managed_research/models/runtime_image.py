"""Typed runtime image selection for Managed Research launches.

# See: Jstack/.jstack/daily_notes/2026-06-12/managed_research_sdk_dataclass_surface_2026-06-12.md
# See: backend/packages/smr/environments/models.py (ImageRecipe authority)
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Self

IMAGE_RECIPE_SCHEMA_VERSION = "2026-05-14-image-recipe-v1"
_IMAGE_REF_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._:/@+-]*$")


class RuntimeImageError(ValueError):
    """Raised when a runtime image request is invalid or not supported."""


class ActorImageId(StrEnum):
    DEFAULT = "default"
    OPEN_RESEARCH = "open_research"
    OPEN_RESEARCH_CRAFTER = "open_research_crafter"
    OPEN_RESEARCH_CRAFTAX = "open_research_craftax"
    OPEN_RESEARCH_NETHACK = "open_research_nethack"
    OPEN_RESEARCH_DUNGEONBENCH = "open_research_dungeonbench"


class RuntimePackageId(StrEnum):
    CRAFTER = "crafter"
    NUMPY = "numpy"
    IMAGEIO = "imageio"
    IMAGEIO_FFMPEG = "imageio-ffmpeg"
    PILLOW = "pillow"
    PYYAML = "pyyaml"
    CRAFTAX = "craftax"
    JAX_CPU = "jax_cpu"
    CHEX = "chex"
    DISTRAX = "distrax"
    FLAX = "flax"
    GYMNAX = "gymnax"
    MODAL = "modal"
    OPTAX = "optax"
    ORBAX_CHECKPOINT = "orbax_checkpoint"
    NLE = "nle"
    GYMNASIUM = "gymnasium"


class RuntimeImageKind(StrEnum):
    CATALOG = "catalog"
    RECIPE = "recipe"
    REF = "ref"


# Backend actor-runtime pins. SDK mirrors backend authority; do not add packages here
# without a matching backend build target and allowlist review.
RUNTIME_PACKAGE_PINS: dict[RuntimePackageId, str] = {
    RuntimePackageId.CRAFTER: "crafter==1.8.3",
    RuntimePackageId.NUMPY: "numpy",
    RuntimePackageId.IMAGEIO: "imageio",
    RuntimePackageId.IMAGEIO_FFMPEG: "imageio-ffmpeg",
    RuntimePackageId.PILLOW: "pillow",
    RuntimePackageId.PYYAML: "pyyaml",
    RuntimePackageId.CRAFTAX: "craftax",
    RuntimePackageId.JAX_CPU: "jax[cpu]>=0.5.0",
    RuntimePackageId.CHEX: "chex",
    RuntimePackageId.DISTRAX: "distrax",
    RuntimePackageId.FLAX: "flax",
    RuntimePackageId.GYMNAX: "gymnax",
    RuntimePackageId.MODAL: "modal",
    RuntimePackageId.OPTAX: "optax",
    RuntimePackageId.ORBAX_CHECKPOINT: "orbax-checkpoint==0.5.0",
    RuntimePackageId.NLE: "nle",
    RuntimePackageId.GYMNASIUM: "gymnasium",
}


@dataclass(frozen=True, slots=True)
class ActorImageCatalogEntry:
    image_id: ActorImageId
    runtime_image_ref: str
    dependency_set_id: str
    label: str
    environment_name: str | None = None
    extends: ActorImageId | None = None
    packages: frozenset[RuntimePackageId] = frozenset()


ACTOR_IMAGE_CATALOG: dict[ActorImageId, ActorImageCatalogEntry] = {
    ActorImageId.DEFAULT: ActorImageCatalogEntry(
        image_id=ActorImageId.DEFAULT,
        runtime_image_ref="synth-local-smr-runtime:latest",
        dependency_set_id="default_2026_05",
        label="Default SMR actor runtime",
    ),
    ActorImageId.OPEN_RESEARCH: ActorImageCatalogEntry(
        image_id=ActorImageId.OPEN_RESEARCH,
        runtime_image_ref="synth-local-smr-runtime:latest",
        dependency_set_id="open_research_2026_05",
        label="Open Research actor runtime",
        extends=ActorImageId.DEFAULT,
        packages=frozenset(
            {
                RuntimePackageId.IMAGEIO,
                RuntimePackageId.IMAGEIO_FFMPEG,
                RuntimePackageId.PILLOW,
                RuntimePackageId.PYYAML,
            }
        ),
    ),
    ActorImageId.OPEN_RESEARCH_CRAFTER: ActorImageCatalogEntry(
        image_id=ActorImageId.OPEN_RESEARCH_CRAFTER,
        runtime_image_ref="synth-local-open-research-crafter:latest",
        dependency_set_id="open_research_crafter_2026_05",
        label="Open Research Crafter actor runtime",
        environment_name="symbolic-crafter-py311",
        extends=ActorImageId.OPEN_RESEARCH,
        packages=frozenset(
            {
                RuntimePackageId.CRAFTER,
                RuntimePackageId.NUMPY,
                RuntimePackageId.IMAGEIO,
            }
        ),
    ),
    ActorImageId.OPEN_RESEARCH_CRAFTAX: ActorImageCatalogEntry(
        image_id=ActorImageId.OPEN_RESEARCH_CRAFTAX,
        runtime_image_ref="synth-local-open-research-craftax:latest",
        dependency_set_id="open_research_craftax_2026_05",
        label="Open Research Craftax actor runtime",
        environment_name="symbolic-craftax-py311",
        extends=ActorImageId.OPEN_RESEARCH,
        packages=frozenset(
            {
                RuntimePackageId.JAX_CPU,
                RuntimePackageId.CHEX,
                RuntimePackageId.CRAFTAX,
                RuntimePackageId.DISTRAX,
                RuntimePackageId.FLAX,
                RuntimePackageId.GYMNAX,
                RuntimePackageId.MODAL,
                RuntimePackageId.OPTAX,
                RuntimePackageId.ORBAX_CHECKPOINT,
            }
        ),
    ),
    ActorImageId.OPEN_RESEARCH_NETHACK: ActorImageCatalogEntry(
        image_id=ActorImageId.OPEN_RESEARCH_NETHACK,
        runtime_image_ref="synth-local-open-research-nethack:latest",
        dependency_set_id="open_research_nethack_2026_05",
        label="Open Research NetHack actor runtime",
        environment_name="symbolic-nethack-py311",
        extends=ActorImageId.OPEN_RESEARCH,
        packages=frozenset({RuntimePackageId.GYMNASIUM, RuntimePackageId.NLE}),
    ),
    ActorImageId.OPEN_RESEARCH_DUNGEONBENCH: ActorImageCatalogEntry(
        image_id=ActorImageId.OPEN_RESEARCH_DUNGEONBENCH,
        runtime_image_ref="synth-local-open-research-dungeongrid:latest",
        dependency_set_id="open_research_dungeonbench_2026_05",
        label="Open Research DungeonGrid actor runtime",
        extends=ActorImageId.OPEN_RESEARCH,
        packages=frozenset(),
    ),
}


@dataclass(frozen=True, slots=True)
class ImageRecipeStep:
    op: str
    args: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ImageRecipe:
    schema_version: str
    base: str
    steps: tuple[ImageRecipeStep, ...] = ()
    digest: str | None = None

    def __post_init__(self) -> None:
        if self.digest is None:
            object.__setattr__(self, "digest", _recipe_digest(self.to_wire(include_digest=False)))

    @classmethod
    def from_wire(cls, payload: Mapping[str, object]) -> Self:
        schema_version = _required_text(payload, "schema_version")
        base = _required_text(payload, "base")
        raw_steps = payload.get("steps")
        steps: list[ImageRecipeStep] = []
        if isinstance(raw_steps, Sequence) and not isinstance(raw_steps, (str, bytes)):
            for index, raw_step in enumerate(raw_steps):
                if not isinstance(raw_step, Mapping):
                    raise RuntimeImageError(f"recipe steps[{index}] must be an object")
                op = _required_text(raw_step, "op")
                args = _string_tuple(raw_step.get("args"), label=f"steps[{index}].args")
                steps.append(ImageRecipeStep(op=op, args=args))
        digest = _optional_text(payload, "digest")
        return cls(
            schema_version=schema_version,
            base=base,
            steps=tuple(steps),
            digest=digest,
        )

    def to_wire(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "base": self.base,
            "steps": [{"op": step.op, "args": list(step.args)} for step in self.steps],
        }
        if include_digest and self.digest is not None:
            payload["digest"] = self.digest
        return payload


@dataclass(frozen=True, slots=True)
class RuntimeImage:
    kind: RuntimeImageKind
    runtime_image_ref: str
    image_id: ActorImageId | None = None
    dependency_set_id: str | None = None
    environment_name: str | None = None
    recipe: ImageRecipe | None = None
    local_only: bool = False
    smoke_test_command: str | None = None

    def __post_init__(self) -> None:
        _validate_image_ref(self.runtime_image_ref)

    @classmethod
    def catalog(cls, image_id: ActorImageId | str) -> RuntimeImage:
        normalized = _coerce_actor_image_id(image_id)
        entry = ACTOR_IMAGE_CATALOG[normalized]
        return cls(
            kind=RuntimeImageKind.CATALOG,
            image_id=entry.image_id,
            runtime_image_ref=entry.runtime_image_ref,
            dependency_set_id=entry.dependency_set_id,
            environment_name=entry.environment_name,
        )

    @classmethod
    def ref(cls, image_ref: str, *, local_only: bool = True) -> RuntimeImage:
        normalized = str(image_ref or "").strip()
        if not normalized:
            raise RuntimeImageError("image_ref is required")
        return cls(
            kind=RuntimeImageKind.REF,
            runtime_image_ref=normalized,
            local_only=local_only,
        )

    @classmethod
    def extend(cls, base: ActorImageId | str) -> ImageBuilder:
        return ImageBuilder(base=_coerce_actor_image_id(base))

    @classmethod
    def from_wire(cls, payload: Mapping[str, object]) -> Self:
        kind = RuntimeImageKind(_required_text(payload, "kind"))
        runtime_image_ref = _required_text(payload, "runtime_image_ref")
        image_id_raw = _optional_text(payload, "image_id")
        image_id = _coerce_actor_image_id(image_id_raw) if image_id_raw else None
        recipe_payload = payload.get("recipe")
        recipe = (
            ImageRecipe.from_wire(recipe_payload) if isinstance(recipe_payload, Mapping) else None
        )
        return cls(
            kind=kind,
            runtime_image_ref=runtime_image_ref,
            image_id=image_id,
            dependency_set_id=_optional_text(payload, "dependency_set_id"),
            environment_name=_optional_text(payload, "environment_name"),
            recipe=recipe,
            local_only=bool(payload.get("local_only")),
            smoke_test_command=_optional_text(payload, "smoke_test_command"),
        )

    def image_ref(self) -> str:
        return self.runtime_image_ref

    def to_sandbox_override(self) -> dict[str, str]:
        return {"image": self.runtime_image_ref}

    def to_environment_patch(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        if self.environment_name:
            payload["name"] = self.environment_name
        if self.image_id is not None:
            payload["actor_runtime_image_id"] = self.image_id.value
        if self.dependency_set_id:
            payload["dependency_set_id"] = self.dependency_set_id
        if self.recipe is not None:
            payload["image_recipe"] = self.recipe.to_wire()
        payload["runtime_image"] = self.to_wire(include_nested=False)
        return payload

    def to_wire(self, *, include_nested: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "kind": self.kind.value,
            "runtime_image_ref": self.runtime_image_ref,
        }
        if self.image_id is not None:
            payload["image_id"] = self.image_id.value
        if self.dependency_set_id is not None:
            payload["dependency_set_id"] = self.dependency_set_id
        if self.environment_name is not None:
            payload["environment_name"] = self.environment_name
        if self.recipe is not None and include_nested:
            payload["recipe"] = self.recipe.to_wire()
        if self.local_only:
            payload["local_only"] = True
        if self.smoke_test_command is not None:
            payload["smoke_test_command"] = self.smoke_test_command
        return payload

    def require_local_docker_image(self) -> None:
        probe = subprocess.run(
            ["docker", "image", "inspect", self.runtime_image_ref],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode != 0:
            raise RuntimeImageError(
                f"missing local Docker image: {self.runtime_image_ref}. "
                "Build the actor runtime image before launching."
            )


@dataclass(frozen=True, slots=True)
class ImageBuilder:
    base: ActorImageId
    packages: tuple[RuntimePackageId, ...] = ()
    smoke_test_command: str | None = None

    def pip(self, *packages: RuntimePackageId | str) -> ImageBuilder:
        normalized = tuple(_coerce_runtime_package_id(package) for package in packages)
        for package in normalized:
            if package not in RUNTIME_PACKAGE_PINS:
                raise RuntimeImageError(f"unsupported runtime package: {package.value}")
        merged = tuple(dict.fromkeys((*self.packages, *normalized)))
        return ImageBuilder(
            base=self.base,
            packages=merged,
            smoke_test_command=self.smoke_test_command,
        )

    def smoke_test(self, command: str) -> ImageBuilder:
        normalized = str(command or "").strip()
        if not normalized:
            raise RuntimeImageError("smoke_test command is required")
        return ImageBuilder(
            base=self.base,
            packages=self.packages,
            smoke_test_command=normalized,
        )

    def build(self) -> RuntimeImage:
        matched = _match_catalog_entry(base=self.base, packages=frozenset(self.packages))
        if matched is not None:
            image = RuntimeImage.catalog(matched.image_id)
            if self.smoke_test_command is None:
                return image
            return RuntimeImage(
                kind=image.kind,
                image_id=image.image_id,
                runtime_image_ref=image.runtime_image_ref,
                dependency_set_id=image.dependency_set_id,
                environment_name=image.environment_name,
                recipe=image.recipe,
                smoke_test_command=self.smoke_test_command,
            )

        if self.packages:
            package_list = ", ".join(package.value for package in self.packages)
            raise RuntimeImageError(
                "runtime image recipe is not pre-built in the actor runtime catalog. "
                f"base={self.base.value} packages=[{package_list}]. "
                "Use RuntimeImage.catalog(...) or request a new catalog entry."
            )
        return RuntimeImage.catalog(self.base)


@dataclass(frozen=True, slots=True)
class EnvironmentSpec:
    name: str
    digest: str | None = None
    runtime_image: RuntimeImage | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise RuntimeImageError("environment name is required")

    def to_wire(self) -> dict[str, object]:
        payload: dict[str, object] = {"name": self.name.strip()}
        if self.digest:
            payload["digest"] = self.digest.strip()
        if self.runtime_image is not None:
            payload.update(self.runtime_image.to_environment_patch())
        return payload


@dataclass(frozen=True, slots=True)
class SandboxOverride:
    image: RuntimeImage | str | None = None
    snapshot: str | None = None
    extra: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.image is None and self.snapshot is None and not self.extra:
            raise RuntimeImageError("sandbox override requires image, snapshot, or extra fields")
        if self.image is not None and self.snapshot is not None:
            raise RuntimeImageError("sandbox override cannot set both image and snapshot")

    def to_wire(self) -> dict[str, str]:
        payload = dict(self.extra)
        if isinstance(self.image, RuntimeImage):
            payload.update(self.image.to_sandbox_override())
        elif isinstance(self.image, str) and self.image.strip():
            payload["image"] = self.image.strip()
        if self.snapshot is not None and self.snapshot.strip():
            payload["snapshot"] = self.snapshot.strip()
            payload.pop("image", None)
        return payload


def align_execution_profile_for_runtime_image(
    runtime_image: RuntimeImage,
    *,
    host_kind: object | None,
    execution_profile: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Keep local docker execution_profile material aligned with runtime_image selection."""
    if execution_profile is None:
        return None
    normalized_host_kind = _normalize_host_kind(host_kind)
    if normalized_host_kind != "docker":
        return {str(key): value for key, value in execution_profile.items()}
    merged = {str(key): value for key, value in execution_profile.items()}
    merged["docker_image"] = runtime_image.runtime_image_ref
    return merged


def runtime_image_launch_patches(
    runtime_image: RuntimeImage | None,
    *,
    environment: EnvironmentSpec | Mapping[str, object] | None = None,
    sandbox_override: SandboxOverride | Mapping[str, str] | None = None,
) -> tuple[dict[str, object] | None, dict[str, str] | None]:
    environment_payload: dict[str, object] | None = None
    sandbox_payload: dict[str, str] | None = None

    if isinstance(environment, EnvironmentSpec):
        environment_payload = environment.to_wire()
    elif isinstance(environment, Mapping):
        environment_payload = {str(key): value for key, value in environment.items()}

    if isinstance(sandbox_override, SandboxOverride):
        sandbox_payload = sandbox_override.to_wire()
    elif isinstance(sandbox_override, Mapping):
        sandbox_payload = {
            str(key): str(value)
            for key, value in sandbox_override.items()
            if str(key).strip() and str(value or "").strip()
        }

    if runtime_image is not None:
        environment_payload = _merge_object_payloads(
            environment_payload,
            runtime_image.to_environment_patch(),
        )
        sandbox_payload = _merge_string_payloads(
            sandbox_payload,
            runtime_image.to_sandbox_override(),
        )

    return environment_payload, sandbox_payload


def _match_catalog_entry(
    *,
    base: ActorImageId,
    packages: frozenset[RuntimePackageId],
) -> ActorImageCatalogEntry | None:
    for entry in ACTOR_IMAGE_CATALOG.values():
        if entry.image_id == base and not packages:
            return entry
        if entry.extends == base and entry.packages == packages:
            return entry
    return None


def _recipe_digest(payload: Mapping[str, object]) -> str:
    material = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(material.encode("utf-8")).hexdigest()


def _coerce_actor_image_id(value: ActorImageId | str) -> ActorImageId:
    if isinstance(value, ActorImageId):
        return value
    normalized = str(value or "").strip()
    if not normalized:
        raise RuntimeImageError("image_id is required")
    try:
        return ActorImageId(normalized)
    except ValueError as exc:
        raise RuntimeImageError(f"unknown actor image id: {normalized}") from exc


def _coerce_runtime_package_id(value: RuntimePackageId | str) -> RuntimePackageId:
    if isinstance(value, RuntimePackageId):
        return value
    normalized = str(value or "").strip()
    if not normalized:
        raise RuntimeImageError("package id is required")
    try:
        return RuntimePackageId(normalized)
    except ValueError as exc:
        raise RuntimeImageError(
            f"unsupported runtime package: {normalized}. "
            "Use RuntimePackageId from the SDK allowlist."
        ) from exc


def _validate_image_ref(image_ref: str) -> None:
    normalized = str(image_ref or "").strip()
    if not normalized:
        raise RuntimeImageError("runtime_image_ref is required")
    if normalized.startswith("sha256:"):
        return
    if not _IMAGE_REF_RE.match(normalized):
        raise RuntimeImageError(f"invalid runtime image ref: {normalized}")


def _required_text(payload: Mapping[str, object], key: str) -> str:
    value = _optional_text(payload, key)
    if value is None:
        raise RuntimeImageError(f"{key} is required")
    return value


def _optional_text(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise RuntimeImageError(f"{key} must be a string")
    text = value.strip()
    return text or None


def _string_tuple(value: object, *, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = [str(item).strip() for item in value if str(item or "").strip()]
        return tuple(items)
    raise RuntimeImageError(f"{label} must be a string or string list")


def _merge_object_payloads(
    left: dict[str, object] | None,
    right: dict[str, object] | None,
) -> dict[str, object] | None:
    if left is None and right is None:
        return None
    merged: dict[str, object] = {}
    if left is not None:
        merged.update(left)
    if right is not None:
        merged.update(right)
    return merged


def _normalize_host_kind(host_kind: object | None) -> str:
    if host_kind is None:
        return ""
    if isinstance(host_kind, str):
        return host_kind.strip().lower()
    value = getattr(host_kind, "value", host_kind)
    return str(value or "").strip().lower()


def _merge_string_payloads(
    left: dict[str, str] | None,
    right: dict[str, str] | None,
) -> dict[str, str] | None:
    if left is None and right is None:
        return None
    merged: dict[str, str] = {}
    if left is not None:
        merged.update(left)
    if right is not None:
        merged.update(right)
    return merged


__all__ = [
    "ACTOR_IMAGE_CATALOG",
    "ActorImageCatalogEntry",
    "ActorImageId",
    "EnvironmentSpec",
    "ImageBuilder",
    "ImageRecipe",
    "ImageRecipeStep",
    "RUNTIME_PACKAGE_PINS",
    "RuntimeImage",
    "RuntimeImageError",
    "RuntimeImageKind",
    "RuntimePackageId",
    "SandboxOverride",
    "align_execution_profile_for_runtime_image",
    "runtime_image_launch_patches",
]
