"""Dataset registry and helpers shared by Task Apps."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import Any, List

from pydantic import BaseModel, Field, field_validator


class TaskDatasetSpec(BaseModel):
    """Declarative metadata describing a dataset that a Task App exposes."""

    id: str
    name: str
    version: str | None = None
    splits: list[str] = Field(default_factory=list)
    default_split: str | None = None
    cardinality: int | None = None
    description: str | None = None

    @field_validator("default_split")
    @classmethod
    def _validate_default_split(cls, value: str | None, info):
        values = info.data if hasattr(info, "data") else {}  # type: ignore[attr-defined]
        if value and value not in (values.get("splits") or []):
            raise ValueError("default_split must be one of splits when provided")
        return value


RegistryLoader = Callable[[TaskDatasetSpec], Any]


class TaskDatasetRegistry:
    """Lightweight registry mapping dataset specs to loader callables."""

    def __init__(self) -> None:
        self._entries: dict[str, tuple[TaskDatasetSpec, RegistryLoader, bool]] = {}
        self._cache: dict[Hashable, Any] = {}

    def register(
        self, spec: TaskDatasetSpec, loader: RegistryLoader, *, cache: bool = True
    ) -> None:
        """Register a dataset loader and its metadata."""

        self._entries[spec.id] = (spec, loader, cache)

    def describe(self, dataset_id: str) -> TaskDatasetSpec:
        if dataset_id not in self._entries:
            raise KeyError(f"Dataset not registered: {dataset_id}")
        spec, _, _ = self._entries[dataset_id]
        return spec

    def list(self) -> List[TaskDatasetSpec]:
        return [entry[0] for entry in self._entries.values()]

    def get(self, spec: TaskDatasetSpec | str) -> Any:
        """Return dataset materialisation (with optional caching)."""

        if isinstance(spec, str):
            if spec not in self._entries:
                raise KeyError(f"Dataset not registered: {spec}")
            base_spec, loader, cache_enabled = self._entries[spec]
            effective_spec = base_spec
        else:
            if spec.id not in self._entries:
                raise KeyError(f"Dataset not registered: {spec.id}")
            base_spec, loader, cache_enabled = self._entries[spec.id]
            effective_spec = base_spec.model_copy(update=spec.model_dump(exclude_unset=True))

        cache_key: Hashable = (
            effective_spec.id,
            effective_spec.version,
            effective_spec.default_split,
        )
        if cache_enabled:
            if cache_key not in self._cache:
                self._cache[cache_key] = loader(effective_spec)
            return self._cache[cache_key]
        return loader(effective_spec)

    @staticmethod
    def ensure_split(spec: TaskDatasetSpec, split: str | None) -> str:
        """Validate that `split` exists on the spec; return a concrete split."""

        if not spec.splits:
            return split or spec.default_split or "default"
        if split is None:
            if spec.default_split:
                return spec.default_split
            raise ValueError(f"split must be provided for dataset {spec.id}")
        if split not in spec.splits:
            raise ValueError(f"Unknown split '{split}' for dataset {spec.id}")
        return split

    @staticmethod
    def normalise_seed(seed: Any, *, cardinality: int | None = None) -> int:
        """Normalise arbitrary seed input into a bounded non-negative integer."""

        try:
            value = int(seed)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"Seed must be convertible to int (got {seed!r})") from exc
        if value < 0:
            value = abs(value)
        if cardinality and cardinality > 0:
            value = value % cardinality
        return value
