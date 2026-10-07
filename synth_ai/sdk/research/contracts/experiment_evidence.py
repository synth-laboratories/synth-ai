"""Native trial/result identities; Forge Result retains its separate identity.

See backend SmrExperimentBundleResponse and Forge decision 0002.
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime

from synth_ai.sdk.research.contracts.run_state import (
    _optional_string,
    _require_mapping,
    _require_string,
)


@dataclass(frozen=True)
class NativeExperimentResult(Mapping[str, object]):
    result_id: str
    experiment_run_id: str | None = None
    run_id: str | None = None
    container_run_id: str | None = None
    scorer_id: str | None = None
    scorer_version: str | None = None
    scorer_config_digest: str | None = None
    raw: dict[str, object] = field(default_factory=dict, repr=False)

    @classmethod
    def from_wire(cls, value: object) -> "NativeExperimentResult":
        wire = _require_mapping(value, label="native result")
        return cls(
            result_id=_require_string(wire, "result_id", label="native result"),
            experiment_run_id=_optional_string(wire, "experiment_run_id"),
            run_id=_optional_string(wire, "run_id"),
            container_run_id=_optional_string(wire, "container_run_id"),
            scorer_id=_optional_string(wire, "scorer_id"),
            scorer_version=_optional_string(wire, "scorer_version"),
            scorer_config_digest=_optional_string(wire, "scorer_config_digest"),
            raw=dict(wire),
        )

    def __getitem__(self, key: str) -> object:
        return self.raw[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.raw)

    def __len__(self) -> int:
        return len(self.raw)


@dataclass(frozen=True)
class NativeExperimentTrial:
    experiment_run_id: str
    experiment_id: str
    experiment_revision: int
    run_id: str
    role: str
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, value: object) -> "NativeExperimentTrial":
        wire = _require_mapping(value, label="native trial")
        revision = wire.get("experiment_revision")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise ValueError("native trial experiment_revision must be a positive integer")
        created = _require_string(wire, "created_at", label="native trial")
        created_at = datetime.fromisoformat(created.replace("Z", "+00:00"))
        if created_at.tzinfo is None:
            raise ValueError("native trial created_at must include a timezone")
        metadata = wire.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("native trial metadata must be an object")
        return cls(
            experiment_run_id=_require_string(wire, "experiment_run_id", label="native trial"),
            experiment_id=_require_string(wire, "experiment_id", label="native trial"),
            experiment_revision=revision,
            run_id=_require_string(wire, "run_id", label="native trial"),
            role=_require_string(wire, "role", label="native trial"),
            created_at=created_at,
            metadata=dict(metadata),
        )
