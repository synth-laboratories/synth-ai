"""Canonical projection digests and API/SDK/MCP surface-parity evidence.

The Gate-2 acceptance authority (``evals/suites/product/intern/
product_acceptance.py``) validates one ``surfaces`` object whose ``api``,
``mcp``, and ``frontend`` entries must observe the same canonical identity and
the same state generations. This module is the single implementation of:

1. the canonical digest -- SHA-256 over compact, key-sorted, ASCII JSON,
   byte-identical to the acceptance authority's ``_canonical_digest``; and
2. the canonical wire form of a Sync/Async projection -- the strict contract
   model's ``to_wire()`` output (null-elided), so that raw API JSON, the typed
   SDK object, and the MCP tool result all normalize to the same bytes.

The backend remains the projection authority; this module only observes and
digests what the backend served. It deliberately does not relax any contract:
payloads that fail strict validation cannot be digested.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from synth_ai.sdk.research.contracts.research_intern import (
    InternAsyncRuntime,
    InternSyncSession,
)

#: The product surfaces the acceptance authority requires evidence from.
REQUIRED_SURFACES: tuple[str, ...] = ("api", "mcp", "frontend")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize ``value`` exactly as the Gate-2 acceptance authority does."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def canonical_digest(value: Any) -> str:
    """SHA-256 digest of the canonical JSON form, ``sha256:<hex>``."""
    return f"sha256:{hashlib.sha256(canonical_json_bytes(value)).hexdigest()}"


def sync_projection_canonical_wire(
    payload: Mapping[str, Any] | InternSyncSession,
) -> dict[str, Any]:
    """Normalize a Sync session projection to its canonical wire form.

    The canonical form is the strict ``InternSyncSession`` contract's
    ``to_wire()`` output: strict validation first (unknown fields fail), then a
    JSON-mode dump with ``None`` values elided. Raw API JSON, the SDK's typed
    object, and the MCP ``intern_sync_get`` result all reduce to this one form.
    """
    if isinstance(payload, InternSyncSession):
        return payload.to_wire()
    return InternSyncSession.from_wire(payload).to_wire()


def sync_projection_digest(payload: Mapping[str, Any] | InternSyncSession) -> str:
    """Canonical digest of a Sync session projection, stable across surfaces."""
    return canonical_digest(sync_projection_canonical_wire(payload))


def async_runtime_canonical_wire(
    payload: Mapping[str, Any] | InternAsyncRuntime,
) -> dict[str, Any]:
    """Normalize an Async runtime projection to its canonical wire form."""
    if isinstance(payload, InternAsyncRuntime):
        return payload.to_wire()
    return InternAsyncRuntime.from_wire(payload).to_wire()


def async_runtime_digest(payload: Mapping[str, Any] | InternAsyncRuntime) -> str:
    """Canonical digest of an Async runtime projection, stable across surfaces."""
    return canonical_digest(async_runtime_canonical_wire(payload))


class _StrictEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CanonicalIdentity(_StrictEvidence):
    """The eight-field identity every product surface must agree on."""

    organization_id: str = Field(min_length=1)
    research_intern_id: str = Field(min_length=1)
    sync_session_id: str = Field(min_length=1)
    async_runtime_id: str = Field(min_length=1)
    factory_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    effort_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)


class SurfaceObservation(_StrictEvidence):
    """One surface's observation of the canonical identity and projections."""

    surface: Literal["api", "mcp", "frontend"]
    status: Literal["passed", "failed"]
    identity: CanonicalIdentity
    sync_state_generation: int = Field(ge=0)
    async_state_generation: int = Field(ge=0)
    receipt_ids: tuple[str, ...] = Field(min_length=1)
    sync_projection_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


def build_surface_parity(
    observations: Iterable[SurfaceObservation],
) -> dict[str, Any]:
    """Assemble the acceptance-authority ``surfaces`` object, or refuse.

    Refuses (``ValueError``) unless exactly the required surfaces are present,
    every surface passed, and all surfaces agree on identity, on both state
    generations, and on the canonical Sync projection digest.
    """
    by_surface: dict[str, SurfaceObservation] = {}
    for observation in observations:
        if observation.surface in by_surface:
            raise ValueError(f"duplicate surface observation: {observation.surface}")
        by_surface[observation.surface] = observation
    missing = [name for name in REQUIRED_SURFACES if name not in by_surface]
    if missing:
        raise ValueError(f"missing surface observations: {missing}")
    failed = sorted(
        name for name, observation in by_surface.items() if observation.status != "passed"
    )
    if failed:
        raise ValueError(f"surfaces did not pass: {failed}")
    identities = {observation.identity for observation in by_surface.values()}
    if len(identities) != 1:
        raise ValueError("surfaces observed divergent canonical identities")
    sync_generations = {o.sync_state_generation for o in by_surface.values()}
    if len(sync_generations) != 1:
        raise ValueError("surfaces observed divergent sync_state_generation values")
    async_generations = {o.async_state_generation for o in by_surface.values()}
    if len(async_generations) != 1:
        raise ValueError("surfaces observed divergent async_state_generation values")
    digests = {o.sync_projection_digest for o in by_surface.values()}
    if len(digests) != 1:
        raise ValueError("surfaces observed divergent canonical projection digests")
    return {
        name: {
            "status": by_surface[name].status,
            "identity": by_surface[name].identity.model_dump(mode="json"),
            "sync_state_generation": by_surface[name].sync_state_generation,
            "async_state_generation": by_surface[name].async_state_generation,
            "receipt_ids": list(by_surface[name].receipt_ids),
            "sync_projection_digest": by_surface[name].sync_projection_digest,
        }
        for name in REQUIRED_SURFACES
    }


def emit_surface_parity_json(
    path: Path,
    observations: Iterable[SurfaceObservation],
) -> dict[str, Any]:
    """Write a validated ``surface_parity.json`` and return its content.

    ``path`` is an explicit argument by policy: evidence locations are wired
    through code or TOML configuration, never environment variables.
    """
    surfaces = build_surface_parity(observations)
    path.write_text(json.dumps(surfaces, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return surfaces


__all__ = [
    "REQUIRED_SURFACES",
    "CanonicalIdentity",
    "SurfaceObservation",
    "async_runtime_canonical_wire",
    "async_runtime_digest",
    "build_surface_parity",
    "canonical_digest",
    "canonical_json_bytes",
    "emit_surface_parity_json",
    "sync_projection_canonical_wire",
    "sync_projection_digest",
]
