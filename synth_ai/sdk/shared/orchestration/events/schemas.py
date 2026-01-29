"""Rust-backed event data schemas for prompt learning jobs."""

from __future__ import annotations

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for orchestration.events.schemas.") from exc

_HAS_RUST_SCHEMAS = synth_ai_py is not None and hasattr(synth_ai_py, "ProgramCandidate")

# =============================================================================
# Size Limits for Event Payloads
# =============================================================================

if _HAS_RUST_SCHEMAS:
    MAX_INSTRUCTION_LENGTH = int(synth_ai_py.orchestration_max_instruction_length())
    MAX_ROLLOUT_SAMPLES = int(synth_ai_py.orchestration_max_rollout_samples())
    MAX_SEED_INFO_COUNT = int(synth_ai_py.orchestration_max_seed_info_count())
else:
    MAX_INSTRUCTION_LENGTH = 0
    MAX_ROLLOUT_SAMPLES = 0
    MAX_SEED_INFO_COUNT = 0

# =============================================================================
# Data Schemas (Rust-backed)
# =============================================================================

if _HAS_RUST_SCHEMAS:
    MutationTypeStats = synth_ai_py.MutationTypeStats
    MutationSummary = synth_ai_py.MutationSummary
    SeedAnalysis = synth_ai_py.SeedAnalysis
    PhaseSummary = synth_ai_py.PhaseSummary
    StageInfo = synth_ai_py.StageInfo
    SeedInfo = synth_ai_py.SeedInfo
    TokenUsage = synth_ai_py.TokenUsage
    ProgramCandidate = synth_ai_py.ProgramCandidate
    BaseCandidateEventData = getattr(synth_ai_py, "BaseCandidateEventData", dict)
else:
    MutationTypeStats = object
    MutationSummary = object
    SeedAnalysis = object
    PhaseSummary = object
    StageInfo = object
    SeedInfo = object
    TokenUsage = object
    ProgramCandidate = object
    BaseCandidateEventData = dict

__all__ = [
    "MAX_INSTRUCTION_LENGTH",
    "MAX_ROLLOUT_SAMPLES",
    "MAX_SEED_INFO_COUNT",
    "MutationTypeStats",
    "MutationSummary",
    "BaseCandidateEventData",
    "SeedAnalysis",
    "PhaseSummary",
    "StageInfo",
    "SeedInfo",
    "TokenUsage",
    "ProgramCandidate",
]
