"""Rust-backed event data schemas for prompt learning jobs."""

from __future__ import annotations

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover
    raise RuntimeError("synth_ai_py is required for orchestration.events.schemas.") from exc


if synth_ai_py is None or not hasattr(synth_ai_py, "ProgramCandidate"):
    raise RuntimeError("Rust core event schemas required; synth_ai_py is unavailable.")

# =============================================================================
# Size Limits for Event Payloads
# =============================================================================

MAX_INSTRUCTION_LENGTH = int(synth_ai_py.orchestration_max_instruction_length())
MAX_ROLLOUT_SAMPLES = int(synth_ai_py.orchestration_max_rollout_samples())
MAX_SEED_INFO_COUNT = int(synth_ai_py.orchestration_max_seed_info_count())

# =============================================================================
# Data Schemas (Rust-backed)
# =============================================================================

MutationTypeStats = synth_ai_py.MutationTypeStats
MutationSummary = synth_ai_py.MutationSummary
SeedAnalysis = synth_ai_py.SeedAnalysis
PhaseSummary = synth_ai_py.PhaseSummary
StageInfo = synth_ai_py.StageInfo
SeedInfo = synth_ai_py.SeedInfo
TokenUsage = synth_ai_py.TokenUsage
ProgramCandidate = synth_ai_py.ProgramCandidate
BaseCandidateEventData = getattr(synth_ai_py, "BaseCandidateEventData", dict)

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
