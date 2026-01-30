"""Stage and content extraction utilities for candidate events.

Rust-backed wrappers that delegate to synth_ai_core via synth_ai_py.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .schemas import ProgramCandidate, SeedInfo, TokenUsage

try:
    import synth_ai_py
except Exception as exc:  # pragma: no cover - rust bindings required
    raise RuntimeError("synth_ai_py is required for orchestration.events.extraction.") from exc


# =============================================================================
# Exceptions
# =============================================================================


class StageExtractionError(Exception):
    """Raised when stage extraction fails and no fallback is acceptable."""

    pass


# =============================================================================
# Rust binding helpers
# =============================================================================


def _require_rust():
    if synth_ai_py is None or not hasattr(
        synth_ai_py, "orchestration_extract_stages_from_candidate"
    ):
        raise RuntimeError("Rust core extraction helpers required; synth_ai_py is unavailable.")
    return synth_ai_py


def _maybe_to_dict(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        try:
            return value.to_dict()
        except Exception:
            pass
    return value


# =============================================================================
# Helper Functions
# =============================================================================


def seed_reward_entry(seed: int, score: Any) -> Dict[str, Any]:
    """Create a properly formatted seed reward entry."""

    rust = _require_rust()
    return rust.orchestration_seed_reward_entry(seed, score)


def seed_score_entry(seed: int, score: Any) -> Dict[str, Any]:
    """Deprecated: use seed_reward_entry instead."""
    return seed_reward_entry(seed, score)


# =============================================================================
# Stage Extraction
# =============================================================================


def extract_stages_from_candidate(
    candidate: Dict[str, Any],
    *,
    require_stages: bool = False,
    candidate_id: Optional[str] = None,
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Extract unified stages format from a candidate object."""

    rust = _require_rust()
    try:
        return rust.orchestration_extract_stages_from_candidate(
            candidate, require_stages, candidate_id
        )
    except Exception as exc:
        if require_stages:
            raise StageExtractionError(str(exc)) from exc
        raise


def extract_stages_required(
    candidate: Dict[str, Any],
    candidate_id: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Extract stages from candidate, raising an error if extraction fails."""

    result = extract_stages_from_candidate(
        candidate, require_stages=True, candidate_id=candidate_id
    )
    assert result is not None
    return result


def extract_program_candidate_content(candidate: Dict[str, Any]) -> str:
    """Extract readable program/candidate content from a candidate object."""

    rust = _require_rust()
    return rust.orchestration_extract_program_candidate_content(candidate)


# Alias for backwards compatibility
extract_prompt_text_from_candidate = extract_program_candidate_content


# =============================================================================
# Transformation Normalization
# =============================================================================


def normalize_transformation(transformation: Any) -> Optional[Dict[str, Any]]:
    """Normalize transformation data to a consistent dict format."""

    rust = _require_rust()
    return rust.orchestration_normalize_transformation(transformation)


# =============================================================================
# Program Candidate Builder
# =============================================================================


def build_program_candidate(
    candidate: Dict[str, Any],
    *,
    candidate_id: Optional[str] = None,
    seed_info: Optional[List[SeedInfo]] = None,
    token_usage: Optional[TokenUsage] = None,
    cost_usd: Optional[float] = None,
    timestamp_ms: Optional[int] = None,
) -> ProgramCandidate:
    """Build a ProgramCandidate from a candidate dictionary."""

    rust = _require_rust()
    payload = rust.orchestration_build_program_candidate(
        candidate,
        candidate_id,
        [_maybe_to_dict(s) for s in seed_info] if seed_info is not None else None,
        _maybe_to_dict(token_usage),
        cost_usd,
        timestamp_ms,
    )

    if not isinstance(payload, dict):
        raise RuntimeError("Rust build_program_candidate returned non-dict payload")

    return ProgramCandidate.from_dict(payload)


__all__ = [
    "StageExtractionError",
    "seed_reward_entry",
    "seed_score_entry",
    "extract_stages_from_candidate",
    "extract_stages_required",
    "extract_program_candidate_content",
    "extract_prompt_text_from_candidate",
    "normalize_transformation",
    "build_program_candidate",
]
