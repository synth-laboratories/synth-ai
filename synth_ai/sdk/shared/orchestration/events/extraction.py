"""Stage and content extraction utilities for candidate events.

This module provides utilities to extract structured data from various candidate
formats used across algorithms. The extraction functions handle:

- Multi-stage genome (Dict[str, StageGene] with instruction_lines)
- Single-stage transformation (text_replacements grouped by role)
- Messages (patterns)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from .schemas import (
    ProgramCandidate,
    SeedInfo,
    StageInfo,
    TokenUsage,
)

_logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class StageExtractionError(Exception):
    """Raised when stage extraction fails and no fallback is acceptable."""

    pass


# =============================================================================
# Helper Functions
# =============================================================================


def seed_score_entry(seed: int, score: Any) -> Dict[str, Any]:
    """Create a properly formatted seed score entry.

    Args:
        seed: The seed ID
        score: The score value (can be None, float, or invalid)

    Returns:
        Dict with seed, score, and optional status for failures
    """
    if score is None:
        return {"seed": seed, "score": None, "status": "failed"}
    try:
        value = float(score)
    except (TypeError, ValueError):
        return {"seed": seed, "score": None, "status": "invalid"}
    # Check for NaN
    if isinstance(value, float) and value != value:
        return {"seed": seed, "score": None, "status": "invalid"}
    return {"seed": seed, "score": value}


def _extract_instruction_text(text: Any) -> str:
    """Extract clean instruction text from various formats (JSON, dict, or string)."""
    if text is None:
        return ""
    if isinstance(text, str):
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, dict):
                    for key in ["instruction", "text", "content", "prompt"]:
                        if key in parsed and isinstance(parsed[key], str):
                            return parsed[key]
                    for v in parsed.values():
                        if isinstance(v, str) and len(v) > 10:
                            return v
            except json.JSONDecodeError:
                pass
        return text
    elif isinstance(text, dict):
        for key in ["instruction", "text", "content", "prompt"]:
            if key in text and isinstance(text[key], str):
                return text[key]
    return str(text)


# =============================================================================
# Stage Extraction
# =============================================================================


def extract_stages_from_candidate(
    candidate: Dict[str, Any],
    *,
    require_stages: bool = False,
    candidate_id: Optional[str] = None,
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Extract unified stages format from a candidate object.

    Returns a dict of:
        {stage_id: {"instruction": str, "rules": dict, "temperature": float|None}}

    Handles:
    - Multi-stage genome (Dict[str, StageGene] with instruction_lines)
    - Single-stage transformation (text_replacements grouped by role)
    - Messages (patterns)

    Args:
        candidate: Candidate dictionary (from attempted_candidates or optimized_candidates)
        require_stages: If True, raise StageExtractionError instead of returning None
        candidate_id: Optional candidate ID for better error messages

    Returns:
        Dict of stages, or None if extraction fails (unless require_stages=True)

    Raises:
        StageExtractionError: If require_stages=True and extraction fails
    """
    cid = candidate_id or candidate.get("version_id", "unknown")[:16]

    obj = candidate.get("object", {})
    if not isinstance(obj, dict):
        obj = {}

    stages: Dict[str, Dict[str, Any]] = {}
    extraction_source = None

    # --------------------------------------------------------
    # 1. Check for multi-stage genome (Dict[str, StageGene])
    # --------------------------------------------------------
    genome_data = candidate.get("genome") or obj.get("genome")
    if genome_data and isinstance(genome_data, dict):
        for stage_id in sorted(genome_data.keys()):
            gene = genome_data.get(stage_id, {})
            if not isinstance(gene, dict):
                _logger.warning(
                    "[EXTRACT_STAGES] Skipping malformed gene for stage_id=%s in candidate %s: "
                    "expected dict, got %s",
                    stage_id,
                    cid,
                    type(gene).__name__,
                )
                continue

            instruction_lines = gene.get("instruction_lines", [])
            if instruction_lines and isinstance(instruction_lines, list):
                instruction_text = "\n".join(str(line) for line in instruction_lines)
                stages[stage_id] = {
                    "instruction": instruction_text,
                    "rules": gene.get("rules", {}),
                    "temperature": gene.get("temperature"),
                }

        if stages:
            extraction_source = "genome"
            _logger.debug(
                "[EXTRACT_STAGES] Built %d stages from genome for candidate %s",
                len(stages),
                cid,
            )

    # --------------------------------------------------------
    # 2. Check for single-stage transformation (text_replacements)
    # --------------------------------------------------------
    if not stages:
        # Try multiple paths to find text_replacements
        text_replacements: List[Any] = []

        # Path 1: obj.text_replacements (attribute)
        if hasattr(obj, "text_replacements") and obj.text_replacements:
            text_replacements = obj.text_replacements
        # Path 2: obj["text_replacements"]
        elif isinstance(obj, dict) and obj.get("text_replacements"):
            text_replacements = obj.get("text_replacements", [])
        # Path 3: obj["data"]["text_replacements"]
        elif isinstance(obj, dict) and isinstance(obj.get("data"), dict):
            text_replacements = obj.get("data", {}).get("text_replacements", [])
        # Path 4: candidate["transformation"]["text_replacements"] (transformation dict)
        elif isinstance(candidate.get("transformation"), dict):
            transformation = candidate.get("transformation", {})
            text_replacements = transformation.get("text_replacements", [])
        # Path 5: candidate["text_replacements"]
        elif candidate.get("text_replacements"):
            text_replacements = candidate.get("text_replacements", [])

        if text_replacements and isinstance(text_replacements, list):
            # Group replacements by role (each role becomes a "stage")
            role_instructions: Dict[str, List[str]] = {}

            for tr in text_replacements:
                if hasattr(tr, "new_text"):
                    new_text = tr.new_text
                    role = getattr(tr, "apply_to_role", "system") or "system"
                elif isinstance(tr, dict):
                    new_text = tr.get("new_text", "")
                    role = tr.get("apply_to_role", "system") or "system"
                else:
                    continue

                if new_text:
                    clean_instruction = _extract_instruction_text(new_text)
                    if clean_instruction:
                        role = role.lower()
                        if role not in role_instructions:
                            role_instructions[role] = []
                        role_instructions[role].append(clean_instruction)

            for role in sorted(role_instructions.keys()):
                instructions = role_instructions[role]
                stages[role] = {
                    "instruction": "\n".join(instructions),
                    "rules": {},
                    "temperature": None,
                }

            if stages:
                extraction_source = "text_replacements"
                _logger.debug(
                    "[EXTRACT_STAGES] Built %d stages from text_replacements for candidate %s",
                    len(stages),
                    cid,
                )

    # --------------------------------------------------------
    # 3. Fallback: Try messages (patterns)
    # --------------------------------------------------------
    if not stages:
        messages = []
        pattern = candidate.get("pattern") or obj.get("pattern")
        if isinstance(pattern, dict):
            messages = pattern.get("messages", []) or []
        if not messages:
            messages = candidate.get("messages") or obj.get("messages", [])
        # Also check transformation["messages"]
        if not messages and isinstance(candidate.get("transformation"), dict):
            transformation = candidate.get("transformation", {})
            messages = transformation.get("messages", [])
        if messages and isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, dict):
                    # Check for content OR pattern (pattern is used in TOML config files)
                    content = msg.get("content") or msg.get("pattern") or ""
                    if content:
                        role = msg.get("role", "system").lower()
                        if role not in stages:
                            stages[role] = {
                                "instruction": content,
                                "rules": {},
                                "temperature": None,
                            }
                        else:
                            # Append to existing role
                            stages[role]["instruction"] += "\n" + content

            if stages:
                extraction_source = "messages"
                _logger.debug(
                    "[EXTRACT_STAGES] Built %d stages from messages for candidate %s",
                    len(stages),
                    cid,
                )

    # --------------------------------------------------------
    # 4. Handle extraction failure
    # --------------------------------------------------------
    if not stages:
        available_keys = list(candidate.keys())
        obj_keys = list(obj.keys()) if isinstance(obj, dict) else []

        if require_stages:
            raise StageExtractionError(
                f"Failed to extract stages for candidate {cid}. "
                f"candidate keys: {available_keys}, object keys: {obj_keys}. "
                f"Expected one of: genome, text_replacements, messages"
            )
        else:
            _logger.warning(
                "[EXTRACT_STAGES] Failed to extract stages for candidate %s. "
                "candidate keys=%s, object keys=%s",
                cid,
                available_keys,
                obj_keys,
            )
            return None

    # --------------------------------------------------------
    # 5. Validate and return stages
    # --------------------------------------------------------
    for stage_id, stage_data in stages.items():
        instruction = stage_data.get("instruction", "")
        if not instruction or not instruction.strip():
            _logger.warning(
                "[EXTRACT_STAGES] Stage '%s' in candidate %s has empty instruction (source=%s)",
                stage_id,
                cid,
                extraction_source,
            )

    return stages


def extract_stages_required(
    candidate: Dict[str, Any],
    candidate_id: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Extract stages from candidate, raising an error if extraction fails.

    This is the strict version of extract_stages_from_candidate that should
    be used when stages are required (e.g., for event emission).

    Args:
        candidate: Candidate dictionary
        candidate_id: Optional candidate ID for better error messages

    Returns:
        Dict of stages (never None)

    Raises:
        StageExtractionError: If extraction fails
    """
    result = extract_stages_from_candidate(
        candidate, require_stages=True, candidate_id=candidate_id
    )
    # Type checker: result is guaranteed non-None due to require_stages=True
    assert result is not None
    return result


def extract_program_candidate_content(candidate: Dict[str, Any]) -> str:
    """Extract readable program/candidate content from a candidate object.

    PREFERRED: Use extract_stages_from_candidate and derive prompt_text from stages.
    This function is kept for backwards compatibility.

    Handles multiple candidate structures (shared across algorithms):
    - Transformations with text_replacements
    - Patterns with messages
    - Raw prompt_text field

    Args:
        candidate: Candidate dictionary

    Returns:
        Extracted program/candidate content string
    """
    # Check for pattern at top level or nested
    pattern = candidate.get("pattern") or candidate.get("object", {}).get("pattern")
    if isinstance(pattern, dict):
        pattern_messages = pattern.get("messages", [])
        if pattern_messages:
            result_parts = []
            for message in pattern_messages[:5]:
                if isinstance(message, dict):
                    role = message.get("role", message.get("name", "system"))
                    content = message.get("pattern") or message.get("content") or ""
                    if content:
                        result_parts.append(f"[{role.upper()}]: {content}")
            if result_parts:
                return "\n".join(result_parts)
    # Legacy template key (sections)
    template = candidate.get("template") or candidate.get("object", {}).get("template")
    if isinstance(template, dict):
        template_sections = template.get("sections", [])
        if template_sections:
            result_parts = []
            for section in template_sections[:5]:
                if isinstance(section, dict):
                    role = section.get("role", section.get("name", "system"))
                    content = section.get("content", "")
                    if content:
                        result_parts.append(f"[{role.upper()}]: {content}")
            if result_parts:
                return "\n".join(result_parts)

    # First, try to derive from stages (preferred method)
    stages = extract_stages_from_candidate(candidate)
    if stages:
        parts = []
        for stage_id in sorted(stages.keys()):
            instruction = stages[stage_id].get("instruction", "")
            if instruction:
                parts.append(f"[{stage_id.upper()}]: {instruction}")
        if parts:
            return "\n".join(parts)

    # Fallback: Try direct prompt_text field
    if candidate.get("prompt_text"):
        return str(candidate["prompt_text"])

    obj = candidate.get("object", {})
    if not isinstance(obj, dict):
        obj = {}

    result_parts: List[str] = []

    # Try text_replacements (transformations)
    text_replacements = obj.get("text_replacements", [])
    if not text_replacements:
        data_dict = obj.get("data", {})
        if isinstance(data_dict, dict):
            text_replacements = data_dict.get("text_replacements", [])
    if not text_replacements and isinstance(candidate.get("transformation"), dict):
        transformation = candidate.get("transformation", {})
        text_replacements = transformation.get("text_replacements", [])
    if not text_replacements:
        text_replacements = candidate.get("text_replacements", [])

    if text_replacements and isinstance(text_replacements, list):
        for repl in text_replacements[:5]:
            if isinstance(repl, dict):
                new_text = repl.get("new_text", "")
                role = repl.get("apply_to_role", "system")
                if new_text:
                    result_parts.append(f"[{role.upper()}]: {new_text}")

    # Try messages (patterns)
    if not result_parts:
        messages = []
        pattern = candidate.get("pattern") or obj.get("pattern")
        if isinstance(pattern, dict):
            messages = pattern.get("messages", []) or []
        if not messages:
            messages = candidate.get("messages") or obj.get("messages", [])
        if not messages and isinstance(candidate.get("transformation"), dict):
            transformation = candidate.get("transformation", {})
            messages = transformation.get("messages", [])
        if messages and isinstance(messages, list):
            for msg in messages[:5]:
                if isinstance(msg, dict):
                    role = msg.get("role", "system")
                    content = msg.get("content", "")
                    if content:
                        result_parts.append(f"[{role.upper()}]: {content}")

    # Try sections (GraphGen legacy template format)
    if not result_parts:
        sections = obj.get("sections", [])
        if not sections:
            data_dict = obj.get("data", {})
            if isinstance(data_dict, dict):
                sections = data_dict.get("sections", [])
        if not sections:
            sections = candidate.get("sections", [])
        if sections and isinstance(sections, list):
            for section in sections[:5]:
                if isinstance(section, dict):
                    role = section.get("role", section.get("name", "system"))
                    content = section.get("content", "")
                    if content:
                        result_parts.append(f"[{role.upper()}]: {content}")

    # Try instruction_text
    if not result_parts:
        instruction = obj.get("instruction_text") or obj.get("prompt_text")
        if not instruction and isinstance(candidate.get("transformation"), dict):
            transformation = candidate.get("transformation", {})
            instruction = transformation.get("instruction_text") or transformation.get(
                "prompt_text"
            )
        if instruction:
            result_parts.append(str(instruction))

    return "\n".join(result_parts)


# Alias for backwards compatibility
extract_prompt_text_from_candidate = extract_program_candidate_content


# =============================================================================
# Transformation Normalization
# =============================================================================


def normalize_transformation(transformation: Any) -> Optional[Dict[str, Any]]:
    """Normalize transformation data to a consistent dict format.

    Transformations can come in various formats:
    - List of dicts (array of text_replacements)
    - Dict with nested structure
    - None

    This function normalizes to a consistent dict format with:
    - text_replacements: List of replacement dicts
    - mutation_type: Type of mutation if available
    - parent_id: Parent transformation ID if available

    Args:
        transformation: Raw transformation data (any format)

    Returns:
        Normalized transformation dict, or None if empty/invalid
    """
    if transformation is None:
        return None

    result: Dict[str, Any] = {}

    # Handle list format (array of text_replacements)
    if isinstance(transformation, list):
        result["text_replacements"] = transformation
        return result

    # Handle dict format
    if isinstance(transformation, dict):
        # Copy known fields
        if "text_replacements" in transformation:
            result["text_replacements"] = transformation["text_replacements"]
        if (
            "data" in transformation
            and isinstance(transformation["data"], dict)
            and "text_replacements" in transformation["data"]
        ):
            result["text_replacements"] = transformation["data"]["text_replacements"]
        if "mutation_type" in transformation:
            result["mutation_type"] = transformation["mutation_type"]
        if "parent_id" in transformation:
            result["parent_id"] = transformation["parent_id"]
        if "version_id" in transformation:
            result["version_id"] = transformation["version_id"]

        # If no specific fields found, return the original dict
        if not result:
            result = dict(transformation)

        return result

    return None


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
    """Build a ProgramCandidate from a candidate dictionary.

    This is the main entry point for creating first-class program candidates
    from the various candidate formats in the codebase.

    Args:
        candidate: Candidate dictionary from optimizer/evaluator
        candidate_id: Override for candidate ID
        seed_info: Seed metadata (query text, expected output)
        token_usage: Token usage for this candidate
        cost_usd: Cost in USD
        timestamp_ms: Timestamp in milliseconds

    Returns:
        ProgramCandidate with all available data

    Raises:
        StageExtractionError: If stages cannot be extracted
    """
    cid = candidate_id or candidate.get("version_id") or candidate.get("candidate_id") or "unknown"

    # Extract stages (required)
    stages_dict = extract_stages_from_candidate(candidate, candidate_id=cid)

    # Convert to StageInfo objects
    stages: Dict[str, StageInfo] = {}
    if stages_dict:
        for stage_id, stage_data in stages_dict.items():
            stages[stage_id] = StageInfo.from_dict(stage_data)
    else:
        # Create a minimal stage with whatever prompt text we can find
        prompt_text = extract_program_candidate_content(candidate)
        if prompt_text:
            stages["system"] = StageInfo(instruction=prompt_text)
        else:
            # Last resort: create empty system stage
            stages["system"] = StageInfo(instruction="")

    # Extract seed_scores
    seed_scores = candidate.get("seed_scores")
    if not seed_scores:
        instance_scores = candidate.get("instance_scores", [])
        eval_seeds = candidate.get("seed_eval_info", {}).get("seeds", [])
        if instance_scores and eval_seeds:
            seed_scores = [
                seed_score_entry(seed, score)
                for seed, score in zip(eval_seeds, instance_scores, strict=False)
            ]

    # Normalize transformation
    transformation = normalize_transformation(candidate.get("transformation"))

    objectives = candidate.get("objectives")
    if objectives is None and isinstance(candidate.get("score"), dict):
        objectives = candidate["score"].get("objectives")
    instance_objectives = candidate.get("instance_objectives")
    if instance_objectives is None and isinstance(candidate.get("score"), dict):
        instance_objectives = candidate["score"].get("instance_objectives")

    return ProgramCandidate(
        candidate_id=cid,
        generation=candidate.get("generation", 0),
        stages=stages,
        parent_id=candidate.get("parent_id"),
        mutation_type=candidate.get("mutation_type") or candidate.get("operator") or "unknown",
        mutation_params=candidate.get("mutation_params"),
        accuracy=candidate.get("accuracy", 0) or 0,
        val_accuracy=candidate.get("val_accuracy") or candidate.get("full_score"),
        minibatch_score=candidate.get("minibatch_score"),
        seed_scores=seed_scores,
        seed_info=seed_info,
        instance_scores=candidate.get("instance_scores"),
        objectives=objectives,
        instance_objectives=instance_objectives,
        newly_solved_seeds=candidate.get("newly_solved_seeds"),
        artifact_refs=candidate.get("artifact_refs"),
        success_statuses=candidate.get("success_statuses"),
        token_usage=token_usage,
        cost_usd=cost_usd,
        timestamp_ms=timestamp_ms or int(time.time() * 1000),
        transformation=transformation,
        prompt_length=candidate.get("prompt_length"),
        status=candidate.get("status", "evaluated"),
        context_override_bundle_id=candidate.get("context_override_bundle_id"),
        context_overrides=candidate.get("context_overrides"),
        override_application_status=candidate.get("override_application_status"),
        override_application_errors=candidate.get("override_application_errors"),
    )


__all__ = [
    # Exceptions
    "StageExtractionError",
    # Helper functions
    "seed_score_entry",
    # Stage extraction
    "extract_stages_from_candidate",
    "extract_stages_required",
    "extract_program_candidate_content",
    "extract_prompt_text_from_candidate",  # alias
    # Transformation
    "normalize_transformation",
    # Program candidate builder
    "build_program_candidate",
]
