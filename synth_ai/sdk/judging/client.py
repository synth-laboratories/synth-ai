"""Judge API types and verifier client alias.

This module provides types for LLM-based evaluation responses and the
VerifierClient alias for graph-based verification.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from synth_ai.sdk.graphs import VerifierClient as GraphVerifierClient

Provider = Literal["groq", "gemini"]


class JudgeOptions(TypedDict, total=False):
    """Options for judge/verifier requests."""
    event: bool
    outcome: bool
    rubric_id: str
    rubric_overrides: dict[str, Any]
    provider: Provider
    model: str
    max_concurrency: int
    verifier_type: str


class JudgeScoreResponse(TypedDict, total=False):
    """Response from judge/verifier scoring."""
    status: str
    event_rewards: list[dict[str, Any]]
    outcome_reward: dict[str, Any]
    details: dict[str, Any]


class VerifierClient(GraphVerifierClient):
    """Alias for graph-based VerifierClient.

    Use synth_ai.sdk.graphs.VerifierClient or GraphCompletionsClient directly.
    """
