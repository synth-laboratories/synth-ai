"""Judging SDK - LLM-based trace evaluation.

This module provides the JudgeClient for scoring traces using
LLM judges. Experimental API - subject to change.

Example:
    from synth_ai.sdk.judging import JudgeClient
    
    client = JudgeClient(base_url, api_key)
    result = await client.score(
        trace=trace_data,
        policy_name="my_policy",
        task_app_id="task_123",
        options={"event": True, "outcome": True},
    )
"""

from __future__ import annotations

# Re-export from existing location
from synth_ai.evals.client import (
    JudgeClient,
    JudgeOptions,
    JudgeScoreResponse,
)

__all__ = [
    "JudgeClient",
    "JudgeOptions",
    "JudgeScoreResponse",
]

