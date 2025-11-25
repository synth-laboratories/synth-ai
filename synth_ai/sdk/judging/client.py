"""Experimental Judge API client.

This surface is experimental and subject to change without notice.
Set environment variable `SYNTH_SILENCE_EXPERIMENTAL=1` to silence warnings.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Literal, TypedDict

from synth_ai.core.http import AsyncHttpClient, HTTPError
from synth_ai.core.tracing_v3.serialization import normalize_for_json

Provider = Literal["groq", "gemini"]


class JudgeOptions(TypedDict, total=False):
    event: bool
    outcome: bool
    rubric_id: str
    rubric_overrides: dict[str, Any]
    provider: Provider
    model: str
    max_concurrency: int


class JudgeScoreResponse(TypedDict, total=False):
    status: str
    event_rewards: list[dict[str, Any]]
    outcome_reward: dict[str, Any]
    details: dict[str, Any]


class JudgeClient:
    """Client for LLM-based evaluation of task app traces.
    
    This client provides programmatic access to Synth AI's judge API, which uses
    LLMs to evaluate task execution traces and generate rewards. The judge can
    evaluate both event-level (step-by-step) and outcome-level (episode-level) rewards.
    
    .. warning::
        This API is experimental and subject to change without notice.
        Set `SYNTH_SILENCE_EXPERIMENTAL=1` to silence warnings.
    
    Example:
        >>> from synth_ai.sdk.judging import JudgeClient, JudgeOptions
        >>> 
        >>> client = JudgeClient(
        ...     base_url="https://api.usesynth.ai",
        ...     api_key=os.environ["SYNTH_API_KEY"],
        ... )
        >>> 
        >>> # Score a trace with outcome reward
        >>> result = await client.score(
        ...     trace=my_trace_dict,
        ...     policy_name="my_policy",
        ...     task_app_id="heartdisease",
        ...     options=JudgeOptions(
        ...         outcome=True,
        ...         rubric_id="accuracy",
        ...         provider="groq",
        ...         model="llama-3.1-8b-instant",
        ...     ),
        ... )
        >>> 
        >>> print(f"Outcome reward: {result['outcome_reward']}")
    """
    
    def __init__(self, base_url: str, api_key: str, *, timeout: float = 60.0) -> None:
        """Initialize the judge client.
        
        Args:
            base_url: Base URL for the Synth AI API
            api_key: API key for authentication
            timeout: Request timeout in seconds (default: 60.0)
        """
        _silence = (os.getenv("SYNTH_SILENCE_EXPERIMENTAL") or "").strip().lower()
        if _silence not in {"1", "true", "t", "yes", "y", "on"}:
            warnings.warn(
                "Experimental API: synth_ai.sdk.judging.JudgeClient is experimental and may change without notice.",
                UserWarning,
                stacklevel=2,
            )
        self._base = base_url.rstrip("/")
        self._key = api_key
        self._timeout = timeout

    async def score(
        self,
        *,
        trace: dict[str, Any] | Any,
        policy_name: str,
        task_app_id: str,
        options: JudgeOptions,
        task_app_base_url: str | None = None,
    ) -> JudgeScoreResponse:
        """Score a task execution trace using LLM-based evaluation.
        
        This method sends a trace to the judge API, which evaluates it according
        to the provided rubric and returns event-level and/or outcome-level rewards.
        
        Args:
            trace: Task execution trace (SessionTrace dict or compatible object)
            policy_name: Name of the policy that generated this trace
            task_app_id: Identifier for the task app (e.g., "heartdisease")
            options: Judge configuration options:
                - event: Whether to generate event-level rewards (default: False)
                - outcome: Whether to generate outcome-level reward (default: False)
                - rubric_id: Rubric identifier to use for evaluation
                - rubric_overrides: Optional rubric modifications
                - provider: LLM provider ("groq" or "gemini")
                - model: Model identifier (e.g., "llama-3.1-8b-instant")
                - max_concurrency: Max concurrent judge calls (default: 1)
            task_app_base_url: Optional base URL for task app (for rubric fetching)
            
        Returns:
            JudgeScoreResponse with:
                - status: "ok" or error status
                - event_rewards: List of event-level reward dicts (if event=True)
                - outcome_reward: Outcome-level reward dict (if outcome=True)
                - details: Additional evaluation details
                
        Raises:
            ValueError: If validation fails or rubric is invalid
            PermissionError: If authentication fails
            FileNotFoundError: If task app or rubric not found
            Exception: For rate limiting or transient errors
        """
        body = {
            "policy_name": policy_name,
            "task_app": {"id": task_app_id, **({"base_url": task_app_base_url} if task_app_base_url else {})},
            "trace": normalize_for_json(trace),
            "options": options or {},
        }
        try:
            async with AsyncHttpClient(self._base, self._key, timeout=self._timeout) as http:
                js = await http.post_json("/api/judge/v1/score", json=body)
                if not isinstance(js, dict):
                    raise ValueError("invalid_judge_response_shape")
                return js  # type: ignore[return-value]
        except HTTPError as err:  # map to friendlier exceptions
            status = int(getattr(err, "status", 0) or 0)
            if status in (400, 422):
                raise ValueError(f"judge_validation_error: {err.detail}") from err
            if status in (401, 403):
                raise PermissionError(f"judge_auth_error: {err.detail}") from err
            if status == 404:
                raise FileNotFoundError(f"judge_route_not_found: {err.detail}") from err
            if status == 429:
                raise Exception("judge_rate_limited") from err  # replace with RetryLater in future
            if status >= 500:
                raise Exception("judge_transient_error") from err  # replace with TransientError in future
            raise
