from __future__ import annotations

"""Experimental Judge API client.

This surface is experimental and subject to change without notice.
Set environment variable `SYNTH_SILENCE_EXPERIMENTAL=1` to silence warnings.
"""

import os
import warnings
from typing import Any, Literal, TypedDict

from synth_ai.http import AsyncHttpClient, HTTPError
from synth_ai.tracing_v3.serialization import normalize_for_json


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
	def __init__(self, base_url: str, api_key: str, *, timeout: float = 60.0) -> None:
		_silence = (os.getenv("SYNTH_SILENCE_EXPERIMENTAL") or "").strip().lower()
		if _silence not in {"1", "true", "t", "yes", "y", "on"}:
			warnings.warn(
				"Experimental API: synth_ai.evals.JudgeClient is experimental and may change without notice.",
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
		except HTTPError as e:  # map to friendlier exceptions
			status = int(getattr(e, "status", 0) or 0)
			if status in (400, 422):
				raise ValueError(f"judge_validation_error: {e.detail}") from e
			if status in (401, 403):
				raise PermissionError(f"judge_auth_error: {e.detail}") from e
			if status == 404:
				raise FileNotFoundError(f"judge_route_not_found: {e.detail}") from e
			if status == 429:
				raise Exception("judge_rate_limited") from e  # replace with RetryLater in future
			if status >= 500:
				raise Exception("judge_transient_error") from e  # replace with TransientError in future
			raise


