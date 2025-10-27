from __future__ import annotations

import os
from typing import Any

import pytest

from synth_ai.evals.client import JudgeClient


def _require_env(var: str) -> str:
	val = os.getenv(var)
	if not val:
		pytest.skip(f"missing env var {var} for prod integration test")
	return val


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_prod_judge_score_minimal_trace_expected_schema():
	"""Hit the prod judge endpoint with a minimal trace and expect a structured response.

	This test is expected to FAIL until the prod backend `/api/judge/v1/score` is deployed.
	Set env:
	- SYNTH_BACKEND_BASE_URL=https://agent-learning.onrender.com/api
	- SYNTH_API_KEY=...
	"""
	base = _require_env("SYNTH_BACKEND_BASE_URL")
	key = _require_env("SYNTH_API_KEY")

	client = JudgeClient(base_url=base, api_key=key, timeout=30.0)
	trace: dict[str, Any] = {"session_id": "it-prod-smoke", "event_history": [], "markov_blanket_message_history": []}
	# Expectation: when backend is live, this returns a JSON dict with event/outcome payloads
	resp = await client.score(
		trace=trace,
		policy_name="crafter-react",
		task_app_id="grpo-crafter",
		options={"event": True, "outcome": True, "provider": "groq", "model": "qwen/Qwen3-32B"},
	)
	# Strict checks we intend to satisfy once deployed
	assert isinstance(resp, dict)
	assert resp.get("status") == "ok"
	assert "details" in resp
	assert "outcome_reward" in resp or "event_rewards" in resp


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_prod_judge_score_invalid_key_401():
	"""With an invalid API key, prod should return an auth error (401/403).

	This test is also expected to FAIL until the route is deployed and auth enforced.
	Set env:
	- SYNTH_BACKEND_BASE_URL=https://agent-learning.onrender.com/api
	"""
	base = _require_env("SYNTH_BACKEND_BASE_URL")
	client = JudgeClient(base_url=base, api_key="invalid", timeout=15.0)
	trace = {"session_id": "it-auth", "event_history": [], "markov_blanket_message_history": []}
	with pytest.raises(PermissionError):
		await client.score(
			trace=trace,
			policy_name="crafter-react",
			task_app_id="grpo-crafter",
			options={"event": True, "outcome": True, "provider": "groq"},
		)


