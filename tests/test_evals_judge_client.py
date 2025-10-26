import asyncio
import json
from typing import Any
from datetime import datetime, UTC
from decimal import Decimal
import numpy as np

import pytest

from synth_ai.http_client import HTTPError

# Placeholder import path; the implementation will be added to synth_ai/evals/client.py
# Tests focus on request/response contracts and error mapping, to be satisfied once implemented.
try:
	from synth_ai.evals.client import JudgeClient  # type: ignore
except Exception:  # pragma: no cover - allow import to fail before impl
	JudgeClient = None  # type: ignore


class DummyHttp:
	"""Simple dummy to monkeypatch AsyncHttpClient context and return canned responses."""

	def __init__(self, responses: list[tuple[int, dict[str, Any] | str]]):
		self._responses = responses
		self.calls: list[dict[str, Any]] = []

	async def __aenter__(self):
		return self

	async def __aexit__(self, exc_type, exc, tb):  # noqa: ANN001
		return None

	async def post_json(self, path: str, *, json: dict[str, Any], headers: dict[str, str] | None = None):
		self.calls.append({"path": path, "json": json, "headers": headers})
		status, payload = self._responses.pop(0)
		if 200 <= status < 300:
			return payload
		# Simulate HTTPError raise contract
		raise HTTPError(status=status, url=path, message="request_failed", body_snippet=str(payload)[:200], detail=payload)


@pytest.mark.asyncio
async def test_score_happy_path(monkeypatch):
	if JudgeClient is None:
		pytest.skip("JudgeClient not yet implemented")

	resp = {
		"status": "ok",
		"event_rewards": [{"turn": 1, "scores": {"k": 0.7}, "aggregate": {"score": 0.71}}],
		"outcome_reward": {"scores": {"k": 0.8}, "aggregate": {"score": 0.82}},
		"details": {"used_rubric": "bundle@v1", "policy": "p"},
	}
	dummy = DummyHttp([(200, resp)])
	monkeypatch.setattr("synth_ai.evals.client.AsyncHttpClient", lambda base, key, timeout=60.0: dummy)

	client = JudgeClient(base_url="https://backend/api", api_key="k")
	out = await client.score(
		trace={"session_id": "s", "event_history": []},
		policy_name="p",
		task_app_id="env",
		options={"event": True, "outcome": True, "provider": "groq", "model": "qwen/Qwen3-32B"},
	)
	assert out.get("status") == "ok"
	call = dummy.calls[0]
	assert call["path"] == "/api/judge/v1/score"
	assert call["json"]["policy_name"] == "p"
	assert call["json"]["task_app"]["id"] == "env"
	assert call["json"]["options"]["provider"] == "groq"


@pytest.mark.asyncio
@pytest.mark.fast
async def test_score_validation_error_422(monkeypatch):
	if JudgeClient is None:
		pytest.skip("JudgeClient not yet implemented")

	detail = {"error": "invalid_trace"}
	dummy = DummyHttp([(422, detail)])
	monkeypatch.setattr("synth_ai.evals.client.AsyncHttpClient", lambda base, key, timeout=60.0: dummy)

	client = JudgeClient(base_url="https://backend/api", api_key="k")
	with pytest.raises(ValueError):
		await client.score(
			trace={}, policy_name="p", task_app_id="env", options={"event": True, "outcome": True, "provider": "groq"}
		)


@pytest.mark.asyncio
async def test_score_rate_limit_429(monkeypatch):
	if JudgeClient is None:
		pytest.skip("JudgeClient not yet implemented")

	detail = {"error": "too_many_requests"}
	dummy = DummyHttp([(429, detail)])
	monkeypatch.setattr("synth_ai.evals.client.AsyncHttpClient", lambda base, key, timeout=60.0: dummy)

	client = JudgeClient(base_url="https://backend/api", api_key="k")
	with pytest.raises(Exception):  # replace with RetryLater once implemented
		await client.score(
			trace={}, policy_name="p", task_app_id="env", options={"event": True, "outcome": True, "provider": "groq"}
		)


@pytest.mark.asyncio
async def test_score_server_error_5xx(monkeypatch):
	if JudgeClient is None:
		pytest.skip("JudgeClient not yet implemented")

	detail = {"error": "upstream_failed"}
	dummy = DummyHttp([(502, detail)])
	monkeypatch.setattr("synth_ai.evals.client.AsyncHttpClient", lambda base, key, timeout=60.0: dummy)

	client = JudgeClient(base_url="https://backend/api", api_key="k")
	with pytest.raises(Exception):  # replace with TransientError once implemented
		await client.score(
			trace={}, policy_name="p", task_app_id="env", options={"event": True, "outcome": True, "provider": "groq"}
		)


@pytest.mark.asyncio
async def test_trace_normalization(monkeypatch):
	if JudgeClient is None:
		pytest.skip("JudgeClient not yet implemented")

	# includes a non-JSON type (bytes) to ensure normalization path is used
	resp = {"status": "ok", "event_rewards": [], "outcome_reward": {"scores": {}, "aggregate": {"score": 0.0}}, "details": {}}
	dummy = DummyHttp([(200, resp)])
	monkeypatch.setattr("synth_ai.evals.client.AsyncHttpClient", lambda base, key, timeout=60.0: dummy)

	client = JudgeClient(base_url="https://backend/api", api_key="k")
	out = await client.score(
		trace={"session_id": "s", "blob": b"abc"},
		policy_name="p",
		task_app_id="env",
		options={"event": True, "outcome": True, "provider": "groq", "model": "qwen/Qwen3-32B"},
	)
	call = dummy.calls[0]
	body = call["json"]
	assert isinstance(body["trace"]["blob"], str)  # base64 encoded


@pytest.mark.asyncio
async def test_score_auth_error_401(monkeypatch):
	if JudgeClient is None:
		pytest.skip("JudgeClient not yet implemented")

	detail = {"error": "auth_failed"}
	dummy = DummyHttp([(401, detail)])
	monkeypatch.setattr("synth_ai.evals.client.AsyncHttpClient", lambda base, key, timeout=60.0: dummy)

	client = JudgeClient(base_url="https://backend/api", api_key="k")
	with pytest.raises(PermissionError):
		await client.score(
			trace={}, policy_name="p", task_app_id="env", options={"event": True, "outcome": True, "provider": "groq"}
		)


@pytest.mark.asyncio
async def test_score_not_found_404(monkeypatch):
	if JudgeClient is None:
		pytest.skip("JudgeClient not yet implemented")

	detail = {"error": "route_missing"}
	dummy = DummyHttp([(404, detail)])
	monkeypatch.setattr("synth_ai.evals.client.AsyncHttpClient", lambda base, key, timeout=60.0: dummy)

	client = JudgeClient(base_url="https://backend/api", api_key="k")
	with pytest.raises(FileNotFoundError):
		await client.score(
			trace={}, policy_name="p", task_app_id="env", options={"event": True, "outcome": True, "provider": "groq"}
		)


@pytest.mark.asyncio
async def test_invalid_response_shape_raises(monkeypatch):
	if JudgeClient is None:
		pytest.skip("JudgeClient not yet implemented")

	# Return a non-dict payload to trigger shape validation
	dummy = DummyHttp([(200, "ok")])
	monkeypatch.setattr("synth_ai.evals.client.AsyncHttpClient", lambda base, key, timeout=60.0: dummy)

	client = JudgeClient(base_url="https://backend/api", api_key="k")
	with pytest.raises(ValueError):
		await client.score(
			trace={}, policy_name="p", task_app_id="env", options={"event": True, "outcome": True, "provider": "groq"}
		)


@pytest.mark.asyncio
async def test_includes_task_app_base_url(monkeypatch):
	if JudgeClient is None:
		pytest.skip("JudgeClient not yet implemented")

	resp = {"status": "ok", "event_rewards": [], "outcome_reward": {"scores": {}, "aggregate": {"score": 0.0}}, "details": {}}
	dummy = DummyHttp([(200, resp)])
	monkeypatch.setattr("synth_ai.evals.client.AsyncHttpClient", lambda base, key, timeout=60.0: dummy)

	client = JudgeClient(base_url="https://backend/api", api_key="k")
	await client.score(
		trace={}, policy_name="p", task_app_id="env", task_app_base_url="https://task.app", options={"event": True, "outcome": True, "provider": "groq"}
	)
	call = dummy.calls[0]
	assert call["json"]["task_app"]["base_url"] == "https://task.app"


@pytest.mark.asyncio
async def test_trace_normalization_numpy_decimal_datetime(monkeypatch):
	if JudgeClient is None:
		pytest.skip("JudgeClient not yet implemented")

	resp = {"status": "ok", "event_rewards": [], "outcome_reward": {"scores": {}, "aggregate": {"score": 0.0}}, "details": {}}
	dummy = DummyHttp([(200, resp)])
	monkeypatch.setattr("synth_ai.evals.client.AsyncHttpClient", lambda base, key, timeout=60.0: dummy)

	now = datetime.now(UTC)
	trace = {
		"session_id": "s",
		"arr": np.array([1, 2, 3]),
		"ts": now,
		"price": Decimal("3.14"),
	}
	client = JudgeClient(base_url="https://backend/api", api_key="k")
	await client.score(
		trace=trace, policy_name="p", task_app_id="env", options={"event": True, "outcome": True, "provider": "groq"}
	)
	body = dummy.calls[0]["json"]["trace"]
	assert body["arr"] == [1, 2, 3]
	assert isinstance(body["ts"], str)
	assert body["price"] == pytest.approx(3.14)


@pytest.mark.asyncio
@pytest.mark.parametrize("event,outcome", [(True, False), (False, True), (True, True), (False, False)])
@pytest.mark.parametrize("provider", ["groq", "gemini"])
@pytest.mark.parametrize("rubric_mode", ["id", "overrides", "both"])  # how we specify rubrics
@pytest.mark.parametrize("extended_resp", [False, True])
async def test_score_rubric_combinations(monkeypatch, event: bool, outcome: bool, provider: str, rubric_mode: str, extended_resp: bool):
	if JudgeClient is None:
		pytest.skip("JudgeClient not yet implemented")

	# Build response depending on validity and whether extended fields are included
	if not event and not outcome:
		responses = [(422, {"error": "event_or_outcome_required"})]
	else:
		base = {
			"status": "ok",
			"event_rewards": [] if event else None,
			"outcome_reward": {"scores": {}, "aggregate": {"score": 0.5}} if outcome else None,
			"details": {"used_rubric": "bundle@v1", "policy": "p"},
		}
		# Remove None values to simulate server not returning irrelevant fields
		base = {k: v for k, v in base.items() if v is not None}
		if extended_resp:
			base["judgements"] = [
				{"judgement": {"key": "progress.unique", "score": 1.0}, "scope": "event", "turn": 1}
			]
			base["aggregates"] = {"progress": {"mean": 0.9, "median": 0.9, "std": 0.0, "n": 1}}
			base["tree"] = {"progress": {"key": "progress", "children": []}}
		responses = [(200, base)]

	dummy = DummyHttp(responses)
	monkeypatch.setattr("synth_ai.evals.client.AsyncHttpClient", lambda base, key, timeout=60.0: dummy)

	# Build options according to rubric_mode
	options: dict[str, Any] = {"event": event, "outcome": outcome, "provider": provider}
	if rubric_mode in ("id", "both"):
		options["rubric_id"] = "env/bundle@v1"
	if rubric_mode in ("overrides", "both"):
		options["rubric_overrides"] = {
			"event": {
				"criteria": [
					{"id": "process.protocol_adherence", "weight": 0.2, "scale": "binary"},
					{"id": "progress.unique_achievements", "weight": 0.8, "scale": "bounded"},
				],
				"tree": {"key": "root", "aggregation": "weighted_mean", "children": []},
			},
			"outcome": {"criteria": [{"id": "outcome.goal", "weight": 1.0, "scale": "binary"}]},
		}
	# Also pass through tracks/weights if present in some configs; SDK should forward transparently
	options["tracks"] = ["process", "reasoning", "progress", "outcome"]
	options["weights"] = {"outcome": 0.5, "progress": 0.3, "reasoning": 0.15, "process": 0.05}

	client = JudgeClient(base_url="https://backend/api", api_key="k")
	if not event and not outcome:
		with pytest.raises(ValueError):
			await client.score(trace={"session_id": "s"}, policy_name="p", task_app_id="env", options=options)  # type: ignore[arg-type]
		return

	result = await client.score(trace={"session_id": "s"}, policy_name="p", task_app_id="env", options=options)  # type: ignore[arg-type]
	call = dummy.calls[0]
	body = call["json"]
	# Verify options forwarding
	assert body["options"]["event"] is event
	assert body["options"]["outcome"] is outcome
	assert body["options"]["provider"] == provider
	if rubric_mode in ("id", "both"):
		assert body["options"]["rubric_id"] == "env/bundle@v1"
	if rubric_mode in ("overrides", "both"):
		assert "rubric_overrides" in body["options"]
	# Check passthrough of tracks/weights
	assert body["options"]["tracks"] == ["process", "reasoning", "progress", "outcome"]
	assert body["options"]["weights"]["outcome"] == pytest.approx(0.5)
	# Validate extended fields are accepted when present
	if extended_resp:
		assert "judgements" in result and "aggregates" in result and "tree" in result
