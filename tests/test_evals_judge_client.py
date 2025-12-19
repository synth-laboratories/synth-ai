"""Legacy judge tests now target verifier graph completions."""

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import numpy as np
import pytest

from synth_ai.core.http import HTTPError

try:
	from synth_ai.sdk.graphs import VerifierClient  # type: ignore
except Exception:  # pragma: no cover - allow import to fail before impl
	VerifierClient = None  # type: ignore


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
		raise HTTPError(status=status, url=path, message="request_failed", body_snippet=str(payload)[:200], detail=payload)


def _build_ok_response() -> dict[str, Any]:
	return {
		"output": {
			"status": "ok",
			"event_rewards": [{"event_index": 0, "reward_value": 0.71, "criteria": {"k": {"score": 0.7}}}],
			"outcome_reward": {"reward_value": 0.82, "criteria": {"k": {"score": 0.8}}},
			"details": {"used_rubric": "bundle@v1", "policy": "p"},
		}
	}


@pytest.mark.asyncio
async def test_verifier_happy_path(monkeypatch):
	if VerifierClient is None:
		pytest.skip("VerifierClient not yet implemented")

	dummy = DummyHttp([(200, _build_ok_response())])
	monkeypatch.setattr("synth_ai.sdk.graphs.completions.AsyncHttpClient", lambda base, key, timeout=60.0: dummy)

	client = VerifierClient(base_url="https://backend/api", api_key="k")
	out = await client.evaluate(
		session_trace={"session_id": "s", "event_history": []},
		rubric={"event": [], "outcome": []},
		policy_name="p",
		task_app_id="env",
		options={"event": True, "outcome": True, "provider": "groq", "model": "qwen/Qwen3-32B"},
		job_id="zero_shot_verifier_single",
	)
	assert out.get("output", {}).get("status") == "ok"
	call = dummy.calls[0]
	assert call["path"] == "/api/graphs/completions"
	assert call["json"]["job_id"] == "zero_shot_verifier_single"
	assert call["json"]["input"]["policy_name"] == "p"
	assert call["json"]["input"]["task_app"]["id"] == "env"
	assert call["json"]["input"]["options"]["provider"] == "groq"


@pytest.mark.asyncio
@pytest.mark.fast
async def test_verifier_validation_error_422(monkeypatch):
	if VerifierClient is None:
		pytest.skip("VerifierClient not yet implemented")

	detail = {"error": "invalid_trace"}
	dummy = DummyHttp([(422, detail)])
	monkeypatch.setattr("synth_ai.sdk.graphs.completions.AsyncHttpClient", lambda base, key, timeout=60.0: dummy)

	client = VerifierClient(base_url="https://backend/api", api_key="k")
	with pytest.raises(ValueError):
		await client.evaluate(
			session_trace={},
			rubric={"event": [], "outcome": []},
			policy_name="p",
			task_app_id="env",
			options={"event": True, "outcome": True, "provider": "groq"},
			job_id="zero_shot_verifier_single",
		)


@pytest.mark.asyncio
async def test_verifier_rate_limit_429(monkeypatch):
	if VerifierClient is None:
		pytest.skip("VerifierClient not yet implemented")

	detail = {"error": "too_many_requests"}
	dummy = DummyHttp([(429, detail)])
	monkeypatch.setattr("synth_ai.sdk.graphs.completions.AsyncHttpClient", lambda base, key, timeout=60.0: dummy)

	client = VerifierClient(base_url="https://backend/api", api_key="k")
	with pytest.raises(Exception):
		await client.evaluate(
			session_trace={},
			rubric={"event": [], "outcome": []},
			policy_name="p",
			task_app_id="env",
			options={"event": True, "outcome": True, "provider": "groq"},
			job_id="zero_shot_verifier_single",
		)


@pytest.mark.asyncio
async def test_verifier_server_error_5xx(monkeypatch):
	if VerifierClient is None:
		pytest.skip("VerifierClient not yet implemented")

	detail = {"error": "upstream_failed"}
	dummy = DummyHttp([(502, detail)])
	monkeypatch.setattr("synth_ai.sdk.graphs.completions.AsyncHttpClient", lambda base, key, timeout=60.0: dummy)

	client = VerifierClient(base_url="https://backend/api", api_key="k")
	with pytest.raises(Exception):
		await client.evaluate(
			session_trace={},
			rubric={"event": [], "outcome": []},
			policy_name="p",
			task_app_id="env",
			options={"event": True, "outcome": True, "provider": "groq"},
			job_id="zero_shot_verifier_single",
		)


@pytest.mark.asyncio
async def test_trace_normalization(monkeypatch):
	if VerifierClient is None:
		pytest.skip("VerifierClient not yet implemented")

	dummy = DummyHttp([(200, _build_ok_response())])
	monkeypatch.setattr("synth_ai.sdk.graphs.completions.AsyncHttpClient", lambda base, key, timeout=60.0: dummy)

	client = VerifierClient(base_url="https://backend/api", api_key="k")
	await client.evaluate(
		session_trace={"session_id": "s", "blob": b"abc"},
		rubric={"event": [], "outcome": []},
		policy_name="p",
		task_app_id="env",
		options={"event": True, "outcome": True, "provider": "groq", "model": "qwen/Qwen3-32B"},
		job_id="zero_shot_verifier_single",
	)
	body = dummy.calls[0]["json"]["input"]["session_trace"]
	assert isinstance(body["blob"], str)


@pytest.mark.asyncio
async def test_verifier_auth_error_401(monkeypatch):
	if VerifierClient is None:
		pytest.skip("VerifierClient not yet implemented")

	detail = {"error": "auth_failed"}
	dummy = DummyHttp([(401, detail)])
	monkeypatch.setattr("synth_ai.sdk.graphs.completions.AsyncHttpClient", lambda base, key, timeout=60.0: dummy)

	client = VerifierClient(base_url="https://backend/api", api_key="k")
	with pytest.raises(PermissionError):
		await client.evaluate(
			session_trace={},
			rubric={"event": [], "outcome": []},
			policy_name="p",
			task_app_id="env",
			options={"event": True, "outcome": True, "provider": "groq"},
			job_id="zero_shot_verifier_single",
		)


@pytest.mark.asyncio
async def test_verifier_not_found_404(monkeypatch):
	if VerifierClient is None:
		pytest.skip("VerifierClient not yet implemented")

	detail = {"error": "route_missing"}
	dummy = DummyHttp([(404, detail)])
	monkeypatch.setattr("synth_ai.sdk.graphs.completions.AsyncHttpClient", lambda base, key, timeout=60.0: dummy)

	client = VerifierClient(base_url="https://backend/api", api_key="k")
	with pytest.raises(FileNotFoundError):
		await client.evaluate(
			session_trace={},
			rubric={"event": [], "outcome": []},
			policy_name="p",
			task_app_id="env",
			options={"event": True, "outcome": True, "provider": "groq"},
			job_id="zero_shot_verifier_single",
		)


@pytest.mark.asyncio
async def test_invalid_response_shape_raises(monkeypatch):
	if VerifierClient is None:
		pytest.skip("VerifierClient not yet implemented")

	dummy = DummyHttp([(200, "ok")])
	monkeypatch.setattr("synth_ai.sdk.graphs.completions.AsyncHttpClient", lambda base, key, timeout=60.0: dummy)

	client = VerifierClient(base_url="https://backend/api", api_key="k")
	with pytest.raises(ValueError):
		await client.evaluate(
			session_trace={},
			rubric={"event": [], "outcome": []},
			policy_name="p",
			task_app_id="env",
			options={"event": True, "outcome": True, "provider": "groq"},
			job_id="zero_shot_verifier_single",
		)


@pytest.mark.asyncio
async def test_includes_task_app_base_url(monkeypatch):
	if VerifierClient is None:
		pytest.skip("VerifierClient not yet implemented")

	dummy = DummyHttp([(200, _build_ok_response())])
	monkeypatch.setattr("synth_ai.sdk.graphs.completions.AsyncHttpClient", lambda base, key, timeout=60.0: dummy)

	client = VerifierClient(base_url="https://backend/api", api_key="k")
	await client.evaluate(
		session_trace={},
		rubric={"event": [], "outcome": []},
		policy_name="p",
		task_app_id="env",
		task_app_base_url="https://task.app",
		options={"event": True, "outcome": True, "provider": "groq"},
		job_id="zero_shot_verifier_single",
	)
	call = dummy.calls[0]
	assert call["json"]["input"]["task_app"]["base_url"] == "https://task.app"


@pytest.mark.asyncio
async def test_trace_normalization_numpy_decimal_datetime(monkeypatch):
	if VerifierClient is None:
		pytest.skip("VerifierClient not yet implemented")

	dummy = DummyHttp([(200, _build_ok_response())])
	monkeypatch.setattr("synth_ai.sdk.graphs.completions.AsyncHttpClient", lambda base, key, timeout=60.0: dummy)

	now = datetime.now(UTC)
	trace = {
		"session_id": "s",
		"arr": np.array([1, 2, 3]),
		"ts": now,
		"price": Decimal("3.14"),
	}
	client = VerifierClient(base_url="https://backend/api", api_key="k")
	await client.evaluate(
		session_trace=trace,
		rubric={"event": [], "outcome": []},
		policy_name="p",
		task_app_id="env",
		options={"event": True, "outcome": True, "provider": "groq"},
		job_id="zero_shot_verifier_single",
	)
	body = dummy.calls[0]["json"]["input"]["session_trace"]
	assert body["arr"] == [1, 2, 3]
	assert isinstance(body["ts"], str)
	assert body["price"] == pytest.approx(3.14)


@pytest.mark.asyncio
@pytest.mark.parametrize("event,outcome", [(True, False), (False, True), (True, True), (False, False)])
@pytest.mark.parametrize("provider", ["groq", "gemini"])
@pytest.mark.parametrize("rubric_mode", ["overrides", "both"])
@pytest.mark.parametrize("extended_resp", [False, True])
async def test_verifier_rubric_combinations(
	monkeypatch,
	event: bool,
	outcome: bool,
	provider: str,
	rubric_mode: str,
	extended_resp: bool,
):
	if VerifierClient is None:
		pytest.skip("VerifierClient not yet implemented")

	if not event and not outcome:
		responses = [(422, {"error": "event_or_outcome_required"})]
	else:
		base = {
			"status": "ok",
			"event_rewards": [] if event else None,
			"outcome_reward": {"reward_value": 0.5} if outcome else None,
			"details": {"used_rubric": "bundle@v1", "policy": "p"},
		}
		base = {k: v for k, v in base.items() if v is not None}
		if extended_resp:
			base["judgements"] = [
				{"judgement": {"key": "progress.unique", "score": 1.0}, "scope": "event", "turn": 1}
			]
			base["aggregates"] = {"progress": {"mean": 0.9, "median": 0.9, "std": 0.0, "n": 1}}
			base["tree"] = {"progress": {"key": "progress", "children": []}}
		responses = [(200, {"output": base})]

	dummy = DummyHttp(responses)
	monkeypatch.setattr("synth_ai.sdk.graphs.completions.AsyncHttpClient", lambda base, key, timeout=60.0: dummy)

	options: dict[str, Any] = {"event": event, "outcome": outcome, "provider": provider}
	rubric: dict[str, Any] = {"event": [], "outcome": []}
	if rubric_mode in ("overrides", "both"):
		rubric = {
			"event": [
				{"id": "process.protocol_adherence", "weight": 0.2, "scale": "binary"},
				{"id": "progress.unique_achievements", "weight": 0.8, "scale": "bounded"},
			],
			"outcome": [{"id": "outcome.goal", "weight": 1.0, "scale": "binary"}],
		}
	options["tracks"] = ["process", "reasoning", "progress", "outcome"]
	options["weights"] = {"outcome": 0.5, "progress": 0.3, "reasoning": 0.15, "process": 0.05}

	client = VerifierClient(base_url="https://backend/api", api_key="k")
	if not event and not outcome:
		with pytest.raises(ValueError):
			await client.evaluate(
				session_trace={"session_id": "s"},
				rubric=rubric,
				policy_name="p",
				task_app_id="env",
				options=options,
				job_id="zero_shot_verifier_single",
			)
		return

	result = await client.evaluate(
		session_trace={"session_id": "s"},
		rubric=rubric,
		policy_name="p",
		task_app_id="env",
		options=options,
		job_id="zero_shot_verifier_single",
	)
	call = dummy.calls[0]
	body = call["json"]["input"]
	assert body["options"]["event"] is event
	assert body["options"]["outcome"] is outcome
	assert body["options"]["provider"] == provider
	assert body["options"]["tracks"] == ["process", "reasoning", "progress", "outcome"]
	assert body["options"]["weights"]["outcome"] == pytest.approx(0.5)
	if extended_resp:
		output = result.get("output", {})
		assert "judgements" in output and "aggregates" in output and "tree" in output
