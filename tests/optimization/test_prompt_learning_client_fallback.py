from __future__ import annotations

import asyncio
from typing import Any

from synth_ai.sdk.optimization.internal.learning import prompt_learning_client as plc
from synth_ai.sdk.optimization.internal.learning.prompt_learning_client import (
    PromptLearningClient,
    _extract_mipro_state_fields,
    _extract_prompt_learning_fields_from_job_payload,
    _merge_prompt_results_from_job_payload,
)
from synth_ai.sdk.optimization.internal.learning.prompt_learning_types import PromptResults


def test_extract_prompt_learning_fields_from_job_payload_prefers_job_metadata() -> None:
    payload = {
        "best_prompt": {"messages": [{"role": "system", "content": "fallback"}]},
        "metadata": {
            "best_score": "0.20",
            "sensor_frames": [{"frame_id": "frame_meta"}],
        },
        "job_metadata": {
            "best_score": "0.91",
            "lever_summary": {
                "prompt_lever_id": "mipro.prompt.pl_123",
                "candidate_lever_versions": {"candidate_1": "7"},
                "best_candidate_id": "candidate_1",
            },
            "sensor_frames": [{"frame_id": "frame_job"}],
        },
    }

    fields = _extract_prompt_learning_fields_from_job_payload(payload)

    assert fields["best_candidate"] == payload["best_prompt"]
    assert fields["best_reward"] == 0.91
    assert fields["sensor_frames"] == [{"frame_id": "frame_job"}]
    assert fields["lever_versions"] == {"mipro.prompt.pl_123": 7}
    assert fields["best_lever_version"] == 7


def test_merge_prompt_results_from_job_payload_preserves_existing_fields() -> None:
    result = PromptResults(
        best_candidate={"existing": True},
        best_reward=0.55,
        lever_versions={"mipro.prompt.existing": 2},
    )
    payload = {
        "best_candidate": {"replacement": True},
        "best_score": 0.99,
        "lever_summary": {"prompt_lever_id": "mipro.prompt.other"},
        "sensor_frames": [{"frame_id": "frame_1"}],
        "lever_versions": {"mipro.prompt.other": "9"},
        "best_lever_version": "9",
    }

    _merge_prompt_results_from_job_payload(result, payload)

    assert result.best_candidate == {"existing": True}
    assert result.best_reward == 0.55
    assert result.lever_versions == {"mipro.prompt.existing": 2}
    assert result.lever_summary == {"prompt_lever_id": "mipro.prompt.other"}
    assert result.sensor_frames == [{"frame_id": "frame_1"}]
    assert result.best_lever_version == 9


def test_get_prompts_falls_back_to_job_payload_when_events_unavailable(monkeypatch) -> None:
    async def _raise_events(self, job_id: str, *, since_seq: int = 0, limit: int = 5000):
        _ = (self, job_id, since_seq, limit)
        raise RuntimeError("404 Not Found")

    async def _job_payload(self, job_id: str):
        _ = (self, job_id)
        return {
            "status": "succeeded",
            "best_prompt": {"messages": [{"role": "system", "content": "from status"}]},
            "best_score": 0.77,
            "lever_summary": {
                "prompt_lever_id": "mipro.prompt.pl_abc",
                "candidate_lever_versions": {"candidate_best": 4},
                "best_candidate_id": "candidate_best",
            },
            "sensor_frames": [{"frame_id": "frame_1"}],
        }

    monkeypatch.setattr(PromptLearningClient, "get_events", _raise_events)
    monkeypatch.setattr(PromptLearningClient, "get_job", _job_payload)

    client = PromptLearningClient(base_url="http://example.com", api_key="key")
    result = asyncio.run(client.get_prompts("pl_fallback"))

    assert result.best_candidate == {"messages": [{"role": "system", "content": "from status"}]}
    assert result.best_reward == 0.77
    assert result.lever_versions == {"mipro.prompt.pl_abc": 4}
    assert result.best_lever_version == 4
    assert result.sensor_frames == [{"frame_id": "frame_1"}]


def test_get_job_merges_canonical_and_legacy_payloads(monkeypatch) -> None:
    class _FakeHttpClient:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            return

        async def __aenter__(self) -> "_FakeHttpClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            _ = (exc_type, exc, tb)

        async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
            _ = params
            if path.startswith("/api/jobs/") and "/events" not in path:
                return {"job_id": "pl_merge", "status": "succeeded", "job_metadata": {"x": 1}}
            if path.startswith("/api/policy-optimization/online/jobs/"):
                return {
                    "job_id": "pl_merge",
                    "status": "succeeded",
                    "best_reward": 0.73,
                    "metadata": {"best_prompt": {"messages": [{"role": "system", "content": "ok"}]}},
                }
            raise RuntimeError("404 Not Found")

    monkeypatch.setattr(plc, "RustCoreHttpClient", _FakeHttpClient)

    client = PromptLearningClient(base_url="http://example.com", api_key="key")
    payload = asyncio.run(client.get_job("pl_merge"))
    assert payload["status"] == "succeeded"
    assert payload["best_reward"] == 0.73
    assert payload["best_score"] == 0.73
    assert payload["best_candidate"] == {"messages": [{"role": "system", "content": "ok"}]}
    assert payload["job_metadata"]["x"] == 1


def test_get_events_prefers_richest_endpoint(monkeypatch) -> None:
    class _FakeHttpClient:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            return

        async def __aenter__(self) -> "_FakeHttpClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            _ = (exc_type, exc, tb)

        async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
            _ = params
            if path.startswith("/api/jobs/") and path.endswith("/events"):
                return {"events": [{"type": "one"}]}
            if path.startswith("/api/policy-optimization/online/jobs/") and path.endswith("/events"):
                return {"events": [{"type": "one"}, {"type": "two"}]}
            if path.startswith("/api/prompt-learning/online/jobs/") and path.endswith("/events"):
                return {"events": []}
            raise RuntimeError("404 Not Found")

    monkeypatch.setattr(plc, "RustCoreHttpClient", _FakeHttpClient)

    client = PromptLearningClient(base_url="http://example.com", api_key="key")
    events = asyncio.run(client.get_events("pl_events", limit=100))
    assert [event.get("type") for event in events] == ["one", "two"]


def test_get_job_mipro_falls_back_to_system_state(monkeypatch) -> None:
    class _FakeHttpClient:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            return

        async def __aenter__(self) -> "_FakeHttpClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            _ = (exc_type, exc, tb)

        async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
            _ = params
            if path == "/api/jobs/pl_state":
                return {
                    "job_id": "pl_state",
                    "status": "succeeded",
                    "metadata": {
                        "algorithm": "mipro",
                        "mipro_system_id": "sys_123",
                    },
                }
            if path == "/api/prompt-learning/online/mipro/systems/sys_123/state":
                return {
                    "best_candidate_id": "cand_a",
                    "best_score": 0.81,
                    "candidates": {
                        "cand_a": {"candidate_id": "cand_a", "avg_reward": 0.81},
                    },
                    "attempted_candidates": [{"candidate_id": "cand_a", "score": 0.81}],
                    "optimized_candidates": [{"candidate_id": "cand_a", "score": 0.81}],
                }
            raise RuntimeError("404 Not Found")

    monkeypatch.setattr(plc, "RustCoreHttpClient", _FakeHttpClient)

    client = PromptLearningClient(base_url="http://example.com", api_key="key")
    payload = asyncio.run(client.get_job("pl_state"))

    assert payload["best_reward"] == 0.81
    assert payload["best_score"] == 0.81
    assert payload["best_candidate"] == {"candidate_id": "cand_a", "avg_reward": 0.81}
    assert payload["attempted_candidates"] == [{"candidate_id": "cand_a", "score": 0.81}]
    assert payload["optimized_candidates"] == [{"candidate_id": "cand_a", "score": 0.81}]


def test_get_job_mipro_state_infers_best_candidate_without_best_fields(monkeypatch) -> None:
    class _FakeHttpClient:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            return

        async def __aenter__(self) -> "_FakeHttpClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            _ = (exc_type, exc, tb)

        async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
            _ = params
            if path == "/api/jobs/pl_infer":
                return {
                    "job_id": "pl_infer",
                    "status": "succeeded",
                    "metadata": {
                        "algorithm": "mipro",
                        "mipro_system_id": "sys_infer",
                    },
                }
            if path == "/api/prompt-learning/online/mipro/systems/sys_infer/state":
                return {
                    "candidates": {
                        "cand_low": {"candidate_id": "cand_low", "avg_reward": 0.11},
                        "cand_high": {"candidate_id": "cand_high", "avg_reward": 0.73},
                    }
                }
            raise RuntimeError("404 Not Found")

    monkeypatch.setattr(plc, "RustCoreHttpClient", _FakeHttpClient)

    client = PromptLearningClient(base_url="http://example.com", api_key="key")
    payload = asyncio.run(client.get_job("pl_infer"))

    assert payload["best_reward"] == 0.73
    assert payload["best_score"] == 0.73
    assert payload["best_candidate"] == {"candidate_id": "cand_high", "avg_reward": 0.73}


def test_extract_mipro_state_fields_normalizes_candidate_versions_to_prompt_lever() -> None:
    fields = _extract_mipro_state_fields(
        {
            "best_candidate_id": "cand_best",
            "prompt_lever_id": "mipro.prompt.sys_123",
            "candidate_lever_versions": {"cand_best": 11, "cand_other": 4},
            "lever_summary": {
                "prompt_lever_id": "mipro.prompt.sys_123",
                "candidate_lever_versions": {"cand_best": 11, "cand_other": 4},
                "best_candidate_id": "cand_best",
            },
            "candidates": {
                "cand_best": {"candidate_id": "cand_best", "avg_reward": 0.88},
            },
        }
    )

    assert fields["lever_versions"] == {"mipro.prompt.sys_123": 11}
    assert fields["best_lever_version"] == 11


def test_extract_prompt_learning_fields_corrects_candidate_mapped_lever_versions() -> None:
    fields = _extract_prompt_learning_fields_from_job_payload(
        {
            "metadata": {
                "lever_versions": {"cand_best": 9},
                "lever_summary": {
                    "prompt_lever_id": "mipro.prompt.sys_987",
                    "candidate_lever_versions": {"cand_best": 9},
                    "best_candidate_id": "cand_best",
                },
            }
        }
    )
    assert fields["lever_versions"] == {"mipro.prompt.sys_987": 9}
    assert fields["best_lever_version"] == 9


def test_get_events_falls_back_to_mipro_system_events(monkeypatch) -> None:
    class _FakeHttpClient:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            return

        async def __aenter__(self) -> "_FakeHttpClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            _ = (exc_type, exc, tb)

        async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
            _ = params
            if path.startswith("/api/jobs/pl_sparse") and path.endswith("/events"):
                return {"events": [{"type": "prompt.learning.created"}]}
            if path == "/api/jobs/pl_sparse":
                return {
                    "job_id": "pl_sparse",
                    "status": "succeeded",
                    "metadata": {
                        "algorithm": "mipro",
                        "mipro_system_id": "sys_sparse",
                    },
                }
            if path == "/api/prompt-learning/online/mipro/systems/sys_sparse/events":
                return {
                    "events": [
                        {
                            "seq": 1,
                            "event_type": "mipro.rollout.complete",
                            "message": "done",
                            "data": {"reward": 0.5},
                        },
                        {
                            "seq": 2,
                            "event_type": "mipro.reward.received",
                            "message": "reward",
                            "data": {"reward": 0.7},
                        },
                    ]
                }
            if path.startswith("/api/policy-optimization/online/jobs/") and path.endswith("/events"):
                return {"events": []}
            if path.startswith("/api/prompt-learning/online/jobs/") and path.endswith("/events"):
                return {"events": []}
            raise RuntimeError("404 Not Found")

    monkeypatch.setattr(plc, "RustCoreHttpClient", _FakeHttpClient)

    client = PromptLearningClient(base_url="http://example.com", api_key="key")
    events = asyncio.run(client.get_events("pl_sparse", limit=100))
    assert [event.get("type") for event in events] == [
        "learning.policy.mipro.rollout.complete",
        "learning.policy.mipro.reward.received",
    ]


def test_get_prompts_falls_back_to_mipro_system_state(monkeypatch) -> None:
    class _FakeHttpClient:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            return

        async def __aenter__(self) -> "_FakeHttpClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            _ = (exc_type, exc, tb)

        async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
            _ = params
            if path.startswith("/api/jobs/pl_prompt_state") and path.endswith("/events"):
                return {"events": [{"type": "prompt.learning.created"}]}
            if path == "/api/jobs/pl_prompt_state":
                return {
                    "job_id": "pl_prompt_state",
                    "status": "succeeded",
                    "metadata": {
                        "algorithm": "mipro",
                        "mipro_system_id": "sys_prompt",
                    },
                }
            if path == "/api/prompt-learning/online/mipro/systems/sys_prompt/state":
                return {
                    "best_candidate_id": "cand_best",
                    "best_score": 0.66,
                    "candidates": {
                        "cand_best": {"candidate_id": "cand_best", "avg_reward": 0.66},
                    },
                    "attempted_candidates": [{"candidate_id": "cand_best", "score": 0.66}],
                    "optimized_candidates": [{"candidate_id": "cand_best", "score": 0.66}],
                }
            if path.startswith("/api/policy-optimization/online/jobs/") and path.endswith("/events"):
                return {"events": []}
            if path.startswith("/api/prompt-learning/online/jobs/") and path.endswith("/events"):
                return {"events": []}
            raise RuntimeError("404 Not Found")

    monkeypatch.setattr(plc, "RustCoreHttpClient", _FakeHttpClient)

    client = PromptLearningClient(base_url="http://example.com", api_key="key")
    result = asyncio.run(client.get_prompts("pl_prompt_state"))

    assert result.best_reward == 0.66
    assert result.best_candidate == {"candidate_id": "cand_best", "avg_reward": 0.66}
    assert result.attempted_candidates == [{"candidate_id": "cand_best", "score": 0.66}]
    assert result.optimized_candidates == [{"candidate_id": "cand_best", "score": 0.66}]
