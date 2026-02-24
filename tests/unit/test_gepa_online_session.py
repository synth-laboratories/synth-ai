from __future__ import annotations

from typing import Any, Dict, List

import pytest

from synth_ai.sdk.optimization.policy import gepa_online_session as module
from synth_ai.sdk.optimization.policy.gepa_online_session import GepaOnlineSession


@pytest.mark.asyncio
async def test_get_prompt_urls_async_raises_on_non_object_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyRustClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        async def __aenter__(self) -> "DummyRustClient":
            return self

        async def __aexit__(self, *args: Any) -> None:
            del args
            return None

        async def get(self, path: str, params: Any = None) -> List[str]:
            del path, params
            return ["bad-shape"]

    monkeypatch.setattr(module, "RustCoreHttpClient", DummyRustClient)
    session = GepaOnlineSession(
        session_id="session-bad-prompt",
        backend_url="https://backend.test",
        api_key="api-key",
        timeout=0.1,
    )
    with pytest.raises(ValueError, match="GEPA prompt endpoint"):
        await session.get_prompt_urls_async()


@pytest.mark.asyncio
async def test_list_candidates_async_uses_session_system_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: Dict[str, Any] = {}

    class DummyPromptLearningClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured["init"] = {"args": args, "kwargs": kwargs}

        async def list_system_candidates(self, system_id: str, **kwargs: Any) -> Dict[str, Any]:
            captured["list"] = {"system_id": system_id, "kwargs": kwargs}
            return {"items": [{"candidate_id": "cand_1"}], "system_id": system_id}

    monkeypatch.setattr(module, "PromptLearningClient", DummyPromptLearningClient)
    session = GepaOnlineSession(
        session_id="gepa-session-1",
        backend_url="https://backend.test",
        api_key="api-key",
        timeout=0.1,
    )

    async def _fake_get_async() -> Dict[str, Any]:
        return {"job_id": "pl_abc"}

    monkeypatch.setattr(session, "_get_async", _fake_get_async)
    result = await session.list_candidates_async(status="evaluated", limit=3)
    assert result["items"][0]["candidate_id"] == "cand_1"
    assert captured["list"]["system_id"] == "gepa-session-1"
    assert captured["list"]["kwargs"]["job_id"] == "pl_abc"


@pytest.mark.asyncio
async def test_get_candidate_async_prefers_job_scoped_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: Dict[str, Any] = {}

    class DummyPromptLearningClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        async def get_candidate(self, job_id: str, candidate_id: str) -> Dict[str, Any]:
            captured["get_candidate"] = {"job_id": job_id, "candidate_id": candidate_id}
            return {"candidate_id": candidate_id, "job_id": job_id}

        async def get_global_candidate(self, candidate_id: str) -> Dict[str, Any]:
            return {"candidate_id": candidate_id}

        async def list_system_candidates(self, system_id: str, **kwargs: Any) -> Dict[str, Any]:
            del system_id, kwargs
            return {"items": []}

    monkeypatch.setattr(module, "PromptLearningClient", DummyPromptLearningClient)
    session = GepaOnlineSession(
        session_id="gepa-session-2",
        backend_url="https://backend.test",
        api_key="api-key",
        timeout=0.1,
    )

    async def _fake_get_async() -> Dict[str, Any]:
        return {"job_id": "pl_scope"}

    monkeypatch.setattr(session, "_get_async", _fake_get_async)
    candidate = await session.get_candidate_async("cand_scope")
    assert candidate["job_id"] == "pl_scope"
    assert captured["get_candidate"] == {
        "job_id": "pl_scope",
        "candidate_id": "cand_scope",
    }


@pytest.mark.asyncio
async def test_list_seed_evals_async_infers_job_and_uses_system_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: Dict[str, Any] = {}

    class DummyPromptLearningClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured["init"] = {"args": args, "kwargs": kwargs}

        async def list_system_seed_evals(self, system_id: str, **kwargs: Any) -> Dict[str, Any]:
            captured["list_seed_evals"] = {"system_id": system_id, "kwargs": kwargs}
            return {"items": [{"candidate_id": "cand_1", "seed": 9}], "system_id": system_id}

    monkeypatch.setattr(module, "PromptLearningClient", DummyPromptLearningClient)
    session = GepaOnlineSession(
        session_id="gepa-session-seeds",
        backend_url="https://backend.test",
        api_key="api-key",
        timeout=0.1,
    )

    async def _fake_get_async() -> Dict[str, Any]:
        return {"job_id": "pl_gepa_seed"}

    monkeypatch.setattr(session, "_get_async", _fake_get_async)
    payload = await session.list_seed_evals_async(limit=6, success=False)
    assert payload["system_id"] == "gepa-session-seeds"
    assert captured["list_seed_evals"]["system_id"] == "gepa-session-seeds"
    assert captured["list_seed_evals"]["kwargs"]["job_id"] == "pl_gepa_seed"
    assert captured["list_seed_evals"]["kwargs"]["success"] is False


@pytest.mark.asyncio
async def test_get_candidate_async_paginates_system_candidates_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[Dict[str, Any]] = []

    class DummyPromptLearningClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        async def get_global_candidate(self, candidate_id: str) -> Dict[str, Any]:
            return {"candidate_id": candidate_id, "system_id": None}

        async def list_system_candidates(self, system_id: str, **kwargs: Any) -> Dict[str, Any]:
            calls.append({"system_id": system_id, "kwargs": kwargs})
            cursor = kwargs.get("cursor")
            if cursor is None:
                return {
                    "items": [{"candidate_id": "cand_1"}],
                    "next_cursor": "cursor-2",
                    "system_id": system_id,
                }
            return {
                "items": [{"candidate_id": "cand_target"}],
                "next_cursor": None,
                "system_id": system_id,
            }

    monkeypatch.setattr(module, "PromptLearningClient", DummyPromptLearningClient)
    session = GepaOnlineSession(
        session_id="gepa-session-pagination",
        backend_url="https://backend.test",
        api_key="api-key",
        timeout=0.1,
    )

    candidate = await session.get_candidate_async("cand_target")
    assert candidate["candidate_id"] == "cand_target"
    assert len(calls) == 2
    assert calls[0]["kwargs"]["cursor"] is None
    assert calls[1]["kwargs"]["cursor"] == "cursor-2"
