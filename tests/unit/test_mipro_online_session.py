from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from synth_ai.sdk.optimization.policy import mipro_online_session as module
from synth_ai.sdk.optimization.policy.mipro_online_session import MiproOnlineSession


def _patch_rust_client(monkeypatch: pytest.MonkeyPatch) -> List[Any]:
    clients: List[Any] = []

    class DummyRustClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            clients.append(self)
            self.args = args
            self.kwargs = kwargs
            self.requests: List[tuple[str, str, Any]] = []

        async def __aenter__(self) -> "DummyRustClient":
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def post_json(self, path: str, json: Any) -> Dict[str, Any]:
            self.requests.append(("post_json", path, json))
            return {"result": "ok", "json": json}

        async def get(self, path: str, params: Any = None) -> Dict[str, Any]:
            self.requests.append(("get", path, params))
            return {"online_url": "https://example.com"}

    monkeypatch.setattr(module, "RustCoreHttpClient", DummyRustClient)
    return clients


def test_resolve_backend_url_strips_trailing_slash() -> None:
    assert module._resolve_backend_url("https://api.synth.ai/") == "https://api.synth.ai"
    default_url = module._resolve_backend_url(None)
    assert default_url == module.BACKEND_URL_BASE.rstrip("/")


def test_resolve_api_key_prefers_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNTH_API_KEY", "env-key")
    assert module._resolve_api_key("explicit") == "explicit"
    assert module._resolve_api_key(None) == "env-key"


def test_resolve_api_key_missing_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SYNTH_API_KEY", raising=False)
    with pytest.raises(ValueError):
        module._resolve_api_key(None)


def test_build_session_payload_conflict_raises() -> None:
    with pytest.raises(ValueError):
        module._build_session_payload(
            config={},
            config_name="config",
            config_path=None,
            config_body=None,
            overrides=None,
            metadata=None,
            session_id=None,
            correlation_id=None,
            agent_id=None,
        )


def test_build_session_payload_includes_metadata_and_overrides() -> None:
    payload = module._build_session_payload(
        config=None,
        config_name=None,
        config_path=None,
        config_body={"steps": []},
        overrides={"tuning": "fast"},
        metadata={"note": "test"},
        session_id="sid",
        correlation_id="cid",
        agent_id="aid",
    )
    assert payload["config_body"] == {"steps": []}
    assert payload["overrides"]["tuning"] == "fast"
    assert payload["metadata"]["note"] == "test"
    assert payload["session_id"] == "sid"
    assert payload["correlation_id"] == "cid"
    assert payload["agent_id"] == "aid"


def test_build_session_payload_requires_one_config_source() -> None:
    with pytest.raises(ValueError, match="config source"):
        module._build_session_payload(
            config=None,
            config_name=None,
            config_path=None,
            config_body=None,
            overrides={"tuning": "fast"},
            metadata={"note": "test"},
            session_id=None,
            correlation_id=None,
            agent_id=None,
        )


def test_build_session_payload_rejects_multiple_explicit_sources() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        module._build_session_payload(
            config=None,
            config_name="named-config",
            config_path=None,
            config_body={"steps": []},
            overrides=None,
            metadata=None,
            session_id=None,
            correlation_id=None,
            agent_id=None,
        )


def test_build_session_payload_resolves_config_path(tmp_path: Path) -> None:
    config_path = tmp_path / "session.toml"
    config_path.write_text("key = 1")
    payload = module._build_session_payload(
        config=None,
        config_name=None,
        config_path=config_path,
        config_body=None,
        overrides=None,
        metadata=None,
        session_id=None,
        correlation_id=None,
        agent_id=None,
    )
    assert payload["config_path"] == str(config_path)


def test_build_session_payload_config_path_missing_raises(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.toml"
    with pytest.raises(FileNotFoundError):
        module._build_session_payload(
            config=None,
            config_name=None,
            config_path=missing_path,
            config_body=None,
            overrides=None,
            metadata=None,
            session_id=None,
            correlation_id=None,
            agent_id=None,
        )


def test_coerce_str_handles_none_and_values() -> None:
    assert module._coerce_str(None) is None
    assert module._coerce_str(123) == "123"


def test_run_async_uses_run_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    called: Dict[str, Any] = {}

    async def dummy_coro() -> str:
        return "ok"

    def fake_run_sync(coro: Any, label: str) -> str:
        called["label"] = label
        return "sync-result"

    monkeypatch.setattr(module, "run_sync", fake_run_sync)
    result = module._run_async(dummy_coro())
    assert result == "sync-result"
    assert "MiproOnlineSession" in called["label"]


@pytest.mark.asyncio
async def test_update_reward_async_builds_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    clients = _patch_rust_client(monkeypatch)
    monkeypatch.setenv("SYNTH_API_KEY", "test-key")
    session = MiproOnlineSession(
        session_id="session-1",
        backend_url="https://backend.test",
        api_key="override",
        timeout=0.1,
    )
    response = await session.update_reward_async(
        reward_info={"score": 0.75},
        candidate_id="cand",
        stop=True,
        rollout_id="rollout-id",
    )

    assert response["result"] == "ok"
    assert clients
    last_client = clients[-1]
    assert last_client.requests[-1][0] == "post_json"
    path = last_client.requests[-1][1]
    assert path.endswith("/reward")
    payload = last_client.requests[-1][2]
    assert payload["candidate_id"] == "cand"
    assert payload["stop"] is True
    assert payload["rollout_id"] == "rollout-id"
    assert payload["reward_info"]["score"] == 0.75


@pytest.mark.asyncio
async def test_get_prompt_urls_async_passes_correlation(monkeypatch: pytest.MonkeyPatch) -> None:
    clients = _patch_rust_client(monkeypatch)
    session = MiproOnlineSession(
        session_id="session-2",
        backend_url="https://backend.test",
        api_key="api-key",
        timeout=0.1,
    )
    await session.get_prompt_urls_async(correlation_id="corr-1")

    last_client = clients[-1]
    assert last_client.requests[-1][0] == "get"
    params = last_client.requests[-1][2]
    assert params == {"correlation_id": "corr-1"}


@pytest.mark.asyncio
async def test_post_action_async_uses_action_path(monkeypatch: pytest.MonkeyPatch) -> None:
    clients = _patch_rust_client(monkeypatch)
    session = MiproOnlineSession(
        session_id="session-3",
        backend_url="https://backend.test",
        api_key="api-key",
        timeout=0.1,
    )
    await session._post_action_async("pause")

    last_client = clients[-1]
    assert last_client.requests[-1][0] == "post_json"
    assert last_client.requests[-1][1].endswith("/pause")


@pytest.mark.asyncio
async def test_status_async_raises_on_non_object_response(
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
    session = MiproOnlineSession(
        session_id="session-bad-status",
        backend_url="https://backend.test",
        api_key="api-key",
        timeout=0.1,
    )
    with pytest.raises(ValueError, match="MIPRO status endpoint"):
        await session.status_async()


@pytest.mark.asyncio
async def test_list_candidates_async_uses_system_candidates_client(
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
    session = MiproOnlineSession(
        session_id="session-list",
        backend_url="https://backend.test",
        api_key="api-key",
        timeout=0.1,
    )
    result = await session.list_candidates_async(
        status="evaluated",
        limit=5,
        sort="objective:desc,created_at:desc",
        include="prompt_text",
    )

    assert result["items"][0]["candidate_id"] == "cand_1"
    assert captured["list"]["system_id"] == "session-list"
    assert captured["list"]["kwargs"]["status"] == "evaluated"
    assert captured["list"]["kwargs"]["limit"] == 5


@pytest.mark.asyncio
async def test_get_candidate_async_rejects_candidate_from_other_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyPromptLearningClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        async def get_global_candidate(self, candidate_id: str) -> Dict[str, Any]:
            return {"candidate_id": candidate_id, "system_id": "other-session"}

        async def list_system_candidates(self, system_id: str, **kwargs: Any) -> Dict[str, Any]:
            del system_id, kwargs
            return {"items": []}

    monkeypatch.setattr(module, "PromptLearningClient", DummyPromptLearningClient)
    session = MiproOnlineSession(
        session_id="session-own",
        backend_url="https://backend.test",
        api_key="api-key",
        timeout=0.1,
    )

    with pytest.raises(ValueError, match="does not belong to MIPRO session"):
        await session.get_candidate_async("cand_2")


@pytest.mark.asyncio
async def test_list_seed_evals_async_uses_system_seed_eval_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: Dict[str, Any] = {}

    class DummyPromptLearningClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured["init"] = {"args": args, "kwargs": kwargs}

        async def list_system_seed_evals(self, system_id: str, **kwargs: Any) -> Dict[str, Any]:
            captured["list_seed_evals"] = {"system_id": system_id, "kwargs": kwargs}
            return {"items": [{"candidate_id": "cand_1", "seed": 3}], "system_id": system_id}

    monkeypatch.setattr(module, "PromptLearningClient", DummyPromptLearningClient)
    session = MiproOnlineSession(
        session_id="session-seeds",
        backend_url="https://backend.test",
        api_key="api-key",
        timeout=0.1,
    )
    payload = await session.list_seed_evals_async(split="held_out", limit=8, success=True)
    assert payload["system_id"] == "session-seeds"
    assert captured["list_seed_evals"]["system_id"] == "session-seeds"
    assert captured["list_seed_evals"]["kwargs"]["split"] == "held_out"
    assert captured["list_seed_evals"]["kwargs"]["limit"] == 8
    assert captured["list_seed_evals"]["kwargs"]["success"] is True


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
    session = MiproOnlineSession(
        session_id="session-pagination",
        backend_url="https://backend.test",
        api_key="api-key",
        timeout=0.1,
    )

    candidate = await session.get_candidate_async("cand_target")
    assert candidate["candidate_id"] == "cand_target"
    assert len(calls) == 2
    assert calls[0]["kwargs"]["cursor"] is None
    assert calls[1]["kwargs"]["cursor"] == "cursor-2"
