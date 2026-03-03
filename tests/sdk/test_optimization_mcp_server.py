from __future__ import annotations

import json

import pytest


def test_initialize_advertises_tools_capability() -> None:
    from synth_ai.mcp.optimization_server import OptimizationMcpServer

    server = OptimizationMcpServer()
    result = server.dispatch("initialize", {"protocolVersion": "2025-06-18"})

    assert result["protocolVersion"] == "2025-06-18"
    assert "tools" in result["capabilities"]


def test_tools_list_contains_stateful_v2_controls() -> None:
    from synth_ai.mcp.optimization_server import OptimizationMcpServer

    server = OptimizationMcpServer()
    result = server.dispatch("tools/list", {})
    names = {tool["name"] for tool in result["tools"]}

    assert "opt_offline_submit_candidates" in names
    assert "opt_offline_get_state_envelope" in names
    assert "opt_offline_list_trial_queue" in names
    assert "opt_offline_get_rollout_dispatch_metrics" in names
    assert "opt_offline_retry_rollout_dispatch" in names
    assert "opt_offline_drain_rollout_queue" in names


def test_tools_call_unknown_tool_returns_tool_error() -> None:
    from synth_ai.mcp.optimization_server import OptimizationMcpServer

    server = OptimizationMcpServer()
    result = server.dispatch("tools/call", {"name": "missing_tool", "arguments": {}})

    assert result["isError"] is True
    assert "Unknown tool" in result["content"][0]["text"]


def test_tool_get_status_uses_synth_client(monkeypatch: pytest.MonkeyPatch) -> None:
    import synth_ai.mcp.optimization_server as mcp_module

    class DummyJob:
        def status(self) -> dict[str, str]:
            return {"job_id": "pl_123", "state": "running"}

    class DummyOfflineClient:
        def get(self, job_id: str, **kwargs):  # noqa: ANN003
            assert job_id == "pl_123"
            assert kwargs.get("api_version") == "v2"
            return DummyJob()

    class DummyOptimization:
        def __init__(self) -> None:
            self.offline = DummyOfflineClient()

    class DummySynthClient:
        def __init__(self, *, api_key=None, base_url=None, timeout=30.0):
            self.api_key = api_key
            self.base_url = base_url
            self.timeout = timeout
            self.optimization = DummyOptimization()

    monkeypatch.setattr(mcp_module, "SynthClient", DummySynthClient)
    server = mcp_module.OptimizationMcpServer()

    result = server.dispatch(
        "tools/call",
        {
            "name": "opt_offline_get_status",
            "arguments": {
                "job_id": "pl_123",
                "api_key": "sk_local",
                "backend_base": "https://infra-api-dev.usesynth.ai",
                "api_version": "v2",
            },
        },
    )

    payload = json.loads(result["content"][0]["text"])
    assert payload["job_id"] == "pl_123"
    assert payload["state"] == "running"


def test_tool_submit_candidates_uses_delegate_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    import synth_ai.mcp.optimization_server as mcp_module

    class DummyJob:
        def submit_candidates(self, **kwargs):  # noqa: ANN003
            assert kwargs["algorithm_kind"] == "gepa"
            assert isinstance(kwargs["candidates"], list)
            return {
                "job_id": "pl_123",
                "accepted": [{"index": 0, "accepted": True}],
                "rejected": [],
            }

    class DummyOfflineClient:
        def get(self, _job_id: str, **kwargs):  # noqa: ANN003
            assert kwargs.get("api_version") == "v2"
            return DummyJob()

    class DummyOptimization:
        def __init__(self) -> None:
            self.offline = DummyOfflineClient()

    class DummySynthClient:
        def __init__(self, *, api_key=None, base_url=None, timeout=30.0):
            self.api_key = api_key
            self.base_url = base_url
            self.timeout = timeout
            self.optimization = DummyOptimization()

    monkeypatch.setattr(mcp_module, "SynthClient", DummySynthClient)
    server = mcp_module.OptimizationMcpServer()
    result = server.dispatch(
        "tools/call",
        {
            "name": "opt_offline_submit_candidates",
            "arguments": {
                "job_id": "pl_123",
                "algorithm_kind": "gepa",
                "candidates": [
                    {
                        "candidate_type": "gepa_prompt_candidate",
                        "stage_prompts": [{"stage_id": "root", "instruction_text": "x"}],
                    }
                ],
            },
        },
    )
    payload = json.loads(result["content"][0]["text"])
    assert payload["job_id"] == "pl_123"
    assert payload["accepted"][0]["accepted"] is True


def test_tool_rollout_retry_and_drain_delegate_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    import synth_ai.mcp.optimization_server as mcp_module

    class DummyJob:
        def retry_rollout_dispatch(self, dispatch_id: str, **kwargs):  # noqa: ANN003
            assert dispatch_id == "trial_0001:0"
            assert kwargs.get("algorithm_kind") == "gepa"
            return {"ok": True, "dispatch_id": dispatch_id, "op": "retry"}

        def drain_rollout_queue(self, **kwargs):  # noqa: ANN003
            assert kwargs.get("algorithm_kind") == "gepa"
            assert kwargs.get("cancel_queued") is True
            return {"ok": True, "dispatcher_status": "draining", "op": "drain"}

    class DummyOfflineClient:
        def get(self, _job_id: str, **kwargs):  # noqa: ANN003
            assert kwargs.get("api_version") == "v2"
            return DummyJob()

    class DummyOptimization:
        def __init__(self) -> None:
            self.offline = DummyOfflineClient()

    class DummySynthClient:
        def __init__(self, *, api_key=None, base_url=None, timeout=30.0):
            self.api_key = api_key
            self.base_url = base_url
            self.timeout = timeout
            self.optimization = DummyOptimization()

    monkeypatch.setattr(mcp_module, "SynthClient", DummySynthClient)
    server = mcp_module.OptimizationMcpServer()

    retry_result = server.dispatch(
        "tools/call",
        {
            "name": "opt_offline_retry_rollout_dispatch",
            "arguments": {
                "job_id": "pl_123",
                "algorithm_kind": "gepa",
                "dispatch_id": "trial_0001:0",
            },
        },
    )
    retry_payload = json.loads(retry_result["content"][0]["text"])
    assert retry_payload["ok"] is True
    assert retry_payload["op"] == "retry"

    drain_result = server.dispatch(
        "tools/call",
        {
            "name": "opt_offline_drain_rollout_queue",
            "arguments": {
                "job_id": "pl_123",
                "algorithm_kind": "gepa",
                "cancel_queued": True,
            },
        },
    )
    drain_payload = json.loads(drain_result["content"][0]["text"])
    assert drain_payload["ok"] is True
    assert drain_payload["op"] == "drain"
