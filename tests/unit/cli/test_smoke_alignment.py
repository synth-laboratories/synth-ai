from __future__ import annotations

from pathlib import Path
from pathlib import Path
from typing import Any

import pytest

from synth_ai.task.contracts import (
    RolloutEnvSpec,
    RolloutMetrics,
    RolloutPolicySpec,
    RolloutRecordConfig,
    RolloutRequest,
    RolloutResponse,
    RolloutStep,
    RolloutTrajectory,
    TaskDescriptor,
    TaskInfo,
)


class _FakeTaskAppClient:
    def __init__(self, base_url: str, api_key: str | None = None, *, timeout: float = 600.0, retries: int = 3) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries
        self.health_calls = 0
        self.task_info_calls = 0
        self.task_info_last_seeds: list[int] | None = None
        self.rollout_calls = 0
        self.last_rollout_request: RolloutRequest | None = None

    async def __aenter__(self) -> "_FakeTaskAppClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        return None

    async def health(self) -> dict[str, Any]:
        self.health_calls += 1
        return {"status": "ok"}

    async def task_info(self, seeds: list[int] | None = None) -> TaskInfo:
        self.task_info_calls += 1
        self.task_info_last_seeds = list(seeds or []) if seeds else None
        # Minimal valid TaskInfo
        return TaskInfo(
            task=TaskDescriptor(id="crafter", name="crafter"),
            environment="crafter",
            dataset={},
            rubric={},
            inference={},
            limits={},
        )

    async def rollout(self, request: RolloutRequest) -> RolloutResponse:  # type: ignore[override]
        self.rollout_calls += 1
        self.last_rollout_request = request

        # Construct a minimal valid RolloutResponse that passes RL validation
        step = RolloutStep(
            obs={"image": [0]},
            tool_calls=[{"type": "function", "function": {"name": "act", "arguments": "{}"}}],
            reward=0.0,
            done=True,
            info={
                "meta": {
                    "inference_url": "https://mock.local/v1/chat/completions?cid=test-cid-123",
                }
            },
        )
        traj = RolloutTrajectory(
            env_id="env-1",
            policy_id="policy-1",
            steps=[step],
            final=None,
            length=1,
            inference_url="https://mock.local/v1/chat/completions?cid=test-cid-123",
            decision_samples=[],
        )
        metrics = RolloutMetrics(episode_returns=[0.0], mean_return=0.0, num_steps=1, num_episodes=1)
        return RolloutResponse(
            run_id=request.run_id,
            trajectories=[traj],
            branches={},
            metrics=metrics,
            aborted=False,
            ops_executed=1,
            trace={"id": request.run_id},
            pipeline_metadata={"inference_url": "https://mock.local/v1/chat/completions?cid=test-cid-123"},
        )


SMOKE_CORE_PATH = Path(__file__).resolve().parents[2] / "synth_ai" / "cli" / "commands" / "smoke" / "core.py"


SMOKE_CORE_PATH = Path(__file__).resolve().parents[3] / "synth_ai" / "cli" / "commands" / "smoke" / "core.py"


@pytest.mark.asyncio
async def test_smoke_rollout_request_alignment_structured_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    # Import by file path to avoid CLI package side effects
    import importlib.util
    import sys as _sys

    spec = importlib.util.spec_from_file_location("smoke_core_test", SMOKE_CORE_PATH)
    assert spec and spec.loader
    smoke_core = importlib.util.module_from_spec(spec)
    _sys.modules[spec.name] = smoke_core
    spec.loader.exec_module(smoke_core)  # type: ignore[arg-type]

    created_instances: list[_FakeTaskAppClient] = []

    def _factory(*args: Any, **kwargs: Any) -> _FakeTaskAppClient:
        inst = _FakeTaskAppClient(*args, **kwargs)
        created_instances.append(inst)
        return inst

    # Patch the TaskAppClient used by the smoke tool
    monkeypatch.setattr(smoke_core, "TaskAppClient", _factory)

    exit_code = await smoke_core._run_smoke_async(
        task_app_url="http://task.local:8000",
        api_key="k1",
        env_name_opt="crafter",
        policy_name="react",
        model="gpt-5-nano",
        inference_url_opt="https://api.openai.com/v1",  # ensure normalization appends /chat/completions
        inference_policy=None,
        max_steps=2,
        return_trace=True,
        use_mock=False,
        mock_port=0,
        mock_backend="synthetic",
        config_path=None,
        rollouts=1,
        group_size=1,
        batch_size=None,
    )

    assert exit_code == 0

    assert created_instances, "Expected TaskAppClient to be instantiated"
    inst = created_instances[-1]
    sent = inst.last_rollout_request
    assert sent is not None
    assert sent.record.return_trace is True
    assert sent.record.trace_format == "structured"
    # inference_url should be normalized to include /chat/completions and have a cid
    url = str((sent.policy.config or {}).get("inference_url"))
    assert "/chat/completions" in url
    assert "?cid=" in url


@pytest.mark.asyncio
async def test_smoke_calls_health_and_task_info_when_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util
    import sys as _sys
    spec = importlib.util.spec_from_file_location("smoke_core_test2", SMOKE_CORE_PATH)
    assert spec and spec.loader
    smoke_core = importlib.util.module_from_spec(spec)
    _sys.modules[spec.name] = smoke_core
    spec.loader.exec_module(smoke_core)  # type: ignore[arg-type]

    created_instances: list[_FakeTaskAppClient] = []

    def _factory(*args: Any, **kwargs: Any) -> _FakeTaskAppClient:  # type: ignore[no-redef]
        inst = _FakeTaskAppClient(*args, **kwargs)
        created_instances.append(inst)
        return inst

    monkeypatch.setattr(smoke_core, "TaskAppClient", _factory)

    exit_code = await smoke_core._run_smoke_async(
        task_app_url="http://task.local:8000",
        api_key="k1",
        env_name_opt=None,  # force task_info path
        policy_name="react",
        model="gpt-5-nano",
        inference_url_opt="https://api.openai.com/v1/chat/completions",
        inference_policy=None,
        max_steps=1,
        return_trace=False,
        use_mock=False,
        mock_port=0,
        mock_backend="synthetic",
        config_path=None,
        rollouts=1,
        group_size=1,
        batch_size=None,
    )

    assert exit_code == 0
    assert created_instances, "Expected TaskAppClient to be instantiated"
    inst = created_instances[-1]
    assert inst.health_calls >= 1
    assert inst.task_info_calls >= 1
    assert inst.task_info_last_seeds == [0]
