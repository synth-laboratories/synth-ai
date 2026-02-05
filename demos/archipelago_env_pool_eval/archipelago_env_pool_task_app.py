"""LocalAPI task app that forwards rollouts to an Archipelago environment pool."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any

from synth_ai.data.artifacts import Artifact
import httpx
from synth_ai.sdk.localapi import LocalAPIConfig, create_local_api
from synth_ai.sdk.localapi._impl import (
    DatasetInfo,
    InferenceInfo,
    LimitsInfo,
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
    TaskDescriptor,
    TaskInfo,
)

APP_ID = "archipelago_env_pool_eval"
APP_NAME = "Archipelago Env Pool Eval Proxy"
APP_DESCRIPTION = "Routes LocalAPI rollouts into Synth environment pools (archipelago)."

DEFAULT_BACKEND_URL = (
    os.environ.get("SYNTH_BACKEND_URL")
    or os.environ.get("SYNTH_BASE_URL")
    or "https://api-dev.usesynth.ai"
)
DEFAULT_ENV_POOLS_URL = os.environ.get("ENV_POOLS_BASE_URL") or DEFAULT_BACKEND_URL
DEFAULT_POLL_INTERVAL = float(os.environ.get("ARCHIPELAGO_POLL_INTERVAL_SEC", "3"))
DEFAULT_ROLLOUT_TIMEOUT = float(os.environ.get("ARCHIPELAGO_ROLLOUT_TIMEOUT_SEC", "900"))
DEFAULT_AGENT_TIMEOUT = int(os.environ.get("ARCHIPELAGO_AGENT_TIMEOUT_SEC", "900"))
DEFAULT_VERIFIER_TIMEOUT = int(os.environ.get("ARCHIPELAGO_VERIFIER_TIMEOUT_SEC", "240"))


@dataclass(frozen=True)
class ArchipelagoDefaults:
    env_image: str
    agent_image: str
    grading_image: str
    env_port: int | None
    initial_snapshot_path: str
    mcp_config_path: str
    initial_messages_path: str
    agent_config_path: str
    orchestrator_config_path: str
    grading_settings_path: str
    verifiers_path: str
    eval_configs_path: str
    scoring_config_path: str
    mcp_gateway_auth_token: str | None


def _env_str(name: str, fallback: str = "") -> str:
    return (os.environ.get(name) or fallback).strip()


def _env_int(name: str, fallback: int | None = None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return fallback
    try:
        return int(raw)
    except ValueError:
        return fallback


def _load_json_env(name: str) -> dict[str, Any]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return {}
    try:
        if raw.endswith(".json") and os.path.exists(raw):
            with open(raw, "r", encoding="utf-8") as handle:
                return json.load(handle) or {}
        return json.loads(raw)
    except Exception as exc:
        raise ValueError(f"Failed to parse {name}: {exc}") from exc


def _normalize_tags(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        tags = [v.strip() for v in value.split(",") if v.strip()]
        return tags or None
    if isinstance(value, list):
        tags = [str(v).strip() for v in value if str(v).strip()]
        return tags or None
    return None


def _build_defaults() -> ArchipelagoDefaults:
    return ArchipelagoDefaults(
        env_image=_env_str("RHODES_APEX_ENV_IMAGE"),
        agent_image=_env_str("RHODES_APEX_AGENT_IMAGE"),
        grading_image=_env_str("RHODES_APEX_GRADING_IMAGE"),
        env_port=_env_int("ARCHIPELAGO_ENV_PORT"),
        initial_snapshot_path=_env_str("ARCHIPELAGO_INITIAL_SNAPSHOT_PATH", "/configs/original_snapshot.zip"),
        mcp_config_path=_env_str("ARCHIPELAGO_MCP_CONFIG_PATH", "/configs/mcp_config.json"),
        initial_messages_path=_env_str(
            "ARCHIPELAGO_INITIAL_MESSAGES_PATH", "/configs/initial_messages.json"
        ),
        agent_config_path=_env_str("ARCHIPELAGO_AGENT_CONFIG_PATH", "/configs/agent_config.json"),
        orchestrator_config_path=_env_str(
            "ARCHIPELAGO_ORCHESTRATOR_CONFIG_PATH", "/configs/orchestrator_config.json"
        ),
        grading_settings_path=_env_str(
            "ARCHIPELAGO_GRADING_SETTINGS_PATH", "/configs/grading_settings.json"
        ),
        verifiers_path=_env_str("ARCHIPELAGO_VERIFIERS_PATH", "/configs/verifiers.json"),
        eval_configs_path=_env_str("ARCHIPELAGO_EVAL_CONFIGS_PATH", "/configs/eval_configs.json"),
        scoring_config_path=_env_str("ARCHIPELAGO_SCORING_CONFIG_PATH", "/configs/scoring_config.json"),
        mcp_gateway_auth_token=_env_str("ARCHIPELAGO_MCP_GATEWAY_AUTH_TOKEN") or None,
    )


def _merge_archipelago_config(overrides: dict[str, Any] | None) -> dict[str, Any]:
    defaults = _build_defaults()
    config: dict[str, Any] = {
        "env_image": defaults.env_image,
        "agent_image": defaults.agent_image,
        "grading_image": defaults.grading_image,
        "env_port": defaults.env_port,
        "initial_snapshot_path": defaults.initial_snapshot_path,
        "mcp_config_path": defaults.mcp_config_path,
        "initial_messages_path": defaults.initial_messages_path,
        "agent_config_path": defaults.agent_config_path,
        "orchestrator_config_path": defaults.orchestrator_config_path,
        "grading_settings_path": defaults.grading_settings_path,
        "verifiers_path": defaults.verifiers_path,
        "eval_configs_path": defaults.eval_configs_path,
        "scoring_config_path": defaults.scoring_config_path,
        "mcp_gateway_auth_token": defaults.mcp_gateway_auth_token,
    }
    extra = _load_json_env("ARCHIPELAGO_CONFIG_JSON")
    if extra:
        config.update(extra)
    if overrides:
        config.update({k: v for k, v in overrides.items() if v is not None})
    return config


def _require_archipelago_images(config: dict[str, Any]) -> None:
    missing = [
        key
        for key in ("env_image", "agent_image", "grading_image")
        if not config.get(key)
    ]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            "Missing required Archipelago image(s): "
            f"{joined}. Set RHODES_APEX_*_IMAGE env vars or override via env_config['archipelago']."
        )


def _poll_rollout(
    rollout_id: str,
    backend_base: str,
    api_key: str,
    timeout_sec: float,
) -> dict[str, Any]:
    deadline = time.time() + timeout_sec
    terminal = {"succeeded", "failed", "cancelled", "error", "completed"}
    last: dict[str, Any] = {}
    while time.time() < deadline:
        last = _env_pools_get_rollout(rollout_id, backend_base=backend_base, api_key=api_key)
        status = last.get("status", "")
        if status in terminal:
            return last
        time.sleep(DEFAULT_POLL_INTERVAL)
    return last


def _env_pools_use_infra(base_url: str) -> bool:
    flag = os.environ.get("ENV_POOLS_USE_INFRA_API", "").strip().lower()
    if flag in ("1", "true", "yes", "on"):
        return True
    return "infra-api" in base_url


def _env_pools_headers(api_key: str, *, base_url: str) -> dict[str, str]:
    header = os.environ.get("ENV_POOLS_AUTH_HEADER", "").strip()
    if header:
        return {header: api_key}
    if _env_pools_use_infra(base_url):
        return {"x-user-api-key": api_key}
    return {"Authorization": f"Bearer {api_key}"}


def _env_pools_url(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    prefix = "/v1" if _env_pools_use_infra(base) else "/api/v1/environment-pools"
    return f"{base}{prefix}/{path.lstrip('/')}"


def _env_pools_create_rollout(
    *,
    backend_base: str,
    api_key: str,
    request: dict[str, Any],
    timeout: float = 60.0,
) -> dict[str, Any]:
    url = _env_pools_url(backend_base, "rollouts")
    resp = httpx.post(url, headers=_env_pools_headers(api_key, base_url=backend_base), json=request, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, dict) else {}


def _env_pools_get_rollout(
    rollout_id: str,
    *,
    backend_base: str,
    api_key: str,
    timeout: float = 30.0,
) -> dict[str, Any]:
    url = _env_pools_url(backend_base, f"rollouts/{rollout_id}")
    resp = httpx.get(url, headers=_env_pools_headers(api_key, base_url=backend_base), timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, dict) else {}


def _build_rollout_response(
    request: RolloutRequest,
    *,
    outcome_reward: float,
    trace: dict[str, Any] | None,
    artifact: list[Any] | None,
    status_detail: str | None,
) -> RolloutResponse:
    trace_payload = trace
    if isinstance(trace_payload, dict) and request.trace_correlation_id:
        trace_payload = dict(trace_payload)
        metadata = trace_payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        metadata.setdefault("trace_correlation_id", request.trace_correlation_id)
        trace_payload["metadata"] = metadata

    return RolloutResponse(
        trace_correlation_id=request.trace_correlation_id,
        reward_info=RolloutMetrics(outcome_reward=float(outcome_reward)),
        trace=trace_payload,
        artifact=artifact,
        status_detail=status_detail,
    )


def provide_taskset_description() -> dict[str, Any]:
    return {
        "id": APP_ID,
        "name": APP_NAME,
        "splits": ["default"],
        "description": APP_DESCRIPTION,
    }


def provide_task_instances(seeds: list[int]) -> list[TaskInfo]:
    instances: list[TaskInfo] = []
    for seed in seeds:
        instances.append(
            TaskInfo(
                task=TaskDescriptor(
                    id=f"archipelago-simple-task-{seed}",
                    name="Archipelago Simple Task",
                    description="Simple task executed via Archipelago pool.",
                ),
                dataset=DatasetInfo(
                    id="archipelago-simple-task",
                    name="archipelago/simple_task",
                    splits=["default"],
                    default_split="default",
                ),
                inference=InferenceInfo(model=os.environ.get("POLICY_MODEL")),
                limits=LimitsInfo(timeout_seconds=int(DEFAULT_ROLLOUT_TIMEOUT)),
            )
        )
    return instances


async def run_rollout(request: RolloutRequest, _http_request: Any) -> Any:
    backend_base = request.env.config.get("backend_url") or DEFAULT_ENV_POOLS_URL
    api_key = os.environ.get("SYNTH_API_KEY", "").strip()
    if not api_key:
        raise ValueError("SYNTH_API_KEY is required to call environment pools")

    env_config = request.env.config or {}
    arch_overrides = env_config.get("archipelago") if isinstance(env_config, dict) else None
    archipelago_config = _merge_archipelago_config(
        arch_overrides if isinstance(arch_overrides, dict) else None
    )
    _require_archipelago_images(archipelago_config)

    pool_id = env_config.get("pool_id") if isinstance(env_config, dict) else None
    pool_tags = _normalize_tags(env_config.get("pool_tags") if isinstance(env_config, dict) else None)
    if pool_tags is None:
        pool_tags = _normalize_tags(os.environ.get("ARCHIPELAGO_POOL_TAGS"))
    if not pool_id:
        pool_id = os.environ.get("ARCHIPELAGO_POOL_ID")

    timeouts = env_config.get("timeouts") if isinstance(env_config, dict) else None
    if not isinstance(timeouts, dict):
        timeouts = {}
    timeouts.setdefault("agent_sec", DEFAULT_AGENT_TIMEOUT)
    timeouts.setdefault("verifier_sec", DEFAULT_VERIFIER_TIMEOUT)

    rollout_request: dict[str, Any] = {
        "archipelago": archipelago_config,
        "timeouts": timeouts,
    }
    if pool_id:
        rollout_request["pool_id"] = pool_id
    if pool_tags:
        rollout_request["pool_tags"] = pool_tags

    rollout = await asyncio.to_thread(
        _env_pools_create_rollout,
        backend_base=backend_base,
        api_key=api_key,
        request=rollout_request,
        timeout=60.0,
    )

    rollout_id = rollout.get("trial_id", "")
    if not rollout_id:
        raise RuntimeError("Environment pool did not return a trial_id")

    final = await asyncio.to_thread(
        _poll_rollout,
        rollout_id,
        backend_base,
        api_key,
        float(env_config.get("rollout_timeout", DEFAULT_ROLLOUT_TIMEOUT))
        if isinstance(env_config, dict)
        else DEFAULT_ROLLOUT_TIMEOUT,
    )

    reward = final.get("reward_primary")
    reward_value = float(reward) if reward is not None else 0.0

    artifact = Artifact(
        content={
            "trial_id": rollout_id,
            "status": final.get("status"),
            "pool_id": final.get("pool_id"),
            "reward_primary": final.get("reward_primary"),
            "reward_metrics": final.get("reward_metrics"),
            "result_details": final.get("result_details"),
        },
        content_type="environment_pool_rollout",
        metadata={"backend_url": backend_base},
    )

    trace = {
        "metadata": {
            "trace_correlation_id": request.trace_correlation_id,
            "pool_trial_id": rollout_id,
        }
    }

    return _build_rollout_response(
        request,
        outcome_reward=reward_value,
        trace=trace,
        artifact=[artifact],
        status_detail=f"archipelago pool status: {final.get('status')}",
    )


app = create_local_api(
    LocalAPIConfig(
        app_id=APP_ID,
        name=APP_NAME,
        description=APP_DESCRIPTION,
        provide_taskset_description=provide_taskset_description,
        provide_task_instances=provide_task_instances,
        rollout=run_rollout,
        cors_origins=["*"],
    )
)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
