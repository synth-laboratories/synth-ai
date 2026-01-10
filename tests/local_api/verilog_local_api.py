"""VerilogEval spec-to-RTL local API for Synth prompt optimization benchmarks.

This is an agentic task where the model uses tools (write_file, compile, simulate, submit)
to implement and verify Verilog hardware designs.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterable, Sequence
from typing import Any, Mapping, cast

from dotenv import load_dotenv
from fastapi import HTTPException, Request

# Synth-AI SDK imports
from synth_ai.sdk.localapi.apps import LocalAPIEntry, ModalDeploymentConfig, register_local_api
from synth_ai.sdk.localapi.helpers import (
    create_http_client_hooks,
    extract_api_key,
    normalize_chat_completion_url,
    preload_dataset_splits,
)
from synth_ai.sdk.localapi.server import (
    LocalAPIConfig,
    RubricBundle,
    create_local_api,
    run_local_api,
)
from synth_ai.sdk.task.contracts import (
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
    TaskInfo,
)
from synth_ai.sdk.task.datasets import TaskDatasetRegistry, TaskDatasetSpec
from synth_ai.sdk.task.rubrics import Rubric, load_rubric
from synth_ai.sdk.task.trace_correlation_helpers import extract_trace_correlation_id

# Business logic imports (no synth-ai dependencies)
from verilog_business_logic import (
    AVAILABLE_SPLITS,
    DATASET_NAME,
    DEFAULT_SPLIT,
    MAX_STEPS,
    REPO_ROOT,
    TOOL_COMPILE,
    TOOL_OPEN_FILE,
    TOOL_SIMULATE,
    TOOL_SUBMIT,
    TOOL_WRITE_FILE,
    VerilogEvalDataset,
    VerilogWorkspace,
    build_verilog_tools,
    format_user_message,
    get_system_message,
)

print(
    f"[verilog_task_app] Module loaded: DATASET_NAME={DATASET_NAME}",
    flush=True,
)

# Dataset spec for registry
VERILOG_DATASET_SPEC = TaskDatasetSpec(
    id="verilog",
    name="VerilogEval v2 Spec-to-RTL",
    version="2.0.0",
    splits=list(AVAILABLE_SPLITS),
    default_split=DEFAULT_SPLIT,
    description="VerilogEval v2 specification-to-RTL translation tasks.",
)


async def call_chat_completion_with_tools(
    policy_config: dict[str, Any],
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]],
    api_key: str | None = None,
    http_client: Any | None = None,
) -> tuple[str, list[dict[str, Any]], dict[str, Any] | None]:
    """Call the chat completion API with tools and return response."""
    model_val = policy_config.get("model")
    if not isinstance(model_val, str) or not model_val.strip():
        raise HTTPException(status_code=400, detail="Missing policy field: model")

    inference_url_raw = policy_config.get("inference_url")
    api_base_raw = policy_config.get("api_base")
    base_url_raw = policy_config.get("base_url")

    if inference_url_raw:
        route_base = str(inference_url_raw).strip()
    else:
        route_base = (api_base_raw or "").strip() or (base_url_raw or "").strip()

    if not route_base:
        raise HTTPException(status_code=400, detail="Missing policy field: inference_url")

    model = policy_config["model"].strip()
    inference_url = normalize_chat_completion_url(str(route_base))
    temperature = policy_config.get("temperature", 0.0)
    max_tokens = policy_config.get("max_completion_tokens", 512)

    headers: dict[str, str] = {"Content-Type": "application/json"}
    lowered = route_base.lower()
    is_provider_host = ("api.openai.com" in lowered) or ("api.groq.com" in lowered)

    if api_key:
        if is_provider_host:
            headers["Authorization"] = f"Bearer {api_key}"
        else:
            headers["X-API-Key"] = api_key

    def _build_payload(message_list: list[dict[str, str]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": message_list,
            "max_completion_tokens": max_tokens,
            "tools": tools,
            "tool_choice": "auto",
        }
        if temperature != 0.0:
            payload["temperature"] = temperature
        return payload

    def _needs_tool_retry(error_text: str) -> bool:
        lowered = error_text.lower()
        return "tool_use_failed" in lowered or "failed to parse tool call arguments" in lowered

    if http_client is None:
        raise HTTPException(status_code=500, detail="HTTP client not initialized")

    response_json: dict[str, Any] | None = None
    try:
        import aiohttp

        is_aiohttp = isinstance(http_client, aiohttp.ClientSession)

        for attempt in range(2):
            message_list = list(messages)
            if attempt == 1:
                message_list = message_list + [
                    {
                        "role": "user",
                        "content": (
                            "If you call a tool, the arguments must be valid JSON. "
                            "Do not include comments or trailing commas."
                        ),
                    }
                ]
            payload = _build_payload(message_list)

            if is_aiohttp:
                async with http_client.post(
                    inference_url, json=payload, headers=headers
                ) as response:
                    status_code = response.status
                    if status_code != 200:
                        error_text = await response.text()
                        if status_code == 400 and attempt == 0 and _needs_tool_retry(error_text):
                            continue
                        raise HTTPException(
                            status_code=status_code, detail=f"API error: {error_text[:200]}"
                        )
                    response_json = await response.json()
            else:
                response = await http_client.post(inference_url, json=payload, headers=headers)
                if response.status_code != 200:
                    error_text = response.text
                    if (
                        response.status_code == 400
                        and attempt == 0
                        and _needs_tool_retry(error_text)
                    ):
                        continue
                    raise HTTPException(
                        status_code=response.status_code, detail=f"API error: {error_text[:200]}"
                    )
                response_json = response.json()
            break
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Request failed: {e}") from e

    if response_json is None:
        raise HTTPException(status_code=502, detail="No response data")

    response_text = ""
    tool_calls = []
    if "choices" in response_json and len(response_json["choices"]) > 0:
        choice = response_json["choices"][0]
        message = choice.get("message", {})
        response_text = message.get("content", "") or ""

        if "tool_calls" in message and message["tool_calls"]:
            for tc in message["tool_calls"]:
                tool_calls.append(
                    {
                        "id": tc.get("id", ""),
                        "type": tc.get("type", "function"),
                        "function": {
                            "name": tc.get("function", {}).get("name", ""),
                            "arguments": tc.get("function", {}).get("arguments", "{}"),
                        },
                    }
                )

    return response_text, tool_calls, response_json


async def rollout_executor(request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
    """Execute a rollout for the Verilog task.

    Note: We emit `event_history` as plain dicts to match current rollout ingestion.
    See `synth-ai/tests/local_api/migrate_event_history.txt` for the plan to move
    to typed SessionTrace/LMCAISEvent objects.
    """
    dataset: VerilogEvalDataset = fastapi_request.app.state.verilog_dataset

    split = str(((request.env.config or {}).get("split")) or DEFAULT_SPLIT)
    seed = request.env.seed or 0

    sample = dataset.sample(split=split, index=seed)

    # Create workspace
    workspace = VerilogWorkspace(
        problem_id=sample["problem_id"],
        prompt=sample["prompt"],
        testbench=sample["test"],
        ref_solution=sample["ref"],
    )

    try:
        # Build initial observation
        observation = {
            "problem_id": sample["problem_id"],
            "instructions": sample["prompt"],
            "files": list(workspace.files.keys()),
            "index": sample["index"],
            "split": sample["split"],
        }

        # Build messages with STATIC system message and DYNAMIC user message
        system_message = get_system_message()
        user_message = format_user_message(
            sample["problem_id"], sample["prompt"], list(workspace.files.keys())
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        tools = build_verilog_tools()

        api_key = extract_api_key(fastapi_request, request.policy.config or {})

        http_client = getattr(fastapi_request.app.state, "http_client", None)

        steps: list[dict[str, Any]] = []
        # Dict-based events for compatibility; see migrate_event_history.txt for typed traces.
        event_history: list[dict[str, Any]] = []
        done = False
        inference_url = (request.policy.config or {}).get("inference_url")
        trace_correlation_id = extract_trace_correlation_id(
            policy_config=request.policy.config or {},
            inference_url=str(inference_url or ""),
        )

        # Agentic loop
        for step_idx in range(MAX_STEPS):
            if done:
                break

            # Get model response with tools
            messages_snapshot = list(messages)
            try:
                response_text, tool_calls, response_json = await call_chat_completion_with_tools(
                    request.policy.config or {},
                    messages,
                    tools,
                    api_key=api_key,
                    http_client=http_client,
                )
            except HTTPException as exc:
                detail = getattr(exc, "detail", "")
                detail_str = str(detail)
                if exc.status_code == 400 and "tool_use_failed" in detail_str.lower():
                    response_text = ""
                    tool_calls = []
                    response_json = {"error": detail_str}
                else:
                    raise

            # Add assistant message to history
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": response_text or ""}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            llm_event = {
                "type": "lm_call",
                "event_type": "lm_call",
                "step_index": step_idx,
                "llm_request": {
                    "messages": messages_snapshot,
                    "model": (request.policy.config or {}).get("model", "unknown"),
                },
                "llm_response": {
                    "message": {
                        "role": "assistant",
                        "content": response_text or "",
                        "tool_calls": tool_calls,
                    },
                    "usage": response_json.get("usage", {})
                    if isinstance(response_json, dict)
                    else {},
                    "model": response_json.get("model")
                    if isinstance(response_json, dict)
                    else None,
                },
            }
            if trace_correlation_id:
                llm_event["correlation_id"] = trace_correlation_id
            event_history.append(llm_event)

            step_reward = 0.0
            step_info: dict[str, Any] = {"step": step_idx}
            if isinstance(response_json, dict) and response_json.get("error"):
                step_info["api_error"] = response_json.get("error")

            if not tool_calls:
                # No tool calls - model is done or confused
                step_info["no_tool_call"] = True
                done = True
            else:
                # Process tool calls
                for tc in tool_calls:
                    fn_name = tc.get("function", {}).get("name", "")
                    fn_args_str = tc.get("function", {}).get("arguments", "{}")

                    try:
                        fn_args = json.loads(fn_args_str)
                    except json.JSONDecodeError:
                        fn_args = {}

                    tool_result: dict[str, Any]

                    if fn_name == TOOL_WRITE_FILE:
                        tool_result = workspace.write_file(
                            fn_args.get("path", "TopModule.v"),
                            fn_args.get("content", ""),
                        )
                    elif fn_name == TOOL_OPEN_FILE:
                        tool_result = workspace.read_file(fn_args.get("path", "TopModule.v"))
                    elif fn_name == TOOL_COMPILE:
                        tool_result = workspace.compile(
                            fn_args.get("sources"),
                            fn_args.get("testbench"),
                        )
                    elif fn_name == TOOL_SIMULATE:
                        tool_result = workspace.simulate(fn_args.get("binary"))
                    elif fn_name == TOOL_SUBMIT:
                        tool_result = workspace.submit()
                        done = True
                        if workspace.passed:
                            step_reward = 1.0
                    else:
                        tool_result = {"ok": False, "error": f"Unknown tool: {fn_name}"}

                    step_info[f"tool_{fn_name}"] = tool_result

                    # Add tool result to messages
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.get("id", ""),
                            "content": json.dumps(tool_result),
                        }
                    )

            steps.append(
                {
                    "obs": observation,
                    "tool_calls": tool_calls,
                    "reward": step_reward,
                    "done": done,
                    "info": step_info,
                }
            )

            print(
                f"[VERILOG_ROLLOUT] seed={seed} step={step_idx} tool_calls={len(tool_calls)} done={done} reward={step_reward}",
                flush=True,
            )

        # Final reward
        final_reward = 1.0 if workspace.passed else 0.0

        event_history.append(
            {
                "type": "outcome",
                "event_type": "outcome",
                "reward": final_reward,
                "metadata": {
                    "passed": workspace.passed,
                    "steps": len(steps),
                },
            }
        )

        corr_ids: dict[str, Any] = {"run_id": request.run_id, "seed": seed}
        if trace_correlation_id:
            corr_ids["trace_correlation_id"] = trace_correlation_id

        trace_payload = {
            "schema_version": "3.0",
            "event_history": event_history,
            "markov_blanket_message_history": [],
            "metadata": {
                "environment": "verilog",
                "split": sample["split"],
                "index": sample["index"],
                "passed": workspace.passed,
                "steps": len(steps),
                "trace_correlation_id": trace_correlation_id,
                "correlation_ids": corr_ids,
            },
        }

        metrics = RolloutMetrics(
            outcome_reward=final_reward,
            details={"passed": workspace.passed, "steps": len(steps)},
        )

        return RolloutResponse(
            run_id=request.run_id,
            metrics=metrics,
            trace_correlation_id=trace_correlation_id,
            trace=trace_payload,
            inference_url=str(inference_url or ""),
        )

    finally:
        # Clean up workspace
        workspace.cleanup()


def build_dataset() -> tuple[TaskDatasetRegistry, VerilogEvalDataset]:
    """Build the dataset registry and dataset instance."""
    registry = TaskDatasetRegistry()
    dataset = VerilogEvalDataset()
    registry.register(VERILOG_DATASET_SPEC, lambda _spec: dataset, cache=True)
    return registry, dataset


def _base_task_info() -> TaskInfo:
    """Get the base task info for Verilog."""
    return TaskInfo(
        task={
            "id": "verilog",
            "name": "VerilogEval v2 Spec-to-RTL",
            "version": "2.0.0",
            "action_space": {
                "type": "tool_call",
                "tools": [TOOL_WRITE_FILE, TOOL_COMPILE, TOOL_SIMULATE, TOOL_SUBMIT],
                "description": "Implement Verilog modules using write, compile, simulate, submit workflow.",
            },
        },
        environment="verilog",
        dataset=VERILOG_DATASET_SPEC.model_dump(),
        rubric={"version": "1", "criteria_count": 1, "source": "inline"},
        inference={"supports_proxy": True, "agentic": True},
        limits={"max_turns": MAX_STEPS},
        task_metadata={"format": "agentic_tool_call"},
    )


def describe_taskset(dataset: VerilogEvalDataset) -> Mapping[str, Any]:
    """Describe the taskset for the API."""
    return {
        **VERILOG_DATASET_SPEC.model_dump(),
        "sizes": {split: dataset.size(split) for split in AVAILABLE_SPLITS},
    }


def provide_task_instances(dataset: VerilogEvalDataset, seeds: Sequence[int]) -> Iterable[TaskInfo]:
    """Provide task instances for the given seeds."""
    base_info = _base_task_info()
    for seed in seeds:
        sample = dataset.sample(split=DEFAULT_SPLIT, index=seed)
        yield TaskInfo(
            task=base_info.task,
            environment=base_info.environment,
            dataset={**base_info.dataset, "split": sample["split"], "index": sample["index"]},
            rubric=base_info.rubric,
            inference=base_info.inference,
            limits=base_info.limits,
            task_metadata={
                **base_info.task_metadata,
                "problem_id": sample["problem_id"],
                "prompt": sample["prompt"][:500],  # Truncate for metadata
            },
        )


OUTCOME_RUBRIC: Rubric = cast(
    Rubric,
    load_rubric(
        {
            "version": "1",
            "goal_text": "Implement Verilog modules that pass testbench verification.",
            "aggregation": "weighted_sum",
            "criteria": [
                {
                    "id": "testbench_pass",
                    "description": "Implementation passes all testbench tests.",
                    "weight": 1.0,
                }
            ],
        }
    ),
)


def build_config() -> LocalAPIConfig:
    """Build the LocalAPIConfig for the local API."""
    registry, dataset = build_dataset()
    base_info = _base_task_info()

    preload_dataset_splits(dataset, AVAILABLE_SPLITS, "verilog_task_app")

    startup_http_client, shutdown_http_client = create_http_client_hooks(
        timeout=60.0,
        log_prefix="verilog_task_app",
    )

    config = LocalAPIConfig(
        app_id="verilog",
        name="VerilogEval v2 Spec-to-RTL Task",
        description="VerilogEval v2 spec-to-RTL local API for Verilog code generation.",
        base_task_info=base_info,
        describe_taskset=lambda: describe_taskset(dataset),
        provide_task_instances=lambda seeds: provide_task_instances(dataset, seeds),
        rollout=rollout_executor,
        dataset_registry=registry,
        rubrics=RubricBundle(outcome=OUTCOME_RUBRIC, events=OUTCOME_RUBRIC),
        proxy=None,
        app_state={"verilog_dataset": dataset},
        cors_origins=["*"],
        startup_hooks=[startup_http_client],
        shutdown_hooks=[shutdown_http_client],
    )
    return config


register_local_api(
    entry=LocalAPIEntry(
        api_id="verilog",
        description="VerilogEval v2 spec-to-RTL local API for Verilog code generation.",
        config_factory=build_config,
        aliases=("verilogeval",),
        modal=ModalDeploymentConfig(
            app_name="synth-verilog",
            pip_packages=(
                "datasets>=2.14.0",
                "fastapi>=0.115.0",
                "pydantic>=2.0.0",
                "aiohttp>=3.9.0",
            ),
        ),
    )
)


def fastapi_app():
    """Return the FastAPI application."""
    with contextlib.suppress(Exception):
        load_dotenv(str(REPO_ROOT / ".env"), override=False)

    app = create_local_api(build_config())
    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the VerilogEval local API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8118)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    run_local_api(
        build_config,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
