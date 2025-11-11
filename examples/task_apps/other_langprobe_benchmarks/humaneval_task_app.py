"""HumanEval code generation task app for LangProBe benchmarks."""

from __future__ import annotations

import contextlib
import os
import re
import subprocess
import tempfile
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Mapping, cast

from datasets import load_dataset
from fastapi import APIRouter, HTTPException, Request

from synth_ai.task.apps import ModalDeploymentConfig, TaskAppEntry, register_task_app
from synth_ai.task.contracts import (
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
    RolloutStep,
    RolloutTrajectory,
    TaskInfo,
)
from synth_ai.task.datasets import TaskDatasetRegistry, TaskDatasetSpec
from synth_ai.task.rubrics import Rubric, load_rubric
from synth_ai.task.server import ProxyConfig, RubricBundle, TaskAppConfig
from synth_ai.task.vendors import normalize_vendor_keys

from ..gepa_benchmarks.common import call_chat_completion

REPO_ROOT = Path(__file__).resolve().parents[3]

HUMANEVAL_DATASET = "openai/humaneval"
DEFAULT_SPLIT = "test"
AVAILABLE_SPLITS: tuple[str, ...] = ("test",)

humaneval_router = APIRouter()


HUMANEVAL_DATASET_SPEC = TaskDatasetSpec(
    id="humaneval",
    name="HumanEval Code Generation",
    version="1.0.0",
    splits=list(AVAILABLE_SPLITS),
    default_split=DEFAULT_SPLIT,
    description="Code generation benchmark with 164 Python programming problems.",
)


def _extract_code_from_response(text: str) -> str:
    """Extract Python code from model response."""
    if not text:
        return ""
    
    # Look for code blocks
    code_block_pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    
    # Look for function definitions (common pattern)
    func_pattern = r"def\s+\w+.*?(?=\n\n|\n```|\Z)"
    matches = re.findall(func_pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    
    # Fallback: return text as-is (might be just code)
    return text.strip()


def _execute_code_safely(code: str, test_code: str, timeout: int = 10) -> tuple[bool, str]:
    """Execute code safely with timeout and capture output.
    
    Returns:
        (success: bool, output: str)
    """
    if not code or not test_code:
        return False, "Missing code or test"
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Write code + test
            full_code = f"{code}\n\n{test_code}\n"
            f.write(full_code)
            temp_path = f.name
        
        try:
            # Execute with timeout
            result = subprocess.run(
                ["python3", temp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "PYTHONPATH": ""},  # Restrict imports
            )
            
            # Check if tests passed (exit code 0)
            success = result.returncode == 0
            output = result.stdout + result.stderr
            
            return success, output
        finally:
            # Clean up temp file
            with contextlib.suppress(OSError):
                os.unlink(temp_path)
                
    except subprocess.TimeoutExpired:
        return False, f"Execution timed out after {timeout}s"
    except Exception as e:
        return False, f"Execution error: {str(e)}"


class HumanEvalDataset:
    """Lazy loader and sampler for the HumanEval dataset."""

    def __init__(self) -> None:
        self._splits: dict[str, Any] = {}

    def _load_split(self, split: str):
        if split not in AVAILABLE_SPLITS:
            raise ValueError(f"Unknown split '{split}'. Available: {AVAILABLE_SPLITS}")
        if split not in self._splits:
            try:
                self._splits[split] = load_dataset(HUMANEVAL_DATASET, split=split)
            except Exception as exc:  # pragma: no cover - network/dataset errors
                raise RuntimeError(
                    f"Failed to download HumanEval split '{split}'. "
                    f"Ensure network access to Hugging Face datasets."
                ) from exc
        return self._splits[split]

    def ensure_ready(self, required_splits: Sequence[str]) -> None:
        for split in required_splits:
            self._load_split(split)

    def size(self, split: str) -> int:
        dataset = self._load_split(split)
        return len(dataset)

    def sample(self, *, split: str, index: int) -> dict[str, Any]:
        dataset = self._load_split(split)
        size = len(dataset)
        if size == 0:
            raise RuntimeError(f"HumanEval split '{split}' is empty")
        idx = int(index) % size
        row = dataset[int(idx)]

        prompt = str(row.get("prompt") or "")
        test = str(row.get("test") or "")
        entry_point = str(row.get("entry_point") or "")
        
        # HumanEval format: prompt is the function signature + docstring
        # test contains the test cases
        # We need to extract the function name from entry_point

        return {
            "index": idx,
            "split": split,
            "prompt": prompt,
            "test": test,
            "entry_point": entry_point,
            "canonical_solution": str(row.get("canonical_solution") or ""),
        }


async def rollout_executor(request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
    dataset: HumanEvalDataset = fastapi_request.app.state.humaneval_dataset

    split = str(((request.env.config or {}).get("split")) or DEFAULT_SPLIT)
    seed = request.env.seed or 0

    sample = dataset.sample(split=split, index=seed)
    observation = {
        "prompt": sample["prompt"],
        "entry_point": sample["entry_point"],
        "index": sample["index"],
        "split": sample["split"],
    }

    placeholders = {
        "prompt": sample["prompt"],
        "entry_point": sample["entry_point"],
    }

    default_messages = [
        {
            "role": "system",
            "pattern": (
                "You are a Python programming assistant. Complete the function implementation "
                "based on the provided prompt. Return only the Python code, no explanations."
            ),
        },
        {
            "role": "user",
            "pattern": (
                "Complete the following function:\n\n{prompt}\n\n"
                "Provide only the function implementation code."
            ),
        },
    ]

    tool_calls: list[dict[str, Any]] = []
    response_json: dict[str, Any] | None = None
    response_text = ""
    error_info: dict[str, Any] = {}

    try:
        response_text, response_json, _ = await call_chat_completion(
            request.policy.config or {},
            placeholders,
            default_messages,
        )
    except HTTPException as http_err:  # pragma: no cover - passthrough to metrics
        error_info = {"error": str(http_err.detail), "code": http_err.status_code}
    except Exception as exc:  # pragma: no cover - defensive logging
        error_info = {"error": str(exc)}

    # Extract code from response
    generated_code = _extract_code_from_response(response_text)
    
    # Execute code with tests
    test_passed = False
    execution_output = ""
    if generated_code:
        test_passed, execution_output = _execute_code_safely(
            generated_code,
            sample["test"],
            timeout=10,
        )
    else:
        execution_output = "No code extracted from response"

    reward = 1.0 if test_passed else 0.0

    info_payload = {
        "test_passed": test_passed,
        "generated_code": generated_code[:500],  # Truncate for logging
        "execution_output": execution_output[:500],
        "entry_point": sample["entry_point"],
        "response_text": response_text[:500],
        "response_json": response_json,
        **error_info,
    }

    with contextlib.suppress(Exception):
        print(
            f"[HUMANEVAL_ROLLOUT] run_id={request.run_id} split={sample['split']} "
            f"index={sample['index']} test_passed={test_passed} "
            f"reward={reward:.3f}",
            flush=True,
        )

    step = RolloutStep(
        obs=observation,
        tool_calls=tool_calls,
        reward=reward,
        done=True,
        info=info_payload,
    )

    inference_url = (request.policy.config or {}).get("inference_url")

    trajectory = RolloutTrajectory(
        env_id=f"humaneval::{sample['split']}::{sample['index']}",
        policy_id=request.policy.policy_id or request.policy.policy_name or "policy",
        steps=[step],
        final={"observation": observation, "reward": reward},
        length=1,
        inference_url=str(inference_url or ""),
    )

    metrics = RolloutMetrics(
        episode_returns=[reward],
        mean_return=reward,
        num_steps=1,
        num_episodes=1,
        outcome_score=reward,
        events_score=reward,
        details={
            "test_passed": bool(test_passed),
        },
    )

    trace_payload = None
    include_trace = bool(
        (request.record and getattr(request.record, "return_trace", False))
        or os.getenv("TASKAPP_TRACING_ENABLED")
    )
    if include_trace:
        trace_payload = {
            "session_id": str(uuid.uuid4()),
            "events_count": 1,
            "decision_rewards": [reward],
            "metadata": {
                "env": "humaneval",
                "split": sample["split"],
                "index": sample["index"],
                "test_passed": test_passed,
            },
        }

    return RolloutResponse(
        run_id=request.run_id,
        trajectories=[trajectory],
        branches={},
        metrics=metrics,
        aborted=False,
        ops_executed=2,
        trace=trace_payload,
    )


def build_dataset() -> tuple[TaskDatasetRegistry, HumanEvalDataset]:
    registry = TaskDatasetRegistry()
    dataset = HumanEvalDataset()
    dataset.ensure_ready([DEFAULT_SPLIT])
    registry.register(HUMANEVAL_DATASET_SPEC, lambda _spec: dataset, cache=True)
    return registry, dataset


def _base_task_info() -> TaskInfo:
    return TaskInfo(
        task={
            "id": "humaneval",
            "name": "HumanEval Code Generation",
            "version": "1.0.0",
            "action_space": {
                "type": "free_text",
                "description": "Generate Python code to complete function implementations.",
            },
        },
        environment="humaneval",
        dataset={
            **HUMANEVAL_DATASET_SPEC.model_dump(),
            "hf_dataset": HUMANEVAL_DATASET,
        },
        rubric={
            "version": "1",
            "criteria_count": 1,
            "source": "inline",
        },
        inference={
            "supports_proxy": True,
            "tool": None,
        },
        limits={"max_turns": 1},
        task_metadata={
            "format": "Python code generation with test execution",
        },
    )


def describe_taskset(dataset: HumanEvalDataset) -> Mapping[str, Any]:
    return {
        **HUMANEVAL_DATASET_SPEC.model_dump(),
        "hf_dataset": HUMANEVAL_DATASET,
        "sizes": {split: dataset.size(split) for split in AVAILABLE_SPLITS},
    }


def provide_task_instances(dataset: HumanEvalDataset, seeds: Sequence[int]) -> Iterable[TaskInfo]:
    base_info = _base_task_info()
    for seed in seeds:
        sample = dataset.sample(split=DEFAULT_SPLIT, index=seed)
        yield TaskInfo(
            task=base_info.task,
            environment=base_info.environment,
            dataset={
                **base_info.dataset,
                "split": sample["split"],
                "index": sample["index"],
            },
            rubric=base_info.rubric,
            inference=base_info.inference,
            limits=base_info.limits,
            task_metadata={
                **base_info.task_metadata,
                "entry_point": sample["entry_point"],
                "prompt": sample["prompt"][:200],  # Preview
            },
        )


OUTCOME_RUBRIC: Rubric = cast(
    Rubric,
    load_rubric(
        {
            "version": "1",
            "goal_text": "Generate correct Python code that passes all test cases.",
            "aggregation": "weighted_sum",
            "criteria": [
                {
                    "id": "test_pass_rate",
                    "description": "Generated code passes all test cases.",
                    "weight": 1.0,
                },
            ],
        }
    ),
)

EVENTS_RUBRIC: Rubric = cast(
    Rubric,
    load_rubric(
        {
            "version": "1",
            "goal_text": "Encourage valid Python code generation.",
            "aggregation": "weighted_sum",
            "criteria": [
                {
                    "id": "code_quality",
                    "description": "Generate syntactically valid Python code.",
                    "weight": 1.0,
                }
            ],
        }
    ),
)


def build_config() -> TaskAppConfig:
    registry, dataset = build_dataset()
    base_info = _base_task_info()

    proxy_keys = normalize_vendor_keys()
    proxy_config = ProxyConfig(
        enable_openai=proxy_keys.get("OPENAI_API_KEY") is not None,
        enable_groq=proxy_keys.get("GROQ_API_KEY") is not None,
        system_hint="Generate Python code to complete function implementations.",
    )

    config = TaskAppConfig(
        app_id="humaneval",
        name="HumanEval Code Generation Task",
        description="HumanEval environment for evaluating prompt optimisers on code generation.",
        base_task_info=base_info,
        describe_taskset=lambda: describe_taskset(dataset),
        provide_task_instances=lambda seeds: provide_task_instances(dataset, seeds),
        rollout=rollout_executor,
        dataset_registry=registry,
        rubrics=RubricBundle(outcome=OUTCOME_RUBRIC, events=EVENTS_RUBRIC),
        proxy=proxy_config,
        routers=(humaneval_router,),
        app_state={"humaneval_dataset": dataset},
        cors_origins=["*"],
    )
    return config


register_task_app(
    entry=TaskAppEntry(
        app_id="humaneval",
        description="HumanEval code generation task app using openai/humaneval.",
        config_factory=build_config,
        aliases=("humaneval-code",),
        modal=ModalDeploymentConfig(
            app_name="synth-humaneval",
            pip_packages=(
                "datasets>=2.14.0",
                "fastapi>=0.115.0",
                "pydantic>=2.0.0",
                "httpx>=0.26.0",
            ),
            extra_local_dirs=((str(REPO_ROOT / "synth_ai"), "/opt/synth_ai_repo/synth_ai"),),
        ),
    )
)


if __name__ == "__main__":  # pragma: no cover - manual local run helper
    import argparse
    from synth_ai.task.server import run_task_app

    parser = argparse.ArgumentParser(description="Run the HumanEval task app locally")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8113)
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn autoreload")
    parser.add_argument(
        "--env-file",
        action="append",
        default=[],
        help="Additional .env files to load before startup",
    )
    args = parser.parse_args()

    default_env = Path(__file__).resolve().parents[2] / ".env"
    env_files = [str(default_env)] if default_env.exists() else []
    env_files.extend(args.env_file or [])

    run_task_app(
        build_config,
        host=args.host,
        port=args.port,
        reload=args.reload,
        env_files=env_files,
    )

