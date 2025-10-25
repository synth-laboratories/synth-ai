"""Task app configuration for a single-step math reasoning environment."""

from __future__ import annotations

import contextlib
import os
import random
import re
import uuid
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import httpx
from datasets import load_dataset
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
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
from synth_ai.task.errors import http_exception
from synth_ai.task.rubrics import Rubric, load_rubric
from synth_ai.task.server import ProxyConfig, RubricBundle, TaskAppConfig
from synth_ai.task.tracing_utils import (
    build_tracer_factory,
    resolve_sft_output_dir,
    resolve_tracing_db_url,
    tracing_env_enabled,
)
from synth_ai.task.vendors import normalize_vendor_keys
from synth_ai.tracing_v3.session_tracer import SessionTracer

REPO_ROOT = Path(__file__).resolve().parents[3]

_modal_volume_candidate = Path(
    os.getenv("MATH_MODAL_DATASET_DIR", "/modal_volumes/math_dataset")
).expanduser()
_modal_volume_root: Path | None = None
try:
    _modal_volume_candidate.mkdir(parents=True, exist_ok=True)
    _modal_volume_root = _modal_volume_candidate
except Exception:
    if _modal_volume_candidate.exists():
        _modal_volume_root = _modal_volume_candidate

if _modal_volume_root is not None:
    hf_cache_path = _modal_volume_root / "hf_cache"
    local_dataset_dir = _modal_volume_root / "jsonl"
    local_dataset_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MATH_DATASET_LOCAL_DIR", str(local_dataset_dir))
else:
    hf_cache_path = Path(
        os.getenv("MATH_DATASET_CACHE_DIR", str(REPO_ROOT / ".cache" / "hf-datasets"))
    ).expanduser()

hf_cache_path.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MATH_DATASET_CACHE_DIR", str(hf_cache_path))
os.environ.setdefault("HF_HOME", str(hf_cache_path))
os.environ.setdefault("HF_DATASETS_CACHE", str(hf_cache_path))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_cache_path))

HF_DATASETS_CACHE = hf_cache_path
DATASET_NAME = os.getenv("MATH_DATASET_NAME", "nlile/hendrycks-MATH-benchmark")
DATASET_CONFIG = os.getenv("MATH_DATASET_CONFIG", "")
DEFAULT_SPLIT = os.getenv("MATH_DATASET_DEFAULT_SPLIT", "train")
AVAILABLE_SPLITS: tuple[str, ...] = (
    DEFAULT_SPLIT,
    os.getenv("MATH_DATASET_VALIDATION_SPLIT", "test"),
    os.getenv("MATH_DATASET_TEST_SPLIT", "test"),
)
TOOL_NAME = "math_submit"
PROBLEM_KEYS: tuple[str, ...] = ("problem", "question", "prompt", "query")
SOLUTION_KEYS: tuple[str, ...] = ("solution", "answer", "final_answer", "solution_text")

REWARD_POSITIVE = float(os.getenv("MATH_REWARD_POSITIVE", "1.0"))
REWARD_NEGATIVE_NO_TOOL = float(os.getenv("MATH_REWARD_NEGATIVE_NO_TOOL", "-1.0"))
REWARD_NEGATIVE_NO_ANSWER = float(os.getenv("MATH_REWARD_NEGATIVE_NO_ANSWER", "-0.5"))

HF_TOKEN_ENV_KEYS: tuple[str, ...] = (
    "HF_DATASETS_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "HUGGINGFACE_TOKEN",
)

## Single-source dataset policy: use a single known-good HF dataset id by default.

MATH_DATASET_SPEC = TaskDatasetSpec(
    id="math_single_step",
    name="MATH Single Step",
    version="1.0.0",
    splits=list(dict.fromkeys(split for split in AVAILABLE_SPLITS if split)),
    default_split=DEFAULT_SPLIT,
    description="Single-step math reasoning problems sourced from the Hendrycks MATH dataset.",
)


_BOXED_MARKERS: tuple[str, ...] = ("\\boxed", "boxed")


def _extract_boxed(text: str) -> str | None:
    if not text:
        return None
    for marker in _BOXED_MARKERS:
        start = text.find(marker)
        if start == -1:
            continue
        brace_start = text.find("{", start)
        if brace_start == -1:
            continue
        depth = 1
        idx = brace_start + 1
        while idx < len(text) and depth > 0:
            ch = text[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            idx += 1
        if depth == 0:
            return text[brace_start + 1 : idx - 1].strip()
    return None


_FRAC_PATTERN = re.compile(r"\\?frac\{([^{}]+)\}\{([^{}]+)\}")
_SQRT_PATTERN = re.compile(r"\\?sqrt\{([^{}]+)\}")


def _normalize_final_answer(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    boxed = _extract_boxed(raw)
    if boxed:
        raw = boxed
    raw = raw.strip().strip("$")
    raw = raw.replace("\\left", "").replace("\\right", "")
    raw = raw.replace("\\!", "").replace("\\,", " ").replace("\\;", " ")
    raw = raw.replace("left", "").replace("right", "")
    raw = raw.replace("\\times", "*").replace("\\cdot", "*")
    raw = raw.replace("\\pi", "pi").replace("\\theta", "theta").replace("\\phi", "phi")
    raw = raw.replace("\\pm", "+/-").replace("\\mp", "-/+")
    raw = raw.replace("^{\\circ}", "deg").replace("^\\circ", "deg").replace("\\circ", "deg")

    def _frac_sub(match: re.Match[str]) -> str:
        num = match.group(1).strip()
        den = match.group(2).strip()
        return f"({num})/({den})"

    def _sqrt_sub(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        return f"sqrt({inner})"

    raw = _FRAC_PATTERN.sub(_frac_sub, raw)
    raw = _SQRT_PATTERN.sub(_sqrt_sub, raw)
    raw = raw.replace("\\", "")
    raw = raw.replace("{", "").replace("}", "")
    raw = raw.replace(" ", "")
    raw = raw.rstrip(".")
    return raw


class MathDataset:
    """Lazy Hugging Face dataset loader for EleutherAI/math splits."""

    def __init__(self, *, name: str, config: str, splits: Sequence[str]) -> None:
        self.name = name
        self.config = config
        self.splits = [split for split in splits if split]
        self._cache: dict[str, Any] = {}
        self._local_dir = os.getenv("MATH_DATASET_LOCAL_DIR")
        self._hf_token: str | None = None
        for key in HF_TOKEN_ENV_KEYS:
            value = os.getenv(key)
            if value:
                trimmed = value.strip()
                if trimmed:
                    self._hf_token = trimmed
                    break
        # No multi-candidate fallback: enforce explicit dataset id

    def _local_file_for_split(self, split: str) -> Path | None:
        specific = os.getenv(f"MATH_DATASET_LOCAL_{split.upper()}_FILE")
        if specific:
            path = Path(specific).expanduser()
            if path.exists():
                return path
        if self._local_dir:
            candidate = Path(self._local_dir).expanduser() / f"{split}.jsonl"
            if candidate.exists():
                return candidate
        return None

    def _load_split(self, split: str):
        # Treat 'validation' as an alias for 'test' for datasets without a separate validation split
        if split not in self.splits and split.lower() == "validation":
            split = "test"
        if split not in self.splits:
            raise ValueError(f"Unknown split '{split}'. Available: {self.splits}")
        if split not in self._cache:
            local_file = self._local_file_for_split(split)
            if local_file is not None:
                dataset = load_dataset(
                    "json", data_files=str(local_file), cache_dir=str(HF_DATASETS_CACHE)
                )
                self._cache[split] = dataset["train"]
            else:
                try:
                    load_kwargs: dict[str, Any] = {"split": split}
                    if self.config:
                        load_kwargs["name"] = self.config
                    if self._hf_token:
                        load_kwargs["use_auth_token"] = self._hf_token
                    ds = load_dataset(self.name, cache_dir=str(HF_DATASETS_CACHE), **load_kwargs)
                    self._cache[split] = ds
                    if self._local_dir:
                        local_dir = Path(self._local_dir).expanduser()
                        target = local_dir / f"{split}.jsonl"
                        if not target.exists() and hasattr(ds, "to_json"):
                            tmp_path = target.with_name(target.name + ".tmp")
                            try:
                                local_dir.mkdir(parents=True, exist_ok=True)
                                ds.to_json(str(tmp_path))
                                tmp_path.replace(target)
                            except Exception:
                                with contextlib.suppress(FileNotFoundError):
                                    tmp_path.unlink()
                except Exception as exc:
                    hints = [
                        "Failed to download MATH dataset from Hugging Face.",
                        f"Dataset: {self.name} | Config: {self.config or 'none'} | Split: {split}",
                        "If this persists, verify MATH_DATASET_NAME/MATH_DATASET_CONFIG or set MATH_DATASET_LOCAL_DIR to pre-downloaded JSONL files.",
                    ]
                    raise RuntimeError(" ".join(hints)) from exc
        return self._cache[split]

    def sample(self, *, split: str, index: int | None = None) -> dict[str, Any]:
        dataset = self._load_split(split)
        if len(dataset) == 0:
            raise RuntimeError(f"Dataset split '{split}' is empty")
        if index is None:
            index = random.randint(0, len(dataset) - 1)
        idx = int(index) % len(dataset)
        item = dataset[int(idx)]

        raw_problem = ""
        for key in PROBLEM_KEYS:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                raw_problem = value.strip()
                break
        if not raw_problem:
            raise RuntimeError(f"Sample missing problem field for split '{split}' index {idx}")

        solution_value: Any = None
        for key in SOLUTION_KEYS:
            if key in item:
                solution_value = item[key]
                break
        if solution_value is None:
            raise RuntimeError(f"Sample missing solution field for split '{split}' index {idx}")

        # Solutions can contain reasoning and final answer; take final line by convention
        if isinstance(solution_value, list):
            solution_text = "\n".join(str(part) for part in solution_value)
        else:
            solution_text = str(solution_value)
        lines = [line.strip() for line in solution_text.strip().splitlines() if line.strip()]
        final_line = ""
        for line in reversed(lines):
            lowered = line.lower()
            if "boxed" in lowered or "answer" in lowered:
                final_line = line
                break
        if not final_line and lines:
            final_line = lines[-1]
        candidate_answer = final_line or solution_text.strip()
        normalized_answer = _normalize_final_answer(candidate_answer)
        if not normalized_answer:
            normalized_answer = _normalize_final_answer(solution_text)
        return {
            "index": idx,
            "split": split,
            "problem": raw_problem,
            "answer": normalized_answer,
            "raw_solution": solution_text,
        }

    def size(self, split: str) -> int:
        dataset = self._load_split(split)
        return len(dataset)

    def ensure_ready(self, required_splits: Sequence[str]) -> None:
        errors: list[str] = []
        for split in required_splits:
            if not split:
                continue
            try:
                self._load_split(split)
            except Exception as exc:
                errors.append(f"{split}: {exc}")
        if errors:
            raise RuntimeError("Dataset preparation failed:\n" + "\n".join(errors))


@dataclass
class MathEnvState:
    env_id: str
    split: str
    index: int
    problem: str
    answer: str
    raw_solution: str
    done: bool = False


class MathEnvironmentManager:
    """Stores in-flight environment state keyed by env_id."""

    def __init__(self, dataset: MathDataset) -> None:
        self.dataset = dataset
        self._states: dict[str, MathEnvState] = {}

    def create(self, *, split: str, index: int | None, seed: int | None) -> MathEnvState:
        if index is None and seed is not None:
            index = seed
        sample = self.dataset.sample(split=split, index=index)
        env_id = str(uuid.uuid4())
        state = MathEnvState(
            env_id=env_id,
            split=split,
            index=int(sample["index"]),
            problem=sample["problem"],
            answer=sample["answer"],
            raw_solution=sample["raw_solution"],
        )
        self._states[env_id] = state
        return state

    def get(self, env_id: str) -> MathEnvState:
        if env_id not in self._states:
            raise KeyError(f"Unknown env_id: {env_id}")
        return self._states[env_id]

    def terminate(self, env_id: str) -> None:
        self._states.pop(env_id, None)


class InitializePayload(BaseModel):
    seed: int | None = None
    config: dict[str, Any] = Field(default_factory=dict)


def _observation_from_state(state: MathEnvState) -> dict[str, Any]:
    return {
        "problem": state.problem,
        "split": state.split,
        "index": state.index,
    }


def _score_submission(
    state: MathEnvState, tool_calls: Sequence[Mapping[str, Any]]
) -> tuple[float, str, bool]:
    if not tool_calls:
        return REWARD_NEGATIVE_NO_TOOL, "missing_tool_call", False
    call = tool_calls[0]
    tool_name = str(call.get("tool") or "").strip()
    if tool_name != TOOL_NAME:
        return REWARD_NEGATIVE_NO_TOOL, "wrong_tool", False
    args = call.get("args") or {}
    answer = _normalize_final_answer(str(args.get("answer") or ""))
    if not answer:
        return REWARD_NEGATIVE_NO_ANSWER, "blank_answer", False
    is_correct = answer == state.answer
    return (
        (REWARD_POSITIVE if is_correct else 0.0),
        ("correct" if is_correct else "incorrect"),
        is_correct,
    )


math_router = APIRouter()


def _preview_tool_calls(tool_calls: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return a compact, log-friendly preview of tool calls.

    Truncates long fields to avoid noisy logs and leaking excessive content.
    """
    preview: list[dict[str, Any]] = []
    for call in list(tool_calls or [])[:3]:
        args = dict(call.get("args") or {})
        answer = str(args.get("answer") or "")
        # Hard truncate to keep logs compact
        answer_short = answer[:120] + ("…" if len(answer) > 120 else "")
        preview.append(
            {
                "tool": call.get("tool"),
                "answer": answer_short,
            }
        )
    return preview


def _event_and_outcome_components(
    tool_calls: Sequence[Mapping[str, Any]], *, correct: bool, reward: float
) -> dict[str, float]:
    """Approximate component-wise scores for RL-style logs.

    - env:     task-level scalar reward (our single-step outcome)
    - rubric_event: 1.0 if a valid tool call with non-empty answer was made else 0.0
    - rubric_outcome: 1.0 if final answer was correct else 0.0
    """
    has_valid_tool = False
    if tool_calls:
        first = tool_calls[0] or {}
        if str(first.get("tool") or "") == TOOL_NAME:
            args = first.get("args") or {}
            ans = str(args.get("answer") or "").strip()
            has_valid_tool = bool(ans)
    return {
        "env": float(reward),
        "rubric_event": 1.0 if has_valid_tool else 0.0,
        "rubric_outcome": 1.0 if bool(correct) else 0.0,
    }


@math_router.post("/env/math/initialize")
async def initialize_env(request: Request, payload: InitializePayload) -> dict[str, Any]:
    manager: MathEnvironmentManager = request.app.state.math_env_manager
    split = str(payload.config.get("split") or DEFAULT_SPLIT)
    seed = payload.seed
    index = None
    if payload.config.get("index") is not None:
        index = int(payload.config["index"])
    state = manager.create(split=split, index=index, seed=seed)
    return {
        "env_id": state.env_id,
        "observation": _observation_from_state(state),
        "info": {"raw_solution": state.raw_solution},
    }


@math_router.post("/env/math/step")
async def step_env(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
    manager: MathEnvironmentManager = request.app.state.math_env_manager
    env_id = str(payload.get("env_id") or "")
    if not env_id:
        raise HTTPException(status_code=400, detail="env_id required")
    try:
        state = manager.get(env_id)
    except KeyError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    action = payload.get("action") or {}
    tool_calls = action.get("tool_calls") or payload.get("tool_calls") or []
    reward, status, correct = _score_submission(state, tool_calls)
    with contextlib.suppress(Exception):
        print(
            "[MATH_STEP] env_id=",
            state.env_id,
            " split=",
            state.split,
            " index=",
            state.index,
            " calls=",
            _preview_tool_calls(tool_calls),
            " reward=",
            reward,
            " status=",
            status,
            " correct=",
            correct,
            " components=",
            _event_and_outcome_components(tool_calls, correct=correct, reward=reward),
            flush=True,
        )
    state.done = True

    observation = _observation_from_state(state)
    observation["status"] = status
    return {
        "observation": observation,
        "done": True,
        "reward": reward,
        "info": {
            "correct": correct,
            "expected_answer": state.answer,
            "raw_solution": state.raw_solution,
        },
    }


@math_router.post("/env/math/terminate")
async def terminate_env(request: Request, payload: dict[str, Any]) -> dict[str, Any]:
    manager: MathEnvironmentManager = request.app.state.math_env_manager
    env_id = str(payload.get("env_id") or "")
    if env_id:
        manager.terminate(env_id)
    return {"ok": True}


def _resolve_inference_url(base_url: str) -> str:
    normalized = (base_url or "").rstrip("/")
    if not normalized:
        raise RuntimeError("policy.config.inference_url required")
    if normalized.endswith("/v1/chat/completions"):
        return normalized
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


async def _call_inference(
    policy_config: Mapping[str, Any], observation: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inference_url = str(policy_config.get("inference_url") or "").rstrip("/")
    if not inference_url:
        raise RuntimeError("policy.config.inference_url required for rollout")
    model = policy_config.get("model")
    max_tokens = policy_config.get("max_tokens", 512)
    temperature = policy_config.get("temperature", 0.0)
    top_p = policy_config.get("top_p", 1.0)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a math solver. Read the problem carefully and respond with a single"
                f" tool call using the function `{TOOL_NAME}`."
                "\nRules:\n"
                "- Do all reasoning internally.\n"
                "- The tool call must include ONLY the final numeric or simplified answer in the"
                " `answer` field.\n"
                "- DO NOT include explanations, units, or extra text in the answer."
            ),
        },
        {
            "role": "user",
            "content": (
                "Problem:\n"
                + str(observation.get("problem") or "")
                + "\nSubmit the final answer via the tool call."
            ),
        },
    ]

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": TOOL_NAME,
                    "description": "Submit the final answer for the math problem.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answer": {
                                "type": "string",
                                "description": "Final answer in simplest form",
                            },
                            "explanation": {
                                "type": "string",
                                "description": "Optional explanation of reasoning",
                            },
                        },
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": TOOL_NAME}},
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    final_url = _resolve_inference_url(inference_url)
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        response = await client.post(final_url, json=payload)
    try:
        data = response.json()
    except Exception as exc:
        raise http_exception(
            502,
            "inference_invalid_response",
            "Inference provider returned invalid JSON",
            extra={"body": response.text[:800]},
        ) from exc
    if response.status_code >= 500:
        raise http_exception(
            502,
            "inference_upstream_error",
            "Inference provider returned an error",
            extra={"status": response.status_code, "body": data},
        )
    if response.status_code >= 400:
        raise http_exception(
            400,
            "inference_request_invalid",
            "Invalid inference request",
            extra={"status": response.status_code, "body": data},
        )

    tool_calls = []
    choices = data.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        raw_calls = message.get("tool_calls") or []
        for call in raw_calls:
            function = call.get("function") or {}
            name = function.get("name")
            arguments = function.get("arguments")
            parsed_args: dict[str, Any]
            if isinstance(arguments, str):
                try:
                    import json

                    parsed_args = json.loads(arguments)
                except Exception:
                    parsed_args = {}
            elif isinstance(arguments, MutableMapping):
                parsed_args = dict(arguments)
            else:
                parsed_args = {}
            tool_calls.append({"tool": name, "args": parsed_args})
    # Lightweight provider-side logging
    with contextlib.suppress(Exception):
        print(
            "[MATH_INFER] model=",
            model,
            " calls=",
            _preview_tool_calls(tool_calls),
            flush=True,
        )
    return tool_calls, data


async def rollout_executor(request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
    dataset: MathDataset = fastapi_request.app.state.math_dataset
    split = str(((request.env.config or {}).get("split")) or DEFAULT_SPLIT)
    sample = dataset.sample(split=split, index=request.env.seed)

    observation = {
        "problem": sample["problem"],
        "split": sample["split"],
        "index": sample["index"],
    }

    tool_calls: list[dict[str, Any]] = []
    inference_payload: dict[str, Any] | None = None
    error_info: dict[str, Any] = {}
    try:
        tool_calls, inference_payload = await _call_inference(
            request.policy.config or {}, observation
        )
    except HTTPException as http_err:
        tool_calls = []
        error_info = {"error": http_err.detail, "code": http_err.status_code}
    except Exception as exc:
        tool_calls = []
        error_info = {"error": str(exc)}

    reward, status, correct = _score_submission(
        MathEnvState(
            env_id="rollout",
            split=sample["split"],
            index=sample["index"],
            problem=sample["problem"],
            answer=sample["answer"],
            raw_solution=sample["raw_solution"],
        ),
        tool_calls,
    )

    # Log a concise summary so we can debug reward=0 issues in production
    with contextlib.suppress(Exception):
        print(
            "[MATH_ROLLOUT] run=",
            request.run_id,
            " split=",
            sample["split"],
            " index=",
            sample["index"],
            " calls=",
            _preview_tool_calls(tool_calls),
            " reward=",
            reward,
            " status=",
            status,
            " correct=",
            correct,
            " components=",
            _event_and_outcome_components(tool_calls, correct=correct, reward=reward),
            flush=True,
        )

    step = RolloutStep(
        obs=observation,
        tool_calls=tool_calls,
        reward=reward,
        done=True,
        info={
            "expected_answer": sample["answer"],
            "status": status,
            "correct": correct,
            "raw_solution": sample["raw_solution"],
            "tool_call_preview": _preview_tool_calls(tool_calls),
            **error_info,
        },
    )

    trajectory = RolloutTrajectory(
        env_id=f"math::{sample['split']}::{sample['index']}",
        policy_id=request.policy.policy_id or "policy",
        steps=[step],
        final={
            "observation": {**observation, "status": status},
            "reward": reward,
        },
        length=1,
    )
    metrics = RolloutMetrics(
        episode_returns=[reward],
        mean_return=reward,
        num_steps=1,
        num_episodes=1,
        outcome_score=reward,
        events_score=reward,
        details={"status": status, "correct": correct},
    )

    # Include a minimal trace when requested or tracing is enabled via env
    include_trace = bool(
        (request.record and getattr(request.record, "return_trace", False))
        or os.getenv("TASKAPP_TRACING_ENABLED")
    )
    trace_payload = None
    if include_trace:
        try:
            # Minimal structured trace for assertions
            trace_payload = {
                "session_id": str(uuid.uuid4()),
                "events_count": 1,
                "decision_rewards": [reward],
                "lm_calls": (
                    [{"prompt": str(observation.get("problem", "")), "response": str(tool_calls)}]
                    if tool_calls
                    else []
                ),
                "metadata": {
                    "env": "math_single_step",
                    "split": sample["split"],
                    "index": sample["index"],
                    "status": status,
                },
            }
        except Exception:
            trace_payload = None

    return RolloutResponse(
        run_id=request.run_id,
        trajectories=[trajectory],
        branches={},
        metrics=metrics,
        aborted=False,
        ops_executed=2,
        trace=trace_payload,
    )


def build_dataset() -> tuple[TaskDatasetRegistry, MathDataset]:
    registry = TaskDatasetRegistry()
    dataset = MathDataset(name=DATASET_NAME, config=DATASET_CONFIG, splits=AVAILABLE_SPLITS)
    # Ensure default split is available when the task app boots
    try:
        dataset.ensure_ready([DEFAULT_SPLIT])
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialise math dataset. Set MATH_DATASET_LOCAL_DIR or ensure network access.\n"
            f"Underlying error: {exc}"
        ) from exc
    registry.register(MATH_DATASET_SPEC, lambda _spec: dataset, cache=True)
    return registry, dataset


def _base_task_info() -> TaskInfo:
    return TaskInfo(
        task={"id": "math_single_step", "name": "Math Single Step", "version": "1.0.0"},
        environments=["math"],
        action_space={
            "type": "tool_call",
            "tools": [
                {
                    "name": TOOL_NAME,
                    "description": "Submit the final answer for the math problem.",
                    "schema": {"answer": "string"},
                }
            ],
            "max_calls": 1,
        },
        observation={
            "summary": "Single math word problem presented as plain text.",
            "keys": ["problem"],
        },
        dataset={
            **MATH_DATASET_SPEC.model_dump(),
            "hf_dataset": DATASET_NAME,
            "hf_config": DATASET_CONFIG,
        },
        rubric={
            "version": "1",
            "criteria_count": 1,
            "source": "inline",
        },
        inference={
            "supports_proxy": True,
            "tool": {"name": TOOL_NAME, "parallel_tool_calls": False},
        },
        capabilities={
            "supports_rollout": True,
            "supports_env_lifecycle": True,
            "requires_api_key_header": True,
        },
        limits={"max_turns": 1},
    )


OUTCOME_RUBRIC: Rubric = cast(
    Rubric,
    load_rubric(
        {
            "version": "1",
            "goal_text": "Encourage correct single-step math answers via tool calls.",
            "aggregation": "weighted_sum",
            "criteria": [
                {
                    "id": "correct_answer",
                    "description": "Submit the correct final answer using the math_submit tool.",
                    "weight": 1.0,
                }
            ],
        }
    ),
)

EVENTS_RUBRIC: Rubric = cast(
    Rubric,
    load_rubric(
        {
            "version": "1",
            "goal_text": "Penalize missing or malformed tool calls.",
            "aggregation": "weighted_sum",
            "criteria": [
                {
                    "id": "tool_usage",
                    "description": "Make exactly one tool call with an answer string.",
                    "weight": 1.0,
                }
            ],
        }
    ),
)


def describe_taskset(dataset: MathDataset) -> dict[str, Any]:
    return {
        **MATH_DATASET_SPEC.model_dump(),
        "hf_dataset": DATASET_NAME,
        "hf_config": DATASET_CONFIG,
        "sizes": {split: dataset.size(split) for split in dataset.splits},
    }


def provide_task_instances(dataset: MathDataset, seeds: Sequence[int]) -> Iterable[TaskInfo]:
    info = _base_task_info()
    for seed in seeds:
        sample = dataset.sample(split=DEFAULT_SPLIT, index=seed)
        yield TaskInfo(
            task=info.task,
            environments=info.environments,
            action_space=info.action_space,
            observation={**info.observation, "sample_index": sample["index"]},
            dataset={
                **info.dataset,
                "split": sample["split"],
                "index": sample["index"],
            },
            rubric=info.rubric,
            inference=info.inference,
            capabilities=info.capabilities,
            limits=info.limits,
        )


def build_config() -> TaskAppConfig:
    registry, dataset = build_dataset()
    base_info = _base_task_info()

    tracing_enabled = tracing_env_enabled()
    tracing_db_url = resolve_tracing_db_url()
    tracer_factory = build_tracer_factory(
        SessionTracer, enabled=tracing_enabled, db_url=tracing_db_url
    )
    sft_output_dir = resolve_sft_output_dir()

    app_state: dict[str, Any] = {
        "math_dataset": dataset,
        "math_env_manager": MathEnvironmentManager(dataset),
        "tracing_enabled": tracing_enabled,
    }
    if tracer_factory is not None:
        app_state["session_tracer_factory"] = tracer_factory
    if sft_output_dir:
        app_state["sft_output_dir"] = sft_output_dir

    proxy_keys = normalize_vendor_keys()
    openai_key = proxy_keys.get("OPENAI_API_KEY")
    groq_key = proxy_keys.get("GROQ_API_KEY")
    proxy_config = ProxyConfig(
        enable_openai=openai_key is not None,
        enable_groq=groq_key is not None,
        system_hint=(
            "You must respond with a single math_submit tool call containing only the final answer."
        ),
    )

    config = TaskAppConfig(
        app_id="math-single-step",
        name="Math Single Step Task",
        description="Single-step math reasoning environment built on the MATH dataset.",
        base_task_info=base_info,
        describe_taskset=lambda: describe_taskset(dataset),
        provide_task_instances=lambda seeds: provide_task_instances(dataset, seeds),
        rollout=rollout_executor,
        dataset_registry=registry,
        rubrics=RubricBundle(outcome=OUTCOME_RUBRIC, events=EVENTS_RUBRIC),
        proxy=proxy_config,
        routers=(math_router,),
        app_state=app_state,
        cors_origins=["*"],
    )
    return config


register_task_app(
    entry=TaskAppEntry(
        app_id="math-single-step",
        description="Single-step math reasoning task app using EleutherAI/math dataset.",
        config_factory=build_config,
        aliases=("math-rl",),
        env_files=("examples/rl/.env",),
        modal=ModalDeploymentConfig(
            app_name="synth-math-single-step",
            pip_packages=(
                "datasets>=4.0.0",
                "fastapi>=0.115.0",
                "pydantic>=2.0.0",
                "httpx>=0.26.0",
                "requests>=2.32.0",
                "python-dotenv>=1.0.0",
                "diskcache>=5.6.3",
                "duckdb>=1.0.0",
                "ty>=0.0.1a5",
                "toml>=0.10.2",
                "aiosqlite>=0.21.0",
                "libsql>=0.1.8",
                "pynacl>=1.5.0",
                "sqlalchemy>=2.0.42",
            ),
            extra_local_dirs=(
                (str(REPO_ROOT / "synth_ai"), "/opt/synth_ai_repo/synth_ai"),
                (str(REPO_ROOT / "examples" / "rl"), "/opt/synth_ai_repo/examples/rl"),
            ),
            volume_mounts=(("math-dataset-cache", "/modal_volumes/math_dataset"),),
        ),
    )
)
