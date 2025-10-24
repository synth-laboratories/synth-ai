"""Task App configuration for the GRPO Enron email QA example."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from uuid import UUID, uuid4

from datasets import load_dataset
import httpx

from fastapi import HTTPException

from synth_ai.environments.examples.enron.environment import EnronEnvironment
from synth_ai.environments.examples.enron.taskset import (
    EnronTaskInstance,
    EnronTaskInstanceMetadata,
)
from synth_ai.environments.tasks.core import (
    Impetus,
    Intent,
    SplitInfo,
    TaskInstanceSet,
)
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
from synth_ai.task.rubrics import load_rubric
from synth_ai.task.server import ProxyConfig, RubricBundle, TaskAppConfig
from synth_ai.task.tracing_utils import (
    build_tracer_factory,
    resolve_sft_output_dir,
    resolve_tracing_db_url,
    tracing_env_enabled,
)
from synth_ai.tracing_v3.session_tracer import SessionTracer
from synth_ai.environments.environment.tools import EnvToolCall

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve()
REPO_ROOT = _HERE.parents[4]

DATASET_SPEC = TaskDatasetSpec(
    id="enron_email_qa",
    name="Enron Email QA",
    version="1.0.0",
    splits=["train", "test"],
    default_split="train",
    description="Question answering over a sample of Enron emails.",
)

HF_DATASET_ID = "corbt/enron_emails_sample_questions"
HF_CACHE_DIR = os.path.join(
    os.getenv("ENRON_DATASET_CACHE_DIR", str(REPO_ROOT / ".cache" / "hf-datasets"))
)

TOOLS = ["search_emails", "read_email", "answer_question", "terminate"]
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "qwen/qwen3-32b"
ENRON_SYSTEM_PROMPT = (
    "You are an Enron investigations analyst. Answer the user's question by reading emails. "
    "You can call tools to search the corpus, read specific messages, and submit a final answer. "
    "Use the tools deliberately, gather evidence before answering, and when confident call "
    "answer_question with your final answer. If you cannot find the answer after thorough search, "
    "answer_question with your best attempt noting uncertainty."
)


def _simplify(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _simplify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_simplify(v) for v in obj]
    return str(obj)


def _render_search_results(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No search results."
    lines = []
    for item in results[:5]:
        message_id = item.get("message_id") or item.get("id") or "<unknown>"
        snippet = (item.get("snippet") or item.get("snip") or "").strip()
        lines.append(f"- {message_id}: {snippet[:280]}")
    return "\n".join(lines)


def _render_email(email: dict[str, Any] | None) -> str:
    if not email:
        return "No email loaded."
    subject = email.get("subject", "<no subject>")
    from_addr = email.get("from_address") or email.get("from_addr") or "<unknown>"
    date = email.get("date", "<unknown date>")
    snippet = (email.get("body") or "")[:600]
    return f"Subject: {subject}\nFrom: {from_addr}\nDate: {date}\nBody Preview:\n{snippet}"


def _render_observation(obs: dict[str, Any]) -> str:
    lines = [
        f"Question: {obs.get('question', '')}",
        f"Already answered: {bool(obs.get('already_answered'))}",
        f"Available tools: {', '.join(obs.get('tools') or [])}",
        f"Inbox address: {obs.get('inbox_address', '<unknown>')}",
        f"Reward Δ: {obs.get('reward_last', 0)}   Total Reward: {obs.get('total_reward', 0)}",
    ]
    tool_error = obs.get("tool_error")
    if tool_error:
        lines.append(f"Last tool error: {tool_error}")
    search_results = obs.get("search_results") or []
    if search_results:
        lines.append("Search Results:")
        lines.append(_render_search_results(search_results))
    email = obs.get("email")
    if email:
        lines.append("Email Content:")
        lines.append(_render_email(email))
    gold = obs.get("gold_answer")
    if gold and obs.get("terminated"):
        lines.append(f"Gold Answer: {gold}")
    return "\n".join(lines)


def _conversation_message(role: str, content: Any, **metadata: Any) -> dict[str, Any]:
    if isinstance(content, (dict, list)):
        rendered = json.dumps(_simplify(content), ensure_ascii=False)
    else:
        rendered = str(content)
    message: dict[str, Any] = {"role": role, "content": rendered}
    message.update({k: v for k, v in metadata.items() if v is not None})
    return message


def _build_trace_payload_enron(
    run_id: str,
    request: RolloutRequest,
    steps: list[RolloutStep],
    metrics: RolloutMetrics,
    *,
    provider: str,
    model: str,
    conversation: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    created_at = datetime.now(timezone.utc)
    event_time = time.time()
    session_steps: list[dict[str, Any]] = []
    event_history: list[dict[str, Any]] = []
    markov_history: list[dict[str, Any]] = []
    for msg in conversation:
        event_time += 0.005
        markov_history.append(
            {
                "content": {"text": msg.get("content", "")},
                "message_type": msg.get("role", "system"),
                "time_record": {"event_time": event_time},
                "metadata": _simplify({k: v for k, v in msg.items() if k not in {"role", "content"}}),
            }
        )

    session_trace = {
        "session_id": run_id,
        "created_at": created_at.isoformat(),
        "metadata": {
            "task": "enron_email_qa",
            "provider": provider,
            "model": model,
            "policy": _simplify(request.policy.model_dump() if request.policy else {}),
            "env": _simplify(request.env.model_dump() if request.env else {}),
            **(_simplify(metadata or {})),
        },
        "session_time_steps": session_steps,
        "event_history": event_history,
        "markov_blanket_message_history": markov_history,
    }

    return {
        "version": 3,
        "session_trace": session_trace,
        "run_id": run_id,
        "policy_id": request.policy.policy_id or request.policy.policy_name,
        "reward": metrics.mean_return,
        "episode_returns": metrics.episode_returns,
        "mean_return": metrics.mean_return,
        "num_steps": metrics.num_steps,
    }


async def _call_groq_chat(
    client: httpx.AsyncClient,
    api_key: str,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    response = await client.post(
        GROQ_CHAT_URL,
        json=payload,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    if response.status_code >= 400:
        try:
            body = response.json()
        except Exception:
            body = {"raw": response.text}
        detail = {
            "status": response.status_code,
            "body": body,
            "headers": dict(response.headers),
        }
        raise HTTPException(status_code=response.status_code, detail=detail)
    data = response.json()
    return data, {
        "status": response.status_code,
        "headers": dict(response.headers),
        "body": data,
    }


def _load_taskset_blocking() -> TaskInstanceSet:
    """Build the Enron taskset synchronously."""

    cache_dir = Path(HF_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    ds_train = load_dataset(HF_DATASET_ID, split="train", cache_dir=cache_dir)
    ds_test = load_dataset(HF_DATASET_ID, split="test", cache_dir=cache_dir)

    def _metadata_from_row(row: dict[str, Any], split: str) -> EnronTaskInstance:
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        message_ids = row.get("message_ids") or []
        if not isinstance(message_ids, list):
            message_ids = list(message_ids)
        impetus = Impetus(instructions=question)
        intent = Intent(
            rubric={"goal": "Answer the question using the Enron emails."},
            gold_trajectories=None,
            gold_state_diff={"answer": answer},
        )
        metadata = EnronTaskInstanceMetadata(
            split=split,
            email_count=len(message_ids),
            message_ids=message_ids,
        )
        return EnronTaskInstance(
            id=uuid4(),
            impetus=impetus,
            intent=intent,
            metadata=metadata,
            is_reproducible=True,
            initial_engine_snapshot=row,
        )

    train_instances = [_metadata_from_row(r, "train") for r in ds_train]
    test_instances = [_metadata_from_row(r, "test") for r in ds_test]

    split_info = SplitInfo(
        val_instance_ids=set(),
        test_instance_ids={inst.id for inst in test_instances},
        _is_split_defined=True,
    )

    return TaskInstanceSet(
        name="Enron-QA",
        description="QA over Enron email dataset sample.",
        instances=train_instances + test_instances,
        split_info=split_info,
    )


def _safe_uuid(value: Any) -> UUID:
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except Exception:
        return UUID(int=0)


@dataclass
class EnronDataset:
    spec: TaskDatasetSpec

    def __post_init__(self) -> None:
        self._taskset = _load_taskset_blocking()
        self.instances: list[EnronTaskInstance] = list(self._taskset.instances)
        self.instance_ids = [str(_safe_uuid(inst.id)) for inst in self.instances]
        self.default_seed = 0
        self.seed_min = 0
        self.seed_max = max(len(self.instances) - 1, 0)

    def describe(self) -> dict[str, Any]:
        return {
            **self.spec.model_dump(),
            "instance_count": len(self.instances),
            "instance_ids": self.instance_ids[:50],
        }

    def instance_by_seed(self, seed: int | None) -> EnronTaskInstance:
        if not self.instances:
            raise ValueError("Enron dataset is empty.")
        if seed is None:
            index = 0
        else:
            index = int(seed) % len(self.instances)
        return self.instances[index]


def build_dataset() -> tuple[TaskDatasetRegistry, EnronDataset]:
    registry = TaskDatasetRegistry()
    dataset = EnronDataset(DATASET_SPEC)
    registry.register(DATASET_SPEC, lambda _spec: dataset, cache=True)
    return registry, dataset


def _base_task_info(dataset: EnronDataset) -> TaskInfo:
    return TaskInfo(
        task={"id": "enron_email_qa", "name": "Enron Email QA", "version": "1.0.0"},
        environment="enron",
        action_space={
            "type": "tool_calls",
            "tools": TOOLS,
            "description": "Tool-assisted QA workflow over an email corpus.",
        },
        observation={
            "summary": "Text observations describing the question, tool status, and last reward.",
            "format": "text",
        },
        dataset={**dataset.describe(), "default_seed": dataset.default_seed},
        rubric={
            "version": "1",
            "criteria_count": 1,
            "source": "inline",
            "aggregation": "weighted_sum",
        },
        inference={
            "supports_proxy": False,
            "endpoints": {},
            "tool": {"name": "enron_tools", "parallel_tool_calls": False},
        },
        limits={"max_ops": 0, "max_time_s": 900},
    )


OUTCOME_RUBRIC = load_rubric(
    {
        "version": "1",
        "goal_text": "Provide the correct answer to the question using the Enron emails.",
        "aggregation": "weighted_sum",
        "criteria": [
            {
                "id": "accuracy",
                "description": "Final answer matches the gold answer.",
                "weight": 1.0,
            }
        ],
    }
)

EVENTS_RUBRIC = load_rubric(
    {
        "version": "1",
        "goal_text": "Encourage efficient use of tools when exploring the corpus.",
        "aggregation": "weighted_sum",
        "criteria": [
            {
                "id": "tool_use",
                "description": "Use search, read, and answer tools deliberately.",
                "weight": 1.0,
            }
        ],
    }
)


def describe_taskset(dataset: EnronDataset) -> dict[str, Any]:
    return dataset.describe()


def provide_task_instances(
    dataset: EnronDataset, base_info: TaskInfo, seeds: Sequence[int]
) -> Iterable[TaskInfo]:
    infos: list[TaskInfo] = []
    base_observation = getattr(base_info, "observation", None)
    if hasattr(base_observation, "model_dump"):
        observation_template = base_observation.model_dump()
    elif isinstance(base_observation, dict):
        observation_template = dict(base_observation)
    else:
        observation_template = {}

    for seed in seeds:
        instance = dataset.instance_by_seed(seed)
        metadata = instance.metadata
        meta_dict = {
            "split": getattr(metadata, "split", None),
            "email_count": getattr(metadata, "email_count", None),
            "message_ids": getattr(metadata, "message_ids", None),
        }
        infos.append(
            TaskInfo(
                task=base_info.task,
                environment=base_info.environment,
                action_space=base_info.action_space,
                observation={
                    **observation_template,
                    "question": instance.impetus.instructions,
                },
                dataset={
                    **base_info.dataset.model_dump(),
                    "instance_id": str(_safe_uuid(instance.id)),
                    "metadata": meta_dict,
                },
                rubric=base_info.rubric,
                inference=base_info.inference,
                limits=base_info.limits,
            )
        )
    return infos


def _ensure_dataset_from_state(fastapi_request, fallback: EnronDataset) -> EnronDataset:
    if fastapi_request is None:
        return fallback
    dataset = getattr(getattr(fastapi_request, "app", None), "state", None)
    candidate = getattr(dataset, "dataset", None)
    return candidate or fallback


def _normalise_observation(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "observation"):
        obs = getattr(value, "observation")
        if isinstance(obs, dict):
            return obs
        return {"text": str(obs)}
    return {"text": str(value)}


async def rollout_executor(request: RolloutRequest, fastapi_request) -> RolloutResponse:
    policy_cfg = dict(request.policy.config or {})
    provider = str(policy_cfg.get("provider") or "").strip().lower()
    if provider == "groq":
        return await _rollout_with_groq(request, fastapi_request, policy_cfg)

    # Fallback: return initial observation but include minimal trace payload
    dataset = _ensure_dataset_from_state(fastapi_request, RUNTIME_DATASET)
    env_seed = getattr(request.env, "seed", None) if request and request.env else None
    instance = dataset.instance_by_seed(env_seed)
    env = EnronEnvironment(task_instance=instance)
    env.custom_obs = None
    try:
        initial_observation = await env.initialize()
    finally:
        with contextlib.suppress(Exception):
            await env.terminate()

    obs_dict = _normalise_observation(initial_observation)
    step = RolloutStep(
        obs=obs_dict,
        tool_calls=[],
        reward=0.0,
        done=True,
        truncated=None,
        info={"note": "No rollout executed; provider unset."},
    )
    # No inference_url for noop policy
    trajectory = RolloutTrajectory(
        env_id=request.env.env_id or "enron",
        policy_id=request.policy.policy_id or request.policy.policy_name or "noop-policy",
        steps=[step],
        final={"observation": obs_dict},
        length=1,
        inference_url=None,  # NEW: No inference for noop policy
        decision_samples=None,
    )
    metrics = RolloutMetrics(
        episode_returns=[0.0],
        mean_return=0.0,
        num_steps=1,
        num_episodes=1,
        outcome_score=None,
        events_score=None,
        details={"note": "Provider not configured; returning initial state."},
    )
    trace_payload = _build_trace_payload_enron(
        request.run_id,
        request,
        [step],
        metrics,
        provider="local",
        model=policy_cfg.get("model") or "noop",
        conversation=[
            _conversation_message("system", ENRON_SYSTEM_PROMPT),
            _conversation_message("user", _render_observation(obs_dict)),
        ],
        metadata={"mode": "noop"},
    )
    return RolloutResponse(
        run_id=request.run_id,
        trajectories=[trajectory],
        branches={},
        metrics=metrics,
        aborted=False,
        ops_executed=0,
        trace=trace_payload,
    )


def _prepare_tool_call(
    tool_name: str,
    raw_args: dict[str, Any],
    current_obs: dict[str, Any],
) -> EnvToolCall:
    if tool_name == "search_emails":
        keywords = raw_args.get("keywords")
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(",") if k.strip()]
        if not isinstance(keywords, list) or not keywords:
            raise ValueError("search_emails requires a non-empty list of keywords.")
        inbox = raw_args.get("inbox") or current_obs.get("inbox_address") or "investigator@enron.com"
        args = {
            "inbox": str(inbox),
            "keywords": [str(k) for k in keywords],
            "from_addr": raw_args.get("from_addr"),
            "to_addr": raw_args.get("to_addr"),
            "sent_after": raw_args.get("sent_after"),
            "sent_before": raw_args.get("sent_before"),
            "max_results": int(raw_args.get("max_results") or 5),
        }
        return EnvToolCall(tool="search_emails", args=args)

    if tool_name == "read_email":
        message_id = raw_args.get("message_id")
        if not message_id:
            raise ValueError("read_email requires 'message_id'.")
        return EnvToolCall(tool="read_email", args={"message_id": str(message_id)})

    if tool_name == "answer_question":
        answer = raw_args.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("answer_question requires a non-empty 'answer'.")
        return EnvToolCall(tool="answer_question", args={"answer": answer.strip()})

    if tool_name == "terminate":
        return EnvToolCall(tool="terminate", args={})

    raise ValueError(f"Unsupported tool '{tool_name}'")


async def _rollout_with_groq(
    request: RolloutRequest,
    fastapi_request,
    config: dict[str, Any],
) -> RolloutResponse:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY environment variable is required for Groq rollouts.",
        )

    dataset = _ensure_dataset_from_state(fastapi_request, RUNTIME_DATASET)
    env_seed = getattr(request.env, "seed", None) if request and request.env else None
    instance = dataset.instance_by_seed(env_seed)
    env = EnronEnvironment(task_instance=instance)
    env.custom_obs = None

    metadata_extra = {
        "split": getattr(instance.metadata, "split", None),
        "email_count": getattr(instance.metadata, "email_count", None),
        "message_ids": list(getattr(instance.metadata, "message_ids", []))[:10],
    }

    model = config.get("model") or DEFAULT_GROQ_MODEL
    temperature = float(config.get("temperature", 0.2) or 0.2)
    top_p = float(config.get("top_p", 0.8) or 0.8)
    max_tokens = int(config.get("max_tokens", 768) or 768)
    max_turns = int(config.get("max_turns", config.get("max_steps", 12)) or 12)

    tool_schemas = [
        {
            "type": "function",
            "function": {
                "name": "search_emails",
                "description": "Search the Enron corpus for emails matching keywords.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "inbox": {"type": "string", "description": "Email address performing the search."},
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "description": "Keywords to include in the search.",
                        },
                        "from_addr": {"type": "string"},
                        "to_addr": {"type": "string"},
                        "sent_after": {"type": "string", "description": "YYYY-MM-DD"},
                        "sent_before": {"type": "string", "description": "YYYY-MM-DD"},
                        "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
                    },
                    "required": ["keywords"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_email",
                "description": "Read the full contents of an email by message_id.",
                "parameters": {
                    "type": "object",
                    "properties": {"message_id": {"type": "string"}},
                    "required": ["message_id"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "answer_question",
                "description": "Submit the final answer to the investigation question.",
                "parameters": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "terminate",
                "description": "Terminate the investigation without answering.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
    ]

    steps: list[RolloutStep] = []
    conversation: list[dict[str, Any]] = []
    executed = 0
    try:
        observation = await env.initialize()
        obs_dict = _normalise_observation(observation)
        conversation.append(_conversation_message("system", ENRON_SYSTEM_PROMPT))
        conversation.append(_conversation_message("user", _render_observation(obs_dict)))

        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            for turn in range(max_turns):
                payload = {
                    "model": model,
                    "messages": conversation,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "tools": tool_schemas,
                    "tool_choice": "auto",
                }
                vendor_attempts: list[dict[str, Any]] = []
                response, response_meta = await _call_groq_chat(client, api_key, payload)
                vendor_attempts.append({"request": payload, "response": response_meta})

                choices = response.get("choices") or []
                if not choices:
                    break
                message = choices[0].get("message") or {}
                tool_calls = message.get("tool_calls") or []
                assistant_msg_meta = {"tool_calls": _simplify(tool_calls)} if tool_calls else {}
                conversation.append(
                    _conversation_message("assistant", message.get("content") or "", **assistant_msg_meta)
                )

                tool_call_records: list[dict[str, Any]] = []
                step_reward = 0.0
                done = False
                truncated = False

                if not tool_calls:
                    final_answer = (message.get("content") or "").strip()
                    if final_answer:
                        env_call = EnvToolCall(tool="answer_question", args={"answer": final_answer})
                        observation = await env.step(env_call)
                        executed += 1
                        obs_dict = _normalise_observation(observation)
                        step_reward += float(obs_dict.get("reward_last") or 0.0)
                        done = bool(obs_dict.get("terminated"))
                        truncated = bool(obs_dict.get("truncated"))
                        tool_call_records.append({"tool": "answer_question", "args": env_call.args})
                        conversation.append(
                            _conversation_message(
                                "tool",
                                {"result": "answer_submitted", "observation": obs_dict},
                                name="answer_question",
                            )
                        )
                    else:
                        break
                else:
                    for call in tool_calls:
                        func = call.get("function") or {}
                        name = func.get("name")
                        raw_args = func.get("arguments")
                        if isinstance(raw_args, str):
                            try:
                                parsed_args = json.loads(raw_args)
                            except json.JSONDecodeError:
                                parsed_args = {}
                        elif isinstance(raw_args, dict):
                            parsed_args = raw_args
                        else:
                            parsed_args = {}

                        env_call = _prepare_tool_call(name, parsed_args, obs_dict)
                        observation = await env.step(env_call)
                        executed += 1
                        obs_dict = _normalise_observation(observation)
                        reward_delta = float(obs_dict.get("reward_last") or 0.0)
                        step_reward += reward_delta
                        done = bool(obs_dict.get("terminated"))
                        truncated = bool(obs_dict.get("truncated"))
                        tool_call_records.append({"tool": env_call.tool, "args": env_call.args})
                        conversation.append(
                            _conversation_message(
                                "tool",
                                {
                                    "tool": env_call.tool,
                                    "args": env_call.args,
                                    "reward_delta": reward_delta,
                                    "observation": obs_dict,
                                },
                                name=env_call.tool,
                                tool_call_id=call.get("id"),
                            )
                        )
                        if done or truncated:
                            break

                conversation.append(_conversation_message("user", _render_observation(obs_dict)))

                step = RolloutStep(
                    obs=obs_dict,
                    tool_calls=tool_call_records,
                    reward=step_reward,
                    done=done,
                    truncated=truncated if truncated else None,
                    info={
                        "provider": "groq",
                        "model": model,
                        "vendor_attempts": vendor_attempts,
                        "turn": turn,
                    },
                )
                steps.append(step)

                if done or truncated:
                    break
    finally:
        with contextlib.suppress(Exception):
            await env.terminate()

    if steps:
        final_obs = steps[-1].obs
        total_reward = float(final_obs.get("total_reward") or 0.0)
    else:
        total_reward = 0.0

    metrics = RolloutMetrics(
        episode_returns=[total_reward],
        mean_return=total_reward if steps else 0.0,
        num_steps=len(steps),
        num_episodes=1,
        outcome_score=None,
        events_score=None,
        details={"provider": "groq", "model": model},
    )
    inference_url_groq = "https://api.groq.com/openai/v1/chat/completions"
    
    trajectory = RolloutTrajectory(
        env_id=request.env.env_id or "enron",
        policy_id=request.policy.policy_id or request.policy.policy_name or "enron-groq",
        steps=steps,
        final={"observation": steps[-1].obs if steps else {}},
        length=len(steps),
        inference_url=inference_url_groq,  # NEW: Required for trace correlation
        decision_samples=None,
    )
    trace_payload = _build_trace_payload_enron(
        request.run_id,
        request,
        steps,
        metrics,
        provider="groq",
        model=model,
        conversation=conversation,
        metadata=metadata_extra,
    )
    return RolloutResponse(
        run_id=request.run_id,
        trajectories=[trajectory],
        branches={},
        metrics=metrics,
        aborted=False,
        ops_executed=executed,
        trace=trace_payload,
    )


RUNTIME_DATASET: EnronDataset
registry, RUNTIME_DATASET = build_dataset()
BASE_INFO = _base_task_info(RUNTIME_DATASET)


def build_config() -> TaskAppConfig:
    tracing_enabled = tracing_env_enabled()
    tracing_db_url = resolve_tracing_db_url()
    tracer_factory = build_tracer_factory(
        SessionTracer, enabled=tracing_enabled, db_url=tracing_db_url
    )
    sft_output_dir = resolve_sft_output_dir()

    app_state: dict[str, Any] = {
        "dataset": RUNTIME_DATASET,
        "allowed_environments": ["enron"],
        "tracing_enabled": tracing_enabled,
    }
    if tracer_factory is not None:
        app_state["session_tracer_factory"] = tracer_factory
    if sft_output_dir:
        app_state["sft_output_dir"] = sft_output_dir

    if tracing_enabled:
        logger.info("[enron:tracing] enabled (db=%s)", tracing_db_url or "default")
    else:
        logger.info("[enron:tracing] disabled")
    if sft_output_dir:
        logger.info("[enron:sft] writing JSONL to %s", sft_output_dir)

    config = TaskAppConfig(
        app_id="grpo-enron",
        name="GRPO Enron Email QA Task App",
        description="Tool-assisted QA environment over Enron emails with GRPO-compatible endpoints.",
        base_task_info=BASE_INFO,
        describe_taskset=lambda: describe_taskset(RUNTIME_DATASET),
        provide_task_instances=lambda seeds: provide_task_instances(RUNTIME_DATASET, BASE_INFO, seeds),
        rollout=rollout_executor,
        dataset_registry=registry,
        rubrics=RubricBundle(outcome=OUTCOME_RUBRIC, events=EVENTS_RUBRIC),
        proxy=ProxyConfig(enable_openai=False, enable_groq=False),
        routers=(),
        app_state=app_state,
        cors_origins=["*"],
    )
    return config


register_task_app(
    entry=TaskAppEntry(
        app_id="grpo-enron",
        description="Enron email QA task app with rollout metadata endpoints.",
        config_factory=build_config,
        aliases=("enron", "enron-task"),
        env_files=(str(REPO_ROOT / "backend" / ".env.dev"),),
        modal=ModalDeploymentConfig(
            app_name="grpo-enron-task-app",
            python_version="3.11",
            pip_packages=(
                "fastapi>=0.100.0",
                "uvicorn>=0.23.0",
                "pydantic>=2.0.0",
                "httpx>=0.24.0",
                "python-dotenv>=1.0.1",
                "datasets>=2.10.0",
            ),
            extra_local_dirs=(
                (str(REPO_ROOT), "/opt/synth_ai_repo"),
                (str(REPO_ROOT / "synth_ai"), "/opt/synth_ai_repo/synth_ai"),
                (str(_HERE.parent), "/opt/synth_ai_repo/examples/task_apps/enron/task_app"),
            ),
            secret_names=("groq-api-key", "openai-api-key"),
            memory=8192,
            cpu=2.0,
            max_containers=4,
        ),
    )
)
