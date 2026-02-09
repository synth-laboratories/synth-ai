"""GEPA compatibility API backed by Synth AI optimization."""
# See: specifications/tanha/master_specification.md

from __future__ import annotations

import asyncio
import os
import re
import threading
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import httpx

from synth_ai.core.config.expansion import expand_gepa_config, gepa_candidate_to_initial_prompt
from synth_ai.sdk.localapi import InProcessTaskApp
from synth_ai.sdk.localapi._impl.rollout_helpers import build_rollout_response
from synth_ai.sdk.localapi._impl.server import TaskAppConfig, create_task_app
from synth_ai.sdk.localapi._impl.validators import normalize_inference_url
from synth_ai.sdk.localapi.auth import ensure_localapi_auth
from synth_ai.sdk.optimization.policy import PolicyOptimizationJob
from synth_ai.sdk.shared.models import detect_model_provider

from .core.adapter import DataInst
from .core.data_loader import DataLoader, ensure_loader
from .core.result import GEPAResult
from .utils.stop_condition import MaxMetricCallsStopper, StopperProtocol

CandidateSelectionStrategy = Literal["pareto", "current_best", "epsilon_greedy"]
BatchSamplerStrategy = Literal["epoch_shuffled"]
ModuleSelectorStrategy = Literal["round_robin", "all"]
FrontierType = Literal["instance", "objective", "hybrid", "cartesian"]


@dataclass(frozen=True)
class _DatasetSource:
    loader: DataLoader[Any, DataInst]
    ids: list[Any]

    def size(self) -> int:
        return len(self.ids)

    def get_by_index(self, index: int) -> DataInst:
        if not self.ids:
            raise ValueError("Dataset is empty.")
        resolved_index = index % len(self.ids)
        data_id = self.ids[resolved_index]
        items = self.loader.fetch([data_id])
        if not items:
            raise ValueError(f"Dataset returned no items for id {data_id}.")
        return items[0]


@dataclass(frozen=True)
class _DatasetBundle:
    train: _DatasetSource
    validation: _DatasetSource
    train_seeds: list[int]
    validation_seeds: list[int]
    validation_seed_offset: int | None

    def resolve_seed(self, seed: int) -> tuple[str, DataInst]:
        if self.validation_seed_offset is None:
            return "train", self.train.get_by_index(seed)
        if seed < self.validation_seed_offset:
            return "train", self.train.get_by_index(seed)
        return "val", self.validation.get_by_index(seed - self.validation_seed_offset)


def _build_dataset_source(data: Sequence[DataInst] | DataLoader[Any, DataInst]) -> _DatasetSource:
    loader = ensure_loader(data)
    ids = list(loader.all_ids())
    if not ids:
        raise ValueError("Dataset must contain at least one item.")
    return _DatasetSource(loader=loader, ids=ids)


def _build_dataset_bundle(
    trainset: Sequence[DataInst] | DataLoader[Any, DataInst],
    valset: Sequence[DataInst] | DataLoader[Any, DataInst] | None,
) -> _DatasetBundle:
    train_source = _build_dataset_source(trainset)
    if valset is None:
        validation_source = train_source
        train_seeds = list(range(train_source.size()))
        validation_seeds = list(range(train_source.size()))
        validation_seed_offset = None
    else:
        validation_source = _build_dataset_source(valset)
        train_seeds = list(range(train_source.size()))
        validation_seed_offset = train_source.size()
        validation_seeds = list(
            range(validation_seed_offset, validation_seed_offset + validation_source.size())
        )
    return _DatasetBundle(
        train=train_source,
        validation=validation_source,
        train_seeds=train_seeds,
        validation_seeds=validation_seeds,
        validation_seed_offset=validation_seed_offset,
    )


def _extract_input(sample: Any) -> str:
    if isinstance(sample, dict):
        for key in ("input", "question", "problem", "prompt", "query"):
            value = sample.get(key)
            if isinstance(value, str) and value.strip():
                return value
    raise ValueError(
        "Dataset item must include an input string (input/question/problem/prompt/query)."
    )


def _extract_answer(sample: Any) -> str:
    if isinstance(sample, dict):
        for key in ("answer", "expected_output", "output", "label"):
            value = sample.get(key)
            if isinstance(value, str) and value.strip():
                return value
    raise ValueError(
        "Dataset item must include an answer string (answer/expected_output/output/label)."
    )


def _extract_additional_context(sample: Any) -> dict[str, str]:
    if isinstance(sample, dict):
        value = sample.get("additional_context")
        if isinstance(value, dict):
            return {str(k): str(v) for k, v in value.items()}
    return {}


def _extract_seed_system_prompt(seed_candidate: dict[str, str]) -> str:
    for key in ("system_prompt", "instruction", "prompt", "system"):
        value = seed_candidate.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if len(seed_candidate) == 1:
        value = next(iter(seed_candidate.values()))
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(
        "seed_candidate must include a system prompt under 'system_prompt', 'instruction', "
        "'prompt', or a single key."
    )


def _extract_system_prompt_from_best_prompt(best_prompt: Any) -> str | None:
    if isinstance(best_prompt, str) and best_prompt.strip():
        return best_prompt.strip()
    if not isinstance(best_prompt, dict):
        return None
    for key in ("system_prompt", "instruction", "prompt"):
        value = best_prompt.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    messages = best_prompt.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                content = msg.get("pattern") or msg.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
    stages = best_prompt.get("stages")
    if isinstance(stages, dict):
        main_stage = stages.get("main")
        if isinstance(main_stage, dict):
            value = main_stage.get("instruction")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _parse_task_model(task_lm: str) -> tuple[str, str]:
    task_lm = task_lm.strip()
    if not task_lm:
        raise ValueError("task_lm must be a non-empty model identifier.")
    if "/" in task_lm:
        provider, model = task_lm.split("/", 1)
        provider = provider.strip().lower()
        model = model.strip()
        if not provider or not model:
            raise ValueError(f"Invalid task_lm identifier: {task_lm}")
        return provider, model
    provider = detect_model_provider(task_lm)
    return provider, task_lm


def _infer_proposer_effort(reflection_lm: str | None) -> str:
    override = os.getenv("SYNTH_GEPA_PROPOSER_EFFORT", "").strip()
    if override:
        return override.upper()
    if not reflection_lm:
        return "LOW"
    lowered = reflection_lm.lower()
    if "gpt-5" in lowered or "o1" in lowered or "o3" in lowered:
        return "HIGH"
    if "gpt-4" in lowered or "claude-3" in lowered or "gemini-2.5-pro" in lowered:
        return "MEDIUM"
    if "nano" in lowered or "tiny" in lowered:
        return "LOW_CONTEXT"
    return "LOW"


def _infer_num_generations(max_metric_calls: int | None) -> int:
    default_generations = int(os.getenv("SYNTH_GEPA_NUM_GENERATIONS", "5"))
    if max_metric_calls is None:
        return max(1, default_generations)
    return max(1, min(default_generations, max_metric_calls))


def _infer_children_per_generation(max_metric_calls: int | None) -> int:
    default_children = int(os.getenv("SYNTH_GEPA_CHILDREN_PER_GENERATION", "4"))
    if max_metric_calls is None:
        return max(1, default_children)
    return max(1, min(default_children, max_metric_calls))


def _warn_if_unsupported(name: str, value: Any, default: Any) -> None:
    if value != default:
        warnings.warn(
            f"gepa.optimize compatibility: '{name}' is not yet supported and will be ignored.",
            RuntimeWarning,
            stacklevel=2,
        )


def _resolve_max_metric_calls(
    max_metric_calls: int | None,
    stop_callbacks: StopperProtocol | Sequence[StopperProtocol] | None,
) -> int | None:
    if max_metric_calls is not None:
        return max_metric_calls
    if stop_callbacks is None:
        return None
    callbacks = stop_callbacks if isinstance(stop_callbacks, Sequence) else (stop_callbacks,)
    for callback in callbacks:
        if isinstance(callback, MaxMetricCallsStopper):
            return callback.max_metric_calls
    raise ValueError("stop_callbacks are not supported yet; use max_metric_calls instead.")


async def _call_llm(
    *,
    system_prompt: str,
    user_prompt: str,
    policy_config: dict[str, Any],
    request_headers: dict[str, str],
) -> tuple[str, str]:
    inference_url_raw = (
        policy_config.get("inference_url")
        or policy_config.get("api_base")
        or policy_config.get("base_url")
    )
    if not isinstance(inference_url_raw, str) or not inference_url_raw.strip():
        raise ValueError("Policy config must include an inference_url.")
    inference_url = normalize_inference_url(inference_url_raw)

    model = policy_config.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Policy config must include a model identifier.")

    api_key = policy_config.get("api_key")
    if not isinstance(api_key, str) or not api_key.strip():
        api_key = request_headers.get("x-api-key") or request_headers.get("authorization")

    headers: dict[str, str] = {"Content-Type": "application/json"}
    lower_url = inference_url.lower()
    if api_key:
        if "api.openai.com" in lower_url or "api.groq.com" in lower_url:
            headers["Authorization"] = (
                api_key if api_key.lower().startswith("bearer ") else f"Bearer {api_key}"
            )
        else:
            headers["X-API-Key"] = api_key

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if "temperature" in policy_config:
        payload["temperature"] = policy_config["temperature"]
    if "max_completion_tokens" in policy_config:
        payload["max_completion_tokens"] = policy_config["max_completion_tokens"]
    elif "max_tokens" in policy_config:
        payload["max_tokens"] = policy_config["max_tokens"]

    async def _post(url: str, req_headers: dict[str, str]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload, headers=req_headers)
            resp.raise_for_status()
            data = resp.json()
        if not isinstance(data, dict):
            raise ValueError("LLM response must be a JSON object.")
        return data

    try:
        data = await _post(inference_url, headers)
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code if exc.response is not None else None
        # The Synth interceptor path can occasionally return transient 5xx errors in dev.
        # Fall back to direct OpenAI calls when possible so local rollouts still complete.
        if status in {500, 502, 503, 504} and "api.openai.com" not in inference_url.lower():
            openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
            if openai_key:
                fallback_url = "https://api.openai.com/v1/chat/completions"
                fallback_headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {openai_key}",
                }
                data = await _post(fallback_url, fallback_headers)
                inference_url = fallback_url
            else:
                raise
        else:
            raise

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("LLM response missing choices.")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = ""
    if isinstance(message, dict):
        content = message.get("content") or ""
    return str(content).strip(), inference_url


def _run_async(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}

    def _runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:
            error["exc"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if error:
        raise error["exc"]
    return result.get("value")


def optimize(
    seed_candidate: dict[str, str],
    trainset: list[DataInst] | DataLoader[Any, DataInst],
    valset: list[DataInst] | DataLoader[Any, DataInst] | None = None,
    adapter: Any | None = None,
    task_lm: str | Any | None = None,
    evaluator: Any | None = None,
    reflection_lm: str | Any | None = None,
    candidate_selection_strategy: CandidateSelectionStrategy = "pareto",
    frontier_type: FrontierType = "instance",
    skip_perfect_score: bool = True,
    batch_sampler: BatchSamplerStrategy = "epoch_shuffled",
    reflection_minibatch_size: int | None = None,
    perfect_score: float = 1.0,
    reflection_prompt_template: str | dict[str, str] | None = None,
    custom_candidate_proposer: Any | None = None,
    module_selector: ModuleSelectorStrategy | str = "round_robin",
    use_merge: bool = False,
    max_merge_invocations: int = 5,
    merge_val_overlap_floor: int = 5,
    max_metric_calls: int | None = None,
    stop_callbacks: StopperProtocol | Sequence[StopperProtocol] | None = None,
    logger: Any | None = None,
    run_dir: str | None = None,
    callbacks: list[Any] | None = None,
    use_wandb: bool = False,
    wandb_api_key: str | None = None,
    wandb_init_kwargs: dict[str, Any] | None = None,
    use_mlflow: bool = False,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str | None = None,
    track_best_outputs: bool = False,
    display_progress_bar: bool = False,
    use_cloudpickle: bool = False,
    cache_evaluation: bool = False,
    seed: int = 0,
    raise_on_exception: bool = True,
    val_evaluation_policy: Any | None = None,
) -> GEPAResult[Any, Any]:
    """Run prompt optimization with a GEPA-compatible interface."""
    # See: specifications/tanha/master_specification.md

    if adapter is not None:
        raise ValueError("Custom adapters are not supported in Synth GEPA compatibility mode.")
    if evaluator is not None:
        raise ValueError("Custom evaluators are not supported in Synth GEPA compatibility mode.")
    if custom_candidate_proposer is not None:
        raise ValueError("Custom candidate proposers are not supported in Synth GEPA mode.")
    if use_merge:
        raise ValueError("Merge-based optimization is not supported in Synth GEPA mode.")

    _warn_if_unsupported("candidate_selection_strategy", candidate_selection_strategy, "pareto")
    _warn_if_unsupported("frontier_type", frontier_type, "instance")
    _warn_if_unsupported("skip_perfect_score", skip_perfect_score, True)
    _warn_if_unsupported("batch_sampler", batch_sampler, "epoch_shuffled")
    _warn_if_unsupported("reflection_minibatch_size", reflection_minibatch_size, None)
    _warn_if_unsupported("perfect_score", perfect_score, 1.0)
    _warn_if_unsupported("reflection_prompt_template", reflection_prompt_template, None)
    _warn_if_unsupported("module_selector", module_selector, "round_robin")
    _warn_if_unsupported("max_merge_invocations", max_merge_invocations, 5)
    _warn_if_unsupported("merge_val_overlap_floor", merge_val_overlap_floor, 5)
    _warn_if_unsupported("logger", logger, None)
    _warn_if_unsupported("run_dir", run_dir, None)
    _warn_if_unsupported("callbacks", callbacks, None)
    _warn_if_unsupported("use_wandb", use_wandb, False)
    _warn_if_unsupported("wandb_api_key", wandb_api_key, None)
    _warn_if_unsupported("wandb_init_kwargs", wandb_init_kwargs, None)
    _warn_if_unsupported("use_mlflow", use_mlflow, False)
    _warn_if_unsupported("mlflow_tracking_uri", mlflow_tracking_uri, None)
    _warn_if_unsupported("mlflow_experiment_name", mlflow_experiment_name, None)
    _warn_if_unsupported("track_best_outputs", track_best_outputs, False)
    _warn_if_unsupported("use_cloudpickle", use_cloudpickle, False)
    _warn_if_unsupported("cache_evaluation", cache_evaluation, False)
    _warn_if_unsupported("val_evaluation_policy", val_evaluation_policy, None)

    if seed_candidate is None or not seed_candidate:
        raise ValueError("seed_candidate must contain at least one prompt component.")

    if task_lm is None or not isinstance(task_lm, str):
        raise ValueError("task_lm must be provided as a model identifier string.")

    resolved_max_metric_calls = _resolve_max_metric_calls(max_metric_calls, stop_callbacks)
    if resolved_max_metric_calls is None and stop_callbacks is None:
        raise ValueError("Either max_metric_calls or stop_callbacks must be provided.")

    dataset_bundle = _build_dataset_bundle(trainset, valset)
    base_system_prompt = _extract_seed_system_prompt(seed_candidate)
    initial_prompt = gepa_candidate_to_initial_prompt(seed_candidate)
    provider, model = _parse_task_model(task_lm)
    proposer_effort = _infer_proposer_effort(
        reflection_lm if isinstance(reflection_lm, str) else None
    )
    proposer_output_tokens = "FAST"
    num_generations = _infer_num_generations(resolved_max_metric_calls)
    children_per_generation = _infer_children_per_generation(resolved_max_metric_calls)

    backend_url = os.environ.get("SYNTH_BACKEND_URL", "").strip() or "https://api.usesynth.ai"
    api_key = os.environ.get("SYNTH_API_KEY", "").strip()
    if not api_key:
        raise ValueError("SYNTH_API_KEY must be set to run Synth GEPA compatibility mode.")

    environment_api_key = ensure_localapi_auth(backend_base=backend_url, synth_api_key=api_key)

    async def _run() -> GEPAResult[Any, Any]:
        def _build_task_app_config() -> TaskAppConfig:
            def provide_taskset_description() -> dict[str, Any]:
                return {
                    "splits": ["train", "val"],
                    "sizes": {
                        "train": dataset_bundle.train.size(),
                        "val": dataset_bundle.validation.size(),
                    },
                }

            def provide_task_instances(seeds: list[int]) -> list[dict[str, Any]]:
                instances = []
                for seed_value in seeds:
                    split_name, sample = dataset_bundle.resolve_seed(seed_value)
                    input_text = _extract_input(sample)
                    expected_answer = _extract_answer(sample)
                    instances.append(
                        {
                            "task": {"id": "gepa-compat", "name": "GEPA Compatibility"},
                            "dataset": {
                                "id": "gepa-compat",
                                "split": split_name,
                                "index": seed_value,
                            },
                            "inference": {},
                            "limits": {"max_turns": 1},
                            "task_metadata": {
                                "input": input_text,
                                "answer": expected_answer,
                                "additional_context": _extract_additional_context(sample),
                            },
                        }
                    )
                return instances

            async def rollout(request: Any, fastapi_request: Any) -> Any:
                seed_value: Any = getattr(getattr(request, "env", None), "seed", None)
                if seed_value is None:
                    env_config = getattr(getattr(request, "env", None), "config", None)
                    if isinstance(env_config, dict):
                        for key in ("seed", "index", "env_seed", "resolved_seed"):
                            if key in env_config:
                                seed_value = env_config.get(key)
                                break
                if seed_value is None:
                    # Backend often encodes the seed in the correlation id:
                    # `trace_prompt-learning-{seed}-{suffix}`.
                    corr = getattr(request, "trace_correlation_id", None)
                    if isinstance(corr, str):
                        match = re.search(r"prompt-learning-(\\d+)-", corr)
                        if match:
                            seed_value = int(match.group(1))
                if seed_value is None:
                    seed_value = 0

                _, sample = dataset_bundle.resolve_seed(int(seed_value))
                input_text = _extract_input(sample)
                expected_answer = _extract_answer(sample)

                policy_config = cast(dict[str, Any], request.policy.config or {})
                predicted = ""
                inference_url = None
                status_detail = None
                try:
                    predicted, inference_url = await _call_llm(
                        system_prompt=base_system_prompt,
                        user_prompt=input_text,
                        policy_config=policy_config,
                        request_headers=dict(getattr(fastapi_request, "headers", {})),
                    )
                except Exception as exc:
                    # Return a valid RolloutResponse even when inference fails so the
                    # backend can attribute the failure to this rollout.
                    status_detail = f"inference_error: {type(exc).__name__}: {exc}"

                normalized_pred = predicted.strip()
                last_line = normalized_pred.splitlines()[-1].strip() if normalized_pred else ""
                if last_line.startswith("###"):
                    last_line = last_line.removeprefix("###").strip()

                # Heuristic:
                # - For label-like answers (Banking77), require exact match.
                # - For free-form answers (AIME-style), allow substring containment.
                is_label_answer = bool(re.fullmatch(r"[A-Za-z0-9_?\\-]{1,80}", expected_answer))
                if is_label_answer:
                    outcome_reward = 1.0 if last_line == expected_answer else 0.0
                else:
                    outcome_reward = 1.0 if predicted and expected_answer in predicted else 0.0

                # Provide a minimal direct trace so the backend can skip trace hydration
                # when the interceptor is unavailable (common in dev).
                trace_payload = {
                    "metadata": {
                        "trace_correlation_id": getattr(request, "trace_correlation_id", ""),
                        "env_name": "gepa-compat",
                        "env_seed": int(seed_value),
                    },
                    "event_history": [
                        {
                            # Backend expects lm_call-style events with both llm_request and
                            # llm_response.message for input/output extraction.
                            "type": "lm_call",
                            "llm_request": {
                                "model": policy_config.get("model"),
                                "messages": [
                                    {"role": "system", "content": base_system_prompt},
                                    {"role": "user", "content": input_text},
                                ],
                            },
                            "llm_response": {
                                "model": policy_config.get("model"),
                                "message": {
                                    "role": "assistant",
                                    "content": predicted,
                                },
                                # Extra debug metadata (ignored by normalizers/extractors).
                                "expected_answer": expected_answer,
                            },
                        },
                    ],
                }

                response = build_rollout_response(
                    request=request,
                    outcome_reward=outcome_reward,
                    inference_url=inference_url,
                    trace=trace_payload,
                    policy_config=policy_config,
                    status_detail=status_detail,
                )
                payload = response.model_dump()
                payload["_hydration_skipped"] = True
                return payload

            return TaskAppConfig(
                app_id="gepa-compat",
                name="GEPA Compatibility",
                description="Synth-backed GEPA compatibility task app",
                provide_taskset_description=provide_taskset_description,
                provide_task_instances=provide_task_instances,
                rollout=rollout,
                cors_origins=["*"],
            )

        app = create_task_app(_build_task_app_config())
        tunnel_mode = (
            os.environ.get("SYNTH_GEPA_TUNNEL_MODE", "synthtunnel").strip() or "synthtunnel"
        )

        # When using local tunnel mode, the backend may not have the ENVIRONMENT_API_KEY
        # stored in its database for this org.  In that case the GEPA optimizer falls back
        # to the well-known internal key "synth-internal-verifier-opt-key-v1".
        # Register it as an accepted alias so the in-process task app accepts it.
        if tunnel_mode in ("local", "localhost"):
            internal_key = "synth-internal-verifier-opt-key-v1"
            existing_aliases = os.environ.get("ENVIRONMENT_API_KEY_ALIASES", "")
            if internal_key not in existing_aliases:
                sep = "," if existing_aliases else ""
                os.environ["ENVIRONMENT_API_KEY_ALIASES"] = f"{existing_aliases}{sep}{internal_key}"

        async with InProcessTaskApp(
            app=app, tunnel_mode=tunnel_mode, api_key=environment_api_key
        ) as task_app:
            task_url = task_app.url or ""
            worker_token = task_app.task_app_worker_token
            if not task_url:
                raise ValueError("Failed to resolve task app URL for optimization.")
            if "s/rt_" in task_url and not worker_token:
                raise ValueError("SynthTunnel worker token is required for task app URL.")

            minimal_config: dict[str, Any] = {
                "task_app_url": task_url,
                "env_name": "gepa-compat",
                "train_seeds": dataset_bundle.train_seeds,
                "validation_seeds": dataset_bundle.validation_seeds,
                "proposer_effort": proposer_effort,
                "proposer_output_tokens": proposer_output_tokens,
                "num_generations": num_generations,
                "children_per_generation": children_per_generation,
                "policy": {
                    "provider": provider,
                    "model": model,
                    "inference_mode": "synth_hosted",
                },
                "initial_prompt": initial_prompt,
                "proposer_type": "gepa-ai",
                "rng_seed": seed,
            }
            if resolved_max_metric_calls is not None:
                minimal_config["max_rollouts"] = resolved_max_metric_calls

            expanded_config = expand_gepa_config(minimal_config)
            config_dict = {"policy_optimization": expanded_config}

            job_overrides: dict[str, Any] = {}
            if resolved_max_metric_calls is not None:
                job_overrides["prompt_learning.gepa.rollout.budget"] = resolved_max_metric_calls
                # Keep demos responsive: avoid large minibatches / high parallelism
                # when users set a tiny rollout budget.
                job_overrides.setdefault("prompt_learning.gepa.rollout.max_concurrent", 1)
                job_overrides.setdefault("prompt_learning.gepa.rollout.minibatch_size", 1)

            job = PolicyOptimizationJob.from_dict(
                config_dict=config_dict,
                backend_url=backend_url,
                api_key=api_key,
                localapi_api_key=environment_api_key,
                task_app_worker_token=worker_token,
                algorithm="gepa",
                overrides=job_overrides,
            )
            # Submitting performs synchronous HTTP requests; keep the task app loop responsive.
            await asyncio.to_thread(job.submit)
            # `InProcessTaskApp` runs an async FastAPI server in this same event loop.
            # `poll_until_complete()` is synchronous and would block the loop, preventing
            # the task app from serving `/rollout` requests coming from the backend worker.
            poll_kwargs: dict[str, Any] = {
                "timeout": float(os.getenv("SYNTH_GEPA_TIMEOUT_SECONDS", "3600")),
                "interval": 15.0,
                "progress": display_progress_bar,
            }
            # Avoid the Rust "poll_until_complete" fast-path when progress is disabled:
            # it can hold the GIL for long periods, starving the in-process task app thread.
            if not display_progress_bar:
                poll_kwargs["on_status"] = lambda _status: None

            result = await asyncio.to_thread(job.poll_until_complete, **poll_kwargs)

            if result.failed and raise_on_exception:
                raise RuntimeError(result.error or "GEPA optimization failed.")

            best_prompt_value = result.best_prompt
            best_system_prompt = _extract_system_prompt_from_best_prompt(best_prompt_value)
            best_candidate = (
                {"system_prompt": best_system_prompt}
                if best_system_prompt
                else {"system_prompt": base_system_prompt}
            )

            best_score = result.best_reward or 0.0
            return GEPAResult(
                candidates=[best_candidate],
                parents=[[]],
                val_aggregate_scores=[float(best_score)],
                val_subscores=[{}],
                per_val_instance_best_candidates={},
                discovery_eval_counts=[resolved_max_metric_calls or 0],
                total_metric_calls=resolved_max_metric_calls,
                run_dir=None,
                seed=seed,
            )

    return _run_async(_run())
