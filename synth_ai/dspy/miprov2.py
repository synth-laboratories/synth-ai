"""DSPy-compatible MIPROv2 wrapper backed by Synth MIPRO."""
# See: specifications/tanha/master_specification.md

from __future__ import annotations

import asyncio
import math
import os
import re
import warnings
from dataclasses import dataclass
from typing import Any, Literal

from synth_ai.core.config.expansion import gepa_candidate_to_initial_prompt
from synth_ai.core.utils.urls import resolve_synth_backend_url
from synth_ai.gepa import api as _gepa_api
from synth_ai.sdk.container import InProcessContainer
from synth_ai.sdk.container._impl.rollout_helpers import build_rollout_response
from synth_ai.sdk.container._impl.server import ContainerConfig, create_container
from synth_ai.sdk.container.auth import ensure_container_auth
from synth_ai.sdk.optimization.policy import PolicyOptimizationJob

from ._compat import (
    apply_system_prompt_to_student,
    attach_detailed_results,
    clone_student,
    coerce_model_identifier,
    extract_seed_candidate_from_student,
    infer_task_lm,
    materialize_dataset,
)

_AUTO_TRIALS: dict[str, int] = {
    "light": 7,
    "medium": 20,
    "heavy": 50,
}


@dataclass(frozen=True)
class MIPROv2DetailedResult:
    """Minimal detailed result payload for DSPy compatibility consumers."""

    job_id: str
    status: str
    best_score: float | None
    best_candidate: str | dict[str, Any] | None
    raw: dict[str, Any]


def _extract_system_prompt(candidate: Any) -> str | None:
    if isinstance(candidate, str):
        rendered = candidate.strip()
        return rendered or None
    if not isinstance(candidate, dict):
        return None

    for key in ("system_prompt", "instruction", "prompt"):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    messages = candidate.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") != "system":
                continue
            value = message.get("pattern") or message.get("content")
            if isinstance(value, str) and value.strip():
                return value.strip()

    sections = candidate.get("sections")
    if isinstance(sections, list):
        for section in sections:
            if not isinstance(section, dict):
                continue
            if section.get("role") != "system":
                continue
            value = section.get("pattern") or section.get("content")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _resolve_num_trials(
    *,
    auto: Literal["light", "medium", "heavy"] | None,
    explicit_trials: int | None,
    num_candidates: int | None,
    student: Any,
) -> int:
    if explicit_trials is not None:
        return max(1, int(explicit_trials))
    if auto is None and num_candidates is not None:
        predictor_count = 1
        named_predictors = getattr(student, "named_predictors", None)
        if callable(named_predictors):
            try:
                predictor_count = max(1, len(list(named_predictors())))
            except Exception:
                predictor_count = 1
        suggested = int(
            max(
                2 * (predictor_count * 2) * math.log2(max(num_candidates, 2)),
                1.5 * num_candidates,
            )
        )
        return max(1, suggested)
    if auto is None:
        raise ValueError("num_trials must be provided when auto is None.")
    return _AUTO_TRIALS[auto]


def _run_synth_mipro(
    *,
    seed_candidate: dict[str, str],
    trainset: list[dict[str, Any]],
    valset: list[dict[str, Any]] | None,
    task_lm: str,
    proposer_lm: str,
    bootstrap_train_seeds: list[int],
    online_pool: list[int],
    validation_seeds: list[int],
    max_rollouts: int | None,
    seed: int,
    display_progress: bool,
    timeout_seconds: float,
) -> Any:
    backend_url = resolve_synth_backend_url()
    api_key = os.environ.get("SYNTH_API_KEY", "").strip()
    if not api_key:
        raise ValueError("SYNTH_API_KEY must be set to run Synth DSPy MIPROv2 compatibility.")

    environment_api_key = ensure_container_auth(backend_base=backend_url, synth_api_key=api_key)
    dataset_bundle = _gepa_api._build_dataset_bundle(trainset, valset)
    base_system_prompt = _gepa_api._extract_seed_system_prompt(seed_candidate)
    initial_prompt = gepa_candidate_to_initial_prompt(seed_candidate)
    provider, model = _gepa_api._parse_task_model(task_lm)
    _proposer_provider, _proposer_model = _gepa_api._parse_task_model(proposer_lm)

    async def _run() -> Any:
        def _build_container_config() -> ContainerConfig:
            def provide_taskset_description() -> dict[str, Any]:
                return {
                    "splits": ["train", "val"],
                    "sizes": {
                        "train": dataset_bundle.train.size(),
                        "val": dataset_bundle.validation.size(),
                    },
                }

            def provide_task_instances(seeds: list[int]) -> list[dict[str, Any]]:
                instances: list[dict[str, Any]] = []
                for seed_value in seeds:
                    split_name, sample = dataset_bundle.resolve_seed(seed_value)
                    input_text = _gepa_api._extract_input(sample)
                    expected_answer = _gepa_api._extract_answer(sample)
                    instances.append(
                        {
                            "task": {"id": "mipro-compat", "name": "MIPRO Compatibility"},
                            "dataset": {
                                "id": "mipro-compat",
                                "split": split_name,
                                "index": seed_value,
                            },
                            "inference": {},
                            "limits": {"max_turns": 1},
                            "task_metadata": {
                                "input": input_text,
                                "answer": expected_answer,
                                "additional_context": _gepa_api._extract_additional_context(sample),
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
                    corr = getattr(request, "trace_correlation_id", None)
                    if isinstance(corr, str):
                        match = re.search(r"prompt-learning-(\\d+)-", corr)
                        if match:
                            seed_value = int(match.group(1))
                if seed_value is None:
                    seed_value = 0

                _, sample = dataset_bundle.resolve_seed(int(seed_value))
                input_text = _gepa_api._extract_input(sample)
                expected_answer = _gepa_api._extract_answer(sample)

                policy_config = dict(getattr(request.policy, "config", {}) or {})
                predicted = ""
                inference_url = None
                status_detail = None
                try:
                    predicted, inference_url = await _gepa_api._call_llm(
                        system_prompt=base_system_prompt,
                        user_prompt=input_text,
                        policy_config=policy_config,
                        request_headers=dict(getattr(fastapi_request, "headers", {})),
                    )
                except Exception as exc:  # pragma: no cover - exercised in integration
                    status_detail = f"inference_error: {type(exc).__name__}: {exc}"

                normalized_pred = predicted.strip()
                last_line = normalized_pred.splitlines()[-1].strip() if normalized_pred else ""
                if last_line.startswith("###"):
                    last_line = last_line.removeprefix("###").strip()

                is_label_answer = bool(re.fullmatch(r"[A-Za-z0-9_?\\-]{1,80}", expected_answer))
                if is_label_answer:
                    outcome_reward = 1.0 if last_line == expected_answer else 0.0
                else:
                    outcome_reward = 1.0 if predicted and expected_answer in predicted else 0.0

                trace_payload = {
                    "metadata": {
                        "trace_correlation_id": getattr(request, "trace_correlation_id", ""),
                        "env_name": "mipro-compat",
                        "env_seed": int(seed_value),
                    },
                    "event_history": [
                        {
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
                                "message": {"role": "assistant", "content": predicted},
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

            return ContainerConfig(
                app_id="mipro-compat",
                name="MIPRO Compatibility",
                description="Synth-backed MIPRO compatibility container",
                provide_taskset_description=provide_taskset_description,
                provide_task_instances=provide_task_instances,
                rollout=rollout,
                cors_origins=["*"],
            )

        app = create_container(_build_container_config())
        tunnel_mode = (
            os.environ.get("SYNTH_MIPRO_TUNNEL_MODE", "").strip()
            or os.environ.get("SYNTH_GEPA_TUNNEL_MODE", "").strip()
            or "synthtunnel"
        )
        if tunnel_mode in ("local", "localhost"):
            internal_key = "synth-internal-verifier-opt-key-v1"
            existing_aliases = os.environ.get("ENVIRONMENT_API_KEY_ALIASES", "")
            if internal_key not in existing_aliases:
                sep = "," if existing_aliases else ""
                os.environ["ENVIRONMENT_API_KEY_ALIASES"] = f"{existing_aliases}{sep}{internal_key}"

        async with InProcessContainer(
            app=app,
            tunnel_mode=tunnel_mode,
            api_key=environment_api_key,
        ) as container:
            task_url = container.url or ""
            worker_token = container.container_worker_token
            if not task_url:
                raise ValueError("Failed to resolve container URL for MIPRO compatibility mode.")
            if "s/rt_" in task_url and not worker_token:
                raise ValueError("SynthTunnel worker token is required for container URL.")

            config_dict: dict[str, Any] = {
                "policy_optimization": {
                    "algorithm": "mipro",
                    "container_url": task_url,
                    "env_name": "mipro-compat",
                    "policy": {
                        "provider": provider,
                        "model": model,
                        "inference_mode": "synth_hosted",
                    },
                    "initial_prompt": initial_prompt,
                    "bootstrap_train_seeds": bootstrap_train_seeds,
                    "online_pool": online_pool,
                    "val_seeds": validation_seeds,
                    "reference_pool": validation_seeds,
                    "mipro": {
                        "env_name": "mipro-compat",
                        "bootstrap_train_seeds": bootstrap_train_seeds,
                        "online_pool": online_pool,
                        "val_seeds": validation_seeds,
                        "reference_pool": validation_seeds,
                        "evaluation": {
                            "val_seeds": validation_seeds,
                        },
                    },
                }
            }

            overrides: dict[str, Any] = {}
            if max_rollouts is not None:
                overrides["prompt_learning.mipro.rollout.budget"] = max_rollouts

            job = PolicyOptimizationJob.from_dict(
                config_dict=config_dict,
                backend_url=backend_url,
                api_key=api_key,
                container_api_key=environment_api_key,
                container_worker_token=worker_token,
                algorithm="mipro",
                overrides=overrides,
                skip_health_check=True,
            )
            await asyncio.to_thread(job.submit)

            poll_kwargs: dict[str, Any] = {
                "timeout": timeout_seconds,
                "interval": 15.0,
                "progress": display_progress,
            }
            if not display_progress:
                poll_kwargs["on_status"] = lambda _status: None

            return await asyncio.to_thread(job.poll_until_complete, **poll_kwargs)

    return _gepa_api._run_async(_run())


class MIPROv2:
    """Drop-in replacement for `dspy.MIPROv2` using Synth policy optimization."""

    def __init__(
        self,
        metric: Any,
        prompt_model: Any | None = None,
        task_model: Any | None = None,
        teacher_settings: dict[str, Any] | None = None,
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 4,
        auto: Literal["light", "medium", "heavy"] | None = "light",
        num_candidates: int | None = None,
        num_threads: int | None = None,
        max_errors: int | None = None,
        seed: int = 9,
        init_temperature: float = 1.0,
        verbose: bool = False,
        track_stats: bool = True,
        log_dir: str | None = None,
        metric_threshold: float | None = None,
        **kwargs: Any,
    ) -> None:
        init_kwargs = dict(kwargs)
        if task_model is None and "task_lm" in init_kwargs:
            task_model = init_kwargs.pop("task_lm")
        if prompt_model is None and "proposer_model" in init_kwargs:
            prompt_model = init_kwargs.pop("proposer_model")

        if auto not in {None, "light", "medium", "heavy"}:
            raise ValueError("auto must be one of: None, light, medium, heavy.")

        if task_model is None or prompt_model is None:
            try:
                import synth_ai.dspy as dspy_module

                default_lm = getattr(dspy_module.settings, "lm", None)
            except Exception:
                default_lm = None
            if task_model is None:
                task_model = default_lm
            if prompt_model is None:
                prompt_model = default_lm

        if task_model is None or prompt_model is None:
            raise ValueError("Provide task_model and prompt_model, or set dspy.configure(lm=...).")

        self.metric = metric
        self.prompt_model = prompt_model
        self.task_model = task_model
        self.teacher_settings = dict(teacher_settings or {})
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.auto = auto
        self.num_candidates = num_candidates
        self.num_threads = num_threads
        self.max_errors = max_errors
        self.seed = seed
        self.init_temperature = init_temperature
        self.verbose = verbose
        self.track_stats = track_stats
        self.log_dir = log_dir
        self.metric_threshold = metric_threshold

        if init_kwargs:
            warnings.warn(
                "Ignoring unsupported MIPROv2 init argument(s): " + ", ".join(sorted(init_kwargs)),
                RuntimeWarning,
                stacklevel=2,
            )

    def compile(
        self,
        student: Any,
        *,
        trainset: list[Any],
        teacher: Any = None,
        valset: list[Any] | None = None,
        num_trials: int | None = None,
        num_candidates: int | None = None,
        max_bootstrapped_demos: int | None = None,
        max_labeled_demos: int | None = None,
        seed: int | None = None,
        minibatch: bool = True,
        minibatch_size: int = 35,
        minibatch_full_eval_steps: int = 5,
        program_aware_proposer: bool = True,
        data_aware_proposer: bool = True,
        view_data_batch_size: int = 10,
        tip_aware_proposer: bool = True,
        fewshot_aware_proposer: bool = True,
        requires_permission_to_run: bool | None = None,
        provide_traceback: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        compile_kwargs = dict(kwargs)
        if "max_rollouts" in compile_kwargs and num_trials is None:
            num_trials = int(compile_kwargs.pop("max_rollouts"))
        display_progress = bool(compile_kwargs.pop("display_progress", self.verbose))
        timeout_seconds = float(
            compile_kwargs.pop(
                "timeout_seconds", float(os.getenv("SYNTH_MIPRO_TIMEOUT_SECONDS", "3600"))
            )
        )
        track_stats = bool(compile_kwargs.pop("track_stats", self.track_stats))

        runtime_task_model = compile_kwargs.pop("task_model", self.task_model)
        runtime_prompt_model = compile_kwargs.pop("prompt_model", self.prompt_model)

        if compile_kwargs:
            warnings.warn(
                "Ignoring unsupported MIPROv2 compile argument(s): "
                + ", ".join(sorted(compile_kwargs)),
                RuntimeWarning,
                stacklevel=2,
            )

        if requires_permission_to_run is False:
            warnings.warn(
                "'requires_permission_to_run' is deprecated and ignored.",
                RuntimeWarning,
                stacklevel=2,
            )
        elif requires_permission_to_run is True:
            raise ValueError(
                "User confirmation is removed from MIPROv2 compatibility mode. "
                "Remove requires_permission_to_run."
            )

        effective_num_candidates = (
            int(num_candidates) if num_candidates is not None else self.num_candidates
        )
        effective_auto = self.auto
        if effective_auto is not None and (
            effective_num_candidates is not None or num_trials is not None
        ):
            warnings.warn(
                "Explicit num_candidates/num_trials overrides auto in MIPROv2 compatibility mode.",
                RuntimeWarning,
                stacklevel=2,
            )
            effective_auto = None

        if effective_auto is None and (effective_num_candidates is not None and num_trials is None):
            warnings.warn(
                "auto=None with num_candidates and no num_trials; inferring num_trials "
                "using DSPy-style heuristic.",
                RuntimeWarning,
                stacklevel=2,
            )
        if effective_auto is None and (effective_num_candidates is None and num_trials is None):
            raise ValueError("If auto is None, provide either num_candidates or num_trials.")

        if teacher is not None:
            warnings.warn(
                "teacher is ignored in Synth MIPROv2 compatibility mode.",
                RuntimeWarning,
                stacklevel=2,
            )
        if self.num_threads is not None:
            warnings.warn(
                "num_threads is not yet supported in Synth MIPROv2 compatibility mode.",
                RuntimeWarning,
                stacklevel=2,
            )
        if self.max_errors is not None:
            warnings.warn(
                "max_errors is not yet supported in Synth MIPROv2 compatibility mode.",
                RuntimeWarning,
                stacklevel=2,
            )
        if self.init_temperature != 1.0:
            warnings.warn(
                "init_temperature is not yet supported in Synth MIPROv2 compatibility mode.",
                RuntimeWarning,
                stacklevel=2,
            )
        if self.metric_threshold is not None:
            warnings.warn(
                "metric_threshold is not yet supported in Synth MIPROv2 compatibility mode.",
                RuntimeWarning,
                stacklevel=2,
            )
        if self.log_dir:
            warnings.warn(
                "log_dir is not yet supported in Synth MIPROv2 compatibility mode.",
                RuntimeWarning,
                stacklevel=2,
            )

        ignored_flags = {
            "program_aware_proposer": program_aware_proposer,
            "data_aware_proposer": data_aware_proposer,
            "view_data_batch_size": view_data_batch_size,
            "tip_aware_proposer": tip_aware_proposer,
            "fewshot_aware_proposer": fewshot_aware_proposer,
            "provide_traceback": provide_traceback,
            "minibatch_full_eval_steps": minibatch_full_eval_steps,
        }
        for name, value in ignored_flags.items():
            if value not in (True, None, 10, 5):
                warnings.warn(
                    f"{name} is not supported in Synth MIPROv2 compatibility mode and is ignored.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        run_seed = int(seed if seed is not None else self.seed)
        if max_bootstrapped_demos is not None:
            self.max_bootstrapped_demos = int(max_bootstrapped_demos)
        if max_labeled_demos is not None:
            self.max_labeled_demos = int(max_labeled_demos)

        train_records = materialize_dataset(trainset)
        val_records = materialize_dataset(valset) if valset is not None else train_records
        if minibatch and minibatch_size < 1:
            raise ValueError("minibatch_size must be >= 1 when minibatch=True.")
        if minibatch and minibatch_size > len(val_records):
            raise ValueError(
                f"minibatch_size ({minibatch_size}) cannot exceed valset size ({len(val_records)})."
            )

        trials = _resolve_num_trials(
            auto=effective_auto,
            explicit_trials=num_trials,
            num_candidates=effective_num_candidates,
            student=student,
        )
        seed_candidate, target_predictor_name = extract_seed_candidate_from_student(student)

        task_lm = infer_task_lm(task_model=runtime_task_model, student=student)
        proposer_lm = coerce_model_identifier(runtime_prompt_model) or task_lm
        if proposer_lm != task_lm:
            warnings.warn(
                "prompt_model differs from task_model, but current Synth MIPRO compatibility "
                "uses task_model for backend policy execution.",
                RuntimeWarning,
                stacklevel=2,
            )

        total_bootstrap = max(1, self.max_bootstrapped_demos + max(self.max_labeled_demos, 0))
        bootstrap_count = min(total_bootstrap, len(train_records))
        if minibatch:
            online_count = min(max(1, minibatch_size), len(train_records))
        else:
            online_count = len(train_records)
        bootstrap_train_seeds = list(range(bootstrap_count))
        online_pool = list(range(online_count))

        if valset is None:
            validation_seeds = list(range(min(max(1, online_count), len(train_records))))
            run_valset: list[dict[str, Any]] | None = None
        else:
            validation_offset = len(train_records)
            validation_seeds = list(range(validation_offset, validation_offset + len(val_records)))
            run_valset = val_records

        result = _run_synth_mipro(
            seed_candidate=seed_candidate,
            trainset=train_records,
            valset=run_valset,
            task_lm=task_lm,
            proposer_lm=proposer_lm,
            bootstrap_train_seeds=bootstrap_train_seeds,
            online_pool=online_pool,
            validation_seeds=validation_seeds,
            max_rollouts=trials,
            seed=run_seed,
            display_progress=display_progress,
            timeout_seconds=timeout_seconds,
        )
        if getattr(result, "failed", False):
            raise RuntimeError(getattr(result, "error", None) or "MIPROv2 optimization failed.")

        optimized_program = clone_student(student)
        best_prompt = _extract_system_prompt(getattr(result, "best_candidate", None))
        if best_prompt:
            apply_system_prompt_to_student(
                optimized_program,
                best_prompt,
                target_predictor_name=target_predictor_name,
            )

        details = MIPROv2DetailedResult(
            job_id=str(getattr(result, "job_id", "")),
            status=str(
                getattr(getattr(result, "status", None), "value", getattr(result, "status", ""))
            ),
            best_score=getattr(result, "best_reward", None),
            best_candidate=getattr(result, "best_candidate", None),
            raw=dict(getattr(result, "raw", {}) or {}),
        )
        if track_stats:
            attach_detailed_results(optimized_program, details)
        return optimized_program
