"""DSPy-compatible MIPROv2 wrapper for prompt-opt."""

from __future__ import annotations

import copy
import json
import math
from typing import Any, Literal

from prompt_opt.mipro import proposer_backends, run_mipro


def _extract_text_field(example: Any, keys: tuple[str, ...]) -> str:
    if isinstance(example, dict):
        for key in keys:
            value = example.get(key)
            if value is not None:
                return str(value)
    for key in keys:
        if hasattr(example, key):
            value = getattr(example, key)
            if value is not None:
                return str(value)
    return ""


def _materialize_dataset(dataset: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in dataset:
        input_text = _extract_text_field(item, ("input", "question", "query", "text"))
        answer_text = _extract_text_field(item, ("answer", "label", "expected", "output"))
        additional_context: dict[str, Any] = {}
        if isinstance(item, dict):
            raw_context = item.get("additional_context")
            if isinstance(raw_context, dict):
                additional_context = dict(raw_context)
        labels_blob = str(additional_context.get("labels", "")).strip()
        if labels_blob:
            input_text = f"{input_text}\n\nLabels: {labels_blob}"
        rows.append({"input": input_text, "answer": answer_text, "metadata": {}})
    return rows


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
        suggested = int(max(2 * (predictor_count * 2) * math.log2(max(num_candidates, 2)), 1.5 * num_candidates))
        return max(1, suggested)
    if auto == "light":
        return 7
    if auto == "medium":
        return 20
    if auto == "heavy":
        return 50
    raise ValueError("num_trials must be provided when auto is None.")


def _extract_seed_candidate_from_student(student: Any) -> tuple[dict[str, str], str | None]:
    named_predictors = getattr(student, "named_predictors", None)
    if callable(named_predictors):
        try:
            entries = list(named_predictors())
            if entries:
                first_name, first_predictor = entries[0]
                signature = getattr(first_predictor, "signature", None)
                instructions = getattr(signature, "instructions", None)
                if isinstance(instructions, str) and instructions.strip():
                    return {"system_prompt": instructions.strip()}, str(first_name)
        except Exception:
            pass
    return {"system_prompt": "You are a helpful assistant."}, None


def _apply_prompt_to_student(student: Any, prompt: str, target_predictor_name: str | None = None) -> Any:
    named_predictors = getattr(student, "named_predictors", None)
    if not callable(named_predictors):
        return student
    try:
        for name, predictor in list(named_predictors()):
            if target_predictor_name is not None and name != target_predictor_name:
                continue
            signature = getattr(predictor, "signature", None)
            if signature is None:
                continue
            with_instructions = getattr(signature, "with_instructions", None)
            if callable(with_instructions):
                predictor.signature = with_instructions(prompt)
            else:
                signature.instructions = prompt
            break
    except Exception:
        return student
    return student


def _extract_best_prompt(best_policy_payload: dict[str, Any]) -> str | None:
    template = best_policy_payload.get("template")
    if isinstance(template, str) and template.strip():
        return template.strip()
    return None


class MIPROv2:
    """Drop-in replacement for `dspy.MIPROv2` running locally only."""

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
        backend_mode: Literal["local"] = "local",
        proposer_backend: Literal["single_prompt", "rlm"] = "single_prompt",
        **kwargs: Any,
    ) -> None:
        del metric, teacher_settings, num_threads, max_errors, init_temperature, log_dir, metric_threshold, kwargs
        if proposer_backend not in proposer_backends():
            raise ValueError(f"Unsupported proposer_backend={proposer_backend!r}")
        self.prompt_model = prompt_model
        self.task_model = task_model
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.auto = auto
        self.num_candidates = num_candidates
        self.seed = seed
        self.verbose = verbose
        self.track_stats = track_stats
        self.backend_mode = backend_mode
        self.proposer_backend = proposer_backend

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
        del (
            teacher,
            max_bootstrapped_demos,
            max_labeled_demos,
            minibatch,
            minibatch_size,
            minibatch_full_eval_steps,
            program_aware_proposer,
            data_aware_proposer,
            view_data_batch_size,
            tip_aware_proposer,
            fewshot_aware_proposer,
            requires_permission_to_run,
            provide_traceback,
        )
        if self.backend_mode != "local":
            raise ValueError(
                "prompt-opt is local-only. Use backend_mode='local'."
            )

        if not callable(getattr(self.task_model, "__call__", None)):
            raise ValueError(
                "Local backend_mode requires task_model to be callable(prompt)->str. "
                "Model-id strings are not supported in local-only mode."
            )

        run_seed = int(seed if seed is not None else self.seed)
        train_records = _materialize_dataset(trainset)
        val_records = _materialize_dataset(valset) if valset is not None else train_records
        combined_records = list(train_records) + list(val_records)
        effective_num_candidates = int(num_candidates) if num_candidates is not None else self.num_candidates
        trials = _resolve_num_trials(
            auto=self.auto,
            explicit_trials=num_trials,
            num_candidates=effective_num_candidates,
            student=student,
        )
        seed_candidate, target_predictor_name = _extract_seed_candidate_from_student(student)
        initial_policy = {"template": seed_candidate.get("system_prompt", "")}
        dataset_payload = {
            "id": "mipro-local",
            "examples": [
                {"input": row.get("input", ""), "expected": row.get("answer", ""), "metadata": {}}
                for row in combined_records
            ],
        }
        config = {
            "num_candidates": max(1, int(effective_num_candidates or 8)),
            "holdout_ratio": 0.2,
            "max_iterations": max(1, int(trials)),
            "early_stop_rounds": 3,
            "min_improvement": 1e-6,
            "seed": run_seed,
            "proposer_backend": self.proposer_backend,
        }

        result = run_mipro(
            config=config,
            initial_policy=initial_policy,
            dataset=dataset_payload,
            task_llm=self.task_model,
        )
        optimized_program = copy.deepcopy(student)
        best_policy = result.get("best_policy", {})
        best_prompt = _extract_best_prompt(best_policy if isinstance(best_policy, dict) else {})
        if best_prompt:
            optimized_program = _apply_prompt_to_student(
                optimized_program,
                best_prompt,
                target_predictor_name=target_predictor_name,
            )
        if self.track_stats:
            setattr(optimized_program, "prompt_opt_result", json.loads(json.dumps(result)))
        return optimized_program
