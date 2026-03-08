"""Shared compatibility helpers for DSPy drop-in shims."""
# See: specifications/tanha/master_specification.md

from __future__ import annotations

import copy
import os
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any

_INPUT_KEYS: tuple[str, ...] = ("input", "question", "problem", "prompt", "query")
_ANSWER_KEYS: tuple[str, ...] = ("answer", "expected_output", "output", "label")


def coerce_model_identifier(model: Any) -> str | None:
    """Convert a model-like object into a model identifier string."""
    if model is None:
        return None
    if isinstance(model, str):
        value = model.strip()
        return value or None
    for attr in ("model", "model_name", "name", "id"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    rendered = str(model).strip()
    return rendered or None


def infer_task_lm(
    *,
    task_model: Any | None,
    student: Any | None = None,
    env_var: str = "SYNTH_DSPY_TASK_LM",
    default: str = "openai/gpt-4.1-mini",
) -> str:
    """Resolve the task model in DSPy-compatible priority order."""
    for candidate in (task_model, getattr(student, "lm", None), os.getenv(env_var, "")):
        model_id = coerce_model_identifier(candidate)
        if model_id:
            return model_id
    return default


def clone_student(student: Any) -> Any:
    """Best-effort clone of a DSPy student program."""
    deepcopy_method = getattr(student, "deepcopy", None)
    if callable(deepcopy_method):
        try:
            return deepcopy_method()
        except Exception:
            pass
    try:
        return copy.deepcopy(student)
    except Exception:
        return student


def _read_field(item: Any, key: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(key)
    if hasattr(item, key):
        return getattr(item, key)
    getter = getattr(item, "get", None)
    if callable(getter):
        try:
            return getter(key)
        except Exception:
            return None
    return None


def _to_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        rendered = value.strip()
        return rendered or None
    rendered = str(value).strip()
    return rendered or None


def _extract_instruction_from_object(obj: Any) -> str | None:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        for key in ("system_prompt", "instruction", "prompt", "system"):
            text = _to_text(obj.get(key))
            if text:
                return text
        if len(obj) == 1:
            return _to_text(next(iter(obj.values())))
        return None

    signature = getattr(obj, "signature", None)
    if signature is not None:
        text = _to_text(getattr(signature, "instructions", None))
        if text:
            return text

    text = _to_text(getattr(obj, "instructions", None))
    if text:
        return text

    predict = getattr(obj, "predict", None)
    if predict is not None:
        return _extract_instruction_from_object(predict)
    return None


def _set_instruction_on_object(obj: Any, prompt: str) -> bool:
    if isinstance(obj, MutableMapping):
        for key in ("system_prompt", "instruction", "prompt"):
            if key in obj:
                obj[key] = prompt
                return True
        if len(obj) == 1:
            only_key = next(iter(obj))
            obj[only_key] = prompt
            return True
        obj["system_prompt"] = prompt
        return True

    signature = getattr(obj, "signature", None)
    if signature is not None and hasattr(signature, "instructions"):
        try:
            signature.instructions = prompt
            return True
        except Exception:
            pass

    if hasattr(obj, "instructions"):
        try:
            obj.instructions = prompt
            return True
        except Exception:
            return False
    return False


def extract_seed_candidate_from_student(student: Any) -> tuple[dict[str, str], str | None]:
    """Build a GEPA seed candidate from a DSPy-style program."""
    if isinstance(student, Mapping):
        prompt = _extract_instruction_from_object(student)
        if prompt:
            return {"system_prompt": prompt}, None
        raise ValueError("Could not derive a seed prompt from mapping student.")

    named_predictors = getattr(student, "named_predictors", None)
    if callable(named_predictors):
        try:
            predictors = list(named_predictors())
        except Exception:
            predictors = []
        for name, predictor in predictors:
            prompt = _extract_instruction_from_object(predictor)
            if prompt:
                return {"system_prompt": prompt}, str(name)

    prompt = _extract_instruction_from_object(student)
    if prompt:
        return {"system_prompt": prompt}, None

    raise ValueError(
        "Could not derive a seed prompt from student. Expected a mapping, "
        "student.signature.instructions, or named predictors with instructions."
    )


def materialize_dataset(dataset: Sequence[Any]) -> list[dict[str, Any]]:
    """Normalize examples into GEPA-compatible dict records."""
    records: list[dict[str, Any]] = []
    for item in dataset:
        input_text = None
        for key in _INPUT_KEYS:
            input_text = _to_text(_read_field(item, key))
            if input_text:
                break
        answer_text = None
        for key in _ANSWER_KEYS:
            answer_text = _to_text(_read_field(item, key))
            if answer_text:
                break
        if not input_text:
            raise ValueError(
                "Dataset examples must include one of: input/question/problem/prompt/query."
            )
        if not answer_text:
            raise ValueError(
                "Dataset examples must include one of: answer/expected_output/output/label."
            )
        raw_context = _read_field(item, "additional_context")
        if isinstance(raw_context, Mapping):
            additional_context = {str(k): str(v) for k, v in raw_context.items()}
        else:
            additional_context = {}
        records.append(
            {
                "input": input_text,
                "answer": answer_text,
                "additional_context": additional_context,
            }
        )
    if not records:
        raise ValueError("Dataset must contain at least one example.")
    return records


def apply_system_prompt_to_student(
    student: Any,
    prompt: str,
    *,
    target_predictor_name: str | None = None,
) -> bool:
    """Apply an optimized prompt back onto a DSPy-style program."""
    if _set_instruction_on_object(student, prompt):
        return True

    if target_predictor_name:
        named_predictors = getattr(student, "named_predictors", None)
        if callable(named_predictors):
            try:
                for name, predictor in named_predictors():
                    if str(name) == target_predictor_name and _set_instruction_on_object(
                        predictor, prompt
                    ):
                        return True
            except Exception:
                pass

    predict = getattr(student, "predict", None)
    if predict is not None and _set_instruction_on_object(predict, prompt):
        return True

    named_predictors = getattr(student, "named_predictors", None)
    if callable(named_predictors):
        try:
            for _name, predictor in named_predictors():
                if _set_instruction_on_object(predictor, prompt):
                    return True
        except Exception:
            pass
    return False


def attach_detailed_results(student: Any, value: Any) -> None:
    """Attach `detailed_results` in a way that works for dicts or objects."""
    if isinstance(student, MutableMapping):
        student["detailed_results"] = value
        return
    try:
        student.detailed_results = value
    except Exception:
        return
