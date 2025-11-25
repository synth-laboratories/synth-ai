"""Utilities for validating and constructing SFT job payloads."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

from synth_ai.sdk.api.models.supported import (
    UnsupportedModelError,
    normalize_model_identifier,
)

_STEP_KEYS = ("n_epochs", "total_steps", "train_steps", "steps")


def _ensure_positive_int(value: Any, *, key: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"hyperparameters.{key} must be an integer greater than zero")
    try:
        ivalue = int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"hyperparameters.{key} must be an integer greater than zero") from exc
    if ivalue <= 0:
        raise ValueError(f"hyperparameters.{key} must be an integer greater than zero")
    return ivalue


def _ensure_non_negative_float(value: Any, *, key: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"hyperparameters.{key} must be a float greater than or equal to zero")
    try:
        fvalue = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"hyperparameters.{key} must be a float greater than or equal to zero"
        ) from exc
    if fvalue < 0:
        raise ValueError(f"hyperparameters.{key} must be a float greater than or equal to zero")
    return fvalue


def _ensure_positive_float(value: Any, *, key: str) -> float:
    fvalue = _ensure_non_negative_float(value, key=key)
    if fvalue == 0.0:
        raise ValueError(f"hyperparameters.{key} must be greater than zero")
    return fvalue


@dataclass(slots=True)
class SFTTrainingHyperparameters:
    """Typed representation of SFT training hyperparameters."""

    n_epochs: int | None = None
    total_steps: int | None = None
    train_steps: int | None = None
    steps: int | None = None
    batch_size: int | None = None
    global_batch: int | None = None
    per_device_batch: int | None = None
    gradient_accumulation_steps: int | None = None
    sequence_length: int | None = None
    learning_rate: float | None = None
    warmup_ratio: float | None = None
    train_kind: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> SFTTrainingHyperparameters:
        if data is None:
            raise ValueError("hyperparameters must not be empty")
        normalized: dict[str, Any] = dict(data)
        if not normalized:
            raise ValueError("hyperparameters must not be empty")

        kwargs: dict[str, Any] = {}

        def pop_int(name: str) -> int | None:
            if name not in normalized:
                return None
            value = _ensure_positive_int(normalized.pop(name), key=name)
            return value

        def pop_optional_int(name: str) -> int | None:
            if name not in normalized:
                return None
            value = _ensure_positive_int(normalized.pop(name), key=name)
            return value

        def pop_positive_float(name: str) -> float | None:
            if name not in normalized:
                return None
            return _ensure_positive_float(normalized.pop(name), key=name)

        def pop_non_negative_float(name: str) -> float | None:
            if name not in normalized:
                return None
            value = _ensure_non_negative_float(normalized.pop(name), key=name)
            return value

        # Step-derived keys
        step_values = {
            "n_epochs": pop_int("n_epochs"),
            "total_steps": pop_int("total_steps"),
            "train_steps": pop_int("train_steps"),
            "steps": pop_int("steps"),
        }
        if not any(step_values.values()):
            keys = ", ".join(_STEP_KEYS)
            raise ValueError(f"hyperparameters must include at least one of: {keys}")
        kwargs.update(step_values)

        kwargs["batch_size"] = pop_optional_int("batch_size")
        kwargs["global_batch"] = pop_optional_int("global_batch")
        kwargs["per_device_batch"] = pop_optional_int("per_device_batch")
        kwargs["gradient_accumulation_steps"] = pop_optional_int("gradient_accumulation_steps")
        kwargs["sequence_length"] = pop_optional_int("sequence_length")
        kwargs["learning_rate"] = pop_positive_float("learning_rate")
        kwargs["warmup_ratio"] = pop_non_negative_float("warmup_ratio")

        if "warmup_ratio" in kwargs and kwargs["warmup_ratio"] is not None:
            ratio = kwargs["warmup_ratio"]
            if ratio > 1:
                raise ValueError("hyperparameters.warmup_ratio must be between 0 and 1 inclusive")

        if "train_kind" in normalized:
            value = normalized.pop("train_kind")
            if not isinstance(value, str):
                raise ValueError("hyperparameters.train_kind must be a string")
            kwargs["train_kind"] = value

        extras = normalized

        return cls(extras=extras, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for field_info in fields(self):
            if field_info.name == "extras":
                continue
            value = getattr(self, field_info.name)
            if value is not None:
                result[field_info.name] = value
        result.update(self.extras)
        return result


def _coerce_mapping(value: Mapping[str, Any] | None, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


@dataclass(slots=True)
class SFTJobConfig:
    """Structured representation of an SFT training job request."""

    model: str
    hyperparameters: Mapping[str, Any] | SFTTrainingHyperparameters
    training_file: str | None = None
    metadata: Mapping[str, Any] | None = None
    training_type: str | None = "sft_offline"
    validation_file: str | None = None
    suffix: str | None = None
    integrations: Mapping[str, Any] | None = None

    def to_payload(
        self,
        *,
        training_file_field: str = "training_file_id",
        require_training_file: bool = True,
        include_training_file_when_none: bool = False,
        allow_finetuned_prefixes: bool = False,
    ) -> dict[str, Any]:
        model = normalize_model_identifier(
            self.model, allow_finetuned_prefixes=allow_finetuned_prefixes
        )
        if isinstance(self.hyperparameters, SFTTrainingHyperparameters):
            hyper_config = self.hyperparameters
        else:
            hyper_config = SFTTrainingHyperparameters.from_mapping(
                _coerce_mapping(self.hyperparameters, name="hyperparameters")
            )
        hyperparameters = hyper_config.to_dict()

        payload: dict[str, Any] = {
            "model": model,
            "hyperparameters": hyperparameters,
        }

        training_type = (self.training_type or "").strip() if self.training_type else ""
        if training_type:
            payload["training_type"] = training_type

        metadata = _coerce_mapping(self.metadata, name="metadata")
        if metadata:
            payload["metadata"] = metadata

        integrations = _coerce_mapping(self.integrations, name="integrations")
        if integrations:
            payload["integrations"] = integrations

        suffix = (self.suffix or "").strip()
        if suffix:
            payload["suffix"] = suffix

        validation_file = (self.validation_file or "").strip()
        if validation_file:
            payload["validation_file"] = validation_file

        if training_file_field:
            training_file = (self.training_file or "").strip() if self.training_file else ""
            if training_file:
                payload[training_file_field] = training_file
            elif require_training_file:
                raise ValueError("training file identifier is required for SFT jobs")
            elif include_training_file_when_none:
                payload[training_file_field] = None

        return payload


def prepare_sft_job_payload(
    *,
    model: str,
    hyperparameters: Mapping[str, Any] | SFTTrainingHyperparameters | None,
    training_file: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    training_type: str | None = "sft_offline",
    validation_file: str | None = None,
    suffix: str | None = None,
    integrations: Mapping[str, Any] | None = None,
    training_file_field: str = "training_file_id",
    require_training_file: bool = True,
    include_training_file_when_none: bool = False,
    allow_finetuned_prefixes: bool = False,
) -> dict[str, Any]:
    """Validate inputs and return an SFT job payload suitable for API calls."""

    if isinstance(hyperparameters, SFTTrainingHyperparameters):
        hyper_config = hyperparameters
    else:
        hyper_config = SFTTrainingHyperparameters.from_mapping(hyperparameters or {})

    config = SFTJobConfig(
        model=model,
        training_file=training_file,
        hyperparameters=hyper_config,
        metadata=metadata,
        training_type=training_type,
        validation_file=validation_file,
        suffix=suffix,
        integrations=integrations,
    )
    return config.to_payload(
        training_file_field=training_file_field,
        require_training_file=require_training_file,
        include_training_file_when_none=include_training_file_when_none,
        allow_finetuned_prefixes=allow_finetuned_prefixes,
    )


__all__ = [
    "SFTTrainingHyperparameters",
    "SFTJobConfig",
    "prepare_sft_job_payload",
    "UnsupportedModelError",
]
