from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

try:
    _models_module = cast(
        Any, importlib.import_module("synth_ai.sdk.api.models.supported")
    )
    RL_SUPPORTED_MODELS = cast(tuple[str, ...], _models_module.RL_SUPPORTED_MODELS)
    SFT_SUPPORTED_MODELS = cast(tuple[str, ...], _models_module.SFT_SUPPORTED_MODELS)
    training_modes_for_model = cast(
        Callable[[str], tuple[str, ...]], _models_module.training_modes_for_model
    )
except Exception as exc:  # pragma: no cover - critical dependency
    raise RuntimeError("Unable to load supported model metadata") from exc


@dataclass(frozen=True)
class AlgorithmSpec:
    algo_type: str
    method: str
    variety: str
    label: str  # Human readable identifier (e.g. "RL / GSPO")
    required_training_mode: str  # Expected training mode on the model record (e.g. "rl", "sft")


def _normalize(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()

RL_ALGORITHM_MODEL_IDS: tuple[str, ...] = tuple(sorted(RL_SUPPORTED_MODELS))
RL_ALGORITHM_MODEL_SET: frozenset[str] = frozenset(RL_ALGORITHM_MODEL_IDS)

SFT_ALGORITHM_MODEL_IDS: tuple[str, ...] = tuple(sorted(SFT_SUPPORTED_MODELS))
SFT_ALGORITHM_MODEL_SET: frozenset[str] = frozenset(SFT_ALGORITHM_MODEL_IDS)


SUPPORTED_ALGORITHMS: tuple[AlgorithmSpec, ...] = (
    AlgorithmSpec(
        "online", "policy_gradient", "gspo", "online policy_gradient / gspo", "rl"
    ),
    AlgorithmSpec(
        "offline", "supervised_finetune", "fft", "offline supervised_finetune / fft", "sft"
    ),
    AlgorithmSpec("offline", "sft", "fft", "offline sft / fft", "sft"),
    # Accept explicit LoRA variety for SFT as a first-class alias of FFT SFT.
    # This allows configs to declare intent with variety="lora" while still
    # using SFT training mode; actual adapter selection is driven by [training].
    AlgorithmSpec("offline", "sft", "lora", "offline sft / lora", "sft"),
)

_SUPPORTED_LOOKUP = {
    (spec.algo_type, spec.method, spec.variety): spec for spec in SUPPORTED_ALGORITHMS
}


class AlgorithmValidationError(ValueError):
    """Raised when an algorithm block contains unsupported combinations."""


def validate_algorithm_config(
    algorithm_block: Mapping[str, object] | None,
    *,
    expected_family: str | None = None,
) -> AlgorithmSpec:
    """Validate the [algorithm] section of a training config.

    Args:
        algorithm_block: Parsed mapping from the TOML config.
        expected_family: Optional expected family label ("rl" or "sft").

    Returns:
        The matched AlgorithmSpec describing the supported combination.

    Raises:
        AlgorithmValidationError: if the combination is missing or unsupported.
    """

    if algorithm_block is None:
        raise AlgorithmValidationError("Missing required [algorithm] section in config.")
    if not isinstance(algorithm_block, Mapping):
        raise AlgorithmValidationError("[algorithm] section must be a mapping/object.")

    algo_type = _normalize(algorithm_block.get("type"))
    method = _normalize(algorithm_block.get("method"))
    variety = _normalize(algorithm_block.get("variety"))

    key = (algo_type, method, variety)
    spec = _SUPPORTED_LOOKUP.get(key)
    if spec is None:
        supported = "; ".join(
            f"type='{entry.algo_type}', method='{entry.method}', variety='{entry.variety}'"
            for entry in SUPPORTED_ALGORITHMS
        )
        raise AlgorithmValidationError(
            "Unsupported algorithm configuration:\n"
            f"  type={algo_type!r}, method={method!r}, variety={variety!r}\n"
            f"Supported combinations are: {supported}"
        )

    if expected_family:
        expected_family = expected_family.lower()
        family_map = {
            ("online", "policy_gradient", "gspo"): "rl",
            ("offline", "supervised_finetune", "fft"): "sft",
            ("offline", "sft", "fft"): "sft",
            ("offline", "sft", "lora"): "sft",
        }
        family = family_map.get(key)
        if family != expected_family:
            raise AlgorithmValidationError(
                f"Config contains algorithm {spec.label!r}, "
                f"but the current command expects {expected_family.upper()}."
            )

    return spec


def ensure_model_supported_for_algorithm(model_id: str, spec: AlgorithmSpec) -> None:
    """Ensure that the given model supports the training mode required by *spec*."""

    modes = {mode.lower() for mode in training_modes_for_model(model_id)}
    required = spec.required_training_mode.lower()
    if required not in modes:
        if required == "rl":
            allowed = ", ".join(RL_ALGORITHM_MODEL_IDS)
        else:
            allowed = ", ".join(SFT_ALGORITHM_MODEL_IDS)
        raise AlgorithmValidationError(
            f"Model '{model_id}' does not support {spec.label} workloads "
            f"(missing training mode '{required}'). Supported model IDs: {allowed}"
        )


__all__ = [
    "AlgorithmSpec",
    "AlgorithmValidationError",
    "RL_ALGORITHM_MODEL_IDS",
    "SFT_ALGORITHM_MODEL_IDS",
    "SUPPORTED_ALGORITHMS",
    "validate_algorithm_config",
    "ensure_model_supported_for_algorithm",
]
