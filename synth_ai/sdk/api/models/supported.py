"""Catalog of Synth-hosted base models and helpers (core vs experimental)."""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

# ------------------------------------------------------------------------------
# Model families
# ------------------------------------------------------------------------------

QWEN3_MODELS: list[str] = [
    # Core Qwen3 base models
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-30B-A3B",
    "Qwen/Qwen3-32B",
    # 2507 baseline models
    "Qwen/Qwen3-4B-2507",
    # Instruct variants (no <think> tags)
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-4B-Instruct-2507-FP8",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
    # Thinking variants (with <think> tags)
    "Qwen/Qwen3-4B-Thinking-2507",
    "Qwen/Qwen3-4B-Thinking-2507-FP8",
    "Qwen/Qwen3-30B-A3B-Thinking-2507",
    "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",
    "Qwen/Qwen3-235B-A22B-Thinking-2507",
    "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8",
]

# Qwen3 Coder family (backend-supported); text-only, SFT/inference
QWEN3_CODER_MODELS: list[str] = [
    # Instruct variants used for coding tasks (no <think> tags)
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
    "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
]

# Qwen3-VL family (vision-language models); multimodal, SFT/inference
QWEN3_VL_MODELS: list[str] = [
    # Vision-Language Models (Qwen3-VL)
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen3-VL-2B-Thinking",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-4B-Thinking",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-8B-Thinking",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "Qwen/Qwen3-VL-30B-A3B-Thinking",
    "Qwen/Qwen3-VL-32B-Instruct",
    "Qwen/Qwen3-VL-32B-Thinking",
    "Qwen/Qwen3-VL-235B-A22B-Instruct",
    "Qwen/Qwen3-VL-235B-A22B-Thinking",
]

# Training support sets
RL_SUPPORTED_MODELS: frozenset[str] = frozenset(
    {
        # Legacy base models
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen3-14B",
        "Qwen/Qwen3-30B-A3B",
        # 2507 models - base
        "Qwen/Qwen3-4B-2507",
        # 2507 models - instruct (no <think> tags)
        "Qwen/Qwen3-4B-Instruct-2507",
        "Qwen/Qwen3-4B-Instruct-2507-FP8",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        # 2507 models - thinking (with <think> tags)
        "Qwen/Qwen3-4B-Thinking-2507",
        "Qwen/Qwen3-4B-Thinking-2507-FP8",
        "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",
        # Coder instruct models
        "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
        # Vision-Language models (Qwen3-VL)
        "Qwen/Qwen3-VL-2B-Instruct",
        "Qwen/Qwen3-VL-2B-Thinking",
        "Qwen/Qwen3-VL-4B-Instruct",
        "Qwen/Qwen3-VL-4B-Thinking",
        "Qwen/Qwen3-VL-8B-Instruct",
        "Qwen/Qwen3-VL-8B-Thinking",
    }
)

# SFT allowlist includes core Qwen3 plus Coder and VL families
SFT_SUPPORTED_MODELS: frozenset[str] = frozenset([*QWEN3_MODELS, *QWEN3_CODER_MODELS, *QWEN3_VL_MODELS])

# Models that support <think> reasoning tags
THINKING_MODELS: frozenset[str] = frozenset(
    {
        "Qwen/Qwen3-4B-Thinking-2507",
        "Qwen/Qwen3-4B-Thinking-2507-FP8",
        "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",
        "Qwen/Qwen3-235B-A22B-Thinking-2507",
        "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8",
        # Vision-Language Thinking models
        "Qwen/Qwen3-VL-2B-Thinking",
        "Qwen/Qwen3-VL-4B-Thinking",
        "Qwen/Qwen3-VL-8B-Thinking",
        "Qwen/Qwen3-VL-30B-A3B-Thinking",
        "Qwen/Qwen3-VL-32B-Thinking",
        "Qwen/Qwen3-VL-235B-A22B-Thinking",
    }
)

# ------------------------------------------------------------------------------
# Lifecycle classification (core vs experimental)
# ------------------------------------------------------------------------------

# Which base models are considered "experimental" by default.
_EXPERIMENTAL_DEFAULTS: frozenset[str] = frozenset(
    {
        # Larger (>= 64B) or bleeding-edge variants are experimental by default.
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
        "Qwen/Qwen3-235B-A22B-Thinking-2507",
        "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8",
        "Qwen/Qwen3-Coder-480B-A35B-Instruct",
        "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
        # Thinking variants can fluctuate more rapidly.
        "Qwen/Qwen3-30B-A3B-Thinking-2507",
        "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",
        "Qwen/Qwen3-4B-Thinking-2507",
        "Qwen/Qwen3-4B-Thinking-2507-FP8",
    }
)


def _parse_experimental_env() -> frozenset[str]:
    raw = os.getenv("SDK_EXPERIMENTAL_MODELS", "").strip()
    if not raw:
        return frozenset()
    return frozenset(s.strip() for s in raw.split(",") if s.strip())


# Final experimental set (defaults ∪ optional env override)
EXPERIMENTAL_MODELS: frozenset[str] = frozenset(_EXPERIMENTAL_DEFAULTS | _parse_experimental_env())

# Build catalog entries for core, coder, and VL families under unified "Qwen3"
_ALL_QWEN3_IDS: list[str] = [*QWEN3_MODELS, *QWEN3_CODER_MODELS, *QWEN3_VL_MODELS]

CORE_MODELS: frozenset[str] = frozenset(m for m in _ALL_QWEN3_IDS if m not in EXPERIMENTAL_MODELS)

# ------------------------------------------------------------------------------
# Experimental gating / warnings
# ------------------------------------------------------------------------------


class ExperimentalWarning(UserWarning):
    """Warning for usage of experimental SDK models/APIs."""


def _experimental_enabled() -> bool:
    # Global toggle to permit experimental usage
    return os.getenv("SDK_EXPERIMENTAL", "0") == "1"


def _warn_if_experimental(model_id: str) -> None:
    if model_id in EXPERIMENTAL_MODELS:
        warnings.warn(
            f"Model '{model_id}' is experimental and may change or be removed.",
            category=ExperimentalWarning,
            stacklevel=2,
        )


# ------------------------------------------------------------------------------
# Model metadata + catalog
# ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SupportedModel:
    """Metadata describing a supported base model."""

    model_id: str
    family: str
    provider: str
    modalities: tuple[str, ...] = ()
    training_modes: tuple[str, ...] = ()
    lifecycle: str = "core"  # "core" | "experimental"
    supports_thinking: bool = False  # Whether model supports <think> reasoning tags

    def as_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "model_id": self.model_id,
            "family": self.family,
            "provider": self.provider,
            "lifecycle": self.lifecycle,
            "supports_thinking": self.supports_thinking,
        }
        if self.modalities:
            data["modalities"] = list(self.modalities)
        if self.training_modes:
            data["training_modes"] = list(self.training_modes)
        return data


SUPPORTED_MODELS: tuple[SupportedModel, ...] = tuple(
    SupportedModel(
        model_id=model,
        family="Qwen3",
        provider="Qwen",
        modalities=("text",),
        training_modes=tuple(
            sorted(
                {
                    *(("sft",) if model in SFT_SUPPORTED_MODELS else ()),
                    *(("rl",) if model in RL_SUPPORTED_MODELS else ()),
                }
            )
        ),
        lifecycle=("experimental" if model in EXPERIMENTAL_MODELS else "core"),
        supports_thinking=(model in THINKING_MODELS),
    )
    for model in _ALL_QWEN3_IDS
)

_BASE_LOOKUP = {model.model_id.lower(): model.model_id for model in SUPPORTED_MODELS}
SUPPORTED_BASE_MODEL_IDS: frozenset[str] = frozenset(_BASE_LOOKUP.values())
FINE_TUNED_PREFIXES: tuple[str, ...] = ("ft:", "fft:", "qft:", "rl:")
_MODEL_BY_ID = {model.model_id: model for model in SUPPORTED_MODELS}

# ------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------


class UnsupportedModelError(ValueError):
    """Raised when a model identifier is not supported by Synth."""


def _extract_base_model(candidate: str, *, allow_finetuned_prefixes: bool) -> str | None:
    cleaned = candidate.strip()
    lowered = cleaned.lower()
    base = _BASE_LOOKUP.get(lowered)
    if base:
        return base
    if not allow_finetuned_prefixes or ":" not in cleaned:
        return None

    segments = cleaned.split(":")
    for segment in segments[1:]:
        candidate_base = segment.strip()
        if not candidate_base:
            continue
        base = _BASE_LOOKUP.get(candidate_base.lower())
        if base:
            return base
    return None


def ensure_supported_model(
    model_id: str,
    *,
    allow_finetuned_prefixes: bool = True,
) -> str:
    """Validate that *model_id* resolves to a supported base model (no lifecycle gate)."""
    candidate = (model_id or "").strip()
    if not candidate:
        raise UnsupportedModelError("Model identifier is empty")

    base = _extract_base_model(candidate, allow_finetuned_prefixes=allow_finetuned_prefixes)
    if base:
        return base

    raise UnsupportedModelError(
        f"Model '{candidate}' is not supported. Call supported_model_ids() for available base models."
    )


def ensure_allowed_model(
    model_id: str,
    *,
    allow_finetuned_prefixes: bool = True,
    allow_experimental: bool | None = None,
) -> str:
    """Validate support + lifecycle; gate experimental unless enabled."""
    base = ensure_supported_model(model_id, allow_finetuned_prefixes=allow_finetuned_prefixes)
    is_exp = base in EXPERIMENTAL_MODELS
    allow_exp = allow_experimental if allow_experimental is not None else _experimental_enabled()
    if is_exp and not allow_exp:
        raise UnsupportedModelError(
            f"Model '{base}' is experimental and disabled. "
            "Set SDK_EXPERIMENTAL=1 or pass allow_experimental=True."
        )
    if is_exp:
        _warn_if_experimental(base)
    return base


def normalize_model_identifier(
    model_id: str,
    *,
    allow_finetuned_prefixes: bool = True,
) -> str:
    """Return a cleaned model identifier suitable for job payloads (no lifecycle gate)."""
    canonical = ensure_supported_model(model_id, allow_finetuned_prefixes=allow_finetuned_prefixes)
    cleaned = (model_id or "").strip()
    if not cleaned:
        return canonical
    if cleaned.lower() in _BASE_LOOKUP:
        return canonical
    return cleaned


def is_supported_model(model_id: str, *, allow_finetuned_prefixes: bool = True) -> bool:
    """Return True if *model_id* resolves to a supported base model (ignores lifecycle)."""
    try:
        ensure_supported_model(model_id, allow_finetuned_prefixes=allow_finetuned_prefixes)
    except UnsupportedModelError:
        return False
    return True


def is_experimental_model(model_id: str) -> bool:
    """Return True if *model_id* is marked experimental."""
    try:
        base = ensure_supported_model(model_id, allow_finetuned_prefixes=True)
    except UnsupportedModelError:
        return False
    return base in EXPERIMENTAL_MODELS


def is_core_model(model_id: str) -> bool:
    """Return True if *model_id* is marked core."""
    try:
        base = ensure_supported_model(model_id, allow_finetuned_prefixes=True)
    except UnsupportedModelError:
        return False
    return base in CORE_MODELS


def iter_supported_models(
    *,
    families: Sequence[str] | None = None,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> Iterator[SupportedModel]:
    """Yield supported models, optionally filtered by family and lifecycle."""
    include_set = {s.lower() for s in include} if include else None
    exclude_set = {s.lower() for s in exclude} if exclude else None
    fam_set = {f.lower() for f in families} if families else None

    for m in SUPPORTED_MODELS:
        if fam_set is not None and m.family.lower() not in fam_set:
            continue
        if include_set is not None and m.lifecycle.lower() not in include_set:
            continue
        if exclude_set is not None and m.lifecycle.lower() in exclude_set:
            continue
        yield m


def list_supported_models(
    *,
    families: Sequence[str] | None = None,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> list[SupportedModel]:
    """Return supported models as a list for easier consumption."""
    return list(iter_supported_models(families=families, include=include, exclude=exclude))


def supported_model_ids(
    *,
    families: Sequence[str] | None = None,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> list[str]:
    """Return just the model identifiers for supported models."""
    return [m.model_id for m in iter_supported_models(families=families, include=include, exclude=exclude)]


def experimental_model_ids(*, families: Sequence[str] | None = None) -> list[str]:
    """Return identifiers for experimental supported models."""
    return supported_model_ids(families=families, include=("experimental",))


def core_model_ids(*, families: Sequence[str] | None = None) -> list[str]:
    """Return identifiers for core supported models."""
    return supported_model_ids(families=families, include=("core",))


def format_supported_models(
    *,
    families: Sequence[str] | None = None,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
) -> str:
    """Produce a human readable table of supported models."""
    rows: Iterable[SupportedModel] = iter_supported_models(families=families, include=include, exclude=exclude)
    lines = ["model_id | family | provider | lifecycle | modalities | training_modes", "-" * 96]
    for model in rows:
        modalities = ",".join(model.modalities) or "-"
        training = ",".join(model.training_modes) or "-"
        lines.append(
            f"{model.model_id} | {model.family} | {model.provider} | {model.lifecycle} | {modalities} | {training}"
        )
    return "\n".join(lines)


def training_modes_for_model(model_id: str) -> tuple[str, ...]:
    """Return the supported training modes (e.g., ('sft','rl')) for the given base model."""
    canonical = ensure_supported_model(model_id, allow_finetuned_prefixes=True)
    model = _MODEL_BY_ID.get(canonical)
    if not model:
        raise UnsupportedModelError(f"Model '{model_id}' is not registered as supported.")
    return model.training_modes


def supports_thinking(model_id: str) -> bool:
    """Return True if the model supports <think> reasoning tags.
    
    Thinking models use structured <think>...</think> tags for reasoning.
    Instruct models do not have these tags and should not use thinking-specific logic.
    
    Args:
        model_id: Model identifier (can include prefixes like 'rl:', 'fft:', etc.)
        
    Returns:
        True if the model supports thinking tags, False otherwise.
        Returns False for unsupported models.
        
    Example:
        >>> supports_thinking("Qwen/Qwen3-4B-Thinking-2507")
        True
        >>> supports_thinking("Qwen/Qwen3-4B-Instruct-2507")
        False
        >>> supports_thinking("rl:Qwen/Qwen3-4B-Thinking-2507")
        True
    """
    try:
        canonical = ensure_supported_model(model_id, allow_finetuned_prefixes=True)
    except UnsupportedModelError:
        return False
    model = _MODEL_BY_ID.get(canonical)
    if not model:
        return False
    return model.supports_thinking


def get_model_metadata(model_id: str) -> SupportedModel | None:
    """Return the full metadata for a supported model, or None if not supported.
    
    Args:
        model_id: Model identifier (can include prefixes like 'rl:', 'fft:', etc.)
        
    Returns:
        SupportedModel instance with full metadata, or None if model is not supported.
        
    Example:
        >>> meta = get_model_metadata("Qwen/Qwen3-4B-Instruct-2507")
        >>> meta.supports_thinking
        False
        >>> meta.training_modes
        ('rl', 'sft')
    """
    try:
        canonical = ensure_supported_model(model_id, allow_finetuned_prefixes=True)
    except UnsupportedModelError:
        return None
    return _MODEL_BY_ID.get(canonical)


__all__ = [
    "QWEN3_MODELS",
    "QWEN3_CODER_MODELS",
    "RL_SUPPORTED_MODELS",
    "SFT_SUPPORTED_MODELS",
    "THINKING_MODELS",
    "EXPERIMENTAL_MODELS",
    "CORE_MODELS",
    "ExperimentalWarning",
    "SupportedModel",
    "SUPPORTED_MODELS",
    "SUPPORTED_BASE_MODEL_IDS",
    "FINE_TUNED_PREFIXES",
    "UnsupportedModelError",
    "ensure_supported_model",
    "ensure_allowed_model",
    "normalize_model_identifier",
    "is_supported_model",
    "is_experimental_model",
    "is_core_model",
    "iter_supported_models",
    "list_supported_models",
    "supported_model_ids",
    "experimental_model_ids",
    "core_model_ids",
    "format_supported_models",
    "training_modes_for_model",
    "supports_thinking",
    "get_model_metadata",
]

