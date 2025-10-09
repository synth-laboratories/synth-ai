from __future__ import annotations

"""Catalog of Synth-hosted base models and helpers for discovery."""

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence


QWEN3_MODELS: list[str] = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
]


@dataclass(frozen=True, slots=True)
class SupportedModel:
    """Metadata describing a supported base model."""

    model_id: str
    family: str
    provider: str
    modalities: tuple[str, ...] = ()
    training_modes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "model_id": self.model_id,
            "family": self.family,
            "provider": self.provider,
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
        training_modes=("sft", "rl"),
    )
    for model in QWEN3_MODELS
)


_BASE_LOOKUP = {model.model_id.lower(): model.model_id for model in SUPPORTED_MODELS}
SUPPORTED_BASE_MODEL_IDS: frozenset[str] = frozenset(_BASE_LOOKUP.values())
FINE_TUNED_PREFIXES: tuple[str, ...] = ("ft:", "fft:", "qft:")


class UnsupportedModelError(ValueError):
    """Raised when a model identifier is not supported by Synth."""


def _extract_base_model(candidate: str, *, allow_finetuned_prefixes: bool) -> str | None:
    cleaned = candidate.strip()
    lowered = cleaned.lower()
    base = _BASE_LOOKUP.get(lowered)
    if base:
        return base
    if not allow_finetuned_prefixes:
        return None
    if ":" not in cleaned:
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
    """Validate that *model_id* resolves to a supported base model.

    Returns the canonical base model identifier on success, otherwise raises
    :class:`UnsupportedModelError`.
    """

    candidate = (model_id or "").strip()
    if not candidate:
        raise UnsupportedModelError("Model identifier is empty")

    base = _extract_base_model(candidate, allow_finetuned_prefixes=allow_finetuned_prefixes)
    if base:
        return base

    raise UnsupportedModelError(
        f"Model '{candidate}' is not supported. Call supported_model_ids() for available base models."
    )


def normalize_model_identifier(
    model_id: str,
    *,
    allow_finetuned_prefixes: bool = True,
) -> str:
    """Return a cleaned model identifier suitable for job payloads."""

    canonical = ensure_supported_model(model_id, allow_finetuned_prefixes=allow_finetuned_prefixes)
    cleaned = (model_id or "").strip()
    if not cleaned:
        return canonical
    if cleaned.lower() in _BASE_LOOKUP:
        return canonical
    return cleaned


def is_supported_model(model_id: str, *, allow_finetuned_prefixes: bool = True) -> bool:
    """Return True if *model_id* resolves to a supported base model."""

    try:
        ensure_supported_model(model_id, allow_finetuned_prefixes=allow_finetuned_prefixes)
    except UnsupportedModelError:
        return False
    return True


def iter_supported_models(*, families: Sequence[str] | None = None) -> Iterator[SupportedModel]:
    """Yield supported models, optionally filtered by family name."""

    if families is None:
        yield from SUPPORTED_MODELS
        return
    family_filter = {family.lower() for family in families}
    for model in SUPPORTED_MODELS:
        if model.family.lower() in family_filter:
            yield model


def list_supported_models(*, families: Sequence[str] | None = None) -> list[SupportedModel]:
    """Return supported models as a list for easier consumption."""

    return list(iter_supported_models(families=families))


def supported_model_ids(*, families: Sequence[str] | None = None) -> list[str]:
    """Return just the model identifiers for supported models."""

    return [model.model_id for model in iter_supported_models(families=families)]


def format_supported_models(*, families: Sequence[str] | None = None) -> str:
    """Produce a human readable table of supported models."""

    rows: Iterable[SupportedModel] = iter_supported_models(families=families)
    lines = ["model_id | family | provider | modalities | training_modes", "-" * 78]
    for model in rows:
        modalities = ",".join(model.modalities) or "-"
        training = ",".join(model.training_modes) or "-"
        lines.append(
            f"{model.model_id} | {model.family} | {model.provider} | {modalities} | {training}"
        )
    return "\n".join(lines)


__all__ = [
    "QWEN3_MODELS",
    "SupportedModel",
    "SUPPORTED_MODELS",
    "SUPPORTED_BASE_MODEL_IDS",
    "FINE_TUNED_PREFIXES",
    "UnsupportedModelError",
    "ensure_supported_model",
    "normalize_model_identifier",
    "is_supported_model",
    "iter_supported_models",
    "list_supported_models",
    "supported_model_ids",
    "format_supported_models",
]
