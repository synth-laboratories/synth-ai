"""Public Managed Research inference-provider enum."""

from __future__ import annotations

from enum import StrEnum


class SmrInferenceProvider(StrEnum):
    OPENAI = "openai"
    GOOGLE = "google"
    GROQ = "groq"


SMR_INFERENCE_PROVIDER_VALUES: tuple[str, ...] = tuple(
    provider.value for provider in SmrInferenceProvider
)


def coerce_smr_inference_provider(
    value: SmrInferenceProvider | str | None,
    *,
    field_name: str = "inference_provider",
) -> SmrInferenceProvider | None:
    if value is None:
        return None
    if isinstance(value, SmrInferenceProvider):
        return value
    normalized = str(value).strip()
    if not normalized:
        return None
    try:
        return SmrInferenceProvider(normalized)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be one of: {', '.join(SMR_INFERENCE_PROVIDER_VALUES)}"
        ) from exc


__all__ = [
    "SMR_INFERENCE_PROVIDER_VALUES",
    "SmrInferenceProvider",
    "coerce_smr_inference_provider",
]
