"""Public Managed Research provider binding and usage-limit models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class Provider(StrEnum):
    OPENROUTER = "openrouter"
    TINKER = "tinker"
    SYNTH_AI = "synth_ai"
    CURSOR = "cursor"
    DEEPSEEK = "deepseek"
    GROQ = "groq"


class ActorResourceCapability(StrEnum):
    INFERENCE = "inference"
    TRAINING = "training"
    MODEL_ARTIFACT = "model_artifact"


PROVIDER_VALUES: tuple[str, ...] = tuple(provider.value for provider in Provider)
PROVIDER_CAPABILITY_VALUES: tuple[str, ...] = tuple(
    capability.value for capability in ActorResourceCapability
)

ACTOR_RESOURCE_CAPABILITIES: dict[Provider, frozenset[ActorResourceCapability]] = {
    Provider.OPENROUTER: frozenset({ActorResourceCapability.INFERENCE}),
    Provider.TINKER: frozenset(
        {
            ActorResourceCapability.INFERENCE,
            ActorResourceCapability.TRAINING,
            ActorResourceCapability.MODEL_ARTIFACT,
        }
    ),
    Provider.SYNTH_AI: frozenset(
        {
            ActorResourceCapability.INFERENCE,
            ActorResourceCapability.TRAINING,
            ActorResourceCapability.MODEL_ARTIFACT,
        }
    ),
    Provider.CURSOR: frozenset({ActorResourceCapability.INFERENCE}),
    Provider.DEEPSEEK: frozenset({ActorResourceCapability.INFERENCE}),
    Provider.GROQ: frozenset({ActorResourceCapability.INFERENCE}),
}


def _clean_string(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _optional_float(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a number when provided")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number when provided") from exc


def _optional_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer when provided")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer when provided") from exc


def _string_tuple(value: Any, *, field_name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be an array of strings when provided")
    normalized = tuple(text for item in value if (text := _clean_string(item)))
    return normalized or None


@dataclass(frozen=True)
class UsageLimit:
    max_spend_usd: float | None = None
    max_wallclock_seconds: int | None = None
    max_gpu_hours: float | None = None
    max_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.max_spend_usd is not None:
            payload["max_spend_usd"] = float(self.max_spend_usd)
        if self.max_wallclock_seconds is not None:
            payload["max_wallclock_seconds"] = int(self.max_wallclock_seconds)
        if self.max_gpu_hours is not None:
            payload["max_gpu_hours"] = float(self.max_gpu_hours)
        if self.max_tokens is not None:
            payload["max_tokens"] = int(self.max_tokens)
        return payload


@dataclass(frozen=True)
class OpenRouterConfig:
    allowed_models: tuple[str, ...] | None = None
    provider_preferences: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.allowed_models is not None:
            payload["allowed_models"] = list(self.allowed_models)
        if self.provider_preferences:
            payload["provider_preferences"] = dict(self.provider_preferences)
        return payload


@dataclass(frozen=True)
class TinkerConfig:
    base_model: str | None = None
    fine_tune_mode: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.base_model is not None:
            payload["base_model"] = self.base_model
        if self.fine_tune_mode is not None:
            payload["fine_tune_mode"] = self.fine_tune_mode
        return payload


@dataclass(frozen=True)
class SynthAIConfig:
    def to_dict(self) -> dict[str, Any]:
        return {}


ProviderConfig = OpenRouterConfig | TinkerConfig | SynthAIConfig

DEFAULT_CONFIGS: dict[Provider, ProviderConfig] = {
    Provider.OPENROUTER: OpenRouterConfig(),
    Provider.TINKER: TinkerConfig(),
    Provider.SYNTH_AI: SynthAIConfig(),
    Provider.CURSOR: SynthAIConfig(),
    Provider.DEEPSEEK: SynthAIConfig(),
    Provider.GROQ: SynthAIConfig(),
}


@dataclass(frozen=True)
class ProviderBinding:
    provider: Provider
    config: ProviderConfig | None = None
    limit: UsageLimit | None = None

    def to_wire(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"provider": self.provider.value}
        config = self.config or DEFAULT_CONFIGS[self.provider]
        config_payload = config.to_dict()
        if config_payload:
            payload["config"] = config_payload
        if self.limit is not None:
            limit_payload = self.limit.to_dict()
            if limit_payload:
                payload["limit"] = limit_payload
        return payload

    @property
    def capabilities(self) -> frozenset[ActorResourceCapability]:
        return ACTOR_RESOURCE_CAPABILITIES[self.provider]


def coerce_provider(
    value: Provider | str | None,
    *,
    field_name: str = "provider",
) -> Provider | None:
    if value is None:
        return None
    if isinstance(value, Provider):
        return value
    normalized = str(value).strip()
    if not normalized:
        return None
    try:
        return Provider(normalized)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be one of: {', '.join(PROVIDER_VALUES)}") from exc


def coerce_usage_limit(
    value: UsageLimit | Mapping[str, Any] | None,
    *,
    field_name: str = "limit",
) -> UsageLimit | None:
    if value is None:
        return None
    if isinstance(value, UsageLimit):
        return value
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object when provided")
    return UsageLimit(
        max_spend_usd=_optional_float(
            value.get("max_spend_usd"),
            field_name=f"{field_name}.max_spend_usd",
        ),
        max_wallclock_seconds=_optional_int(
            value.get("max_wallclock_seconds"),
            field_name=f"{field_name}.max_wallclock_seconds",
        ),
        max_gpu_hours=_optional_float(
            value.get("max_gpu_hours"),
            field_name=f"{field_name}.max_gpu_hours",
        ),
        max_tokens=_optional_int(
            value.get("max_tokens"),
            field_name=f"{field_name}.max_tokens",
        ),
    )


def _coerce_config(
    provider: Provider,
    value: ProviderConfig | Mapping[str, Any] | None,
    *,
    field_name: str,
) -> ProviderConfig:
    if value is None:
        return DEFAULT_CONFIGS[provider]
    if isinstance(value, (OpenRouterConfig, TinkerConfig, SynthAIConfig)):
        return value
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object when provided")
    if provider is Provider.OPENROUTER:
        provider_preferences = value.get("provider_preferences")
        if provider_preferences is not None and not isinstance(provider_preferences, Mapping):
            raise ValueError(f"{field_name}.provider_preferences must be an object")
        return OpenRouterConfig(
            allowed_models=_string_tuple(
                value.get("allowed_models"),
                field_name=f"{field_name}.allowed_models",
            ),
            provider_preferences=(
                dict(provider_preferences) if isinstance(provider_preferences, Mapping) else None
            ),
        )
    if provider is Provider.TINKER:
        return TinkerConfig(
            base_model=_clean_string(value.get("base_model")),
            fine_tune_mode=_clean_string(value.get("fine_tune_mode")),
        )
    return SynthAIConfig()


def coerce_provider_binding(
    value: ProviderBinding | Provider | str | Mapping[str, Any],
    *,
    field_name: str = "providers[]",
) -> ProviderBinding:
    if isinstance(value, ProviderBinding):
        return value
    if isinstance(value, Provider):
        return ProviderBinding(provider=value)
    if isinstance(value, str):
        provider = coerce_provider(value, field_name=field_name)
        if provider is None:
            raise ValueError(f"{field_name}.provider is required")
        return ProviderBinding(provider=provider)
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} entries must be provider strings or objects")
    provider = coerce_provider(value.get("provider"), field_name=f"{field_name}.provider")
    if provider is None:
        raise ValueError(f"{field_name}.provider is required")
    return ProviderBinding(
        provider=provider,
        config=_coerce_config(
            provider,
            value.get("config"),
            field_name=f"{field_name}.config",
        ),
        limit=coerce_usage_limit(value.get("limit"), field_name=f"{field_name}.limit"),
    )


def coerce_provider_bindings(
    values: Sequence[ProviderBinding | Provider | str | Mapping[str, Any]] | None,
    *,
    field_name: str = "providers",
) -> tuple[ProviderBinding, ...]:
    if values is None:
        raise ValueError(f"{field_name} is required")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{field_name} must be an array")
    normalized = tuple(
        coerce_provider_binding(item, field_name=f"{field_name}[{index}]")
        for index, item in enumerate(values)
    )
    if not normalized:
        raise ValueError(f"{field_name} must include at least one provider binding")
    return normalized


def provider_capabilities(
    bindings: Sequence[ProviderBinding | Provider | str | Mapping[str, Any]],
) -> frozenset[ActorResourceCapability]:
    capabilities: set[ActorResourceCapability] = set()
    for binding in coerce_provider_bindings(bindings):
        capabilities.update(binding.capabilities)
    return frozenset(capabilities)


__all__ = [
    "DEFAULT_CONFIGS",
    "ACTOR_RESOURCE_CAPABILITIES",
    "PROVIDER_CAPABILITY_VALUES",
    "PROVIDER_VALUES",
    "OpenRouterConfig",
    "Provider",
    "ProviderBinding",
    "ActorResourceCapability",
    "ProviderConfig",
    "SynthAIConfig",
    "TinkerConfig",
    "UsageLimit",
    "coerce_provider",
    "coerce_provider_binding",
    "coerce_provider_bindings",
    "coerce_usage_limit",
    "provider_capabilities",
]
