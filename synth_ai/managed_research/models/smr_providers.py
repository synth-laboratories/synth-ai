"""Public Managed Research provider binding and usage-limit models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ResourceProvider(StrEnum):
    """External resource vendors an actor may use during work (``providers[]``).

    Not the actor runtime — these are inference/training/tool backends. Wire key
    (``provider``) and string values are unchanged.
    """

    OPENROUTER = "openrouter"
    TINKER = "tinker"
    SYNTH_AI = "synth_ai"
    CURSOR = "cursor"
    DEEPSEEK = "deepseek"
    XAI = "xai"
    MODAL = "modal"
    OPENAI_CHATGPT = "openai_chatgpt"
    BASETEN = "baseten"


# Deprecated alias retained one release; prefer ``ResourceProvider``.
Provider = ResourceProvider


class ResourceCapability(StrEnum):
    INFERENCE = "inference"
    TRAINING = "training"
    MODEL_ARTIFACT = "model_artifact"


# Deprecated alias retained one release; prefer ``ResourceCapability``.
ActorResourceCapability = ResourceCapability


PROVIDER_VALUES: tuple[str, ...] = tuple(provider.value for provider in Provider)
PROVIDER_CAPABILITY_VALUES: tuple[str, ...] = tuple(
    capability.value for capability in ActorResourceCapability
)
DEFAULT_PROVIDER_POLICY_ALLOWED_PROVIDERS: tuple[str, ...] = (
    "openai",
    "baseten",
    "gemini",
    "grok",
)
DEFAULT_PROVIDER_POLICY_ALLOWED_DOMICILES: tuple[str, ...] = ("us",)
DEFAULT_PROVIDER_POLICY_ALLOWED_REGIONS: tuple[str, ...] = ("us",)

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
    Provider.XAI: frozenset({ActorResourceCapability.INFERENCE}),
    Provider.MODAL: frozenset({ActorResourceCapability.INFERENCE}),
    Provider.OPENAI_CHATGPT: frozenset({ActorResourceCapability.INFERENCE}),
    Provider.BASETEN: frozenset({ActorResourceCapability.INFERENCE}),
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


def _optional_bool(value: Any, *, field_name: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(f"{field_name} must be a boolean when provided")


def _string_tuple(value: Any, *, field_name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be an array of strings when provided")
    normalized = tuple(text for item in value if (text := _clean_string(item)))
    return normalized or None


def _lower_string_tuple(value: Any, *, field_name: str) -> tuple[str, ...] | None:
    normalized = _string_tuple(value, field_name=field_name)
    if normalized is None:
        return None
    return tuple(item.lower() for item in normalized)


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


@dataclass(frozen=True)
class OpenAIChatGPTConfig:
    """ChatGPT/Codex subscription pool (`openai_chatgpt_pool`), not OpenAI Platform API."""

    def to_dict(self) -> dict[str, Any]:
        return {}


ProviderConfig = OpenRouterConfig | TinkerConfig | SynthAIConfig | OpenAIChatGPTConfig

DEFAULT_CONFIGS: dict[Provider, ProviderConfig] = {
    Provider.OPENROUTER: OpenRouterConfig(),
    Provider.TINKER: TinkerConfig(),
    Provider.SYNTH_AI: SynthAIConfig(),
    Provider.CURSOR: SynthAIConfig(),
    Provider.DEEPSEEK: SynthAIConfig(),
    Provider.XAI: SynthAIConfig(),
    Provider.MODAL: SynthAIConfig(),
    Provider.OPENAI_CHATGPT: OpenAIChatGPTConfig(),
    Provider.BASETEN: SynthAIConfig(),
}


@dataclass(frozen=True)
class ResourceProviderBinding:
    provider: ResourceProvider
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
    def capabilities(self) -> frozenset[ResourceCapability]:
        return ACTOR_RESOURCE_CAPABILITIES[self.provider]


# Deprecated alias retained one release; prefer ``ResourceProviderBinding``.
ProviderBinding = ResourceProviderBinding


@dataclass(frozen=True)
class ResourceRoutingPolicy:
    allowed_providers: tuple[str, ...] | None = None
    denied_providers: tuple[str, ...] | None = None
    preferred_providers: tuple[str, ...] | None = None
    allowed_models: tuple[str, ...] | None = None
    denied_models: tuple[str, ...] | None = None
    preferred_models: tuple[str, ...] | None = None
    require_zdr: bool | None = None
    allowed_domiciles: tuple[str, ...] | None = None
    allowed_regions: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "allowed_providers",
            "denied_providers",
            "preferred_providers",
            "allowed_domiciles",
            "allowed_regions",
        ):
            normalized = _lower_string_tuple(
                getattr(self, field_name),
                field_name=field_name,
            )
            object.__setattr__(self, field_name, normalized)
        for field_name in ("allowed_models", "denied_models", "preferred_models"):
            normalized = _string_tuple(getattr(self, field_name), field_name=field_name)
            object.__setattr__(self, field_name, normalized)
        allowed = set(self.allowed_providers or ())
        denied = set(self.denied_providers or ())
        preferred = set(self.preferred_providers or ())
        if allowed and allowed.intersection(denied):
            raise ValueError("allowed_providers and denied_providers must not overlap")
        if preferred.intersection(denied):
            raise ValueError("preferred_providers and denied_providers must not overlap")
        allowed_models = set(self.allowed_models or ())
        denied_models = set(self.denied_models or ())
        preferred_models = set(self.preferred_models or ())
        if allowed_models and allowed_models.intersection(denied_models):
            raise ValueError("allowed_models and denied_models must not overlap")
        if preferred_models.intersection(denied_models):
            raise ValueError("preferred_models and denied_models must not overlap")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.allowed_providers is not None:
            payload["allowed_providers"] = list(self.allowed_providers)
        if self.denied_providers is not None:
            payload["denied_providers"] = list(self.denied_providers)
        if self.preferred_providers is not None:
            payload["preferred_providers"] = list(self.preferred_providers)
        if self.allowed_models is not None:
            payload["allowed_models"] = list(self.allowed_models)
        if self.denied_models is not None:
            payload["denied_models"] = list(self.denied_models)
        if self.preferred_models is not None:
            payload["preferred_models"] = list(self.preferred_models)
        if self.require_zdr is not None:
            payload["require_zdr"] = bool(self.require_zdr)
        if self.allowed_domiciles is not None:
            payload["allowed_domiciles"] = list(self.allowed_domiciles)
        if self.allowed_regions is not None:
            payload["allowed_regions"] = list(self.allowed_regions)
        return payload


# Deprecated alias retained one release; prefer ``ResourceRoutingPolicy``.
ProviderRoutingPolicy = ResourceRoutingPolicy


def default_provider_routing_policy() -> ResourceRoutingPolicy:
    return ResourceRoutingPolicy(
        allowed_providers=DEFAULT_PROVIDER_POLICY_ALLOWED_PROVIDERS,
        require_zdr=True,
        allowed_domiciles=DEFAULT_PROVIDER_POLICY_ALLOWED_DOMICILES,
        allowed_regions=DEFAULT_PROVIDER_POLICY_ALLOWED_REGIONS,
    )


@dataclass(frozen=True)
class ResourceProviderPolicy:
    default: ResourceRoutingPolicy | None = field(
        default_factory=default_provider_routing_policy
    )
    agent_inference: ResourceRoutingPolicy | None = None
    experiment_inference: ResourceRoutingPolicy | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key in ("default", "agent_inference", "experiment_inference"):
            policy = getattr(self, key)
            if policy is not None:
                policy_payload = policy.to_dict()
                if policy_payload:
                    payload[key] = policy_payload
        return payload


# Deprecated alias retained one release; prefer ``ResourceProviderPolicy``.
ProviderPolicy = ResourceProviderPolicy


def default_provider_policy() -> ResourceProviderPolicy:
    return ResourceProviderPolicy()


def provider_policy_zdr_override(*, require_zdr: bool) -> ProviderPolicy:
    """Partial ``provider_policy`` that overrides ZDR for a hosted launch.

    Hosted prod/staging always enforces US domicile/regions on the backend.
    Only ``require_zdr`` is intended to be overridden from the SDK; domicile
    cannot be turned off.
    """

    return ProviderPolicy(
        default=ProviderRoutingPolicy(require_zdr=require_zdr),
    )


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
    if isinstance(value, (OpenRouterConfig, TinkerConfig, SynthAIConfig, OpenAIChatGPTConfig)):
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


def _coerce_policy_provider_tuple(
    value: Any,
    *,
    field_name: str,
) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be an array of providers when provided")
    return _lower_string_tuple(value, field_name=field_name)


def _coerce_routing_policy(
    value: ProviderRoutingPolicy | Mapping[str, Any] | None,
    *,
    field_name: str,
) -> ProviderRoutingPolicy | None:
    if value is None:
        return None
    if isinstance(value, ProviderRoutingPolicy):
        return value
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object when provided")
    return ProviderRoutingPolicy(
        allowed_providers=_coerce_policy_provider_tuple(
            value.get("allowed_providers"),
            field_name=f"{field_name}.allowed_providers",
        ),
        denied_providers=_coerce_policy_provider_tuple(
            value.get("denied_providers"),
            field_name=f"{field_name}.denied_providers",
        ),
        preferred_providers=_coerce_policy_provider_tuple(
            value.get("preferred_providers"),
            field_name=f"{field_name}.preferred_providers",
        ),
        allowed_models=_string_tuple(
            value.get("allowed_models"),
            field_name=f"{field_name}.allowed_models",
        ),
        denied_models=_string_tuple(
            value.get("denied_models"),
            field_name=f"{field_name}.denied_models",
        ),
        preferred_models=_string_tuple(
            value.get("preferred_models"),
            field_name=f"{field_name}.preferred_models",
        ),
        require_zdr=_optional_bool(
            value.get("require_zdr"),
            field_name=f"{field_name}.require_zdr",
        ),
        allowed_domiciles=_lower_string_tuple(
            value.get("allowed_domiciles"),
            field_name=f"{field_name}.allowed_domiciles",
        ),
        allowed_regions=_lower_string_tuple(
            value.get("allowed_regions"),
            field_name=f"{field_name}.allowed_regions",
        ),
    )


def coerce_provider_policy(
    value: ProviderPolicy | Mapping[str, Any] | None,
    *,
    field_name: str = "provider_policy",
) -> ProviderPolicy | None:
    if value is None:
        return None
    if isinstance(value, ProviderPolicy):
        return value
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object when provided")
    return ProviderPolicy(
        default=(
            _coerce_routing_policy(
                value.get("default"),
                field_name=f"{field_name}.default",
            )
            if "default" in value
            else default_provider_routing_policy()
        ),
        agent_inference=_coerce_routing_policy(
            value.get("agent_inference"),
            field_name=f"{field_name}.agent_inference",
        ),
        experiment_inference=_coerce_routing_policy(
            value.get("experiment_inference"),
            field_name=f"{field_name}.experiment_inference",
        ),
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
    "OpenAIChatGPTConfig",
    "OpenRouterConfig",
    "Provider",
    "ProviderBinding",
    "ProviderPolicy",
    "ProviderRoutingPolicy",
    "ActorResourceCapability",
    "ProviderConfig",
    "SynthAIConfig",
    "TinkerConfig",
    "UsageLimit",
    "coerce_provider",
    "coerce_provider_binding",
    "coerce_provider_bindings",
    "coerce_provider_policy",
    "coerce_usage_limit",
    "default_provider_policy",
    "default_provider_routing_policy",
    "provider_capabilities",
    "provider_policy_zdr_override",
]
