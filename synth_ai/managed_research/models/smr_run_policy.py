"""Typed public run-policy helpers for the rewritten SDK surface."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

from synth_ai.managed_research.models.smr_credential_providers import (
    SmrCredentialProvider,
    coerce_smr_credential_provider,
)
from synth_ai.managed_research.models.smr_funding_sources import (
    SmrFundingSource,
    coerce_smr_funding_source,
)
from synth_ai.managed_research.models.smr_inference_providers import (
    SmrInferenceProvider,
    coerce_smr_inference_provider,
)
from synth_ai.managed_research.models.smr_tool_providers import (
    SmrToolProvider,
    coerce_smr_tool_provider,
)

_ProviderEnumT = TypeVar("_ProviderEnumT")


def _optional_provider_values(
    values: tuple[_ProviderEnumT, ...] | list[_ProviderEnumT] | None,
) -> list[str] | None:
    if values is None:
        return None
    return [str(value) for value in values]


@dataclass(frozen=True)
class SmrRunPolicyAccess:
    credential_providers: tuple[SmrCredentialProvider, ...] | None = None
    inference_providers: tuple[SmrInferenceProvider, ...] | None = None
    tool_providers: tuple[SmrToolProvider, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.credential_providers is not None:
            payload["credential_providers"] = _optional_provider_values(self.credential_providers)
        if self.inference_providers is not None:
            payload["inference_providers"] = _optional_provider_values(self.inference_providers)
        if self.tool_providers is not None:
            payload["tool_providers"] = _optional_provider_values(self.tool_providers)
        return payload


@dataclass(frozen=True)
class SmrRunPolicyLimits:
    total_cost_cents: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.total_cost_cents is not None:
            payload["total_cost_cents"] = int(self.total_cost_cents)
        return payload


@dataclass(frozen=True)
class SmrRunPolicy:
    funding_source: SmrFundingSource | None = None
    access: SmrRunPolicyAccess | None = None
    limits: SmrRunPolicyLimits | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.funding_source is not None:
            payload["funding_source"] = self.funding_source.value
        if self.access is not None:
            payload["access"] = self.access.to_dict()
        if self.limits is not None:
            payload["limits"] = self.limits.to_dict()
        return payload


def _coerce_credential_provider_list(
    values: Any,
    *,
    field_name: str,
) -> tuple[SmrCredentialProvider, ...] | None:
    if values is None:
        return None
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{field_name} must be an array when provided")
    normalized = [coerce_smr_credential_provider(value, field_name=field_name) for value in values]
    return tuple(value for value in normalized if value is not None)


def _coerce_inference_provider_list(
    values: Any,
    *,
    field_name: str,
) -> tuple[SmrInferenceProvider, ...] | None:
    if values is None:
        return None
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{field_name} must be an array when provided")
    normalized = [coerce_smr_inference_provider(value, field_name=field_name) for value in values]
    return tuple(value for value in normalized if value is not None)


def _coerce_tool_provider_list(
    values: Any,
    *,
    field_name: str,
) -> tuple[SmrToolProvider, ...] | None:
    if values is None:
        return None
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{field_name} must be an array when provided")
    normalized = [coerce_smr_tool_provider(value, field_name=field_name) for value in values]
    return tuple(value for value in normalized if value is not None)


def coerce_smr_run_policy_access(
    value: SmrRunPolicyAccess | Mapping[str, Any] | None,
    *,
    field_name: str = "run_policy.access",
) -> SmrRunPolicyAccess | None:
    if value is None:
        return None
    if isinstance(value, SmrRunPolicyAccess):
        return value
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping when provided")
    return SmrRunPolicyAccess(
        credential_providers=_coerce_credential_provider_list(
            value.get("credential_providers"),
            field_name=f"{field_name}.credential_providers",
        ),
        inference_providers=_coerce_inference_provider_list(
            value.get("inference_providers"),
            field_name=f"{field_name}.inference_providers",
        ),
        tool_providers=_coerce_tool_provider_list(
            value.get("tool_providers"),
            field_name=f"{field_name}.tool_providers",
        ),
    )


def coerce_smr_run_policy_limits(
    value: SmrRunPolicyLimits | Mapping[str, Any] | None,
    *,
    field_name: str = "run_policy.limits",
) -> SmrRunPolicyLimits | None:
    if value is None:
        return None
    if isinstance(value, SmrRunPolicyLimits):
        return value
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping when provided")
    total_cost_cents = value.get("total_cost_cents")
    if total_cost_cents is not None and (
        isinstance(total_cost_cents, bool) or not isinstance(total_cost_cents, int)
    ):
        raise ValueError(f"{field_name}.total_cost_cents must be an integer when provided")
    return SmrRunPolicyLimits(
        total_cost_cents=int(total_cost_cents) if total_cost_cents is not None else None
    )


def coerce_smr_run_policy(
    value: SmrRunPolicy | Mapping[str, Any] | None,
    *,
    field_name: str = "run_policy",
) -> SmrRunPolicy | None:
    if value is None:
        return None
    if isinstance(value, SmrRunPolicy):
        return value
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping when provided")
    funding_source = coerce_smr_funding_source(
        value.get("funding_source"),
        field_name=f"{field_name}.funding_source",
    )
    access = coerce_smr_run_policy_access(
        value.get("access") if isinstance(value.get("access"), Mapping) else value.get("access"),
        field_name=f"{field_name}.access",
    )
    limits = coerce_smr_run_policy_limits(
        value.get("limits") if isinstance(value.get("limits"), Mapping) else value.get("limits"),
        field_name=f"{field_name}.limits",
    )
    return SmrRunPolicy(
        funding_source=funding_source,
        access=access,
        limits=limits,
    )


__all__ = [
    "SmrRunPolicy",
    "SmrRunPolicyAccess",
    "SmrRunPolicyLimits",
    "coerce_smr_run_policy",
    "coerce_smr_run_policy_access",
    "coerce_smr_run_policy_limits",
]
