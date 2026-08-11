"""Managed-inference intent shared by Swarms, Factories, and pool tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from synth_ai.core.contracts.json_value import JsonObject, JsonValue


@dataclass(frozen=True, slots=True)
class ManagedInferenceLimits:
    max_tokens: int | None = None
    max_spend_usd: float | None = None
    max_concurrent_calls: int | None = None
    max_calls: int | None = None

    def __post_init__(self) -> None:
        for name in ("max_tokens", "max_concurrent_calls", "max_calls"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_spend_usd is not None and self.max_spend_usd <= 0:
            raise ValueError("max_spend_usd must be positive")

    def to_wire(self) -> JsonObject:
        return {
            name: value
            for name, value in (
                ("max_tokens", self.max_tokens),
                ("max_spend_usd", self.max_spend_usd),
                ("max_concurrent_calls", self.max_concurrent_calls),
                ("max_calls", self.max_calls),
            )
            if value is not None
        }


@dataclass(frozen=True, slots=True)
class ManagedInference:
    model: str
    wire_apis: tuple[str, ...] = ("chat_completions", "responses")
    limits: ManagedInferenceLimits = field(default_factory=ManagedInferenceLimits)
    credential_ref: str | None = None
    mode: str = "synth_managed"
    metadata: Mapping[str, JsonValue] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if self.mode != "synth_managed":
            raise ValueError("managed inference mode must be synth_managed")
        if not self.model.strip():
            raise ValueError("managed inference model must be non-empty")
        allowed = {"chat_completions", "responses"}
        normalized = tuple(dict.fromkeys(str(value).strip() for value in self.wire_apis))
        if not normalized or not set(normalized).issubset(allowed):
            raise ValueError("wire_apis must contain chat_completions and/or responses")
        object.__setattr__(self, "wire_apis", normalized)
        if self.credential_ref is not None and not self.credential_ref.strip():
            raise ValueError("credential_ref must be non-empty")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_wire(self) -> JsonObject:
        payload: JsonObject = {
            "mode": self.mode,
            "model": self.model,
            "wire_apis": list(self.wire_apis),
            "limits": self.limits.to_wire(),
        }
        if self.credential_ref is not None:
            payload["credential_ref"] = self.credential_ref
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload
