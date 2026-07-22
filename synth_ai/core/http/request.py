"""Typed HTTP operation metadata.

# See: specifications/sdk/core_research_migration.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping

from synth_ai.core.contracts.json_value import JsonObject, JsonValue


class HttpMethod(StrEnum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


class OperationId(str):
    """Stable language-neutral backend operation identifier."""


@dataclass(frozen=True, slots=True)
class OperationMetadata:
    operation_id: OperationId
    method: HttpMethod
    path_template: str
    mutation: bool = False
    idempotent: bool = False


@dataclass(frozen=True, slots=True)
class HttpRequest:
    operation: OperationMetadata
    path: str
    query: Mapping[str, JsonValue] = field(default_factory=dict)
    body: JsonObject | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    timeout_seconds: float | None = None


__all__ = ["HttpMethod", "HttpRequest", "OperationId", "OperationMetadata"]
