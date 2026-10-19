"""Bounded request policy. See the public contract snapshots in contracts/"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Self

import httpx
from pydantic import Field, InstanceOf, StrictInt, TypeAdapter, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema

from synth_ai.types.base import Digest, Identifier, Model
from synth_ai.types.identity import MutationIdentity
from synth_ai.types.rate_limits import ResponseMetadata


@lru_cache(maxsize=1)
def _warn_retry_delay_alias() -> None:
    warnings.warn(
        "retry_delay_seconds is deprecated; use retry_base_delay_seconds",
        DeprecationWarning,
        stacklevel=3,
    )


@dataclass(frozen=True, slots=True)
class ClientOptions:
    release_manifest_sha256: str | None = None
    candidate_manifest_sha256: str | None = None
    timeout_seconds: float = 30.0
    connect_timeout_seconds: float = 10.0
    maximum_response_bytes: int = 1024 * 1024
    maximum_connections: int = 20
    read_retry_count: int = 2
    retry_delay_seconds: float | None = None
    retry_base_delay_seconds: float = 0.2
    retry_max_delay_seconds: float = 10.0
    upload_hosts: tuple[str, ...] = ()
    http_client: httpx.Client | None = None
    async_http_client: httpx.AsyncClient | None = None
    # Reviewed Index egress includes Monitor review; other reads stay at 30s.
    index_result_timeout_seconds: float | None = None
    # Explicitly accept the local server debug archive; never relax hosted ZDR.
    local_development_retention: bool = False

    def request_timeout(self, operation_id: str) -> float:
        if operation_id != "index.searches.result":
            return self.timeout_seconds
        if self.index_result_timeout_seconds is not None:
            return self.index_result_timeout_seconds
        # Preserve existing explicit global deadline overrides. Only the
        # ordinary 30s default gains the reviewed-result allowance.
        return 75.0 if self.timeout_seconds == 30.0 else self.timeout_seconds

    def __post_init__(self) -> None:
        if self.http_client is not None and not isinstance(self.http_client, httpx.Client):
            raise ValueError("http_client must be an httpx.Client")
        if self.async_http_client is not None and not isinstance(
            self.async_http_client, httpx.AsyncClient
        ):
            raise ValueError("async_http_client must be an httpx.AsyncClient")
        if type(self.local_development_retention) is not bool:
            raise ValueError("local_development_retention must be a boolean")
        if self.release_manifest_sha256 is not None:
            TypeAdapter(Digest).validate_python(self.release_manifest_sha256)
        if self.candidate_manifest_sha256 is not None:
            TypeAdapter(Digest).validate_python(self.candidate_manifest_sha256)
        if self.release_manifest_sha256 is not None and self.candidate_manifest_sha256 is not None:
            raise ValueError(
                "Select either a qualified release pin or a nonproduction candidate pin"
            )
        if len(self.upload_hosts) > 32 or any(
            not host or host != host.lower() or any(character in host for character in "/:@ \n\r")
            for host in self.upload_hosts
        ):
            raise ValueError("upload_hosts must contain at most 32 plain lowercase hostnames")
        if self.retry_delay_seconds is not None:
            _bounded_number("retry_delay_seconds", self.retry_delay_seconds, 0, 60)
            if self.retry_base_delay_seconds not in (0.2, self.retry_delay_seconds):
                raise ValueError("Select retry_base_delay_seconds or retry_delay_seconds, not both")
            _warn_retry_delay_alias()
            object.__setattr__(self, "retry_base_delay_seconds", self.retry_delay_seconds)
        for name in ("timeout_seconds", "connect_timeout_seconds", "index_result_timeout_seconds"):
            value = getattr(self, name)
            if value is not None or name != "index_result_timeout_seconds":
                _bounded_number(name, value, 0, 86400, exclusive_minimum=True)
        _bounded_number("retry_base_delay_seconds", self.retry_base_delay_seconds, 0, 60)
        _bounded_number("retry_max_delay_seconds", self.retry_max_delay_seconds, 0, 300)
        for name, minimum, maximum in (
            ("maximum_response_bytes", 1, 32 * 1024 * 1024),
            ("maximum_connections", 1, 1000),
            ("read_retry_count", 0, 10),
        ):
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")


def _bounded_number(
    name: str, value: object, minimum: float, maximum: float, *, exclusive_minimum: bool = False
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not minimum <= value <= maximum
        or not math.isfinite(value)
        or (exclusive_minimum and value == minimum)
    ):
        boundary = "greater than" if exclusive_minimum else "at least"
        raise ValueError(f"{name} must be finite, {boundary} {minimum}, and at most {maximum}")


class ReadOptions(Model):
    timeout_seconds: float | None = Field(default=None, gt=0, le=86400)
    request_id: Identifier | None = None
    read_retry_count: StrictInt | None = Field(default=None, ge=0, le=10)
    response_metadata: SkipJsonSchema[InstanceOf[ResponseMetadata] | None] = Field(
        default=None, exclude=True, repr=False
    )


class MutationOptions(ReadOptions):
    identity: MutationIdentity | None = None
    idempotency_key: Identifier | None = None

    @model_validator(mode="after")
    def validate_identity_selection(self) -> Self:
        if self.identity is not None and self.idempotency_key is not None:
            raise ValueError("select an exact mutation identity or an idempotency key, not both")
        return self


class ObservationOptions(Model):
    timeout_seconds: float = Field(gt=0, le=86400)
    idle_timeout_seconds: float = Field(gt=0, le=86400)
    reconnect_deadline_seconds: float = Field(gt=0, le=3600)
    retry_count_max: StrictInt = Field(ge=0, le=10)
    maximum_events: StrictInt = Field(ge=1, le=10000)
    poll_interval_seconds: float = Field(ge=0.01, le=60)


class TransferOptions(Model):
    timeout_seconds: float = Field(gt=0, le=86400)
    maximum_bytes: StrictInt = Field(ge=1, le=32 * 1024 * 1024)
    chunk_bytes: StrictInt = Field(ge=1024, le=1024 * 1024)
    retry_count_max: Literal[0]

    @field_validator("retry_count_max", mode="before")
    @classmethod
    def validate_transfer_retry(cls, value: object) -> object:
        if type(value) is not int or value != 0:
            raise ValueError(
                "automatic transfer retry is not admitted; reconcile status explicitly"
            )
        return value
