"""Bounded request policy. See backend/notes/specifications/sdk/index_contract.md."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Self

import httpx
from pydantic import Field, InstanceOf, StrictInt, TypeAdapter, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema

from synth_ai.types.base import Digest, Identifier, Model
from synth_ai.types.identity import MutationIdentity
from synth_ai.types.rate_limits import ResponseMetadata


@dataclass(frozen=True, slots=True)
class ClientOptions:
    release_manifest_sha256: str | None = None
    candidate_manifest_sha256: str | None = None
    timeout_seconds: float = 30.0
    connect_timeout_seconds: float = 10.0
    maximum_response_bytes: int = 1024 * 1024
    maximum_connections: int = 20
    read_retry_count: int = 2
    retry_delay_seconds: float = 0.1
    upload_hosts: tuple[str, ...] = ()
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
        for value in (
            self.timeout_seconds,
            self.connect_timeout_seconds,
            *(
                ()
                if self.index_result_timeout_seconds is None
                else (self.index_result_timeout_seconds,)
            ),
        ):
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 86400:
                raise ValueError("timeouts must be finite and positive")
        if (
            isinstance(self.retry_delay_seconds, bool)
            or not math.isfinite(self.retry_delay_seconds)
            or not 0 <= self.retry_delay_seconds <= 60
        ):
            raise ValueError("retry delay must be finite and nonnegative")
        for value, minimum, maximum in (
            (self.maximum_response_bytes, 1, 32 * 1024 * 1024),
            (self.maximum_connections, 1, 1000),
            (self.read_retry_count, 0, 10),
        ):
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError("connection, retry and byte limits are outside safety bounds")


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
