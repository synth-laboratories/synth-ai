"""Credential-safe request construction. See SDK index_contract.md."""

from __future__ import annotations

from hashlib import sha256
from typing import Mapping
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel

from synth_ai.contracts.operations import PREFIX, Operation
from synth_ai.errors import AuthenticationError
from synth_ai.options import MutationOptions, ReadOptions
from synth_ai.types.capabilities import NegotiatedCapability
from synth_ai.types.identity import MutationIdentity

CREDENTIAL_HEADERS = frozenset(
    {
        "authorization",
        "cookie",
        "x-api-key",
        "x_api_key",
        "x-user-api-key",
        "x-clerk-session-token",
        "x-search-token",
        "proxy-authorization",
    }
)


def validate_base_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
    ):
        raise ValueError(
            "base_url must be an absolute origin without credentials, path, query or fragment"
        )
    if parsed.scheme != "https" and not (
        parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
    ):
        raise ValueError("base_url requires HTTPS except for loopback development")
    return value.rstrip("/")


def encode_body(body: BaseModel | None) -> bytes:
    return b"" if body is None else body.model_dump_json(exclude_none=False).encode("utf-8")


def mutation_identity(
    operation: Operation, path: str | None, body: bytes, options: ReadOptions | None
) -> MutationIdentity | None:
    if not operation.mutation:
        return None
    if options is not None and not isinstance(options, MutationOptions):
        raise ValueError("Mutations require MutationOptions")
    if not operation.keyed:
        if isinstance(options, MutationOptions) and (
            options.identity is not None or options.idempotency_key is not None
        ):
            raise ValueError("Anonymous mutations have no durable keyed identity")
        return None
    identity = options.identity if isinstance(options, MutationOptions) else None
    if identity is None:
        return MutationIdentity.create(
            operation.id,
            body,
            PREFIX + (path or operation.path),
            key=options.idempotency_key if isinstance(options, MutationOptions) else None,
        )
    if (
        identity.operation != operation.id
        or identity.resource_path != PREFIX + (path or operation.path)
        or identity.payload_sha256 != sha256(body).hexdigest()
    ):
        raise ValueError("mutation identity must match the operation and exact serialized intent")
    return identity


def build_request(
    client: httpx.Client | httpx.AsyncClient,
    base_url: str,
    api_key: str | None,
    operation: Operation,
    path: str | None,
    body: bytes,
    query: Mapping[str, str | int] | None,
    options: ReadOptions | None,
    identity: MutationIdentity | None,
    remaining_seconds: float,
    connect_seconds: float,
    negotiated: NegotiatedCapability | None = None,
) -> httpx.Request:
    if operation.keyed and api_key is None:
        raise AuthenticationError("This operation requires an API key")
    selected_path = path if path is not None else operation.path
    if not operation.matches_resource_path(PREFIX + selected_path):
        raise ValueError("request path does not match the selected operation")
    request = client.build_request(
        operation.method,
        base_url + PREFIX + selected_path,
        params=query,
        content=body if body else None,
        timeout=httpx.Timeout(remaining_seconds, connect=min(remaining_seconds, connect_seconds)),
    )
    for header in CREDENTIAL_HEADERS:
        request.headers.pop(header, None)
    for header in (
        "X-Synth-Schema-SHA256",
        "X-Synth-Capabilities-SHA256",
        "X-Synth-Release-Manifest-SHA256",
    ):
        request.headers.pop(header, None)
    if negotiated is not None:
        request.headers["X-Synth-Schema-SHA256"] = negotiated.schema_sha256
        request.headers["X-Synth-Capabilities-SHA256"] = negotiated.capability_sha256
        request.headers["X-Synth-Release-Manifest-SHA256"] = negotiated.release_manifest_sha256
    request.headers["X-Synth-Contract"] = operation.contract
    request.headers["Accept-Encoding"] = "identity"
    request.headers["X-Synth-Request-Timeout-Ms"] = str(max(1, int(remaining_seconds * 1000)))
    if body:
        request.headers["Content-Type"] = "application/json"
    if api_key is not None and operation.id != "capabilities.get":
        request.headers["Authorization"] = "Bearer " + api_key
    if options is not None and options.request_id is not None:
        request.headers["X-Request-Id"] = options.request_id
    if identity is not None:
        request.headers["Idempotency-Key"] = identity.key
        request.headers["X-Synth-Intent-SHA256"] = identity.payload_sha256
    return request
