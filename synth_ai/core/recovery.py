"""Shared scoped creation recovery without repeating a mutation."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from synth_ai.contracts.operations import Operation
from synth_ai.contracts.registry import recovery_operation
from synth_ai.core.http.async_client import AsyncHttpTransport
from synth_ai.core.http.client import HttpTransport
from synth_ai.core.parameters import verified
from synth_ai.errors import ContractError
from synth_ai.options import ReadOptions
from synth_ai.types.identity import MutationIdentity
from synth_ai.types.receipts import CreationResolution, MutationResolution, ResourceReference

T = TypeVar("T")


def validate_creation(identity: MutationIdentity, operation: Operation) -> None:
    if identity.operation != operation.id:
        raise ValueError("Creation resolver requires the matching operation identity")
    recovery_operation(identity)


def validate_resource(result: MutationResolution, kind: str) -> ResourceReference | None:
    if result.resource is not None and result.resource.kind != kind:
        raise ContractError("Creation resolver returned a different resource kind")
    return result.resource


def resolve_creation(
    transport: HttpTransport,
    identity: MutationIdentity,
    operation: Operation,
    kind: str,
    attach: Callable[[str], T],
    request: ReadOptions | None,
) -> CreationResolution[T]:
    validate_creation(identity, operation)
    result = transport.request(
        recovery_operation(identity),
        MutationResolution,
        query={
            "key": identity.key,
            "operation": identity.operation,
            "resource_path": identity.resource_path,
            "payload_sha256": identity.payload_sha256,
        },
        options=request,
        validate=lambda result: verified(
            result, result.identity == identity, "Resolver identity mismatch"
        ),
    )
    reference = validate_resource(result, kind)
    return CreationResolution(
        result.state, attach(reference.resource_id) if reference else None, result.metadata
    )


async def async_resolve_creation(
    transport: AsyncHttpTransport,
    identity: MutationIdentity,
    operation: Operation,
    kind: str,
    attach: Callable[[str], Awaitable[T]],
    request: ReadOptions | None,
) -> CreationResolution[T]:
    validate_creation(identity, operation)
    result = await transport.request(
        recovery_operation(identity),
        MutationResolution,
        query={
            "key": identity.key,
            "operation": identity.operation,
            "resource_path": identity.resource_path,
            "payload_sha256": identity.payload_sha256,
        },
        options=request,
        validate=lambda result: verified(
            result, result.identity == identity, "Resolver identity mismatch"
        ),
    )
    reference = validate_resource(result, kind)
    return CreationResolution(
        result.state, await attach(reference.resource_id) if reference else None, result.metadata
    )
