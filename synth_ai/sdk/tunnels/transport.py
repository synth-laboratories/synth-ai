"""Adapt typed control routes to the SDK-owned transport; see backend v2_catalog.md."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import ConfigDict, JsonValue, RootModel

from synth_ai.contracts.operations import PREFIX
from synth_ai.contracts.tunnels import CONTROL_OPERATIONS, RESPONSE_MODELS
from synth_ai.core.http.async_client import AsyncHttpTransport
from synth_ai.core.http.client import HttpTransport
from synth_ai.errors import ContractError
from synth_ai.options import MutationOptions, ReadOptions
from synth_ai.types.tunnels.contracts import CatalogMutation, ConnectorStatus


class Payload(RootModel[dict[str, JsonValue]]):
    model_config = ConfigDict(allow_inf_nan=False, hide_input_in_errors=True)


def prepare(method: str, path: str, json_body: dict | None, operation_id: str | None):
    operation = next(
        (
            value
            for value in CONTROL_OPERATIONS
            if value.method == method and value.matches_resource_path(path)
        ),
        None,
    )
    if operation is None:
        raise ValueError("Unknown canonical Tunnel v2 route")
    for part in path.removeprefix(PREFIX + "/synthtunnel/v2/connectors").split("/"):
        if len(part.split(":", 1)[0]) == 36 and UUID(part.split(":", 1)[0]).int == 0:
            raise ValueError("nil Tunnel resource identity")
    if (
        json_body is not None
        and "connector_id" in json_body
        and UUID(str(json_body["connector_id"])).int == 0
    ):
        raise ValueError("nil Tunnel connector identity")
    # Catalog deadlines are immutable effect policy, not local observation
    # deadlines: an expired command may already have a durable receipt.
    options = MutationOptions(idempotency_key=operation_id) if operation.mutation else ReadOptions()
    return operation, Payload(json_body) if json_body is not None else None, options


def verified(result: Any, body: dict | None, path: str):
    if isinstance(result, CatalogMutation):
        expected = body.get("command_id") if body else path.rsplit("/", 1)[-1]
        if expected is not None and str(result.command) != expected:
            raise ContractError("Tunnel command receipt identity mismatch")
    if isinstance(result, (CatalogMutation, ConnectorStatus)):
        connector = path.removeprefix(PREFIX + "/synthtunnel/v2/connectors/").split("/")[0]
        expected = body.get("connector_id") if body and "connector_id" in body else connector
        if str(result.connector) != expected:
            raise ContractError("Tunnel connector receipt identity mismatch")
    return result.root if isinstance(result, RootModel) else result


class TunnelTransport:
    def __init__(self, transport: HttpTransport) -> None:
        self.transport = transport

    def close(self) -> None:
        self.transport.close()

    def request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: dict | None = None,
        operation_id: str | None = None,
    ) -> Any:
        operation, body, options = prepare(method, path, json_body, operation_id)
        return self.transport.request(
            operation,
            RESPONSE_MODELS[operation.id],
            body=body,
            path=path.removeprefix(PREFIX),
            options=options,
            validate=lambda result: verified(result, json_body, path),
        )


class AsyncTunnelTransport:
    def __init__(self, transport: AsyncHttpTransport) -> None:
        self.transport = transport

    async def close(self) -> None:
        await self.transport.aclose()

    async def request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: dict | None = None,
        operation_id: str | None = None,
    ) -> Any:
        operation, body, options = prepare(method, path, json_body, operation_id)
        return await self.transport.request(
            operation,
            RESPONSE_MODELS[operation.id],
            body=body,
            path=path.removeprefix(PREFIX),
            options=options,
            validate=lambda result: verified(result, json_body, path),
        )
