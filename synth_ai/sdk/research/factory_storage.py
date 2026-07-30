"""Read-only Factory storage-authority consumers."""

from __future__ import annotations

from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.request import HttpRequest
from synth_ai.core.http.transport import HttpTransport
from synth_ai.sdk.research.contracts.common import FactoryId
from synth_ai.sdk.research.contracts.project_runtime import (
    FactoryStorageAuthorityResponse,
)
from synth_ai.sdk.research.operations import research_operation


def _request(factory_id: FactoryId) -> HttpRequest:
    return HttpRequest(
        research_operation("get_factory_storage_authority"),
        f"/smr/factories/{factory_id}/storage-authority",
    )


class FactoryStorageAPI:
    """Opaque, generation-bound Postgres/S3/Turso authority descriptors."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def retrieve(self, factory_id: FactoryId) -> FactoryStorageAuthorityResponse:
        """Retrieve the descriptor and its owner-authored lifecycle receipt."""
        response = FactoryStorageAuthorityResponse.from_wire(
            self._transport.execute(_request(factory_id))
        )
        if response.descriptor.factory_id != str(factory_id):
            raise ValueError("Factory storage authority crossed its Factory boundary")
        return response


class AsyncFactoryStorageAPI:
    """Native asynchronous peer of :class:`FactoryStorageAPI`."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def retrieve(
        self,
        factory_id: FactoryId,
    ) -> FactoryStorageAuthorityResponse:
        """Retrieve the descriptor and its owner-authored lifecycle receipt."""
        response = FactoryStorageAuthorityResponse.from_wire(
            await self._transport.execute(_request(factory_id))
        )
        if response.descriptor.factory_id != str(factory_id):
            raise ValueError("Factory storage authority crossed its Factory boundary")
        return response


__all__ = ["AsyncFactoryStorageAPI", "FactoryStorageAPI"]
