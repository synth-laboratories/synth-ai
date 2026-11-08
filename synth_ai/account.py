"""Account and session-authorized API key operations."""

from __future__ import annotations

import builtins

from ._generated.models import (
    Account,
    ApiKey,
    ApiKeyPage,
    LimitList,
    MintApiKeyRequest,
    MintApiKeyRequestPurpose,
    MintedApiKey,
    Scope,
    UpdateApiKeyRequest,
    UsageExport,
    UsageRecordPage,
    UsageSummary,
)
from ._transport import Transport


class AsyncApiKeys:
    def __init__(self, transport: Transport) -> None:
        self._transport = transport

    async def list(
        self, *, purpose: str | None = None, cursor: str | None = None, limit: int = 20
    ) -> ApiKeyPage:
        return await self._transport.request(
            "listApiKeys", ApiKeyPage, query={"purpose": purpose, "cursor": cursor, "limit": limit}
        )

    async def mint(
        self,
        *,
        purpose: MintApiKeyRequestPurpose | None = None,
        name: str | None = None,
        scopes: builtins.list[Scope] | None = None,
        expires_at: str | None = None,
        request_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> MintedApiKey:
        body = MintApiKeyRequest(
            purpose=purpose, name=name, scopes=scopes, expires_at=expires_at, request_id=request_id
        )
        body = MintApiKeyRequest.model_validate(body.model_dump(exclude_none=True))
        return await self._transport.request(
            "createApiKey", MintedApiKey, body=body, idempotency_key=idempotency_key
        )

    async def update_grants(
        self,
        key_id: str,
        *,
        scopes: builtins.list[Scope],
        expected_version: int,
        name: str | None = None,
    ) -> ApiKey:
        body = UpdateApiKeyRequest(scopes=scopes, expected_version=expected_version, name=name)
        body = UpdateApiKeyRequest.model_validate(body.model_dump(exclude_none=True))
        return await self._transport.request(
            "updateApiKey", ApiKey, path={"key_id": key_id}, body=body
        )

    async def rotate(self, key_id: str, *, idempotency_key: str | None = None) -> MintedApiKey:
        return await self._transport.request(
            "rotateApiKey", MintedApiKey, path={"key_id": key_id}, idempotency_key=idempotency_key
        )

    async def revoke(self, key_id: str) -> None:
        await self._transport.invoke("revokeApiKey", path={"key_id": key_id})


class AsyncAccount:
    def __init__(self, transport: Transport) -> None:
        self._transport = transport
        self.api_keys = AsyncApiKeys(transport)

    async def get(self) -> Account:
        return await self._transport.request("getAccount", Account)

    async def usage_summary(
        self, *, since: str | None = None, until: str | None = None
    ) -> UsageSummary:
        return await self._transport.request(
            "getUsageSummary", UsageSummary, query={"since": since, "until": until}
        )

    async def usage_records(
        self,
        *,
        since: str | None = None,
        until: str | None = None,
        run_id: str | None = None,
        cursor: str | None = None,
        limit: int = 20,
    ) -> UsageRecordPage:
        return await self._transport.request(
            "listUsageRecords",
            UsageRecordPage,
            query={
                "since": since,
                "until": until,
                "run_id": run_id,
                "cursor": cursor,
                "limit": limit,
            },
        )

    async def export_usage(
        self, *, since: str, until: str, idempotency_key: str | None = None
    ) -> UsageExport:
        return await self._transport.request(
            "createUsageExport",
            UsageExport,
            body={"since": since, "until": until, "format": "csv"},
            idempotency_key=idempotency_key,
        )

    async def usage_export(self, export_id: str) -> UsageExport:
        return await self._transport.request(
            "getUsageExport", UsageExport, path={"export_id": export_id}
        )

    async def limits(self) -> LimitList:
        return await self._transport.request("listLimits", LimitList)
