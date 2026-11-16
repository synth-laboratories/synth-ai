"""Account and session-authorized API key operations."""

from __future__ import annotations

import builtins
from collections.abc import AsyncGenerator, Sequence

from pydantic import ValidationError

from ._generated.models import (
    Account,
    ApiKey,
    ApiKeyPage,
    LimitList,
    MintApiKeyRequest,
    MintApiKeyRequestPurpose,
    MintedApiKey,
    PlanRequest,
    PlanRequestList,
    Scope,
    SpecialFreeRequestInput,
    TermsAcceptance,
    TermsAcceptanceRequest,
    UpdateApiKeyRequest,
    UsageExport,
    UsageRecord,
    UsageRecordPage,
    UsageSummary,
)
from ._pagination import paginate
from ._transport import Transport


class AsyncApiKeys:
    def __init__(self, transport: Transport) -> None:
        self._transport = transport

    async def list(
        self, *, purpose: str | None = None, cursor: str | None = None, limit: int = 20
    ) -> ApiKeyPage:
        """List the caller's personal and automation keys (session only)."""
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
        """Mint a key. The secret is returned once (session only)."""
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
        """Replace a key's scopes (optimistic on authorization_version, session only)."""
        body = UpdateApiKeyRequest(scopes=scopes, expected_version=expected_version, name=name)
        body = UpdateApiKeyRequest.model_validate(body.model_dump(exclude_none=True))
        return await self._transport.request(
            "updateApiKey", ApiKey, path={"key_id": key_id}, body=body
        )

    async def rotate(self, key_id: str, *, idempotency_key: str | None = None) -> MintedApiKey:
        """Revoke a key and mint its replacement in the same rotation family (session only)."""
        return await self._transport.request(
            "rotateApiKey", MintedApiKey, path={"key_id": key_id}, idempotency_key=idempotency_key
        )

    async def revoke(self, key_id: str) -> None:
        """Revoke a key (idempotent, session only)."""
        await self._transport.invoke("revokeApiKey", path={"key_id": key_id})

    async def iter_keys(
        self, *, purpose: str | None = None, cursor: str | None = None, limit: int = 20
    ) -> AsyncGenerator[ApiKey, None]:
        """Yield API key metadata across cursor pages."""

        async def fetch(next_cursor: str | None) -> ApiKeyPage:
            return await self.list(purpose=purpose, cursor=next_cursor, limit=limit)

        async for item in paginate(fetch, cursor=cursor):
            yield item


class AsyncAccount:
    def __init__(self, transport: Transport) -> None:
        self._transport = transport
        self.api_keys = AsyncApiKeys(transport)

    async def get(self) -> Account:
        """The caller's org, principal and limits."""
        return await self._transport.request("getAccount", Account)

    async def accept_terms(
        self, *, version: str, accepted_at: str, idempotency_key: str | None = None
    ) -> TermsAcceptance:
        """Record the caller's explicit acceptance of a terms version and claimed UTC time."""
        return await self._transport.request(
            "acceptTerms",
            TermsAcceptance,
            body=TermsAcceptanceRequest(version=version, accepted_at=accepted_at),
            idempotency_key=idempotency_key,
        )

    async def usage_summary(
        self, *, since: str | None = None, until: str | None = None
    ) -> UsageSummary:
        """A projection of settled `billing.charge_usage` rows, not a separate meter: `total` equals the sum of
        `GET /v1/usage/records` over the same window. `data_through` is the newest settled row in the window
        (null when there is none) and `observed_at` is when this answer was read. `pending` is the org's live
        unsettled holds at `observed_at`; it is not limited to the window. `by_source.sessions` is usage from
        sessions outside a run, and `by_source.unattributed` means attribution is unknown, not "not a run".
        Refusals: `invalid_filter` (422) on an unparsable time, `usage_window_invalid` (422) when `until` is not
        after `since` or the window exceeds 366 days."""
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
        """The same rows and window as `GET /v1/usage/summary`. Resend the same filters with `next_cursor`.
        `run_id` is null when the record has no run. Refusals: `invalid_filter`, `usage_window_invalid` and
        `invalid_cursor` (422)."""
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
        """Export settled usage records."""
        return await self._transport.request(
            "createUsageExport",
            UsageExport,
            body={"since": since, "until": until, "format": "csv"},
            idempotency_key=idempotency_key,
        )

    async def usage_export(self, export_id: str) -> UsageExport:
        """Read an export and its download link."""
        return await self._transport.request(
            "getUsageExport", UsageExport, path={"export_id": export_id}
        )

    async def limits(self) -> LimitList:
        """Plan limits, the concurrent Swarms run cap and the public API rate limit. `cap` is null when the limit is
        unbounded. `current` is null when core has no exact measure for that row. `reset_at` is null when core defines
        no reset for that row, such as concurrency."""
        return await self._transport.request("listLimits", LimitList)

    async def request_special_free(
        self, *, use_case: str, links: Sequence[str] = (), idempotency_key: str | None = None
    ) -> PlanRequest:
        """Ask for owner review to move this org to Special Free. `use_case` (at most 4096 UTF-8 bytes) and
        `links` (at most 5) are customer content: stored for review, never returned by the API and never logged by
        this SDK. Refusals: `plan_request_pending_exists` (409), `plan_request_not_needed` (409, already on
        Special Free or a paid plan), `plan_request_cooldown` (429 after a denial; never retried, the wait is
        `retry_after_seconds` on the error), `plan_request_too_large` (422)."""
        try:
            body = SpecialFreeRequestInput(use_case=use_case, links=list(links))
        except ValidationError:
            # Pydantic's message echoes the rejected input; the use case must not reach logs.
            raise ValueError(
                "use_case must be a non-empty string and links a sequence of strings"
            ) from None
        return await self._transport.request(
            "requestSpecialFree", PlanRequest, body=body, idempotency_key=idempotency_key
        )

    async def plan_requests(self) -> PlanRequestList:
        """This org's plan requests, newest first (at most 100), as status projections without the use case."""
        return await self._transport.request("listPlanRequests", PlanRequestList)

    async def withdraw_plan_request(self, request_id: str) -> PlanRequest:
        """Withdraw this org's pending plan request. Refusals: `plan_request_not_pending` (409),
        `plan_request_unknown` (404)."""
        return await self._transport.request(
            "withdrawPlanRequest", PlanRequest, path={"request_id": request_id}
        )

    async def iter_usage_records(
        self,
        *,
        since: str | None = None,
        until: str | None = None,
        run_id: str | None = None,
        cursor: str | None = None,
        limit: int = 20,
    ) -> AsyncGenerator[UsageRecord, None]:
        """Yield usage records across cursor pages, preserving the supplied filters."""

        async def fetch(next_cursor: str | None) -> UsageRecordPage:
            return await self.usage_records(
                since=since, until=until, run_id=run_id, cursor=next_cursor, limit=limit
            )

        async for item in paginate(fetch, cursor=cursor):
            yield item
