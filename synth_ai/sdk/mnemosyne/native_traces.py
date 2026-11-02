"""Project-retained native Rhodes traces through negotiated core.v1 reads.

See backend notes/specifications/mnemosyne-evaluation-evidence.md. Bytes are
accepted only when scoped metadata, owner headers, size and SHA-256 agree.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from typing import Annotated, Literal

import httpx
from pydantic import AwareDatetime, Field, StrictInt, model_validator

from synth_ai.contracts import mnemosyne_native as operations
from synth_ai.contracts.operations import AUTH_CONTEXT
from synth_ai.core.http.cleanup import handle_cleanup_failure
from synth_ai.core.http.response import classify_exchange
from synth_ai.core.parameters import verified
from synth_ai.errors import ContractError, RequestTimeoutError, TransportError
from synth_ai.sdk.core.repo_archive import error_response
from synth_ai.sdk.forge.contracts.contracts import Digest, Scope
from synth_ai.sdk.forge.records import project_identity
from synth_ai.types.auth import AuthContext
from synth_ai.types.base import Model, ResponseModel

MAX_NATIVE_TRACE_BYTES = 8 * 1024 * 1024
_ID = re.compile(r"^mna_[0-9a-f]{64}$")


class NativeTraceItem(Model):
    association_id: str = Field(pattern=r"^mna_[0-9a-f]{64}$")
    source_kind: Literal["swarm_aa1", "configured_customer", "swarm_program"]
    rollout_id: str = Field(min_length=1, max_length=128)
    execution_epoch: str = Field(min_length=1, max_length=128)
    run_id: str | None = Field(
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    )
    created_at: AwareDatetime
    archive_sha256: Digest
    archive_size_bytes: Annotated[StrictInt, Field(gt=0, le=MAX_NATIVE_TRACE_BYTES)]

    @model_validator(mode="after")
    def actual_source(self):
        if self.source_kind in {"swarm_aa1", "swarm_program"} and self.run_id is None:
            raise ValueError("Swarm source requires its original run")
        return self


class NativeTraceRecord(NativeTraceItem):
    schema_version: Literal["synth.mnemosyne.native-trace.v1"]
    scope: Scope


class NativeTracePageData(Model):
    schema_version: Literal["synth.mnemosyne.native-trace-page.v1"]
    scope: Scope
    items: tuple[NativeTraceItem, ...] = Field(max_length=100)
    next_cursor: str | None = Field(pattern=r"^mna_[0-9a-f]{64}$")


class NativeTraceResponse(ResponseModel):
    data: NativeTraceRecord


class NativeTracePage(ResponseModel):
    data: NativeTracePageData


def native_identifier(value):
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ValueError("association_id must identify a native Mnemosyne trace")
    return value


def _query(after, limit):
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    return {
        "limit": limit,
        **({"after": native_identifier(after)} if after is not None else {}),
    }


def _scope(scope, project_id, organization_id):
    return scope.project_id == project_id and scope.organization_id == organization_id


def _page(result, project_id, organization_id, after, limit):
    data = result.data
    ids = [item.association_id for item in data.items]
    return verified(
        result,
        _scope(data.scope, project_id, organization_id)
        and len(ids) <= limit
        and ids == sorted(set(ids))
        and (after is None or all(item > after for item in ids))
        and (data.next_cursor is None or bool(ids) and data.next_cursor == ids[-1]),
        "Mnemosyne returned a different native trace page",
    )


def _record(result, project_id, organization_id, association_id):
    return verified(
        result,
        _scope(result.data.scope, project_id, organization_id)
        and result.data.association_id == association_id,
        "Mnemosyne returned a different native trace",
    )


def _headers(response, item):
    exchange = classify_exchange(
        response,
        None,
        contract="core.v1",
        mutation=False,
        identity=None,
        attempt=1,
        attempts=1,
        operation=operations.CONTENT.id,
    )
    if exchange.error is not None:
        raise exchange.error
    if response.headers.get("Content-Encoding", "identity").lower() != "identity":
        raise ContractError("Compressed trace delivery is unsupported")
    if response.status_code != 200:
        return
    expected = {
        "X-Synth-Contract": "core.v1",
        "X-Content-SHA256": item.archive_sha256,
        "X-Mnemosyne-Organization-Id": item.scope.organization_id,
        "X-Mnemosyne-Project-Id": item.scope.project_id,
        "X-Mnemosyne-Association-Id": item.association_id,
    }
    if any(response.headers.get(k) != v for k, v in expected.items()):
        raise ContractError("Native trace delivery changed its scoped identity")
    if (
        response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        != "application/zip"
    ):
        raise ContractError("Native trace delivery has an unexpected media type")
    length = response.headers.get("Content-Length")
    if length is not None and (
        re.fullmatch(r"[0-9]{1,20}", length) is None
        or int(length) != item.archive_size_bytes
    ):
        raise ContractError("Native trace delivery changed its size")


def _check_bytes(data, item):
    if (
        len(data) != item.archive_size_bytes
        or hashlib.sha256(data).hexdigest() != item.archive_sha256
    ):
        raise ContractError("Native trace bytes differ from retained metadata")
    return bytes(data)


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RequestTimeoutError("Native trace download timed out")
    return remaining


def _deadline(transport, request):
    timeout = getattr(
        request, "timeout_seconds", None
    ) or transport._options.request_timeout(operations.CONTENT.id)
    return time.monotonic() + timeout


class NativeTraces:
    def __init__(self, transport):
        self._transport = transport

    def _organization(self, request):
        return self._transport.request(
            AUTH_CONTEXT, AuthContext, options=request
        ).tenant_id

    def list(self, project_id, *, after=None, limit=50, request=None):
        project_id = project_identity(project_id)
        query = _query(after, limit)
        organization = self._organization(request)
        return self._transport.request(
            operations.LIST,
            NativeTracePage,
            path=operations.LIST.path_for(project_id=project_id),
            query=query,
            options=request,
            validate=lambda r: _page(r, project_id, organization, after, limit),
        )

    def get(self, project_id, association_id, *, request=None):
        project_id, association_id = (
            project_identity(project_id),
            native_identifier(association_id),
        )
        organization = self._organization(request)
        return self._transport.request(
            operations.GET,
            NativeTraceResponse,
            path=operations.GET.path_for(
                project_id=project_id, identifier=association_id
            ),
            options=request,
            validate=lambda r: _record(r, project_id, organization, association_id),
        )

    def download(self, project_id, association_id, *, request=None):
        deadline = _deadline(self._transport, request)
        item = self.get(project_id, association_id, request=request).data
        response = None
        primary = None
        try:
            response = self._transport._open_stream(
                operations.CONTENT,
                path=operations.CONTENT.path_for(
                    project_id=item.scope.project_id, identifier=item.association_id
                ),
                query={},
                remaining_seconds=_remaining(deadline),
                digest=None,
                accept="application/zip",
            )
            _headers(response, item)
            data = bytearray()
            bound = item.archive_size_bytes if response.status_code == 200 else 65536
            for chunk in response.iter_raw(65536):
                _remaining(deadline)
                if len(data) + len(chunk) > bound:
                    raise ContractError("Native trace response exceeded its byte bound")
                data.extend(chunk)
            if response.status_code != 200:
                error_response(response, bytes(data), operations.CONTENT)
            return _check_bytes(data, item)
        except httpx.TimeoutException as cause:
            primary = RequestTimeoutError("Native trace download timed out")
            raise primary from cause
        except httpx.TransportError as cause:
            primary = TransportError("Native trace download failed")
            raise primary from cause
        except BaseException as error:
            primary = error
            raise
        finally:
            if response is not None:
                try:
                    self._transport._release_response(response)
                except Exception as cause:
                    handle_cleanup_failure(
                        cause, primary, mutation=False, identity=None
                    )


class AsyncNativeTraces:
    def __init__(self, transport):
        self._transport = transport

    async def _organization(self, request):
        return (
            await self._transport.request(AUTH_CONTEXT, AuthContext, options=request)
        ).tenant_id

    async def list(self, project_id, *, after=None, limit=50, request=None):
        project_id = project_identity(project_id)
        query = _query(after, limit)
        organization = await self._organization(request)
        return await self._transport.request(
            operations.LIST,
            NativeTracePage,
            path=operations.LIST.path_for(project_id=project_id),
            query=query,
            options=request,
            validate=lambda r: _page(r, project_id, organization, after, limit),
        )

    async def get(self, project_id, association_id, *, request=None):
        project_id, association_id = (
            project_identity(project_id),
            native_identifier(association_id),
        )
        organization = await self._organization(request)
        return await self._transport.request(
            operations.GET,
            NativeTraceResponse,
            path=operations.GET.path_for(
                project_id=project_id, identifier=association_id
            ),
            options=request,
            validate=lambda r: _record(r, project_id, organization, association_id),
        )

    async def download(self, project_id, association_id, *, request=None):
        deadline = _deadline(self._transport, request)
        item = (await self.get(project_id, association_id, request=request)).data
        response = None
        primary = None
        try:
            async with asyncio.timeout(_remaining(deadline)):
                response = await self._transport._open_stream(
                    operations.CONTENT,
                    path=operations.CONTENT.path_for(
                        project_id=item.scope.project_id, identifier=item.association_id
                    ),
                    query={},
                    remaining_seconds=_remaining(deadline),
                    digest=None,
                    accept="application/zip",
                )
                _headers(response, item)
                data = bytearray()
                bound = (
                    item.archive_size_bytes if response.status_code == 200 else 65536
                )
                async for chunk in response.aiter_raw(65536):
                    _remaining(deadline)
                    if len(data) + len(chunk) > bound:
                        raise ContractError(
                            "Native trace response exceeded its byte bound"
                        )
                    data.extend(chunk)
                if response.status_code != 200:
                    error_response(response, bytes(data), operations.CONTENT)
                return _check_bytes(data, item)
        except (httpx.TimeoutException, TimeoutError) as cause:
            primary = RequestTimeoutError("Native trace download timed out")
            raise primary from cause
        except httpx.TransportError as cause:
            primary = TransportError("Native trace download failed")
            raise primary from cause
        except BaseException as error:
            primary = error
            raise
        finally:
            if response is not None:
                try:
                    await self._transport._release_response(response)
                except (Exception, asyncio.CancelledError) as cause:
                    handle_cleanup_failure(
                        cause, primary, mutation=False, identity=None
                    )
