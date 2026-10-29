"""Menagerie export, archive/restore, lineage and serving identity (SYN-5183).

Thin facade over ``synth_ai.contracts.menagerie``: the backend owns every
decision (ownership, dependents, conflicts). Writes are server-idempotent, so
repeating a call after ``OutcomeUncertainError`` is safe. Hard byte deletion
is Artifact Platform retention, not a Menagerie operation.
"""

from __future__ import annotations

import re
from typing import TypeVar

from pydantic import BaseModel

from synth_ai.contracts import menagerie as ops
from synth_ai.contracts.operations import Operation
from synth_ai.core.http.async_client import AsyncHttpTransport
from synth_ai.core.http.client import HttpTransport
from synth_ai.errors import ContractError

from .types import (
    DEPLOYMENT_ID,
    Deployment,
    DeploymentWrite,
    LineageEdge,
    LineageView,
    LineageWrite,
    MenagerieExport,
    ModelArchive,
    ModelResolution,
    ModelRestore,
    RecordArchive,
    RecordRestore,
)

T = TypeVar("T", bound=BaseModel)
_Call = tuple[Operation, type[BaseModel], str, BaseModel | None]


def _deployment(deployment_id: str) -> str:
    if not re.fullmatch(DEPLOYMENT_ID, deployment_id):
        raise ValueError("deployment_id must match " + DEPLOYMENT_ID)
    return deployment_id


def _model_call(operation: Operation, model, identifier: str) -> _Call:
    return operation, model, operation.path_for(identifier=identifier), None


def _check(result: T, expected: str, actual: str) -> T:
    if expected != actual:
        raise ContractError("Menagerie receipt names another subject than requested")
    return result


class _Calls:
    @staticmethod
    def export() -> _Call:
        return ops.EXPORT, MenagerieExport, ops.EXPORT.path, None

    @staticmethod
    def resolve(identifier: str) -> _Call:
        return _model_call(ops.RESOLVE_MODEL, ModelResolution, identifier)

    @staticmethod
    def archive_model(identifier: str) -> _Call:
        return _model_call(ops.ARCHIVE_MODEL, ModelArchive, identifier)

    @staticmethod
    def restore_model(identifier: str) -> _Call:
        return _model_call(ops.RESTORE_MODEL, ModelRestore, identifier)

    @staticmethod
    def lineage(identifier: str) -> _Call:
        return _model_call(ops.MODEL_LINEAGE, LineageView, identifier)

    @staticmethod
    def record_lineage(request: LineageWrite) -> _Call:
        return ops.RECORD_LINEAGE, LineageEdge, ops.RECORD_LINEAGE.path, request

    @staticmethod
    def record_deployment(deployment_id: str, request: DeploymentWrite) -> _Call:
        path = ops.RECORD_DEPLOYMENT.path_for(deployment_id=_deployment(deployment_id))
        return ops.RECORD_DEPLOYMENT, Deployment, path, request

    @staticmethod
    def retire_deployment(deployment_id: str) -> _Call:
        path = ops.RETIRE_DEPLOYMENT.path_for(deployment_id=_deployment(deployment_id))
        return ops.RETIRE_DEPLOYMENT, Deployment, path, None

    @staticmethod
    def archive_record(record_id: str) -> _Call:
        return (
            ops.ARCHIVE_RECORD,
            RecordArchive,
            ops.ARCHIVE_RECORD.path_for(record_id=record_id),
            None,
        )

    @staticmethod
    def restore_record(record_id: str) -> _Call:
        return (
            ops.RESTORE_RECORD,
            RecordRestore,
            ops.RESTORE_RECORD.path_for(record_id=record_id),
            None,
        )


def _validate(call: _Call, subject: str | None):
    operation = call[0]
    if operation in (ops.RECORD_DEPLOYMENT, ops.RETIRE_DEPLOYMENT):
        return lambda result: _check(result, subject, result.deployment_id)
    if operation in (ops.ARCHIVE_RECORD, ops.RESTORE_RECORD):
        return lambda result: _check(result, subject, result.record_id)
    return None


class MenagerieClient:
    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def _send(self, call: _Call, subject: str | None = None):
        operation, model, path, body = call
        return self._transport.request(
            operation, model, body=body, path=path, validate=_validate(call, subject)
        )

    def export(self) -> MenagerieExport:
        """The caller's organization's whole Menagerie state, as logical identity."""
        return self._send(_Calls.export())

    def resolve(self, identifier: str) -> ModelResolution:
        return self._send(_Calls.resolve(identifier))

    def archive_model(self, identifier: str) -> ModelArchive:
        """Delete boundary: archive (never hard-delete); refuses while dependents are active."""
        return self._send(_Calls.archive_model(identifier))

    def restore_model(self, identifier: str) -> ModelRestore:
        return self._send(_Calls.restore_model(identifier))

    def lineage(self, identifier: str) -> LineageView:
        return self._send(_Calls.lineage(identifier))

    def record_lineage(self, request: LineageWrite) -> LineageEdge:
        return self._send(_Calls.record_lineage(request))

    def record_deployment(self, deployment_id: str, request: DeploymentWrite) -> Deployment:
        return self._send(_Calls.record_deployment(deployment_id, request), deployment_id)

    def retire_deployment(self, deployment_id: str) -> Deployment:
        return self._send(_Calls.retire_deployment(deployment_id), deployment_id)

    def archive_record(self, record_id: str) -> RecordArchive:
        return self._send(_Calls.archive_record(record_id), record_id)

    def restore_record(self, record_id: str) -> RecordRestore:
        return self._send(_Calls.restore_record(record_id), record_id)


class AsyncMenagerieClient:
    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def _send(self, call: _Call, subject: str | None = None):
        operation, model, path, body = call
        return await self._transport.request(
            operation, model, body=body, path=path, validate=_validate(call, subject)
        )

    async def export(self) -> MenagerieExport:
        return await self._send(_Calls.export())

    async def resolve(self, identifier: str) -> ModelResolution:
        return await self._send(_Calls.resolve(identifier))

    async def archive_model(self, identifier: str) -> ModelArchive:
        return await self._send(_Calls.archive_model(identifier))

    async def restore_model(self, identifier: str) -> ModelRestore:
        return await self._send(_Calls.restore_model(identifier))

    async def lineage(self, identifier: str) -> LineageView:
        return await self._send(_Calls.lineage(identifier))

    async def record_lineage(self, request: LineageWrite) -> LineageEdge:
        return await self._send(_Calls.record_lineage(request))

    async def record_deployment(self, deployment_id: str, request: DeploymentWrite) -> Deployment:
        return await self._send(_Calls.record_deployment(deployment_id, request), deployment_id)

    async def retire_deployment(self, deployment_id: str) -> Deployment:
        return await self._send(_Calls.retire_deployment(deployment_id), deployment_id)

    async def archive_record(self, record_id: str) -> RecordArchive:
        return await self._send(_Calls.archive_record(record_id), record_id)

    async def restore_record(self, record_id: str) -> RecordRestore:
        return await self._send(_Calls.restore_record(record_id), record_id)
