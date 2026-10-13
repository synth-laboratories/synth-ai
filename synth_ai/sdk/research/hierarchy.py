"""Direct project-owner APIs with fresh, isolated capability transport per call.

# See: testing/specifications/sdk/hierarchy_owner.md
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Annotated, Literal, TypeVar, cast

from pydantic import Field, TypeAdapter

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.transport import HttpTransport
from synth_ai.sdk.research.hierarchy_models import (
    Closed,
    HierarchyCommand,
    HierarchyCommandReceipt,
    HierarchyOriginal,
    HierarchyPermission,
    HierarchyRead,
    HierarchyReadReply,
    HierarchyReference,
    HierarchyScope,
    HierarchyTransferBundle,
    HierarchyTransferReceipt,
    OperationId,
    Positive,
)
from synth_ai.sdk.research.owner_reads import _origin

Model = TypeVar("Model", bound=Closed)
_operation = TypeAdapter(OperationId)


def _wire(model: Closed) -> JsonObject:
    return cast(JsonObject, model.model_dump(mode="json", by_alias=True))


def _decode(model: type[Model], value: JsonValue) -> Model:
    return model.model_validate_json(json.dumps(value, ensure_ascii=False, allow_nan=False))


class _Access(Closed):
    schema_version: Literal["synth.hierarchy-access.v1"]
    scope: HierarchyScope
    owner_origin: str
    token: Annotated[str, Field(min_length=1, max_length=8192, repr=False)]
    expires_at_ms: Positive

    def checked(self, scope: HierarchyScope) -> _Access:
        if self.scope != scope or self.expires_at_ms <= time.time_ns() // 1_000_000:
            raise ValueError("hierarchy access scope or expiry invalid")
        if _origin(self.owner_origin) != self.owner_origin:
            raise ValueError("hierarchy owner origin must be normalized")
        if self.owner_origin == "http://orchestra:8790":
            raise ValueError("hierarchy credential belongs to Sublinear only")
        if any(ord(c) < 33 or ord(c) > 126 for c in self.token):
            raise ValueError("hierarchy credential must be a header token")
        return self


def _access_request(
    scope: HierarchyScope, permissions: tuple[HierarchyPermission, ...]
) -> JsonObject:
    allowed = {"read", "plan", "evidence", "review", "association", "transfer"}
    if not permissions or set(permissions) - allowed or len(set(permissions)) != len(permissions):
        raise ValueError("hierarchy permissions invalid")
    return {
        "schema_version": "synth.hierarchy-access-request.v1",
        "scope": _wire(scope),
        "permissions": sorted(permissions),
    }


def _scoped(result: Model, scope: HierarchyScope) -> Model:
    if getattr(result, "scope", None) != scope:
        raise ValueError("hierarchy owner reply scope mismatch")
    if isinstance(result, HierarchyReadReply):
        for original in result.originals:
            if original.document().get("scope") != _wire(
                scope
            ) or not original.reference.reference.startswith(f"project.{scope.project_id}."):
                raise ValueError("hierarchy original reply scope mismatch")
    return result


def _transfer(
    scope: HierarchyScope, phase: str, operation_id: str, **fields: JsonValue
) -> JsonObject:
    _operation.validate_python(operation_id)
    return {
        "schema_version": f"sublinear.hierarchy-transfer-{phase}.v1",
        "scope": _wire(scope),
        "operation_id": operation_id,
        **fields,
    }


@contextmanager
def _owner_transport(origin: str, token: str) -> Iterator[HttpTransport]:
    owner = HttpTransport(
        base_url=origin, headers={"Authorization": f"Bearer {token}"}, follow_redirects=False
    )
    try:
        yield owner
    finally:
        owner.close()


@asynccontextmanager
async def _async_owner_transport(origin: str, token: str) -> AsyncIterator[AsyncHttpTransport]:
    owner = AsyncHttpTransport(
        base_url=origin, headers={"Authorization": f"Bearer {token}"}, follow_redirects=False
    )
    try:
        yield owner
    finally:
        await owner.close()


class HierarchyClient:
    """Project-only owner APIs; no legacy projector fallback or implicit transfer.

    # See: testing/specifications/sdk/hierarchy_owner.md
    """

    def __init__(
        self,
        backend: HttpTransport,
        *,
        scope: HierarchyScope,
        permissions: tuple[HierarchyPermission, ...] = ("read",),
    ) -> None:
        self._backend = backend
        self.scope = scope
        self._access_body = _access_request(scope, permissions)

    def _access(self) -> _Access:
        return _decode(
            _Access,
            self._backend.request_json(
                "POST", "/auth/capabilities/hierarchy", json_body=self._access_body
            ),
        ).checked(self.scope)

    def _owner(self, suffix: str, body: JsonObject) -> JsonValue:
        access = self._access()
        with _owner_transport(access.owner_origin, access.token) as owner:
            return owner.request_json(
                "POST", f"/v1/projects/{self.scope.project_id}/hierarchy/{suffix}", json_body=body
            )

    def read(self, request: HierarchyRead) -> HierarchyReadReply:
        """Read exact owner originals/current children; see hierarchy_owner.md."""
        _scoped(request, self.scope)
        return _scoped(_decode(HierarchyReadReply, self._owner("read", _wire(request))), self.scope)

    def command(self, command: HierarchyCommand) -> HierarchyCommandReceipt:
        """Apply a typed owner CAS command; see hierarchy_owner.md."""
        _scoped(command, self.scope)
        result = _scoped(
            _decode(HierarchyCommandReceipt, self._owner("commands", _wire(command))), self.scope
        )
        if result.operation_id != command.operation_id or result.entity != command.target:
            raise ValueError("hierarchy command receipt identity mismatch")
        return result

    def enroll(self, operation_id: str) -> HierarchyOriginal:
        """Explicitly enroll this selected project; see hierarchy_owner.md."""
        _operation.validate_python(operation_id)
        access = self._access()
        value = self._backend.request_json(
            "POST",
            "/smr/v1/hierarchy/enrollment-challenge",
            json_body={"scope": _wire(self.scope), "operation_id": operation_id},
        )
        if not isinstance(value, dict) or set(value) != {"challenge", "token"}:
            raise ValueError("hierarchy enrollment challenge invalid")
        challenge = _decode(HierarchyOriginal, value["challenge"])
        document = challenge.document()
        token = value["token"]
        if (
            document.get("scope") != _wire(self.scope)
            or document.get("operation_id") != operation_id
            or document.get("owner_origin") != access.owner_origin
            or not isinstance(token, str)
            or not token
            or len(token) > 8192
            or any(ord(c) < 33 or ord(c) > 126 for c in token)
        ):
            raise ValueError("hierarchy enrollment challenge scope/origin invalid")
        with _owner_transport(access.owner_origin, token) as owner:
            reply = owner.request_json(
                "POST",
                f"/v1/projects/{self.scope.project_id}/hierarchy/enrollment",
                json_body={
                    "schema_version": "sublinear.hierarchy-enrollment-request.v1",
                    "scope": _wire(self.scope),
                    "operation_id": operation_id,
                    "challenge": _wire(challenge),
                },
            )
        result = _decode(HierarchyOriginal, reply)
        if (
            result.reference.owner != "sublinear"
            or result.reference.reference
            != f"project.{self.scope.project_id}.hierarchy-enrollment.{operation_id}"
            or result.document().get("operation_id") != operation_id
            or result.document().get("challenge") != _wire(challenge.reference)
            or result.reference.schema_ != "sublinear.hierarchy-enrollment.v1"
            or result.document().get("scope") != _wire(self.scope)
        ):
            raise ValueError("hierarchy enrollment original invalid")
        return result

    def freeze_transfer(
        self, operation_id: str, owner_enrollment: HierarchyOriginal
    ) -> HierarchyTransferBundle:
        """Freeze/export only the explicitly enrolled scope; see hierarchy_owner.md."""
        _operation.validate_python(operation_id)
        value = self._backend.request_json(
            "POST",
            "/smr/v1/hierarchy/freeze-transfer",
            json_body={
                "schema_version": "synth.hierarchy-freeze-transfer.v1",
                "scope": _wire(self.scope),
                "operation_id": operation_id,
                "owner_enrollment": _wire(owner_enrollment),
            },
        )
        bundle = _decode(HierarchyTransferBundle, value)
        if (
            bundle.decision.document().get("scope") != _wire(self.scope)
            or bundle.decision.document().get("operation_id") != operation_id
        ):
            raise ValueError("hierarchy frozen transfer scope mismatch")
        return bundle

    def prepare(
        self, operation_id: str, bundle: HierarchyTransferBundle
    ) -> HierarchyTransferReceipt:
        """Prepare the exact frozen export; see hierarchy_owner.md."""
        return self._phase(
            "prepare",
            operation_id,
            decision=_wire(bundle.decision),
            manifest=_wire(bundle.manifest),
        )

    def stage(self, operation_id: str, chunk: HierarchyOriginal) -> HierarchyTransferReceipt:
        """Stage one original export chunk; see hierarchy_owner.md."""
        return self._phase("stage", operation_id, chunk=_wire(chunk))

    def activate(
        self,
        operation_id: str,
        *,
        transfer_operation_id: str,
        expected_authority_epoch: int,
        decision: HierarchyReference,
        manifest: HierarchyReference,
    ) -> HierarchyTransferReceipt:
        """Activate under the exact source epoch; see hierarchy_owner.md."""
        _operation.validate_python(transfer_operation_id)
        TypeAdapter(Positive).validate_python(expected_authority_epoch)
        return self._phase(
            "activate",
            operation_id,
            transfer_operation_id=transfer_operation_id,
            expected_authority_epoch=expected_authority_epoch,
            decision=_wire(decision),
            manifest=_wire(manifest),
        )

    def _phase(
        self, phase: str, operation_id: str, **fields: JsonValue
    ) -> HierarchyTransferReceipt:
        result = _scoped(
            _decode(
                HierarchyTransferReceipt,
                self._owner(
                    "transfer/" + phase, _transfer(self.scope, phase, operation_id, **fields)
                ),
            ),
            self.scope,
        )
        if (
            result.operation_id != operation_id
            or result.phase
            != {"prepare": "prepared", "stage": "staged", "activate": "owned"}[phase]
        ):
            raise ValueError("hierarchy transfer receipt identity mismatch")
        return result

    def confirm_activation(
        self, *, transfer_operation_id: str, activation_operation_id: str
    ) -> HierarchyOriginal:
        """Backend rereads the genuine owner receipt before confirming; see hierarchy_owner.md."""
        _operation.validate_python(transfer_operation_id)
        _operation.validate_python(activation_operation_id)
        result = _decode(
            HierarchyOriginal,
            self._backend.request_json(
                "POST",
                "/smr/v1/hierarchy/activation-confirm",
                json_body={
                    "schema_version": "synth.hierarchy-activation-confirm-request.v1",
                    "scope": _wire(self.scope),
                    "transfer_operation_id": transfer_operation_id,
                    "activation_operation_id": activation_operation_id,
                },
            ),
        )
        if (
            result.reference.owner != "backend"
            or result.reference.schema_ != "synth.hierarchy-owner-activation-custody.v1"
            or result.reference.reference
            != f"project.{self.scope.project_id}.hierarchy-owner-activation.{activation_operation_id}"
            or result.document().get("scope") != _wire(self.scope)
        ):
            raise ValueError("hierarchy activation confirmation scope mismatch")
        original = _decode(HierarchyOriginal, result.document().get("original"))
        receipt = _decode(HierarchyTransferReceipt, original.document())
        if (
            receipt.scope != self.scope
            or receipt.phase != "owned"
            or receipt.transfer_operation_id != transfer_operation_id
            or receipt.operation_id != activation_operation_id
        ):
            raise ValueError("hierarchy activation owner receipt mismatch")
        return result


class AsyncHierarchyClient:
    """Project-only owner APIs; no legacy projector fallback or implicit transfer.

    # See: testing/specifications/sdk/hierarchy_owner.md
    """

    def __init__(
        self,
        backend: AsyncHttpTransport,
        *,
        scope: HierarchyScope,
        permissions: tuple[HierarchyPermission, ...] = ("read",),
    ) -> None:
        self._backend = backend
        self.scope = scope
        self._access_body = _access_request(scope, permissions)

    async def _access(self) -> _Access:
        return _decode(
            _Access,
            await self._backend.request_json(
                "POST", "/auth/capabilities/hierarchy", json_body=self._access_body
            ),
        ).checked(self.scope)

    async def _owner(self, suffix: str, body: JsonObject) -> JsonValue:
        access = await self._access()
        async with _async_owner_transport(access.owner_origin, access.token) as owner:
            return await owner.request_json(
                "POST", f"/v1/projects/{self.scope.project_id}/hierarchy/{suffix}", json_body=body
            )

    async def read(self, request: HierarchyRead) -> HierarchyReadReply:
        """Read exact owner originals/current children; see hierarchy_owner.md."""
        _scoped(request, self.scope)
        return _scoped(
            _decode(HierarchyReadReply, await self._owner("read", _wire(request))), self.scope
        )

    async def command(self, command: HierarchyCommand) -> HierarchyCommandReceipt:
        """Apply a typed owner CAS command; see hierarchy_owner.md."""
        _scoped(command, self.scope)
        result = _scoped(
            _decode(HierarchyCommandReceipt, await self._owner("commands", _wire(command))),
            self.scope,
        )
        if result.operation_id != command.operation_id or result.entity != command.target:
            raise ValueError("hierarchy command receipt identity mismatch")
        return result

    async def enroll(self, operation_id: str) -> HierarchyOriginal:
        """Explicitly enroll this selected project; see hierarchy_owner.md."""
        _operation.validate_python(operation_id)
        access = await self._access()
        value = await self._backend.request_json(
            "POST",
            "/smr/v1/hierarchy/enrollment-challenge",
            json_body={"scope": _wire(self.scope), "operation_id": operation_id},
        )
        if not isinstance(value, dict) or set(value) != {"challenge", "token"}:
            raise ValueError("hierarchy enrollment challenge invalid")
        challenge = _decode(HierarchyOriginal, value["challenge"])
        document = challenge.document()
        token = value["token"]
        if (
            document.get("scope") != _wire(self.scope)
            or document.get("operation_id") != operation_id
            or document.get("owner_origin") != access.owner_origin
            or not isinstance(token, str)
            or not token
            or len(token) > 8192
            or any(ord(c) < 33 or ord(c) > 126 for c in token)
        ):
            raise ValueError("hierarchy enrollment challenge scope/origin invalid")
        async with _async_owner_transport(access.owner_origin, token) as owner:
            reply = await owner.request_json(
                "POST",
                f"/v1/projects/{self.scope.project_id}/hierarchy/enrollment",
                json_body={
                    "schema_version": "sublinear.hierarchy-enrollment-request.v1",
                    "scope": _wire(self.scope),
                    "operation_id": operation_id,
                    "challenge": _wire(challenge),
                },
            )
        result = _decode(HierarchyOriginal, reply)
        if (
            result.reference.owner != "sublinear"
            or result.reference.reference
            != f"project.{self.scope.project_id}.hierarchy-enrollment.{operation_id}"
            or result.document().get("operation_id") != operation_id
            or result.document().get("challenge") != _wire(challenge.reference)
            or result.reference.schema_ != "sublinear.hierarchy-enrollment.v1"
            or result.document().get("scope") != _wire(self.scope)
        ):
            raise ValueError("hierarchy enrollment original invalid")
        return result

    async def freeze_transfer(
        self, operation_id: str, owner_enrollment: HierarchyOriginal
    ) -> HierarchyTransferBundle:
        """Freeze/export only the explicitly enrolled scope; see hierarchy_owner.md."""
        _operation.validate_python(operation_id)
        value = await self._backend.request_json(
            "POST",
            "/smr/v1/hierarchy/freeze-transfer",
            json_body={
                "schema_version": "synth.hierarchy-freeze-transfer.v1",
                "scope": _wire(self.scope),
                "operation_id": operation_id,
                "owner_enrollment": _wire(owner_enrollment),
            },
        )
        bundle = _decode(HierarchyTransferBundle, value)
        if (
            bundle.decision.document().get("scope") != _wire(self.scope)
            or bundle.decision.document().get("operation_id") != operation_id
        ):
            raise ValueError("hierarchy frozen transfer scope mismatch")
        return bundle

    async def prepare(
        self, operation_id: str, bundle: HierarchyTransferBundle
    ) -> HierarchyTransferReceipt:
        """Prepare the exact frozen export; see hierarchy_owner.md."""
        return await self._phase(
            "prepare",
            operation_id,
            decision=_wire(bundle.decision),
            manifest=_wire(bundle.manifest),
        )

    async def stage(self, operation_id: str, chunk: HierarchyOriginal) -> HierarchyTransferReceipt:
        """Stage one original export chunk; see hierarchy_owner.md."""
        return await self._phase("stage", operation_id, chunk=_wire(chunk))

    async def activate(
        self,
        operation_id: str,
        *,
        transfer_operation_id: str,
        expected_authority_epoch: int,
        decision: HierarchyReference,
        manifest: HierarchyReference,
    ) -> HierarchyTransferReceipt:
        """Activate under the exact source epoch; see hierarchy_owner.md."""
        _operation.validate_python(transfer_operation_id)
        TypeAdapter(Positive).validate_python(expected_authority_epoch)
        return await self._phase(
            "activate",
            operation_id,
            transfer_operation_id=transfer_operation_id,
            expected_authority_epoch=expected_authority_epoch,
            decision=_wire(decision),
            manifest=_wire(manifest),
        )

    async def _phase(
        self, phase: str, operation_id: str, **fields: JsonValue
    ) -> HierarchyTransferReceipt:
        result = _scoped(
            _decode(
                HierarchyTransferReceipt,
                await self._owner(
                    "transfer/" + phase, _transfer(self.scope, phase, operation_id, **fields)
                ),
            ),
            self.scope,
        )
        if (
            result.operation_id != operation_id
            or result.phase
            != {"prepare": "prepared", "stage": "staged", "activate": "owned"}[phase]
        ):
            raise ValueError("hierarchy transfer receipt identity mismatch")
        return result

    async def confirm_activation(
        self, *, transfer_operation_id: str, activation_operation_id: str
    ) -> HierarchyOriginal:
        """Backend rereads the genuine owner receipt before confirming; see hierarchy_owner.md."""
        _operation.validate_python(transfer_operation_id)
        _operation.validate_python(activation_operation_id)
        result = _decode(
            HierarchyOriginal,
            await self._backend.request_json(
                "POST",
                "/smr/v1/hierarchy/activation-confirm",
                json_body={
                    "schema_version": "synth.hierarchy-activation-confirm-request.v1",
                    "scope": _wire(self.scope),
                    "transfer_operation_id": transfer_operation_id,
                    "activation_operation_id": activation_operation_id,
                },
            ),
        )
        if (
            result.reference.owner != "backend"
            or result.reference.schema_ != "synth.hierarchy-owner-activation-custody.v1"
            or result.reference.reference
            != f"project.{self.scope.project_id}.hierarchy-owner-activation.{activation_operation_id}"
            or result.document().get("scope") != _wire(self.scope)
        ):
            raise ValueError("hierarchy activation confirmation scope mismatch")
        original = _decode(HierarchyOriginal, result.document().get("original"))
        receipt = _decode(HierarchyTransferReceipt, original.document())
        if (
            receipt.scope != self.scope
            or receipt.phase != "owned"
            or receipt.transfer_operation_id != transfer_operation_id
            or receipt.operation_id != activation_operation_id
        ):
            raise ValueError("hierarchy activation owner receipt mismatch")
        return result
