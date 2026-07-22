"""Stable Environment catalog APIs over the shared core transport.

# See: specifications/sdk/core_research_migration.md
"""

from __future__ import annotations

from urllib.parse import quote

from synth_ai.core.contracts.json_value import JsonObject
from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.request import HttpRequest
from synth_ai.core.http.transport import HttpTransport
from synth_ai.core.research.contracts._environment_wire import digest, integer, text
from synth_ai.core.research.contracts._wire import array_value
from synth_ai.core.research.contracts.common import EnvironmentDigest, EnvironmentName
from synth_ai.core.research.contracts.environment_manifest import EnvironmentManifest
from synth_ai.core.research.contracts.environments import (
    Environment,
    EnvironmentDetail,
    EnvironmentPreflight,
)
from synth_ai.core.research.operations import research_operation


def _request(
    operation_id: str,
    path: str,
    *,
    query: JsonObject | None = None,
    body: JsonObject | None = None,
) -> HttpRequest:
    return HttpRequest(
        research_operation(operation_id),
        path,
        query=query or {},
        body=body,
    )


def _name_path(name: EnvironmentName) -> tuple[EnvironmentName, str]:
    normalized = EnvironmentName(text(name, field="environment name", maximum=255))
    return normalized, quote(normalized, safe="")


def _selector_query(manifest_digest: EnvironmentDigest | None) -> JsonObject:
    if manifest_digest is None:
        return {}
    return {
        "manifest_digest": digest(
            manifest_digest,
            field="manifest_digest",
        )
    }


def _environments(value: object) -> tuple[Environment, ...]:
    return tuple(
        Environment.from_wire(item)
        for item in array_value(value, operation_id="list_research_environments")
    )


def _verify_detail(
    detail: EnvironmentDetail,
    *,
    name: EnvironmentName,
    manifest_digest: EnvironmentDigest | None,
) -> EnvironmentDetail:
    if detail.name != name:
        raise ValueError("environment response crossed its requested name boundary")
    if manifest_digest is not None and detail.manifest_digest != manifest_digest:
        raise ValueError("environment response crossed its requested digest boundary")
    return detail


def _verify_preflight(
    result: EnvironmentPreflight,
    *,
    name: EnvironmentName,
    manifest_digest: EnvironmentDigest | None,
) -> EnvironmentPreflight:
    if result.name != name:
        raise ValueError("environment preflight crossed its requested name boundary")
    if manifest_digest is not None and result.manifest_digest != manifest_digest:
        raise ValueError("environment preflight crossed its requested digest boundary")
    return result


class EnvironmentsAPI:
    """Immutable Environment catalog and deterministic preflight operations."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def list(self, *, limit: int = 100) -> tuple[Environment, ...]:
        validated_limit = integer(limit, field="limit", minimum=1, maximum=500)
        value = self._transport.execute(
            _request(
                "list_research_environments",
                "/smr/environments",
                query={"limit": validated_limit},
            )
        )
        return _environments(value)

    def create(self, manifest: EnvironmentManifest) -> EnvironmentDetail:
        if not isinstance(manifest, EnvironmentManifest):
            raise ValueError("manifest must be EnvironmentManifest")
        value = self._transport.execute(
            _request(
                "create_research_environment",
                "/smr/environments",
                body={"manifest": manifest.to_wire()},
            )
        )
        detail = EnvironmentDetail.from_wire(value)
        return _verify_detail(
            detail,
            name=manifest.name,
            manifest_digest=manifest.digest,
        )

    def retrieve(
        self,
        name: EnvironmentName,
        *,
        manifest_digest: EnvironmentDigest | None = None,
    ) -> EnvironmentDetail:
        normalized_name, path_name = _name_path(name)
        value = self._transport.execute(
            _request(
                "retrieve_research_environment",
                f"/smr/environments/{path_name}",
                query=_selector_query(manifest_digest),
            )
        )
        return _verify_detail(
            EnvironmentDetail.from_wire(value),
            name=normalized_name,
            manifest_digest=manifest_digest,
        )

    def preflight(
        self,
        name: EnvironmentName,
        *,
        manifest_digest: EnvironmentDigest | None = None,
    ) -> EnvironmentPreflight:
        normalized_name, path_name = _name_path(name)
        value = self._transport.execute(
            _request(
                "preflight_research_environment",
                f"/smr/environments/{path_name}/preflight",
                query=_selector_query(manifest_digest),
            )
        )
        return _verify_preflight(
            EnvironmentPreflight.from_wire(value),
            name=normalized_name,
            manifest_digest=manifest_digest,
        )


class AsyncEnvironmentsAPI:
    """Native-async peer of :class:`EnvironmentsAPI`."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def list(self, *, limit: int = 100) -> tuple[Environment, ...]:
        validated_limit = integer(limit, field="limit", minimum=1, maximum=500)
        value = await self._transport.execute(
            _request(
                "list_research_environments",
                "/smr/environments",
                query={"limit": validated_limit},
            )
        )
        return _environments(value)

    async def create(self, manifest: EnvironmentManifest) -> EnvironmentDetail:
        if not isinstance(manifest, EnvironmentManifest):
            raise ValueError("manifest must be EnvironmentManifest")
        value = await self._transport.execute(
            _request(
                "create_research_environment",
                "/smr/environments",
                body={"manifest": manifest.to_wire()},
            )
        )
        return _verify_detail(
            EnvironmentDetail.from_wire(value),
            name=manifest.name,
            manifest_digest=manifest.digest,
        )

    async def retrieve(
        self,
        name: EnvironmentName,
        *,
        manifest_digest: EnvironmentDigest | None = None,
    ) -> EnvironmentDetail:
        normalized_name, path_name = _name_path(name)
        value = await self._transport.execute(
            _request(
                "retrieve_research_environment",
                f"/smr/environments/{path_name}",
                query=_selector_query(manifest_digest),
            )
        )
        return _verify_detail(
            EnvironmentDetail.from_wire(value),
            name=normalized_name,
            manifest_digest=manifest_digest,
        )

    async def preflight(
        self,
        name: EnvironmentName,
        *,
        manifest_digest: EnvironmentDigest | None = None,
    ) -> EnvironmentPreflight:
        normalized_name, path_name = _name_path(name)
        value = await self._transport.execute(
            _request(
                "preflight_research_environment",
                f"/smr/environments/{path_name}/preflight",
                query=_selector_query(manifest_digest),
            )
        )
        return _verify_preflight(
            EnvironmentPreflight.from_wire(value),
            name=normalized_name,
            manifest_digest=manifest_digest,
        )


__all__ = ["AsyncEnvironmentsAPI", "EnvironmentsAPI"]
