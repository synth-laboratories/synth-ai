"""Synchronous and asynchronous first-class Research Visual operations.

# See: Jstack/.jstack/daily_notes/2026-07-27/SPEC_visuals_resource.md
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from synth_ai.core.contracts.json_value import JsonObject, JsonValue
from synth_ai.core.http.async_transport import AsyncHttpTransport
from synth_ai.core.http.request import HttpRequest
from synth_ai.core.http.transport import HttpTransport
from synth_ai.core.research.contracts.common import (
    ArtifactId,
    EffortId,
    FactoryId,
    ProjectId,
    SwarmId,
)
from synth_ai.core.research.contracts.visuals import (
    Visual,
    VisualPage,
    VisualPatch,
    VisualPromotion,
    VisualVisibility,
    VisualVersions,
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


def _html_bytes(html: str | bytes | Path) -> bytes:
    if isinstance(html, Path):
        return html.read_bytes()
    if isinstance(html, str):
        return html.encode("utf-8")
    return bytes(html)


def _upload_parts(
    *,
    title: str,
    html: str | bytes | Path,
    visual_kind: str,
    source_run_ids: Iterable[SwarmId | str],
    metadata: Mapping[str, JsonValue] | None,
) -> tuple[dict[str, str], dict[str, tuple[str, bytes, str]]]:
    normalized_title = title.strip()
    if not normalized_title:
        raise ValueError("title must not be empty")
    normalized_kind = visual_kind.strip()
    if not normalized_kind:
        raise ValueError("visual_kind must not be empty")
    data = {
        "title": normalized_title,
        "visual_kind": normalized_kind,
        "source_run_ids_json": json.dumps([str(item) for item in source_run_ids]),
        "metadata_json": json.dumps(dict(metadata or {})),
    }
    return data, {"html": ("index.html", _html_bytes(html), "text/html")}


def _list_query(
    *,
    visual_kind: str | None,
    visibility: VisualVisibility | None,
    factory_id: FactoryId | None,
    effort_id: EffortId | None,
    include_deleted: bool,
    cursor: str | None,
    limit: int,
) -> JsonObject:
    if limit < 1 or limit > 250:
        raise ValueError("limit must be between 1 and 250")
    query: JsonObject = {
        "include_deleted": include_deleted,
        "limit": limit,
    }
    if visual_kind is not None:
        normalized_kind = visual_kind.strip()
        if not normalized_kind:
            raise ValueError("visual_kind must not be empty when provided")
        query["visual_kind"] = normalized_kind
    if visibility is not None:
        query["visibility"] = visibility.value
    if factory_id is not None:
        query["factory_id"] = str(factory_id)
    if effort_id is not None:
        query["effort_id"] = str(effort_id)
    if cursor is not None:
        normalized_cursor = cursor.strip()
        if not normalized_cursor:
            raise ValueError("cursor must not be empty when provided")
        query["cursor"] = normalized_cursor
    return query


class VisualsAPI:
    """Typed Visual publication, lifecycle, sharing, and content operations."""

    def __init__(self, transport: HttpTransport) -> None:
        self._transport = transport

    def publish_run(
        self,
        run_id: SwarmId,
        *,
        title: str,
        html: str | bytes | Path,
        visual_kind: str = "research_visual",
        source_run_ids: Iterable[SwarmId | str] = (),
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> Visual:
        """Publish self-contained HTML produced by a Research run."""
        data, files = _upload_parts(
            title=title,
            html=html,
            visual_kind=visual_kind,
            source_run_ids=source_run_ids,
            metadata=metadata,
        )
        data["visibility"] = VisualVisibility.ORGANIZATION.value
        value = self._transport.request_multipart_json(
            _request(
                "publish_run_visual",
                f"/smr/runs/{run_id}/visuals",
            ),
            data=data,
            files=files,
        )
        return Visual.from_wire(value)

    def create(
        self,
        project_id: ProjectId,
        *,
        title: str,
        html: str | bytes | Path,
        visual_kind: str = "research_visual",
        source_run_ids: Iterable[SwarmId | str] = (),
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> Visual:
        """Publish a project-owned Visual without requiring a run."""
        data, files = _upload_parts(
            title=title,
            html=html,
            visual_kind=visual_kind,
            source_run_ids=source_run_ids,
            metadata=metadata,
        )
        value = self._transport.request_multipart_json(
            _request(
                "create_project_visual",
                f"/smr/projects/{project_id}/visuals",
            ),
            data=data,
            files=files,
        )
        return Visual.from_wire(value)

    def list(
        self,
        project_id: ProjectId,
        *,
        visual_kind: str | None = None,
        visibility: VisualVisibility | None = None,
        factory_id: FactoryId | None = None,
        effort_id: EffortId | None = None,
        include_deleted: bool = False,
        cursor: str | None = None,
        limit: int = 100,
    ) -> VisualPage:
        """List one cursor-paginated page of project Visuals."""
        value = self._transport.execute(
            _request(
                "list_project_visuals",
                f"/smr/projects/{project_id}/visuals",
                query=_list_query(
                    visual_kind=visual_kind,
                    visibility=visibility,
                    factory_id=factory_id,
                    effort_id=effort_id,
                    include_deleted=include_deleted,
                    cursor=cursor,
                    limit=limit,
                ),
            )
        )
        return VisualPage.from_wire(value)

    def retrieve(self, visual_id: ArtifactId) -> Visual:
        """Retrieve one organization-authorized Visual."""
        value = self._transport.execute(
            _request("retrieve_visual", f"/smr/visuals/{visual_id}")
        )
        return Visual.from_wire(value)

    def list_versions(self, visual_id: ArtifactId) -> VisualVersions:
        """List immutable versions for a Visual lineage."""
        value = self._transport.execute(
            _request(
                "list_visual_versions",
                f"/smr/visuals/{visual_id}/versions",
            )
        )
        return VisualVersions.from_wire(value)

    def update(self, visual_id: ArtifactId, request: VisualPatch) -> Visual:
        """Update mutable Visual metadata."""
        value = self._transport.execute(
            _request(
                "update_visual",
                f"/smr/visuals/{visual_id}",
                body=request.to_wire(),
            )
        )
        return Visual.from_wire(value)

    def delete(self, visual_id: ArtifactId) -> None:
        """Soft-delete a Visual and schedule blob cleanup."""
        self._transport.execute(
            _request("delete_visual", f"/smr/visuals/{visual_id}")
        )

    def restore(self, visual_id: ArtifactId) -> Visual:
        """Restore a soft-deleted Visual while its blob remains available."""
        value = self._transport.execute(
            _request("restore_visual", f"/smr/visuals/{visual_id}/restore")
        )
        return Visual.from_wire(value)

    def promote(self, visual_id: ArtifactId, request: VisualPromotion) -> Visual:
        """Promote a Visual through the backend's ADMIN/OWNER authority."""
        value = self._transport.execute(
            _request(
                "promote_visual",
                f"/smr/visuals/{visual_id}/promote",
                body=request.to_wire(),
            )
        )
        return Visual.from_wire(value)

    def unpublish(self, visual_id: ArtifactId) -> Visual:
        """Remove public access while retaining the organization Visual."""
        value = self._transport.execute(
            _request("unpublish_visual", f"/smr/visuals/{visual_id}/unpublish")
        )
        return Visual.from_wire(value)

    def retrieve_content(self, visual_id: ArtifactId) -> bytes:
        """Download organization-authorized self-contained HTML."""
        operation_id = "retrieve_visual_content"
        return self._transport.request_bytes(
            "GET",
            f"/smr/visuals/{visual_id}/content",
            operation_id=operation_id,
        )

    def retrieve_public(self, slug: str) -> Visual:
        """Retrieve the safe public projection for a promoted slug."""
        normalized_slug = _slug(slug)
        value = self._transport.execute(
            _request(
                "retrieve_public_visual",
                f"/smr/public/visuals/{normalized_slug}",
            )
        )
        return Visual.from_wire(value)

    def retrieve_public_content(self, slug: str) -> bytes:
        """Download public self-contained HTML for a promoted slug."""
        normalized_slug = _slug(slug)
        return self._transport.request_bytes(
            "GET",
            f"/smr/public/visuals/{normalized_slug}/content",
            operation_id="retrieve_public_visual_content",
        )

    def retrieve_public_preview(self, slug: str) -> bytes:
        """Download the service-authored public preview image."""
        normalized_slug = _slug(slug)
        return self._transport.request_bytes(
            "GET",
            f"/smr/public/visuals/{normalized_slug}/preview",
            operation_id="retrieve_public_visual_preview",
        )


class AsyncVisualsAPI:
    """Native asynchronous peer of :class:`VisualsAPI`."""

    def __init__(self, transport: AsyncHttpTransport) -> None:
        self._transport = transport

    async def publish_run(
        self,
        run_id: SwarmId,
        *,
        title: str,
        html: str | bytes | Path,
        visual_kind: str = "research_visual",
        source_run_ids: Iterable[SwarmId | str] = (),
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> Visual:
        """Publish self-contained HTML produced by a Research run."""
        data, files = _upload_parts(
            title=title,
            html=html,
            visual_kind=visual_kind,
            source_run_ids=source_run_ids,
            metadata=metadata,
        )
        data["visibility"] = VisualVisibility.ORGANIZATION.value
        value = await self._transport.request_multipart_json(
            _request("publish_run_visual", f"/smr/runs/{run_id}/visuals"),
            data=data,
            files=files,
        )
        return Visual.from_wire(value)

    async def create(
        self,
        project_id: ProjectId,
        *,
        title: str,
        html: str | bytes | Path,
        visual_kind: str = "research_visual",
        source_run_ids: Iterable[SwarmId | str] = (),
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> Visual:
        """Publish a project-owned Visual without requiring a run."""
        data, files = _upload_parts(
            title=title,
            html=html,
            visual_kind=visual_kind,
            source_run_ids=source_run_ids,
            metadata=metadata,
        )
        value = await self._transport.request_multipart_json(
            _request(
                "create_project_visual",
                f"/smr/projects/{project_id}/visuals",
            ),
            data=data,
            files=files,
        )
        return Visual.from_wire(value)

    async def list(
        self,
        project_id: ProjectId,
        *,
        visual_kind: str | None = None,
        visibility: VisualVisibility | None = None,
        factory_id: FactoryId | None = None,
        effort_id: EffortId | None = None,
        include_deleted: bool = False,
        cursor: str | None = None,
        limit: int = 100,
    ) -> VisualPage:
        """List one cursor-paginated page of project Visuals."""
        value = await self._transport.execute(
            _request(
                "list_project_visuals",
                f"/smr/projects/{project_id}/visuals",
                query=_list_query(
                    visual_kind=visual_kind,
                    visibility=visibility,
                    factory_id=factory_id,
                    effort_id=effort_id,
                    include_deleted=include_deleted,
                    cursor=cursor,
                    limit=limit,
                ),
            )
        )
        return VisualPage.from_wire(value)

    async def retrieve(self, visual_id: ArtifactId) -> Visual:
        """Retrieve one organization-authorized Visual."""
        return Visual.from_wire(
            await self._transport.execute(
                _request("retrieve_visual", f"/smr/visuals/{visual_id}")
            )
        )

    async def list_versions(self, visual_id: ArtifactId) -> VisualVersions:
        """List immutable versions for a Visual lineage."""
        return VisualVersions.from_wire(
            await self._transport.execute(
                _request(
                    "list_visual_versions",
                    f"/smr/visuals/{visual_id}/versions",
                )
            )
        )

    async def update(self, visual_id: ArtifactId, request: VisualPatch) -> Visual:
        """Update mutable Visual metadata."""
        return Visual.from_wire(
            await self._transport.execute(
                _request(
                    "update_visual",
                    f"/smr/visuals/{visual_id}",
                    body=request.to_wire(),
                )
            )
        )

    async def delete(self, visual_id: ArtifactId) -> None:
        """Soft-delete a Visual and schedule blob cleanup."""
        await self._transport.execute(
            _request("delete_visual", f"/smr/visuals/{visual_id}")
        )

    async def restore(self, visual_id: ArtifactId) -> Visual:
        """Restore a soft-deleted Visual while its blob remains available."""
        return Visual.from_wire(
            await self._transport.execute(
                _request("restore_visual", f"/smr/visuals/{visual_id}/restore")
            )
        )

    async def promote(
        self,
        visual_id: ArtifactId,
        request: VisualPromotion,
    ) -> Visual:
        """Promote a Visual through the backend's ADMIN/OWNER authority."""
        return Visual.from_wire(
            await self._transport.execute(
                _request(
                    "promote_visual",
                    f"/smr/visuals/{visual_id}/promote",
                    body=request.to_wire(),
                )
            )
        )

    async def unpublish(self, visual_id: ArtifactId) -> Visual:
        """Remove public access while retaining the organization Visual."""
        return Visual.from_wire(
            await self._transport.execute(
                _request("unpublish_visual", f"/smr/visuals/{visual_id}/unpublish")
            )
        )

    async def retrieve_content(self, visual_id: ArtifactId) -> bytes:
        """Download organization-authorized self-contained HTML."""
        return await self._transport.request_bytes(
            "GET",
            f"/smr/visuals/{visual_id}/content",
            operation_id="retrieve_visual_content",
        )

    async def retrieve_public(self, slug: str) -> Visual:
        """Retrieve the safe public projection for a promoted slug."""
        normalized_slug = _slug(slug)
        return Visual.from_wire(
            await self._transport.execute(
                _request(
                    "retrieve_public_visual",
                    f"/smr/public/visuals/{normalized_slug}",
                )
            )
        )

    async def retrieve_public_content(self, slug: str) -> bytes:
        """Download public self-contained HTML for a promoted slug."""
        normalized_slug = _slug(slug)
        return await self._transport.request_bytes(
            "GET",
            f"/smr/public/visuals/{normalized_slug}/content",
            operation_id="retrieve_public_visual_content",
        )

    async def retrieve_public_preview(self, slug: str) -> bytes:
        """Download the service-authored public preview image."""
        normalized_slug = _slug(slug)
        return await self._transport.request_bytes(
            "GET",
            f"/smr/public/visuals/{normalized_slug}/preview",
            operation_id="retrieve_public_visual_preview",
        )


def _slug(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("slug must not be empty")
    return normalized


ResearchVisualsAPI = VisualsAPI
AsyncResearchVisualsAPI = AsyncVisualsAPI


__all__ = [
    "AsyncResearchVisualsAPI",
    "AsyncVisualsAPI",
    "ResearchVisualsAPI",
    "VisualsAPI",
]
