"""Ontology client for the Synth AI SDK."""

from __future__ import annotations

from typing import Any, Mapping
from urllib.parse import quote

from synth_ai.core.utils.env import get_api_key, get_backend_url
from synth_ai.core.utils.urls import normalize_base_url
from synth_ai.sdk.shared import AsyncHttpClient


class OntologyClient:
    """Client for accessing the public ontology API via the Synth backend."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        timeout: float = 30.0,
    ) -> None:
        if base_url is None:
            base_url = get_backend_url()
        self._base_url = normalize_base_url(base_url)

        if api_key is None:
            api_key = get_api_key("SYNTH_API_KEY", required=True)
        self._api_key = api_key
        self._timeout = timeout

    @staticmethod
    def _normalize_params(params: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if not params:
            return None
        return dict(params)

    @staticmethod
    def _encode_node_name(node_name: str) -> str:
        if not node_name:
            raise ValueError("node_name is required")
        return quote(node_name, safe="")

    def _node_path(self, node_name: str, suffix: str | None = None) -> str:
        encoded = self._encode_node_name(node_name)
        if suffix:
            return f"/api/ontology/nodes/{encoded}/{suffix}"
        return f"/api/ontology/nodes/{encoded}"

    async def _get(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
        async with AsyncHttpClient(self._base_url, self._api_key, timeout=self._timeout) as http:
            return await http.get(path, params=self._normalize_params(params))

    async def list_nodes(self, *, params: Mapping[str, Any] | None = None) -> Any:
        """List public ontology nodes."""
        return await self._get("/api/ontology/nodes", params=params)

    async def get_node(self, node_name: str) -> Any:
        """Fetch a specific node by name."""
        return await self._get(self._node_path(node_name))

    async def get_node_context(self, node_name: str) -> Any:
        """Fetch the full context for a node."""
        return await self._get(self._node_path(node_name, "context"))

    async def get_node_neighborhood(self, node_name: str) -> Any:
        """Fetch the neighborhood for a node."""
        return await self._get(self._node_path(node_name, "neighborhood"))

    async def list_properties(self, *, params: Mapping[str, Any] | None = None) -> Any:
        """List public property claims."""
        return await self._get("/api/ontology/properties", params=params)

    async def get_node_properties(self, node_name: str) -> Any:
        """Fetch properties for a specific node."""
        return await self._get(self._node_path(node_name, "properties"))

    async def get_node_relationships_outgoing(self, node_name: str) -> Any:
        """Fetch outgoing relationships for a node."""
        return await self._get(self._node_path(node_name, "relationships/outgoing"))

    async def get_node_relationships_incoming(self, node_name: str) -> Any:
        """Fetch incoming relationships for a node."""
        return await self._get(self._node_path(node_name, "relationships/incoming"))

    async def health(self) -> Any:
        """Check ontology service health."""
        return await self._get("/api/ontology/health")


__all__ = [
    "OntologyClient",
]
