"""Ontology API Client.

Typed async client for accessing public ontology data from the Synth backend.
Routes through the backend API gateway (not directly to HelixDB).

Example:
    from synth_ai.sdk.api.ontology import OntologyClient

    # Using async context manager
    async with OntologyClient() as client:
        nodes = await client.list_nodes()
        for node in nodes:
            print(f"{node.name}: {node.description}")

    # Using explicit credentials
    client = OntologyClient(base_url="http://localhost:8000", api_key="sk_...")
    async with client:
        context = await client.get_node_context("Entity")
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from synth_ai.core.http import AsyncHttpClient
from synth_ai.core.env import get_backend_from_env


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class OntologyNode:
    """An ontology node (entity, concept, etc.)."""

    id: str
    name: str
    node_type: str
    description: str
    relevance: float
    created_at: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OntologyNode":
        """Create from API response dict."""
        return cls(
            id=str(data.get("id", "")),
            name=data.get("name", ""),
            node_type=data.get("node_type", ""),
            description=data.get("description", ""),
            relevance=float(data.get("relevance", 0.0)),
            created_at=int(data.get("created_at", 0)),
        )


@dataclass
class PropertyClaim:
    """A property claim attached to a node."""

    id: str
    predicate: str
    value: str
    confidence: float
    status: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PropertyClaim":
        """Create from API response dict."""
        return cls(
            id=str(data.get("id", "")),
            predicate=data.get("predicate", ""),
            value=data.get("value", ""),
            confidence=float(data.get("confidence", 0.0)),
            status=data.get("status", ""),
        )


@dataclass
class Relationship:
    """A relationship between two nodes."""

    id: str
    from_node_id: str
    to_node_id: str
    relation_type: str
    value: str
    confidence: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relationship":
        """Create from API response dict."""
        return cls(
            id=str(data.get("id", "")),
            from_node_id=str(data.get("from_node_id", "")),
            to_node_id=str(data.get("to_node_id", "")),
            relation_type=data.get("relation_type", ""),
            value=data.get("value", ""),
            confidence=float(data.get("confidence", 0.0)),
        )


@dataclass
class NodeContext:
    """Full context for a node (properties and relationships)."""

    node: Optional[OntologyNode] = None
    properties: List[PropertyClaim] = field(default_factory=list)
    relationships_from: List[Relationship] = field(default_factory=list)
    relationships_to: List[Relationship] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeContext":
        """Create from API response dict."""
        node_data = data.get("node")
        node = OntologyNode.from_dict(node_data) if node_data else None

        properties = [
            PropertyClaim.from_dict(p)
            for p in (data.get("properties") or [])
            if isinstance(p, dict)
        ]
        rels_from = [
            Relationship.from_dict(r)
            for r in (data.get("relationships_from") or [])
            if isinstance(r, dict)
        ]
        rels_to = [
            Relationship.from_dict(r)
            for r in (data.get("relationships_to") or [])
            if isinstance(r, dict)
        ]

        return cls(
            node=node,
            properties=properties,
            relationships_from=rels_from,
            relationships_to=rels_to,
        )


@dataclass
class Neighborhood:
    """Neighborhood of a node (outgoing and incoming relationships)."""

    outgoing: List[Relationship] = field(default_factory=list)
    incoming: List[Relationship] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Neighborhood":
        """Create from API response dict."""
        outgoing = [
            Relationship.from_dict(r)
            for r in (data.get("outgoing") or [])
            if isinstance(r, dict)
        ]
        incoming = [
            Relationship.from_dict(r)
            for r in (data.get("incoming") or [])
            if isinstance(r, dict)
        ]
        return cls(outgoing=outgoing, incoming=incoming)


# =============================================================================
# Client
# =============================================================================


class OntologyClient:
    """Async client for the Ontology API.

    Uses the backend API gateway to access public ontology data.
    Authentication is via the standard Synth API key.

    Example:
        async with OntologyClient() as client:
            nodes = await client.list_nodes()
            context = await client.get_node_context("Player")
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the ontology client.

        Args:
            base_url: Backend base URL (defaults to env-based resolution)
            api_key: Synth API key (defaults to env-based resolution)
            timeout: Request timeout in seconds
        """
        if base_url is None or api_key is None:
            env_url, env_key = get_backend_from_env()
            base_url = base_url or env_url
            api_key = api_key or env_key

        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._http: Optional[AsyncHttpClient] = None

    async def __aenter__(self) -> "OntologyClient":
        self._http = AsyncHttpClient(
            self._base_url,
            self._api_key,
            timeout=self._timeout,
        )
        await self._http.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._http is not None:
            await self._http.__aexit__(exc_type, exc, tb)
            self._http = None

    def _ensure_http(self) -> AsyncHttpClient:
        """Ensure HTTP client is initialized."""
        if self._http is None:
            raise RuntimeError(
                "OntologyClient must be used as async context manager: "
                "async with OntologyClient() as client: ..."
            )
        return self._http

    # -------------------------------------------------------------------------
    # Node Operations
    # -------------------------------------------------------------------------

    async def list_nodes(
        self, node_type: Optional[str] = None
    ) -> List[OntologyNode]:
        """List public ontology nodes.

        Args:
            node_type: Optional filter by node type

        Returns:
            List of public OntologyNode objects
        """
        http = self._ensure_http()
        params = {"node_type": node_type} if node_type else None
        data = await http.get("/api/ontology/nodes", params=params)

        if not isinstance(data, dict):
            return []

        nodes_data = data.get("nodes", [])
        return [
            OntologyNode.from_dict(n)
            for n in nodes_data
            if isinstance(n, dict)
        ]

    async def get_node(self, name: str) -> Optional[OntologyNode]:
        """Get a specific public node by name.

        Args:
            name: Node name

        Returns:
            OntologyNode if found, None otherwise
        """
        http = self._ensure_http()
        try:
            data = await http.get(f"/api/ontology/nodes/{name}")
            if isinstance(data, dict):
                return OntologyNode.from_dict(data)
        except Exception:
            pass
        return None

    async def get_node_context(self, name: str) -> NodeContext:
        """Get full context for a public node.

        Includes the node, its properties, and relationships.

        Args:
            name: Node name

        Returns:
            NodeContext with node data and related items
        """
        http = self._ensure_http()
        try:
            data = await http.get(f"/api/ontology/nodes/{name}/context")
            if isinstance(data, dict):
                return NodeContext.from_dict(data)
        except Exception:
            pass
        return NodeContext()

    async def get_neighborhood(self, name: str) -> Neighborhood:
        """Get the neighborhood of a public node.

        Returns outgoing and incoming relationships.

        Args:
            name: Node name

        Returns:
            Neighborhood with relationship lists
        """
        http = self._ensure_http()
        try:
            data = await http.get(f"/api/ontology/nodes/{name}/neighborhood")
            if isinstance(data, dict):
                return Neighborhood.from_dict(data)
        except Exception:
            pass
        return Neighborhood()

    # -------------------------------------------------------------------------
    # Property Operations
    # -------------------------------------------------------------------------

    async def list_properties(self) -> List[PropertyClaim]:
        """List all public active property claims.

        Returns:
            List of PropertyClaim objects
        """
        http = self._ensure_http()
        data = await http.get("/api/ontology/properties")

        if not isinstance(data, dict):
            return []

        props_data = data.get("properties", [])
        return [
            PropertyClaim.from_dict(p)
            for p in props_data
            if isinstance(p, dict)
        ]

    async def get_properties_for_node(self, name: str) -> List[PropertyClaim]:
        """Get public properties for a specific node.

        Args:
            name: Node name

        Returns:
            List of PropertyClaim objects
        """
        http = self._ensure_http()
        try:
            data = await http.get(f"/api/ontology/nodes/{name}/properties")
            if isinstance(data, dict):
                props_data = data.get("properties", [])
                return [
                    PropertyClaim.from_dict(p)
                    for p in props_data
                    if isinstance(p, dict)
                ]
        except Exception:
            pass
        return []

    # -------------------------------------------------------------------------
    # Relationship Operations
    # -------------------------------------------------------------------------

    async def get_relationships_from(self, name: str) -> List[Relationship]:
        """Get public outgoing relationships from a node.

        Args:
            name: Source node name

        Returns:
            List of Relationship objects
        """
        http = self._ensure_http()
        try:
            data = await http.get(
                f"/api/ontology/nodes/{name}/relationships/outgoing"
            )
            if isinstance(data, list):
                return [
                    Relationship.from_dict(r)
                    for r in data
                    if isinstance(r, dict)
                ]
        except Exception:
            pass
        return []

    async def get_relationships_to(self, name: str) -> List[Relationship]:
        """Get public incoming relationships to a node.

        Args:
            name: Target node name

        Returns:
            List of Relationship objects
        """
        http = self._ensure_http()
        try:
            data = await http.get(
                f"/api/ontology/nodes/{name}/relationships/incoming"
            )
            if isinstance(data, list):
                return [
                    Relationship.from_dict(r)
                    for r in data
                    if isinstance(r, dict)
                ]
        except Exception:
            pass
        return []

    # -------------------------------------------------------------------------
    # Health Check
    # -------------------------------------------------------------------------

    async def health(self) -> Dict[str, Any]:
        """Check ontology service health.

        Returns:
            Health status dict
        """
        http = self._ensure_http()
        try:
            data = await http.get("/api/ontology/health")
            return data if isinstance(data, dict) else {"status": "unknown"}
        except Exception as e:
            return {"status": "error", "error": str(e)}


# =============================================================================
# Sync Wrapper
# =============================================================================


def list_nodes_sync(
    node_type: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> List[OntologyNode]:
    """Synchronous wrapper for list_nodes.

    Args:
        node_type: Optional filter by node type
        base_url: Backend base URL
        api_key: Synth API key

    Returns:
        List of OntologyNode objects
    """
    async def _run():
        async with OntologyClient(base_url=base_url, api_key=api_key) as client:
            return await client.list_nodes(node_type=node_type)

    return asyncio.get_event_loop().run_until_complete(_run())


def get_node_context_sync(
    name: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> NodeContext:
    """Synchronous wrapper for get_node_context.

    Args:
        name: Node name
        base_url: Backend base URL
        api_key: Synth API key

    Returns:
        NodeContext with node data
    """
    async def _run():
        async with OntologyClient(base_url=base_url, api_key=api_key) as client:
            return await client.get_node_context(name)

    return asyncio.get_event_loop().run_until_complete(_run())
