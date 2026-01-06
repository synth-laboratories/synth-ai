"""Ontology API client - access public ontology data from the Synth backend.

This module provides a typed client for querying the organization's ontology
graph data. Only public data is accessible through this API.

Example:
    from synth_ai.sdk.api.ontology import OntologyClient

    async with OntologyClient() as client:
        # List all public nodes
        nodes = await client.list_nodes()

        # Get a specific node with context
        context = await client.get_node_context("PlayerCharacter")

        # Get relationships
        rels = await client.get_relationships_from("PlayerCharacter")
"""

from .client import (
    OntologyClient,
    OntologyNode,
    PropertyClaim,
    Relationship,
    NodeContext,
    Neighborhood,
)

__all__ = [
    "OntologyClient",
    "OntologyNode",
    "PropertyClaim",
    "Relationship",
    "NodeContext",
    "Neighborhood",
]
