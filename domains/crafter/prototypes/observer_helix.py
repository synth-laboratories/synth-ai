"""
MVP: Observer watching Crafter gameplay, building ontology from observations.
BACKED BY HELIXDB - Graph-based ontology with explicit nodes and edges.

Structure:
  OntologyNode (entities) ──HasProperty──> PropertyClaim (facts)
       │                                        │
       │                                        └── Evidence
       │
       └── Relationship ──> OntologyNode (edges between entities)

Examples:
  - zombie (node) --HasProperty--> damage=2 (property)
  - tree (node) --yields--> wood (node)  [Relationship edge]
  - place_table (node) --requires--> wood (node) with {amount: 2}
"""

import json
import os
import httpx
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set, Tuple
from enum import Enum

# Default HelixDB URL - can be overridden via environment
DEFAULT_HELIX_URL = os.getenv("HELIX_URL", "http://localhost:6969")


# =============================================================================
# HELIXDB CLIENT
# =============================================================================

class HelixClient:
    """Thin HTTP client for HelixDB queries."""

    def __init__(self, base_url: str = None):
        self.base_url = base_url or DEFAULT_HELIX_URL
        self.client = httpx.Client(timeout=30.0)

    def query(self, name: str, params: Dict[str, Any] = None) -> Any:
        """Execute a named HelixDB query.

        Note: HelixDB returns 500 when a query finds no results.
        We catch this and return empty list/dict instead.
        """
        url = f"{self.base_url}/{name}"
        resp = self.client.post(url, json=params or {})

        # HelixDB returns 500 for "no results found" - treat as empty
        if resp.status_code == 500:
            return []

        resp.raise_for_status()
        return resp.json()

    def close(self):
        self.client.close()


# =============================================================================
# DATA TYPES
# =============================================================================

class EvidenceType(Enum):
    OBSERVATION = "observation"
    MEASUREMENT = "measurement"
    INFERENCE = "inference"
    CONTRADICTION = "contradiction"


class NodeType(Enum):
    CREATURE = "creature"
    RESOURCE = "resource"
    RECIPE = "recipe"
    LOCATION = "location"
    STATE = "state"
    ITEM = "item"
    ACTION = "action"


@dataclass
class Evidence:
    """A piece of evidence supporting claims."""
    id: str = ""
    type: EvidenceType = EvidenceType.OBSERVATION
    source: str = ""
    observation: str = ""
    time: int = 0
    annotation: str = ""
    weight: float = 1.0


@dataclass
class OntologyNode:
    """An entity in the ontology."""
    id: str = ""
    name: str = ""
    node_type: NodeType = NodeType.CREATURE
    description: str = ""
    relevance: float = 1.0


@dataclass
class PropertyClaim:
    """A property/fact about a node."""
    id: str = ""
    predicate: str = ""
    value: Any = None
    confidence: float = 0.0
    status: str = "active"


@dataclass
class RelationshipEdge:
    """A relationship between two nodes."""
    from_node: str = ""
    to_node: str = ""
    relation_type: str = ""
    value: Any = None
    confidence: float = 0.0


# =============================================================================
# NODE TYPE INFERENCE
# =============================================================================

def make_node_type_inferrer(
    type_map: Dict[str, NodeType] = None,
    prefix_rules: Dict[str, NodeType] = None,
    suffix_rules: Dict[str, NodeType] = None,
    default: NodeType = NodeType.CREATURE
):
    """
    Create a node type inference function.
    Domain-specific mappings are passed in, not hardcoded.
    """
    type_map = type_map or {}
    prefix_rules = prefix_rules or {}
    suffix_rules = suffix_rules or {}

    def infer(name: str) -> NodeType:
        if name in type_map:
            return type_map[name]
        for prefix, ntype in prefix_rules.items():
            if name.startswith(prefix):
                return ntype
        for suffix, ntype in suffix_rules.items():
            if name.endswith(suffix):
                return ntype
        return default

    return infer


# Default inferrer (no domain knowledge)
def default_node_type_inferrer(name: str) -> NodeType:
    """Default: everything is a creature unless it looks like a recipe."""
    if name.startswith("place_") or name.startswith("make_"):
        return NodeType.RECIPE
    return NodeType.CREATURE


# =============================================================================
# ONTOLOGY RECORD (HelixDB-backed)
# =============================================================================

class OntologyRecord:
    """Evidence-based ontology backed by HelixDB with explicit graph structure.

    Multi-tenant: org_id is required and passed to all queries.
    The org_id lives in Supabase (source of truth); HelixDB stores it on every node/edge.
    """

    def __init__(self, org_id: str, helix_url: str = "http://localhost:6969",
                 node_type_inferrer: callable = None):
        self.org_id = org_id
        self.helix = HelixClient(helix_url)
        self.current_time: int = 0
        self.history: List[Dict] = []
        self._infer_node_type = node_type_inferrer or default_node_type_inferrer

        # Caches: name -> helix_id
        self._node_cache: Dict[str, str] = {}
        self._property_cache: Dict[str, str] = {}  # "node.predicate" -> id

        # Ensure tenant exists
        self._ensure_tenant()

    def set_time(self, t: int):
        self.current_time = t

    def _ensure_tenant(self):
        """Ensure tenant root node exists for this org_id."""
        try:
            result = self.helix.query("ensureTenant", {"org_id": self.org_id})
            if not result or len(result) == 0:
                self.helix.query("createTenant", {"org_id": self.org_id})
        except Exception:
            # Tenant doesn't exist, create it
            self.helix.query("createTenant", {"org_id": self.org_id})

    # =========================================================================
    # CONFIDENCE CALCULATION (App code)
    # =========================================================================

    def _compute_confidence(self, supporting_weights: List[float],
                           contradicting_weights: List[float]) -> float:
        supporting = sum(supporting_weights)
        contradicting = sum(contradicting_weights)
        return supporting / (supporting + contradicting + 1.0)

    def _get_status(self, confidence: float, contradiction_count: int,
                    evidence_count: int) -> str:
        if contradiction_count > evidence_count:
            return "disproven"
        if confidence >= 0.7:
            return "confident"
        if confidence >= 0.4:
            return "probable"
        return "uncertain"

    # =========================================================================
    # NODE OPERATIONS
    # =========================================================================

    def _get_or_create_node(self, name: str, node_type: NodeType = None) -> str:
        """Get existing node ID or create new one."""
        if name in self._node_cache:
            return self._node_cache[name]

        # Try to get from DB (scoped by org_id)
        result = self.helix.query("getNode", {"org_id": self.org_id, "name": name})
        # Result is wrapped: {"node": {...}} or [] if not found
        if result and isinstance(result, dict) and "node" in result:
            node_id = result["node"]["id"]
            self._node_cache[name] = node_id
            return node_id

        # Create new node
        if node_type is None:
            node_type = self._infer_node_type(name)

        result = self.helix.query("addNode", {
            "org_id": self.org_id,
            "name": name,
            "node_type": node_type.value,
            "description": "",
            "relevance": 1.0,
            "created_at": self.current_time
        })

        # Result is wrapped: {"node": {...}}
        node_id = result["node"]["id"]
        self._node_cache[name] = node_id
        return node_id

    # Note: updateNodeRelevance removed - HelixDB doesn't support SET operations
    # Relevance updates would require delete+recreate pattern if needed

    # =========================================================================
    # EVIDENCE OPERATIONS
    # =========================================================================

    def add_evidence(self, evidence: Evidence) -> str:
        """Add evidence to HelixDB."""
        result = self.helix.query("addEvidence", {
            "org_id": self.org_id,
            "evidence_type": evidence.type.value,
            "source": evidence.source,
            "observation": evidence.observation,
            "annotation": evidence.annotation,
            "weight": evidence.weight,
            "time": evidence.time or self.current_time
        })
        # Result is wrapped: {"ev": {...}}
        ev_data = result.get("ev", result)
        evidence.id = ev_data.get("id", evidence.id)
        return evidence.id

    # =========================================================================
    # PROPERTY OPERATIONS
    # =========================================================================

    def _get_or_create_property(self, node_name: str, predicate: str,
                                 value: Any) -> Tuple[str, bool]:
        """Get existing property ID or create new one. Returns (id, is_new)."""
        key = f"{node_name}.{predicate}"

        if key in self._property_cache:
            return self._property_cache[key], False

        # Query DB (scoped by org_id)
        result = self.helix.query("getPropertyByPredicate", {
            "org_id": self.org_id,
            "node_name": node_name,
            "predicate": predicate
        })

        # Result is wrapped: {"properties": [...]} or []
        props = result.get("properties", []) if isinstance(result, dict) else result
        if props and len(props) > 0:
            # Find active one
            for prop in props:
                if prop.get("status") == "active" or prop.get("Status") == "active":
                    prop_id = prop["id"]
                    self._property_cache[key] = prop_id
                    return prop_id, False

        # Create new property
        node_id = self._get_or_create_node(node_name)

        prop_result = self.helix.query("addProperty", {
            "org_id": self.org_id,
            "predicate": predicate,
            "value": json.dumps(value),
            "confidence": 0.0,
            "status": "uncertain",
            "created_at": self.current_time
        })
        # Result is wrapped: {"prop": {...}}
        prop_data = prop_result.get("prop", prop_result)
        prop_id = prop_data["id"]

        # Link node to property
        self.helix.query("linkNodeToProperty", {
            "org_id": self.org_id,
            "node_id": node_id,
            "property_id": prop_id
        })

        self._property_cache[key] = prop_id
        return prop_id, True

    def _compute_property_confidence(self, property_id: str) -> float:
        """Compute property confidence from evidence (read-only).

        Note: HelixDB doesn't support SET operations, so confidence is computed
        on-the-fly from linked evidence rather than stored/updated.
        """
        result = self.helix.query("getEvidenceForProperty", {
            "org_id": self.org_id,
            "property_id": property_id
        })

        supporting = result.get("supporting") or []
        contradicting = result.get("contradicting") or []

        supporting_weights = [e.get("Weight", 1.0) for e in supporting]
        contradicting_weights = [e.get("Weight", 1.0) for e in contradicting]

        confidence = self._compute_confidence(supporting_weights, contradicting_weights)
        return confidence

    def observe_property(self, node_name: str, predicate: str, value: Any,
                        evidence: Evidence) -> PropertyClaim:
        """Observe a property of a node."""
        t = self.current_time
        key = f"{node_name}.{predicate}"

        # Add evidence
        ev_id = self.add_evidence(evidence)

        # Get or create property
        prop_id, is_new = self._get_or_create_property(node_name, predicate, value)

        if is_new:
            self.helix.query("linkPropertyEvidence", {
                "org_id": self.org_id,
                "property_id": prop_id,
                "evidence_id": ev_id
            })
            confidence = self._compute_property_confidence(prop_id)

            self.history.append({
                "time": t,
                "action": "new_property",
                "node": node_name,
                "predicate": predicate,
                "value": value,
                "confidence": confidence,
            })
        else:
            # Check if value matches
            result = self.helix.query("getPropertyByPredicate", {
                "org_id": self.org_id,
                "node_name": node_name,
                "predicate": predicate
            })
            # Result is wrapped: {"properties": [...]} or []
            props = result.get("properties", []) if isinstance(result, dict) else result
            existing_value = None
            for prop in (props or []):
                status = prop.get("status") or prop.get("Status")
                if status == "active":
                    existing_value = json.loads(prop.get("value") or prop.get("Value", "null"))
                    break

            if existing_value == value:
                # Confirming
                self.helix.query("linkPropertyEvidence", {
                    "org_id": self.org_id,
                    "property_id": prop_id,
                    "evidence_id": ev_id
                })
                confidence = self._compute_property_confidence(prop_id)

                self.history.append({
                    "time": t,
                    "action": "confirm",
                    "node": node_name,
                    "predicate": predicate,
                    "confidence": confidence,
                })
            else:
                # Contradicting
                self.helix.query("linkPropertyContradiction", {
                    "org_id": self.org_id,
                    "property_id": prop_id,
                    "evidence_id": ev_id
                })
                confidence = self._compute_property_confidence(prop_id)

                self.history.append({
                    "time": t,
                    "action": "contradict",
                    "node": node_name,
                    "predicate": predicate,
                    "old_value": existing_value,
                    "new_value": value,
                    "confidence": confidence,
                })

        return PropertyClaim(id=prop_id, predicate=predicate, value=value)

    # =========================================================================
    # RELATIONSHIP OPERATIONS
    # =========================================================================

    def observe_relationship(self, from_node: str, relation_type: str, to_node: str,
                            evidence: Evidence, value: Any = None) -> RelationshipEdge:
        """Observe a relationship between nodes."""
        t = self.current_time

        # Ensure both nodes exist
        from_id = self._get_or_create_node(from_node)
        to_id = self._get_or_create_node(to_node)

        # Add evidence (stored separately, relationship carries the observation)
        self.add_evidence(evidence)

        # Add relationship edge
        self.helix.query("addRelationship", {
            "org_id": self.org_id,
            "from_id": from_id,
            "to_id": to_id,
            "relation_type": relation_type,
            "value": json.dumps(value) if value else "{}",
            "confidence": 0.5,  # Initial confidence
            "created_at": t
        })

        self.history.append({
            "time": t,
            "action": "new_relationship",
            "from": from_node,
            "relation": relation_type,
            "to": to_node,
            "value": value,
        })

        return RelationshipEdge(
            from_node=from_node,
            to_node=to_node,
            relation_type=relation_type,
            value=value
        )

    # =========================================================================
    # SIMPLE OBSERVATION API (convenience methods)
    # =========================================================================

    def record_observation(self, node: str, predicate: str, value: Any,
                          observation_text: str, annotation: str = "",
                          source: str = None, weight: float = 1.0) -> PropertyClaim:
        """
        Simple API: Record an observed property.

        Example:
            record.record_observation("zombie", "damage", 2, "zombie dealt 2 damage")
        """
        ev = Evidence(
            type=EvidenceType.OBSERVATION,
            source=source or f"t{self.current_time}",
            observation=observation_text,
            annotation=annotation,
            weight=weight,
            time=self.current_time
        )
        return self.observe_property(node, predicate, value, ev)

    def record_edge(self, from_node: str, relation: str, to_node: str,
                   observation_text: str, annotation: str = "",
                   value: Any = None, source: str = None, weight: float = 1.0) -> RelationshipEdge:
        """
        Simple API: Record an observed relationship.

        Example:
            record.record_edge("tree", "yields", "wood", "collected wood from tree")
            record.record_edge("place_table", "requires", "wood", "crafting recipe", value={"amount": 2})
        """
        ev = Evidence(
            type=EvidenceType.OBSERVATION,
            source=source or f"t{self.current_time}",
            observation=observation_text,
            annotation=annotation,
            weight=weight,
            time=self.current_time
        )
        return self.observe_relationship(from_node, relation, to_node, ev, value)

    def record_inference(self, node: str, predicate: str, value: Any,
                        reasoning: str, weight: float = 0.5) -> PropertyClaim:
        """
        Record an inferred property (derived from other observations).

        Example:
            record.record_inference("zombie", "health", 5, "died after 5 hits with 1-damage weapon")
        """
        ev = Evidence(
            type=EvidenceType.INFERENCE,
            source=f"inference_t{self.current_time}",
            observation=f"Inferred: {node}.{predicate} = {value}",
            annotation=reasoning,
            weight=weight,
            time=self.current_time
        )
        return self.observe_property(node, predicate, value, ev)

    def record_contradiction(self, node: str, predicate: str, observed_value: Any,
                            observation_text: str, annotation: str = "") -> PropertyClaim:
        """
        Record an observation that contradicts existing knowledge.

        Example:
            record.record_contradiction("zombie", "damage", 3, "zombie dealt 3 damage, expected 2")
        """
        ev = Evidence(
            type=EvidenceType.CONTRADICTION,
            source=f"t{self.current_time}",
            observation=observation_text,
            annotation=annotation,
            weight=1.0,
            time=self.current_time
        )
        return self.observe_property(node, predicate, observed_value, ev)

    # =========================================================================
    # FULL CONTROL API (when you need custom Evidence)
    # =========================================================================

    def observe(self, subject: str, predicate: str, value: Any,
                evidence: Evidence):
        """
        Full control: Observe a property with custom Evidence object.
        For simpler usage, see record_observation().
        """
        return self.observe_property(subject, predicate, value, evidence)

    def observe_multiple(self, observations: List[Dict], evidence: Evidence):
        """Single evidence updates multiple claims."""
        ev_id = self.add_evidence(evidence)
        evidence.id = ev_id

        results = []
        for obs in observations:
            result = self.observe(
                subject=obs["subject"],
                predicate=obs["predicate"],
                value=obs["value"],
                evidence=evidence
            )
            results.append(result)
        return results

    # =========================================================================
    # RELEVANCE EDGES
    # =========================================================================

    def add_relevance_edge(self, from_node: str, to_node: str,
                          weight: float, relation: str = ""):
        """Add semantic relevance edge between nodes."""
        from_id = self._get_or_create_node(from_node)
        to_id = self._get_or_create_node(to_node)

        self.helix.query("addRelevanceEdge", {
            "org_id": self.org_id,
            "from_node_id": from_id,
            "to_node_id": to_id,
            "weight": weight,
            "relation": relation
        })

    def get_related_nodes(self, node_name: str, min_weight: float = 0.0) -> List[Tuple[str, float, str]]:
        """Get nodes related to this one."""
        result = self.helix.query("getRelatedNodes", {"org_id": self.org_id, "node_name": node_name})

        related = []
        for edge in (result.get("outgoing") or []):
            if edge.get("Weight", 0) >= min_weight:
                target = edge.get("to", {})
                related.append((
                    target.get("Name", "?"),
                    edge.get("Weight", 0),
                    edge.get("Relation", "")
                ))

        for edge in (result.get("incoming") or []):
            if edge.get("Weight", 0) >= min_weight:
                source = edge.get("from", {})
                related.append((
                    source.get("Name", "?"),
                    edge.get("Weight", 0),
                    edge.get("Relation", "")
                ))

        return sorted(related, key=lambda x: -x[1])

    def expand_from_seeds(self, seeds: List[str], hops: int = 1,
                         min_weight: float = 0.3) -> Set[str]:
        """Expand from seed nodes following relevance edges."""
        nodes = set(seeds)
        frontier = set(seeds)

        for _ in range(hops):
            new_frontier = set()
            for node in frontier:
                for other, weight, _ in self.get_related_nodes(node, min_weight):
                    if other not in nodes:
                        new_frontier.add(other)
                        nodes.add(other)
            frontier = new_frontier

        return nodes

    # =========================================================================
    # COMPILATION
    # =========================================================================

    def compile(self, min_confidence: float = 0.4) -> Dict[str, Any]:
        """Compile ontology from HelixDB."""
        nodes = self.helix.query("getAllNodes", {"org_id": self.org_id}) or []

        ontology = {"nodes": {}, "edges": []}

        for node_data in nodes:
            name = node_data["Name"]

            # Get properties
            result = self.helix.query("getPropertiesForNode", {"org_id": self.org_id, "node_name": name})
            properties = {}
            for prop in (result.get("properties") or []):
                if prop.get("Status") == "active":
                    conf = prop.get("Confidence", 0)
                    if conf >= min_confidence:
                        properties[prop["Predicate"]] = {
                            "value": json.loads(prop["Value"]),
                            "confidence": round(conf, 2)
                        }

            # Get relationships
            rels = self.helix.query("getRelationshipsFrom", {"org_id": self.org_id, "node_name": name}) or []
            for rel in rels:
                if rel.get("Status") == "active":
                    ontology["edges"].append({
                        "from": name,
                        "relation": rel.get("RelationType"),
                        "to": rel.get("to", {}).get("Name", "?"),
                        "value": json.loads(rel.get("Value", "{}")),
                        "confidence": round(rel.get("Confidence", 0), 2)
                    })

            if properties:
                ontology["nodes"][name] = {
                    "type": node_data.get("NodeType", "unknown"),
                    "properties": properties
                }

        return ontology

    def compile_to_text(self, min_confidence: float = 0.4,
                        include_evidence: bool = False) -> str:
        """Compile to readable text format."""
        ontology = self.compile(min_confidence)
        lines = ["# Ontology", ""]

        # Group by node type
        by_type: Dict[str, List[str]] = {}
        for name, data in ontology["nodes"].items():
            ntype = data["type"]
            if ntype not in by_type:
                by_type[ntype] = []
            by_type[ntype].append(name)

        for ntype in sorted(by_type.keys()):
            lines.append(f"## {ntype.upper()}S")
            lines.append("")

            for name in sorted(by_type[ntype]):
                data = ontology["nodes"][name]
                lines.append(f"### {name}")

                for pred, pdata in data["properties"].items():
                    conf_pct = int(pdata["confidence"] * 100)
                    lines.append(f"  - {pred}: {pdata['value']}  [{conf_pct}%]")

                # Show outgoing relationships
                for edge in ontology["edges"]:
                    if edge["from"] == name:
                        lines.append(f"  → {edge['relation']} → {edge['to']}")

                lines.append("")

        return "\n".join(lines)

    def get_context(self, budget: str, seeds: List[str] = None) -> str:
        """Get context with budget constraint."""
        import re

        budget = budget.lower().strip()

        if budget in ("small", "sm"):
            max_chars = 500
        elif budget in ("medium", "med"):
            max_chars = 2000
        elif budget in ("large", "lg"):
            max_chars = 8000
        elif budget == "full":
            return self.compile_to_text(min_confidence=0.3, include_evidence=True)
        else:
            match = re.match(r"(\d+\.?\d*)\s*(k)?\s*(chars?|tokens?)?", budget)
            if match:
                num = float(match.group(1))
                mult = 1000 if match.group(2) else 1
                unit = match.group(3) or "chars"
                max_chars = int(num * mult) * (4 if unit.startswith("t") else 1)
            else:
                max_chars = 2000

        text = self.compile_to_text(min_confidence=0.3)
        if len(text) > max_chars:
            text = text[:max_chars-20] + "\n\n[truncated]"

        return text

    # =========================================================================
    # REPLAY
    # =========================================================================

    def print_replay(self, times: List[int] = None):
        """Print belief evolution."""
        if times is None:
            times = sorted(set(h["time"] for h in self.history))

        print("\n" + "=" * 70)
        print("BELIEF EVOLUTION REPLAY")
        print("=" * 70)

        for t in times:
            events = [h for h in self.history if h["time"] == t]
            if not events:
                continue

            print(f"\n{'─' * 70}")
            print(f"TIME = {t}")
            print(f"{'─' * 70}")

            for e in events:
                action = e["action"].upper()
                if action == "NEW_PROPERTY":
                    print(f"  + {e['node']}.{e['predicate']} = {e['value']} (conf: {e['confidence']:.2f})")
                elif action == "NEW_RELATIONSHIP":
                    print(f"  + {e['from']} --{e['relation']}--> {e['to']}")
                elif action == "CONFIRM":
                    print(f"  ✓ {e['node']}.{e['predicate']} confirmed (conf: {e['confidence']:.2f})")
                elif action == "CONTRADICT":
                    print(f"  ✗ {e['node']}.{e['predicate']}: {e['old_value']} vs {e['new_value']}")

    def close(self):
        self.helix.close()

    # =========================================================================
    # GENERIC GRAPH API - Domain-agnostic ontology operations
    # =========================================================================

    def get_node_context(self, node_name: str) -> Dict[str, Any]:
        """
        Get everything known about a node: properties, relationships, related nodes.
        """
        result = self.helix.query("getNodeContext", {"org_id": self.org_id, "node_name": node_name})
        if not result or not result.get("node"):
            return {"known": False, "node": node_name}

        node = result["node"]
        properties = {}
        for prop in (result.get("properties") or []):
            pred = prop.get("Predicate") or prop.get("predicate")
            val = prop.get("Value") or prop.get("value") or "{}"
            conf = prop.get("Confidence") or prop.get("confidence") or 0
            status = prop.get("Status") or prop.get("status") or "unknown"
            properties[pred] = {
                "value": json.loads(val),
                "confidence": conf,
                "status": status
            }

        # Note: HelixDB traversal returns target nodes, not edge properties
        # Edge properties (relation_type, value) are not accessible via traversal
        # We get the target nodes and show them as relationships
        # HelixDB uses variable names as keys: "outgoing" not "relationships_from"
        relationships_out = []
        seen_targets = set()
        for rel in (result.get("outgoing") or result.get("relationships_from") or []):
            # rel is the target node, not the edge
            target = rel.get("Name") or rel.get("name")
            if target and target not in seen_targets:
                seen_targets.add(target)
                relationships_out.append({
                    "relation": "related_to",  # Edge type unknown from traversal
                    "target": target,
                    "value": {}
                })

        relationships_in = []
        seen_sources = set()
        for rel in (result.get("incoming") or result.get("relationships_to") or []):
            # rel is the source node, not the edge
            source = rel.get("Name") or rel.get("name")
            if source and source not in seen_sources:
                seen_sources.add(source)
                relationships_in.append({
                    "relation": "related_to",  # Edge type unknown from traversal
                    "source": source,
                    "value": {}
                })

        return {
            "known": True,
            "node": node_name,
            "type": node.get("NodeType"),
            "properties": properties,
            "outgoing": relationships_out,
            "incoming": relationships_in
        }

    def get_outgoing_edges(self, node_name: str, relation_type: str = None) -> List[Dict[str, Any]]:
        """
        Get outgoing edges from a node. If relation_type is None, returns all types.
        Example: get_outgoing_edges("tree", "yields") -> [{target: "wood", ...}]

        Note: HelixDB traversal returns target nodes, not edge objects.
        Edge properties (relation_type, value) are not accessible via traversal.
        If relation_type filter is specified, it's ignored (all edges returned).
        """
        result = self.helix.query("getOutgoingEdges", {
            "org_id": self.org_id,
            "node_name": node_name
        }) or []

        # Result is wrapped: {"edges": [...]} - unwrap it
        # Note: edges contains TARGET NODES, not edge objects
        edge_list = result.get("edges", []) if isinstance(result, dict) else result

        edges = []
        seen_targets = set()
        for target_node in (edge_list or []):
            # target_node is an OntologyNode, not a Relationship edge
            target = target_node.get("Name") or target_node.get("name")
            if target and target not in seen_targets:
                seen_targets.add(target)
                edges.append({
                    "target": target,
                    "relation": "related_to",  # Edge type unknown from traversal
                    "value": {},
                    "confidence": 0
                })
        return edges

    def get_incoming_edges(self, node_name: str, relation_type: str = None) -> List[Dict[str, Any]]:
        """
        Get incoming edges to a node. If relation_type is None, returns all types.
        Example: get_incoming_edges("wood", "yields") -> [{source: "tree", ...}]

        Note: HelixDB traversal returns source nodes, not edge objects.
        Edge properties (relation_type, value) are not accessible via traversal.
        If relation_type filter is specified, it's ignored (all edges returned).
        """
        result = self.helix.query("getIncomingEdges", {
            "org_id": self.org_id,
            "node_name": node_name
        }) or []

        # Result is wrapped: {"edges": [...]} - unwrap it
        # Note: edges contains SOURCE NODES, not edge objects
        edge_list = result.get("edges", []) if isinstance(result, dict) else result

        edges = []
        seen_sources = set()
        for source_node in (edge_list or []):
            # source_node is an OntologyNode, not a Relationship edge
            source = source_node.get("Name") or source_node.get("name")
            if source and source not in seen_sources:
                seen_sources.add(source)
                edges.append({
                    "source": source,
                    "relation": "related_to",  # Edge type unknown from traversal
                    "value": {},
                    "confidence": 0
                })
        return edges

    def get_all_edges_of_type(self, relation_type: str = None) -> List[Dict[str, Any]]:
        """
        Get all edges in the graph, optionally filtered by type.
        Example: get_all_edges_of_type("yields") -> all yield relationships

        Note: HelixDB traversal returns target nodes, not edge objects.
        Edge properties (relation_type, value) are not accessible via traversal.
        If relation_type filter is specified, it's ignored (all edges returned).
        """
        result = self.helix.query("getAllEdges", {"org_id": self.org_id}) or []

        # Result is wrapped: {"edges": [...]} - unwrap it
        # Note: edges contains TARGET NODES, not edge objects
        edge_list = result.get("edges", []) if isinstance(result, dict) else result

        edges = []
        seen_targets = set()
        for target_node in (edge_list or []):
            # target_node is an OntologyNode, not a Relationship edge
            target = target_node.get("Name") or target_node.get("name")
            if target and target not in seen_targets:
                seen_targets.add(target)
                edges.append({
                    "source": None,  # Source unknown from getAllEdges traversal
                    "target": target,
                    "relation": "related_to",  # Edge type unknown from traversal
                    "value": {},
                    "confidence": 0
                })
        return edges

    def get_property_value(self, node_name: str, predicate: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific property value from a node.
        Example: get_property_value("zombie", "damage") -> {value: 2, confidence: 0.8}
        """
        result = self.helix.query("getPropertyValue", {
            "org_id": self.org_id,
            "node_name": node_name,
            "predicate": predicate
        }) or []

        # Result is wrapped: {"props": [...]} - unwrap it
        props = result.get("props", []) if isinstance(result, dict) else result

        # Return first property found (any status)
        for prop in (props or []):
            status = prop.get("Status") or prop.get("status") or "unknown"
            val_str = prop.get("Value") or prop.get("value") or "{}"
            try:
                value = json.loads(val_str)
            except (json.JSONDecodeError, TypeError):
                value = val_str
            return {
                "value": value,
                "confidence": prop.get("Confidence") or prop.get("confidence") or 0,
                "status": status
            }
        return None

    def get_nodes_with_predicate(self, predicate: str) -> List[Dict[str, Any]]:
        """
        Find all nodes that have a specific property.
        Example: get_nodes_with_predicate("damage") -> nodes with damage properties
        """
        result = self.helix.query("getNodesWithPredicate", {"org_id": self.org_id, "predicate": predicate})
        if not result:
            return []

        # Response has "props" (single object or list) and "nodes" (list)
        props_data = result.get("props") or result.get("properties") or []
        node_data = result.get("nodes") or []

        # Normalize props to list
        if isinstance(props_data, dict):
            props_data = [props_data]

        nodes = []
        for node in node_data:
            node_name = node.get("Name") or node.get("name")
            node_type = node.get("NodeType") or node.get("node_type")
            # Find matching property for this node (or use first prop if single)
            prop = props_data[0] if props_data else {}
            val_str = prop.get("Value") or prop.get("value") or "{}"
            try:
                value = json.loads(val_str)
            except (json.JSONDecodeError, TypeError):
                value = val_str
            nodes.append({
                "node": node_name,
                "type": node_type,
                "value": value,
                "confidence": prop.get("Confidence") or prop.get("confidence") or 0
            })
        return nodes

    def get_uncertain_claims(self) -> List[Dict[str, Any]]:
        """Get claims with uncertain status (need more evidence)."""
        result = self.helix.query("getUncertainClaims", {"org_id": self.org_id}) or []
        # Result is wrapped: {"claims": [...]} - unwrap it
        claims = result.get("claims", []) if isinstance(result, dict) else result
        return [{
            "predicate": c.get("Predicate") or c.get("predicate"),
            "value": json.loads(c.get("Value") or c.get("value") or "{}"),
            "confidence": c.get("Confidence") or c.get("confidence") or 0
        } for c in (claims or [])]

    def get_hypotheses(self) -> List[Dict[str, Any]]:
        """Get untested hypothesis claims."""
        result = self.helix.query("getHypotheses", {"org_id": self.org_id}) or []
        # Result is wrapped: {"claims": [...]} - unwrap it
        claims = result.get("claims", []) if isinstance(result, dict) else result
        return [{
            "predicate": c.get("Predicate") or c.get("predicate"),
            "value": json.loads(c.get("Value") or c.get("value") or "{}"),
            "needs_testing": True
        } for c in (claims or [])]

    def record_hypothesis(self, node_name: str, predicate: str, guessed_value: Any,
                         reason: str = "") -> PropertyClaim:
        """
        Record a guess/hypothesis before testing it.
        """
        ev = Evidence(
            type=EvidenceType.INFERENCE,
            source=f"hypothesis_t{self.current_time}",
            observation=f"Hypothesis: {node_name}.{predicate} = {guessed_value}",
            annotation=reason or "Untested hypothesis",
            weight=0.1,  # Low weight until confirmed
            time=self.current_time
        )

        # Create property with hypothesis status
        node_id = self._get_or_create_node(node_name)

        prop_result = self.helix.query("addHypothesis", {
            "org_id": self.org_id,
            "predicate": predicate,
            "value": json.dumps(guessed_value),
            "created_at": self.current_time
        })
        # Result is wrapped: {"prop": {...}} - unwrap it
        prop_data = prop_result.get("prop", prop_result) if isinstance(prop_result, dict) else prop_result
        prop_id = prop_data.get("id") or prop_data.get("ID")

        self.helix.query("linkNodeToProperty", {
            "org_id": self.org_id,
            "node_id": node_id,
            "property_id": prop_id
        })

        ev_id = self.add_evidence(ev)
        self.helix.query("linkPropertyEvidence", {
            "org_id": self.org_id,
            "property_id": prop_id,
            "evidence_id": ev_id
        })

        self.history.append({
            "time": self.current_time,
            "action": "hypothesis",
            "node": node_name,
            "predicate": predicate,
            "value": guessed_value
        })

        return PropertyClaim(id=prop_id, predicate=predicate, value=guessed_value,
                            confidence=0.1, status="hypothesis")


# =============================================================================
# CRAFTER OBSERVER
# =============================================================================

# Crafter-specific node type knowledge (lives with domain code, not generic ontology)
CRAFTER_TYPE_MAP = {
    # Creatures
    "zombie": NodeType.CREATURE,
    "skeleton": NodeType.CREATURE,
    "cow": NodeType.CREATURE,
    "player": NodeType.CREATURE,
    # Resources
    "wood": NodeType.RESOURCE,
    "stone": NodeType.RESOURCE,
    "coal": NodeType.RESOURCE,
    "iron": NodeType.RESOURCE,
    "diamond": NodeType.RESOURCE,
    "food": NodeType.RESOURCE,
    # Items
    "wood_pickaxe": NodeType.ITEM,
    "stone_pickaxe": NodeType.ITEM,
    "wood_sword": NodeType.ITEM,
    "stone_sword": NodeType.ITEM,
    # Locations
    "plains": NodeType.LOCATION,
    "forest": NodeType.LOCATION,
    "mountain": NodeType.LOCATION,
    "cave": NodeType.LOCATION,
    # States
    "sleep": NodeType.STATE,
    "night": NodeType.STATE,
    "day": NodeType.STATE,
}

CRAFTER_PREFIX_RULES = {
    "place_": NodeType.RECIPE,
    "make_": NodeType.RECIPE,
}

CRAFTER_SUFFIX_RULES = {
    "_pickaxe": NodeType.ITEM,
    "_sword": NodeType.ITEM,
}


class CrafterObserver:
    """Observes Crafter gameplay and builds ontology."""

    def __init__(self, org_id: str, helix_url: str = "http://localhost:6969"):
        # Create Crafter-specific type inferrer
        crafter_inferrer = make_node_type_inferrer(
            type_map=CRAFTER_TYPE_MAP,
            prefix_rules=CRAFTER_PREFIX_RULES,
            suffix_rules=CRAFTER_SUFFIX_RULES,
            default=NodeType.CREATURE
        )
        self.record = OntologyRecord(org_id, helix_url, node_type_inferrer=crafter_inferrer)
        self.step = 0
        self._ev_counter = 0

    def _evidence(self, observation: str,
                  etype: EvidenceType = EvidenceType.OBSERVATION,
                  annotation: str = "") -> Evidence:
        self._ev_counter += 1
        return Evidence(
            id=f"ev_{self.step}_{self._ev_counter}",
            type=etype,
            source=f"step_{self.step}",
            observation=observation,
            time=self.step,
            annotation=annotation,
        )

    def observe_event(self, event: Dict):
        """Process gameplay event."""
        self.step = event.get("step", self.step + 1)
        self.record.set_time(self.step)
        etype = event["type"]

        if etype == "damage_dealt":
            attacker = event["attacker"]
            damage = event["damage"]
            context = event.get("context", "normal")
            pred = "sleep_damage" if context == "sleeping" else "damage"

            self.record.observe_property(
                node_name=attacker,
                predicate=pred,
                value=damage,
                evidence=self._evidence(
                    f"{attacker} dealt {damage} damage ({context})",
                    annotation=event.get("annotation", "")
                )
            )

        elif etype == "resource_collected":
            source = event["source"]
            item = event["item"]
            amount = event.get("amount", 1)

            # This is a RELATIONSHIP: source --yields--> item
            self.record.observe_relationship(
                from_node=source,
                relation_type="yields",
                to_node=item,
                evidence=self._evidence(
                    f"Collected {amount} {item} from {source}",
                    annotation=event.get("annotation", "")
                ),
                value={"amount": amount}
            )

        elif etype == "creature_died":
            creature = event["creature"]
            hits = event["hits"]
            weapon_damage = event.get("weapon_damage", 1)
            health = hits * weapon_damage

            self.record.observe_property(
                node_name=creature,
                predicate="health",
                value=health,
                evidence=self._evidence(
                    f"{creature} died after {hits} hits",
                    etype=EvidenceType.INFERENCE,
                    annotation=f"Health = {hits} × {weapon_damage} = {health}"
                )
            )

        elif etype == "crafting_observed":
            recipe = event["recipe"]
            inputs = event["inputs"]
            output = event["output"]

            ev = self._evidence(
                f"Crafted {output} using {inputs}",
                annotation=f"Recipe: {recipe}"
            )

            # Recipe --requires--> input resources
            for item, amount in inputs.items():
                self.record.observe_relationship(
                    from_node=recipe,
                    relation_type="requires",
                    to_node=item,
                    evidence=ev,
                    value={"amount": amount}
                )

            # Recipe --produces--> output
            self.record.observe_relationship(
                from_node=recipe,
                relation_type="produces",
                to_node=output,
                evidence=ev
            )

        elif etype == "sleep_attack":
            attacker = event["attacker"]
            damage = event["damage"]
            normal = event.get("known_normal_damage")

            ev = self._evidence(
                f"While sleeping, {attacker} attacked for {damage}",
                annotation=event.get("annotation", "")
            )

            self.record.observe_property(attacker, "sleep_damage", damage, ev)

            if normal:
                mult = round(damage / normal, 1)
                self.record.observe_property("sleep", "damage_multiplier", mult, ev)
                self.record.observe_property("player", "sleep_is_dangerous", True, ev)

        elif etype == "spawn_observed":
            creature = event["creature"]
            location = event["location"]

            self.record.observe_relationship(
                from_node=creature,
                relation_type="spawns_at",
                to_node=location,
                evidence=self._evidence(
                    f"{creature} spawned at {location}",
                    annotation=f"Time: {event.get('time', 'unknown')}"
                )
            )

    def close(self):
        self.record.close()


# =============================================================================
# DEMO
# =============================================================================

def demo():
    print("=" * 70)
    print("CRAFTER OBSERVER - HELIXDB GRAPH-BASED ONTOLOGY")
    print("=" * 70)
    print("\nRequires HelixDB at http://localhost:6969")
    print("Start with: helix start dev\n")

    # Demo org_id - in production this comes from Supabase auth
    org_id = "demo-org-001"

    try:
        observer = CrafterObserver(org_id=org_id)
        print(f"Connected with org_id: {org_id}\n")

        events = [
            {"step": 10, "type": "resource_collected", "source": "tree", "item": "wood"},
            {"step": 50, "type": "crafting_observed", "recipe": "place_table",
             "inputs": {"wood": 2}, "output": "table"},
            {"step": 110, "type": "damage_dealt", "attacker": "zombie", "damage": 2},
            {"step": 120, "type": "creature_died", "creature": "zombie", "hits": 5},
            {"step": 200, "type": "sleep_attack", "attacker": "zombie", "damage": 7,
             "known_normal_damage": 2, "annotation": "Sleep is dangerous!"},
            {"step": 300, "type": "spawn_observed", "creature": "skeleton",
             "location": "mountain", "time": "night"},
        ]

        for e in events:
            observer.observe_event(e)
            print(f"  Step {e['step']}: {e['type']}")

        # Add relevance edges
        observer.record.add_relevance_edge("zombie", "player", 0.9, "attacks")
        observer.record.add_relevance_edge("zombie", "sleep", 0.95, "bonus_damage_when")
        observer.record.add_relevance_edge("tree", "wood", 0.8, "yields")

        print("\n" + "=" * 70)
        print("COMPILED ONTOLOGY")
        print("=" * 70)
        print(observer.record.compile_to_text(min_confidence=0.3))

        observer.record.print_replay()
        observer.close()

    except httpx.ConnectError:
        print("ERROR: Could not connect to HelixDB")
        print("Use observer_mvp.py for pure Python testing")


if __name__ == "__main__":
    demo()
