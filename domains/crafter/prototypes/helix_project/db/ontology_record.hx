// =============================================================================
// ONTOLOGY RECORD - Multi-tenant Graph-Based Ontology Schema
// =============================================================================
//
// Architecture:
//   - org_id lives in Supabase (source of truth)
//   - HelixDB stores graph data with org_id on every node/edge
//   - API resolves org_id from auth, passes to all queries
//   - Tenant root node for fast org-scoped traversals
//
// Structure:
//   Tenant (org root) ──Owns──> OntologyNode ──HasProperty──> PropertyClaim
//                                    │                              │
//                                    │                              └── SupportedBy ──> Evidence
//                                    │
//                                    └── Relationship ──> OntologyNode
//

// -----------------------------------------------------------------------------
// TENANT ROOT (fast entrypoint for org-scoped queries)
// -----------------------------------------------------------------------------

N::Tenant {
    INDEX org_id: String
}

// -----------------------------------------------------------------------------
// CORE NODES
// -----------------------------------------------------------------------------

N::OntologyNode {
    INDEX org_id: String,
    INDEX name: String,
    node_type: String,
    description: String,
    relevance: F64,
    created_at: I64
}

N::PropertyClaim {
    INDEX org_id: String,
    INDEX predicate: String,
    value: String,
    confidence: F64,
    status: String,
    created_at: I64,
    updated_at: I64
}

N::Evidence {
    INDEX org_id: String,
    evidence_type: String,
    source: String,
    observation: String,
    annotation: String,
    weight: F64,
    time: I64
}

// -----------------------------------------------------------------------------
// EDGES
// -----------------------------------------------------------------------------

E::Owns {
    From: Tenant,
    To: OntologyNode,
    Properties: {
        org_id: String
    }
}

E::HasProperty {
    From: OntologyNode,
    To: PropertyClaim,
    Properties: {
        org_id: String
    }
}

E::Relationship {
    From: OntologyNode,
    To: OntologyNode,
    Properties: {
        org_id: String,
        relation_type: String,
        value: String,
        confidence: F64,
        status: String,
        created_at: I64
    }
}

E::SupportedBy {
    From: PropertyClaim,
    To: Evidence,
    Properties: {
        org_id: String
    }
}

E::ContradictedBy {
    From: PropertyClaim,
    To: Evidence,
    Properties: {
        org_id: String
    }
}

E::RelatedTo {
    From: OntologyNode,
    To: OntologyNode,
    Properties: {
        org_id: String,
        weight: F64,
        relation: String
    }
}

// =============================================================================
// QUERIES - Tenant Management
// =============================================================================

QUERY ensureTenant(org_id: String) =>
    t <- N<Tenant>({ org_id: org_id })
    RETURN t

QUERY createTenant(org_id: String) =>
    t <- AddN<Tenant>({ org_id: org_id })
    RETURN t

// =============================================================================
// QUERIES - Node Operations (all scoped by org_id)
// =============================================================================

QUERY addNode(org_id: String, name: String, node_type: String, description: String, relevance: F64, created_at: I64) =>
    tenant <- N<Tenant>({ org_id: org_id })
    node <- AddN<OntologyNode>({
        org_id: org_id,
        name: name,
        node_type: node_type,
        description: description,
        relevance: relevance,
        created_at: created_at
    })
    edge <- AddE<Owns>::From(tenant)::To(node)
    RETURN node

QUERY getNode(org_id: String, name: String) =>
    node <- N<OntologyNode>({ name: name })::WHERE(_::{org_id}::EQ(org_id))
    RETURN node

QUERY getNodesByType(org_id: String, node_type: String) =>
    nodes <- N<OntologyNode>::WHERE(_::{org_id}::EQ(org_id))::WHERE(_::{node_type}::EQ(node_type))
    RETURN nodes

QUERY getAllNodes(org_id: String) =>
    nodes <- N<OntologyNode>::WHERE(_::{org_id}::EQ(org_id))
    RETURN nodes

// =============================================================================
// QUERIES - Property Operations
// =============================================================================

QUERY addProperty(org_id: String, predicate: String, value: String, confidence: F64, status: String, created_at: I64) =>
    prop <- AddN<PropertyClaim>({
        org_id: org_id,
        predicate: predicate,
        value: value,
        confidence: confidence,
        status: status,
        created_at: created_at,
        updated_at: created_at
    })
    RETURN prop

QUERY linkNodeToProperty(org_id: String, node_id: ID, property_id: ID) =>
    edge <- AddE<HasProperty>::From(node_id)::To(property_id)
    RETURN edge

QUERY getPropertiesForNode(org_id: String, node_name: String) =>
    node <- N<OntologyNode>({ name: node_name })::WHERE(_::{org_id}::EQ(org_id))
    properties <- node::Out<HasProperty>
    RETURN { node: node, properties: properties }

QUERY getPropertyByPredicate(org_id: String, node_name: String, predicate: String) =>
    node <- N<OntologyNode>({ name: node_name })::WHERE(_::{org_id}::EQ(org_id))
    properties <- node::Out<HasProperty>::WHERE(_::{predicate}::EQ(predicate))
    RETURN properties

QUERY getActiveProperties(org_id: String) =>
    properties <- N<PropertyClaim>::WHERE(_::{org_id}::EQ(org_id))::WHERE(_::{status}::EQ("active"))
    RETURN properties

// =============================================================================
// QUERIES - Relationship Operations
// =============================================================================

QUERY addRelationship(org_id: String, from_id: ID, to_id: ID, relation_type: String, value: String, confidence: F64, created_at: I64) =>
    edge <- AddE<Relationship>({
        org_id: org_id,
        relation_type: relation_type,
        value: value,
        confidence: confidence,
        status: "active",
        created_at: created_at
    })::From(from_id)::To(to_id)
    RETURN edge

QUERY getRelationshipsFrom(org_id: String, node_name: String) =>
    node <- N<OntologyNode>({ name: node_name })::WHERE(_::{org_id}::EQ(org_id))
    relationships <- node::Out<Relationship>
    RETURN relationships

QUERY getRelationshipsTo(org_id: String, node_name: String) =>
    node <- N<OntologyNode>({ name: node_name })::WHERE(_::{org_id}::EQ(org_id))
    relationships <- node::In<Relationship>
    RETURN relationships

QUERY getOutgoingEdges(org_id: String, node_name: String) =>
    node <- N<OntologyNode>({ name: node_name })::WHERE(_::{org_id}::EQ(org_id))
    edges <- node::Out<Relationship>
    RETURN edges

QUERY getIncomingEdges(org_id: String, node_name: String) =>
    node <- N<OntologyNode>({ name: node_name })::WHERE(_::{org_id}::EQ(org_id))
    edges <- node::In<Relationship>
    RETURN edges

QUERY getAllEdges(org_id: String) =>
    nodes <- N<OntologyNode>::WHERE(_::{org_id}::EQ(org_id))
    edges <- nodes::Out<Relationship>
    RETURN edges

// =============================================================================
// QUERIES - Evidence Operations
// =============================================================================

QUERY addEvidence(org_id: String, evidence_type: String, source: String, observation: String, annotation: String, weight: F64, time: I64) =>
    ev <- AddN<Evidence>({
        org_id: org_id,
        evidence_type: evidence_type,
        source: source,
        observation: observation,
        annotation: annotation,
        weight: weight,
        time: time
    })
    RETURN ev

QUERY linkPropertyEvidence(org_id: String, property_id: ID, evidence_id: ID) =>
    edge <- AddE<SupportedBy>::From(property_id)::To(evidence_id)
    RETURN edge

QUERY linkPropertyContradiction(org_id: String, property_id: ID, evidence_id: ID) =>
    edge <- AddE<ContradictedBy>::From(property_id)::To(evidence_id)
    RETURN edge

QUERY getEvidenceForProperty(org_id: String, property_id: ID) =>
    prop <- N<PropertyClaim>::WHERE(_::ID::EQ(property_id))::WHERE(_::{org_id}::EQ(org_id))
    supporting <- prop::Out<SupportedBy>
    contradicting <- prop::Out<ContradictedBy>
    RETURN { supporting: supporting, contradicting: contradicting }

// =============================================================================
// QUERIES - Context / Full Graph
// =============================================================================

QUERY getNodeContext(org_id: String, node_name: String) =>
    node <- N<OntologyNode>({ name: node_name })::WHERE(_::{org_id}::EQ(org_id))
    properties <- node::Out<HasProperty>
    outgoing <- node::Out<Relationship>
    incoming <- node::In<Relationship>
    related <- node::Out<RelatedTo>
    RETURN {
        node: node,
        properties: properties,
        relationships_from: outgoing,
        relationships_to: incoming,
        related_nodes: related
    }

QUERY getNeighborhood(org_id: String, seed_name: String) =>
    seed <- N<OntologyNode>({ name: seed_name })::WHERE(_::{org_id}::EQ(org_id))
    out_rels <- seed::Out<Relationship>
    in_rels <- seed::In<Relationship>
    similar <- seed::Out<RelatedTo>
    RETURN {
        outgoing: out_rels,
        incoming: in_rels,
        similar: similar
    }

// =============================================================================
// QUERIES - Uncertainty / Learning
// =============================================================================

QUERY getNodesWithPredicate(org_id: String, predicate: String) =>
    props <- N<PropertyClaim>({ predicate: predicate })::WHERE(_::{org_id}::EQ(org_id))
    nodes <- props::In<HasProperty>
    RETURN { properties: props, nodes: nodes }

QUERY getPropertyValue(org_id: String, node_name: String, predicate: String) =>
    node <- N<OntologyNode>({ name: node_name })::WHERE(_::{org_id}::EQ(org_id))
    props <- node::Out<HasProperty>::WHERE(_::{predicate}::EQ(predicate))
    RETURN props

QUERY getUncertainClaims(org_id: String) =>
    claims <- N<PropertyClaim>::WHERE(_::{org_id}::EQ(org_id))::WHERE(_::{status}::EQ("uncertain"))
    RETURN claims

QUERY getHypotheses(org_id: String) =>
    claims <- N<PropertyClaim>::WHERE(_::{org_id}::EQ(org_id))::WHERE(_::{status}::EQ("hypothesis"))
    RETURN claims

QUERY addHypothesis(org_id: String, predicate: String, value: String, created_at: I64) =>
    prop <- AddN<PropertyClaim>({
        org_id: org_id,
        predicate: predicate,
        value: value,
        confidence: 0.1,
        status: "hypothesis",
        created_at: created_at,
        updated_at: created_at
    })
    RETURN prop

// =============================================================================
// QUERIES - Relevance Edges
// =============================================================================

QUERY addRelevanceEdge(org_id: String, from_node_id: ID, to_node_id: ID, weight: F64, relation: String) =>
    edge <- AddE<RelatedTo>::From(from_node_id)::To(to_node_id)
    RETURN edge

QUERY getRelatedNodes(org_id: String, node_name: String) =>
    node <- N<OntologyNode>({ name: node_name })::WHERE(_::{org_id}::EQ(org_id))
    outgoing <- node::Out<RelatedTo>
    incoming <- node::In<RelatedTo>
    RETURN { outgoing: outgoing, incoming: incoming }
