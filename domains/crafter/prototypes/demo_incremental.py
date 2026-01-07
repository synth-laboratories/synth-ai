"""
Demo: Incremental ontology building for Crafter.
Shows the generic API working with domain-specific data.

Run with: python demo_incremental.py
(Requires HelixDB at localhost:6969, or use --mock for in-memory testing)
"""

import sys
import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

# For mock mode (no HelixDB required)
MOCK_MODE = "--mock" in sys.argv


# =============================================================================
# MOCK ONTOLOGY RECORD (in-memory, for testing without HelixDB)
# =============================================================================

class MockOntologyRecord:
    """In-memory mock for testing without HelixDB."""

    def __init__(self):
        self.current_time = 0
        self.nodes: Dict[str, Dict] = {}
        self.properties: Dict[str, Dict] = {}  # "node.pred" -> {value, evidence: [], confidence}
        self.edges: List[Dict] = []
        self.hypotheses: List[Dict] = []

    def set_time(self, t: int):
        self.current_time = t

    def record_observation(self, node: str, predicate: str, value: Any,
                          observation_text: str, annotation: str = "", **kwargs):
        key = f"{node}.{predicate}"
        if key not in self.properties:
            self.nodes[node] = {"name": node, "type": "unknown"}
            self.properties[key] = {"node": node, "predicate": predicate, "value": value,
                                    "evidence": [], "confidence": 0.0}

        self.properties[key]["evidence"].append({
            "text": observation_text, "time": self.current_time, "type": "observation"
        })
        # Update confidence: n / (n + 1)
        n = len(self.properties[key]["evidence"])
        self.properties[key]["confidence"] = n / (n + 1.0)
        return self.properties[key]

    def record_edge(self, from_node: str, relation: str, to_node: str,
                   observation_text: str, value: Any = None, **kwargs):
        self.nodes[from_node] = self.nodes.get(from_node, {"name": from_node})
        self.nodes[to_node] = self.nodes.get(to_node, {"name": to_node})
        edge = {"from": from_node, "relation": relation, "to": to_node,
                "value": value, "observation": observation_text, "time": self.current_time}
        self.edges.append(edge)
        return edge

    def record_inference(self, node: str, predicate: str, value: Any,
                        reasoning: str, weight: float = 0.5):
        key = f"{node}.{predicate}"
        self.nodes[node] = self.nodes.get(node, {"name": node})
        self.properties[key] = {"node": node, "predicate": predicate, "value": value,
                                "evidence": [{"text": reasoning, "type": "inference"}],
                                "confidence": weight}
        return self.properties[key]

    def record_hypothesis(self, node: str, predicate: str, value: Any, reason: str = ""):
        hyp = {"node": node, "predicate": predicate, "value": value, "reason": reason,
               "status": "hypothesis", "confidence": 0.1}
        self.hypotheses.append(hyp)
        return hyp

    def get_node_context(self, node_name: str) -> Dict[str, Any]:
        if node_name not in self.nodes:
            return {"known": False, "node": node_name}

        props = {k.split(".")[1]: v for k, v in self.properties.items()
                 if k.startswith(f"{node_name}.")}
        out_edges = [e for e in self.edges if e["from"] == node_name]
        in_edges = [e for e in self.edges if e["to"] == node_name]

        return {"known": True, "node": node_name, "properties": props,
                "outgoing": out_edges, "incoming": in_edges}

    def get_outgoing_edges(self, node_name: str, relation_type: str) -> List[Dict]:
        return [{"target": e["to"], "value": e["value"]}
                for e in self.edges if e["from"] == node_name and e["relation"] == relation_type]

    def get_incoming_edges(self, node_name: str, relation_type: str) -> List[Dict]:
        return [{"source": e["from"], "value": e["value"]}
                for e in self.edges if e["to"] == node_name and e["relation"] == relation_type]

    def get_property_value(self, node_name: str, predicate: str) -> Optional[Dict]:
        key = f"{node_name}.{predicate}"
        if key in self.properties:
            p = self.properties[key]
            return {"value": p["value"], "confidence": p["confidence"]}
        return None

    def get_nodes_with_predicate(self, predicate: str) -> List[Dict]:
        return [{"node": v["node"], "value": v["value"], "confidence": v["confidence"]}
                for k, v in self.properties.items() if k.endswith(f".{predicate}")]

    def get_uncertain_claims(self) -> List[Dict]:
        return [{"node": v["node"], "predicate": v["predicate"], "confidence": v["confidence"]}
                for v in self.properties.values() if v["confidence"] < 0.5]

    def get_hypotheses(self) -> List[Dict]:
        return self.hypotheses


# =============================================================================
# DEMO
# =============================================================================

def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_result(label: str, result: Any):
    print(f"\n{label}:")
    if isinstance(result, dict):
        print(f"  {json.dumps(result, indent=2, default=str)}")
    elif isinstance(result, list):
        for item in result:
            print(f"  - {item}")
    else:
        print(f"  {result}")


def run_demo():
    print_section("CRAFTER ONTOLOGY - INCREMENTAL BUILDING DEMO")

    # Demo org_id - in production this comes from Supabase auth
    org_id = "demo-org-001"

    if MOCK_MODE:
        print("\n[Running in MOCK mode - no HelixDB required]")
        record = MockOntologyRecord()
    else:
        print(f"\n[Connecting to HelixDB at localhost:6969 with org_id={org_id}]")
        try:
            from observer_helix import OntologyRecord, NodeType, make_node_type_inferrer

            # Create with Crafter type inference
            crafter_types = {
                "zombie": NodeType.CREATURE, "skeleton": NodeType.CREATURE,
                "cow": NodeType.CREATURE, "player": NodeType.CREATURE,
                "wood": NodeType.RESOURCE, "stone": NodeType.RESOURCE,
                "food": NodeType.RESOURCE, "tree": NodeType.RESOURCE,
            }
            inferrer = make_node_type_inferrer(type_map=crafter_types)
            record = OntologyRecord(org_id=org_id, node_type_inferrer=inferrer)
        except Exception as e:
            print(f"[HelixDB connection failed: {e}]")
            print("[Falling back to MOCK mode]")
            record = MockOntologyRecord()

    # =========================================================================
    # STEP 1: First exploration - find some trees, collect wood
    # =========================================================================
    print_section("STEP 1: First Exploration (t=10)")
    record.set_time(10)

    record.record_edge("tree", "yields", "wood", "chopped tree, got wood")
    print("  Recorded: tree --yields--> wood")

    # Query: what do I know about trees?
    ctx = record.get_node_context("tree")
    print_result("Query: What do I know about 'tree'?", ctx)

    # =========================================================================
    # STEP 2: First combat - zombie attacks!
    # =========================================================================
    print_section("STEP 2: First Combat (t=50)")
    record.set_time(50)

    record.record_observation("zombie", "damage", 2, "zombie hit me for 2 damage")
    print("  Recorded: zombie.damage = 2")

    # Query: what's dangerous?
    dangerous = record.get_nodes_with_predicate("damage")
    print_result("Query: What has 'damage' property?", dangerous)

    # =========================================================================
    # STEP 3: More observations - confidence increases
    # =========================================================================
    print_section("STEP 3: Repeated Observations (t=60-80)")

    record.set_time(60)
    record.record_observation("zombie", "damage", 2, "zombie hit again for 2")
    print("  t=60: zombie.damage = 2 (confirmation)")

    record.set_time(70)
    record.record_observation("zombie", "damage", 2, "third zombie attack, 2 damage")
    print("  t=70: zombie.damage = 2 (another confirmation)")

    record.set_time(80)
    record.record_edge("tree", "yields", "wood", "chopped another tree")
    print("  t=80: tree --yields--> wood (confirmation)")

    # Check confidence
    damage_prop = record.get_property_value("zombie", "damage")
    print_result("Query: zombie.damage confidence after 3 observations", damage_prop)

    # =========================================================================
    # STEP 4: Kill a zombie - infer health
    # =========================================================================
    print_section("STEP 4: Combat Victory (t=100)")
    record.set_time(100)

    record.record_inference("zombie", "health", 5,
                           "zombie died after 5 hits with 1-damage weapon")
    print("  Inferred: zombie.health = 5 (from combat)")

    ctx = record.get_node_context("zombie")
    print_result("Query: Full zombie knowledge", ctx)

    # =========================================================================
    # STEP 5: Discover crafting
    # =========================================================================
    print_section("STEP 5: Crafting Discovery (t=150)")
    record.set_time(150)

    record.record_edge("place_table", "requires", "wood",
                      "crafted table, used 2 wood", value={"amount": 2})
    record.record_edge("place_table", "produces", "table",
                      "crafted table successfully")
    print("  Recorded: place_table --requires--> wood {amount: 2}")
    print("  Recorded: place_table --produces--> table")

    # Query: how do I make a table?
    reqs = record.get_outgoing_edges("place_table", "requires")
    produces = record.get_outgoing_edges("place_table", "produces")
    print_result("Query: place_table requirements", reqs)
    print_result("Query: place_table produces", produces)

    # =========================================================================
    # STEP 6: Form a hypothesis
    # =========================================================================
    print_section("STEP 6: Hypothesis Formation (t=200)")
    record.set_time(200)

    record.record_hypothesis("skeleton", "damage", 1,
                            "skeletons look weaker, probably 1 damage")
    print("  Hypothesis: skeleton.damage = 1 (untested)")

    # Query: what needs testing?
    hypotheses = record.get_hypotheses()
    print_result("Query: Untested hypotheses", hypotheses)

    # =========================================================================
    # STEP 7: Find food source
    # =========================================================================
    print_section("STEP 7: Resource Discovery (t=250)")
    record.set_time(250)

    record.record_edge("cow", "yields", "food", "killed cow, got 6 food",
                      value={"amount": 6})
    print("  Recorded: cow --yields--> food {amount: 6}")

    # Query: where can I find food?
    food_sources = record.get_incoming_edges("food", "yields")
    print_result("Query: What yields 'food'?", food_sources)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_section("FINAL ONTOLOGY SUMMARY")

    print("\nKnown entities:")
    for node in ["zombie", "tree", "cow", "place_table", "skeleton"]:
        ctx = record.get_node_context(node)
        if ctx.get("known"):
            props = ctx.get("properties", {})
            out = ctx.get("outgoing", [])
            print(f"\n  {node}:")
            for pred, pdata in props.items():
                conf = pdata.get("confidence", 0)
                print(f"    - {pred}: {pdata.get('value')}  [conf={conf:.0%}]")
            for edge in out:
                rel = edge.get("relation", "?")
                target = edge.get("target") or edge.get("to", "?")
                print(f"    → {rel} → {target}")

    uncertain = record.get_uncertain_claims()
    if uncertain:
        print("\nUncertain claims (need more evidence):")
        for c in uncertain:
            print(f"  - {c}")

    print("\n" + "="*60)
    print("  DEMO COMPLETE")
    print("="*60)


if __name__ == "__main__":
    run_demo()
