#!/usr/bin/env python3
"""
Run MIPRO with ontology once, save snapshot, and print it.
"""
import json
import os
import subprocess
import sys
import time
from urllib.parse import quote

import httpx

HELIX_URL = "https://helix-prod-production.up.railway.app"
BACKEND_URL = os.environ.get("SYNTH_BACKEND_URL", "https://synth-rust-backend-production.up.railway.app")
API_KEY = os.environ.get("SYNTH_API_KEY", "")
ORG_ID = "e77ef3a8-677d-4ddd-92d6-0f114d6bbdaf"

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def fetch_full_ontology(org_id: str) -> dict:
    """Fetch all nodes, relationships, and properties for an org from HelixDB."""
    nodes_resp = httpx.post(
        f"{HELIX_URL}/getPublicNodes",
        json={"org_id": org_id},
        timeout=30.0,
    )
    nodes_resp.raise_for_status()
    nodes = nodes_resp.json().get("nodes", [])

    # Fetch relationships for each node
    all_relationships = []
    seen_rel_ids = set()
    for node in nodes:
        name = node.get("name", "")
        try:
            out_resp = httpx.post(
                f"{HELIX_URL}/getRelationshipsFrom",
                json={"org_id": org_id, "node_name": name},
                timeout=10.0,
            )
            if out_resp.status_code == 200:
                for rel in out_resp.json().get("relationships", []):
                    rid = rel.get("id", "")
                    if rid and rid not in seen_rel_ids:
                        seen_rel_ids.add(rid)
                        all_relationships.append(rel)
        except Exception:
            pass

    return {
        "org_id": org_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "node_count": len(nodes),
        "relationship_count": len(all_relationships),
        "nodes": nodes,
        "relationships": all_relationships,
    }


def run_demo(rollouts_per_split: int = 5, system_id: str = None) -> str:
    """Run the banking77 demo and return the results file path."""
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "run_mipro_continual.py"),
        "--rollouts-per-split", str(rollouts_per_split),
    ]
    if system_id:
        cmd.extend(["--system-id", system_id])
    print(f"\n{'='*70}")
    print(f"Running demo: {' '.join(cmd)}")
    print(f"{'='*70}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Demo exited with code {result.returncode}")

    # Find most recent results file
    results_files = sorted(
        [f for f in os.listdir(RESULTS_DIR) if f.startswith("mipro_continual_")],
        reverse=True,
    )
    if results_files:
        return os.path.join(RESULTS_DIR, results_files[0])
    return ""


def summarize_ontology(snapshot: dict, label: str):
    """Print a summary of an ontology snapshot."""
    nodes = snapshot.get("nodes", [])
    rels = snapshot.get("relationships", [])

    print(f"\n{'='*70}")
    print(f"ONTOLOGY SNAPSHOT: {label}")
    print(f"{'='*70}")
    print(f"  Nodes: {len(nodes)}")
    print(f"  Relationships: {len(rels)}")

    # Group nodes by type
    by_type: dict[str, list] = {}
    for n in nodes:
        t = n.get("node_type", "unknown")
        by_type.setdefault(t, []).append(n)

    print(f"\n  Nodes by type:")
    for t, ns in sorted(by_type.items()):
        print(f"    {t}: {len(ns)}")
        for n in ns[:5]:
            desc = n.get("description", "")
            if len(desc) > 80:
                desc = desc[:80] + "..."
            print(f"      - {n['name']}: {desc}")
        if len(ns) > 5:
            print(f"      ... and {len(ns) - 5} more")

    # Group relationships by type
    rel_by_type: dict[str, int] = {}
    for r in rels:
        t = r.get("relation_type", "unknown")
        rel_by_type[t] = rel_by_type.get(t, 0) + 1

    if rel_by_type:
        print(f"\n  Relationships by type:")
        for t, count in sorted(rel_by_type.items()):
            print(f"    {t}: {count}")


def compare_ontologies(before: dict, after: dict):
    """Compare two ontology snapshots."""
    before_names = {n["name"] for n in before.get("nodes", [])}
    after_names = {n["name"] for n in after.get("nodes", [])}

    new_nodes = after_names - before_names
    removed_nodes = before_names - after_names

    # Build name->node map for after
    after_map = {n["name"]: n for n in after.get("nodes", [])}

    print(f"\n{'='*70}")
    print(f"ONTOLOGY EVOLUTION COMPARISON")
    print(f"{'='*70}")
    print(f"  Before: {len(before_names)} nodes, {len(before.get('relationships', []))} relationships")
    print(f"  After:  {len(after_names)} nodes, {len(after.get('relationships', []))} relationships")
    print(f"  New nodes: {len(new_nodes)}")
    print(f"  Removed nodes: {len(removed_nodes)}")

    if new_nodes:
        print(f"\n  NEW NODES ({len(new_nodes)}):")
        for name in sorted(new_nodes):
            node = after_map.get(name, {})
            ntype = node.get("node_type", "?")
            desc = node.get("description", "")
            if len(desc) > 80:
                desc = desc[:80] + "..."
            print(f"    + [{ntype}] {name}: {desc}")

    if removed_nodes:
        print(f"\n  REMOVED NODES ({len(removed_nodes)}):")
        for name in sorted(removed_nodes):
            print(f"    - {name}")

    # Compare relationship counts by type
    def rel_type_counts(snapshot):
        counts = {}
        for r in snapshot.get("relationships", []):
            t = r.get("relation_type", "unknown")
            counts[t] = counts.get(t, 0) + 1
        return counts

    before_rels = rel_type_counts(before)
    after_rels = rel_type_counts(after)
    all_rel_types = sorted(set(list(before_rels.keys()) + list(after_rels.keys())))

    if all_rel_types:
        print(f"\n  RELATIONSHIP CHANGES:")
        for t in all_rel_types:
            b = before_rels.get(t, 0)
            a = after_rels.get(t, 0)
            if a != b:
                print(f"    {t}: {b} -> {a} ({'+' if a > b else ''}{a - b})")


def main():
    print("ONTOLOGY EVOLUTION EXPERIMENT")
    print("=" * 70)
    print(f"HelixDB: {HELIX_URL}")
    print(f"Org ID: {ORG_ID}")

    # =====================================================================
    # Single phase: Run 400 rollouts (100 per split)
    # =====================================================================
    print("\n\nPHASE 1: Running 400 rollouts (100 per split)...")
    run_demo(rollouts_per_split=100)

    # Wait for async ontology writes to complete
    print("\nWaiting 45s for async ontology writes to complete...")
    time.sleep(45)

    # Save ontology snapshot
    print("Fetching ontology snapshot after run...")
    snapshot_1 = fetch_full_ontology(ORG_ID)
    path_1 = os.path.join(RESULTS_DIR, "ontology_after_run.json")
    with open(path_1, "w") as f:
        json.dump(snapshot_1, f, indent=2)
    print(f"Saved to {path_1}")
    summarize_ontology(snapshot_1, "After run (400 rollouts)")

    # Print full ontology JSON
    print(f"\n{'='*70}")
    print("FULL ONTOLOGY JSON:")
    print(f"{'='*70}")
    print(json.dumps(snapshot_1, indent=2))
    print(f"\n\nSnapshot saved to:")
    print(f"  {path_1}")


if __name__ == "__main__":
    main()
