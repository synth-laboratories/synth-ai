"""
Compare SQLite vs HelixDB queries for the Crafter ontology.
Runs equivalent queries on both and measures complexity/results.
"""

import sqlite3
import json
import time
import requests
from pathlib import Path

SQLITE_DB = Path(__file__).parent / "crafter_ontology.db"
HELIX_URL = "http://localhost:6969"


# =============================================================================
# SQLITE QUERIES
# =============================================================================

def sqlite_queries():
    """Run queries on SQLite and return results with timing."""
    conn = sqlite3.connect(SQLITE_DB)
    cursor = conn.cursor()
    results = {}

    # Q1: Get all actions
    start = time.perf_counter()
    cursor.execute("SELECT name, description FROM action_types")
    results["Q1_all_actions"] = {
        "data": cursor.fetchall(),
        "time_ms": (time.perf_counter() - start) * 1000,
        "sql": "SELECT name, description FROM action_types"
    }

    # Q2: Get action by name
    start = time.perf_counter()
    cursor.execute("SELECT * FROM action_types WHERE name = ?", ("MakeWoodPickaxe",))
    row = cursor.fetchone()
    results["Q2_action_by_name"] = {
        "data": row,
        "time_ms": (time.perf_counter() - start) * 1000,
        "sql": "SELECT * FROM action_types WHERE name = ?"
    }

    # Q3: Get interfaces for Player (requires JSON parsing)
    start = time.perf_counter()
    cursor.execute("SELECT implements FROM object_types WHERE name = ?", ("Player",))
    row = cursor.fetchone()
    interfaces = json.loads(row[0]) if row else []
    results["Q3_player_interfaces"] = {
        "data": interfaces,
        "time_ms": (time.perf_counter() - start) * 1000,
        "sql": "SELECT implements FROM object_types WHERE name = ? (+ JSON parse)"
    }

    # Q4: Get interface inheritance (requires recursive CTE)
    start = time.perf_counter()
    cursor.execute("""
        WITH RECURSIVE interface_tree AS (
            SELECT name, extends, 0 as depth
            FROM interfaces WHERE name = 'LivingCreature'
            UNION ALL
            SELECT i.name, i.extends, it.depth + 1
            FROM interfaces i, interface_tree it, json_each(it.extends) je
            WHERE je.value = i.name
        )
        SELECT name, depth FROM interface_tree
    """)
    results["Q4_interface_inheritance"] = {
        "data": cursor.fetchall(),
        "time_ms": (time.perf_counter() - start) * 1000,
        "sql": "WITH RECURSIVE interface_tree AS (...) -- 8 lines"
    }

    # Q5: Find hostile creatures (JSON property search)
    start = time.perf_counter()
    cursor.execute("""
        SELECT name FROM object_types
        WHERE json_extract(properties, '$.damage') IS NOT NULL
    """)
    results["Q5_hostile_creatures"] = {
        "data": [r[0] for r in cursor.fetchall()],
        "time_ms": (time.perf_counter() - start) * 1000,
        "sql": "SELECT name FROM object_types WHERE json_extract(properties, '$.damage') IS NOT NULL"
    }

    # Q6: Find actions requiring table (string search in JSON)
    start = time.perf_counter()
    cursor.execute("""
        SELECT name FROM action_types
        WHERE preconditions LIKE '%table%'
    """)
    results["Q6_table_actions"] = {
        "data": [r[0] for r in cursor.fetchall()],
        "time_ms": (time.perf_counter() - start) * 1000,
        "sql": "SELECT name FROM action_types WHERE preconditions LIKE '%table%'"
    }

    # Q7: Get crafting costs (requires JSON parsing in Python)
    start = time.perf_counter()
    cursor.execute("SELECT name, effects FROM action_types WHERE name LIKE 'Make%' OR name LIKE 'Place%'")
    crafting = []
    for name, effects_json in cursor.fetchall():
        effects = json.loads(effects_json)
        costs = [e for e in effects if isinstance(e, dict) and "decrement" in e]
        crafting.append((name, costs))
    results["Q7_crafting_costs"] = {
        "data": crafting,
        "time_ms": (time.perf_counter() - start) * 1000,
        "sql": "SELECT name, effects FROM action_types WHERE name LIKE 'Make%' (+ Python JSON processing)"
    }

    # Q8: Get dynamics affecting Player
    start = time.perf_counter()
    cursor.execute("SELECT name, effects FROM dynamics WHERE for_each IS NULL OR for_each = 'Player'")
    results["Q8_player_dynamics"] = {
        "data": cursor.fetchall(),
        "time_ms": (time.perf_counter() - start) * 1000,
        "sql": "SELECT name, effects FROM dynamics WHERE for_each IS NULL OR for_each = 'Player'"
    }

    conn.close()
    return results


# =============================================================================
# HELIX QUERIES
# =============================================================================

def helix_query(query_name: str, params: dict = None):
    """Execute a HelixDB query."""
    try:
        resp = requests.post(
            f"{HELIX_URL}/{query_name}",
            json=params or {},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        return resp.json() if resp.status_code == 200 else {"error": resp.text, "status": resp.status_code}
    except Exception as e:
        return {"error": str(e)}


def helix_add_data():
    """Load test data into HelixDB."""
    print("Loading data into HelixDB...")

    # Add interfaces
    interfaces = [
        ("Positioned", "Entity with position", "[]", "{}"),
        ("Inventoried", "Entity with inventory", "[]", "{}"),
        ("LivingCreature", "Mortal entity", '["move", "do"]', '{"is_alive": "health > 0"}'),
    ]
    for name, desc, caps, derived in interfaces:
        helix_query("addInterface", {
            "name": name, "description": desc, "capabilities": caps, "derived": derived
        })

    # Add object types
    object_types = [
        ("Player", "Agent-controlled entity", "id", '{"facing": "Direction"}', "{}"),
        ("Zombie", "Hostile creature", "id", '{"health": 5, "damage": 2}', "{}"),
        ("Skeleton", "Ranged hostile", "id", '{"health": 3, "damage": 1}', "{}"),
        ("Cow", "Passive creature", "id", '{"health": 3}', "{}"),
    ]
    for name, desc, pk, props, derived in object_types:
        helix_query("addObjectType", {
            "name": name, "description": desc, "pk": pk, "props": props, "derived": derived
        })

    # Add action types
    actions = [
        ("Noop", "0", "Do nothing", "{}", "[]", "[]"),
        ("Move", "[1,4]", "Move in direction", '{"direction": "Direction"}',
         '["facing_tile.material not blocked"]', '[]'),
        ("MakeWoodPickaxe", "11", "Craft wood pickaxe", "{}",
         '["Player.inventory.wood >= 1", "nearby(Player.pos, 1).materials contains table"]',
         '[{"decrement": "Player.inventory.wood", "by": 1}]'),
        ("PlaceTable", "9", "Place crafting table", "{}",
         '["Player.inventory.wood >= 2"]',
         '[{"decrement": "Player.inventory.wood", "by": 2}]'),
    ]
    for name, aid, desc, params, preconds, effects in actions:
        helix_query("addActionType", {
            "name": name, "action_id": aid, "description": desc,
            "params": params, "preconditions": preconds, "effects": effects
        })

    # Link interfaces (LivingCreature extends Positioned)
    helix_query("linkExtends", {"child_name": "LivingCreature", "parent_name": "Positioned"})

    # Link object types to interfaces
    for obj in ["Player", "Zombie", "Skeleton", "Cow"]:
        helix_query("linkImplements", {"obj_name": obj, "iface_name": "Positioned"})
        helix_query("linkImplements", {"obj_name": obj, "iface_name": "LivingCreature"})
    helix_query("linkImplements", {"obj_name": "Player", "iface_name": "Inventoried"})

    print("Data loaded.")


def helix_queries():
    """Run queries on HelixDB and return results with timing."""
    results = {}

    # Q1: Get all actions
    start = time.perf_counter()
    data = helix_query("getAllActions")
    results["Q1_all_actions"] = {
        "data": data,
        "time_ms": (time.perf_counter() - start) * 1000,
        "hql": "actions <- N<ActionType>  RETURN actions"
    }

    # Q2: Get action by name
    start = time.perf_counter()
    data = helix_query("getActionByName", {"action_name": "MakeWoodPickaxe"})
    results["Q2_action_by_name"] = {
        "data": data,
        "time_ms": (time.perf_counter() - start) * 1000,
        "hql": "action <- N<ActionType>({Name: action_name})  RETURN action"
    }

    # Q3: Get interfaces for Player (graph traversal!)
    start = time.perf_counter()
    data = helix_query("getObjectInterfaces", {"type_name": "Player"})
    results["Q3_player_interfaces"] = {
        "data": data,
        "time_ms": (time.perf_counter() - start) * 1000,
        "hql": "obj <- N<ObjectType>({Name: type_name})  interfaces <- obj::Out<Implements>  RETURN interfaces"
    }

    # Q4: Get parent interfaces (one hop - full recursion needs ShortestPath)
    start = time.perf_counter()
    data = helix_query("getParentInterfaces", {"interface_name": "LivingCreature"})
    results["Q4_interface_inheritance"] = {
        "data": data,
        "time_ms": (time.perf_counter() - start) * 1000,
        "hql": "iface <- N<Interface>({Name: interface_name})  parents <- iface::Out<Extends>  RETURN parents"
    }

    # Q5: Get hostile creatures (reverse traversal from property)
    start = time.perf_counter()
    data = helix_query("getHostileCreatures")
    results["Q5_hostile_creatures"] = {
        "data": data,
        "time_ms": (time.perf_counter() - start) * 1000,
        "hql": "damage_prop <- N<Property>({Name: 'damage'})  hostile <- damage_prop::In<HasProperty>  RETURN hostile"
    }

    # Q6: Get all actions (filter in app layer)
    start = time.perf_counter()
    data = helix_query("getAllActionsWithPreconditions")
    results["Q6_table_actions"] = {
        "data": data,
        "time_ms": (time.perf_counter() - start) * 1000,
        "hql": "actions <- N<ActionType>  RETURN actions  // + app-layer filter"
    }

    # Q7: Get crafting dependencies (graph traversal)
    start = time.perf_counter()
    data = helix_query("getCraftingDependencies", {"action_name": "MakeWoodPickaxe"})
    results["Q7_crafting_costs"] = {
        "data": data,
        "time_ms": (time.perf_counter() - start) * 1000,
        "hql": "action <- N<ActionType>({Name: action_name})  resources <- action::Out<Consumes>  RETURN resources"
    }

    # Q8: Get player dynamics (reverse traversal)
    start = time.perf_counter()
    data = helix_query("getPlayerDynamics")
    results["Q8_player_dynamics"] = {
        "data": data,
        "time_ms": (time.perf_counter() - start) * 1000,
        "hql": "player <- N<ObjectType>({Name: 'Player'})  dynamics <- player::In<Affects>  RETURN dynamics"
    }

    return results


# =============================================================================
# COMPARISON
# =============================================================================

def print_comparison():
    print("=" * 80)
    print("SQLITE vs HELIX QUERY COMPARISON")
    print("=" * 80)

    # Check if HelixDB is running (try a simple query)
    try:
        resp = requests.post(f"{HELIX_URL}/getAllActions", json={}, timeout=2)
        helix_available = resp.status_code == 200
    except:
        helix_available = False

    print(f"\nSQLite DB: {SQLITE_DB}")
    print(f"HelixDB: {HELIX_URL} ({'RUNNING' if helix_available else 'NOT AVAILABLE'})")

    # Run SQLite queries
    print("\n" + "-" * 40)
    print("SQLITE RESULTS")
    print("-" * 40)
    sqlite_results = sqlite_queries()

    for qname, result in sqlite_results.items():
        print(f"\n[{qname}]")
        print(f"  SQL: {result['sql']}")
        print(f"  Time: {result['time_ms']:.3f}ms")
        data = result['data']
        if isinstance(data, list) and len(data) > 3:
            print(f"  Results: {data[:3]} ... ({len(data)} total)")
        else:
            print(f"  Results: {data}")

    if helix_available:
        # Load data into HelixDB
        helix_add_data()

        print("\n" + "-" * 40)
        print("HELIX RESULTS")
        print("-" * 40)
        helix_results = helix_queries()

        for qname, result in helix_results.items():
            print(f"\n[{qname}]")
            print(f"  HQL: {result['hql']}")
            print(f"  Time: {result['time_ms']:.3f}ms")
            data = result['data']
            if isinstance(data, list) and len(data) > 3:
                print(f"  Results: {data[:3]} ... ({len(data)} total)")
            else:
                print(f"  Results: {data}")

    # Summary
    print("\n" + "=" * 80)
    print("QUERY COMPLEXITY COMPARISON")
    print("=" * 80)
    print("""
| Query                      | SQLite                          | HelixDB                        |
|----------------------------|--------------------------------|--------------------------------|
| Q1: All actions            | Simple SELECT                  | Simple node scan               |
| Q2: Action by name         | WHERE clause                   | Index lookup                   |
| Q3: Object interfaces      | JSON parse in app              | ::Out<Implements> traversal    |
| Q4: Interface inheritance  | 8-line recursive CTE           | ::Out<Extends> (1 hop)         |
| Q5: Hostile creatures      | json_extract() function        | ::In<HasProperty> reverse trav |
| Q6: Actions with condition | LIKE '%pattern%'               | App-layer filter (no substr)   |
| Q7: Crafting costs         | SELECT + Python JSON parse     | ::Out<Consumes> traversal      |
| Q8: Entity dynamics        | WHERE for_each filter          | ::In<Affects> reverse traversal|
""")


if __name__ == "__main__":
    print_comparison()
