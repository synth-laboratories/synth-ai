"""
SQLite Prototype for Crafter Ontology

This demonstrates modeling the Crafter ontology in a relational database.
We'll see where it maps naturally and where we need JSON workarounds.
"""

import sqlite3
import json
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent / "crafter_ontology.db"


def create_schema(conn: sqlite3.Connection):
    """Create the relational schema for the ontology."""
    cursor = conn.cursor()

    # ==========================================================================
    # CORE TABLES
    # ==========================================================================

    # Interfaces - shared capabilities
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interfaces (
            name TEXT PRIMARY KEY,
            description TEXT,
            extends TEXT,  -- JSON array of parent interface names
            properties TEXT,  -- JSON object {name: {type, required}}
            derived TEXT,  -- JSON object {name: expression}
            capabilities TEXT  -- JSON array of action names
        )
    """)

    # Object Types - entities in the world
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS object_types (
            name TEXT PRIMARY KEY,
            description TEXT,
            implements TEXT,  -- JSON array of interface names
            primary_key TEXT,
            properties TEXT,  -- JSON object {name: {type, min, max, initial, internal}}
            derived TEXT  -- JSON object {name: expression}
        )
    """)

    # Enums - fixed value sets
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS enums (
            name TEXT PRIMARY KEY,
            enum_values TEXT  -- JSON array of values
        )
    """)

    # Link Types - relationships between object types
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS link_types (
            name TEXT PRIMARY KEY,
            from_type TEXT,  -- single type or JSON array
            to_type TEXT,
            cardinality TEXT,  -- one_to_one, one_to_many, etc.
            computed INTEGER DEFAULT 0,  -- boolean
            condition TEXT,  -- expression for computed links
            properties TEXT  -- JSON object for edge properties
        )
    """)

    # Action Types - operations the agent can perform
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS action_types (
            name TEXT PRIMARY KEY,
            action_id TEXT,  -- single int or JSON array for parameterized
            description TEXT,
            parameters TEXT,  -- JSON object {name: {type, enum}}
            preconditions TEXT,  -- JSON array of boolean expressions
            effects TEXT  -- JSON array of effect objects
        )
    """)

    # Functions - computed values
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS functions (
            name TEXT PRIMARY KEY,
            description TEXT,
            parameters TEXT,  -- JSON object {name: type}
            returns TEXT,
            logic TEXT
        )
    """)

    # Dynamics - autonomous world updates
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dynamics (
            name TEXT PRIMARY KEY,
            every TEXT DEFAULT 'step',
            condition TEXT,
            for_each TEXT,  -- object type for per-entity dynamics
            effects TEXT  -- JSON array of effect objects
        )
    """)

    # ==========================================================================
    # NORMALIZED RELATIONSHIP TABLES (alternative to JSON arrays)
    # ==========================================================================

    # Interface inheritance
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interface_extends (
            child_interface TEXT,
            parent_interface TEXT,
            PRIMARY KEY (child_interface, parent_interface),
            FOREIGN KEY (child_interface) REFERENCES interfaces(name),
            FOREIGN KEY (parent_interface) REFERENCES interfaces(name)
        )
    """)

    # Object type implements interface
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS object_implements (
            object_type TEXT,
            interface TEXT,
            PRIMARY KEY (object_type, interface),
            FOREIGN KEY (object_type) REFERENCES object_types(name),
            FOREIGN KEY (interface) REFERENCES interfaces(name)
        )
    """)

    # Action preconditions (normalized)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS action_preconditions (
            action_name TEXT,
            precondition_order INTEGER,
            expression TEXT,
            PRIMARY KEY (action_name, precondition_order),
            FOREIGN KEY (action_name) REFERENCES action_types(name)
        )
    """)

    # Properties table (fully normalized alternative)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS properties (
            owner_type TEXT,  -- 'interface' or 'object_type'
            owner_name TEXT,
            property_name TEXT,
            property_type TEXT,
            min_value REAL,
            max_value REAL,
            initial_value REAL,
            is_internal INTEGER DEFAULT 0,
            is_required INTEGER DEFAULT 1,
            PRIMARY KEY (owner_type, owner_name, property_name)
        )
    """)

    conn.commit()


def load_crafter_data(conn: sqlite3.Connection):
    """Load Crafter ontology data into SQLite."""
    cursor = conn.cursor()

    # ==========================================================================
    # INTERFACES
    # ==========================================================================
    interfaces = [
        ("Positioned", "Entity with position in world", None,
         json.dumps({"pos": {"type": "tuple[int,int]", "required": True}}),
         None, None),

        ("Inventoried", "Entity with inventory", None,
         json.dumps({
             "health": {"type": "int", "required": True},
             "food": {"type": "int", "required": True},
             "drink": {"type": "int", "required": True},
             "energy": {"type": "int", "required": True},
         }),
         None, None),

        ("LivingCreature", "Mortal entity with health",
         json.dumps(["Positioned"]),
         json.dumps({"health": {"type": "int", "min": 0, "max": 9, "required": True}}),
         json.dumps({"is_alive": "health > 0"}),
         json.dumps(["move", "do"])),
    ]

    cursor.executemany("""
        INSERT OR REPLACE INTO interfaces (name, description, extends, properties, derived, capabilities)
        VALUES (?, ?, ?, ?, ?, ?)
    """, interfaces)

    # ==========================================================================
    # OBJECT TYPES
    # ==========================================================================
    object_types = [
        ("Player", "The agent-controlled entity",
         json.dumps(["Positioned", "Inventoried", "LivingCreature"]),
         "id",
         json.dumps({
             "facing": {"type": "Direction"},
             "sleeping": {"type": "bool", "initial": False},
             "_hunger": {"type": "int", "initial": 0, "internal": True},
             "_thirst": {"type": "int", "initial": 0, "internal": True},
             "_fatigue": {"type": "int", "initial": 0, "internal": True},
             "_recover": {"type": "int", "initial": 0, "internal": True},
         }),
         None),

        ("Zombie", "Hostile creature, melee attacks",
         json.dumps(["Positioned", "LivingCreature"]),
         "id",
         json.dumps({
             "health": {"type": "int", "initial": 5, "max": 5},
             "damage": {"type": "int", "initial": 2},
             "sleep_damage": {"type": "int", "initial": 7},
         }),
         None),

        ("Skeleton", "Hostile creature, ranged attacks",
         json.dumps(["Positioned", "LivingCreature"]),
         "id",
         json.dumps({
             "health": {"type": "int", "initial": 3, "max": 3},
             "damage": {"type": "int", "initial": 1},
             "range": {"type": "int", "initial": 4},
             "cooldown": {"type": "int", "initial": 2},
         }),
         None),

        ("Cow", "Passive creature, food source",
         json.dumps(["Positioned", "LivingCreature"]),
         "id",
         json.dumps({
             "health": {"type": "int", "initial": 3, "max": 3},
             "food_value": {"type": "int", "initial": 6},
         }),
         None),

        ("Plant", "Grows to produce food",
         json.dumps(["Positioned"]),
         "id",
         json.dumps({
             "stage": {"type": "int", "min": 0, "max": 5, "initial": 0},
             "growth_rate": {"type": "float", "initial": 0.02},
         }),
         json.dumps({"is_ripe": "stage >= 4"})),
    ]

    cursor.executemany("""
        INSERT OR REPLACE INTO object_types (name, description, implements, primary_key, properties, derived)
        VALUES (?, ?, ?, ?, ?, ?)
    """, object_types)

    # ==========================================================================
    # ENUMS
    # ==========================================================================
    enums = [
        ("Direction", json.dumps(["up", "down", "left", "right"])),
        ("Material", json.dumps(["water", "grass", "sand", "stone", "path", "tree",
                                  "coal", "iron", "diamond", "lava", "table", "furnace"])),
        ("Item", json.dumps(["wood", "stone", "coal", "iron", "diamond", "sapling",
                             "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
                             "wood_sword", "stone_sword", "iron_sword"])),
    ]

    cursor.executemany("""
        INSERT OR REPLACE INTO enums (name, enum_values) VALUES (?, ?)
    """, enums)

    # ==========================================================================
    # LINK TYPES
    # ==========================================================================
    link_types = [
        ("occupies", "Player", "Material", "many_to_one", 0, None, None),
        ("threatens", json.dumps(["Zombie", "Skeleton"]), "Player", "many_to_one",
         1, "distance(creature.pos, player.pos) <= range", None),
        ("yields_on_collect", "Material", "Item", "one_to_one", 0, None,
         json.dumps({"requires_tool": {"type": "Item", "nullable": True}})),
    ]

    cursor.executemany("""
        INSERT OR REPLACE INTO link_types (name, from_type, to_type, cardinality, computed, condition, properties)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, link_types)

    # ==========================================================================
    # ACTION TYPES
    # ==========================================================================
    action_types = [
        ("Noop", "0", "Do nothing", None,
         json.dumps([]),
         json.dumps([])),

        ("Move", json.dumps([1, 4]), "Move in direction",
         json.dumps({"direction": {"type": "Direction"}}),
         json.dumps([
             "facing_tile.material not in [water, lava, stone, tree, coal, iron, diamond, table, furnace]",
             "no creature on facing_tile"
         ]),
         json.dumps([
             {"set": "Player.pos", "to": "facing_tile.pos"},
             {"set": "Player.facing", "to": "direction"}
         ])),

        ("Do", "5", "Interact with facing tile",
         None,
         json.dumps(["facing_tile has interaction"]),
         json.dumps([
             {"match": "facing_tile.material",
              "cases": [
                  {"when": "tree", "then": [
                      {"increment": "inventory.wood", "by": 1},
                      {"set": "facing_tile.material", "to": "grass"}
                  ]},
                  {"when": "water", "then": [
                      {"increment": "Player.drink", "by": 1},
                      {"set": "Player._thirst", "to": 0}
                  ]}
              ]}
         ])),

        ("MakeWoodPickaxe", "11", "Craft wood pickaxe",
         None,
         json.dumps([
             "Player.inventory.wood >= 1",
             "nearby(Player.pos, 1).materials contains table"
         ]),
         json.dumps([
             {"decrement": "Player.inventory.wood", "by": 1},
             {"increment": "Player.inventory.wood_pickaxe", "by": 1},
             {"trigger_achievement": "make_wood_pickaxe"}
         ])),

        ("PlaceTable", "9", "Place crafting table",
         None,
         json.dumps([
             "Player.inventory.wood >= 2",
             "facing_tile.material in [grass, sand, path]",
             "no object on facing_tile"
         ]),
         json.dumps([
             {"decrement": "Player.inventory.wood", "by": 2},
             {"set": "facing_tile.material", "to": "table"},
             {"trigger_achievement": "place_table"}
         ])),

        ("Sleep", "8", "Rest to recover energy",
         None,
         json.dumps(["Player.energy < 9"]),
         json.dumps([
             {"set": "Player.sleeping", "to": True}
         ])),
    ]

    cursor.executemany("""
        INSERT OR REPLACE INTO action_types (name, action_id, description, parameters, preconditions, effects)
        VALUES (?, ?, ?, ?, ?, ?)
    """, action_types)

    # ==========================================================================
    # DYNAMICS
    # ==========================================================================
    dynamics = [
        ("hunger", "step", None, None,
         json.dumps([
             {"increment": "Player._hunger", "by": 1},
             {"if": "Player._hunger >= 25", "then": [
                 {"decrement": "Player.food", "by": 1},
                 {"set": "Player._hunger", "to": 0}
             ]}
         ])),

        ("thirst", "step", None, None,
         json.dumps([
             {"increment": "Player._thirst", "by": 1},
             {"if": "Player._thirst >= 20", "then": [
                 {"decrement": "Player.drink", "by": 1},
                 {"set": "Player._thirst", "to": 0}
             ]}
         ])),

        ("health_regen", "step", None, None,
         json.dumps([
             {"increment": "Player._recover", "by": 1},
             {"if": "Player._recover >= 10 and all_vitals_positive(Player)", "then": [
                 {"increment": "Player.health", "by": 1},
                 {"set": "Player._recover", "to": 0}
             ]},
             {"if": "any_vital_zero(Player)", "then": [
                 {"decrement": "Player.health", "by": 1}
             ]}
         ])),

        ("plant_growth", "step", None, "Plant",
         json.dumps([
             {"if": "random() < Plant.growth_rate and Plant.stage < 5", "then": [
                 {"increment": "Plant.stage", "by": 1}
             ]}
         ])),
    ]

    cursor.executemany("""
        INSERT OR REPLACE INTO dynamics (name, every, condition, for_each, effects)
        VALUES (?, ?, ?, ?, ?)
    """, dynamics)

    conn.commit()
    print(f"Loaded Crafter ontology data")


# =============================================================================
# QUERY EXAMPLES
# =============================================================================

def run_queries(conn: sqlite3.Connection):
    """Run example queries to demonstrate capabilities and limitations."""
    cursor = conn.cursor()
    results = {}

    # -------------------------------------------------------------------------
    # QUERY 1: Simple lookup - get all action names
    # -------------------------------------------------------------------------
    cursor.execute("SELECT name, description FROM action_types")
    results["all_actions"] = cursor.fetchall()

    # -------------------------------------------------------------------------
    # QUERY 2: Get preconditions for an action
    # -------------------------------------------------------------------------
    cursor.execute("""
        SELECT name, preconditions FROM action_types WHERE name = 'MakeWoodPickaxe'
    """)
    row = cursor.fetchone()
    results["make_wood_pickaxe_preconditions"] = json.loads(row[1]) if row else None

    # -------------------------------------------------------------------------
    # QUERY 3: Find all hostile creatures (need to check properties JSON)
    # This is where SQLite gets awkward - we need JSON functions
    # -------------------------------------------------------------------------
    cursor.execute("""
        SELECT name, properties FROM object_types
        WHERE json_extract(properties, '$.damage') IS NOT NULL
    """)
    results["hostile_creatures"] = cursor.fetchall()

    # -------------------------------------------------------------------------
    # QUERY 4: Find what interfaces an object type implements
    # Need to parse JSON array
    # -------------------------------------------------------------------------
    cursor.execute("""
        SELECT name, implements FROM object_types WHERE name = 'Player'
    """)
    row = cursor.fetchone()
    results["player_interfaces"] = json.loads(row[1]) if row else None

    # -------------------------------------------------------------------------
    # QUERY 5: Find all actions that require a table nearby
    # This requires searching inside JSON - gets complex
    # -------------------------------------------------------------------------
    cursor.execute("""
        SELECT name, preconditions FROM action_types
        WHERE preconditions LIKE '%table%'
    """)
    results["actions_requiring_table"] = [(r[0], json.loads(r[1])) for r in cursor.fetchall()]

    # -------------------------------------------------------------------------
    # QUERY 6: Interface inheritance traversal
    # This is hard in SQLite without recursive CTEs
    # -------------------------------------------------------------------------
    cursor.execute("""
        WITH RECURSIVE interface_tree AS (
            -- Base case: start with LivingCreature
            SELECT name, extends, 0 as depth
            FROM interfaces
            WHERE name = 'LivingCreature'

            UNION ALL

            -- Recursive: follow extends links
            SELECT i.name, i.extends, it.depth + 1
            FROM interfaces i
            JOIN interface_tree it ON json_each.value = i.name
            JOIN json_each(it.extends)
        )
        SELECT name, depth FROM interface_tree
    """)
    results["living_creature_inheritance"] = cursor.fetchall()

    # -------------------------------------------------------------------------
    # QUERY 7: Find all crafting actions and their costs
    # -------------------------------------------------------------------------
    cursor.execute("""
        SELECT name, effects FROM action_types
        WHERE name LIKE 'Make%' OR name LIKE 'Place%'
    """)
    crafting = []
    for name, effects_json in cursor.fetchall():
        effects = json.loads(effects_json)
        costs = [e for e in effects if "decrement" in e]
        crafting.append((name, costs))
    results["crafting_costs"] = crafting

    return results


def main():
    # Remove existing DB to start fresh
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)

    print("=" * 60)
    print("SQLite Ontology Prototype")
    print("=" * 60)

    print("\n1. Creating schema...")
    create_schema(conn)

    print("\n2. Loading Crafter data...")
    load_crafter_data(conn)

    print("\n3. Running example queries...")
    results = run_queries(conn)

    print("\n" + "=" * 60)
    print("QUERY RESULTS")
    print("=" * 60)

    print("\n[Q1] All actions:")
    for name, desc in results["all_actions"]:
        print(f"  - {name}: {desc}")

    print("\n[Q2] MakeWoodPickaxe preconditions:")
    for p in results["make_wood_pickaxe_preconditions"]:
        print(f"  - {p}")

    print("\n[Q3] Hostile creatures (have damage property):")
    for name, props in results["hostile_creatures"]:
        print(f"  - {name}")

    print("\n[Q4] Player implements interfaces:")
    print(f"  {results['player_interfaces']}")

    print("\n[Q5] Actions requiring table:")
    for name, preconds in results["actions_requiring_table"]:
        print(f"  - {name}")

    print("\n[Q6] LivingCreature inheritance tree:")
    for name, depth in results["living_creature_inheritance"]:
        print(f"  {'  ' * depth}{name}")

    print("\n[Q7] Crafting costs:")
    for name, costs in results["crafting_costs"]:
        if costs:
            print(f"  - {name}: {costs}")

    conn.close()
    print(f"\n\nDatabase saved to: {DB_PATH}")


if __name__ == "__main__":
    main()
