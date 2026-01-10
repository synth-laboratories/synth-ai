"""
Compare SQLite vs HelixDB queries when SPEC heavily references ONTOLOGY.

Queries like:
- "What rules reference the zombie entity?"
- "Get full details of all actions mentioned in early_game heuristics"
- "What are the preconditions for actions in the spec's anticipated_transitions?"
"""

import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "spec_ontology.db"


def create_sqlite_schema(conn):
    """Create tables for both ontology and spec with references."""
    c = conn.cursor()

    # ==========================================================================
    # ONTOLOGY TABLES
    # ==========================================================================
    c.execute("""
        CREATE TABLE entities (
            name TEXT PRIMARY KEY,
            type TEXT,  -- 'creature', 'material', 'item'
            properties TEXT  -- JSON
        )
    """)

    c.execute("""
        CREATE TABLE actions (
            name TEXT PRIMARY KEY,
            action_id TEXT,
            preconditions TEXT,  -- JSON array
            effects TEXT  -- JSON array
        )
    """)

    c.execute("""
        CREATE TABLE dynamics (
            name TEXT PRIMARY KEY,
            every TEXT,
            effects TEXT  -- JSON
        )
    """)

    # ==========================================================================
    # SPEC TABLES
    # ==========================================================================
    c.execute("""
        CREATE TABLE rules (
            id TEXT PRIMARY KEY,
            constraint_text TEXT,
            rationale TEXT,
            refs TEXT  -- JSON array of ontology references
        )
    """)

    c.execute("""
        CREATE TABLE heuristics (
            id TEXT PRIMARY KEY,
            phase TEXT,  -- early_game, mid_game, late_game
            guidance TEXT,
            rationale TEXT,
            refs TEXT  -- JSON array of ontology references
        )
    """)

    c.execute("""
        CREATE TABLE anticipated_transitions (
            id INTEGER PRIMARY KEY,
            action_ref TEXT,  -- reference to actions.name
            target TEXT,
            preconditions TEXT,  -- JSON
            expected_effects TEXT  -- JSON
        )
    """)

    conn.commit()


def load_sqlite_data(conn):
    """Load both ontology and spec data."""
    c = conn.cursor()

    # Ontology: Entities
    entities = [
        ("zombie", "creature", '{"health": 5, "damage": 2, "sleep_damage": 7}'),
        ("skeleton", "creature", '{"health": 3, "damage": 1, "range": 4}'),
        ("cow", "creature", '{"health": 3, "food_value": 6}'),
        ("plant", "object", '{"stage": 0, "growth_rate": 0.02}'),
        ("water", "material", '{"walkable": false, "drinkable": true}'),
        ("lava", "material", '{"walkable": false, "deadly": true}'),
        ("tree", "material", '{"collectible": true, "yields": "wood"}'),
        ("stone", "material", '{"collectible": true, "requires": "wood_pickaxe"}'),
        ("table", "material", '{"placeable": true, "enables_crafting": true}'),
        ("furnace", "material", '{"placeable": true, "enables_smelting": true}'),
        ("wood", "item", '{"stackable": true}'),
        ("wood_pickaxe", "item", '{"tool": true, "tier": 1}'),
        ("wood_sword", "item", '{"weapon": true, "damage": 2}'),
        ("stone_pickaxe", "item", '{"tool": true, "tier": 2}'),
        ("iron_pickaxe", "item", '{"tool": true, "tier": 3}'),
    ]
    c.executemany("INSERT INTO entities VALUES (?, ?, ?)", entities)

    # Ontology: Actions
    actions = [
        ("do", "5", '["facing_tile has target"]', '[{"varies_by": "target"}]'),
        (
            "place_table",
            "9",
            '["wood >= 2", "facing grass/sand/path"]',
            '[{"decrement": "wood", "by": 2}, {"place": "table"}]',
        ),
        (
            "place_furnace",
            "10",
            '["stone >= 4", "facing grass/sand/path"]',
            '[{"decrement": "stone", "by": 4}, {"place": "furnace"}]',
        ),
        (
            "make_wood_pickaxe",
            "11",
            '["table nearby", "wood >= 1"]',
            '[{"decrement": "wood", "by": 1}, {"grant": "wood_pickaxe"}]',
        ),
        (
            "make_stone_pickaxe",
            "12",
            '["table nearby", "wood >= 1", "stone >= 1"]',
            '[{"decrement": "wood", "by": 1}, {"decrement": "stone", "by": 1}, {"grant": "stone_pickaxe"}]',
        ),
        (
            "make_wood_sword",
            "14",
            '["table nearby", "wood >= 1"]',
            '[{"decrement": "wood", "by": 1}, {"grant": "wood_sword"}]',
        ),
        (
            "place_stone",
            "17",
            '["stone >= 1", "facing water/lava"]',
            '[{"decrement": "stone", "by": 1}, {"set_tile": "path"}]',
        ),
        ("sleep", "8", '["energy < 9"]', '[{"set": "sleeping", "to": true}]'),
    ]
    c.executemany("INSERT INTO actions VALUES (?, ?, ?, ?)", actions)

    # Ontology: Dynamics
    dynamics = [
        ("thirst", "step", '{"counter": "_thirst", "threshold": 20, "depletes": "drink"}'),
        ("hunger", "step", '{"counter": "_hunger", "threshold": 25, "depletes": "food"}'),
        ("health_regen", "step", '{"requires": "all_vitals_positive", "heals": 1}'),
        ("daylight", "step", '{"cycle": 300, "affects": "spawning"}'),
        ("sleeping", "while_asleep", '{"regen_energy": true, "vulnerability": 7}'),
    ]
    c.executemany("INSERT INTO dynamics VALUES (?, ?, ?)", dynamics)

    # Spec: Rules (with ontology references)
    rules = [
        ("no_lava", "Never move onto lava tiles", "Instant death", '["entities.lava"]'),
        (
            "no_sleep_near_zombies",
            "Never sleep when zombies within 8 tiles",
            "Zombie damage 2->7 while sleeping",
            '["entities.zombie", "dynamics.sleeping"]',
        ),
        (
            "maintain_vitals",
            "Never let vitals reach 0",
            "Each depleted vital drains health",
            '["dynamics.health_regen", "dynamics.thirst", "dynamics.hunger"]',
        ),
        (
            "escape_route",
            "Maintain escape route in combat",
            "Combat is risky",
            '["entities.zombie", "entities.skeleton"]',
        ),
        (
            "armed_in_tunnels",
            "Never enter tunnels without weapon",
            "Skeletons deal ranged damage",
            '["entities.skeleton", "entities.wood_sword"]',
        ),
    ]
    c.executemany("INSERT INTO rules VALUES (?, ?, ?, ?)", rules)

    # Spec: Heuristics (with ontology references)
    heuristics = [
        (
            "collect_wood_first",
            "early_game",
            "Collect 5-10 wood immediately",
            "Wood unlocks tech tree",
            '["entities.tree", "entities.wood"]',
        ),
        (
            "place_table_early",
            "early_game",
            "Place table with 2 wood",
            "Enables crafting",
            '["actions.place_table"]',
        ),
        (
            "craft_pickaxe_immediately",
            "early_game",
            "Craft wood_pickaxe after table",
            "Unlocks stone mining",
            '["actions.make_wood_pickaxe"]',
        ),
        (
            "find_water_early",
            "early_game",
            "Find water before drink < 5",
            "Drink depletes fast",
            '["entities.water", "dynamics.thirst"]',
        ),
        (
            "avoid_zombies_early",
            "early_game",
            "Flee until wood_sword",
            "Running > unarmed combat",
            '["entities.zombie", "entities.wood_sword"]',
        ),
        (
            "establish_base",
            "mid_game",
            "Create base with table+furnace+water",
            "Centralizes operations",
            '["entities.table", "entities.furnace", "entities.water"]',
        ),
        (
            "collect_stone_for_furnace",
            "mid_game",
            "Collect 4 stone for furnace",
            "Furnace needs 4 stone",
            '["actions.place_furnace", "entities.stone"]',
        ),
        (
            "pickaxe_before_sword",
            "mid_game",
            "Upgrade pickaxe before sword",
            "Pickaxe unlocks resources",
            '["entities.stone_pickaxe", "entities.iron_pickaxe"]',
        ),
        (
            "farm_plants",
            "mid_game",
            "Plant saplings near base",
            "Sustainable food",
            '["entities.plant"]',
        ),
        (
            "hunt_daytime",
            "mid_game",
            "Hunt cows during day",
            "More cows, fewer zombies",
            '["entities.cow", "dynamics.daylight"]',
        ),
        (
            "prioritize_iron_pickaxe",
            "late_game",
            "Get iron_pickaxe for diamonds",
            "Diamonds need full tech",
            '["entities.iron_pickaxe"]',
        ),
        (
            "clear_tunnels",
            "late_game",
            "Clear skeletons from tunnels",
            "Only skeleton spawns",
            '["entities.skeleton"]',
        ),
        (
            "build_bridges",
            "late_game",
            "Use stone to bridge water/lava",
            "Access isolated areas",
            '["actions.place_stone", "entities.water", "entities.lava"]',
        ),
    ]
    c.executemany("INSERT INTO heuristics VALUES (?, ?, ?, ?, ?)", heuristics)

    # Spec: Anticipated transitions (referencing actions)
    transitions = [
        (1, "do", "tree", '["facing tree"]', '["tree->grass", "wood+=1"]'),
        (2, "do", "stone", '["facing stone", "have pickaxe"]', '["stone->path", "stone+=1"]'),
        (3, "do", "water", '["facing water"]', '["drink+=1", "_thirst=0"]'),
        (4, "do", "creature", '["creature adjacent"]', '["creature.health -= damage"]'),
        (5, "place_table", None, '["wood>=2", "valid terrain"]', '["table placed", "wood-=2"]'),
        (6, "make_wood_pickaxe", None, '["table nearby", "wood>=1"]', '["wood-=1", "pickaxe+=1"]'),
        (7, "sleep", None, '["energy<9"]', '["sleeping=true", "energy regen"]'),
    ]
    c.executemany("INSERT INTO anticipated_transitions VALUES (?, ?, ?, ?, ?)", transitions)

    conn.commit()
    print("Data loaded.")


# =============================================================================
# SQLITE QUERIES (showing complexity)
# =============================================================================


def sqlite_queries(conn):
    c = conn.cursor()
    print("\n" + "=" * 70)
    print("SQLITE QUERIES")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Q1: What rules reference the zombie entity?
    # -------------------------------------------------------------------------
    print("\n[Q1] What rules reference zombie?")
    print("SQL:")
    sql = """
        SELECT id, constraint_text, refs
        FROM rules
        WHERE refs LIKE '%zombie%'
    """
    print(f"  {sql.strip()}")
    c.execute(sql)
    for row in c.fetchall():
        print(f"  -> {row[0]}: {row[1]}")
    print("  NOTE: LIKE search - no referential integrity, could match 'zombie_king'")

    # -------------------------------------------------------------------------
    # Q2: Get full action details for all actions in early_game heuristics
    # -------------------------------------------------------------------------
    print("\n[Q2] Get action details for early_game heuristics")
    print("SQL:")
    sql = """
        SELECT h.id, h.guidance, h.refs
        FROM heuristics h
        WHERE h.phase = 'early_game' AND h.refs LIKE '%actions.%'
    """
    print(f"  {sql.strip()}")
    c.execute(sql)
    heuristics_with_actions = c.fetchall()

    print("  Then for each, extract action name and JOIN:")
    for h_id, _guidance, refs_json in heuristics_with_actions:
        refs = json.loads(refs_json)
        action_refs = [r.replace("actions.", "") for r in refs if r.startswith("actions.")]
        for action_name in action_refs:
            c.execute(
                "SELECT name, preconditions, effects FROM actions WHERE name = ?", (action_name,)
            )
            action = c.fetchone()
            if action:
                print(f"  -> {h_id}: {action[0]} - preconds: {action[1][:50]}...")
    print("  NOTE: Requires Python loop + multiple queries")

    # -------------------------------------------------------------------------
    # Q3: What dynamics are referenced by rules? Get full dynamic details.
    # -------------------------------------------------------------------------
    print("\n[Q3] Get dynamics referenced by rules (with details)")
    print("SQL + Python:")
    sql = "SELECT id, refs FROM rules WHERE refs LIKE '%dynamics.%'"
    print(f"  {sql}")
    c.execute(sql)
    print("  Then parse JSON, extract dynamics refs, query dynamics table:")

    all_dynamics = set()
    for _rule_id, refs_json in c.fetchall():
        refs = json.loads(refs_json)
        for ref in refs:
            if ref.startswith("dynamics."):
                dyn_name = ref.replace("dynamics.", "")
                all_dynamics.add(dyn_name)

    for dyn_name in all_dynamics:
        c.execute("SELECT name, effects FROM dynamics WHERE name = ?", (dyn_name,))
        row = c.fetchone()
        if row:
            print(f"  -> {row[0]}: {row[1][:60]}...")
    print("  NOTE: 3 steps - query rules, parse JSON, query dynamics")

    # -------------------------------------------------------------------------
    # Q4: Get preconditions for all actions in anticipated_transitions
    # -------------------------------------------------------------------------
    print("\n[Q4] Cross-reference anticipated_transitions with actions table")
    print("SQL:")
    sql = """
        SELECT at.action_ref, at.target, a.preconditions as ontology_preconds, at.preconditions as spec_preconds
        FROM anticipated_transitions at
        LEFT JOIN actions a ON at.action_ref = a.name
    """
    print(f"  {sql.strip()}")
    c.execute(sql)
    for row in c.fetchall():
        print(
            f"  -> {row[0]} ({row[1]}): ontology={row[2][:30] if row[2] else 'N/A'}... spec={row[3][:30]}..."
        )
    print("  NOTE: This one is actually clean with JOIN!")

    # -------------------------------------------------------------------------
    # Q5: What entities are referenced by BOTH rules AND heuristics?
    # -------------------------------------------------------------------------
    print("\n[Q5] Entities referenced by both rules AND heuristics")
    print("SQL + Python (very complex):")

    # Get all entity refs from rules
    c.execute("SELECT refs FROM rules")
    rule_entities = set()
    for (refs_json,) in c.fetchall():
        refs = json.loads(refs_json)
        for ref in refs:
            if ref.startswith("entities."):
                rule_entities.add(ref.replace("entities.", ""))

    # Get all entity refs from heuristics
    c.execute("SELECT refs FROM heuristics")
    heuristic_entities = set()
    for (refs_json,) in c.fetchall():
        refs = json.loads(refs_json)
        for ref in refs:
            if ref.startswith("entities."):
                heuristic_entities.add(ref.replace("entities.", ""))

    common = rule_entities & heuristic_entities
    print(f"  Rule entities: {rule_entities}")
    print(f"  Heuristic entities: {heuristic_entities}")
    print(f"  Common: {common}")
    print("  NOTE: No way to do this in pure SQL with JSON arrays")

    # -------------------------------------------------------------------------
    # Q6: Get all items needed for early_game (parse refs, filter items)
    # -------------------------------------------------------------------------
    print("\n[Q6] What items are needed for early_game phase?")
    print("SQL + Python:")
    c.execute("SELECT refs FROM heuristics WHERE phase = 'early_game'")
    items = set()
    for (refs_json,) in c.fetchall():
        refs = json.loads(refs_json)
        for ref in refs:
            if ref.startswith("entities."):
                ent_name = ref.replace("entities.", "")
                c.execute("SELECT type FROM entities WHERE name = ?", (ent_name,))
                row = c.fetchone()
                if row and row[0] == "item":
                    items.add(ent_name)
    print(f"  Items needed: {items}")
    print("  NOTE: Multiple queries per reference to check entity type")


# =============================================================================
# HELIX QUERIES (showing simplicity)
# =============================================================================


def helix_queries():
    print("\n" + "=" * 70)
    print("HELIX QUERIES (what they WOULD look like)")
    print("=" * 70)

    print("""
[Q1] What rules reference zombie?

    HelixQL:
        zombie <- N<Entity>({Name: "zombie"})
        rules <- zombie::In<ReferencedBy>
        RETURN rules

    READS LIKE: "From zombie, follow incoming ReferencedBy edges"

---

[Q2] Get action details for early_game heuristics

    HelixQL:
        heuristics <- N<Heuristic>({Phase: "early_game"})
        actions <- heuristics::Out<References>::WHERE(_::{label}::EQ("Action"))
        RETURN {heuristics, actions}

    READS LIKE: "From early_game heuristics, follow References edges to Actions"

---

[Q3] Get dynamics referenced by rules (with details)

    HelixQL:
        rules <- N<Rule>
        dynamics <- rules::Out<References>::WHERE(_::{label}::EQ("Dynamic"))
        RETURN dynamics

    READS LIKE: "From all rules, follow References to Dynamics"
    ONE QUERY - no parsing, no loops

---

[Q4] Cross-reference anticipated_transitions with actions

    HelixQL:
        transitions <- N<AnticipatedTransition>
        action_details <- transitions::Out<ForAction>
        RETURN {transitions, action_details}

    READS LIKE: "From transitions, follow ForAction edge to get details"

---

[Q5] Entities referenced by BOTH rules AND heuristics

    HelixQL:
        entities <- N<Entity>
        has_rule_ref <- EXISTS(entities::In<ReferencedBy>::WHERE(_::{label}::EQ("Rule")))
        has_heuristic_ref <- EXISTS(entities::In<ReferencedBy>::WHERE(_::{label}::EQ("Heuristic")))
        common <- entities::WHERE(has_rule_ref AND has_heuristic_ref)
        RETURN common

    READS LIKE: "Find entities that have both Rule and Heuristic pointing to them"
    PURE QUERY - no Python needed

---

[Q6] What items are needed for early_game?

    HelixQL:
        heuristics <- N<Heuristic>({Phase: "early_game"})
        entities <- heuristics::Out<References>
        items <- entities::WHERE(_::{Type}::EQ("item"))
        RETURN items

    READS LIKE: "From early_game heuristics, follow refs, filter to items"
    ONE TRAVERSAL - type check is inline
""")


# =============================================================================
# SIDE BY SIDE COMPARISON
# =============================================================================


def comparison_table():
    print("\n" + "=" * 70)
    print("COMPLEXITY COMPARISON")
    print("=" * 70)
    print("""
| Query | SQLite | HelixDB |
|-------|--------|---------|
| Q1: Rules referencing zombie | LIKE '%zombie%' (unsafe) | `zombie::In<ReferencedBy>` |
| Q2: Action details for heuristics | Query + JSON parse + loop + query | `heuristics::Out<References>` |
| Q3: Dynamics from rules | Query + JSON parse + loop + query | `rules::Out<References>` |
| Q4: Cross-ref transitions/actions | JOIN (clean!) | `transitions::Out<ForAction>` |
| Q5: Entities in both rules+heuristics | 2 queries + 2 JSON parses + set ops | Single query with EXISTS |
| Q6: Items for early_game | Query + parse + type-check loop | `heuristics::Out<References>::WHERE` |

KEY INSIGHT:
- SQLite Q4 is clean because it's a direct FK relationship (JOIN works)
- SQLite Q1-3, Q5-6 are messy because refs are JSON arrays (no JOINs possible)
- HelixDB is CONSISTENTLY simple because ALL references are edges
""")


def main():
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    create_sqlite_schema(conn)
    load_sqlite_data(conn)

    sqlite_queries(conn)
    helix_queries()
    comparison_table()

    conn.close()


if __name__ == "__main__":
    main()
