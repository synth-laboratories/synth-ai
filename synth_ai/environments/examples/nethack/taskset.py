"""TaskSet generation for NetHack environment."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from synth_ai.environments.tasks.core import (
    Impetus,
    Intent,
    SplitInfo,
    TaskInstance,
    TaskInstanceMetadata,
    TaskInstanceSet,
)


@dataclass
class NetHackTaskInstanceMetadata(TaskInstanceMetadata):
    """Task-specific metadata for NetHack."""

    character_role: str  # "wizard", "knight", etc.
    starting_level: int  # Dungeon level to start on
    target_depth: int  # Goal depth to reach
    time_limit: int  # Maximum turns
    difficulty: str  # "easy", "medium", "hard"
    special_objectives: List[str]  # Additional goals beyond survival
    seed: Optional[int] = None  # Random seed for reproducibility


@dataclass
class NetHackTaskInstance(TaskInstance):
    """NetHack task instance."""

    async def serialize(self) -> dict:
        """Convert to serializable format."""
        return {
            "id": str(self.id),
            "impetus": {"instructions": self.impetus.instructions},
            "intent": {
                "rubric": self.intent.rubric,
                "gold_trajectories": None,
                "gold_state_diff": self.intent.gold_state_diff,
            },
            "metadata": {
                "character_role": self.metadata.character_role,
                "starting_level": self.metadata.starting_level,
                "target_depth": self.metadata.target_depth,
                "time_limit": self.metadata.time_limit,
                "difficulty": self.metadata.difficulty,
                "special_objectives": self.metadata.special_objectives,
                "seed": self.metadata.seed,
            },
            "is_reproducible": self.is_reproducible,
            "initial_engine_snapshot": None,
        }

    @classmethod
    async def deserialize(cls, data: dict) -> "NetHackTaskInstance":
        """Restore from serialized data."""
        return cls(
            id=uuid4() if not data.get("id") else data["id"],
            impetus=Impetus(instructions=data["impetus"]["instructions"]),
            intent=Intent(
                rubric=data["intent"]["rubric"],
                gold_trajectories=None,
                gold_state_diff=data["intent"]["gold_state_diff"],
            ),
            metadata=NetHackTaskInstanceMetadata(
                character_role=data["metadata"]["character_role"],
                starting_level=data["metadata"]["starting_level"],
                target_depth=data["metadata"]["target_depth"],
                time_limit=data["metadata"]["time_limit"],
                difficulty=data["metadata"]["difficulty"],
                special_objectives=data["metadata"]["special_objectives"],
                seed=data["metadata"].get("seed"),
            ),
            is_reproducible=data.get("is_reproducible", True),
            initial_engine_snapshot=None,
        )


# Character role definitions
CHARACTER_ROLES = {
    "tourist": {
        "description": "A tourist with a camera and Hawaiian shirt",
        "difficulty_modifier": 0.8,  # Easier
        "starting_items": ["camera", "credit card", "hawaiian shirt"],
        "strengths": ["gold finding", "luck"],
        "weaknesses": ["combat", "magic"],
    },
    "knight": {
        "description": "A noble knight in shining armor",
        "difficulty_modifier": 1.0,
        "starting_items": ["long sword", "armor", "shield"],
        "strengths": ["combat", "riding"],
        "weaknesses": ["magic"],
    },
    "wizard": {
        "description": "A powerful wizard with magical abilities",
        "difficulty_modifier": 1.2,
        "starting_items": ["quarterstaff", "spellbook", "cloak"],
        "strengths": ["magic", "identify"],
        "weaknesses": ["physical combat", "low hp"],
    },
    "barbarian": {
        "description": "A fierce barbarian warrior",
        "difficulty_modifier": 0.9,
        "starting_items": ["battle axe", "leather armor"],
        "strengths": ["combat", "hp", "strength"],
        "weaknesses": ["magic", "intelligence"],
    },
    "ranger": {
        "description": "A skilled ranger and tracker",
        "difficulty_modifier": 1.0,
        "starting_items": ["bow", "arrows", "cloak"],
        "strengths": ["ranged combat", "stealth"],
        "weaknesses": ["melee combat"],
    },
    "priest": {
        "description": "A holy priest with divine powers",
        "difficulty_modifier": 1.1,
        "starting_items": ["mace", "robe", "holy water"],
        "strengths": ["healing", "undead turning"],
        "weaknesses": ["edged weapons"],
    },
    "monk": {
        "description": "A disciplined monk with martial arts skills",
        "difficulty_modifier": 1.3,
        "starting_items": ["robe"],
        "strengths": ["martial arts", "speed"],
        "weaknesses": ["armor", "weapons"],
    },
    "rogue": {
        "description": "A stealthy rogue and thief",
        "difficulty_modifier": 1.1,
        "starting_items": ["dagger", "leather armor", "lock pick"],
        "strengths": ["stealth", "backstab", "traps"],
        "weaknesses": ["direct combat"],
    },
}

# Special objectives for variety
SPECIAL_OBJECTIVES = {
    "exploration": [
        "Explore at least 3 different dungeon levels",
        "Find and enter a shop",
        "Discover a special room (vault, zoo, etc.)",
        "Find the entrance to the Gnomish Mines",
    ],
    "combat": [
        "Defeat 10 monsters",
        "Defeat a monster using magic",
        "Defeat a monster using ranged weapons",
        "Survive an encounter with a tough monster",
    ],
    "collection": [
        "Collect 100 gold pieces",
        "Find and identify a magical item",
        "Collect food rations for survival",
        "Find a valuable gem",
    ],
    "survival": [
        "Survive for 500 turns",
        "Maintain full health for 100 turns",
        "Never let hunger status reach 'Weak'",
        "Avoid all traps",
    ],
    "progression": [
        "Reach experience level 3",
        "Improve at least one skill",
        "Successfully pray to your deity",
        "Complete a quest or mission",
    ],
}


async def create_nethack_taskset() -> TaskInstanceSet:
    """Generate diverse NetHack scenarios."""
    instances = []

    # Configuration for different difficulty levels
    DIFFICULTY_CONFIGS = {
        "tutorial": {
            "roles": ["tourist"],
            "target_depth_range": (1, 3),
            "time_limit_range": (500, 1000),
            "objective_count": 1,
            "count": 20,
        },
        "beginner": {
            "roles": ["knight", "barbarian"],
            "target_depth_range": (3, 5),
            "time_limit_range": (1000, 2000),
            "objective_count": 2,
            "count": 30,
        },
        "intermediate": {
            "roles": ["wizard", "ranger", "priest"],
            "target_depth_range": (5, 10),
            "time_limit_range": (2000, 5000),
            "objective_count": 2,
            "count": 25,
        },
        "advanced": {
            "roles": ["monk", "rogue"],
            "target_depth_range": (10, 15),
            "time_limit_range": (5000, 10000),
            "objective_count": 3,
            "count": 15,
        },
        "expert": {
            "roles": list(CHARACTER_ROLES.keys()),
            "target_depth_range": (15, 20),
            "time_limit_range": (10000, 20000),
            "objective_count": 4,
            "count": 10,
        },
    }

    # Generate instances for each difficulty
    for difficulty, config in DIFFICULTY_CONFIGS.items():
        for i in range(config["count"]):
            # Random role selection
            role = random.choice(config["roles"])
            role_info = CHARACTER_ROLES[role]

            # Random parameters within difficulty range
            min_depth, max_depth = config["target_depth_range"]
            target_depth = random.randint(min_depth, max_depth)
            min_time, max_time = config["time_limit_range"]
            time_limit = random.randint(min_time, max_time)

            # Select random objectives
            objectives = []
            objective_categories = list(SPECIAL_OBJECTIVES.keys())
            for _ in range(config["objective_count"]):
                category = random.choice(objective_categories)
                objective = random.choice(SPECIAL_OBJECTIVES[category])
                objectives.append(objective)

            # Create instruction text
            instructions = f"""You are a {role_info["description"]}.

Your primary goal is to descend to dungeon level {target_depth} within {time_limit} turns.

Additional objectives:
{chr(10).join(f"- {obj}" for obj in objectives)}

Character strengths: {", ".join(role_info["strengths"])}
Character weaknesses: {", ".join(role_info["weaknesses"])}

Tips:
- Use 'inventory' to check your items
- Use 'search' to find secret doors
- Eat food before you become weak from hunger
- Save valuable items for when you need them
- Be cautious around unfamiliar monsters

Remember: In NetHack, careful planning often beats hasty action!"""

            # Create success criteria
            rubric = {
                "goal": f"Reach dungeon level {target_depth}",
                "success_criteria": {
                    "primary": f"Reach dungeon level {target_depth} within {time_limit} turns",
                    "secondary": objectives,
                },
                "evaluation_metrics": {
                    "depth_reached": target_depth,
                    "time_limit": time_limit,
                    "objectives_completed": len(objectives),
                },
            }

            # Create metadata
            metadata = NetHackTaskInstanceMetadata(
                character_role=role,
                starting_level=1,
                target_depth=target_depth,
                time_limit=time_limit,
                difficulty=difficulty,
                special_objectives=objectives,
                seed=random.randint(0, 2**31 - 1),
            )

            # Create task instance
            instance = NetHackTaskInstance(
                id=uuid4(),
                impetus=Impetus(instructions=instructions),
                intent=Intent(rubric=rubric, gold_trajectories=None, gold_state_diff={}),
                metadata=metadata,
                is_reproducible=True,
                initial_engine_snapshot=None,
            )

            instances.append(instance)

    # Define splits (80% train, 10% val, 10% test)
    random.shuffle(instances)
    n_instances = len(instances)
    n_val = n_instances // 10
    n_test = n_instances // 10

    val_ids = {inst.id for inst in instances[:n_val]}
    test_ids = {inst.id for inst in instances[n_val : n_val + n_test]}

    split_info = SplitInfo(
        val_instance_ids=val_ids, test_instance_ids=test_ids, _is_split_defined=True
    )

    return TaskInstanceSet(
        name="NetHack TaskSet",
        description="A comprehensive set of NetHack dungeon exploration tasks with varying difficulty levels, character roles, and objectives",
        instances=instances,
        split_info=split_info,
    )


# Module-level export
taskset = create_nethack_taskset
