from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from synth_ai.environments.tasks.core import (
    Impetus,
    Intent,
    Task,
    TaskInstance,
    TaskInstanceMetadata,
)

# Define the main task for Pokemon Red
TASK = Task(
    global_premises="You are playing Pokemon Red. Start in Pewter City with a level-10 Pikachu.",
    global_constraints="No glitches or exploits. Play within normal game mechanics.",
    global_objectives="Defeat Brock at the Pewter Gym to earn the Boulder Badge.",
    shared_env_params={},
)

# Path to initial save state (would contain a save near Pewter Gym)
INITIAL_SNAPSHOT = Path(__file__).parent / "snapshots" / "pewter_start.state"


@dataclass
class PokemonRedTaskInstance(TaskInstance):
    """Task instance for Pokemon Red challenges"""

    async def serialize(self) -> dict:
        """Serialize the task instance to a dictionary"""
        return {
            "id": str(self.id),
            "impetus": {"instructions": self.impetus.instructions},
            "intent": {
                "rubric": self.intent.rubric,
                "gold_trajectories": None,
                "gold_state_diff": self.intent.gold_state_diff,
            },
            "metadata": {},
            "is_reproducible": self.is_reproducible,
            "initial_engine_snapshot": str(self.initial_engine_snapshot)
            if self.initial_engine_snapshot
            else None,
        }

    @classmethod
    async def deserialize(cls, data: dict) -> "PokemonRedTaskInstance":
        """Deserialize a task instance from a dictionary"""
        return cls(
            id=uuid.UUID(data["id"]),
            impetus=Impetus(instructions=data["impetus"]["instructions"]),
            intent=Intent(
                rubric=data["intent"]["rubric"],
                gold_trajectories=None,
                gold_state_diff=data["intent"]["gold_state_diff"],
            ),
            metadata=TaskInstanceMetadata(),
            is_reproducible=data["is_reproducible"],
            initial_engine_snapshot=None,
        )


# Main task instance - beat Brock for Boulder Badge
INSTANCE = PokemonRedTaskInstance(
    id=uuid.UUID("12345678-1234-5678-9abc-123456789abc"),
    impetus=Impetus(
        instructions="Navigate to Pewter Gym and defeat Brock to earn the Boulder Badge. Use strategic Pokemon battles and item management."
    ),
    intent=Intent(
        rubric="Successfully obtain the Boulder Badge by defeating Brock at Pewter Gym. Efficiency measured by minimal steps and strategic Pokemon usage.",
        gold_trajectories=None,
        gold_state_diff={"badges": 1},
    ),
    metadata=TaskInstanceMetadata(),
    is_reproducible=True,
    initial_engine_snapshot=INITIAL_SNAPSHOT if INITIAL_SNAPSHOT.exists() else None,
)
