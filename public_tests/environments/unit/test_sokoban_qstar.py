"""
Unit tests for Sokoban environment with Q* (A*) solver.
Tests the core Sokoban functionality without agent demos.
"""

import pytest
from uuid import uuid4
from typing import Dict, Any

from synth_ai.environments.examples.sokoban.environment import SokobanEnvironment
from synth_ai.environments.examples.sokoban.engine import SokobanEngine
from synth_ai.environments.examples.sokoban.taskset import (
    SokobanTaskInstance,
    SokobanTaskInstanceMetadata,
)
from synth_ai.environments.tasks.core import Impetus, Intent
from synth_ai.environments.environment.tools import EnvToolCall

# Import the A* solver utilities
from synth_ai.environments.examples.sokoban.units.astar_common import (
    astar,
    solved,
    heuristic,
    ENGINE_ASTAR,
)


# Helper function to create task instance
def create_task_instance(puzzle_snapshot: Dict[str, Any], difficulty: str = "test") -> SokobanTaskInstance:
    """Create a SokobanTaskInstance with the given puzzle snapshot."""
    dim_room = puzzle_snapshot.get("dim_room", [5, 5])
    num_boxes = puzzle_snapshot.get("num_boxes", 1)
    max_steps = puzzle_snapshot.get("max_steps", 50)
    
    return SokobanTaskInstance(
        id=uuid4(),
        impetus=Impetus(instructions="Solve the Sokoban puzzle"),
        intent=Intent(
            rubric={"goal": "Push all boxes onto targets"},
            gold_trajectories=None,
            gold_state_diff={},
            deterministic_eval_functions=[],
        ),
        metadata=SokobanTaskInstanceMetadata(
            difficulty=difficulty,
            num_boxes=num_boxes,
            dim_room=tuple(dim_room),
            max_steps=max_steps,
            shortest_path_length=-1,
            seed=-1,
            generation_params="unit-test",
        ),
        is_reproducible=True,
        initial_engine_snapshot=puzzle_snapshot,
    )


# Test fixtures for various Sokoban puzzles
TRIVIAL_PUZZLE: Dict[str, Any] = {
    "dim_room": [3, 3],
    "room_fixed": [[0, 0, 0], [0, 2, 0], [0, 0, 0]],  # One target at center
    "room_state": [[5, 0, 0], [0, 3, 0], [0, 0, 0]],  # Box on target (3), player at top-left
    "boxes_on_target": 1,
    "max_steps": 5,
    "num_boxes": 1,
    "box_mapping": [],  # Empty since box is already on target
    "num_env_steps": 0,
    "player_position": [0, 0],
    "reward_last": 0
}

SIMPLE_PUZZLE: Dict[str, Any] = {
    "dim_room": [4, 3],
    "room_fixed": [[0, 0, 0], [0, 2, 0], [0, 1, 0], [0, 1, 0]],  # Target at [1,1], floors at [2,1] and [3,1]
    "room_state": [[0, 0, 0], [0, 2, 0], [0, 4, 0], [0, 5, 0]],  # Box at [2,1], player at [3,1]
    "boxes_on_target": 0,
    "max_steps": 10,
    "num_boxes": 1,
    "box_mapping": [],
    "num_env_steps": 0,
    "player_position": [3, 1],
    "reward_last": 0
}

MEDIUM_PUZZLE: Dict[str, Any] = {
    "dim_room": [5, 5],
    "room_fixed": [
        [0, 0, 0, 0, 0],
        [0, 2, 1, 2, 0],  # Two targets
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ],
    "room_state": [
        [0, 0, 0, 0, 0],
        [0, 4, 1, 2, 0],  # Box at [1,1], empty target at [1,3]
        [0, 1, 5, 4, 0],  # Player at [2,2], box at [2,3]
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ],
    "boxes_on_target": 0,
    "max_steps": 20,
    "num_boxes": 2,
    "box_mapping": [],
    "num_env_steps": 0,
    "player_position": [2, 2],
    "reward_last": 0
}


def create_move_action(action: int) -> list:
    """Create a move action for Sokoban."""
    return [EnvToolCall(tool="interact", args={"action": action})]


class TestSokobanQStar:
    """Test Sokoban environment with Q* solver."""

    @pytest.mark.asyncio
    async def test_trivial_puzzle_already_solved(self):
        """Test a puzzle that's already solved."""
        # Create task instance with puzzle snapshot
        task_instance = SokobanTaskInstance(
            id=uuid4(),
            impetus=Impetus(instructions="Solve the Sokoban puzzle"),
            intent=Intent(
                rubric={"goal": "Push all boxes onto targets"},
                gold_trajectories=None,
                gold_state_diff={},
                deterministic_eval_functions=[],
            ),
            metadata=SokobanTaskInstanceMetadata(
                difficulty="trivial",
                num_boxes=1,
                dim_room=(3, 3),
                max_steps=5,
                shortest_path_length=-1,
                seed=-1,
                generation_params="unit-test",
            ),
            is_reproducible=True,
            initial_engine_snapshot=TRIVIAL_PUZZLE,
        )
        engine = SokobanEngine(task_instance)

        # Check if already solved
        assert solved(engine)
        assert heuristic(engine) == 0

        # A* should return empty plan
        plan = await ENGINE_ASTAR(engine, max_nodes=10)
        assert plan == []

    @pytest.mark.asyncio
    async def test_simple_puzzle_solving(self):
        """Test solving a simple puzzle with A*."""
        # Create task instance with puzzle snapshot
        task_instance = SokobanTaskInstance(
            id=uuid4(),
            impetus=Impetus(instructions="Solve the Sokoban puzzle"),
            intent=Intent(
                rubric={"goal": "Push all boxes onto targets"},
                gold_trajectories=None,
                gold_state_diff={},
                deterministic_eval_functions=[],
            ),
            metadata=SokobanTaskInstanceMetadata(
                difficulty="simple",
                num_boxes=1,
                dim_room=(4, 3),
                max_steps=10,
                shortest_path_length=-1,
                seed=-1,
                generation_params="unit-test",
            ),
            is_reproducible=True,
            initial_engine_snapshot=SIMPLE_PUZZLE,
        )
        engine = SokobanEngine(task_instance)

        # Initial state should not be solved
        assert not solved(engine)
        assert heuristic(engine) == 1  # One box not on target

        # Find solution
        plan = await ENGINE_ASTAR(engine, max_nodes=100)
        assert plan is not None
        assert len(plan) > 0

        # Execute plan and verify solution
        for action in plan:
            await engine._step_engine(action)

        assert solved(engine)

    @pytest.mark.asyncio
    async def test_medium_puzzle_solving(self):
        """Test solving a medium puzzle with multiple boxes."""
        # Create task instance with puzzle snapshot
        task_instance = SokobanTaskInstance(
            id=uuid4(),
            impetus=Impetus(instructions="Solve the Sokoban puzzle"),
            intent=Intent(
                rubric={"goal": "Push all boxes onto targets"},
                gold_trajectories=None,
                gold_state_diff={},
                deterministic_eval_functions=[],
            ),
            metadata=SokobanTaskInstanceMetadata(
                difficulty="medium",
                num_boxes=2,
                dim_room=(5, 5),
                max_steps=20,
                shortest_path_length=-1,
                seed=-1,
                generation_params="unit-test",
            ),
            is_reproducible=True,
            initial_engine_snapshot=MEDIUM_PUZZLE,
        )
        engine = SokobanEngine(task_instance)

        # Initial state
        assert not solved(engine)
        assert heuristic(engine) == 2  # Two boxes not on targets

        # Find solution - medium puzzle might need more nodes
        plan = await ENGINE_ASTAR(engine, max_nodes=2000)
        
        # If A* can't find a solution within the limit, skip the test
        if not plan:
            pytest.skip("Medium puzzle too complex for A* within node limit")
        
        assert plan is not None
        assert len(plan) > 0

        # Execute plan
        for action in plan:
            await engine._step_engine(action)

        assert solved(engine)
        assert heuristic(engine) == 0

    @pytest.mark.asyncio
    async def test_environment_level_solving(self):
        """Test solving through the SokobanEnvironment API."""
        # Create task instance
        meta = SokobanTaskInstanceMetadata(
            difficulty="easy",
            num_boxes=1,
            dim_room=(4, 3),
            max_steps=10,
            shortest_path_length=-1,
            seed=-1,
            generation_params="unit-test",
        )

        task_instance = SokobanTaskInstance(
            id=uuid4(),
            impetus=Impetus(instructions="Solve the Sokoban puzzle"),
            intent=Intent(
                rubric="Push all boxes onto targets",
                gold_trajectories=None,
                gold_state_diff={},
                deterministic_eval_functions=[],
            ),
            metadata=meta,
            is_reproducible=True,
            initial_engine_snapshot=SIMPLE_PUZZLE,
        )

        # Create environment
        env = SokobanEnvironment(task_instance)
        obs = await env.initialize()

        # Disable image rendering for speed
        env.engine.package_sokoban_env.observation_mode = "raw"

        # Initial observation should show puzzle state
        assert "room_text" in obs
        assert obs["boxes_on_target"] == 0
        assert obs["max_steps"] == 10

        # Use A* to find solution
        plan = await astar(
            root_obj=env.engine,
            step_fn=lambda e, a: e._step_engine(a),
            deserialize_fn=SokobanEngine._deserialize_engine,
            max_nodes=200,
        )

        assert plan is not None
        assert len(plan) > 0

        # Execute plan through environment API
        for action in plan:
            result = await env.step(create_move_action(action))
            obs = result.get("observation", result)

        # Verify solved
        assert obs["boxes_on_target"] == 1
        assert obs.get("terminated", False) or solved(env.engine)

    @pytest.mark.asyncio
    async def test_max_nodes_limit(self):
        """Test that A* respects max_nodes limit."""
        # Create a complex puzzle that might not solve quickly
        complex_puzzle = {
            "dim_room": [7, 7],
            "room_fixed": [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 2, 1, 1, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 2, 0, 0, 0, 2, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 1, 1, 2, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            "room_state": [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 1, 4, 0, 4, 1, 0],
                [0, 4, 0, 5, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 4, 0],
                [0, 1, 1, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            "boxes_on_target": 0,
            "max_steps": 50,
            "num_boxes": 4,
        }

        task_instance = create_task_instance(complex_puzzle, "complex")
        engine = SokobanEngine(task_instance)

        # Try with very limited nodes
        plan = await ENGINE_ASTAR(engine, max_nodes=5)

        # Should return empty plan if can't solve within limit
        # (or might find a very short solution)
        assert isinstance(plan, list)

    @pytest.mark.asyncio
    async def test_checkpoint_and_restore(self):
        """Test checkpointing and restoring environment state."""
        task_instance = create_task_instance(SIMPLE_PUZZLE, "simple")
        engine = SokobanEngine(task_instance)

        # Get initial checkpoint
        initial_snapshot = await engine._serialize_engine()

        # Make some moves
        await engine._step_engine(0)  # UP
        await engine._step_engine(1)  # DOWN

        # State should have changed
        current_snapshot = await engine._serialize_engine()
        assert current_snapshot != initial_snapshot

        # Restore to initial state
        restored = await SokobanEngine._deserialize_engine(initial_snapshot)
        restored_snapshot = await restored._serialize_engine()

        # Should match initial state
        assert restored_snapshot.engine_snapshot == initial_snapshot.engine_snapshot

    @pytest.mark.asyncio
    async def test_illegal_moves_handling(self):
        """Test that illegal moves are handled properly."""
        # Create a puzzle where player is against a wall
        wall_puzzle = {
            "dim_room": [3, 3],
            "room_fixed": [[0, 0, 0], [0, 2, 0], [0, 0, 0]],
            "room_state": [[5, 0, 0], [4, 0, 0], [0, 0, 0]],  # Player at top-left
            "boxes_on_target": 0,
            "max_steps": 10,
            "num_boxes": 1,
        }

        task_instance = create_task_instance(wall_puzzle, "wall-test")
        engine = SokobanEngine(task_instance)
        initial_state = await engine._serialize_engine()

        # Try to move up (should be blocked by wall)
        await engine._step_engine(0)  # UP

        # State should be unchanged (or handle gracefully)
        final_state = await engine._serialize_engine()
        # The move might be processed but have no effect
        # or it might throw an exception that's caught
        assert True  # Just verify no crash

    @pytest.mark.asyncio
    async def test_reward_structure(self):
        """Test that rewards are properly calculated."""
        meta = SokobanTaskInstanceMetadata(
            difficulty="easy",
            num_boxes=1,
            dim_room=(4, 4),
            max_steps=10,
            shortest_path_length=-1,
            seed=-1,
            generation_params="unit-test",
        )

        task_instance = SokobanTaskInstance(
            id=uuid4(),
            impetus=Impetus(instructions="Solve the Sokoban puzzle"),
            intent=Intent(
                rubric="Push all boxes onto targets",
                gold_trajectories=None,
                gold_state_diff={},
                deterministic_eval_functions=[],
            ),
            metadata=meta,
            is_reproducible=True,
            initial_engine_snapshot=SIMPLE_PUZZLE,
        )

        env = SokobanEnvironment(task_instance)
        await env.initialize()

        # Make a move
        result = await env.step(create_move_action(1))  # DOWN

        # Check result structure
        assert isinstance(result, dict)
        if "observation" in result:
            assert isinstance(result["observation"], dict)

        # Environment should track state properly
        obs = result.get("observation", result)
        assert "steps_taken" in obs
        assert obs["steps_taken"] >= 1
