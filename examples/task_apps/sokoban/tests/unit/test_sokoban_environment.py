"""Unit tests for Sokoban environment and rewards."""
import pytest


@pytest.mark.fast
def test_sokoban_module_imports():
    """Test that Sokoban modules can be imported."""
    from synth_ai.environments.examples.sokoban import environment, engine
    
    assert hasattr(environment, "SokobanEnvironment")
    assert hasattr(engine, "SokobanEngine")


@pytest.mark.asyncio
async def test_sokoban_reward_components():
    """Test that Sokoban reward components exist and work."""
    from synth_ai.environments.examples.sokoban.engine import (
        SokobanEngine,
        SokobanGoalAchievedComponent,
        SokobanStepPenaltyComponent,
        SokobanPublicState,
    )
    from synth_ai.environments.tasks.core import TaskInstance, Impetus, Intent
    
    # Create a minimal task instance
    task = TaskInstance(
        id="test",
        impetus=Impetus(instructions="Test"),
        intent=Intent(
            rubric={"goal": "test"},
            gold_trajectories=None,
            gold_state_diff={},
            deterministic_eval_functions=[],
        ),
        metadata={"difficulty": "easy", "max_steps": 50, "seed": 0},
        is_reproducible=False,
        initial_engine_snapshot=None,
    )
    
    engine = SokobanEngine(task)
    
    # Test that reward components exist
    assert hasattr(engine, "reward_stack")
    assert engine.reward_stack is not None
    
    # Test reward components directly
    goal_reward = SokobanGoalAchievedComponent()
    penalty = SokobanStepPenaltyComponent()
    
    # Mock state for reward calculation
    import numpy as np
    
    state = SokobanPublicState(
        dim_room=(3, 3),
        room_fixed=np.array([[0]]),
        room_state=np.array([[0]]),
        player_position=(0, 0),
        boxes_on_target=0,
        num_steps=0,
        max_steps=50,
        last_action_name="NONE",
        num_boxes=1,
        error_info=None,
    )
    
    # Test goal reward (should be 0 for incomplete puzzle)
    reward1 = await goal_reward.score(state, {"action": 0})
    assert reward1 == 0.0
    
    # Test completed state
    state_complete = SokobanPublicState(
        dim_room=(3, 3),
        room_fixed=np.array([[0]]),
        room_state=np.array([[0]]),
        player_position=(0, 0),
        boxes_on_target=1,
        num_steps=10,
        max_steps=50,
        last_action_name="RIGHT",
        num_boxes=1,
        error_info=None,
    )
    reward_complete = await goal_reward.score(state_complete, {"action": 0})
    assert reward_complete > 0
    
    # Test penalty (should be negative small value)
    penalty_reward = await penalty.score(state, {"action": 0})
    assert penalty_reward < 0
    assert penalty_reward > -1  # Should be a small penalty


def test_sokoban_difficulty_settings():
    """Test that Sokoban engine can be created with task metadata."""
    from synth_ai.environments.examples.sokoban.engine import SokobanEngine
    from synth_ai.environments.tasks.core import TaskInstance, Impetus, Intent
    
    # Test with different difficulty metadata
    for difficulty in ["easy", "medium", "hard"]:
        task = TaskInstance(
            id="test",
            impetus=Impetus(instructions="Test"),
            intent=Intent(
                rubric={"goal": "test"},
                gold_trajectories=None,
                gold_state_diff={},
                deterministic_eval_functions=[],
            ),
            metadata={"difficulty": difficulty, "max_steps": 50, "seed": 0},
            is_reproducible=False,
            initial_engine_snapshot=None,
        )
        
        engine = SokobanEngine(task)
        assert engine is not None
