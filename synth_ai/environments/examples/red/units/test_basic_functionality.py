import pytest
from synth_ai.environments.examples.red.environment import PokemonRedEnvironment
from synth_ai.environments.examples.red.taskset import INSTANCE as POKEMON_TASK
from synth_ai.environments.environment.tools import EnvToolCall


class PressButtonCall(EnvToolCall):
    """Helper class for creating button press calls"""

    def __init__(self, button: str, frames: int = 1):
        super().__init__(tool="press_button", args={"button": button, "frames": frames})


@pytest.mark.asyncio
async def test_pokemon_red_basic():
    """Test basic Pokemon Red environment functionality"""
    env = PokemonRedEnvironment(POKEMON_TASK)

    # Initialize environment
    obs = await env.initialize()
    assert "position" in obs
    assert "badges_earned" in obs
    assert obs["badges_earned"] == 0  # Should start with no badges

    # Test a few button presses
    obs = await env.step(PressButtonCall("A"))
    assert "step_count" in obs
    assert obs["step_count"] == 1

    obs = await env.step(PressButtonCall("RIGHT", 2))
    assert obs["step_count"] == 2

    # Test termination
    final_obs = await env.terminate()
    assert final_obs["terminated"] is True


@pytest.mark.asyncio
async def test_pokemon_red_multiple_actions():
    """Test sequence of actions in Pokemon Red"""
    env = PokemonRedEnvironment(POKEMON_TASK)

    obs = await env.initialize()
    initial_reward = obs["total_reward"]

    # Sequence of movements and actions
    actions = [
        PressButtonCall("RIGHT"),
        PressButtonCall("UP"),
        PressButtonCall("A"),
        PressButtonCall("DOWN"),
        PressButtonCall("B"),
    ]

    for action in actions:
        obs = await env.step(action)
        assert "position" in obs
        assert "hp_status" in obs
        assert "party_level" in obs

    # Should have accumulated some reward (mostly negative from step penalty)
    assert obs["total_reward"] <= initial_reward  # Step penalties
    assert obs["step_count"] == len(actions)


@pytest.mark.asyncio
async def test_pokemon_red_checkpointing():
    """Test environment checkpointing functionality"""
    env = PokemonRedEnvironment(POKEMON_TASK)

    # Initialize and take some steps
    await env.initialize()
    await env.step(PressButtonCall("RIGHT"))
    await env.step(PressButtonCall("A"))

    # Create checkpoint
    checkpoint_obs = await env.checkpoint()
    assert "engine_snapshot_data" in checkpoint_obs
    assert checkpoint_obs["step_count"] == 2

    # Verify checkpoint contains expected data
    snapshot_data = checkpoint_obs["engine_snapshot_data"]
    assert "state_data" in snapshot_data
    assert "total_reward" in snapshot_data
    assert "step_count" in snapshot_data


@pytest.mark.asyncio
async def test_pokemon_red_invalid_button():
    """Test handling of invalid button inputs"""
    env = PokemonRedEnvironment(POKEMON_TASK)
    await env.initialize()

    # Test with invalid button - should handle gracefully
    obs = await env.step(PressButtonCall("INVALID_BUTTON"))
    # Should still return valid observation even if action failed
    assert "position" in obs
