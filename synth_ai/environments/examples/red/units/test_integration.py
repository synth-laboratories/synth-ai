import pytest
from synth_ai.environments.examples.red.environment import PokemonRedEnvironment
from synth_ai.environments.examples.red.taskset import INSTANCE as POKEMON_TASK
from synth_ai.environments.environment.tools import EnvToolCall


class TestPokemonRedIntegration:
    """Integration tests for Pokemon Red environment with REAL ROM"""

    @pytest.mark.asyncio
    async def test_full_workflow_real(self):
        """Test complete workflow from initialization to termination with REAL ROM"""
        # Initialize environment with real ROM
        env = PokemonRedEnvironment(POKEMON_TASK)

        # Test initialization
        obs = await env.initialize()
        assert "position" in obs
        assert "badges_earned" in obs
        # Note: badges_earned might be 0 or could have some initial value from ROM
        assert isinstance(obs["badges_earned"], int)

        # Test series of actions
        actions = [
            EnvToolCall(tool="press_button", args={"button": "RIGHT", "frames": 1}),
            EnvToolCall(tool="press_button", args={"button": "UP", "frames": 2}),
            EnvToolCall(tool="press_button", args={"button": "A", "frames": 1}),
        ]

        for action in actions:
            obs = await env.step(action)
            assert "step_count" in obs
            assert "total_reward" in obs
            assert isinstance(obs["step_count"], int)
            assert isinstance(obs["total_reward"], float)

        # Test checkpointing
        checkpoint_obs = await env.checkpoint()
        assert "engine_snapshot_data" in checkpoint_obs

        # Test termination
        final_obs = await env.terminate()
        assert final_obs["terminated"] is True

    @pytest.mark.asyncio
    async def test_button_sequence_real(self):
        """Test sequence of different button presses with real ROM"""
        env = PokemonRedEnvironment(POKEMON_TASK)
        await env.initialize()

        # Test all basic buttons
        buttons = ["A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT"]

        for i, button in enumerate(buttons):
            action = EnvToolCall(tool="press_button", args={"button": button, "frames": 1})
            obs = await env.step(action)

            assert obs["step_count"] == i + 1
            assert "position" in obs
            assert "badges_earned" in obs
            assert "hp_status" in obs
            assert "party_level" in obs

    @pytest.mark.asyncio
    async def test_multiple_frame_actions_real(self):
        """Test actions with multiple frames using real ROM"""
        env = PokemonRedEnvironment(POKEMON_TASK)
        await env.initialize()

        # Test holding buttons for multiple frames
        action = EnvToolCall(tool="press_button", args={"button": "RIGHT", "frames": 5})
        obs = await env.step(action)

        assert obs["step_count"] == 1  # Should count as one step
        assert "position" in obs

        # Test another multi-frame action
        action = EnvToolCall(tool="press_button", args={"button": "A", "frames": 3})
        obs = await env.step(action)

        assert obs["step_count"] == 2

    @pytest.mark.asyncio
    async def test_state_consistency_real(self):
        """Test that game state remains consistent across steps with real ROM"""
        env = PokemonRedEnvironment(POKEMON_TASK)

        obs1 = await env.initialize()
        initial_position = obs1["position"]
        initial_badges = obs1["badges_earned"]
        initial_hp = obs1["hp_status"]

        # Take some actions
        for _ in range(5):
            action = EnvToolCall(tool="press_button", args={"button": "A", "frames": 1})
            obs = await env.step(action)

            # State should remain valid even if unchanged
            assert "position" in obs
            assert "badges_earned" in obs
            assert "hp_status" in obs
            assert isinstance(obs["badges_earned"], int)

    @pytest.mark.asyncio
    async def test_reward_accumulation_real(self):
        """Test that rewards accumulate properly with real ROM"""
        env = PokemonRedEnvironment(POKEMON_TASK)

        obs = await env.initialize()
        initial_reward = obs["total_reward"]

        # Take several steps and track reward changes
        for i in range(3):
            action = EnvToolCall(tool="press_button", args={"button": "DOWN", "frames": 1})
            obs = await env.step(action)

            # Reward should change (likely negative step penalty)
            assert obs["total_reward"] != initial_reward
            assert isinstance(obs["total_reward"], float)
            assert obs["step_count"] == i + 1

    @pytest.mark.asyncio
    async def test_checkpointing_real(self):
        """Test checkpointing functionality with real ROM"""
        env = PokemonRedEnvironment(POKEMON_TASK)

        # Initialize and take some steps
        await env.initialize()
        action = EnvToolCall(tool="press_button", args={"button": "RIGHT", "frames": 1})
        await env.step(action)

        # Create checkpoint
        checkpoint_obs = await env.checkpoint()

        assert "engine_snapshot_data" in checkpoint_obs
        snapshot = checkpoint_obs["engine_snapshot_data"]
        assert "state_data" in snapshot
        assert "total_reward" in snapshot
        assert "step_count" in snapshot
        assert isinstance(snapshot["total_reward"], float)
        assert isinstance(snapshot["step_count"], int)

    @pytest.mark.asyncio
    async def test_invalid_button_handling_real(self):
        """Test handling of invalid buttons with real ROM"""
        env = PokemonRedEnvironment(POKEMON_TASK)
        await env.initialize()

        # Try invalid button
        action = EnvToolCall(tool="press_button", args={"button": "INVALID", "frames": 1})

        # Should handle gracefully and return valid observation
        obs = await env.step(action)
        assert "position" in obs
        assert "step_count" in obs

    @pytest.mark.asyncio
    async def test_observation_format_real(self):
        """Test that observations have expected format with real ROM"""
        env = PokemonRedEnvironment(POKEMON_TASK)

        obs = await env.initialize()

        # Check required observation keys (based on actual observation format)
        required_keys = [
            "position",
            "badges_earned",
            "badges_bitfield",
            "hp_status",
            "party_level",
            "party_xp",
            "in_battle",
            "step_count",
            "reward_last_step",
            "total_reward",
            "terminated",
        ]

        for key in required_keys:
            assert key in obs, f"Missing key: {key}"

        # Check types
        assert isinstance(obs["position"], str)
        assert isinstance(obs["badges_earned"], int)
        assert isinstance(obs["badges_bitfield"], int)
        assert isinstance(obs["hp_status"], str)
        assert isinstance(obs["party_level"], int)
        assert isinstance(obs["party_xp"], int)
        assert isinstance(obs["in_battle"], bool)
        assert isinstance(obs["step_count"], int)
        assert isinstance(obs["reward_last_step"], float)
        assert isinstance(obs["total_reward"], float)
        assert isinstance(obs["terminated"], bool)

    @pytest.mark.asyncio
    async def test_rom_memory_integration_real(self):
        """Test that we can access and read ROM memory consistently"""
        env = PokemonRedEnvironment(POKEMON_TASK)
        await env.initialize()

        # Should be able to access engine and emulator
        assert env.engine is not None
        assert env.engine.emulator is not None
        assert hasattr(env.engine.emulator, "memory")

        # Memory reads should be consistent
        memory = env.engine.emulator.memory
        badge_flags1 = memory[0xD356]
        badge_flags2 = memory[0xD356]
        assert badge_flags1 == badge_flags2  # Should be deterministic

        # After taking an action, memory should still be accessible
        action = EnvToolCall(tool="press_button", args={"button": "A", "frames": 1})
        await env.step(action)

        badge_flags3 = memory[0xD356]
        assert isinstance(badge_flags3, int)  # Should still be valid
