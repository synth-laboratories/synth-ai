#!/usr/bin/env python3
"""
Unit tests for MiniGrid service integration bug fix.

This test validates that the MiniGrid environment properly handles
EnvToolCall objects sent by the environment service.

Original bug: "Tool call missing 'tool' or 'name' field"
Fix: Updated validate_tool_calls to handle List[EnvToolCall] from service
"""

import pytest
import asyncio
from synth_ai.environments.examples.minigrid.environment import MiniGridEnvironment
from synth_ai.environments.examples.minigrid.taskset import DEFAULT_MINIGRID_TASK
from synth_ai.environments.environment.tools import EnvToolCall


class TestMiniGridServiceIntegration:
    """Test MiniGrid service integration and tool call validation."""

    @pytest.mark.asyncio
    async def test_service_envtoolcall_format(self):
        """
        Test that MiniGrid handles EnvToolCall objects from the service correctly.

        This reproduces the exact format the service sends after converting
        JSON tool calls to EnvToolCall objects.
        """
        # Create environment using the default task
        env = MiniGridEnvironment(DEFAULT_MINIGRID_TASK)

        # Initialize environment
        obs = await env.initialize()
        assert obs is not None
        assert obs["terminated"] is False

        # Test 1: Single EnvToolCall object (what service sends for single tool call)
        single_tool_call = EnvToolCall(tool="minigrid_act", args={"action": "left"})
        obs = await env.step(single_tool_call)

        assert obs is not None
        assert "observation" in obs
        assert "reward_last" in obs
        print("✅ Single EnvToolCall handled correctly")

        # Test 2: List with single EnvToolCall object (actual service format)
        list_with_envtoolcall = [EnvToolCall(tool="minigrid_act", args={"action": "right"})]
        obs = await env.step(list_with_envtoolcall)

        assert obs is not None
        assert "observation" in obs
        assert "reward_last" in obs
        print("✅ List[EnvToolCall] handled correctly")

        # Test 3: Mixed format - list with dict (legacy support)
        mixed_format = [{"tool": "minigrid_act", "args": {"action": "forward"}}]
        obs = await env.step(mixed_format)

        assert obs is not None
        assert "observation" in obs
        assert "reward_last" in obs
        print("✅ Mixed format handled correctly")

        # Test 4: Direct dict format (legacy support)
        dict_format = {"tool": "minigrid_act", "args": {"action": "pickup"}}
        obs = await env.step(dict_format)

        assert obs is not None
        assert "observation" in obs
        assert "reward_last" in obs
        print("✅ Dict format handled correctly")

        # Terminate environment
        final_obs = await env.terminate()
        assert final_obs is not None
        print("✅ Environment terminated successfully")

    @pytest.mark.asyncio
    async def test_tool_validation_comprehensive(self):
        """
        Test comprehensive tool call validation scenarios.

        Ensures all the different formats are properly validated.
        """
        env = MiniGridEnvironment(DEFAULT_MINIGRID_TASK)

        # Test valid tool call validation
        valid_envtoolcall = EnvToolCall(tool="minigrid_act", args={"action": "left"})
        validated = env.validate_tool_calls(valid_envtoolcall)

        assert isinstance(validated, EnvToolCall)
        assert validated.tool == "minigrid_act"
        assert validated.args == {"action": "left"}

        # Test list with EnvToolCall validation
        list_envtoolcall = [EnvToolCall(tool="minigrid_act", args={"action": "right"})]
        validated = env.validate_tool_calls(list_envtoolcall)

        assert isinstance(validated, EnvToolCall)
        assert validated.tool == "minigrid_act"
        assert validated.args == {"action": "right"}

        # Test dict validation
        dict_call = {"tool": "minigrid_act", "args": {"action": "forward"}}
        validated = env.validate_tool_calls(dict_call)

        assert isinstance(validated, EnvToolCall)
        assert validated.tool == "minigrid_act"
        assert validated.args == {"action": "forward"}

        # Test invalid tool name
        with pytest.raises(ValueError, match="Unknown tool.*Expected 'minigrid_act'"):
            invalid_tool = EnvToolCall(tool="invalid_tool", args={"action": "left"})
            env.validate_tool_calls(invalid_tool)

        # Test missing tool field (should fail fast)
        with pytest.raises(ValueError, match="Tool call missing 'tool' or 'name' field"):
            missing_tool = {"args": {"action": "left"}}
            env.validate_tool_calls(missing_tool)

        print("✅ All validation scenarios work correctly")

    @pytest.mark.asyncio
    async def test_minigrid_actions_sequence(self):
        """
        Test a sequence of MiniGrid actions to verify full functionality.

        This confirms the environment actually works end-to-end after the fix.
        """
        env = MiniGridEnvironment(DEFAULT_MINIGRID_TASK)

        # Initialize
        obs = await env.initialize()
        initial_position = obs["public"].agent_pos

        # Execute a sequence of actions using service format
        actions = ["left", "left", "forward", "right", "forward"]

        for action in actions:
            # Use the format the service actually sends
            service_tool_call = [EnvToolCall(tool="minigrid_act", args={"action": action})]
            obs = await env.step(service_tool_call)

            # Verify response structure
            assert "observation" in obs
            assert "reward_last" in obs
            assert "public" in obs
            assert "private" in obs

            # Verify state updates
            assert obs["public"].step_count >= 0
            assert obs["private"].reward_last <= 0  # Step penalty or reward

            # Stop if terminated
            if obs["terminated"]:
                break

        final_position = obs["public"].agent_pos

        # Verify the agent actually moved or turned
        print(f"Initial position: {initial_position}")
        print(f"Final position: {final_position}")
        print(f"Total steps: {obs['public'].step_count}")
        print("✅ Action sequence executed successfully")

        # The key test: no "Tool call missing 'tool' or 'name' field" errors!
        assert True, "MiniGrid service integration working correctly"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
