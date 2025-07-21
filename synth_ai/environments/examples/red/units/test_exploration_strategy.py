import pytest
import asyncio
import uuid

from synth_ai.environments.examples.red.environment import (
    PokemonRedEnvironment,
    PokemonRedPublicState,
    PokemonRedPrivateState,
)
from synth_ai.environments.environment.shared_engine import (
    GetObservationCallable,
    InternalObservation,
)
from synth_ai.environments.examples.red.taskset import PokemonRedTaskInstance
from synth_ai.environments.tasks.core import Impetus, Intent, TaskInstanceMetadata
from synth_ai.environments.environment.tools import EnvToolCall


class PressButtonCall(EnvToolCall):
    """Helper class for creating button press calls"""

    def __init__(self, button: str, frames: int = 1):
        super().__init__(tool="press_button", args={"button": button, "frames": frames})


class ExplorationObservationCallable(GetObservationCallable):
    """Observation callable for exploration testing"""

    def __init__(self):
        self.screen_buffer = None

    async def get_observation(
        self, pub: PokemonRedPublicState, priv: PokemonRedPrivateState
    ) -> InternalObservation:
        if pub is None or priv is None:
            raise RuntimeError("Missing public or private state in get_observation")

        formatted_obs = (
            f"Step: {pub.step_count}, Position: ({pub.player_x}, {pub.player_y}), Map: {pub.map_id}"
        )

        return {
            "public": pub,
            "private": priv,
            "formatted_obs": formatted_obs,
            "screen_buffer": self.screen_buffer,
        }


async def test_exploration_when_stuck():
    """
    Test what happens when we try different buttons in the initial game state.
    """
    print("\n" + "=" * 80)
    print("EXPLORATION STRATEGY TEST - FINDING AVAILABLE ACTIONS")
    print("=" * 80)

    # Create a task instance
    task_metadata = TaskInstanceMetadata()
    inst = PokemonRedTaskInstance(
        id=uuid.uuid4(),
        impetus=Impetus(instructions="Explore available actions in initial state."),
        intent=Intent(
            rubric={"goal": "Find working actions"},
            gold_trajectories=None,
            gold_state_diff={},
        ),
        metadata=task_metadata,
        is_reproducible=True,
        initial_engine_snapshot=None,
    )

    exploration_obs = ExplorationObservationCallable()
    env = PokemonRedEnvironment(inst, custom_step_obs=exploration_obs)

    try:
        # Initialize environment
        print("\n[DEBUG] Initializing environment...")
        obs_payload = await env.initialize()

        if "error" in obs_payload:
            pytest.fail(f"Environment initialization failed: {obs_payload['error']}")

        print("[DEBUG] Environment initialized successfully")

        # Get initial state
        initial_pub = obs_payload["public"]
        initial_position = (initial_pub.player_x, initial_pub.player_y)
        initial_map_id = initial_pub.map_id

        print(f"[DEBUG] Initial position: {initial_position}")
        print(f"[DEBUG] Initial map ID: {initial_map_id}")

        # Test all available buttons systematically
        buttons_to_test = ["A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT"]

        results = {}

        for button in buttons_to_test:
            print(f"\n--- Testing {button} button ---")

            # Get state before button press
            before_pub = obs_payload["public"]
            before_position = (before_pub.player_x, before_pub.player_y)
            before_map = before_pub.map_id

            print(f"Before {button}: pos={before_position}, map={before_map}")

            # Press the button
            step_result = await env.step([[PressButtonCall(button)]])

            if "error" in step_result:
                print(f"[ERROR] {button} button failed: {step_result['error']}")
                results[button] = {"error": step_result["error"]}
                continue

            # Check state after button press
            after_pub = step_result["public"]
            after_position = (after_pub.player_x, after_pub.player_y)
            after_map = after_pub.map_id

            print(f"After {button}: pos={after_position}, map={after_map}")

            # Analyze what changed
            position_changed = after_position != before_position
            map_changed = after_map != before_map

            # Check if any other state changed
            state_changes = []
            if position_changed:
                state_changes.append(f"position: {before_position} -> {after_position}")
            if map_changed:
                state_changes.append(f"map: {before_map} -> {after_map}")

            # Check other state attributes
            if hasattr(before_pub, "party_level") and hasattr(after_pub, "party_level"):
                if before_pub.party_level != after_pub.party_level:
                    state_changes.append(
                        f"party_level: {before_pub.party_level} -> {after_pub.party_level}"
                    )

            if hasattr(before_pub, "badges") and hasattr(after_pub, "badges"):
                if before_pub.badges != after_pub.badges:
                    state_changes.append(f"badges: {before_pub.badges} -> {after_pub.badges}")

            results[button] = {
                "position_changed": position_changed,
                "map_changed": map_changed,
                "state_changes": state_changes,
                "effective": len(state_changes) > 0,
            }

            if state_changes:
                print(f"[SUCCESS] {button} caused changes: {', '.join(state_changes)}")
            else:
                print(f"[NO EFFECT] {button} had no visible effect")

            # Update obs_payload for next test
            obs_payload = step_result

        # Analysis and recommendations
        print("\n" + "=" * 80)
        print("EXPLORATION RESULTS AND RECOMMENDATIONS")
        print("=" * 80)

        effective_buttons = [
            btn
            for btn, result in results.items()
            if isinstance(result, dict) and result.get("effective", False)
        ]

        ineffective_buttons = [
            btn
            for btn, result in results.items()
            if isinstance(result, dict) and not result.get("effective", False)
        ]

        error_buttons = [
            btn for btn, result in results.items() if isinstance(result, dict) and "error" in result
        ]

        print(f"\n✅ EFFECTIVE BUTTONS ({len(effective_buttons)}): {', '.join(effective_buttons)}")
        for btn in effective_buttons:
            changes = results[btn]["state_changes"]
            print(f"  {btn}: {', '.join(changes)}")

        print(
            f"\n❌ INEFFECTIVE BUTTONS ({len(ineffective_buttons)}): {', '.join(ineffective_buttons)}"
        )

        if error_buttons:
            print(f"\n🚫 ERROR BUTTONS ({len(error_buttons)}): {', '.join(error_buttons)}")

        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        if effective_buttons:
            print(f"  - Agent should prioritize: {', '.join(effective_buttons[:3])}")
            print("  - These buttons cause state changes and may lead to progress")
        else:
            print("  - No buttons caused state changes in this initial position")
            print("  - May need to investigate game state or save file")

        if "LEFT" in effective_buttons or "RIGHT" in effective_buttons:
            print("  - Movement is working - agent should explore the area")

        if "A" not in effective_buttons:
            print("  - 'A' button ineffective at this position - agent needs to move first")

        return results

    except Exception as e:
        print(f"[ERROR] Test failed with exception: {e}")
        raise


@pytest.mark.asyncio
async def test_exploration_strategy():
    """Main test function"""
    results = await test_exploration_when_stuck()

    # The test always passes but provides diagnostic information
    assert True, "Exploration strategy test completed - see output for recommendations"


if __name__ == "__main__":
    # Run the test directly
    asyncio.run(test_exploration_strategy())
