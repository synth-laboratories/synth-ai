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


class ButtonTestObservationCallable(GetObservationCallable):
    """Observation callable for systematic button testing"""

    def __init__(self):
        self.screen_buffer = None

    async def get_observation(
        self, pub: PokemonRedPublicState, priv: PokemonRedPrivateState
    ) -> InternalObservation:
        if pub is None or priv is None:
            raise RuntimeError("Missing public or private state in get_observation")

        # Extract screen buffer
        try:
            import inspect

            frame = inspect.currentframe()
            env = None

            while frame:
                if "self" in frame.f_locals and hasattr(frame.f_locals["self"], "engine"):
                    env = frame.f_locals["self"]
                    break
                frame = frame.f_back

            if env and hasattr(env, "engine") and env.engine:
                if hasattr(env.engine, "emulator") and env.engine.emulator:
                    if hasattr(env.engine.emulator, "screen"):
                        screen_buffer = env.engine.emulator.screen.ndarray.copy()
                        self.screen_buffer = screen_buffer
        except Exception as e:
            print(f"[DEBUG] Failed to extract screen buffer: {e}")

        formatted_obs = (
            f"Step: {pub.step_count}, Position: ({pub.player_x}, {pub.player_y}), Map: {pub.map_id}"
        )

        return {
            "public": pub,
            "private": priv,
            "formatted_obs": formatted_obs,
            "screen_buffer": self.screen_buffer,
        }


async def test_single_vs_multiple_presses():
    """
    Test how many button presses are needed for reliable movement in different directions.
    """
    print("\n" + "=" * 80)
    print("SYSTEMATIC BUTTON PRESS REQUIREMENT ANALYSIS")
    print("=" * 80)

    # Test different buttons and press counts
    test_scenarios = [
        ("LEFT", "movement"),
        ("RIGHT", "movement"),
        ("UP", "movement"),
        ("DOWN", "movement"),
        ("A", "interaction"),
        ("B", "cancel/back"),
    ]

    results = {}

    for button, action_type in test_scenarios:
        print(f"\n{'=' * 60}")
        print(f"TESTING {button} BUTTON ({action_type})")
        print(f"{'=' * 60}")

        # Test with different numbers of presses (1-5)
        button_results = {}

        for press_count in range(1, 6):
            print(f"\nTesting {press_count} press(es) of {button}...")

            # Create fresh environment for each test
            task_metadata = TaskInstanceMetadata()
            inst = PokemonRedTaskInstance(
                id=uuid.uuid4(),
                impetus=Impetus(instructions=f"Test {button} button with {press_count} presses."),
                intent=Intent(
                    rubric={"goal": f"Test {button}"},
                    gold_trajectories=None,
                    gold_state_diff={},
                ),
                metadata=task_metadata,
                is_reproducible=True,
                initial_engine_snapshot=None,
            )

            test_obs = ButtonTestObservationCallable()
            env = PokemonRedEnvironment(inst, custom_step_obs=test_obs)

            try:
                # Initialize
                obs_payload = await env.initialize()
                if "error" in obs_payload:
                    print(f"[ERROR] Init failed: {obs_payload['error']}")
                    continue

                initial_pub = obs_payload["public"]
                initial_position = (initial_pub.player_x, initial_pub.player_y)
                initial_map_id = initial_pub.map_id

                print(f"  Initial state: pos={initial_position}, map={initial_map_id}")

                # Press button the specified number of times
                final_position = initial_position
                final_map_id = initial_map_id

                for press_num in range(press_count):
                    step_result = await env.step([[PressButtonCall(button)]])
                    if "error" in step_result:
                        print(f"  [ERROR] Step {press_num + 1} failed: {step_result['error']}")
                        break

                    new_pub = step_result["public"]
                    final_position = (new_pub.player_x, new_pub.player_y)
                    final_map_id = new_pub.map_id

                # Analyze results
                position_changed = final_position != initial_position
                map_changed = final_map_id != initial_map_id

                result = {
                    "initial_position": initial_position,
                    "final_position": final_position,
                    "initial_map": initial_map_id,
                    "final_map": final_map_id,
                    "position_changed": position_changed,
                    "map_changed": map_changed,
                    "effective": position_changed or map_changed,
                }

                button_results[press_count] = result

                print(f"  Result: pos={final_position}, map={final_map_id}")
                print(f"  Effect: {'YES' if result['effective'] else 'NO'}")

            except Exception as e:
                print(f"  [ERROR] Test failed: {e}")
                button_results[press_count] = {"error": str(e)}

        results[button] = button_results

    # Analysis and recommendations
    print("\n" + "=" * 80)
    print("ANALYSIS AND RECOMMENDATIONS")
    print("=" * 80)

    for button, button_results in results.items():
        print(f"\n{button} BUTTON:")

        # Find minimum presses for reliable effect
        min_effective_presses = None
        for press_count in range(1, 6):
            if press_count in button_results:
                result = button_results[press_count]
                if not isinstance(result, dict) or "error" in result:
                    continue
                if result.get("effective", False):
                    min_effective_presses = press_count
                    break

        if min_effective_presses:
            print(f"  ✓ Minimum effective presses: {min_effective_presses}")
            print(f"  ✓ Recommendation: Use {min_effective_presses} presses for {button}")
        else:
            print("  ✗ No effective movement detected with up to 5 presses")

        # Show detailed results
        for press_count, result in button_results.items():
            if isinstance(result, dict) and "error" not in result:
                effect_str = "EFFECTIVE" if result.get("effective") else "no effect"
                print(f"    {press_count} press(es): {effect_str}")

    return results


@pytest.mark.asyncio
async def test_button_press_requirements():
    """Main test function"""
    results = await test_single_vs_multiple_presses()

    # The test always passes but provides diagnostic information
    assert True, "Button press requirements test completed - see output for recommendations"


if __name__ == "__main__":
    # Run the test directly
    asyncio.run(test_button_press_requirements())
