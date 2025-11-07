#!/usr/bin/env python3
"""
Quick evaluation script to test the new Pokemon Red reward system with GPT-4o-mini.
This directly uses the environment with the new reward system rather than the task app.
"""

import asyncio
import os
from typing import Any, Dict

import httpx
from dotenv import load_dotenv
from synth_ai.environments.environment.tools import EnvToolCall
from synth_ai.environments.examples.red.environment import PokemonRedEnvironment
from synth_ai.environments.examples.red.taskset import INSTANCE as POKEMON_TASK
try:
    from synth_ai.environments.examples.red.trace_hooks_v3 import POKEMON_RED_HOOKS
except ImportError:
    import pytest
    pytest.skip("trace_hooks_v3 module missing; Pokemon Red reward test requires it", allow_module_level=True)
from synth_ai.tracing_v3.session_tracer import SessionTracer

# Load environment variables after imports
load_dotenv()


def get_reward_achievement_type(reward_value: float) -> str:
    """Map reward values to intuitive Pokemon Red gameplay achievements."""
    achievement_types = {
        50.0: "DEFEAT_BROCK",                # 🏆 Defeated Brock, won Boulder Badge!
        5.0: "ENTER_GYM_BUILDING",           # 🏢 Entered the gym building
        3.0: "POKEMON_READY_FOR_BATTLE",     # ⚔️ Pokemon strong enough for gym challenge
        2.0: "LEAVE_PALLET_TOWN",            # 🏠 Left home town, entered the world
        1.5: "ENTER_NEW_TOWN",               # 🏙️ Discovered Viridian or Pewter City
        1.0: "EXIT_STARTING_HOUSE",          # 🚪 Left the starter house
        0.8: "VISIT_POKEMON_CENTER",         # 💊 Healed at Pokemon Center
        0.5: "FIND_USEFUL_ITEM",             # 🎒 Found Pokeball, Potion, or TM
        0.3: "POKEMON_GOT_STRONGER",         # ⚡ Pokemon reached milestone level
        0.2: "POKEMON_LEVEL_INCREASED",      # 📈 Pokemon gained experience
        0.1: "ENCOUNTER_WILD_POKEMON",       # ⚔️ Started battle with wild Pokemon
        0.05: "KEEP_POKEMON_HEALTHY",        # ❤️ Pokemon stayed in good health
        0.02: "EXPLORE_NEW_AREA",            # 🗺️ Moved to unexplored location
        0.0: "REPEAT_ACTION",                # 🔄 Tried same action again
    }

    # Find exact match
    if reward_value in achievement_types:
        return achievement_types[reward_value]

    # Find closest match for unknown values (within 0.01 tolerance)
    closest_match = min(achievement_types.keys(), key=lambda x: abs(x - reward_value))
    if abs(reward_value - closest_match) < 0.01:
        return achievement_types[closest_match]

    # Handle special cases
    if 0.03 <= reward_value <= 0.05:
        return "QUICK_EXPLORATION"           # Covered multiple new areas quickly

    return f"UNKNOWN_ACHIEVEMENT_{reward_value}"


async def call_gpt_model(image_data_url: str, step_count: int, model_name: str = "gpt-4o-mini") -> Dict[str, Any]:
    """Call GPT model with image-only input to get next action."""

    system_prompt = """You are playing Pokemon Red. Look at the game screen and decide what button to press.

Available buttons: UP, DOWN, LEFT, RIGHT, A, B, START, SELECT

Your goal is to progress through the game efficiently. Focus on:
- Moving around and exploring
- Talking to NPCs (press A when facing them)
- Battling Pokemon
- Collecting items

Respond with a single tool call. Use execute_sequence for multiple actions."""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url}
                }
            ]
        }
    ]

    # Map model names
    model_mapping = {
        "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
        "gpt-4o": "gpt-4o-2024-08-06",
        "gpt-5-nano": "gpt-4o-mini-2024-07-18",  # Map gpt-5-nano to gpt-4o-mini
        "gpt-4-turbo": "gpt-4-turbo-2024-04-09"
    }

    actual_model = model_mapping.get(model_name, model_name)

    payload = {
        "model": actual_model,
        "messages": messages,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "execute_sequence",
                    "description": "Execute multiple button presses in sequence",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "actions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "button": {"type": "string", "enum": ["UP", "DOWN", "LEFT", "RIGHT", "A", "B", "START", "SELECT"]},
                                        "frames": {"type": "integer", "minimum": 1, "maximum": 60}
                                    },
                                    "required": ["button", "frames"]
                                },
                                "minItems": 1,
                                "maxItems": 10
                            }
                        },
                        "required": ["actions"]
                    }
                }
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "execute_sequence"}},
        "temperature": 0.7,
        "max_tokens": 500
    }

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment")

    headers = {"Authorization": f"Bearer {api_key}"}

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            json=payload,
            headers=headers
        )

    response.raise_for_status()
    data = response.json()

    # Extract tool call
    choices = data.get("choices", [])
    if not choices:
        return {"actions": [{"button": "A", "frames": 30}]}  # Fallback

    message = choices[0].get("message", {})
    tool_calls = message.get("tool_calls", [])

    if not tool_calls:
        return {"actions": [{"button": "A", "frames": 30}]}  # Fallback

    function = tool_calls[0].get("function", {})
    args_str = function.get("arguments", "{}")

    import json
    try:
        args = json.loads(args_str)
        return args
    except Exception:
        return {"actions": [{"button": "A", "frames": 30}]}  # Fallback


async def run_evaluation(seed: int = 0, max_steps: int = 10):
    """Run a single evaluation episode with the new reward system."""

    print(f"\n🚀 Starting Pokemon Red evaluation (seed={seed})")
    print("=" * 60)

    # Initialize tracer for reward/event tracking
    tracer = SessionTracer(hooks=POKEMON_RED_HOOKS)
    await tracer.initialize()

    # Start tracing session
    model_name = "gpt-5-nano"
    session_id = f"pokemon_red_eval_seed_{seed}_{model_name.replace('-', '')}"
    await tracer.start_session(session_id=session_id, metadata={
        "environment": "pokemon_red",
        "evaluation_seed": seed,
        "max_steps": max_steps,
        "model": model_name
    })

    # Initialize environment with tracer
    env = PokemonRedEnvironment(POKEMON_TASK, tracer=tracer)
    obs = await env.initialize()

    total_reward = 0.0
    rewards_history = []

    print(f"📍 Initial state: {obs.get('position', 'unknown')}")
    print("🎯 Goal: Beat Brock at Pewter Gym to earn Boulder Badge")
    print(f"💰 Initial reward: {total_reward}")
    print()

    for step in range(max_steps):
        print(f"Step {step + 1}/{max_steps}")
        print("-" * 30)

        # Get image data for GPT-4o-mini
        image_url = obs.get("observation_image_data_url")
        if not image_url:
            print("❌ No image data available, ending episode")
            break

        # Start timestep for this step
        await tracer.start_timestep(f"step_{step}", turn_number=step)

        # Call GPT model
        model_name = "gpt-5-nano"  # Use the model the user requested
        print(f"🤖 Calling {model_name}...")
        try:
            action = await call_gpt_model(image_url, step, model_name)
            print(f"🎮 Action: {action}")
        except Exception as e:
            print(f"❌ Error calling {model_name}: {e}")
            await tracer.end_timestep()
            break

        # Execute action sequence
        step_reward = 0.0
        actions_taken = []

        for action_item in action.get("actions", []):
            button = action_item.get("button", "A")
            frames = action_item.get("frames", 30)

            # Execute single button press using EnvToolCall
            tool_call = EnvToolCall(tool="press_button", args={"button": button, "frames": frames})
            obs = await env.step(tool_call)
            actions_taken.append(f"{button}({frames}f)")

            # Accumulate reward from this step
            step_reward += obs.get("reward_last_step", 0.0)

        total_reward += step_reward
        rewards_history.append(step_reward)

        # Print step results
        position = obs.get("position", "unknown")
        badges = obs.get("badges_earned", 0)
        hp_status = obs.get("hp_status", "unknown")
        party_level = obs.get("party_level", 0)

        print(f"📍 Position: {position}")
        print(f"🎖️  Badges: {badges}")
        print(f"❤️  HP: {hp_status}")
        print(f"⭐  Party Level: {party_level}")
        print(f"💰 Step Reward: {step_reward:.2f}")
        print(f"💰 Total Reward: {total_reward:.2f}")

        if step_reward > 0:
            print(f"🎉 GOT REWARD! (+{step_reward:.2f})")

        print(f"🎮 Actions: {' → '.join(actions_taken)}")
        print()

        # End timestep for this step
        await tracer.end_timestep()

        # Check if episode should end (got Boulder Badge)
        current_badges = obs.get("badges", 0)
        if current_badges & 0x01:  # Boulder Badge (bit 0)
            print("🏆 SUCCESS! Earned Boulder Badge!")
            break

        # Small delay to avoid rate limiting
        await asyncio.sleep(0.5)

    # Final summary
    print("=" * 60)
    print("📊 EVALUATION SUMMARY")
    print("=" * 60)
    print(f"🎯 Seed: {seed}")
    print(f"📏 Steps taken: {step + 1}")
    print(f"💰 Total reward: {total_reward:.2f}")
    print(f"📈 Reward history: {[f'{r:.2f}' for r in rewards_history]}")

    final_badges = obs.get("badges_earned", 0)
    final_position = obs.get("position", "unknown")
    final_party_level = obs.get("party_level", 0)

    print(f"🎖️  Final badges: {final_badges}")
    print(f"📍 Final position: {final_position}")
    print(f"⭐  Final party level: {final_party_level}")

    success = final_badges >= 1
    print(f"🏆 Success: {'YES' if success else 'NO'}")

    # End tracing session
    await tracer.end_session()

    return {
        "seed": seed,
        "steps": step + 1,
        "total_reward": total_reward,
        "rewards_history": rewards_history,
        "final_badges": final_badges,
        "final_position": final_position,
        "final_party_level": final_party_level,
        "success": success
    }


async def main():
    """Run multiple evaluation episodes."""
    model_name = "gpt-5-nano"
    print(f"🎮 Pokemon Red Reward System Evaluation with {model_name}")
    print("Using the NEW comprehensive reward system!")
    print("Running 20 tool calls per episode to check for non-zero rewards")
    print()

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key: export OPENAI_API_KEY=your_key_here")
        return

    seeds = [10, 11, 12, 13, 14]  # New seeds for comparison
    results = []
    all_rewards = []  # Collect all individual rewards

    print(f"🚀 Running {len(seeds)} episodes with 20 tool calls each...")
    print("=" * 60)

    for seed in seeds:
        try:
            result = await run_evaluation(seed, max_steps=20)  # 20 tool calls per episode
            results.append(result)
            all_rewards.extend(result["rewards_history"])
        except Exception as e:
            print(f"❌ Error in evaluation for seed {seed}: {e}")
            continue

    # Aggregate results
    if results:
        print("\n" + "=" * 80)
        print("📊 AGGREGATE RESULTS")
        print("=" * 80)

        total_episodes = len(results)
        successful_episodes = sum(1 for r in results if r["success"])
        avg_reward = sum(r["total_reward"] for r in results) / total_episodes
        avg_steps = sum(r["steps"] for r in results) / total_episodes
        max_reward = max(r["total_reward"] for r in results)

        print(f"📈 Episodes run: {total_episodes}")
        print(f"🏆 Success rate: {successful_episodes}/{total_episodes} ({successful_episodes/total_episodes*100:.1f}%)")
        print(f"💰 Average reward: {avg_reward:.2f}")
        print(f"📏 Average steps: {avg_steps:.1f}")
        print(f"🏅 Best reward: {max_reward:.2f}")

        # Analyze achievement frequencies (only genuine accomplishments - positive rewards)
        print("\n🎯 ACHIEVEMENT ANALYSIS (Genuine Accomplishments Only)")
        print("-" * 55)

        from collections import Counter
        achievement_counts = Counter()

        # Only count positive rewards as genuine accomplishments
        positive_rewards = [r for r in all_rewards if r > 0.0]
        total_achievement_events = len(positive_rewards)
        total_action_events = len(all_rewards)

        # Convert each positive reward value to achievement type
        for reward_value in positive_rewards:
            achievement_type = get_reward_achievement_type(reward_value)
            achievement_counts[achievement_type] += 1

        print(f"📊 Total action events: {total_action_events}")
        print(f"🏆 Genuine achievements: {total_achievement_events}")
        print(f"🏆 Unique achievement types: {len(achievement_counts)}")

        # Sort by frequency (most common first)
        sorted_achievements = sorted(achievement_counts.items(), key=lambda x: x[1], reverse=True)

        print("\n🏆 ACHIEVEMENT FREQUENCIES:")
        for achievement_type, count in sorted_achievements:
            percentage = (count / total_achievement_events) * 100
            print(f"  {achievement_type:<25} ({count:3d} times, {percentage:5.1f}%)")

        # Summary of achievement patterns
        print("\n📈 ACHIEVEMENT DISTRIBUTION:")
        print(f"  🟢 Genuine achievements: {total_achievement_events} ({total_achievement_events/total_action_events*100:.1f}% of actions)")
        print(f"  ⚪ Non-productive actions: {total_action_events - total_achievement_events} ({(total_action_events - total_achievement_events)/total_action_events*100:.1f}% of actions)")
        print(f"  🎯 Achievement efficiency: {total_achievement_events/total_action_events*100:.1f}%")

        if successful_episodes > 0:
            print("\n🎉 SUCCESS! The new reward system is working!")
            print("GPT-4o-mini earned rewards and made progress in Pokemon Red!")
        else:
            print("\n🤔 No successes yet, but rewards were earned:")
            for _, r in enumerate(results):
                print(f"  Seed {r['seed']}: {r['total_reward']:.2f} reward")


if __name__ == "__main__":
    asyncio.run(main())
