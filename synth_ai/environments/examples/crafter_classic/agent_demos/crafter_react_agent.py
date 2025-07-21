import asyncio
import uuid
import pytest
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Deque, Set, Union
from pydantic import BaseModel, Field
from collections import deque
import toml
from synth_ai.lm.core.main import LM
from synth_ai.lm.tools.base import BaseTool
from synth_sdk.tracing.decorators import trace_event_async
from synth_sdk.tracing.trackers import SynthTracker
from synth_sdk.tracing.abstractions import RewardSignal, Dataset, TrainingQuestion
from synth_sdk.tracing.utils import get_system_id

# Crafter specific imports
from synth_ai.environments.examples.crafter_classic.environment import (
    CrafterClassicEnvironment,
    CrafterPublicState,
    CrafterPrivateState,
)
from synth_ai.environments.examples.crafter_classic.engine import (
    CRAFTER_ACTION_MAP,  # map of action name to int
)

# Convert CRAFTER_ACTION_MAP to ACTION_STRING_TO_INT and INT_TO_ACTION_STRING
ACTION_STRING_TO_INT: Dict[str, int] = CRAFTER_ACTION_MAP
INT_TO_ACTION_STRING: Dict[int, str] = {v: k for k, v in CRAFTER_ACTION_MAP.items()}


from synth_ai.environments.environment.shared_engine import (
    GetObservationCallable,
    InternalObservation,
)
from synth_ai.environments.examples.crafter_classic.taskset import (
    CrafterTaskInstance,
    CrafterTaskInstanceMetadata,
)
from synth_ai.environments.tasks.core import Impetus, Intent
from synth_ai.environments.environment.tools import EnvToolCall

import logging

logging.disable(logging.CRITICAL)


# --- Helper to build crafter semantic mapping ---
def get_crafter_semantic_mapping():
    """Build the crafter semantic ID to item name mapping."""
    import crafter
    import itertools

    # Create a dummy env to get ID mappings (same as environment.py)
    dummyenv = None
    try:
        dummyenv = crafter.Env()
        max_id = (
            max(
                max(dummyenv._world._mat_ids.values()),
                max(dummyenv._sem_view._obj_ids.values()),
            )
            + 1
        )
        id_to_item = ["void"] * max_id
        for name, ind in itertools.chain(
            dummyenv._world._mat_ids.items(), dummyenv._sem_view._obj_ids.items()
        ):
            if name is None:
                clean = "none"
            elif hasattr(name, "__name__"):
                clean = name.__name__
            else:
                clean = str(name)
            id_to_item[ind] = clean.lower()
        player_idx = id_to_item.index("player")
        return id_to_item, player_idx
    finally:
        if dummyenv:
            try:
                dummyenv.close()
            except Exception:
                pass
        del dummyenv


# --- Helper function to format observation for LLM ---
def format_obs_for_llm_from_states(pub: CrafterPublicState, priv: CrafterPrivateState) -> str:
    inventory_str = ", ".join(f"{k}:{v}" for k, v in pub.inventory.items() if v > 0)
    if not inventory_str:
        inventory_str = "empty"

    achievements_str = ", ".join(k for k, v in pub.achievements_status.items() if v)
    if not achievements_str:
        achievements_str = "none"

    # Add map view around player using the real crafter semantic mapping
    map_view = ""
    if pub.semantic_map is not None:
        px, py = pub.player_position
        view_size = 7  # 7x7 view around player
        half_view = view_size // 2

        # Get the real crafter semantic mapping
        id_to_item, player_idx = get_crafter_semantic_mapping()

        # Create a local view around the player using same logic as _plain_grid
        map_view += f"\nLocal Map View ({view_size}x{view_size} around player):\n"
        matrix = []
        for dy in range(-half_view, half_view + 1):
            row = []
            for dx in range(-half_view, half_view + 1):
                x, y = px + dx, py + dy
                if pub.semantic_map is None or not (
                    0 <= x < pub.semantic_map.shape[0] and 0 <= y < pub.semantic_map.shape[1]
                ):
                    row.append("void")
                else:
                    idx = pub.semantic_map[x, y]
                    if dx == 0 and dy == 0:
                        row.append("player")  # Player position
                    else:
                        # Use the real crafter mapping
                        item_name = id_to_item[idx] if idx < len(id_to_item) else "unknown"
                        row.append(item_name)
            matrix.append(row)

        # Transpose the matrix like _plain_grid does
        transposed = list(zip(*matrix))
        # Convert each row to a space-separated string
        for row in transposed:
            map_view += " ".join(row) + "\n"

        # Create a legend of items actually visible in the map
        visible_items = set()
        for row in transposed:
            for item in row:
                if item not in ["void", "player"]:
                    visible_items.add(item)

        if visible_items:
            map_view += f"\nVisible items: {', '.join(sorted(visible_items))}"
        else:
            map_view += "\nNo special items visible (mostly grass/empty)"

    # Simplified observation, focusing on key elements
    return (
        f"Steps: {pub.num_steps_taken}/{pub.max_steps_episode}\n"
        f"Health: {priv.player_internal_stats.get('health', 'N/A')}\n"
        f"Inventory: {inventory_str}\n"
        f"Unlocked Achievements: {achievements_str}\n"
        f"Player Position: {pub.player_position}\n"
        f"Last Reward: {priv.reward_last_step:.2f}\n"
        f"Terminated: {priv.terminated} | Truncated: {priv.truncated}"
        f"{map_view}"
    )


# ---------------------------------- custom observation callable (Optional, can be simpler for Crafter) ------------------------------ #
# For now, let's assume the default observation from the environment is sufficient,
# or we will use the direct public/private states.
# If history is needed, we can adapt the Sokoban HistoryObservationCallable.
class CrafterHistoryObservationCallable(GetObservationCallable):
    def __init__(self, max_history: int = 1):  # Keep only current obs for simplicity now
        self._hist_obs: Deque[str] = deque(maxlen=max_history)
        self._hist_pub_state: Deque[CrafterPublicState] = deque(maxlen=max_history)
        self._hist_priv_state: Deque[CrafterPrivateState] = deque(maxlen=max_history)

    async def get_observation(
        self, pub: CrafterPublicState, priv: CrafterPrivateState
    ) -> InternalObservation:
        if pub is None or priv is None:
            return {
                "error": "Missing public or private state in get_observation",
                "history_formatted_obs": list(self._hist_obs),
            }  # type: ignore[return-value]

        formatted_obs = format_obs_for_llm_from_states(pub, priv)
        self._hist_obs.append(formatted_obs)
        self._hist_pub_state.append(pub)
        self._hist_priv_state.append(priv)

        return {
            "public": pub,
            "private": priv,
            "formatted_obs": formatted_obs,  # Current formatted obs
            "history_formatted_obs": list(self._hist_obs),  # History of formatted obs
            "history_public_states": list(self._hist_pub_state),
            "history_private_states": list(self._hist_priv_state),
        }  # type: ignore[return-value]


# --- Pydantic Models for Tool Arguments ---
class CrafterInteractArgs(BaseModel):
    actions_list: List[str] = Field(
        description="A list of action names to execute in sequence in the Crafter environment (e.g., ['move_up', 'move_up', 'place_stone']). Can contain 1-10 actions."
    )
    reasoning: str = Field(description="A brief explanation of why these actions were chosen.")


# class TerminateArgs(BaseModel):
#     reason: str = Field(
#         description="A detailed reason for why the agent is terminating."
#     )


# --- ReAct agent for Crafter -------------------------------------------------- #
class CrafterInteractTool(BaseTool):
    """Tool for interacting with Crafter environment"""

    name: str = "crafter_interact"
    arguments: type[BaseModel] = CrafterInteractArgs
    description: str = (
        "Interacts with the Crafter environment by proposing a sequence of 1-10 actions to execute."
    )


# class TerminateTool(BaseTool):
#     """Tool for terminating agent execution"""
#     name: str = "terminate"
#     arguments: type[BaseModel] = TerminateArgs
#     description: str = "Terminates the agent's execution if the task is considered complete or no useful progress can be made."


class CrafterMove(EnvToolCall):  # Simple EnvToolCall wrapper
    def __init__(self, action: int):
        super().__init__(tool="interact", args={"action": action})


class ReActAgent:
    def __init__(self, llm, max_turns: int = 50):  # Increased max_turns for Crafter
        self.llm, self.max_turns = llm, max_turns
        self.history: List[Dict[str, Any]] = []
        self.system_name: str = "crafter-react-ex"  # Changed system name
        self.system_id: Any = get_system_id(self.system_name)
        self.system_instance_id: str = str(uuid.uuid4())
        self.last_obs_dict: Optional[Dict[str, Any]] = (
            None  # To store raw observation for terminate guardrails
        )
        self.current_achievements: Set[str] = set()  # To track unique achievements

        self.tools = [
            CrafterInteractTool(),
            # TerminateTool(),  # Commented out to prevent early quitting
        ]

    def _format_history_for_prompt(self) -> str:
        prompt_history = []
        for entry in self.history:
            if entry["type"] == "obs":
                prompt_history.append(f"OBSERVATION:\n{entry['content']}")
            elif entry["type"] == "tool_call":
                args_str = json.dumps(entry["tool_arguments"])
                prompt_history.append(
                    f"THOUGHT:\nI will call the tool `{entry['tool_name']}` with arguments: {args_str}\nACTION: (Tool call executed)"
                )
            elif entry["type"] == "tool_response":
                prompt_history.append(
                    "TOOL_RESPONSE:\n(Action executed, new observation will follow if not terminal)"
                )
        return "\n".join(prompt_history)

    @trace_event_async(event_type="react_agent_decide")
    async def decide(
        self, obs_str: str, current_raw_obs: Dict[str, Any]
    ) -> List[int]:  # Return list of action integers
        self.history.append({"type": "obs", "content": obs_str})
        self.last_obs_dict = current_raw_obs  # Store for terminate guardrail

        # Update current achievements from the raw observation
        if current_raw_obs and isinstance(current_raw_obs.get("public"), CrafterPublicState):
            pub_state: CrafterPublicState = current_raw_obs["public"]
            for ach, unlocked in pub_state.achievements_status.items():
                if unlocked:
                    self.current_achievements.add(ach)

        formatted_prompt_history = self._format_history_for_prompt()

        # Updated prompt for Crafter
        prompt = (
            f"{formatted_prompt_history}\n\n"
            "Based on the history above, particularly the last observation (health, inventory, achievements, position), "
            "what is your reasoning and which `crafter_interact` tool should you call next? "
            "Prioritize actions that lead to new achievements or ensure survival (e.g., find food if health is low)."
        )

        system_message = (
            "You are an agent playing Crafter. Your goal is to survive and unlock as many achievements as possible. "
            "Review the history of observations, thoughts, and actions. "
            "Based on this history, particularly the last observation, decide on the best sequence of actions. "
            "You MUST call the available tool: `crafter_interact`.\\n\\n"
            "For `crafter_interact`, provide a list of 1-10 actions to execute in sequence. "
            "Planning ahead with multiple actions is often more efficient than single actions. "
            f"Available actions are: {', '.join(ACTION_STRING_TO_INT.keys())}.\\n"
            "Always provide a `reasoning` field in your tool call."
        )

        # Trace the LLM interaction input so that full messages (system & user) are included in the trace
        SynthTracker.track_lm(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            model_name=self.llm.model_name,
            model_params=None,
            finetune=False,
        )

        response_obj = await self.llm.respond_async(
            system_message=system_message, user_message=prompt, tools=self.tools
        )

        # Trace the assistant's reply/output so that it is captured alongside the inputs
        SynthTracker.track_lm_output(
            messages=[{"role": "assistant", "content": response_obj.raw_response}],
            model_name=self.llm.model_name,
            finetune=False,
        )

        tool_calls = response_obj.tool_calls

        # Handle case where tool_calls is None or empty (noop to prevent crash)
        if not tool_calls:
            # print(f"[WARNING] No tool calls returned by {self.llm.model_name}, returning noop action")
            self.history.append(
                {
                    "type": "tool_call",
                    "tool_name": "noop",
                    "tool_arguments": {"reason": "no_tool_calls_returned"},
                }
            )
            self.history.append(
                {
                    "type": "tool_response",
                    "content": "Noop executed due to missing tool calls",
                }
            )
            return [0]  # Return 'noop' action (action index 0)

        tool_call_data = tool_calls[0]

        # Handle both dict and object formats
        if isinstance(tool_call_data, dict):
            tool_name = tool_call_data["function"]["name"]
            tool_args_str = tool_call_data["function"]["arguments"]
        else:
            tool_name = tool_call_data.function.name
            tool_args_str = tool_call_data.function.arguments

        tool_arguments = json.loads(tool_args_str)

        # Track the tool call details for richer debugging and training signals
        SynthTracker.track_state(
            variable_name="tool_call",
            variable_value={"tool_name": tool_name, "arguments": tool_arguments},
            origin="agent",
        )

        self.history.append(
            {
                "type": "tool_call",
                "tool_name": tool_name,
                "tool_arguments": tool_arguments,
            }
        )
        self.history.append({"type": "tool_response", "content": "Tool executed"})

        if tool_name == "crafter_interact":
            actions_list = tool_arguments["actions_list"]

            # Convert action names to integers
            action_ints = []
            for action_str in actions_list:
                if action_str in ACTION_STRING_TO_INT:
                    action_ints.append(ACTION_STRING_TO_INT[action_str])
                else:
                    print(f"[WARNING] Invalid action '{action_str}', using noop instead")
                    action_ints.append(0)  # noop action

            return action_ints

        # elif tool_name == "terminate":
        #     reason = tool_arguments["reason"]
        #
        #     # Add the human-readable termination reason to the history
        #     self.history.append({
        #         "type": "termination",
        #         "content": f"Agent terminated: {reason}",
        #         "reason": reason
        #     })
        #
        #     return [-1]  # Special termination indicator


# --- Test for a single agent run ---
@pytest.mark.asyncio
async def test_react_agent_crafter(tmp_path: Path):
    # Create a simple Crafter task instance for testing
    # For Crafter, the seed in metadata is important for reproducibility.
    # initial_engine_snapshot can be None if the engine handles reset with seed.
    task_metadata = CrafterTaskInstanceMetadata(
        difficulty="easy",
        seed=42,
        # Other metadata fields can be default or placeholders if not critical for this test
        num_trees_radius=0,  # Placeholder, actual values depend on seed and world gen
        num_cows_radius=0,  # Placeholder
        num_hostiles_radius=0,  # Placeholder
    )
    inst = CrafterTaskInstance(
        id=uuid.uuid4(),
        impetus=Impetus(instructions="Survive and unlock achievements."),
        intent=Intent(
            rubric={"goal": "Unlock achievements and survive"},
            gold_trajectories=None,
            gold_state_diff={},
        ),
        metadata=task_metadata,
        is_reproducible=True,
        initial_engine_snapshot=None,  # Engine will init with seed from metadata
    )

    hist_cb = CrafterHistoryObservationCallable(max_history=1)
    env = CrafterClassicEnvironment(inst, custom_step_obs=hist_cb)
    # env.engine.package_sokoban_env.render_mode = "raw" # Not applicable to Crafter

    llm = LM(model_name="gpt-4.1-nano", formatting_model_name="gpt-4.1-nano", temperature=0.0)
    agent = ReActAgent(llm, max_turns=30)  # Increased for meaningful progress
    print("[DEBUG] Created agent with max_turns=30")

    async def run_episode():
        obs_payload = await env.initialize()

        if "error" in obs_payload:
            print(f"Error during env.initialize: {obs_payload['error']}")
            return False, 0

        # Initial observation for the agent
        # The CrafterHistoryObservationCallable returns a dict with 'public', 'private', 'formatted_obs'
        current_formatted_obs = obs_payload["formatted_obs"]
        raw_obs_for_agent_decision = (
            obs_payload  # Pass the whole payload which includes public and private states
        )

        for turn in range(agent.max_turns):
            action_sequence = await agent.decide(current_formatted_obs, raw_obs_for_agent_decision)

            if action_sequence == [-1]:  # Agent decided to terminate
                obs_payload_next = obs_payload  # No new observation if terminated by agent
                break

            # Execute each action in the sequence
            for act_idx in action_sequence:
                step_result = await env.step([[CrafterMove(act_idx)]])
                obs_payload_next = step_result

                if "error" in obs_payload_next:
                    break

                # Update observation for next action in sequence
                current_formatted_obs = obs_payload_next["formatted_obs"]
                raw_obs_for_agent_decision = obs_payload_next
                obs_payload = obs_payload_next

                # Check if environment terminated after this sub-action
                if obs_payload_next["private"].terminated or obs_payload_next["private"].truncated:
                    priv_state = obs_payload_next["private"]
                    pub_state = obs_payload_next["public"]
                    player_health = priv_state.player_internal_stats.get("health", "N/A")
                    steps_taken = pub_state.num_steps_taken
                    max_steps = pub_state.max_steps_episode

                    break

            if "error" in obs_payload_next:
                break

            # Update observations for the next agent decision
            current_formatted_obs = obs_payload_next["formatted_obs"]
            raw_obs_for_agent_decision = obs_payload_next

            agent.history.append({"type": "tool_response", "content": "Action executed"})

            obs_payload = obs_payload_next

            if obs_payload_next["private"].terminated or obs_payload_next["private"].truncated:
                break

        # Ensure obs_payload_next is defined even if loop didn't run or agent terminated early
        if "obs_payload_next" not in locals():
            obs_payload_next = obs_payload

        if "error" in obs_payload_next:
            return False, len(agent.current_achievements)

        # Success could be defined as surviving some steps or achieving something
        # For this test, let's say it's successful if it ran and terminated/truncated by env
        final_private_state: CrafterPrivateState = obs_payload_next["private"]
        episode_successful = final_private_state.terminated or final_private_state.truncated
        return episode_successful, len(agent.current_achievements)

    episode_completed, num_achievements = await run_episode()

    dataset = Dataset(
        questions=[
            TrainingQuestion(
                id="crafter_ep_test",
                intent="survive and achieve",
                criteria="completed_episode_or_achieved_something",
            )
        ],
        reward_signals=[
            RewardSignal(
                question_id="crafter_ep_test",
                system_instance_id=agent.system_instance_id,
                reward=1
                if episode_completed or num_achievements > 0
                else 0,  # Reward if completed or got any achievement
            )
        ],
    )
    # upload(dataset=dataset) # Optional: uncomment to upload trace

    assert episode_completed or num_achievements > 0, (
        "Agent failed to complete the episode or unlock any achievement in the test."
    )


async def eval_react_crafter(
    model_name: str = "gpt-4.1-nano",
    formatting_model_name: str = "gpt-4.1-nano",
    modes: Optional[List[str]] = None,
    n_instances_per_mode: int = 3,
) -> List[Dict[str, Any]]:
    """
    Run ReAct agents on Crafter instances of different difficulties,
    and returns a list of dictionaries containing aggregated results for each mode.
    """
    # Import the new evaluation framework
    from synth_ai.environments.examples.crafter_classic.agent_demos.eval_framework import (
        run_crafter_eval,
    )

    if modes is None:
        modes = ["easy", "hard"]

    print(f"🎯 Running Crafter evaluation with new standardized framework")
    print(f"   Model: {model_name}")
    print(f"   Modes: {modes}")
    print(f"   Trajectories per mode: {n_instances_per_mode}")

    # Use the new comprehensive evaluation framework
    report = await run_crafter_eval(
        model_names=[model_name],
        difficulties=modes,
        num_trajectories=n_instances_per_mode,
        max_turns=30,
    )

    # Convert to old format for backward compatibility
    results_for_model = []
    for agg_result in report["raw_aggregate_results"]:
        results_for_model.append(
            {
                "Model": agg_result["model_name"],
                "Difficulty": agg_result["difficulty"],
                "Successful Runs": f"{int(agg_result['success_rate'] * agg_result['num_trajectories'])}/{agg_result['num_trajectories']}",
                "Avg Unique Achievements": f"{agg_result['avg_achievements_per_trajectory']:.2f}",
            }
        )

    return results_for_model


# Keep the old function for backward compatibility
async def eval_react_crafter_legacy(
    model_name: str = "gpt-4.1-nano",
    formatting_model_name: str = "gpt-4.1-nano",
    modes: Optional[List[str]] = None,
    n_instances_per_mode: int = 3,
) -> List[Dict[str, Any]]:
    """
    LEGACY VERSION - Run ReAct agents on Crafter instances of different difficulties,
    and returns a list of dictionaries containing aggregated results for each mode.
    """

    if modes is None:
        modes = ["easy", "hard"]

    current_model_name_for_eval = model_name

    _temp_llm_for_names = LM(
        model_name=current_model_name_for_eval,
        formatting_model_name=formatting_model_name,
        temperature=0.0,
    )
    _temp_agent_for_names = ReActAgent(_temp_llm_for_names)
    actual_system_name = (
        _temp_agent_for_names.system_name
    )  # Still useful for logging within this func

    # ------------------------------------------------------------------ helpers
    async def run_episode_eval(
        inst: CrafterTaskInstance, agent_max_turns: int
    ) -> tuple[bool, int, list[str], int]:  # Added achievements list and steps taken
        """Run single episode and return (success, num_achievements, achievements_list, steps_taken)"""
        llm = LM(
            model_name=current_model_name_for_eval,
            formatting_model_name=current_model_name_for_eval,
            temperature=0.0,
        )

        hist_cb = CrafterHistoryObservationCallable(max_history=1)
        env = CrafterClassicEnvironment(inst, custom_step_obs=hist_cb)
        agent = ReActAgent(llm, max_turns=agent_max_turns)

        obs_payload = await env.initialize()
        if "error" in obs_payload:
            return False, 0, [], 0

        current_formatted_obs = obs_payload["formatted_obs"]
        raw_obs_for_agent_decision = obs_payload

        turn_count = 0
        for turn_idx in range(agent.max_turns):
            turn_count += 1
            # Remove noisy progress output

            action_sequence = await agent.decide(current_formatted_obs, raw_obs_for_agent_decision)

            if action_sequence == [-1]:  # agent terminated
                break

            # Execute each action in the sequence
            for i, act_idx in enumerate(action_sequence):
                obs_payload_next = await env.step([[CrafterMove(act_idx)]])

                if "error" in obs_payload_next:
                    break  # Break out of action sequence on error

                # Update observation for next action in sequence
                current_formatted_obs = obs_payload_next["formatted_obs"]
                raw_obs_for_agent_decision = obs_payload_next

                # Check if environment terminated after this sub-action
                if obs_payload_next["private"].terminated or obs_payload_next["private"].truncated:
                    break

            if "error" in obs_payload_next:
                return (
                    False,
                    len(agent.current_achievements),
                    list(agent.current_achievements),
                    0,
                )

            current_formatted_obs = obs_payload_next["formatted_obs"]
            raw_obs_for_agent_decision = obs_payload_next
            agent.history.append({"type": "tool_response", "content": "Action executed"})

            obs_payload = obs_payload_next
            if obs_payload["private"].terminated or obs_payload["private"].truncated:
                break

        final_private_state: CrafterPrivateState = obs_payload["private"]
        final_public_state: CrafterPublicState = obs_payload["public"]

        run_successful = (final_private_state.terminated or final_private_state.truncated) or len(
            agent.current_achievements
        ) >= 1

        num_unique_achievements = len(agent.current_achievements)
        achievements_list = list(agent.current_achievements)
        steps_taken = final_public_state.num_steps_taken

        return run_successful, num_unique_achievements, achievements_list, steps_taken

    # ---------------------------------------------------------------- instance factory
    async def make_crafter_instances(
        difficulty: str, n_instances: int = 3, start_seed: int = 0
    ) -> List[CrafterTaskInstance]:
        instances = []
        for i in range(n_instances):
            current_seed = start_seed + i
            metadata = CrafterTaskInstanceMetadata(
                difficulty=difficulty,
                seed=current_seed,
                num_trees_radius=0,
                num_cows_radius=0,
                num_hostiles_radius=0,
            )
            instance = CrafterTaskInstance(
                id=uuid.uuid4(),
                impetus=Impetus(
                    instructions=f"Survive and unlock achievements in a {difficulty} environment."
                ),
                intent=Intent(rubric={}, gold_trajectories=None, gold_state_diff={}),
                metadata=metadata,
                is_reproducible=True,
                initial_engine_snapshot=None,
            )
            instances.append(instance)
        return instances

    # ---------------------------------------------------------------- evaluation
    configs = []
    for mode in modes:
        if mode == "easy":
            configs.append(("easy", n_instances_per_mode, 15))
        elif mode == "hard":
            configs.append(("hard", n_instances_per_mode, 15))

    results_for_model = []  # Stores dicts for each mode for the current model
    base_seed_for_difficulty = {"easy": 1000, "hard": 2000}

    print(
        f"Starting Crafter ReAct Agent Evaluation for Model: {current_model_name_for_eval}, System: {actual_system_name}"
    )

    all_generated_task_data = []
    print("\nGenerating task instances...")
    all_tasks_for_eval: Dict[str, List[CrafterTaskInstance]] = {}
    for label, num_agents, _ in configs:
        insts = await make_crafter_instances(
            label, n_instances=num_agents, start_seed=base_seed_for_difficulty[label]
        )
        all_tasks_for_eval[label] = insts
        for inst in insts:
            instance_dict = await inst.serialize()
            all_generated_task_data.append(instance_dict)
        print(f"Generated {len(insts)} instances for {label} difficulty.")

    dataset_dir = Path(__file__).parent.parent / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    synthetic_mix_path = dataset_dir / "synthetic_mix.json"
    with open(synthetic_mix_path, "w") as f:
        json.dump(all_generated_task_data, f, indent=2)
    print(
        f"Saved all {len(all_generated_task_data)} generated task instances to {synthetic_mix_path}"
    )

    for label, num_agents, max_episode_turns in configs:
        print(
            f"\nRunning {num_agents} agents on {label} difficulty tasks (max_turns: {max_episode_turns}) for model {current_model_name_for_eval}..."
        )
        current_difficulty_instances = all_tasks_for_eval[label]

        import time

        start_time = time.time()
        results_per_episode = await asyncio.gather(
            *(run_episode_eval(inst, max_episode_turns) for inst in current_difficulty_instances)
        )
        end_time = time.time()
        print(
            f"Completed {len(current_difficulty_instances)} episodes in {end_time - start_time:.1f}s"
        )

        # Process detailed results
        successful_episodes = 0
        total_achievements = 0
        detailed_results = []

        for i, (success, num_achievements, achievements_list, steps_taken) in enumerate(
            results_per_episode
        ):
            episode_result = {
                "episode_id": i + 1,
                "instance_id": current_difficulty_instances[i].id,
                "success": success,
                "achievements_count": num_achievements,
                "achievements": achievements_list,
                "steps_taken": steps_taken,
                "turns_used": "unknown",  # Could track this if needed
            }
            detailed_results.append(episode_result)

            if success:
                successful_episodes += 1
            total_achievements += num_achievements

        avg_achievements = total_achievements / len(results_per_episode)

        # Print detailed trajectory information
        print(f"\n📊 Detailed Results for {model_name} on {label}:")
        print("-" * 80)
        for result in detailed_results:
            status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
            achievements_str = (
                ", ".join(result["achievements"]) if result["achievements"] else "None"
            )
            print(
                f"Episode {result['episode_id']}: {status} | "
                f"Steps: {result['steps_taken']} | "
                f"Achievements ({result['achievements_count']}): {achievements_str}"
            )
        print("-" * 80)

        print(
            f"Completed {label} for model {model_name}: {successful_episodes}/{len(results_per_episode)} successful, Avg. Achievements: {avg_achievements:.2f}"
        )

        results_for_model.append(
            {
                "Model": model_name,
                "Difficulty": label,
                "Successful Runs": f"{successful_episodes}/{len(results_per_episode)}",
                "Avg Unique Achievements": f"{avg_achievements:.2f}",
            }
        )

    return results_for_model


async def run_model_comparison_from_config():
    """Run model comparison using parameters from eval_config.toml"""
    # Load configuration
    config_path = Path(__file__).parent / "eval_config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config = toml.load(config_path)
    eval_config = config["evaluation"]

    models = eval_config["models"]
    difficulties = eval_config["difficulties"]
    max_turns = eval_config["max_turns"]
    n_trajectories = eval_config["trajectories_per_condition"]

    # Update global max_turns from config
    global agent_max_turns
    agent_max_turns = max_turns

    print("🎯 Crafter Multi-Action Model Comparison")
    print("=" * 50)
    print(f"Models: {', '.join(models)}")
    print(f"Difficulties: {', '.join(difficulties)}")
    print(f"Max turns: {max_turns}")
    print(f"Trajectories per condition: {n_trajectories}")
    print("=" * 50)

    all_results = []

    for model_name in models:
        print(f"\n🤖 Running {model_name}...")

        # Update the global variable for the model
        global current_model_name_for_eval
        current_model_name_for_eval = model_name

        model_results = await eval_react_crafter_legacy(
            model_name=model_name,
            formatting_model_name=model_name,
            modes=difficulties,
            n_instances_per_mode=n_trajectories,
        )

        all_results.extend(model_results)
        print(f"✅ {model_name} completed")

    print("\n" + "=" * 60)
    print("🏆 FINAL COMPARISON RESULTS")
    print("=" * 60)

    from tabulate import tabulate

    print(tabulate(all_results, headers="keys", tablefmt="github"))

    return all_results


if __name__ == "__main__":
    asyncio.run(run_model_comparison_from_config())
