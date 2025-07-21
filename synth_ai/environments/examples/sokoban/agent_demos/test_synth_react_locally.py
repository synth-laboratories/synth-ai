#!/usr/bin/env python3

import asyncio
import uuid
import pytest
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Deque
from pydantic import BaseModel, Field
from collections import deque
from synth_ai.zyk import LM
from synth_sdk.tracing.decorators import trace_event_async
from synth_sdk.tracing.abstractions import RewardSignal, Dataset, TrainingQuestion
from synth_sdk.tracing.utils import get_system_id
from synth_ai.environments.examples.sokoban.environment import (
    SokobanEnvironment,
    SokobanPublicState,
    SokobanPrivateState,
)
from synth_ai.environments.examples.sokoban.engine import (
    _grid_to_text,
    ACTION_STRING_TO_INT,
)
from synth_ai.environments.environment.shared_engine import (
    GetObservationCallable,
    InternalObservation,
)
from synth_ai.environments.examples.sokoban.taskset import (
    SokobanTaskInstance,
    SokobanTaskInstanceMetadata,
)
from synth_ai.environments.tasks.core import Impetus, Intent
from synth_ai.environments.environment.tools import EnvToolCall
from dataclasses import dataclass

import logging

logging.disable(logging.CRITICAL)


@dataclass
class AgentDecisionRecord:
    """Record of agent's decision-making process including messages, tool calls, and results."""

    action_int: int
    input_messages: List[Dict[str, Any]]
    output_messages: List[Dict[str, Any]]
    tool_calls: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    reasoning_text: str
    model_name: str
    raw_response: Any = None


# --- Helper function to format observation for LLM ---
def format_obs_for_llm_from_states(pub: SokobanPublicState, priv: SokobanPrivateState) -> str:
    room_text = _grid_to_text(pub.room_state)

    if pub.last_action_name.startswith("INVALID_ACTION_NO_CHANGE"):
        # Return a message indicating the invalid action directly, along with key state info
        return (
            f"Previous action ({pub.last_action_name.split(': ')[-1]}) resulted in NO CHANGE to the board.\n"
            f"{room_text}\n"
            f"Boxes on Target: {pub.boxes_on_target} / {pub.num_boxes}\n"
            f"Steps Taken: {pub.num_steps} / {pub.max_steps}\n"
            f"Terminated: {priv.terminated}\n"
            f"Last Reward: {priv.reward_last}"
        )

    # Default formatting for valid actions or initial state
    return (
        f"{room_text}\n"
        f"Boxes on Target: {pub.boxes_on_target} / {pub.num_boxes}\n"
        f"Steps Taken: {pub.num_steps} / {pub.max_steps}\n"
        f"Terminated: {priv.terminated}\n"
        f"Last Reward: {priv.reward_last}"
    )


# ---------------------------------- custom observation callable ------------------------------ #
class HistoryObservationCallable(GetObservationCallable):
    def __init__(self, max_history: int = 3):
        self._hist: Deque[str] = deque(maxlen=max_history)

    async def get_observation(
        self, pub: SokobanPublicState, priv: SokobanPrivateState
    ) -> InternalObservation:
        if pub is None or priv is None:
            # This case might occur if env.terminate() is called and doesn't provide full states.
            # For normal steps/reset, pub/priv should be valid.
            # Consider how to handle this if it becomes an issue.
            # For now, returning a dict that leads to an error or specific handling.
            return {
                "error": "Missing public or private state in get_observation",
                "history_boards": list(self._hist),
            }  # type: ignore[return-value]

        current_board_text = _grid_to_text(pub.room_state)
        self._hist.append(current_board_text)

        # Return public and private states along with history of board strings
        return {"public": pub, "private": priv, "history_boards": list(self._hist)}  # type: ignore[return-value]


# --- Pydantic Models for Tool Arguments ---
class SokobanInteractArgs(BaseModel):
    actions_list: List[str] = Field(
        description="List of actions to execute. Valid actions: move up, move down, move left, move right, push up, push down, push left, push right, no operation"
    )
    reasoning: str = Field(description="Reasoning for the chosen actions")


class TerminateArgs(BaseModel):
    reasoning: str = Field(description="Reasoning for terminating the agent")


# --- tiny ReAct agent -------------------------------------------------- #
class Move(EnvToolCall):
    def __init__(self, action: int):
        super().__init__(tool="interact", args={"action": action})


class ReActAgent:
    def __init__(self, llm, max_turns: int = 10):
        self.llm, self.max_turns = llm, max_turns
        self.history: List[Dict[str, Any]] = []
        self.system_name: str = "sokoban-react-ex"
        self.system_id: Any = get_system_id(self.system_name)
        self.system_instance_id: str = str(uuid.uuid4())
        self.last_obs_dict: Optional[Dict[str, Any]] = None
        self.num_total_boxes: int = 0

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "sokoban_interact",
                    "description": "Interacts with the Sokoban environment by proposing a single action.",
                    "parameters": SokobanInteractArgs.model_json_schema(),
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "terminate",
                    "description": "Terminates the agent's execution if the puzzle is solved or no further moves are required.",
                    "parameters": TerminateArgs.model_json_schema(),
                },
            },
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
    async def decide(self, obs: str) -> AgentDecisionRecord:
        """
        Make a decision and return a complete record of the reasoning process.
        """
        self.history.append({"type": "obs", "content": obs})

        formatted_prompt_history = self._format_history_for_prompt()
        user_prompt = f"{formatted_prompt_history}\n\nBased on the history above, particularly the last observation, what is your reasoning and which tool should you call next?"

        system_prompt = (
            "You are an agent playing Sokoban. Your goal is to push all boxes onto the target locations. "
            "Review the history of observations, thoughts, and actions. "
            "Based on this history, particularly the last observation, decide on the best next action. "
            "You MUST call one of the two available tools: `sokoban_interact` or `terminate`.\n\n"
            "Action Guide:\n"
            "- Use 'move' actions (move up, move down, move left, move right) when moving to empty spaces\n"
            "- Use 'push' actions (push up, push down, push left, push right) when pushing boxes onto targets\n"
            "- Use 'no operation' to skip a turn\n\n"
            "Please use the tools available to you. Do not attempt to include a tool call in your reasoning"
        )

        # Create input messages
        input_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Get response from LLM
        response_obj = await self.llm.respond_async(
            system_message=system_prompt, user_message=user_prompt, tools=self.tools
        )

        # Initialize record fields
        tool_calls = []
        tool_results = []
        reasoning_text = ""
        action_int = ACTION_STRING_TO_INT["no operation"]  # Default fallback

        # Handle cases where the model does **not** return a structured tool call.
        if not getattr(response_obj, "tool_calls", None):
            # Record fallback in history for later analysis
            reasoning_text = "LLM failed to provide tool_calls; fallback action."
            self.history.append(
                {
                    "type": "tool_call",
                    "tool_name": "sokoban_interact",
                    "tool_arguments": {
                        "actions_list": ["move down"],
                        "reasoning": reasoning_text,
                    },
                }
            )
            action_int = ACTION_STRING_TO_INT["move down"]

            # Create fallback tool call record
            tool_calls = [
                {
                    "id": "fallback_0",
                    "type": "function",
                    "function": {
                        "name": "sokoban_interact",
                        "arguments": json.dumps(
                            {"actions_list": ["move down"], "reasoning": reasoning_text}
                        ),
                    },
                }
            ]

            tool_results = [{"tool_call_id": "fallback_0", "content": "Fallback action: move down"}]
        else:
            # Process successful tool calls
            response_tool_calls = None

            try:
                if hasattr(response_obj, "tool_calls") and response_obj.tool_calls:
                    response_tool_calls = response_obj.tool_calls
                elif isinstance(response_obj, str):
                    try:
                        potential_tool_call_json = json.loads(response_obj)
                        if (
                            isinstance(potential_tool_call_json, dict)
                            and "tool_calls" in potential_tool_call_json
                        ):
                            response_tool_calls = potential_tool_call_json["tool_calls"]
                        elif (
                            isinstance(potential_tool_call_json, list)
                            and len(potential_tool_call_json) > 0
                            and potential_tool_call_json[0].get("type") == "function"
                        ):
                            response_tool_calls = potential_tool_call_json
                    except json.JSONDecodeError:
                        pass

                if response_tool_calls and len(response_tool_calls) > 0:
                    tool_call_data = response_tool_calls[0]

                    tool_name = ""
                    tool_args_str = ""

                    if (
                        hasattr(tool_call_data, "function")
                        and hasattr(tool_call_data.function, "name")
                        and hasattr(tool_call_data.function, "arguments")
                    ):
                        tool_name = tool_call_data.function.name
                        tool_args_str = tool_call_data.function.arguments
                    elif (
                        isinstance(tool_call_data, dict)
                        and "function" in tool_call_data
                        and isinstance(tool_call_data["function"], dict)
                    ):
                        tool_name = tool_call_data["function"].get("name")
                        tool_args_str = tool_call_data["function"].get("arguments")
                        if not isinstance(tool_args_str, str):
                            tool_args_str = json.dumps(tool_args_str)

                    if tool_name and tool_args_str:
                        tool_arguments = json.loads(tool_args_str)

                        # Create proper tool call record
                        tool_calls = [
                            {
                                "id": f"call_{len(self.history)}",
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": tool_args_str,
                                },
                            }
                        ]

                        # Record in history
                        self.history.append(
                            {
                                "type": "tool_call",
                                "tool_name": tool_name,
                                "tool_arguments": tool_arguments,
                            }
                        )

                        # Process the tool call
                        if tool_name == "sokoban_interact":
                            validated_args = SokobanInteractArgs(**tool_arguments)
                            reasoning_text = validated_args.reasoning

                            if validated_args.actions_list:
                                action_str = validated_args.actions_list[0]
                                action_int = ACTION_STRING_TO_INT.get(
                                    action_str.lower(),
                                    ACTION_STRING_TO_INT["no operation"],
                                )

                                tool_results = [
                                    {
                                        "tool_call_id": f"call_{len(self.history) - 1}",
                                        "content": f"Executed action: {action_str}",
                                    }
                                ]
                            else:
                                tool_results = [
                                    {
                                        "tool_call_id": f"call_{len(self.history) - 1}",
                                        "content": "No action specified, using no operation",
                                    }
                                ]

                        elif tool_name == "terminate":
                            validated_args = TerminateArgs(**tool_arguments)
                            reasoning_text = validated_args.reasoning

                            # Check if termination is valid
                            if self.last_obs_dict:
                                terminated_by_env = self.last_obs_dict.get("terminated", False)
                                boxes_on_target = int(self.last_obs_dict.get("boxes_on_target", 0))
                                is_solved_state = (
                                    self.num_total_boxes > 0
                                    and boxes_on_target == self.num_total_boxes
                                )

                                if terminated_by_env or is_solved_state:
                                    action_int = -1  # Terminate
                                    tool_results = [
                                        {
                                            "tool_call_id": f"call_{len(self.history) - 1}",
                                            "content": "Termination accepted - puzzle solved or environment terminated",
                                        }
                                    ]
                                else:
                                    action_int = ACTION_STRING_TO_INT["no operation"]
                                    tool_results = [
                                        {
                                            "tool_call_id": f"call_{len(self.history) - 1}",
                                            "content": f"Termination rejected - puzzle not solved. Boxes on target: {boxes_on_target}/{self.num_total_boxes}",
                                        }
                                    ]
                            else:
                                action_int = ACTION_STRING_TO_INT["no operation"]
                                tool_results = [
                                    {
                                        "tool_call_id": f"call_{len(self.history) - 1}",
                                        "content": "Termination rejected - cannot verify puzzle state",
                                    }
                                ]

            except Exception as e:
                reasoning_text = f"Error processing LLM response: {str(e)}"
                self.history.append({"type": "error", "content": reasoning_text})
                action_int = ACTION_STRING_TO_INT["no operation"]

                tool_calls = [
                    {
                        "id": f"error_{len(self.history)}",
                        "type": "function",
                        "function": {
                            "name": "sokoban_interact",
                            "arguments": json.dumps(
                                {
                                    "actions_list": ["no operation"],
                                    "reasoning": reasoning_text,
                                }
                            ),
                        },
                    }
                ]

                tool_results = [
                    {
                        "tool_call_id": f"error_{len(self.history)}",
                        "content": f"Error occurred: {str(e)}",
                    }
                ]

        # Create output messages
        output_messages = [
            {"role": "assistant", "content": reasoning_text, "tool_calls": tool_calls}
        ]

        # Add tool results as separate messages
        for tool_result in tool_results:
            output_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_result["tool_call_id"],
                    "content": tool_result["content"],
                }
            )

        # Create and return the decision record
        return AgentDecisionRecord(
            action_int=action_int,
            input_messages=input_messages,
            output_messages=output_messages,
            tool_calls=tool_calls,
            tool_results=tool_results,
            reasoning_text=reasoning_text,
            model_name=self.llm.model_name,
            raw_response=response_obj,
        )


# --- test ---------------------------------------------------------------- #
SIMPLE_SNAPSHOT: Dict[str, Any] = {
    "dim_room": [4, 4],
    "room_fixed": [
        [0, 0, 0, 0],
        [0, 1, 2, 1],  # target at (1,2)
        [0, 1, 1, 1],
        [0, 0, 0, 0],
    ],
    "room_state": [
        [0, 0, 0, 0],
        [0, 1, 1, 1],
        [0, 1, 4, 1],  # box at (2,2)
        [0, 5, 1, 1],  # player at (3,1)
    ],
    "boxes_on_target": 0,
    "max_steps": 10,
    "num_boxes": 1,
}


@pytest.mark.asyncio
async def test_react_agent_sokoban(tmp_path: Path):
    inst = SokobanTaskInstance(
        id=uuid.uuid4(),
        impetus=Impetus(instructions="solve"),
        intent=Intent(rubric={}, gold_trajectories=None, gold_state_diff={}),
        metadata=SokobanTaskInstanceMetadata("easy", 1, (4, 4), 10, -1, -1, "unit"),
        is_reproducible=True,
        initial_engine_snapshot=SIMPLE_SNAPSHOT,
    )
    hist_cb = HistoryObservationCallable(max_history=3)
    env = SokobanEnvironment(inst, custom_step_obs=hist_cb)
    env.engine.package_sokoban_env.render_mode = "raw"  # type: ignore[attr-defined]

    llm = LM(model_name="gpt-4.1-nano", formatting_model_name="gpt-4.1-nano", temperature=0.0)
    agent = ReActAgent(llm)

    async def run_episode():
        obs_payload = await env.initialize()

        # Ensure payload is not an error structure from callable
        if "error" in obs_payload:
            return False  # Or handle error appropriately

        agent.last_obs_dict = {
            "terminated": obs_payload["private"].terminated,
            "boxes_on_target": obs_payload["public"].boxes_on_target,
        }
        agent.num_total_boxes = obs_payload["public"].num_boxes
        current_input_to_agent = format_obs_for_llm_from_states(
            obs_payload["public"], obs_payload["private"]
        )

        for turn in range(agent.max_turns):
            decision_record = await agent.decide(current_input_to_agent)
            act_idx = decision_record.action_int

            if act_idx == -1:
                obs_payload_next = obs_payload
                break

            step_result = await env.step([[Move(act_idx)]])

            obs_payload_next = step_result
            if "error" in obs_payload_next:
                break  # Or handle error appropriately

            agent.last_obs_dict = {
                "terminated": obs_payload_next["private"].terminated,
                "boxes_on_target": obs_payload_next["public"].boxes_on_target,
            }
            # agent.num_total_boxes is assumed constant after initialization

            agent.history.append({"type": "tool_response", "content": "Action executed"})

            current_input_to_agent = format_obs_for_llm_from_states(
                obs_payload_next["public"], obs_payload_next["private"]
            )

            # obs_payload_next["history_boards"] already contains the history *including* the most recent board
            # due to how _hist.append() and list(self._hist) is structured in the callable now.
            # So, history_boards is a list of the N most recent board states, newest last.
            displayed_boards = obs_payload_next["history_boards"]
            # for i, board_text in enumerate(displayed_boards):
            # t-0 is the newest, t-(N-1) is the oldest in the deque

            obs_payload = obs_payload_next

            if obs_payload_next["private"].terminated:
                break

        if "obs_payload_next" not in locals():
            obs_payload_next = obs_payload

        if "error" in obs_payload_next:
            return False  # Indicate failure

        return obs_payload_next["private"].terminated

    solved_status = await run_episode()
    dataset = Dataset(
        questions=[TrainingQuestion(id="sokoban_ep", intent="solve", criteria="solved")],
        reward_signals=[
            RewardSignal(
                question_id="sokoban_ep",
                system_instance_id=agent.system_instance_id,
                reward=1 if solved_status else 0,
                annotation=json.dumps({"agent_history": agent.history}),
            )
        ],
    )
    # upload(dataset=dataset)
    # assert solved_status

    # Print the agent's final reward using checkpoint observation
    # final_obs = await env.checkpoint()
    # if isinstance(final_obs, dict):
    # else:


async def eval_react_sokoban(
    model_name: str = "gpt-4.1-nano",  # Default will be overridden by caller
    formatting_model_name: str = "gpt-4.1-nano",  # Default will be overridden by caller
) -> List[Dict[str, Any]]:
    """
    Run ReAct agents on Sokoban instances of different difficulties for a given model,
    and returns a list of dictionaries containing aggregated results for each mode.
    """
    from synth_ai.environments.examples.sokoban.engine_helpers.room_utils import (
        generate_room,
        get_shortest_action_path,
    )
    import asyncio
    import uuid

    current_model_name_for_eval = model_name  # Use passed-in model name

    _temp_llm_for_names = LM(
        model_name=current_model_name_for_eval,
        formatting_model_name=formatting_model_name,  # Use passed-in formatting model name
        temperature=0.0,
    )
    _temp_agent_for_names = ReActAgent(_temp_llm_for_names)
    actual_system_name = _temp_agent_for_names.system_name

    # Helper to run a single episode (remains largely the same, but uses current_model_name_for_eval)
    async def run_episode(inst) -> bool:
        """Run a single agent/instance episode and return True on success."""
        hist_cb = HistoryObservationCallable(max_history=3)
        env = SokobanEnvironment(inst, custom_step_obs=hist_cb)
        env.engine.package_sokoban_env.render_mode = "raw"  # type: ignore[attr-defined]
        llm_for_episode = LM(
            model_name=current_model_name_for_eval,  # Uses the model for this eval_react_sokoban call
            formatting_model_name=formatting_model_name,  # Uses the formatting model for this call
            temperature=0.0,
        )
        agent = ReActAgent(llm_for_episode)

        obs = await env.initialize()
        agent.last_obs_dict = {
            "terminated": obs["private"].terminated,
            "boxes_on_target": obs["public"].boxes_on_target,
        }
        agent.num_total_boxes = obs["public"].num_boxes
        prompt_obs = format_obs_for_llm_from_states(obs["public"], obs["private"])

        for _ in range(agent.max_turns):
            decision_record = await agent.decide(prompt_obs)
            act_idx = decision_record.action_int
            if act_idx == -1:  # agent terminated
                break
            obs = await env.step([{"tool": "interact", "args": {"action": act_idx}}])
            if "error" in obs:  # safety guard
                return False
            agent.last_obs_dict = {
                "terminated": obs["private"].terminated,
                "boxes_on_target": obs["public"].boxes_on_target,
            }
            agent.history.append({"type": "tool_response", "content": "Action executed"})
            prompt_obs = format_obs_for_llm_from_states(obs["public"], obs["private"])
            if obs["private"].terminated:  # env solved
                break
        return obs["private"].terminated

    # Instance factory (remains the same)
    async def make_instances(label: str, target_len: int, n: int = 3):
        instances = []
        seed = 0
        while len(instances) < n:
            room_structure, room_state, _, _ = generate_room(
                dim=(5, 5),
                initial_seed=seed,
                num_boxes=1,
                search_depth=max(10, target_len + 2),
            )
            path = get_shortest_action_path(room_structure, room_state, MAX_DEPTH=20)
            if len(path) == target_len:
                inst = SokobanTaskInstance(
                    id=uuid.uuid4(),
                    impetus=Impetus(instructions="Solve"),
                    intent=Intent(rubric={}, gold_trajectories=None, gold_state_diff={}),
                    metadata=SokobanTaskInstanceMetadata(
                        label, 1, (5, 5), 20, len(path), seed, f"len={target_len}"
                    ),
                    is_reproducible=True,
                    initial_engine_snapshot={
                        "dim_room": (5, 5),
                        "room_fixed": room_structure,
                        "room_state": room_state,
                        "boxes_on_target": 0,
                        "max_steps": 20,
                        "num_boxes": 1,
                    },
                )
                instances.append(inst)
            seed += 1
        return instances

    # Evaluation logic
    configs = [("ultra-easy", 1), ("easy", 3), ("medium", 5)]
    results_for_this_model = []  # Store list of dicts for this model's run

    print(
        f"\nStarting Sokoban ReAct Agent Evaluation for Model: {current_model_name_for_eval}, System: {actual_system_name}"
    )

    for label, step_len in configs:
        print(f"  Processing difficulty: {label} for model {current_model_name_for_eval}...")
        insts = await make_instances(label, step_len, n=3)  # 3 instances per difficulty
        solved_statuses = await asyncio.gather(*(run_episode(i) for i in insts))
        num_solved = sum(solved_statuses)
        rate = num_solved / len(insts) if insts else 0.0
        results_for_this_model.append(
            {
                "Model": current_model_name_for_eval,
                "Difficulty": label,
                "Solved": f"{num_solved}/{len(insts)}",
                "Success Rate": f"{rate:.0%}",
            }
        )
        print(
            f"    Completed {label} for model {current_model_name_for_eval}: {num_solved}/{len(insts)} solved ({rate:.0%})"
        )

    return results_for_this_model


if __name__ == "__main__":
    # asyncio.run(eval_react_sokoban()) # Old way of running a single model

    async def run_all_sokoban_evals_parallel():
        models_to_evaluate = [
            {"model_name": "gpt-4.1-nano", "formatting_model_name": "gpt-4.1-nano"},
            {"model_name": "gpt-4.1", "formatting_model_name": "gpt-4.1"},
            {
                "model_name": "o4-mini",
                "formatting_model_name": "o4-mini",
            },  # Assuming o4-mini uses itself for formatting
        ]

        print("Starting parallel Sokoban evaluation for all specified models...")

        # eval_react_sokoban returns List[Dict[str, Any]]
        # all_model_results will be a List[List[Dict[str, Any]]]
        all_model_results = await asyncio.gather(
            *[
                eval_react_sokoban(
                    model_name=model_config["model_name"],
                    formatting_model_name=model_config["formatting_model_name"],
                )
                for model_config in models_to_evaluate
            ]
        )

        print("\n=== ALL SOKOBAN EVALUATIONS COMPLETED ===")

        # Flatten the list of lists into a single list of dictionaries
        combined_sokoban_results = []
        for model_result_list in all_model_results:
            combined_sokoban_results.extend(model_result_list)

        print("\n--- Combined Sokoban Evaluation Summary Table ---")
        from tabulate import tabulate  # Ensure tabulate is imported

        if combined_sokoban_results:
            # Headers="keys" will use the dictionary keys as headers
            print(
                tabulate(
                    combined_sokoban_results,
                    headers="keys",
                    tablefmt="github",
                )
            )
        else:
            print("No Sokoban evaluation data to display.")

    asyncio.run(run_all_sokoban_evals_parallel())

# Model: o4-mini, System: sokoban-react-ex
# | Difficulty   | Solved   | Success Rate   |
# |--------------|----------|----------------|
# | ultra-easy   | 3/3      | 100%           |
# | easy         | 3/3      | 100%           |
# | medium       | 3/3      | 100%           |


# Model: gpt-4.1, System: sokoban-react-ex
# | Difficulty   | Solved   | Success Rate   |
# |--------------|----------|----------------|
# | ultra-easy   | 1/3      | 33%            |
# | easy         | 0/3      | 0%             |
# | medium       | 0/3      | 0%             |

# Model: gpt-4.1-nano, System: sokoban-react-ex
# | Difficulty   | Solved   | Success Rate   |
# |--------------|----------|----------------|
# | ultra-easy   | 0/3      | 0%             |
# | easy         | 0/3      | 0%             |
# | medium       | 0/3      | 0%             |
