#!/usr/bin/env python3
"""
Comprehensive Sokoban evaluation framework with trace generation.
Generates proper trace files for the Streamlit viewer.
"""

import asyncio
import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import time
import os

from synth_ai.zyk import LM
from synth_sdk.tracing.decorators import trace_event_async
from synth_sdk.tracing.abstractions import (
    RewardSignal,
    Dataset,
    TrainingQuestion,
    EventPartitionElement,
    SystemTrace,
)
from synth_sdk.tracing.utils import get_system_id

from synth_ai.environments.examples.sokoban.environment import SokobanEnvironment
from synth_ai.environments.examples.sokoban.taskset import (
    SokobanTaskInstance,
    SokobanTaskInstanceMetadata,
)
from synth_ai.environments.examples.sokoban.engine import _grid_to_text, ACTION_STRING_TO_INT
from synth_ai.environments.examples.sokoban.engine_helpers.room_utils import (
    generate_room,
    get_shortest_action_path,
)
from synth_ai.environments.tasks.core import Impetus, Intent
from synth_ai.environments.environment.tools import EnvToolCall

from test_synth_react_locally import (
    ReActAgent,
    HistoryObservationCallable,
    format_obs_for_llm_from_states,
    SokobanInteractArgs,
    Move,
    AgentDecisionRecord,
)


@dataclass
class SokobanTrajectoryResult:
    """Result from a single Sokoban trajectory."""

    trajectory_id: str
    model_name: str
    difficulty: str
    seed: int
    success: bool
    final_reward: float
    num_steps: int
    boxes_solved: int
    total_boxes: int
    trace_file_path: str
    metadata: Dict[str, Any]


class SokobanEvalFramework:
    """Comprehensive evaluation framework for Sokoban with trace generation."""

    def __init__(self):
        self.trajectory_results: List[SokobanTrajectoryResult] = []

    async def run_single_trajectory_with_trace(
        self,
        model_name: str,
        difficulty: str,
        seed: int,
        max_turns: int = 20,
        collect_detailed_data: bool = True,
        eval_dir: Path = None,
    ) -> SokobanTrajectoryResult:
        """Run a single trajectory with comprehensive trace capture."""

        # Generate Sokoban instance
        difficulty_configs = {
            "ultra-easy": {"target_len": 1, "dim": (5, 5), "boxes": 1},
            "easy": {"target_len": 3, "dim": (5, 5), "boxes": 1},
            "medium": {"target_len": 5, "dim": (6, 6), "boxes": 1},
            "hard": {"target_len": 7, "dim": (7, 7), "boxes": 2},
        }

        config = difficulty_configs.get(difficulty, difficulty_configs["easy"])

        # Generate room
        room_structure, room_state, _, _ = generate_room(
            dim=config["dim"],
            initial_seed=seed,
            num_boxes=config["boxes"],
            search_depth=max(10, config["target_len"] + 2),
        )

        # Convert numpy arrays to lists for JSON serialization
        room_structure = room_structure.tolist()
        room_state = room_state.tolist()

        # Create task instance
        metadata = SokobanTaskInstanceMetadata(
            difficulty=difficulty,
            num_boxes=config["boxes"],
            dim_room=config["dim"],
            max_steps=max_turns,
            shortest_path_length=config["target_len"],
            seed=seed,
            generation_params=f"dim={config['dim']}, boxes={config['boxes']}, steps={max_turns}",
        )

        instance = SokobanTaskInstance(
            id=uuid.uuid4(),
            impetus=Impetus(
                instructions="Solve this Sokoban puzzle by pushing all boxes onto targets."
            ),
            intent=Intent(rubric={}, gold_trajectories=None, gold_state_diff={}),
            metadata=metadata,
            is_reproducible=True,
            initial_engine_snapshot={
                "dim_room": config["dim"],
                "room_fixed": room_structure,
                "room_state": room_state,
                "boxes_on_target": 0,
                "max_steps": max_turns,
                "num_boxes": config["boxes"],
            },
        )

        # Setup environment and agent
        hist_cb = HistoryObservationCallable(max_history=1)
        env = SokobanEnvironment(instance, custom_step_obs=hist_cb)

        llm = LM(model_name=model_name, formatting_model_name=model_name, temperature=0.0)
        agent = ReActAgent(llm, max_turns=max_turns)

        # Initialize tracking
        trajectory_id = str(uuid.uuid4())
        turn_count = 0
        actions_per_turn = []
        turn_by_turn_data = [] if collect_detailed_data else None
        partition_index = 0

        # Initialize environment
        obs_payload = await env.initialize()
        if "error" in obs_payload:
            raise Exception(f"Environment initialization failed: {obs_payload['error']}")

        # Record initial turn before any agent action
        initial_pub_state = obs_payload["public"]
        initial_priv_state = obs_payload["private"]
        initial_turn_data = {
            "turn_number": 0,
            "room_text": _grid_to_text(initial_pub_state.room_state),
            "player_position": [int(x) for x in initial_pub_state.player_position],
            "boxes_on_target": int(initial_pub_state.boxes_on_target),
            "num_steps": int(initial_pub_state.num_steps),
            "last_action": "Initial",
            "terminated": bool(initial_priv_state.terminated),
            "truncated": bool(initial_priv_state.truncated),
            "reward": float(initial_priv_state.reward_last),
            "total_reward": float(initial_priv_state.total_reward),
            "action_taken": -1,
            "action_name": "initial",
        }
        if collect_detailed_data:
            turn_by_turn_data.append(initial_turn_data)
        partition_index = 0
        event_partition_initial = EventPartitionElement(
            partition_index=partition_index,
            events=[
                {
                    "event_type": "sokoban_turn",
                    "event_metadata": {
                        "turn_number": 0,
                        "turn_data": initial_turn_data,
                    },
                    "environment_compute_steps": [
                        {
                            "compute_output": [
                                {
                                    "outputs": {
                                        "room_text": initial_turn_data["room_text"],
                                        "action": -1,
                                        "action_name": "initial",
                                        "player_position": initial_turn_data["player_position"],
                                        "boxes_on_target": initial_turn_data["boxes_on_target"],
                                        "num_steps": initial_turn_data["num_steps"],
                                        "reward": initial_turn_data["reward"],
                                        "terminated": initial_turn_data["terminated"],
                                        "truncated": initial_turn_data["truncated"],
                                    }
                                }
                            ]
                        }
                    ],
                }
            ],
        )
        partition_index += 1

        agent.last_obs_dict = {
            "terminated": obs_payload["private"].terminated,
            "boxes_on_target": obs_payload["public"].boxes_on_target,
        }
        agent.num_total_boxes = obs_payload["public"].num_boxes

        try:
            while turn_count < max_turns:
                turn_count += 1

                current_formatted_obs = format_obs_for_llm_from_states(
                    obs_payload["public"], obs_payload["private"]
                )

                # Get current game state for trace
                pub_state = obs_payload["public"]
                priv_state = obs_payload["private"]

                # Create turn data
                turn_data = {
                    "turn_number": turn_count,
                    "room_text": _grid_to_text(pub_state.room_state),
                    "player_position": [int(x) for x in pub_state.player_position],
                    "boxes_on_target": int(pub_state.boxes_on_target),
                    "num_steps": int(pub_state.num_steps),
                    "last_action": pub_state.last_action_name,
                    "terminated": priv_state.terminated,
                    "truncated": priv_state.truncated,
                    "reward": float(priv_state.reward_last),
                    "total_reward": float(priv_state.total_reward),
                }

                # Agent decision - get full reasoning record
                decision_record = await agent.decide(current_formatted_obs)
                action_int = decision_record.action_int

                if action_int == -1:  # Agent terminated
                    break

                # Execute action
                obs_payload_next = await env.step([Move(action_int)])

                if "error" in obs_payload_next:
                    break

                # Update turn data with action taken
                turn_data["action_taken"] = action_int
                turn_data["action_name"] = (
                    list(ACTION_STRING_TO_INT.keys())[
                        list(ACTION_STRING_TO_INT.values()).index(action_int)
                    ]
                    if action_int in ACTION_STRING_TO_INT.values()
                    else f"unknown_{action_int}"
                )

                # Store detailed turn data
                if collect_detailed_data:
                    turn_by_turn_data.append(turn_data)

                # Create event partition for this turn with BOTH agent and environment compute steps
                event_partition = EventPartitionElement(
                    partition_index=partition_index,
                    events=[
                        {
                            "event_type": "sokoban_turn",
                            "event_metadata": {
                                "turn_number": turn_count,
                                "boxes_on_target": pub_state.boxes_on_target,
                                "total_boxes": pub_state.num_boxes,
                                "action_taken": turn_data["action_name"],
                                "player_position": turn_data["player_position"],
                            },
                            "agent_compute_step": {
                                "event_order": 1,
                                "compute_began": datetime.now().isoformat(),
                                "compute_ended": datetime.now().isoformat(),
                                "model_name": decision_record.model_name,
                                "model_params": {"temperature": 0.0},
                                "compute_input": [{"messages": decision_record.input_messages}],
                                "compute_output": [{"messages": decision_record.output_messages}],
                            },
                            "environment_compute_steps": [
                                {
                                    "event_order": 2,
                                    "compute_began": datetime.now().isoformat(),
                                    "compute_ended": datetime.now().isoformat(),
                                    "compute_input": [
                                        {
                                            "action": action_int,
                                            "action_name": turn_data["action_name"],
                                        }
                                    ],
                                    "compute_output": [
                                        {
                                            "outputs": {
                                                "room_text": turn_data["room_text"],
                                                "action": action_int,
                                                "action_name": turn_data["action_name"],
                                                "player_position": turn_data["player_position"],
                                                "boxes_on_target": turn_data["boxes_on_target"],
                                                "num_steps": turn_data["num_steps"],
                                                "reward": turn_data["reward"],
                                                "terminated": turn_data["terminated"],
                                                "truncated": turn_data["truncated"],
                                            }
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                )

                actions_per_turn.append(action_int)
                partition_index += 1

                # Update for next iteration
                obs_payload = obs_payload_next
                agent.last_obs_dict = {
                    "terminated": obs_payload["private"].terminated,
                    "boxes_on_target": obs_payload["public"].boxes_on_target,
                }

                # Check termination - if terminated, record final state
                if obs_payload["private"].terminated or obs_payload["private"].truncated:
                    # Record final state after the terminating action
                    final_pub_state = obs_payload["public"]
                    final_priv_state = obs_payload["private"]

                    final_turn_data = {
                        "turn_number": turn_count + 1,
                        "room_text": _grid_to_text(final_pub_state.room_state),
                        "player_position": [int(x) for x in final_pub_state.player_position],
                        "boxes_on_target": int(final_pub_state.boxes_on_target),
                        "num_steps": int(final_pub_state.num_steps),
                        "last_action": final_pub_state.last_action_name,
                        "terminated": final_priv_state.terminated,
                        "truncated": final_priv_state.truncated,
                        "reward": float(final_priv_state.reward_last),
                        "total_reward": float(final_priv_state.total_reward),
                        "action_taken": -1,  # No action taken in final state
                        "action_name": "final_state",
                    }

                    if collect_detailed_data:
                        turn_by_turn_data.append(final_turn_data)

                    # Create event partition for final state
                    final_event_partition = EventPartitionElement(
                        partition_index=partition_index,
                        events=[
                            {
                                "event_type": "sokoban_turn",
                                "event_metadata": {
                                    "turn_number": turn_count + 1,
                                    "turn_data": final_turn_data,
                                },
                                "environment_compute_steps": [
                                    {
                                        "compute_output": [
                                            {
                                                "outputs": {
                                                    "room_text": final_turn_data["room_text"],
                                                    "action": -1,
                                                    "action_name": "final_state",
                                                    "player_position": final_turn_data[
                                                        "player_position"
                                                    ],
                                                    "boxes_on_target": final_turn_data[
                                                        "boxes_on_target"
                                                    ],
                                                    "num_steps": final_turn_data["num_steps"],
                                                    "reward": final_turn_data["reward"],
                                                    "terminated": final_turn_data["terminated"],
                                                    "truncated": final_turn_data["truncated"],
                                                }
                                            }
                                        ]
                                    }
                                ],
                            }
                        ],
                    )
                    partition_index += 1
                    break

        except Exception as e:
            print(f"Error during trajectory execution: {e}")

        # Final state
        final_private_state = obs_payload["private"]
        final_public_state = obs_payload["public"]

        success = bool(final_public_state.boxes_on_target == final_public_state.num_boxes)
        final_reward = float(final_private_state.total_reward)
        num_steps = int(final_public_state.num_steps)

        # Create trace data
        trace_data = {
            "trace": {
                "metadata": {
                    "model_name": model_name,
                    "difficulty": difficulty,
                    "seed": seed,
                    "trajectory_id": trajectory_id,
                    "success": success,
                    "final_reward": final_reward,
                    "num_steps": num_steps,
                    "boxes_solved": int(final_public_state.boxes_on_target),
                    "total_boxes": int(final_public_state.num_boxes),
                    "max_turns": max_turns,
                },
                "partition": [
                    {
                        "partition_index": i,
                        "events": [
                            {
                                "event_type": "sokoban_turn",
                                "event_metadata": {
                                    "turn_number": i + 1,
                                    "turn_data": turn_data,
                                },
                                "agent_compute_step": {
                                    "event_order": 1,
                                    "compute_began": datetime.now().isoformat(),
                                    "compute_ended": datetime.now().isoformat(),
                                    "model_name": model_name,
                                    "model_params": {"temperature": 0.0},
                                    "compute_input": [
                                        {
                                            "messages": [
                                                {
                                                    "role": "system",
                                                    "content": "You are playing Sokoban. Push all boxes onto targets.",
                                                },
                                                {
                                                    "role": "user",
                                                    "content": f"Turn {i + 1}: {turn_data['room_text']}",
                                                },
                                            ]
                                        }
                                    ],
                                    "compute_output": [
                                        {
                                            "messages": [
                                                {
                                                    "role": "assistant",
                                                    "content": f"Taking action: {turn_data.get('action_name', 'initial')}",
                                                    "tool_calls": [
                                                        {
                                                            "id": f"turn_{i + 1}",
                                                            "type": "function",
                                                            "function": {
                                                                "name": "sokoban_interact",
                                                                "arguments": json.dumps(
                                                                    {
                                                                        "actions_list": [
                                                                            turn_data.get(
                                                                                "action_name",
                                                                                "initial",
                                                                            )
                                                                        ],
                                                                        "reasoning": f"Turn {i + 1} action",
                                                                    }
                                                                ),
                                                            },
                                                        }
                                                    ],
                                                },
                                                {
                                                    "role": "tool",
                                                    "tool_call_id": f"turn_{i + 1}",
                                                    "content": f"Executed: {turn_data.get('action_name', 'initial')}",
                                                },
                                            ]
                                        }
                                    ],
                                },
                                "environment_compute_steps": [
                                    {
                                        "event_order": 2,
                                        "compute_began": datetime.now().isoformat(),
                                        "compute_ended": datetime.now().isoformat(),
                                        "compute_input": [
                                            {
                                                "action": turn_data.get("action_taken", -1),
                                                "action_name": turn_data.get(
                                                    "action_name", "initial"
                                                ),
                                            }
                                        ],
                                        "compute_output": [
                                            {
                                                "outputs": {
                                                    "room_text": turn_data["room_text"],
                                                    "action": turn_data.get("action_taken", -1),
                                                    "action_name": turn_data.get(
                                                        "action_name", "initial"
                                                    ),
                                                    "player_position": turn_data["player_position"],
                                                    "boxes_on_target": turn_data["boxes_on_target"],
                                                    "num_steps": turn_data["num_steps"],
                                                    "reward": turn_data["reward"],
                                                    "terminated": turn_data["terminated"],
                                                    "truncated": turn_data["truncated"],
                                                }
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                    for i, turn_data in enumerate(turn_by_turn_data or [])
                ],
            },
            "dataset": {
                "questions": [
                    {
                        "id": "sokoban_puzzle",
                        "intent": "solve",
                        "criteria": "push_all_boxes_to_targets",
                    }
                ],
                "reward_signals": [
                    {
                        "question_id": "sokoban_puzzle",
                        "system_instance_id": agent.system_instance_id,
                        "reward": final_reward,
                        "annotation": json.dumps(
                            {
                                "success": success,
                                "boxes_solved": final_public_state.boxes_on_target,
                                "total_boxes": final_public_state.num_boxes,
                                "num_steps": num_steps,
                                "actions_taken": len(actions_per_turn),
                            }
                        ),
                    }
                ],
            },
        }

        # Save trace file
        eval_dir = eval_dir or Path(
            f"src/evals/sokoban/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        eval_dir.mkdir(parents=True, exist_ok=True)
        traces_dir = eval_dir / "traces"
        traces_dir.mkdir(exist_ok=True)

        trace_file_path = traces_dir / f"{trajectory_id}.json"
        with open(trace_file_path, "w") as f:
            json.dump(trace_data, f, indent=2)

        # Create trajectory result
        result = SokobanTrajectoryResult(
            trajectory_id=trajectory_id,
            model_name=model_name,
            difficulty=difficulty,
            seed=seed,
            success=success,
            final_reward=final_reward,
            num_steps=num_steps,
            boxes_solved=int(final_public_state.boxes_on_target),
            total_boxes=int(final_public_state.num_boxes),
            trace_file_path=str(trace_file_path),
            metadata={
                "max_turns": max_turns,
                "actions_taken": len(actions_per_turn),
                "evaluation_timestamp": datetime.now().isoformat(),
            },
        )

        self.trajectory_results.append(result)
        return result

    async def run_evaluation(
        self,
        model_names: List[str],
        difficulties: List[str] = ["ultra-easy", "easy", "medium"],
        num_trajectories_per_condition: int = 3,
        max_turns: int = 20,
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation across models and difficulties."""

        print(f"🎯 Starting Sokoban evaluation")
        print(f"   Models: {model_names}")
        print(f"   Difficulties: {difficulties}")
        print(f"   Trajectories per condition: {num_trajectories_per_condition}")
        print(f"   Max turns per trajectory: {max_turns}")

        eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir = Path(f"src/evals/sokoban/run_{eval_timestamp}")
        eval_dir.mkdir(parents=True, exist_ok=True)

        all_results = []

        for model_name in model_names:
            for difficulty in difficulties:
                print(f"\n🤖 Running {model_name} on {difficulty} difficulty...")

                # Run trajectories for this condition
                condition_results = []
                for traj_idx in range(num_trajectories_per_condition):
                    seed = hash(f"{model_name}_{difficulty}_{traj_idx}") % 10000

                    print(
                        f"   Trajectory {traj_idx + 1}/{num_trajectories_per_condition} (seed={seed})"
                    )

                    result = await self.run_single_trajectory_with_trace(
                        model_name=model_name,
                        difficulty=difficulty,
                        seed=seed,
                        max_turns=max_turns,
                        eval_dir=eval_dir,
                    )

                    condition_results.append(result)
                    all_results.append(result)

                # Print condition summary
                successful = sum(1 for r in condition_results if r.success)
                avg_steps = sum(r.num_steps for r in condition_results) / len(condition_results)
                avg_boxes = sum(r.boxes_solved for r in condition_results) / len(condition_results)

                print(f"   ✅ {successful}/{len(condition_results)} successful")
                print(f"   📊 Avg steps: {avg_steps:.1f}, Avg boxes solved: {avg_boxes:.1f}")

        # Generate evaluation summary
        summary = {
            "evaluation_timestamp": eval_timestamp,
            "models_evaluated": model_names,
            "difficulties_evaluated": difficulties,
            "evaluation_metadata": {
                "num_trajectories": len(all_results),
                "max_turns": max_turns,
                "trajectories_per_condition": num_trajectories_per_condition,
            },
            "aggregate_results": [],
        }

        # Aggregate results by model and difficulty
        for model_name in model_names:
            for difficulty in difficulties:
                condition_results = [
                    r
                    for r in all_results
                    if r.model_name == model_name and r.difficulty == difficulty
                ]

                if condition_results:
                    success_rate = sum(1 for r in condition_results if r.success) / len(
                        condition_results
                    )
                    avg_reward = sum(r.final_reward for r in condition_results) / len(
                        condition_results
                    )
                    avg_steps = sum(r.num_steps for r in condition_results) / len(condition_results)
                    avg_boxes = sum(r.boxes_solved for r in condition_results) / len(
                        condition_results
                    )

                    summary["aggregate_results"].append(
                        {
                            "model_name": model_name,
                            "difficulty": difficulty,
                            "num_trajectories": len(condition_results),
                            "success_rate": success_rate,
                            "avg_reward": avg_reward,
                            "avg_steps": avg_steps,
                            "avg_boxes_solved": avg_boxes,
                        }
                    )

        # Save evaluation summary
        summary_file = eval_dir / "evaluation_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        # Print detailed aggregated results
        print("\n" + "=" * 80)
        print("🏆 FINAL SOKOBAN EVALUATION RESULTS")
        print("=" * 80)

        # Overall metrics
        all_successes = [r.success for r in all_results]
        all_rewards = [r.final_reward for r in all_results]
        all_steps = [r.num_steps for r in all_results]
        all_boxes_solved = [r.boxes_solved for r in all_results]

        print(f"📊 EVAL METRICS:")
        print(f"  Episodes: {len(all_results)}")
        print(f"  Individual Success: {[str(x) for x in all_successes]}")
        print(f"  Success Rate: {sum(all_successes) / len(all_successes):.3f}")

        print(f"\n🏆 REWARD METRICS:")
        print(f"  Individual Rewards: {[f'{x:.2f}' for x in all_rewards]}")
        print(f"  Mean Reward: {sum(all_rewards) / len(all_rewards):.2f}")

        print(f"\n📈 EFFICIENCY METRICS:")
        print(f"  Individual Steps: {[str(x) for x in all_steps]}")
        print(f"  Mean Steps: {sum(all_steps) / len(all_steps):.1f}")

        print(f"\n📦 BOX SOLVING METRICS:")
        print(f"  Individual Boxes Solved: {[str(x) for x in all_boxes_solved]}")
        print(f"  Mean Boxes Solved: {sum(all_boxes_solved) / len(all_boxes_solved):.1f}")

        # Results by difficulty
        print(f"\n🎯 RESULTS BY DIFFICULTY:")
        for model_name in model_names:
            print(f"  Model: {model_name}")
            for difficulty in difficulties:
                condition_results = [
                    r
                    for r in all_results
                    if r.model_name == model_name and r.difficulty == difficulty
                ]
                if condition_results:
                    success_rate = sum(1 for r in condition_results if r.success) / len(
                        condition_results
                    )
                    avg_reward = sum(r.final_reward for r in condition_results) / len(
                        condition_results
                    )
                    avg_steps = sum(r.num_steps for r in condition_results) / len(condition_results)
                    avg_boxes = sum(r.boxes_solved for r in condition_results) / len(
                        condition_results
                    )
                    print(
                        f"    {difficulty}: {success_rate:.1%} success, {avg_reward:.1f} reward, {avg_steps:.1f} steps, {avg_boxes:.1f} boxes"
                    )

        # Overall assessment
        overall_success_rate = sum(all_successes) / len(all_successes)
        overall_reward = sum(all_rewards) / len(all_rewards)

        print(f"\n🔍 ASSESSMENT:")
        if overall_success_rate >= 0.8:
            print("🎉 Excellent performance - mastering puzzle solving!")
        elif overall_success_rate >= 0.6:
            print("✅ Good performance - solving most puzzles!")
        elif overall_success_rate >= 0.4:
            print("⚠️  Moderate performance - learning puzzle mechanics")
        elif overall_success_rate >= 0.2:
            print("📈 Early progress - understanding basic moves")
        else:
            print("🧩 Learning phase - focus on understanding Sokoban rules")

        # Output markdown table row for README collation
        print(f"\n📋 MARKDOWN TABLE ROW:")
        print(
            "| Model            | Episodes | Success Rate | Mean Reward | Mean Steps | Mean Boxes | Assessment |"
        )
        print(
            "|------------------|----------|--------------|-------------|------------|------------|------------|"
        )

        if overall_success_rate >= 0.6:
            assessment = "Excellent"
        elif overall_success_rate >= 0.4:
            assessment = "Good"
        elif overall_success_rate >= 0.2:
            assessment = "Moderate"
        else:
            assessment = "Learning"

        main_model = model_names[0] if model_names else "Unknown"
        mean_steps = sum(all_steps) / len(all_steps)
        mean_boxes = sum(all_boxes_solved) / len(all_boxes_solved)

        print(
            f"| {main_model:<16} | {len(all_results):>8} | {overall_success_rate:>12.3f} | {overall_reward:>11.2f} | {mean_steps:>10.1f} | {mean_boxes:>10.1f} | {assessment:<10} |"
        )

        print(f"\n📁 Evaluation saved to: {eval_dir}")
        print(f"📁 Summary: {summary_file}")
        print(f"📁 Traces: {eval_dir / 'traces'}")

        return summary


# Convenience function for quick evaluations
async def run_sokoban_eval(
    model_names: List[str],
    difficulties: List[str] = ["ultra-easy", "easy", "medium"],
    num_trajectories: int = 3,
    max_turns: int = 20,
) -> Dict[str, Any]:
    """Quick evaluation runner with automatic report generation."""

    framework = SokobanEvalFramework()
    report = await framework.run_evaluation(
        model_names=model_names,
        difficulties=difficulties,
        num_trajectories_per_condition=num_trajectories,
        max_turns=max_turns,
    )

    return report


# --- Configuration Class ---
class SokobanConfig:
    """Configuration for Sokoban evaluation."""

    def __init__(self, config_path: Optional[str] = None):
        # Defaults
        self.model_name = "gpt-4.1-mini"
        self.num_instances = 3
        self.max_turns = 20
        self.difficulty_levels = ["ultra-easy", "easy", "medium"]
        self.service_base_url = "http://localhost:8901"
        self.service_timeout = 30.0
        self.seed = 42
        self.save_traces = True
        self.save_detailed_results = True

        if config_path and os.path.exists(config_path):
            try:
                import toml

                cfg = toml.load(config_path)

                e = cfg.get("eval", {})
                self.model_name = e.get("model_name", self.model_name)
                self.num_instances = e.get("episodes", self.num_instances)
                self.max_turns = e.get("max_steps", self.max_turns)
                diff = e.get("difficulty", None)
                if diff:
                    # allow comma-separated list or single value
                    if isinstance(diff, str) and "," in diff:
                        self.difficulty_levels = [d.strip() for d in diff.split(",")]
                    elif isinstance(diff, list):
                        self.difficulty_levels = diff
                    else:
                        self.difficulty_levels = [str(diff)]

                self.seed = e.get("seed", self.seed)

                s = cfg.get("service", {})
                self.service_base_url = s.get("base_url", self.service_base_url)
                self.service_timeout = s.get("timeout", self.service_timeout)

                o = cfg.get("output", {})
                self.save_traces = o.get("save_traces", self.save_traces)
                self.save_detailed_results = o.get(
                    "save_detailed_results", self.save_detailed_results
                )
            except Exception as exc:
                print(f"[WARNING] Failed to load Sokoban config: {exc}")


# --- Helper to run evaluation with config ---
async def _run_with_config(cfg: SokobanConfig):
    await run_sokoban_eval(
        model_names=[cfg.model_name],
        difficulties=cfg.difficulty_levels,
        num_trajectories=cfg.num_instances,
        max_turns=cfg.max_turns,
    )


# --- CLI Entry Point ---
if __name__ == "__main__":
    import argparse, asyncio

    parser = argparse.ArgumentParser(description="Run Sokoban evaluation with optional TOML config")
    parser.add_argument("--config", "-c", type=str, help="Path to TOML configuration file")
    parser.add_argument("--model", "-m", type=str, help="Model name override")
    parser.add_argument("--episodes", "-e", type=int, help="Episodes override")
    parser.add_argument("--max-turns", "-t", type=int, help="Max turns override")
    parser.add_argument(
        "--difficulty", "-d", type=str, help="Difficulty (single or comma-separated list)"
    )

    args = parser.parse_args()

    cfg = SokobanConfig(args.config)
    if args.model:
        cfg.model_name = args.model
    if args.episodes:
        cfg.num_instances = args.episodes
    if args.max_turns:
        cfg.max_turns = args.max_turns
    if args.difficulty:
        if "," in args.difficulty:
            cfg.difficulty_levels = [d.strip() for d in args.difficulty.split(",")]
        else:
            cfg.difficulty_levels = [args.difficulty]

    asyncio.run(_run_with_config(cfg))
