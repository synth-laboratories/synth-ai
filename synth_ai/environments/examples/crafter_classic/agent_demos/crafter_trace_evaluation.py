"""
Full Crafter Evaluation with Traces, Rewards, and Viewer
Extends eval_framework.py with SystemTrace capture, viewer, and comprehensive logging.
"""

import asyncio
import base64
import io
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# Import base evaluation framework
from src.synth_env.examples.crafter_classic.agent_demos.crafter_evaluation_framework import (
    ACHIEVEMENT_CATEGORIES,
    ALL_ACHIEVEMENTS,
    TERMINATION_REASONS,
    AggregateResults,
    CrafterEvalFramework,
    TrajectoryResult,
    balrog_score,
    crafter_score,
)

# Import synth-sdk trace structures
from synth_sdk.tracing.abstractions import (
    AgentComputeStep,
    ArbitraryInputs,
    ArbitraryOutputs,
    Dataset,
    EnvironmentComputeStep,
    Event,
    EventPartitionElement,
    MessageInputs,
    MessageOutputs,
    RewardSignal,
    SystemTrace,
    TrainingQuestion,
)
from tqdm import tqdm

# Action names mapping for Crafter
ACTION_NAMES = {
    -1: "initial_state",
    0: "noop",
    1: "move_left",
    2: "move_right",
    3: "move_up",
    4: "move_down",
    5: "do",
    6: "sleep",
    7: "place_stone",
    8: "place_table",
    9: "place_furnace",
    10: "place_plant",
    11: "make_wood_pickaxe",
    12: "make_stone_pickaxe",
    13: "make_iron_pickaxe",
    14: "make_wood_sword",
    15: "make_stone_sword",
    16: "make_iron_sword",
}


class FullCrafterEvalFramework(CrafterEvalFramework):
    """Extended evaluation framework with trace capture and visualization."""

    def __init__(self, capture_images: bool = True, output_dir: Optional[str] = None):
        super().__init__()
        self.capture_images = capture_images

        # Use standardized eval directory structure
        if output_dir is None:
            # Create timestamp-based directory under src/evals/
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path("src/evals") / "crafter" / f"run_{timestamp}"
        else:
            self.output_dir = Path(output_dir)

        self.traces_dir = self.output_dir / "traces"
        self.viewer_dir = self.output_dir / "viewer"

        # Create directories
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.viewer_dir.mkdir(parents=True, exist_ok=True)

        # Store traces and datasets
        self.system_traces: Dict[str, SystemTrace] = {}
        self.datasets: Dict[str, Dataset] = {}

    def _encode_image_to_base64(self, rgb_array: np.ndarray) -> str:
        """Convert RGB numpy array to base64 PNG string."""
        image = Image.fromarray(rgb_array.astype("uint8"), "RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    async def run_single_trajectory_with_trace(
        self,
        model_name: str,
        difficulty: str,
        seed: int,
        max_turns: int = 30,
        collect_detailed_data: bool = True,
    ) -> TrajectoryResult:
        """Run a single trajectory with comprehensive trace capture."""
        from src.synth_env.examples.crafter_classic.agent_demos.crafter_react_agent import (
            CrafterHistoryObservationCallable,
            CrafterMove,
            ReActAgent,
        )
        from src.synth_env.examples.crafter_classic.environment import (
            CrafterClassicEnvironment,
        )
        from src.synth_env.examples.crafter_classic.taskset import (
            CrafterTaskInstance,
            CrafterTaskInstanceMetadata,
        )
        from src.synth_env.tasks.core import Impetus, Intent
        from synth_ai.lm.core.main import LM

        # Create task instance
        metadata = CrafterTaskInstanceMetadata(
            difficulty=difficulty,
            seed=seed,
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

        # Setup environment and agent
        start_time = time.perf_counter()
        hist_cb = CrafterHistoryObservationCallable(max_history=1)
        env = CrafterClassicEnvironment(instance, custom_step_obs=hist_cb)

        llm = LM(model_name=model_name, formatting_model_name=model_name, temperature=0.0)
        agent = ReActAgent(llm, max_turns=max_turns)

        # Initialize tracking
        trajectory_id = str(uuid.uuid4())
        achievements_unlocked = set()
        achievement_turn_unlocked = {}
        actions_per_turn = []
        turn_by_turn_data = [] if collect_detailed_data else None

        # Initialize SystemTrace
        system_trace = SystemTrace(
            system_name="crafter_evaluation",
            system_id=f"crafter_{model_name}_{difficulty}",
            system_instance_id=trajectory_id,
            partition=[],
            metadata={
                "model_name": model_name,
                "difficulty": difficulty,
                "seed": seed,
                "max_turns": max_turns,
            },
            instance_metadata={
                "start_time": datetime.now().isoformat(),
                "capture_images": self.capture_images,
            },
        )

        # Create TrainingQuestion for this trajectory
        training_question = TrainingQuestion(
            id=trajectory_id,
            intent=f"Survive and unlock achievements in a {difficulty} Crafter environment (seed={seed})",
            criteria="Maximize the number of achievements unlocked and survive as long as possible",
        )

        # Run episode
        obs_payload = await env.initialize()
        turn_count = 0
        termination_reason = "unknown"
        partition_index = 0

        # Create progress bar for this trajectory
        pbar = tqdm(
            total=max_turns,
            desc=f"{model_name} ({difficulty}) Seed {seed}",
            unit="turn",
            leave=False,
            ncols=100,
        )

        try:
            while turn_count < max_turns:
                turn_count += 1
                pbar.update(1)

                # Track achievements
                easy_count = len(
                    [a for a in achievements_unlocked if a in ACHIEVEMENT_CATEGORIES["easy"]]
                )
                medium_count = len(
                    [a for a in achievements_unlocked if a in ACHIEVEMENT_CATEGORIES["medium"]]
                )
                hard_count = len(
                    [a for a in achievements_unlocked if a in ACHIEVEMENT_CATEGORIES["hard"]]
                )
                total_count = len(achievements_unlocked)

                achievement_display = f"{total_count}({easy_count}/{medium_count}/{hard_count})"

                pbar.set_postfix(
                    {
                        "achievements": achievement_display,
                        "steps": obs_payload.get("public", {}).num_steps_taken
                        if hasattr(obs_payload.get("public", {}), "num_steps_taken")
                        else 0,
                    }
                )

                current_formatted_obs = obs_payload["formatted_obs"]

                # Track achievements at start of turn
                current_achievements = set()
                if "public" in obs_payload and hasattr(
                    obs_payload["public"], "achievements_status"
                ):
                    current_achievements = {
                        ach
                        for ach, status in obs_payload["public"].achievements_status.items()
                        if status
                    }

                # Check for new achievements
                new_achievements = current_achievements - achievements_unlocked
                for ach in new_achievements:
                    achievements_unlocked.add(ach)
                    achievement_turn_unlocked[ach] = turn_count
                    agent.current_achievements.add(ach)

                # Create EventPartitionElement for this turn
                event_partition = EventPartitionElement(partition_index=partition_index, events=[])

                # Capture initial state image before actions
                initial_image = None
                if self.capture_images and turn_count == 1:
                    try:
                        if hasattr(env, "engine") and hasattr(env.engine, "env"):
                            crafter_env = env.engine.env
                            if hasattr(crafter_env, "_render_mode"):
                                crafter_env._render_mode = "rgb_array"
                            initial_rgb = crafter_env.render()
                            if initial_rgb is not None:
                                initial_image = self._encode_image_to_base64(initial_rgb)
                                print(f"✓ Initial state captured before actions")
                    except Exception as e:
                        print(f"Warning: Failed to capture initial image: {e}")

                # Agent decision phase
                agent_compute_began = datetime.now()

                # Create proper system and user prompts
                system_prompt = "You are playing Crafter. Your goal is to survive and unlock as many achievements as possible."

                # Get agent decision with proper message structure
                agent_decision = await agent.decide(current_formatted_obs, obs_payload)
                agent_compute_ended = datetime.now()

                if agent_decision == [-1]:  # Agent terminated
                    termination_reason = "agent_quit"
                    break

                action_sequence = agent_decision
                actions_per_turn.append(len(action_sequence))

                # Create proper tool calls for the actions
                tool_calls = [
                    {
                        "id": f"crafter_action_{turn_count}",
                        "type": "function",
                        "function": {
                            "name": "crafter_interact",
                            "arguments": json.dumps(
                                {
                                    "actions": action_sequence,
                                    "reasoning": f"Executing {len(action_sequence)} actions: {[ACTION_NAMES.get(act, f'action_{act}') for act in action_sequence]}",
                                }
                            ),
                        },
                    }
                ]

                tool_results = [
                    {
                        "tool_call_id": f"crafter_action_{turn_count}",
                        "content": f"Planned actions: {[ACTION_NAMES.get(act, f'action_{act}') for act in action_sequence]}",
                    }
                ]

                # Create input messages
                input_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": current_formatted_obs},
                ]

                # Create output messages with tool calls
                output_messages = [
                    {
                        "role": "assistant",
                        "content": f"I need to execute {len(action_sequence)} actions to progress in the game: {[ACTION_NAMES.get(act, f'action_{act}') for act in action_sequence]}",
                        "tool_calls": tool_calls,
                    }
                ]

                # Add tool results
                for tool_result in tool_results:
                    output_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_result["tool_call_id"],
                            "content": tool_result["content"],
                        }
                    )

                # Create AgentComputeStep with proper message structure
                agent_compute_step = AgentComputeStep(
                    event_order=0,
                    compute_began=agent_compute_began,
                    compute_ended=agent_compute_ended,
                    compute_input=[MessageInputs(messages=input_messages)],
                    compute_output=[MessageOutputs(messages=output_messages)],
                    model_name=model_name,
                    model_params={"temperature": 0.0},
                    should_learn=True,
                )

                # Collect turn data
                if collect_detailed_data:
                    turn_data = {
                        "turn": turn_count,
                        "actions_planned": len(action_sequence),
                        "achievements_at_start": list(current_achievements),
                        "new_achievements_this_turn": list(new_achievements),
                        "steps_before_turn": obs_payload.get("public", {}).num_steps_taken
                        if hasattr(obs_payload.get("public", {}), "num_steps_taken")
                        else 0,
                    }
                    turn_by_turn_data.append(turn_data)

                # Execute actions and collect environment steps
                environment_compute_steps = []

                # Add initial state as first "step" if we have it
                if initial_image and turn_count == 1:
                    initial_step = EnvironmentComputeStep(
                        event_order=0,
                        compute_began=agent_compute_began,
                        compute_ended=agent_compute_began,
                        compute_input=[ArbitraryInputs(inputs={"action": "initial_state"})],
                        compute_output=[
                            ArbitraryOutputs(
                                outputs={
                                    "action_index": -1,  # Special index for initial state
                                    "action_order": -1,
                                    "image_base64": initial_image,
                                    "error": None,
                                    "terminated": False,
                                    "truncated": False,
                                    "reward": 0.0,
                                    "total_reward": 0.0,
                                    "num_steps": 0,
                                }
                            )
                        ],
                    )
                    environment_compute_steps.append(initial_step)

                for i, act_idx in enumerate(action_sequence):
                    env_compute_began = datetime.now()

                    # Execute action
                    obs_payload = await env.step([[CrafterMove(act_idx)]])
                    env_compute_ended = datetime.now()

                    # Capture image after step if enabled
                    post_step_image = None
                    if self.capture_images:
                        try:
                            # Access the underlying crafter environment through the engine
                            if hasattr(env, "engine") and hasattr(env.engine, "env"):
                                # Force render mode to 'rgb_array' if needed
                                crafter_env = env.engine.env
                                if hasattr(crafter_env, "_render_mode"):
                                    crafter_env._render_mode = "rgb_array"

                                rgb_array = crafter_env.render()
                                if rgb_array is not None:
                                    post_step_image = self._encode_image_to_base64(rgb_array)
                                    # Debug: check if images are different
                                    # if (
                                    #     turn_count == 1 and i < 3
                                    # ):  # Debug first turn, first 3 actions
                                    #     print(
                                    #         f"✓ Action {i} ({act_idx}): Image captured, first pixel: {rgb_array[0, 0]}, shape: {rgb_array.shape}"
                                    #     )
                                else:
                                    print(f"Warning: render() returned None")
                            else:
                                print(f"Warning: Cannot access env.engine.env for rendering")
                        except Exception as e:
                            print(f"Warning: Failed to capture image: {e}")

                    # Create EnvironmentComputeStep
                    env_outputs = {
                        "action_index": act_idx,
                        "action_order": i,
                        "error": obs_payload.get("error", None),
                        "terminated": obs_payload.get("private", {}).terminated
                        if hasattr(obs_payload.get("private", {}), "terminated")
                        else False,
                        "truncated": obs_payload.get("private", {}).truncated
                        if hasattr(obs_payload.get("private", {}), "truncated")
                        else False,
                        "reward": obs_payload.get("private", {}).reward
                        if hasattr(obs_payload.get("private", {}), "reward")
                        else 0.0,
                        "total_reward": obs_payload.get("private", {}).total_reward_episode
                        if hasattr(obs_payload.get("private", {}), "total_reward_episode")
                        else 0.0,
                        "num_steps": obs_payload.get("public", {}).num_steps_taken
                        if hasattr(obs_payload.get("public", {}), "num_steps_taken")
                        else 0,
                    }

                    # Add image if captured
                    if post_step_image:
                        env_outputs["image_base64"] = post_step_image

                    # Add player stats
                    if hasattr(obs_payload.get("private", {}), "player_internal_stats"):
                        stats = obs_payload["private"].player_internal_stats
                        env_outputs["player_stats"] = {
                            "health": stats.get("health"),
                            "food": stats.get("food"),
                            "drink": stats.get("drink"),
                        }

                    # Adjust event order if we have initial state
                    event_order = i + 2 if (initial_image and turn_count == 1) else i + 1
                    env_compute_step = EnvironmentComputeStep(
                        event_order=event_order,
                        compute_began=env_compute_began,
                        compute_ended=env_compute_ended,
                        compute_input=[ArbitraryInputs(inputs={"action": act_idx})],
                        compute_output=[ArbitraryOutputs(outputs=env_outputs)],
                    )
                    environment_compute_steps.append(env_compute_step)

                    if "error" in obs_payload:
                        termination_reason = "environment_error"
                        break

                    if obs_payload["private"].terminated or obs_payload["private"].truncated:
                        termination_reason = (
                            "timeout" if obs_payload["private"].truncated else "death"
                        )
                        break

                # Create Event for this turn
                event = Event(
                    system_instance_id=trajectory_id,
                    event_type="turn",
                    opened=agent_compute_began,
                    closed=environment_compute_steps[-1].compute_ended
                    if environment_compute_steps
                    else agent_compute_ended,
                    partition_index=partition_index,
                    agent_compute_step=agent_compute_step,
                    environment_compute_steps=environment_compute_steps,
                    event_metadata={
                        "turn_number": turn_count,
                        "new_achievements": list(new_achievements),
                        "total_achievements": len(achievements_unlocked),
                    },
                )

                event_partition.events.append(event)
                system_trace.partition.append(event_partition)
                partition_index += 1

                if termination_reason in ["environment_error", "timeout", "death"]:
                    break

            # Final metrics
            if termination_reason == "unknown":
                termination_reason = "timeout"

            final_private = obs_payload.get("private")
            final_public = obs_payload.get("public")

            total_steps = (
                final_public.num_steps_taken if hasattr(final_public, "num_steps_taken") else 0
            )
            total_reward = (
                final_private.total_reward_episode
                if hasattr(final_private, "total_reward_episode")
                else 0.0
            )

            # Health/survival stats
            final_health = None
            final_food = None
            final_drink = None
            if hasattr(final_private, "player_internal_stats"):
                stats = final_private.player_internal_stats
                final_health = stats.get("health")
                final_food = stats.get("food")
                final_drink = stats.get("drink")

            # Success determination
            success = len(achievements_unlocked) > 0 or (
                hasattr(final_private, "terminated") and final_private.terminated
            )

            avg_actions_per_turn = (
                sum(actions_per_turn) / len(actions_per_turn) if actions_per_turn else 0.0
            )

            # Create RewardSignal
            hafner_score_value = crafter_score(
                [(achievement_turn_unlocked.get(ach, 0) > 0) * 100.0 for ach in ALL_ACHIEVEMENTS]
            )

            reward_signal = RewardSignal(
                question_id=trajectory_id,
                system_instance_id=trajectory_id,
                reward=hafner_score_value,
                annotation=f"Termination: {termination_reason}, Achievements: {len(achievements_unlocked)}/22",
            )

            # Create Dataset
            dataset = Dataset(questions=[training_question], reward_signals=[reward_signal])

            # Store trace and dataset
            self.system_traces[trajectory_id] = system_trace
            self.datasets[trajectory_id] = dataset

            # Save to disk
            self._save_trace_to_disk(trajectory_id, system_trace, dataset)

            # Total duration
            total_duration_sec = time.perf_counter() - start_time

            # Create trajectory result
            result = TrajectoryResult(
                trajectory_id=trajectory_id,
                model_name=model_name,
                difficulty=difficulty,
                seed=seed,
                success=success,
                total_steps=total_steps,
                total_turns=turn_count,
                total_reward=total_reward,
                total_duration_sec=total_duration_sec,
                achievements_unlocked=achievements_unlocked,
                achievement_turn_unlocked=achievement_turn_unlocked,
                actions_per_turn=actions_per_turn,
                avg_actions_per_turn=avg_actions_per_turn,
                termination_reason=termination_reason,
                final_health=final_health,
                final_food=final_food,
                final_drink=final_drink,
                turn_by_turn_data=turn_by_turn_data,
            )

            return result

        finally:
            pbar.close()

    def _save_trace_to_disk(self, trajectory_id: str, trace: SystemTrace, dataset: Dataset):
        """Save trace and dataset to JSON file."""
        trace_file = self.traces_dir / f"{trajectory_id}.json"

        trace_data = {
            "trace": trace.to_dict(),
            "dataset": dataset.to_dict(),
            "metadata": {
                "saved_at": datetime.now().isoformat(),
                "trajectory_id": trajectory_id,
            },
        }

        with open(trace_file, "w") as f:
            json.dump(trace_data, f, indent=2)

    def _save_evaluation_summary(self, report: Dict[str, Any]):
        """Save evaluation summary and metadata."""
        summary_file = self.output_dir / "evaluation_summary.json"

        # Extract key metrics for summary
        summary_data = {
            "evaluation_metadata": {
                "timestamp": datetime.now().isoformat(),
                "output_directory": str(self.output_dir),
                "traces_directory": str(self.traces_dir),
                "viewer_directory": str(self.viewer_dir),
                "num_trajectories": len(self.trajectory_results),
            },
            "evaluation_summary": report.get("evaluation_summary").to_dict()
            if hasattr(report.get("evaluation_summary"), "to_dict")
            else None,
            "traces_info": report.get("traces"),
            "models_evaluated": list(set(t.model_name for t in self.trajectory_results)),
            "difficulties_evaluated": list(set(t.difficulty for t in self.trajectory_results)),
        }

        with open(summary_file, "w") as f:
            json.dump(summary_data, f, indent=2)

        # Also save the full report as CSV tables
        if "evaluation_summary" in report and hasattr(report["evaluation_summary"], "to_csv"):
            report["evaluation_summary"].to_csv(self.output_dir / "summary_table.csv", index=False)

        if "trajectory_by_trajectory_breakdown" in report and hasattr(
            report["trajectory_by_trajectory_breakdown"], "to_csv"
        ):
            report["trajectory_by_trajectory_breakdown"].to_csv(
                self.output_dir / "trajectories.csv", index=False
            )

    async def run_evaluation(
        self,
        model_names: List[str],
        difficulties: List[str] = ["easy", "hard"],
        num_trajectories_per_condition: int = 3,
        max_turns: int = 30,
        collect_detailed_data: bool = True,
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation with trace capture."""

        print(f"🎯 Starting Full Enchilada Crafter Evaluation")
        print(f"   Models: {model_names}")
        print(f"   Difficulties: {difficulties}")
        print(f"   Trajectories per condition: {num_trajectories_per_condition}")
        print(f"   Max turns per trajectory: {max_turns}")
        print(f"   Output directory: {self.output_dir}")

        all_results = []

        for model_name in model_names:
            for difficulty in difficulties:
                print(f"\n🔄 Running {model_name} on {difficulty} difficulty...")

                # Run trajectories for this condition
                trajectory_tasks = []
                for i in range(num_trajectories_per_condition):
                    seed = 1000 + i if difficulty == "easy" else 2000 + i
                    trajectory_tasks.append(
                        self.run_single_trajectory_with_trace(
                            model_name=model_name,
                            difficulty=difficulty,
                            seed=seed,
                            max_turns=max_turns,
                            collect_detailed_data=collect_detailed_data,
                        )
                    )

                condition_results = await asyncio.gather(*trajectory_tasks)
                all_results.extend(condition_results)

        self.trajectory_results = all_results

        # Generate report
        report = self._generate_comprehensive_report()

        # Create viewer files
        self._create_viewer_files()

        # Add trace info to report
        report["traces"] = {
            "count": len(self.system_traces),
            "directory": str(self.traces_dir),
            "viewer_url": f"http://localhost:8999",
        }

        # Save evaluation summary
        self._save_evaluation_summary(report)

        return report

    def _create_viewer_files(self):
        """Create the viewer HTML/JS/CSS files."""

        # Create index.html
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crafter Evaluation Viewer</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div id="app">
        <div class="header">
            <h1>🎮 Crafter Evaluation Viewer</h1>
            <div class="trace-selector">
                <label for="trace-select">Select Trace:</label>
                <select id="trace-select"></select>
                <button id="refresh-btn">🔄 Refresh</button>
            </div>
        </div>
        
        <div class="main-container">
            <div class="sidebar">
                <h2>Timeline</h2>
                <div id="timeline" class="timeline"></div>
                
                <div class="trace-info">
                    <h3>Trace Info</h3>
                    <div id="trace-metadata"></div>
                </div>
            </div>
            
            <div class="content">
                <div class="question-reward">
                    <h2>Training Question</h2>
                    <div id="question-display" class="info-box"></div>
                    
                    <h2>Reward Signal</h2>
                    <div id="reward-display" class="info-box"></div>
                </div>
                
                <div class="turn-details">
                    <h2>Turn Details</h2>
                    <div id="turn-content">
                        <p class="placeholder">Select a turn from the timeline</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="viewer.js"></script>
</body>
</html>"""

        # Create style.css
        css_content = """* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background-color: #f5f5f5;
    color: #333;
}

.header {
    background-color: #2c3e50;
    color: white;
    padding: 1rem 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.header h1 {
    font-size: 1.5rem;
}

.trace-selector {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.trace-selector select {
    padding: 0.5rem;
    border-radius: 4px;
    border: 1px solid #34495e;
    background-color: white;
    min-width: 200px;
}

.trace-selector button {
    padding: 0.5rem 1rem;
    background-color: #3498db;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    transition: background-color 0.2s;
}

.trace-selector button:hover {
    background-color: #2980b9;
}

.main-container {
    display: flex;
    height: calc(100vh - 60px);
}

.sidebar {
    width: 300px;
    background-color: white;
    border-right: 1px solid #ddd;
    overflow-y: auto;
    padding: 1rem;
}

.timeline {
    margin-bottom: 2rem;
}

.timeline-item {
    padding: 0.75rem;
    margin-bottom: 0.5rem;
    background-color: #f8f9fa;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.2s;
    border: 2px solid transparent;
}

.timeline-item:hover {
    background-color: #e9ecef;
}

.timeline-item.active {
    background-color: #3498db;
    color: white;
    border-color: #2980b9;
}

.timeline-item .turn-number {
    font-weight: bold;
    margin-bottom: 0.25rem;
}

.timeline-item .turn-stats {
    font-size: 0.85rem;
    opacity: 0.8;
}

.trace-info {
    padding: 1rem;
    background-color: #f8f9fa;
    border-radius: 4px;
}

.trace-info h3 {
    margin-bottom: 0.5rem;
    color: #2c3e50;
}

.content {
    flex: 1;
    padding: 2rem;
    overflow-y: auto;
}

.question-reward {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
    margin-bottom: 2rem;
}

.info-box {
    padding: 1rem;
    background-color: white;
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.turn-details {
    background-color: white;
    border-radius: 4px;
    padding: 2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.turn-details h2 {
    margin-bottom: 1rem;
    color: #2c3e50;
}

.placeholder {
    color: #999;
    text-align: center;
    padding: 3rem;
}

.agent-section, .environment-section {
    margin-bottom: 2rem;
}

.agent-section h3, .environment-section h3 {
    color: #2c3e50;
    margin-bottom: 1rem;
}

.message-box {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 4px;
    margin-bottom: 1rem;
}

.message-box .role {
    font-weight: bold;
    color: #2c3e50;
    margin-bottom: 0.5rem;
}

.actions-box {
    background-color: #e3f2fd;
    padding: 1rem;
    border-radius: 4px;
    margin-bottom: 1rem;
    font-family: monospace;
}

.env-step {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 4px;
    margin-bottom: 1rem;
    border-left: 4px solid #3498db;
}

.env-step h4 {
    color: #2c3e50;
    margin-bottom: 0.5rem;
}

.env-image {
    width: 256px;
    height: 256px;
    border-radius: 4px;
    margin-top: 1rem;
    image-rendering: pixelated;
    border: 2px solid #ddd;
}

.stats-grid {
    display: inline-flex;
    gap: 1rem;
    margin-top: 0.5rem;
    margin-bottom: 0.5rem;
}

.stat-item {
    background-color: white;
    padding: 0.25rem 0.5rem;
    border-radius: 3px;
    text-align: center;
    border: 1px solid #e0e0e0;
    font-size: 0.75rem;
}

.stat-label {
    font-size: 0.7rem;
    color: #666;
    margin-right: 0.25rem;
}

.stat-value {
    font-size: 0.8rem;
    font-weight: bold;
    color: #2c3e50;
    display: inline;
}

.metadata-item {
    margin-bottom: 0.5rem;
}

.metadata-label {
    font-weight: bold;
    color: #666;
}

.achievement-badge {
    display: inline-block;
    background-color: #27ae60;
    color: white;
    padding: 0.25rem 0.5rem;
    border-radius: 3px;
    font-size: 0.85rem;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
}

.images-row {
    display: flex;
    gap: 1rem;
    overflow-x: auto;
    padding: 1rem 0;
}

.image-container {
    text-align: center;
    flex-shrink: 0;
}

.image-caption {
    margin-top: 0.5rem;
    font-size: 0.85rem;
    color: #666;
    font-family: monospace;
}

@media (max-width: 768px) {
    .main-container {
        flex-direction: column;
    }
    
    .sidebar {
        width: 100%;
        height: 200px;
        border-right: none;
        border-bottom: 1px solid #ddd;
    }
    
    .question-reward {
        grid-template-columns: 1fr;
    }
}"""

        # Create viewer.js
        js_content = """let currentTrace = null;
let currentTurnIndex = null;

// Action mapping
const ACTION_NAMES = {
    -1: 'initial_state',
    0: 'noop',
    1: 'move_left',
    2: 'move_right',
    3: 'move_up',
    4: 'move_down',
    5: 'do',
    6: 'sleep',
    7: 'place_stone',
    8: 'place_table',
    9: 'place_furnace',
    10: 'place_plant',
    11: 'make_wood_pickaxe',
    12: 'make_stone_pickaxe',
    13: 'make_iron_pickaxe',
    14: 'make_wood_sword',
    15: 'make_stone_sword',
    16: 'make_iron_sword'
};

// Load available traces
async function loadTraceList() {
    try {
        const response = await fetch('/api/traces');
        const traces = await response.json();
        
        const select = document.getElementById('trace-select');
        select.innerHTML = '';
        
        traces.forEach(trace => {
            const option = document.createElement('option');
            option.value = trace.id;
            option.textContent = `${trace.model_name} - ${trace.difficulty} - ${trace.id.substring(0, 8)}`;
            select.appendChild(option);
        });
        
        if (traces.length > 0) {
            loadTrace(traces[0].id);
        }
    } catch (error) {
        console.error('Failed to load traces:', error);
    }
}

// Load specific trace
async function loadTrace(traceId) {
    try {
        const response = await fetch(`/api/trace/${traceId}`);
        const data = await response.json();
        
        currentTrace = data;
        currentTurnIndex = null;
        
        displayTraceInfo();
        displayTimeline();
        displayQuestionAndReward();
        clearTurnDetails();
    } catch (error) {
        console.error('Failed to load trace:', error);
    }
}

// Display trace metadata
function displayTraceInfo() {
    const metadataDiv = document.getElementById('trace-metadata');
    const metadata = currentTrace.trace.metadata;
    
    metadataDiv.innerHTML = `
        <div class="metadata-item">
            <span class="metadata-label">Model:</span> ${metadata.model_name}
        </div>
        <div class="metadata-item">
            <span class="metadata-label">Difficulty:</span> ${metadata.difficulty}
        </div>
        <div class="metadata-item">
            <span class="metadata-label">Seed:</span> ${metadata.seed}
        </div>
        <div class="metadata-item">
            <span class="metadata-label">Max Turns:</span> ${metadata.max_turns}
        </div>
    `;
}

// Display timeline
function displayTimeline() {
    const timeline = document.getElementById('timeline');
    timeline.innerHTML = '';
    
    currentTrace.trace.partition.forEach((partition, index) => {
        const event = partition.events[0];
        const metadata = event.event_metadata;
        
        const item = document.createElement('div');
        item.className = 'timeline-item';
        item.innerHTML = `
            <div class="turn-number">Turn ${metadata.turn_number}</div>
            <div class="turn-stats">
                Actions: ${event.environment_compute_steps.length} | 
                Achievements: ${metadata.total_achievements}
                ${metadata.new_achievements.length > 0 ? ' (+' + metadata.new_achievements.length + ')' : ''}
            </div>
        `;
        
        item.addEventListener('click', () => selectTurn(index));
        timeline.appendChild(item);
    });
}

// Display question and reward
function displayQuestionAndReward() {
    const question = currentTrace.dataset.questions[0];
    const reward = currentTrace.dataset.reward_signals[0];
    
    document.getElementById('question-display').innerHTML = `
        <p><strong>Intent:</strong> ${question.intent}</p>
        <p><strong>Criteria:</strong> ${question.criteria}</p>
    `;
    
    document.getElementById('reward-display').innerHTML = `
        <p><strong>Hafner Score:</strong> ${reward.reward.toFixed(2)}%</p>
        <p><strong>Annotation:</strong> ${reward.annotation}</p>
    `;
}

// Select turn
function selectTurn(index) {
    currentTurnIndex = index;
    
    // Update timeline selection
    document.querySelectorAll('.timeline-item').forEach((item, i) => {
        item.classList.toggle('active', i === index);
    });
    
    // Display turn details
    displayTurnDetails();
}

// Display turn details - SIMPLIFIED VERSION
function displayTurnDetails() {
    if (currentTurnIndex === null) return;
    
    const partition = currentTrace.trace.partition[currentTurnIndex];
    const event = partition.events[0];
    const agentStep = event.agent_compute_step;
    const envSteps = event.environment_compute_steps;
    
    let html = '';
    
    // Display actions planned
    if (agentStep.compute_output[0] && agentStep.compute_output[0].outputs) {
        const outputs = agentStep.compute_output[0].outputs;
        const actionNames = outputs.actions.map(idx => `${ACTION_NAMES[idx] || 'unknown'}`);
        html += `
            <div class="actions-box" style="margin-bottom: 1.5rem;">
                <strong>Turn ${event.event_metadata.turn_number} Actions:</strong> ${actionNames.join(' → ')}
            </div>
        `;
    }
    
    // Display all images in a row
    html += '<div class="images-row">';
    envSteps.forEach((step, i) => {
        const outputs = step.compute_output[0].outputs;
        const actionName = ACTION_NAMES[outputs.action_index] || 'unknown';
        
        if (outputs.image_base64) {
            // For initial state, show "0. initial state", otherwise show action number
            const stepNumber = outputs.action_index === -1 ? 0 : i;
            html += `
                <div class="image-container">
                    <img src="data:image/png;base64,${outputs.image_base64}" class="env-image" alt="Game state">
                    <div class="image-caption">${stepNumber}. ${actionName}</div>
                </div>
            `;
        }
    });
    html += '</div>';
    
    // New achievements
    if (event.event_metadata.new_achievements.length > 0) {
        html += '<div class="achievements-section" style="margin-top: 1rem;">';
        html += '<strong>New achievements: </strong>';
        event.event_metadata.new_achievements.forEach(ach => {
            html += `<span class="achievement-badge">${ach}</span>`;
        });
        html += '</div>';
    }
    
    document.getElementById('turn-content').innerHTML = html;
}

// Clear turn details
function clearTurnDetails() {
    document.getElementById('turn-content').innerHTML = '<p class="placeholder">Select a turn from the timeline</p>';
}

// Event listeners
document.getElementById('trace-select').addEventListener('change', (e) => {
    if (e.target.value) {
        loadTrace(e.target.value);
    }
});

document.getElementById('refresh-btn').addEventListener('click', () => {
    loadTraceList();
});

// Initial load
loadTraceList();"""

        # Save files
        with open(self.viewer_dir / "index.html", "w") as f:
            f.write(html_content)

        with open(self.viewer_dir / "style.css", "w") as f:
            f.write(css_content)

        with open(self.viewer_dir / "viewer.js", "w") as f:
            f.write(js_content)


# Global variable to store current eval directory
_current_eval_dir = None


def set_current_eval_dir(eval_dir: Path):
    """Set the current evaluation directory for the viewer."""
    global _current_eval_dir
    _current_eval_dir = eval_dir


# FastAPI app for viewer
app = FastAPI()


@app.get("/api/traces")
async def get_traces():
    """Get list of available traces."""
    global _current_eval_dir
    if _current_eval_dir is None:
        return []

    traces_dir = _current_eval_dir / "traces"
    if not traces_dir.exists():
        return []

    traces = []
    for trace_file in traces_dir.glob("*.json"):
        try:
            with open(trace_file, "r") as f:
                data = json.load(f)
                trace_meta = data["trace"]["metadata"]
                traces.append(
                    {
                        "id": trace_file.stem,
                        "model_name": trace_meta["model_name"],
                        "difficulty": trace_meta["difficulty"],
                        "seed": trace_meta["seed"],
                    }
                )
        except Exception as e:
            print(f"Error loading trace {trace_file}: {e}")

    return sorted(traces, key=lambda x: x["id"])


@app.get("/api/trace/{trace_id}")
async def get_trace(trace_id: str):
    """Get specific trace data."""
    global _current_eval_dir
    if _current_eval_dir is None:
        raise HTTPException(status_code=404, detail="No evaluation directory set")

    trace_file = _current_eval_dir / "traces" / f"{trace_id}.json"
    if not trace_file.exists():
        raise HTTPException(status_code=404, detail="Trace not found")

    with open(trace_file, "r") as f:
        return json.load(f)


@app.get("/api/eval_info")
async def get_eval_info():
    """Get evaluation metadata."""
    global _current_eval_dir
    if _current_eval_dir is None:
        return {"error": "No evaluation directory set"}

    summary_file = _current_eval_dir / "evaluation_summary.json"
    if summary_file.exists():
        with open(summary_file, "r") as f:
            return json.load(f)
    return {"error": "No evaluation summary found"}


# Convenience function for running evaluation
async def run_full_crafter_eval(
    model_names: List[str],
    difficulties: List[str] = ["easy", "hard"],
    num_trajectories: int = 3,
    max_turns: int = 30,
    capture_images: bool = True,
    launch_viewer: bool = True,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run full Crafter evaluation with traces and viewer."""

    framework = FullCrafterEvalFramework(capture_images=capture_images, output_dir=output_dir)
    report = await framework.run_evaluation(
        model_names=model_names,
        difficulties=difficulties,
        num_trajectories_per_condition=num_trajectories,
        max_turns=max_turns,
    )

    framework.print_report(report)

    if launch_viewer:
        print(f"\n📁 Evaluation saved to: {framework.output_dir}")
        print("🌐 Launching viewer at http://localhost:8999")
        print("   Press Ctrl+C to stop the viewer")

        # Set the current eval directory for the viewer
        set_current_eval_dir(framework.output_dir)

        # Mount static files from the viewer directory
        app.mount(
            "/",
            StaticFiles(directory=str(framework.viewer_dir), html=True),
            name="viewer",
        )

        # Run viewer
        config = uvicorn.Config(app, host="0.0.0.0", port=8999, log_level="error")
        server = uvicorn.Server(config)
        await server.serve()

    return report


if __name__ == "__main__":
    # Example usage
    async def main():
        await run_full_crafter_eval(
            model_names=["gpt-4.1-mini"],
            difficulties=["easy"],
            num_trajectories=10,
            max_turns=50,
            capture_images=True,
            launch_viewer=True,
        )

    asyncio.run(main())
