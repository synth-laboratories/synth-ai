"""
MiniGrid Evaluation Framework
Provides detailed metrics, trajectory analysis, and achievement statistics for MiniGrid environments.
"""

import asyncio
import json
import time
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
import uuid
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Synth-SDK tracing imports
from synth_sdk.tracing.context import trace_context
from synth_sdk.tracing.events.store import event_store

# MiniGrid-specific achievements based on task complexity
MINIGRID_ACHIEVEMENTS = {
    "basic": [
        "reach_goal",  # Complete any goal-reaching task
        "first_pickup",  # Pick up first object
        "first_door_open",  # Open first door
        "first_key_use",  # Use key to unlock door
        "navigate_empty_room",  # Complete Empty room tasks
        "complete_5_tasks",  # Complete 5 different tasks
    ],
    "intermediate": [
        "door_key_master",  # Complete DoorKey tasks consistently
        "multi_room_navigator",  # Complete MultiRoom tasks
        "unlock_pickup_combo",  # Complete UnlockPickup tasks
        "four_rooms_explorer",  # Complete FourRooms tasks
        "complete_20_tasks",  # Complete 20 different tasks
        "efficiency_expert",  # Complete task in <50% of max steps
    ],
    "advanced": [
        "lava_crosser",  # Complete LavaCrossing tasks
        "large_room_master",  # Complete 16x16+ room tasks
        "complex_multi_room",  # Complete N6+ MultiRoom tasks
        "speed_runner",  # Complete task in <25% of max steps
        "complete_50_tasks",  # Complete 50 different tasks
        "perfect_navigator",  # 90%+ success rate across all task types
    ],
}

ALL_ACHIEVEMENTS = [ach for category in MINIGRID_ACHIEVEMENTS.values() for ach in category]

TERMINATION_REASONS = [
    "timeout",
    "goal_reached",
    "agent_quit",
    "environment_error",
    "lava_death",
]

# Task difficulty mapping
MINIGRID_DIFFICULTY_MAPPING = {
    "easy": [
        "MiniGrid-Empty-5x5-v0",
        "MiniGrid-Empty-6x6-v0",
        "MiniGrid-Empty-8x8-v0",
        "MiniGrid-FourRooms-v0",
    ],
    "medium": [
        "MiniGrid-DoorKey-5x5-v0",
        "MiniGrid-DoorKey-6x6-v0",
        "MiniGrid-DoorKey-8x8-v0",
        "MiniGrid-Unlock-v0",
        "MiniGrid-UnlockPickup-v0",
    ],
    "hard": [
        "MiniGrid-DoorKey-16x16-v0",
        "MiniGrid-MultiRoom-N2-S4-v0",
        "MiniGrid-MultiRoom-N4-S5-v0",
        "MiniGrid-MultiRoom-N6-v0",
        "MiniGrid-LavaGapS5-v0",
        "MiniGrid-LavaGapS6-v0",
        "MiniGrid-LavaGapS7-v0",
        "MiniGrid-LavaCrossingS9N1-v0",
        "MiniGrid-LavaCrossingS9N2-v0",
        "MiniGrid-LavaCrossingS9N3-v0",
    ],
}


def minigrid_composite_score(
    achievements_unlocked: int,
    task_completion_rate: float,
    avg_efficiency: float,
    exploration_coverage: float,
) -> float:
    """
    MiniGrid composite scoring based on:
    - Achievement unlocking (30%)
    - Task completion rate (40%)
    - Movement efficiency (20%)
    - Exploration coverage (10%)
    """
    achievement_score = (achievements_unlocked / len(ALL_ACHIEVEMENTS)) * 30
    completion_score = task_completion_rate * 40
    efficiency_score = avg_efficiency * 20
    exploration_score = exploration_coverage * 10
    return achievement_score + completion_score + efficiency_score + exploration_score


def minigrid_navigation_score(
    success_rate: float, efficiency_ratio: float, wall_collision_rate: float
) -> float:
    """Navigation-specific score focusing on pathfinding ability."""
    # Penalize wall collisions
    collision_penalty = min(wall_collision_rate * 10, 20)  # Cap at 20% penalty
    base_score = (success_rate * 70) + (efficiency_ratio * 30)
    return max(0, base_score - collision_penalty)


@dataclass
class MiniGridTrajectoryResult:
    """Results from a single MiniGrid trajectory/episode."""

    trajectory_id: str
    model_name: str
    difficulty: str
    task_type: str  # "Empty", "DoorKey", "MultiRoom", etc.
    seed: int

    # Core metrics
    success: bool
    total_steps: int
    total_turns: int  # Number of agent decision turns
    total_reward: float

    # MiniGrid-specific fields
    grid_size: Tuple[int, int]  # (width, height)
    steps_to_goal: int  # Actual steps taken
    optimal_steps: Optional[int]  # Theoretical minimum steps
    efficiency_ratio: float  # optimal_steps / steps_to_goal (higher is better)
    objects_interacted: List[str]  # ["door", "key", "goal"]
    rooms_visited: int  # Number of different rooms visited

    # Navigation metrics
    backtrack_count: int  # Number of revisited positions
    wall_collision_count: int  # Number of invalid moves
    exploration_coverage: float  # % of accessible area explored

    # Achievement tracking
    achievements_unlocked: Set[str]
    achievement_turn_unlocked: Dict[str, int]  # achievement -> turn when unlocked

    # Multi-action metrics
    actions_per_turn: List[int]  # Number of actions per turn
    avg_actions_per_turn: float

    # Termination analysis
    termination_reason: (
        str  # "timeout", "goal_reached", "agent_quit", "environment_error", "lava_death"
    )
    final_position: Optional[Tuple[int, int]]
    final_direction: Optional[int]

    # Trajectory data for detailed analysis
    turn_by_turn_data: Optional[List[Dict[str, Any]]] = None


@dataclass
class MiniGridAggregateResults:
    """Aggregate results across multiple MiniGrid trajectories."""

    model_name: str
    difficulty: str
    num_trajectories: int

    # Success metrics
    success_rate: float
    avg_total_steps: float
    avg_total_turns: float
    avg_total_reward: float

    # MiniGrid-specific metrics
    task_completion_rates: Dict[str, float]  # task_type -> completion rate
    avg_efficiency_ratio: float
    avg_exploration_coverage: float
    avg_wall_collisions: float
    avg_backtrack_count: float

    # Achievement metrics
    unique_achievements_unlocked: Set[str]
    total_achievement_count: int
    avg_achievements_per_trajectory: float
    achievement_unlock_rates: Dict[str, float]  # achievement -> % of trajectories that unlocked it

    # MiniGrid-specific scores
    composite_score_avg: float  # Average composite score across trajectories
    composite_score_best: float  # Best single composite score
    navigation_score_avg: float  # Average navigation score
    navigation_score_best: float  # Best navigation score

    # Multi-action metrics
    avg_actions_per_turn_overall: float
    actions_per_turn_distribution: Dict[int, int]  # num_actions -> count

    # Termination analysis
    termination_breakdown: Dict[str, float]  # reason -> percentage
    avg_final_position: Optional[Tuple[float, float]]


def get_pure_success_scores(
    aggregate_results: List[MiniGridAggregateResults],
) -> Dict[str, float]:
    """
    Extract pure success scores - the percentage of tasks completed successfully.

    This is the simplest, most direct metric: did the agent reach the goal?

    Returns:
        Dict mapping "model_name (difficulty)" to success rate percentage (0-100)
    """
    success_scores = {}
    for agg in aggregate_results:
        key = f"{agg.model_name} ({agg.difficulty})"
        success_scores[key] = agg.success_rate * 100.0  # Convert to percentage

    return success_scores


def print_pure_success_summary(aggregate_results: List[MiniGridAggregateResults]):
    """Print a clean summary focused on pure success rates."""
    print("\n🎯 PURE SUCCESS RATES (Task Completion)")
    print("=" * 50)

    success_scores = get_pure_success_scores(aggregate_results)

    # Sort by success rate (highest first)
    sorted_results = sorted(aggregate_results, key=lambda x: x.success_rate, reverse=True)

    for agg in sorted_results:
        success_pct = agg.success_rate * 100.0
        print(f"{agg.model_name:25} ({agg.difficulty:6}): {success_pct:5.1f}%")

    print("=" * 50)
    print("✓ Success = Agent reached the goal")
    print("✗ Failure = Timeout, quit, or error")


def get_success_rate(report: Dict[str, Any], model_name: str, difficulty: str = None) -> float:
    """
    Quick helper to get the success rate for a specific model.

    Args:
        report: Evaluation report from run_minigrid_eval()
        model_name: Name of the model
        difficulty: Specific difficulty, or None for all difficulties

    Returns:
        Success rate as percentage (0-100)
    """
    if "pure_success_scores" not in report:
        return 0.0

    success_scores = report["pure_success_scores"]

    if difficulty:
        key = f"{model_name} ({difficulty})"
        return success_scores.get(key, 0.0)
    else:
        # Return average across all difficulties for this model
        matching_scores = [
            score for key, score in success_scores.items() if key.startswith(model_name)
        ]
        return sum(matching_scores) / len(matching_scores) if matching_scores else 0.0


class MiniGridEvalFramework:
    """Evaluation framework for MiniGrid environments."""

    def __init__(self):
        self.trajectory_results: List[MiniGridTrajectoryResult] = []

    async def run_single_trajectory(
        self,
        model_name: str,
        difficulty: str,
        task_type: str,
        seed: int,
        max_turns: int = 30,
        collect_detailed_data: bool = True,
    ) -> MiniGridTrajectoryResult:
        """Run a single trajectory and collect detailed metrics."""
        import sys
        import os

        # Add the agent_demos directory to path
        agent_demos_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, agent_demos_dir)
        # Add the minigrid directory to path
        minigrid_dir = os.path.dirname(agent_demos_dir)
        sys.path.insert(0, minigrid_dir)

        from test_synth_react import MiniGridReActAgent
        from environment import MiniGridEnvironment
        from taskset import MiniGridTaskInstance, MiniGridTaskInstanceMetadata
        from synth_ai.environments.tasks.core import Impetus, Intent
        from synth_ai.zyk import LM

        # Create task instance based on task type
        # Extract grid size from task name
        grid_size = (6, 6)  # Default
        if "5x5" in task_type:
            grid_size = (5, 5)
        elif "6x6" in task_type:
            grid_size = (6, 6)
        elif "8x8" in task_type:
            grid_size = (8, 8)
        elif "16x16" in task_type:
            grid_size = (16, 16)

        # Determine features
        has_key = "DoorKey" in task_type or "Unlock" in task_type
        has_door = "Door" in task_type or "Room" in task_type
        has_lava = "Lava" in task_type

        metadata = MiniGridTaskInstanceMetadata(
            env_name=task_type,
            grid_size=grid_size,
            difficulty=difficulty,
            has_key=has_key,
            has_door=has_door,
            has_lava=has_lava,
            num_objects=1 if has_key or has_door else 0,
            seed=seed,
        )

        instance = MiniGridTaskInstance(
            id=uuid.uuid4(),
            impetus=Impetus(instructions=f"Navigate and complete the {task_type} environment."),
            intent=Intent(rubric={}, gold_trajectories=None, gold_state_diff={}),
            metadata=metadata,
            is_reproducible=True,
            initial_engine_snapshot=None,
        )

        # Setup environment and agent
        env = MiniGridEnvironment(instance)

        llm = LM(model_name=model_name, formatting_model_name=model_name, temperature=0.0)
        agent = MiniGridReActAgent(llm, max_turns=max_turns, verbose=True)

        # Initialize tracking
        trajectory_id = str(uuid.uuid4())
        achievements_unlocked = set()
        achievement_turn_unlocked = {}
        actions_per_turn = []
        turn_by_turn_data = [] if collect_detailed_data else None

        # Navigation tracking
        positions_visited = set()
        wall_collisions = 0
        backtrack_count = 0
        objects_interacted = []

        # Wrap in trace context for synth-sdk tracing
        with trace_context(
            system_name="minigrid_evaluation",
            system_id="minigrid_evaluation",
            system_instance_id=trajectory_id,
        ):
            # Run episode
            obs_payload = await env.initialize()
            turn_count = 0
            termination_reason = "unknown"

            # Extract grid size from initial observation
            grid_size = self._extract_grid_size(obs_payload)

            # Create progress bar for this trajectory
            pbar = tqdm(
                total=max_turns,
                desc=f"{model_name} ({difficulty}) {task_type} Seed {seed}",
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
                        [a for a in achievements_unlocked if a in MINIGRID_ACHIEVEMENTS["basic"]]
                    )
                    medium_count = len(
                        [
                            a
                            for a in achievements_unlocked
                            if a in MINIGRID_ACHIEVEMENTS["intermediate"]
                        ]
                    )
                    hard_count = len(
                        [a for a in achievements_unlocked if a in MINIGRID_ACHIEVEMENTS["advanced"]]
                    )
                    total_count = len(achievements_unlocked)

                    achievement_display = f"{total_count}({easy_count}/{medium_count}/{hard_count})"

                    pbar.set_postfix(
                        {
                            "achievements": achievement_display,
                            "steps": obs_payload.get("public", {}).step_count
                            if hasattr(obs_payload.get("public", {}), "step_count")
                            else 0,
                        }
                    )

                    current_formatted_obs = obs_payload.get("formatted_obs", "")

                    # Track current position
                    current_position = self._extract_position(obs_payload)
                    if current_position:
                        if current_position in positions_visited:
                            backtrack_count += 1
                        positions_visited.add(current_position)

                    # Check for new achievements
                    new_achievements = self._check_achievements(
                        obs_payload, achievements_unlocked, turn_count, task_type
                    )
                    for ach in new_achievements:
                        achievements_unlocked.add(ach)
                        achievement_turn_unlocked[ach] = turn_count

                    # Agent decision
                    task_description = f"Complete the {task_type} task"
                    action_decision = await agent.decide(
                        current_formatted_obs, task_description, turn_count
                    )

                    if action_decision["name"] == "terminate":
                        termination_reason = "agent_quit"
                        break

                    # Convert to environment action format
                    env_action = self._convert_action_format(action_decision)
                    actions_per_turn.append(1)  # MiniGrid typically uses single actions

                    # Collect turn data
                    if collect_detailed_data:
                        turn_data = {
                            "turn": turn_count,
                            "action_planned": action_decision,
                            "achievements_at_start": list(achievements_unlocked),
                            "new_achievements_this_turn": list(new_achievements),
                            "position": current_position,
                            "steps_before_turn": obs_payload.get("public", {}).step_count
                            if hasattr(obs_payload.get("public", {}), "step_count")
                            else 0,
                        }
                        turn_by_turn_data.append(turn_data)

                    # Execute action
                    obs_payload = await env.step(env_action)

                    # Check for wall collision
                    if "blocked" in obs_payload.get("formatted_obs", "").lower():
                        wall_collisions += 1

                    # Check for object interaction
                    objects_interacted.extend(self._extract_object_interactions(obs_payload))

                    if "error" in obs_payload:
                        termination_reason = "environment_error"
                        break

                    # Fix the terminated/truncated check
                    private_data = obs_payload.get("private", {})
                    if (hasattr(private_data, "terminated") and private_data.terminated) or (
                        hasattr(private_data, "truncated") and private_data.truncated
                    ):
                        if "lava" in obs_payload.get("formatted_obs", "").lower():
                            termination_reason = "lava_death"
                        elif hasattr(private_data, "terminated") and private_data.terminated:
                            termination_reason = "goal_reached"
                        else:
                            termination_reason = "timeout"
                        break

                # Final metrics
                if termination_reason == "unknown":
                    termination_reason = "timeout"

                final_private = obs_payload.get("private", {})
                final_public = obs_payload.get("public", {})

                total_steps = getattr(final_public, "step_count", 0)
                total_reward = getattr(final_private, "total_reward", 0.0)

                # Calculate efficiency
                optimal_steps = self._estimate_optimal_steps(task_type, grid_size)
                efficiency_ratio = optimal_steps / max(total_steps, 1) if optimal_steps else 1.0

                # Calculate exploration coverage
                total_accessible_cells = self._estimate_accessible_cells(grid_size, task_type)
                exploration_coverage = len(positions_visited) / max(total_accessible_cells, 1)

                # Success determination
                success = termination_reason == "goal_reached"

                # Final position and direction
                final_position = self._extract_position(obs_payload)
                final_direction = self._extract_direction(obs_payload)

                avg_actions_per_turn = (
                    sum(actions_per_turn) / len(actions_per_turn) if actions_per_turn else 0.0
                )

                return MiniGridTrajectoryResult(
                    trajectory_id=trajectory_id,
                    model_name=model_name,
                    difficulty=difficulty,
                    task_type=task_type,
                    seed=seed,
                    success=success,
                    total_steps=total_steps,
                    total_turns=turn_count,
                    total_reward=total_reward,
                    grid_size=grid_size,
                    steps_to_goal=total_steps,
                    optimal_steps=optimal_steps,
                    efficiency_ratio=efficiency_ratio,
                    objects_interacted=list(set(objects_interacted)),
                    rooms_visited=1,  # TODO: Implement room detection
                    backtrack_count=backtrack_count,
                    wall_collision_count=wall_collisions,
                    exploration_coverage=exploration_coverage,
                    achievements_unlocked=achievements_unlocked,
                    achievement_turn_unlocked=achievement_turn_unlocked,
                    actions_per_turn=actions_per_turn,
                    avg_actions_per_turn=avg_actions_per_turn,
                    termination_reason=termination_reason,
                    final_position=final_position,
                    final_direction=final_direction,
                    turn_by_turn_data=turn_by_turn_data,
                )
            finally:
                pbar.close()

    async def run_evaluation(
        self,
        model_names: List[str],
        difficulties: List[str] = ["easy", "medium"],
        task_types: List[str] = None,
        num_trajectories_per_condition: int = 3,
        max_turns: int = 30,
        collect_detailed_data: bool = True,
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation across models and difficulties."""

        if task_types is None:
            task_types = ["MiniGrid-Empty-6x6-v0", "MiniGrid-DoorKey-5x5-v0"]

        print(f"🎯 Starting MiniGrid Evaluation")
        print(f"   Models: {model_names}")
        print(f"   Difficulties: {difficulties}")
        print(f"   Task Types: {task_types}")
        print(f"   Trajectories per condition: {num_trajectories_per_condition}")
        print(f"   Max turns per trajectory: {max_turns}")

        all_results = []

        for model_name in model_names:
            for difficulty in difficulties:
                for task_type in task_types:
                    print(f"\n🔄 Running {model_name} on {difficulty} difficulty, {task_type}...")

                    # Run trajectories for this condition
                    trajectory_tasks = []
                    for i in range(num_trajectories_per_condition):
                        seed = hash(f"{difficulty}_{task_type}_{i}") % 10000
                        trajectory_tasks.append(
                            self.run_single_trajectory(
                                model_name=model_name,
                                difficulty=difficulty,
                                task_type=task_type,
                                seed=seed,
                                max_turns=max_turns,
                                collect_detailed_data=collect_detailed_data,
                            )
                        )

                    condition_results = await asyncio.gather(*trajectory_tasks)
                    all_results.extend(condition_results)

        self.trajectory_results = all_results

        # Save synth-sdk traces after evaluation
        self._save_traces()

        return self._generate_comprehensive_report()

    def _extract_grid_size(self, obs_payload: Dict[str, Any]) -> Tuple[int, int]:
        """Extract grid size from observation."""
        # Try to extract from public state
        public = obs_payload.get("public", {})
        if hasattr(public, "grid_array"):
            grid = public.grid_array
            return (grid.shape[1], grid.shape[0])  # (width, height)

        # Default fallback
        return (6, 6)

    def _extract_position(self, obs_payload: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        """Extract agent position from observation."""
        public = obs_payload.get("public", {})
        if hasattr(public, "agent_pos"):
            return public.agent_pos
        return None

    def _extract_direction(self, obs_payload: Dict[str, Any]) -> Optional[int]:
        """Extract agent direction from observation."""
        public = obs_payload.get("public", {})
        if hasattr(public, "agent_dir"):
            return public.agent_dir
        return None

    def _extract_object_interactions(self, obs_payload: Dict[str, Any]) -> List[str]:
        """Extract object interactions from observation."""
        interactions = []
        formatted_obs = obs_payload.get("formatted_obs", "").lower()

        if "pickup" in formatted_obs:
            interactions.append("pickup")
        if "door" in formatted_obs:
            interactions.append("door")
        if "key" in formatted_obs:
            interactions.append("key")
        if "goal" in formatted_obs:
            interactions.append("goal")

        return interactions

    def _check_achievements(
        self,
        obs_payload: Dict[str, Any],
        current_achievements: Set[str],
        turn: int,
        task_type: str,
    ) -> Set[str]:
        """Check for new achievements based on current state."""
        new_achievements = set()
        formatted_obs = obs_payload.get("formatted_obs", "").lower()

        # Basic achievements
        if "reach_goal" not in current_achievements and "goal" in formatted_obs:
            new_achievements.add("reach_goal")

        if "first_pickup" not in current_achievements and "pickup" in formatted_obs:
            new_achievements.add("first_pickup")

        if (
            "first_door_open" not in current_achievements
            and "door" in formatted_obs
            and "open" in formatted_obs
        ):
            new_achievements.add("first_door_open")

        if "first_key_use" not in current_achievements and "key" in formatted_obs:
            new_achievements.add("first_key_use")

        # Task-specific achievements
        if "navigate_empty_room" not in current_achievements and "empty" in task_type.lower():
            private_data = obs_payload.get("private", {})
            if hasattr(private_data, "terminated") and private_data.terminated:
                new_achievements.add("navigate_empty_room")

        # Count-based achievements
        task_completions = len([a for a in current_achievements if "complete" not in a])
        if task_completions >= 5 and "complete_5_tasks" not in current_achievements:
            new_achievements.add("complete_5_tasks")

        return new_achievements

    def _convert_action_format(self, action_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Convert agent action decision to environment format."""
        if action_decision["name"] == "minigrid_act":
            action = action_decision["parameters"]["action"]
            return {"tool": "minigrid_act", "args": {"action": action}}

        # Fail fast if not minigrid_act
        raise ValueError(f"Expected minigrid_act tool, got {action_decision['name']}")

    def _estimate_optimal_steps(self, task_type: str, grid_size: Tuple[int, int]) -> Optional[int]:
        """Estimate optimal steps for a task type."""
        width, height = grid_size

        if "empty" in task_type.lower():
            # Manhattan distance estimate
            return width + height - 2
        elif "doorkey" in task_type.lower():
            # Need to find key, then door, then goal
            return (width + height) * 2
        else:
            # Conservative estimate
            return width * height // 2

    def _estimate_accessible_cells(self, grid_size: Tuple[int, int], task_type: str) -> int:
        """Estimate number of accessible cells."""
        width, height = grid_size
        total_cells = width * height

        # Account for walls (rough estimate)
        if "empty" in task_type.lower():
            return int(total_cells * 0.8)  # 80% accessible
        else:
            return int(total_cells * 0.6)  # 60% accessible with obstacles

    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report with all metrics and tables."""

        # Group results by model and difficulty
        grouped_results = defaultdict(lambda: defaultdict(list))
        for result in self.trajectory_results:
            grouped_results[result.model_name][result.difficulty].append(result)

        # Generate aggregate results
        aggregate_results = []
        for model_name, difficulties in grouped_results.items():
            for difficulty, trajectories in difficulties.items():
                agg = self._compute_aggregate_metrics(model_name, difficulty, trajectories)
                aggregate_results.append(agg)

        # Generate all tables and analyses
        report = {
            "evaluation_summary": self._generate_summary_table(aggregate_results),
            "achievement_percentage_table": self._generate_achievement_percentage_table(
                grouped_results
            ),
            "task_completion_breakdown": self._generate_task_completion_table(aggregate_results),
            "navigation_analysis": self._generate_navigation_analysis(aggregate_results),
            "trajectory_by_trajectory_breakdown": self._generate_trajectory_breakdown(),
            "raw_aggregate_results": [asdict(agg) for agg in aggregate_results],
            "raw_trajectory_results": [asdict(traj) for traj in self.trajectory_results],
        }

        return report

    def _compute_aggregate_metrics(
        self,
        model_name: str,
        difficulty: str,
        trajectories: List[MiniGridTrajectoryResult],
    ) -> MiniGridAggregateResults:
        """Compute aggregate metrics for a model-difficulty condition."""

        num_trajectories = len(trajectories)
        if num_trajectories == 0:
            return MiniGridAggregateResults(
                model_name=model_name,
                difficulty=difficulty,
                num_trajectories=0,
                success_rate=0.0,
                avg_total_steps=0.0,
                avg_total_turns=0.0,
                avg_total_reward=0.0,
                task_completion_rates={},
                avg_efficiency_ratio=0.0,
                avg_exploration_coverage=0.0,
                avg_wall_collisions=0.0,
                avg_backtrack_count=0.0,
                unique_achievements_unlocked=set(),
                total_achievement_count=0,
                avg_achievements_per_trajectory=0.0,
                achievement_unlock_rates={},
                composite_score_avg=0.0,
                composite_score_best=0.0,
                navigation_score_avg=0.0,
                navigation_score_best=0.0,
                avg_actions_per_turn_overall=0.0,
                actions_per_turn_distribution={},
                termination_breakdown={},
                avg_final_position=None,
            )

        # Success metrics
        success_rate = sum(1 for t in trajectories if t.success) / num_trajectories
        avg_total_steps = sum(t.total_steps for t in trajectories) / num_trajectories
        avg_total_turns = sum(t.total_turns for t in trajectories) / num_trajectories
        avg_total_reward = sum(t.total_reward for t in trajectories) / num_trajectories

        # MiniGrid-specific metrics
        task_completion_rates = {}
        task_counts = defaultdict(int)
        task_successes = defaultdict(int)

        for traj in trajectories:
            task_counts[traj.task_type] += 1
            if traj.success:
                task_successes[traj.task_type] += 1

        for task_type in task_counts:
            task_completion_rates[task_type] = task_successes[task_type] / task_counts[task_type]

        avg_efficiency_ratio = sum(t.efficiency_ratio for t in trajectories) / num_trajectories
        avg_exploration_coverage = (
            sum(t.exploration_coverage for t in trajectories) / num_trajectories
        )
        avg_wall_collisions = sum(t.wall_collision_count for t in trajectories) / num_trajectories
        avg_backtrack_count = sum(t.backtrack_count for t in trajectories) / num_trajectories

        # Achievement analysis
        all_achievements = set()
        total_achievement_count = 0
        achievement_counts = defaultdict(int)

        for traj in trajectories:
            all_achievements.update(traj.achievements_unlocked)
            total_achievement_count += len(traj.achievements_unlocked)
            for ach in traj.achievements_unlocked:
                achievement_counts[ach] += 1

        achievement_unlock_rates = {
            ach: count / num_trajectories for ach, count in achievement_counts.items()
        }
        avg_achievements_per_trajectory = total_achievement_count / num_trajectories

        # Compute MiniGrid-specific scores
        composite_scores = [
            minigrid_composite_score(
                len(traj.achievements_unlocked),
                1.0 if traj.success else 0.0,
                traj.efficiency_ratio,
                traj.exploration_coverage,
            )
            for traj in trajectories
        ]
        composite_score_avg = (
            sum(composite_scores) / len(composite_scores) if composite_scores else 0.0
        )
        composite_score_best = max(composite_scores) if composite_scores else 0.0

        # Navigation scores
        navigation_scores = [
            minigrid_navigation_score(
                1.0 if traj.success else 0.0,
                traj.efficiency_ratio,
                traj.wall_collision_count / max(traj.total_turns, 1),
            )
            for traj in trajectories
        ]
        navigation_score_avg = (
            sum(navigation_scores) / len(navigation_scores) if navigation_scores else 0.0
        )
        navigation_score_best = max(navigation_scores) if navigation_scores else 0.0

        # Multi-action analysis
        all_actions_per_turn = []
        actions_per_turn_dist = defaultdict(int)
        for traj in trajectories:
            all_actions_per_turn.extend(traj.actions_per_turn)
            for count in traj.actions_per_turn:
                actions_per_turn_dist[count] += 1

        avg_actions_per_turn_overall = (
            sum(all_actions_per_turn) / len(all_actions_per_turn) if all_actions_per_turn else 0.0
        )

        # Termination analysis
        termination_counts = defaultdict(int)
        for traj in trajectories:
            termination_counts[traj.termination_reason] += 1
        termination_breakdown = {
            reason: count / num_trajectories for reason, count in termination_counts.items()
        }

        # Average final position
        final_positions = [t.final_position for t in trajectories if t.final_position is not None]
        avg_final_position = None
        if final_positions:
            avg_x = sum(pos[0] for pos in final_positions) / len(final_positions)
            avg_y = sum(pos[1] for pos in final_positions) / len(final_positions)
            avg_final_position = (avg_x, avg_y)

        return MiniGridAggregateResults(
            model_name=model_name,
            difficulty=difficulty,
            num_trajectories=num_trajectories,
            success_rate=success_rate,
            avg_total_steps=avg_total_steps,
            avg_total_turns=avg_total_turns,
            avg_total_reward=avg_total_reward,
            task_completion_rates=task_completion_rates,
            avg_efficiency_ratio=avg_efficiency_ratio,
            avg_exploration_coverage=avg_exploration_coverage,
            avg_wall_collisions=avg_wall_collisions,
            avg_backtrack_count=avg_backtrack_count,
            unique_achievements_unlocked=all_achievements,
            total_achievement_count=total_achievement_count,
            avg_achievements_per_trajectory=avg_achievements_per_trajectory,
            achievement_unlock_rates=achievement_unlock_rates,
            avg_actions_per_turn_overall=avg_actions_per_turn_overall,
            actions_per_turn_distribution=dict(actions_per_turn_dist),
            termination_breakdown=termination_breakdown,
            avg_final_position=avg_final_position,
            composite_score_avg=composite_score_avg,
            composite_score_best=composite_score_best,
            navigation_score_avg=navigation_score_avg,
            navigation_score_best=navigation_score_best,
        )

    def _generate_summary_table(
        self, aggregate_results: List[MiniGridAggregateResults]
    ) -> pd.DataFrame:
        """Generate main summary table with key metrics."""

        data = []
        for agg in aggregate_results:
            data.append(
                {
                    "Model": agg.model_name,
                    "Difficulty": agg.difficulty,
                    "✓ Success Rate": f"{agg.success_rate:.1%}",  # Made more prominent with checkmark
                    "Composite Score": f"{agg.composite_score_avg:.1f}",
                    "Navigation Score": f"{agg.navigation_score_avg:.1f}",
                    "Avg Steps": f"{agg.avg_total_steps:.1f}",
                    "Avg Turns": f"{agg.avg_total_turns:.1f}",
                    "Efficiency": f"{agg.avg_efficiency_ratio:.2f}",
                    "Exploration": f"{agg.avg_exploration_coverage:.1%}",
                    "Wall Collisions": f"{agg.avg_wall_collisions:.1f}",
                    "Achievements": len(agg.unique_achievements_unlocked),
                    "Avg Actions/Turn": f"{agg.avg_actions_per_turn_overall:.1f}",
                }
            )

        return pd.DataFrame(data)

    def _generate_achievement_percentage_table(
        self, grouped_results: Dict[str, Dict[str, List[MiniGridTrajectoryResult]]]
    ) -> pd.DataFrame:
        """Generate table showing percentage of trajectories achieving each achievement."""

        data = []

        for model_name, difficulties in grouped_results.items():
            for difficulty, trajectories in difficulties.items():
                if not trajectories:
                    continue

                num_trajectories = len(trajectories)
                row = {"Model": model_name, "Difficulty": difficulty}

                # Count achievements
                achievement_counts = defaultdict(int)
                for traj in trajectories:
                    for ach in traj.achievements_unlocked:
                        achievement_counts[ach] += 1

                # Add percentage for each achievement
                for achievement in ALL_ACHIEVEMENTS:
                    count = achievement_counts[achievement]
                    percentage = count / num_trajectories if num_trajectories > 0 else 0.0
                    row[achievement] = f"{percentage:.1%}"

                data.append(row)

        df = pd.DataFrame(data)

        # Reorder columns: Model, Difficulty, then achievements by category
        base_cols = ["Model", "Difficulty"]
        achievement_cols = []
        for category in ["basic", "intermediate", "advanced"]:
            for ach in MINIGRID_ACHIEVEMENTS[category]:
                if ach in df.columns:
                    achievement_cols.append(ach)

        return df[base_cols + achievement_cols]

    def _generate_task_completion_table(
        self, aggregate_results: List[MiniGridAggregateResults]
    ) -> pd.DataFrame:
        """Generate table showing completion rates by task type."""

        data = []
        for agg in aggregate_results:
            row = {
                "Model": agg.model_name,
                "Difficulty": agg.difficulty,
            }

            for task_type, completion_rate in agg.task_completion_rates.items():
                row[task_type] = f"{completion_rate:.1%}"

            data.append(row)

        return pd.DataFrame(data)

    def _generate_navigation_analysis(
        self, aggregate_results: List[MiniGridAggregateResults]
    ) -> pd.DataFrame:
        """Generate analysis of navigation metrics."""

        data = []
        for agg in aggregate_results:
            data.append(
                {
                    "Model": agg.model_name,
                    "Difficulty": agg.difficulty,
                    "Efficiency Ratio": f"{agg.avg_efficiency_ratio:.3f}",
                    "Exploration Coverage": f"{agg.avg_exploration_coverage:.1%}",
                    "Wall Collisions": f"{agg.avg_wall_collisions:.1f}",
                    "Backtrack Count": f"{agg.avg_backtrack_count:.1f}",
                    "Navigation Score": f"{agg.navigation_score_avg:.1f}",
                    "Final Position": f"({agg.avg_final_position[0]:.1f}, {agg.avg_final_position[1]:.1f})"
                    if agg.avg_final_position
                    else "N/A",
                }
            )

        return pd.DataFrame(data)

    def _generate_trajectory_breakdown(self) -> pd.DataFrame:
        """Generate detailed trajectory-by-trajectory breakdown."""

        data = []
        for traj in self.trajectory_results:
            # Achievement category breakdown
            easy_achievements = len(
                [a for a in traj.achievements_unlocked if a in MINIGRID_ACHIEVEMENTS["basic"]]
            )
            medium_achievements = len(
                [
                    a
                    for a in traj.achievements_unlocked
                    if a in MINIGRID_ACHIEVEMENTS["intermediate"]
                ]
            )
            hard_achievements = len(
                [a for a in traj.achievements_unlocked if a in MINIGRID_ACHIEVEMENTS["advanced"]]
            )

            data.append(
                {
                    "Trajectory ID": traj.trajectory_id[:8],  # Short ID
                    "Model": traj.model_name,
                    "Difficulty": traj.difficulty,
                    "Task Type": traj.task_type,
                    "Seed": traj.seed,
                    "Success": "✓" if traj.success else "✗",
                    "Steps": traj.total_steps,
                    "Turns": traj.total_turns,
                    "Efficiency": f"{traj.efficiency_ratio:.3f}",
                    "Exploration": f"{traj.exploration_coverage:.1%}",
                    "Wall Collisions": traj.wall_collision_count,
                    "Total Achievements": len(traj.achievements_unlocked),
                    "Basic": easy_achievements,
                    "Intermediate": medium_achievements,
                    "Advanced": hard_achievements,
                    "Termination": traj.termination_reason,
                    "Final Position": f"({traj.final_position[0]}, {traj.final_position[1]})"
                    if traj.final_position
                    else "N/A",
                    "Achievements": ", ".join(sorted(traj.achievements_unlocked))
                    if traj.achievements_unlocked
                    else "None",
                }
            )

        return pd.DataFrame(data)

    def print_report(self, report: Dict[str, Any]):
        """Print a formatted evaluation report."""

        print("\n" + "=" * 80)
        print("🎯 MINIGRID EVALUATION REPORT")
        print("=" * 80)

        # Pure success summary first - the most important metric
        aggregate_results = [
            MiniGridAggregateResults(**agg) for agg in report["raw_aggregate_results"]
        ]
        print_pure_success_summary(aggregate_results)

        # Summary table
        print("\n📊 EVALUATION SUMMARY")
        summary_df = report["evaluation_summary"]
        print(summary_df.to_string(index=False, max_colwidth=12))

        # Achievement breakdown
        print("\n🏆 ACHIEVEMENT UNLOCK RATES")
        achievement_df = report["achievement_percentage_table"]
        if not achievement_df.empty:
            print("Format: percentage of trajectories that unlocked each achievement")

            # Print by category for better readability
            for category in ["basic", "intermediate", "advanced"]:
                category_cols = ["Model", "Difficulty"] + [
                    col for col in achievement_df.columns if col in MINIGRID_ACHIEVEMENTS[category]
                ]
                if len(category_cols) > 2:
                    category_data = achievement_df[category_cols]
                    if not category_data.empty:
                        print(f"\n{category.upper()} ACHIEVEMENTS:")
                        print(category_data.to_string(index=False))

        # Task completion breakdown
        print("\n📋 TASK COMPLETION RATES")
        task_df = report["task_completion_breakdown"]
        print(task_df.to_string(index=False))

        # Navigation analysis
        print("\n🧭 NAVIGATION ANALYSIS")
        nav_df = report["navigation_analysis"]
        print(nav_df.to_string(index=False))

        # Trajectory breakdown (summary stats only for space)
        traj_df = report["trajectory_by_trajectory_breakdown"]
        print(f"\n📋 TRAJECTORY BREAKDOWN ({len(traj_df)} total trajectories)")
        print("Sample trajectories:")
        sample_cols = [
            "Model",
            "Difficulty",
            "Task Type",
            "Success",
            "Steps",
            "Total Achievements",
            "Termination",
        ]
        sample_df = traj_df[sample_cols].head(5)
        print(sample_df.to_string(index=False, max_colwidth=12))
        if len(traj_df) > 5:
            print(f"... and {len(traj_df) - 5} more trajectories")

        print("\n" + "=" * 80)

    def _save_traces(self):
        """Save synth-sdk traces to disk."""
        # Get all traces from event store
        traces = event_store.get_system_traces()

        if not traces:
            print("⚠️ No traces found in event store")
            return

        # Create traces directory
        traces_dir = Path("src/evals/minigrid") / f"run_{int(time.time())}" / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)

        print(f"💾 Saving {len(traces)} traces to {traces_dir}")

        for trace in traces:
            trace_file = traces_dir / f"minigrid_trace_{trace.system_instance_id}.json"
            with open(trace_file, "w") as f:
                json.dump(trace.to_dict(), f, indent=2)

        print(f"✅ Traces saved. To view: ./run_viewer.sh {traces_dir.parent}")
        return traces_dir


# Convenience function for quick evaluations
async def run_minigrid_eval(
    model_names: List[str],
    difficulties: List[str] = ["easy", "medium"],
    task_types: List[str] = None,
    num_trajectories: int = 3,
    max_turns: int = 30,
) -> Dict[str, Any]:
    """Quick evaluation runner with automatic report generation."""

    framework = MiniGridEvalFramework()
    report = await framework.run_evaluation(
        model_names=model_names,
        difficulties=difficulties,
        task_types=task_types,
        num_trajectories_per_condition=num_trajectories,
        max_turns=max_turns,
    )

    framework.print_report(report)

    # Add pure success scores to the report for easy access
    aggregate_results = [MiniGridAggregateResults(**agg) for agg in report["raw_aggregate_results"]]
    report["pure_success_scores"] = get_pure_success_scores(aggregate_results)

    return report
