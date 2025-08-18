"""
NetHack Evaluation Framework
============================
Provides detailed metrics, trajectory analysis, and achievement statistics for NetHack.
Mirrors the Crafter evaluation structure but adapted for NetHack specifics.
"""

import asyncio
import json
import math
import os
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from src.synth_env.examples.nethack.achievements import NetHackAchievements
from src.synth_env.examples.nethack.agent_demos.test_synth_react import (
    NetHackReActAgent,
)
from src.synth_env.examples.nethack.engine import NetHackObservationCallable

# NetHack specific imports
from src.synth_env.examples.nethack.environment import NetHackEnvironment
from src.synth_env.examples.nethack.taskset import (
    NetHackTaskInstance,
    NetHackTaskInstanceMetadata,
)
from src.synth_env.tasks.core import Impetus, Intent
from synth_ai.zyk import LM
from tqdm import tqdm

# Load achievements mapping for BALROG scoring
_achievements_path = os.path.join(os.path.dirname(__file__), "..", "helpers", "achievements.json")
with open(_achievements_path, "r") as f:
    BALROG_ACHIEVEMENTS = json.load(f)["3.4.3"]

# Achievement categories based on difficulty/complexity
ACHIEVEMENT_CATEGORIES = {
    "basic": [
        "first_kill",
        "first_spell_cast",
        "first_prayer",
        "survived_100_turns",
        "reached_dlvl_2",
        "reached_dlvl_5",
        "killed_10_monsters",
    ],
    "intermediate": [
        "reached_dlvl_10",
        "reached_dlvl_20",
        "killed_50_monsters",
        "killed_100_monsters",
        "collected_1000_gold",
        "reached_level_5",
        "reached_level_10",
        "reached_minetown",
    ],
    "advanced": [
        "reached_dlvl_30",
        "reached_castle",
        "got_quest",
        "completed_quest",
        "reached_level_20",
        "collected_10000_gold",
        "found_artifact",
        "reached_mines_end",
    ],
}

# Get all achievements from NetHackAchievements
_sample_achievements = NetHackAchievements()
ALL_ACHIEVEMENTS = list(_sample_achievements.get_unlocked_achievements().keys())

TERMINATION_REASONS = ["timeout", "death", "agent_quit", "environment_error"]

# SOTA scores (NetHack doesn't have published Hafner scores, only BALROG)
BALROG_SOTA_SCORES = {
    "balrog_leaderboard": {
        # TODO: Add real BALROG leaderboard scores when available
        "Claude 3.5 Sonnet": 25.0,  # Placeholder
        "GPT-4o": 20.0,  # Placeholder
        "GPT-4o-mini": 15.0,  # Placeholder
        "Gemini 1.5 Flash": 12.0,  # Placeholder
    }
}

# Model name mapping for SOTA percentage calculations
MODEL_NAME_TO_SOTA = {
    "claude-3-5-sonnet-latest": "Claude 3.5 Sonnet",
    "gpt-4o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o-mini",
    "gemini-1.5-flash": "Gemini 1.5 Flash",
    "gemini-1.5-flash-latest": "Gemini 1.5 Flash",
}


def hafner_score(success_rates_percent: List[float]) -> float:
    """Compute the Hafner adjusted score (log-mean) for NetHack."""
    if not success_rates_percent:
        return 0.0
    N = len(success_rates_percent)
    g = sum(math.log(1 + s) for s in success_rates_percent) / N
    return math.exp(g) - 1


def balrog_score_simple(percent: float) -> float:
    """BALROG score is already a percentage (0-100)."""
    return percent


@dataclass
class TrajectoryResult:
    """Results from a single NetHack trajectory/episode."""

    trajectory_id: str
    model_name: str
    difficulty: str
    seed: int

    # Core metrics
    success: bool
    total_steps: int
    total_turns: int
    total_reward: float

    # Achievement tracking
    achievements_unlocked: Set[str]
    achievement_turn_unlocked: Dict[str, int]

    # Multi-action metrics (if applicable)
    actions_per_turn: List[int]
    avg_actions_per_turn: float

    # Termination analysis
    termination_reason: str
    final_depth: Optional[int]
    final_level: Optional[int]
    final_gold: Optional[int]

    # BALROG scoring
    balrog_percent: float

    # Trajectory data for detailed analysis
    turn_by_turn_data: Optional[List[Dict[str, Any]]] = None


@dataclass
class AggregateResults:
    """Aggregate results across multiple NetHack trajectories."""

    model_name: str
    difficulty: str
    num_trajectories: int

    # Success metrics
    success_rate: float
    avg_total_steps: float
    avg_total_turns: float
    avg_total_reward: float

    # Achievement metrics
    unique_achievements_unlocked: Set[str]
    total_achievement_count: int
    avg_achievements_per_trajectory: float
    achievement_unlock_rates: Dict[str, float]
    hafner_score: float
    balrog_score_avg: float
    balrog_score_best: float

    # Multi-action metrics
    avg_actions_per_turn_overall: float
    actions_per_turn_distribution: Dict[int, int]

    # Termination analysis
    termination_breakdown: Dict[str, float]
    avg_final_depth: Optional[float]
    avg_final_level: Optional[float]
    avg_final_gold: Optional[float]


class NetHackEvalFramework:
    """Standardized evaluation framework for NetHack environments."""

    def __init__(self):
        self.trajectory_results: List[TrajectoryResult] = []

    async def run_single_trajectory(
        self,
        model_name: str,
        difficulty: str,
        seed: int,
        max_turns: int = 200,
        collect_detailed_data: bool = True,
    ) -> TrajectoryResult:
        """Run a single NetHack trajectory and collect detailed metrics."""

        # Create task instance
        metadata = NetHackTaskInstanceMetadata(
            character_role="knight",  # Default role
            starting_level=1,
            target_depth=5 if difficulty == "easy" else 10,
            time_limit=max_turns * 10,  # Generous time limit
            difficulty=difficulty,
            special_objectives=[
                "Survive for as long as possible",
                "Collect gold",
                "Kill monsters",
            ],
            seed=seed,
        )
        instance = NetHackTaskInstance(
            id=uuid.uuid4(),
            impetus=Impetus(
                instructions=f"Explore the NetHack dungeon on {difficulty} difficulty. Survive as long as possible, kill monsters, collect items, and descend to deeper levels."
            ),
            intent=Intent(rubric={}, gold_trajectories=None, gold_state_diff={}),
            metadata=metadata,
            is_reproducible=True,
            initial_engine_snapshot=None,
        )

        # Setup environment and agent
        obs_callback = NetHackObservationCallable()
        env = NetHackEnvironment(instance, custom_step_obs=obs_callback)

        llm = LM(model_name=model_name, formatting_model_name=model_name, temperature=0.0)
        agent = NetHackReActAgent(llm, max_turns=max_turns)

        # Set system prompt for agent
        task_instructions = instance.impetus.instructions
        agent.system_prompt = agent._create_system_prompt(task_instructions)

        # Initialize tracking
        trajectory_id = str(uuid.uuid4())
        achievements = NetHackAchievements()
        achievements_unlocked = set()
        achievement_turn_unlocked = {}
        actions_per_turn = []
        turn_by_turn_data = [] if collect_detailed_data else None

        # Progress tracking for BALROG score
        class BalrogProgress:
            def __init__(self):
                self.percent = 0.0
                self.end_reason = None

            def update(self, depth: int, level: int, done: bool = False, end_reason: str = ""):
                # Simple progress based on depth and level
                depth_score = min(depth * 2, 50)  # Max 50 from depth
                level_score = min(level * 3, 50)  # Max 50 from level
                self.percent = max(depth_score, level_score)
                if done:
                    self.end_reason = end_reason

        balrog_progress = BalrogProgress()

        # Run episode
        obs_payload = await env.initialize()
        turn_count = 0
        termination_reason = "unknown"

        # Create progress bar
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

                # Extract stats from observation for progress tracking
                if "formatted_obs" in obs_payload:
                    current_formatted_obs = obs_payload["formatted_obs"]
                elif "message" in obs_payload:
                    # Format the observation for the agent
                    current_formatted_obs = f"""
=== NetHack Observation ===
Message: {obs_payload.get("message", "")}
Map:
{obs_payload.get("ascii_map", "")}

Stats: {obs_payload.get("player_stats", {})}
Inventory: {obs_payload.get("inventory", [])}
In Menu: {obs_payload.get("in_menu", False)}
"""
                else:
                    # Fallback to string representation
                    current_formatted_obs = str(obs_payload)

                # Update achievements (simplified - would need real obs parsing)
                prev_achievements = achievements_unlocked.copy()

                # Extract game state for BALROG scoring
                try:
                    # Parse the actual game state from obs
                    player_stats = obs_payload.get("player_stats", {})
                    current_depth = player_stats.get("depth", 1)
                    current_level = player_stats.get("experience_level", 1)
                    balrog_progress.update(current_depth, current_level)
                except:
                    current_depth = 1
                    current_level = 1
                    balrog_progress.update(current_depth, current_level)

                # Update progress bar
                easy_count = len(
                    [a for a in achievements_unlocked if a in ACHIEVEMENT_CATEGORIES["basic"]]
                )
                inter_count = len(
                    [
                        a
                        for a in achievements_unlocked
                        if a in ACHIEVEMENT_CATEGORIES["intermediate"]
                    ]
                )
                adv_count = len(
                    [a for a in achievements_unlocked if a in ACHIEVEMENT_CATEGORIES["advanced"]]
                )
                total_count = len(achievements_unlocked)
                achievement_display = f"{total_count}({easy_count}/{inter_count}/{adv_count})"

                pbar.set_postfix(
                    {
                        "achievements": achievement_display,
                        "balrog": f"{balrog_progress.percent:.1f}%",
                    }
                )

                # Agent decision
                decision = await agent.decide(current_formatted_obs)

                # Check for termination - NetHack agent uses different format
                if isinstance(decision, dict):
                    # Handle tool call format: {'name': 'tool_name', 'parameters': {...}}
                    if decision.get("name") == "terminate":
                        termination_reason = "agent_quit"
                        break

                    # Extract actions from NetHack agent response
                    if "parameters" in decision and isinstance(decision["parameters"], dict):
                        params = decision["parameters"]
                        if "actions" in params:
                            actions = params["actions"]
                        elif "action" in params:
                            actions = [params["action"]]
                        else:
                            actions = ["wait"]  # Default action
                    elif "actions" in decision:
                        actions = decision["actions"]
                    elif "action" in decision:
                        actions = [decision["action"]]
                    else:
                        actions = ["wait"]  # Default action
                else:
                    # If decision is not a dict, assume it's a single action or termination
                    if decision == -1 or decision == [-1]:
                        termination_reason = "agent_quit"
                        break
                    elif isinstance(decision, list):
                        actions = decision
                    else:
                        actions = [str(decision)]

                if not isinstance(actions, list):
                    actions = [str(actions)]

                actions_per_turn.append(len(actions))

                # Collect turn data
                if collect_detailed_data:
                    turn_data = {
                        "turn": turn_count,
                        "actions_planned": len(actions),
                        "achievements_at_start": list(achievements_unlocked),
                        "balrog_percent": balrog_progress.percent,
                    }
                    turn_by_turn_data.append(turn_data)

                # Execute actions
                for action in actions:
                    obs_payload = await env.step(action)

                    # Check for REAL environment errors (not NetHack game messages)
                    if "error" in obs_payload:
                        error_msg = obs_payload["error"]
                        # NetHack game messages like "No stairs here" are normal, not environment errors
                        if error_msg and not any(
                            phrase in error_msg.lower()
                            for phrase in [
                                "no stairs",
                                "can't go",
                                "there is nothing",
                                "you can't",
                                "you don't",
                                "you aren't",
                                "you have no",
                                "invalid action",
                                "stairs here to",
                                "can't",
                                "there's nothing",
                                "no door",
                            ]
                        ):
                            print(f"   ⚠️ Real environment error: {error_msg}")
                            termination_reason = "environment_error"
                            break
                        # This is just a NetHack game message, continue playing

                    # Check termination status
                    private_state = obs_payload.get("private")
                    if private_state:
                        if getattr(private_state, "terminated", False) or getattr(
                            private_state, "truncated", False
                        ):
                            termination_reason = (
                                "timeout" if getattr(private_state, "truncated", False) else "death"
                            )
                            balrog_progress.update(
                                current_depth,
                                current_level,
                                done=True,
                                end_reason=termination_reason,
                            )
                            break

                if termination_reason in ["environment_error", "timeout", "death"]:
                    break

            # Final metrics
            if termination_reason == "unknown":
                termination_reason = "timeout"

            final_private = obs_payload.get("private")
            final_public = obs_payload.get("public")

            total_steps = getattr(final_public, "step_count", turn_count)
            total_reward = getattr(final_private, "total_reward", 0.0)

            # Final stats from player_stats
            player_stats = obs_payload.get("player_stats", {})
            final_depth = player_stats.get("depth", current_depth)
            final_level = player_stats.get("experience_level", current_level)
            final_gold = player_stats.get("gold", 0)

            # Success determination
            success = len(achievements_unlocked) > 0 or balrog_progress.percent > 5.0

            avg_actions_per_turn = (
                sum(actions_per_turn) / len(actions_per_turn) if actions_per_turn else 0.0
            )

            return TrajectoryResult(
                trajectory_id=trajectory_id,
                model_name=model_name,
                difficulty=difficulty,
                seed=seed,
                success=success,
                total_steps=total_steps,
                total_turns=turn_count,
                total_reward=total_reward,
                achievements_unlocked=achievements_unlocked,
                achievement_turn_unlocked=achievement_turn_unlocked,
                actions_per_turn=actions_per_turn,
                avg_actions_per_turn=avg_actions_per_turn,
                termination_reason=termination_reason,
                final_depth=final_depth,
                final_level=final_level,
                final_gold=final_gold,
                balrog_percent=balrog_progress.percent,
                turn_by_turn_data=turn_by_turn_data,
            )
        finally:
            pbar.close()

    async def run_evaluation(
        self,
        model_names: List[str],
        difficulties: List[str] = ["easy", "hard"],
        num_trajectories_per_condition: int = 3,
        max_turns: int = 200,
        collect_detailed_data: bool = True,
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation across models and difficulties."""

        print(f"🎯 Starting NetHack Evaluation")
        print(f"   Models: {model_names}")
        print(f"   Difficulties: {difficulties}")
        print(f"   Trajectories per condition: {num_trajectories_per_condition}")
        print(f"   Max turns per trajectory: {max_turns}")

        all_results = []

        for model_name in model_names:
            for difficulty in difficulties:
                print(f"\n🔄 Running {model_name} on {difficulty} difficulty...")

                # Run trajectories for this condition
                trajectory_tasks = []
                for i in range(num_trajectories_per_condition):
                    seed = 1000 + i if difficulty == "easy" else 2000 + i
                    trajectory_tasks.append(
                        self.run_single_trajectory(
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
        return self._generate_comprehensive_report()

    def _compute_aggregate_metrics(
        self, model_name: str, difficulty: str, trajectories: List[TrajectoryResult]
    ) -> AggregateResults:
        """Compute aggregate metrics for a model-difficulty condition."""

        num_trajectories = len(trajectories)
        if num_trajectories == 0:
            return AggregateResults(
                model_name=model_name,
                difficulty=difficulty,
                num_trajectories=0,
                success_rate=0.0,
                avg_total_steps=0.0,
                avg_total_turns=0.0,
                avg_total_reward=0.0,
                unique_achievements_unlocked=set(),
                total_achievement_count=0,
                avg_achievements_per_trajectory=0.0,
                achievement_unlock_rates={},
                hafner_score=0.0,
                balrog_score_avg=0.0,
                balrog_score_best=0.0,
                avg_actions_per_turn_overall=0.0,
                actions_per_turn_distribution={},
                termination_breakdown={},
                avg_final_depth=None,
                avg_final_level=None,
                avg_final_gold=None,
            )

        # Success metrics
        success_rate = sum(1 for t in trajectories if t.success) / num_trajectories
        avg_total_steps = sum(t.total_steps for t in trajectories) / num_trajectories
        avg_total_turns = sum(t.total_turns for t in trajectories) / num_trajectories
        avg_total_reward = sum(t.total_reward for t in trajectories) / num_trajectories

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

        # Compute Hafner score
        all_achievement_rates = []
        for achievement in ALL_ACHIEVEMENTS:
            unlock_rate = achievement_counts.get(achievement, 0) / num_trajectories
            all_achievement_rates.append(unlock_rate * 100.0)

        hafner_adjusted_score = hafner_score(all_achievement_rates)

        # Compute BALROG scores
        balrog_scores = [t.balrog_percent for t in trajectories]
        balrog_score_avg = sum(balrog_scores) / len(balrog_scores) if balrog_scores else 0.0
        balrog_score_best = max(balrog_scores) if balrog_scores else 0.0

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

        # Final stats
        depth_values = [t.final_depth for t in trajectories if t.final_depth is not None]
        level_values = [t.final_level for t in trajectories if t.final_level is not None]
        gold_values = [t.final_gold for t in trajectories if t.final_gold is not None]

        avg_final_depth = sum(depth_values) / len(depth_values) if depth_values else None
        avg_final_level = sum(level_values) / len(level_values) if level_values else None
        avg_final_gold = sum(gold_values) / len(gold_values) if gold_values else None

        return AggregateResults(
            model_name=model_name,
            difficulty=difficulty,
            num_trajectories=num_trajectories,
            success_rate=success_rate,
            avg_total_steps=avg_total_steps,
            avg_total_turns=avg_total_turns,
            avg_total_reward=avg_total_reward,
            unique_achievements_unlocked=all_achievements,
            total_achievement_count=total_achievement_count,
            avg_achievements_per_trajectory=avg_achievements_per_trajectory,
            achievement_unlock_rates=achievement_unlock_rates,
            hafner_score=hafner_adjusted_score,
            balrog_score_avg=balrog_score_avg,
            balrog_score_best=balrog_score_best,
            avg_actions_per_turn_overall=avg_actions_per_turn_overall,
            actions_per_turn_distribution=dict(actions_per_turn_dist),
            termination_breakdown=termination_breakdown,
            avg_final_depth=avg_final_depth,
            avg_final_level=avg_final_level,
            avg_final_gold=avg_final_gold,
        )

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
            "termination_breakdown_table": self._generate_termination_breakdown_table(
                aggregate_results
            ),
            "trajectory_by_trajectory_breakdown": self._generate_trajectory_breakdown(),
            "sota_comparison": self._generate_sota_comparison(aggregate_results),
            "raw_aggregate_results": [asdict(agg) for agg in aggregate_results],
            "raw_trajectory_results": [asdict(traj) for traj in self.trajectory_results],
        }

        return report

    def _generate_summary_table(self, aggregate_results: List[AggregateResults]) -> pd.DataFrame:
        """Generate main summary table with key metrics."""

        data = []
        for agg in aggregate_results:
            data.append(
                {
                    "Model": agg.model_name,
                    "Difficulty": agg.difficulty,
                    "Success Rate": f"{agg.success_rate:.1%}",
                    "Hafner Score": f"{agg.hafner_score:.1f}%",
                    "BALROG Avg": f"{agg.balrog_score_avg:.1f}%",
                    "BALROG Best": f"{agg.balrog_score_best:.1f}%",
                    "Avg Steps": f"{agg.avg_total_steps:.1f}",
                    "Avg Turns": f"{agg.avg_total_turns:.1f}",
                    "Avg Reward": f"{agg.avg_total_reward:.3f}",
                    "Unique Achievements": len(agg.unique_achievements_unlocked),
                    "Avg Achievements/Traj": f"{agg.avg_achievements_per_trajectory:.2f}",
                    "Avg Actions/Turn": f"{agg.avg_actions_per_turn_overall:.1f}",
                }
            )

        return pd.DataFrame(data)

    def _generate_achievement_percentage_table(
        self, grouped_results: Dict[str, Dict[str, List[TrajectoryResult]]]
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
            for ach in ACHIEVEMENT_CATEGORIES[category]:
                if ach in df.columns:
                    achievement_cols.append(ach)

        return df[base_cols + achievement_cols]

    def _generate_termination_breakdown_table(
        self, aggregate_results: List[AggregateResults]
    ) -> pd.DataFrame:
        """Generate table showing termination reason percentages."""

        data = []
        for agg in aggregate_results:
            row = {
                "Model": agg.model_name,
                "Difficulty": agg.difficulty,
            }

            for reason in TERMINATION_REASONS:
                percentage = agg.termination_breakdown.get(reason, 0.0)
                row[f"{reason.title()} %"] = f"{percentage:.1%}"

            data.append(row)

        return pd.DataFrame(data)

    def _generate_trajectory_breakdown(self) -> pd.DataFrame:
        """Generate detailed trajectory-by-trajectory breakdown."""

        data = []
        for traj in self.trajectory_results:
            # Achievement category breakdown
            basic_achievements = len(
                [a for a in traj.achievements_unlocked if a in ACHIEVEMENT_CATEGORIES["basic"]]
            )
            inter_achievements = len(
                [
                    a
                    for a in traj.achievements_unlocked
                    if a in ACHIEVEMENT_CATEGORIES["intermediate"]
                ]
            )
            adv_achievements = len(
                [a for a in traj.achievements_unlocked if a in ACHIEVEMENT_CATEGORIES["advanced"]]
            )

            data.append(
                {
                    "Trajectory ID": traj.trajectory_id[:8],
                    "Model": traj.model_name,
                    "Difficulty": traj.difficulty,
                    "Seed": traj.seed,
                    "Success": "✓" if traj.success else "✗",
                    "Steps": traj.total_steps,
                    "Turns": traj.total_turns,
                    "Reward": f"{traj.total_reward:.3f}",
                    "Total Achievements": len(traj.achievements_unlocked),
                    "Basic": basic_achievements,
                    "Intermediate": inter_achievements,
                    "Advanced": adv_achievements,
                    "BALROG Score": f"{traj.balrog_percent:.1f}%",
                    "Termination": traj.termination_reason,
                    "Final Depth": traj.final_depth,
                    "Achievements": ", ".join(sorted(traj.achievements_unlocked))
                    if traj.achievements_unlocked
                    else "None",
                }
            )

        return pd.DataFrame(data)

    def _generate_sota_comparison(
        self, aggregate_results: List[AggregateResults]
    ) -> Dict[str, pd.DataFrame]:
        """Generate comparison tables with SOTA benchmarks, separating Hafner and BALROG methodologies."""

        # Create our results table for both methodologies
        our_hafner_data = []
        our_balrog_data = []

        for agg in aggregate_results:
            # Hafner results
            hafner_row = {
                "System": f"{agg.model_name} (multi-action)",
                "Hafner Score": f"{agg.hafner_score:.1f}%",
                "Category": "Current Evaluation (Hafner)",
            }
            our_hafner_data.append(hafner_row)

            # BALROG results
            balrog_row = {
                "System": f"{agg.model_name} (multi-action)",
                "BALROG Score (Avg)": f"{agg.balrog_score_avg:.1f}%",
                "BALROG Score (Best)": f"{agg.balrog_score_best:.1f}%",
                "Category": "Current Evaluation (BALROG)",
            }

            # Add percentage comparison to BALROG SOTA if we can map the model name
            if agg.model_name in MODEL_NAME_TO_SOTA:
                sota_name = MODEL_NAME_TO_SOTA[agg.model_name]
                if sota_name in BALROG_SOTA_SCORES["balrog_leaderboard"]:
                    balrog_sota_score = BALROG_SOTA_SCORES["balrog_leaderboard"][sota_name]
                    percentage_of_balrog_sota_avg = (agg.balrog_score_avg / balrog_sota_score) * 100
                    percentage_of_balrog_sota_best = (
                        agg.balrog_score_best / balrog_sota_score
                    ) * 100
                    balrog_row["% of BALROG SOTA (Avg)"] = f"{percentage_of_balrog_sota_avg:.1f}%"
                    balrog_row["% of BALROG SOTA (Best)"] = f"{percentage_of_balrog_sota_best:.1f}%"
                    balrog_row["BALROG SOTA Reference"] = f"{sota_name} ({balrog_sota_score:.1f}%)"

            our_balrog_data.append(balrog_row)

        our_hafner_df = pd.DataFrame(our_hafner_data)
        our_balrog_df = pd.DataFrame(our_balrog_data)

        return {
            "our_hafner_results": our_hafner_df,
            "our_balrog_results": our_balrog_df,
            "methodology_note": "⚠️  CRITICAL: Hafner scores (log-adjusted multi-episode) and BALROG scores (simple single-episode percentage) use different methodologies and are NOT directly comparable!",
        }

    def print_report(self, report: Dict[str, Any]):
        """Print a formatted evaluation report."""

        print("\n" + "=" * 80)
        print("🎯 NETHACK EVALUATION REPORT")
        print("=" * 80)

        # Summary table
        print("\n📊 EVALUATION SUMMARY")
        summary_df = report["evaluation_summary"]
        # Clean formatting for summary table
        for col in summary_df.columns:
            if len(col) > 12:  # Truncate long column names
                summary_df = summary_df.rename(columns={col: col[:12]})
        print(summary_df.to_string(index=False, max_colwidth=12))

        # Create and show vertical achievement table
        print("\n🏆 ACHIEVEMENT UNLOCK RATES")
        print("Format: unlocked/total (percentage)")

        # Group results for achievement summary
        grouped_results = defaultdict(lambda: defaultdict(list))
        for traj in self.trajectory_results:
            grouped_results[traj.model_name][traj.difficulty].append(traj)

        achievement_summary = self._generate_achievement_summary_table(grouped_results)

        # Print by category for better readability
        for category in ["Basic", "Intermediate", "Advanced"]:
            category_data = achievement_summary[achievement_summary["Category"] == category]
            if not category_data.empty:
                print(f"\n{category.upper()} ACHIEVEMENTS:")
                category_display = category_data.drop("Category", axis=1)
                print(category_display.to_string(index=False))

        # Trajectory breakdown (summary stats only for space)
        traj_df = report["trajectory_by_trajectory_breakdown"]
        print(f"\n📋 TRAJECTORY BREAKDOWN ({len(traj_df)} total trajectories)")
        print("Sample trajectories:")
        sample_cols = [
            "Model",
            "Difficulty",
            "Success",
            "Steps",
            "Total Achievements",
            "BALROG Score",
            "Termination",
        ]
        sample_df = traj_df[sample_cols].head(5)
        print(sample_df.to_string(index=False, max_colwidth=12))
        if len(traj_df) > 5:
            print(f"... and {len(traj_df) - 5} more trajectories")

        # SOTA comparison
        sota_comparison = report["sota_comparison"]
        print("\n🏆 SOTA COMPARISON")
        print(sota_comparison["methodology_note"])

        print("\n📊 HAFNER METHODOLOGY RESULTS (Multi-episode log-adjusted)")
        hafner_df = sota_comparison["our_hafner_results"]
        print(hafner_df.to_string(index=False, max_colwidth=20))

        print("\n📊 BALROG METHODOLOGY RESULTS (Single-episode percentage)")
        balrog_df = sota_comparison["our_balrog_results"]
        # Clean up column names for better display
        balrog_clean = balrog_df.copy()
        if "% of BALROG SOTA (Avg)" in balrog_clean.columns:
            balrog_clean = balrog_clean.rename(columns={"% of BALROG SOTA (Avg)": "% SOTA Avg"})
        if "% of BALROG SOTA (Best)" in balrog_clean.columns:
            balrog_clean = balrog_clean.rename(columns={"% of BALROG SOTA (Best)": "% SOTA Best"})
        print(balrog_clean.to_string(index=False, max_colwidth=20))

        print("\n" + "=" * 80)

    def _generate_achievement_summary_table(
        self, grouped_results: Dict[str, Dict[str, List[TrajectoryResult]]]
    ) -> pd.DataFrame:
        """Generate a vertical achievement summary table that's easier to read."""

        data = []

        # For each achievement, show rates across all model/difficulty combinations
        for category_name, achievements in ACHIEVEMENT_CATEGORIES.items():
            for achievement in achievements:
                row = {
                    "Category": category_name.capitalize(),
                    "Achievement": achievement.replace("_", " ").title(),
                }

                # Add columns for each model/difficulty combination
                for model_name, difficulties in grouped_results.items():
                    for difficulty, trajectories in difficulties.items():
                        if not trajectories:
                            continue

                        num_trajectories = len(trajectories)
                        count = sum(
                            1 for traj in trajectories if achievement in traj.achievements_unlocked
                        )
                        percentage = count / num_trajectories if num_trajectories > 0 else 0.0

                        col_name = f"{model_name} ({difficulty})"
                        row[col_name] = f"{count}/{num_trajectories} ({percentage:.1%})"

                data.append(row)

        return pd.DataFrame(data)


# Convenience function for quick evaluations
async def run_nethack_eval(
    model_names: List[str],
    difficulties: List[str] = ["easy", "hard"],
    num_trajectories: int = 3,
    max_turns: int = 200,
) -> Dict[str, Any]:
    """Quick evaluation runner with automatic report generation."""

    framework = NetHackEvalFramework()
    report = await framework.run_evaluation(
        model_names=model_names,
        difficulties=difficulties,
        num_trajectories_per_condition=num_trajectories,
        max_turns=max_turns,
    )

    framework.print_report(report)
    return report
