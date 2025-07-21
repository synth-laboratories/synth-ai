"""
Standardized Crafter Evaluation Framework
Provides detailed metrics, trajectory analysis, and achievement statistics.
"""

import asyncio
import json
import time
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
import uuid

import pandas as pd
from tqdm import tqdm
from synth_ai.lm.core.main import LM

# Achievement categories based on difficulty/complexity
ACHIEVEMENT_CATEGORIES = {
    "easy": [
        "collect_wood",
        "collect_stone",
        "collect_sapling",
        "place_stone",
        "place_table",
        "wake_up",
        "eat_plant",
        "collect_drink",
    ],
    "medium": [
        "make_wood_pickaxe",
        "make_wood_sword",
        "place_furnace",
        "place_plant",
        "collect_coal",
        "collect_iron",
        "eat_cow",
    ],
    "hard": [
        "make_stone_pickaxe",
        "make_stone_sword",
        "make_iron_pickaxe",
        "make_iron_sword",
        "collect_diamond",
        "defeat_skeleton",
        "defeat_zombie",
    ],
}

ALL_ACHIEVEMENTS = [ach for category in ACHIEVEMENT_CATEGORIES.values() for ach in category]

TERMINATION_REASONS = ["timeout", "death", "agent_quit", "environment_error"]

# SOTA Benchmarks for comparison
# ⚠️  IMPORTANT: These use different scoring methodologies and are NOT directly comparable!

HAFNER_SOTA_SCORES = {
    # Official Hafner scores use log-adjusted multi-episode success rates
    "rl_baselines_hafner": {
        "Achievement Distillation + EnvGen (COLM 2024)": 35.3,
        "PPO + EnvGen": 32.2,
        "Curious Replay": 19.4,
        "Human experts": 50.5,
        "SPRING (GPT-4 planner)": 27.3,
        "Plan2Explore (unsupervised)": 2.1,
    }
}

BALROG_SOTA_SCORES = {
    # BALROG scores use simple percentage: achievements_unlocked/22 * 100
    "balrog_leaderboard": {
        "Claude 3.5 Sonnet": 37.3,
        "Gemini 1.5 Pro": 36.4,
        "GPT-4o": 33.6,
        "Claude 3 Opus": 33.1,
        "GPT-4 Turbo": 32.7,
        "Gemini 1.5 Flash": 31.7,
        "Claude 3.5 Haiku": 31.2,
        "GPT-4o-mini": 30.2,
        "Llama 3.1 405B": 28.6,
        "Gemini 1.0 Pro": 27.7,
        "Claude 3 Haiku": 27.3,
        "Llama 3.1 70B": 26.4,
        "GPT-3.5 Turbo": 26.2,
        "Llama 3.1 8B": 25.5,
        "Gemini 1.5 Flash-8B": 25.0,
        "Llama 3 70B": 22.7,
        "Llama 3 8B": 20.0,
        "Gemini 1.0 Pro Vision": 17.3,
        "GPT-3.5 Turbo Instruct": 16.4,
    }
}

# Model name mapping for SOTA percentage calculations
MODEL_NAME_TO_SOTA = {
    "claude-3-5-sonnet-latest": "Claude 3.5 Sonnet",
    "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet",
    "claude-3-5-haiku-latest": "Claude 3.5 Haiku",
    "claude-3-5-haiku-20241022": "Claude 3.5 Haiku",
    "claude-3-opus-20240229": "Claude 3 Opus",
    "claude-3-haiku-20240307": "Claude 3 Haiku",
    "gpt-4o": "GPT-4o",
    "gpt-4o-2024-11-20": "GPT-4o",
    "gpt-4o-mini": "GPT-4o-mini",
    "gpt-4o-mini-2024-07-18": "GPT-4o-mini",
    "gpt-4-turbo": "GPT-4 Turbo",
    "gpt-4-turbo-2024-04-09": "GPT-4 Turbo",
    "gpt-3.5-turbo": "GPT-3.5 Turbo",
    "gpt-3.5-turbo-instruct": "GPT-3.5 Turbo Instruct",
    "gemini-1.5-pro-latest": "Gemini 1.5 Pro",
    "gemini-1.5-flash-latest": "Gemini 1.5 Flash",
    "gemini-1.0-pro": "Gemini 1.0 Pro",
}


def crafter_score(success_rates_percent: List[float]) -> float:
    """
    Compute the official Hafner adjusted Crafter score (2022).

    Args:
        success_rates_percent: List of success rates for each achievement (0-100 scale)

    Returns:
        Log-adjusted Crafter score as percentage (0-100)

    Formula: exp(mean(log(1+si))) - 1 where si is each achievement's success rate in %
    """
    if not success_rates_percent:
        return 0.0

    N = len(success_rates_percent)
    g = sum(math.log(1 + s) for s in success_rates_percent) / N
    return math.exp(g) - 1


def balrog_score(achievements_unlocked: int, total_achievements: int = 22) -> float:
    """
    Compute BALROG-style Crafter score (simple percentage).

    Args:
        achievements_unlocked: Number of achievements unlocked in episode
        total_achievements: Total possible achievements (22 in Crafter)

    Returns:
        Simple percentage score (0-100)

    Formula: (achievements_unlocked / total_achievements) * 100
    """
    return (achievements_unlocked / total_achievements) * 100.0


@dataclass
class TrajectoryResult:
    """Results from a single trajectory/episode."""

    trajectory_id: str
    model_name: str
    difficulty: str
    seed: int

    # Core metrics
    success: bool
    total_steps: int
    total_turns: int  # Number of agent decision turns
    total_reward: float

    # Time metrics
    total_duration_sec: float  # Episode wall-clock duration in seconds

    # Achievement tracking
    achievements_unlocked: Set[str]
    achievement_turn_unlocked: Dict[str, int]  # achievement -> turn when unlocked

    # Multi-action metrics
    actions_per_turn: List[int]  # Number of actions per turn
    avg_actions_per_turn: float

    # Termination analysis
    termination_reason: str  # "timeout", "death", "agent_quit", "environment_error"
    final_health: Optional[float]
    final_food: Optional[int]
    final_drink: Optional[int]

    # Trajectory data for detailed analysis
    turn_by_turn_data: Optional[List[Dict[str, Any]]] = None


@dataclass
class AggregateResults:
    """Aggregate results across multiple trajectories."""

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
    achievement_unlock_rates: Dict[str, float]  # achievement -> % of trajectories that unlocked it
    hafner_score: float  # Official Hafner adjusted score (log-mean)
    balrog_score_avg: float  # Average BALROG-style score across trajectories
    balrog_score_best: float  # Best single BALROG-style score

    # Multi-action metrics
    avg_actions_per_turn_overall: float
    actions_per_turn_distribution: Dict[int, int]  # num_actions -> count

    # Termination analysis
    termination_breakdown: Dict[str, float]  # reason -> percentage
    avg_final_health: Optional[float]
    avg_final_food: Optional[float]
    avg_final_drink: Optional[float]

    # Rollout duration stats (seconds)
    median_duration_sec: float
    p90_duration_sec: float
    max_duration_sec: float


class CrafterEvalFramework:
    """Standardized evaluation framework for Crafter environments."""

    def __init__(self):
        self.trajectory_results: List[TrajectoryResult] = []

    async def run_single_trajectory(
        self,
        model_name: str,
        difficulty: str,
        seed: int,
        max_turns: int = 30,
        collect_detailed_data: bool = True,
    ) -> TrajectoryResult:
        """Run a single trajectory and collect detailed metrics."""
        from src.synth_env.examples.crafter_classic.agent_demos.crafter_react_agent import (
            ReActAgent,
            CrafterHistoryObservationCallable,
            CrafterMove,
        )
        from src.synth_env.examples.crafter_classic.environment import (
            CrafterClassicEnvironment,
        )
        from synth_ai.environments.examples.crafter_classic.taskset import (
            CrafterTaskInstance,
            CrafterTaskInstanceMetadata,
        )
        from synth_ai.environments.tasks.core import Impetus, Intent
        # LM import moved to top level

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

        # Run episode
        start_time = time.perf_counter()
        obs_payload = await env.initialize()
        turn_count = 0
        termination_reason = "unknown"

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
                # Calculate achievement breakdown by difficulty
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

                # Agent decision
                action_sequence = await agent.decide(current_formatted_obs, obs_payload)

                if action_sequence == [-1]:  # Agent terminated
                    termination_reason = "agent_quit"
                    break

                actions_per_turn.append(len(action_sequence))

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

                # Execute actions
                for i, act_idx in enumerate(action_sequence):
                    obs_payload = await env.step([[CrafterMove(act_idx)]])

                    if "error" in obs_payload:
                        termination_reason = "environment_error"
                        break

                    if obs_payload["private"].terminated or obs_payload["private"].truncated:
                        termination_reason = (
                            "timeout" if obs_payload["private"].truncated else "death"
                        )
                        break

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

            return TrajectoryResult(
                trajectory_id=trajectory_id,
                model_name=model_name,
                difficulty=difficulty,
                seed=seed,
                success=success,
                total_steps=total_steps,
                total_turns=turn_count,
                total_reward=total_reward,
                total_duration_sec=time.perf_counter() - start_time,
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
        finally:
            pbar.close()

    async def run_evaluation(
        self,
        model_names: List[str],
        difficulties: List[str] = ["easy", "hard"],
        num_trajectories_per_condition: int = 3,
        max_turns: int = 30,
        collect_detailed_data: bool = True,
    ) -> Dict[str, Any]:
        """Run comprehensive evaluation across models and difficulties."""

        print(f"🎯 Starting Crafter Evaluation")
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
            "multi_action_analysis": self._generate_multi_action_analysis(aggregate_results),
            "trajectory_by_trajectory_breakdown": self._generate_trajectory_breakdown(),
            "model_comparison_tables": self._generate_model_comparison_tables(aggregate_results),
            "sota_comparison": self._generate_sota_comparison(aggregate_results),
            "raw_aggregate_results": [asdict(agg) for agg in aggregate_results],
            "raw_trajectory_results": [asdict(traj) for traj in self.trajectory_results],
        }

        return report

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
                avg_final_health=None,
                avg_final_food=None,
                avg_final_drink=None,
                median_duration_sec=0.0,
                p90_duration_sec=0.0,
                max_duration_sec=0.0,
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

        # Compute Hafner adjusted score across all achievements
        all_achievement_rates = []
        for achievement in ALL_ACHIEVEMENTS:
            unlock_rate = achievement_counts.get(achievement, 0) / num_trajectories
            all_achievement_rates.append(unlock_rate * 100.0)  # Convert to percentage

        hafner_adjusted_score = crafter_score(all_achievement_rates)

        # Compute BALROG scores
        balrog_scores = [balrog_score(len(traj.achievements_unlocked)) for traj in trajectories]
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

        # Survival stats
        health_values = [t.final_health for t in trajectories if t.final_health is not None]
        food_values = [t.final_food for t in trajectories if t.final_food is not None]
        drink_values = [t.final_drink for t in trajectories if t.final_drink is not None]

        avg_final_health = sum(health_values) / len(health_values) if health_values else None
        avg_final_food = sum(food_values) / len(food_values) if food_values else None
        avg_final_drink = sum(drink_values) / len(drink_values) if drink_values else None

        # Duration stats
        durations = [t.total_duration_sec for t in trajectories]
        durations.sort()
        median_duration_sec = durations[len(durations) // 2] if durations else 0.0
        p90_duration_sec = durations[int(len(durations) * 0.9)] if durations else 0.0
        max_duration_sec = durations[-1] if durations else 0.0

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
            avg_actions_per_turn_overall=avg_actions_per_turn_overall,
            actions_per_turn_distribution=dict(actions_per_turn_dist),
            termination_breakdown=termination_breakdown,
            avg_final_health=avg_final_health,
            avg_final_food=avg_final_food,
            avg_final_drink=avg_final_drink,
            hafner_score=hafner_adjusted_score,
            balrog_score_avg=balrog_score_avg,
            balrog_score_best=balrog_score_best,
            median_duration_sec=median_duration_sec,
            p90_duration_sec=p90_duration_sec,
            max_duration_sec=max_duration_sec,
        )

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
                    "Q2 Secs": f"{agg.median_duration_sec:.1f}",
                    "P90 Secs": f"{agg.p90_duration_sec:.1f}",
                    "Max Secs": f"{agg.max_duration_sec:.1f}",
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
        for category in ["easy", "medium", "hard"]:
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

    def _generate_multi_action_analysis(
        self, aggregate_results: List[AggregateResults]
    ) -> Dict[str, pd.DataFrame]:
        """Generate analysis of multi-action tool calls."""

        # Summary table
        summary_data = []
        for agg in aggregate_results:
            summary_data.append(
                {
                    "Model": agg.model_name,
                    "Difficulty": agg.difficulty,
                    "Avg Actions/Turn": f"{agg.avg_actions_per_turn_overall:.2f}",
                    "Most Common": max(
                        agg.actions_per_turn_distribution.items(), key=lambda x: x[1]
                    )[0]
                    if agg.actions_per_turn_distribution
                    else 0,
                    "Distribution": str(dict(sorted(agg.actions_per_turn_distribution.items()))),
                }
            )

        summary_df = pd.DataFrame(summary_data)

        # Detailed distribution table
        all_action_counts = set()
        for agg in aggregate_results:
            all_action_counts.update(agg.actions_per_turn_distribution.keys())

        dist_data = []
        for agg in aggregate_results:
            row = {"Model": agg.model_name, "Difficulty": agg.difficulty}
            total_turns = sum(agg.actions_per_turn_distribution.values())

            for count in sorted(all_action_counts):
                turns_with_count = agg.actions_per_turn_distribution.get(count, 0)
                percentage = turns_with_count / total_turns if total_turns > 0 else 0.0
                row[f"{count} Actions"] = f"{percentage:.1%}"

            dist_data.append(row)

        distribution_df = pd.DataFrame(dist_data)

        return {"summary": summary_df, "distribution": distribution_df}

    def _generate_trajectory_breakdown(self) -> pd.DataFrame:
        """Generate detailed trajectory-by-trajectory breakdown."""

        data = []
        for traj in self.trajectory_results:
            # Achievement category breakdown
            easy_achievements = len(
                [a for a in traj.achievements_unlocked if a in ACHIEVEMENT_CATEGORIES["easy"]]
            )
            medium_achievements = len(
                [a for a in traj.achievements_unlocked if a in ACHIEVEMENT_CATEGORIES["medium"]]
            )
            hard_achievements = len(
                [a for a in traj.achievements_unlocked if a in ACHIEVEMENT_CATEGORIES["hard"]]
            )

            data.append(
                {
                    "Trajectory ID": traj.trajectory_id[:8],  # Short ID
                    "Model": traj.model_name,
                    "Difficulty": traj.difficulty,
                    "Seed": traj.seed,
                    "Success": "✓" if traj.success else "✗",
                    "Steps": traj.total_steps,
                    "Turns": traj.total_turns,
                    "Reward": f"{traj.total_reward:.3f}",
                    "Total Achievements": len(traj.achievements_unlocked),
                    "Easy": easy_achievements,
                    "Medium": medium_achievements,
                    "Hard": hard_achievements,
                    "Avg Actions/Turn": f"{traj.avg_actions_per_turn:.1f}",
                    "Termination": traj.termination_reason,
                    "Final Health": traj.final_health,
                    "Achievements": ", ".join(sorted(traj.achievements_unlocked))
                    if traj.achievements_unlocked
                    else "None",
                }
            )

        return pd.DataFrame(data)

    def _generate_model_comparison_tables(
        self, aggregate_results: List[AggregateResults]
    ) -> Dict[str, Any]:
        """Generate model-to-model comparison tables and deltas."""

        if len(set(agg.model_name for agg in aggregate_results)) < 2:
            return {"note": "Need at least 2 models for comparison"}

        # Group by difficulty for comparison
        by_difficulty = defaultdict(list)
        for agg in aggregate_results:
            by_difficulty[agg.difficulty].append(agg)

        comparison_tables = {}

        for difficulty, agg_list in by_difficulty.items():
            if len(agg_list) < 2:
                continue

            # Sort by model name for consistent ordering
            agg_list.sort(key=lambda x: x.model_name)

            # Create comparison table
            comparison_data = []
            for agg in agg_list:
                comparison_data.append(
                    {
                        "Model": agg.model_name,
                        "Success Rate": agg.success_rate,
                        "Avg Steps": agg.avg_total_steps,
                        "Avg Achievements": agg.avg_achievements_per_trajectory,
                        "Avg Actions/Turn": agg.avg_actions_per_turn_overall,
                    }
                )

            comparison_df = pd.DataFrame(comparison_data)

            # Create delta table (difference from first model)
            if len(agg_list) > 1:
                baseline = agg_list[0]
                delta_data = []

                for agg in agg_list[1:]:
                    delta_data.append(
                        {
                            "Model vs Baseline": f"{agg.model_name} vs {baseline.model_name}",
                            "Success Rate Δ": f"{agg.success_rate - baseline.success_rate:+.1%}",
                            "Avg Steps Δ": f"{agg.avg_total_steps - baseline.avg_total_steps:+.1f}",
                            "Avg Achievements Δ": f"{agg.avg_achievements_per_trajectory - baseline.avg_achievements_per_trajectory:+.2f}",
                            "Avg Actions/Turn Δ": f"{agg.avg_actions_per_turn_overall - baseline.avg_actions_per_turn_overall:+.2f}",
                        }
                    )

                delta_df = pd.DataFrame(delta_data) if delta_data else None
            else:
                delta_df = None

            comparison_tables[difficulty] = {
                "comparison": comparison_df,
                "deltas": delta_df,
            }

        return comparison_tables

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

    def _generate_sota_comparison(
        self, aggregate_results: List[AggregateResults]
    ) -> Dict[str, pd.DataFrame]:
        """Generate comparison tables with SOTA benchmarks, separating Hafner and BALROG methodologies."""

        # ⚠️  CRITICAL: Hafner and BALROG scores use different methodologies and are NOT comparable!

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

        # Create nearby comparisons for BALROG methodology only (since that's what we can compare to)
        balrog_nearby_comparisons = []
        all_balrog_scores = []

        # Add BALROG leaderboard scores
        for system, score in BALROG_SOTA_SCORES["balrog_leaderboard"].items():
            all_balrog_scores.append(
                {"System": system, "Score": score, "Category": "BALROG Leaderboard"}
            )

        # Sort BALROG scores
        all_balrog_scores.sort(key=lambda x: x["Score"], reverse=True)

        # For each of our models, find nearby BALROG scores
        for agg in aggregate_results:
            # Use average BALROG score for comparison
            model_balrog_score = agg.balrog_score_avg

            # Find position where this model would fit
            insert_pos = 0
            for i, sota_entry in enumerate(all_balrog_scores):
                if model_balrog_score > sota_entry["Score"]:
                    insert_pos = i
                    break
                insert_pos = i + 1

            # Get 2 scores above and 2 scores below (if available)
            start_idx = max(0, insert_pos - 2)
            end_idx = min(len(all_balrog_scores), insert_pos + 3)

            nearby_scores = all_balrog_scores[start_idx:end_idx]

            # Create comparison table for this model
            comparison_data = []

            # Add scores above
            for sota_entry in nearby_scores[: insert_pos - start_idx]:
                comparison_data.append(
                    {
                        "System": sota_entry["System"],
                        "BALROG Score": f"{sota_entry['Score']:.1f}%",
                        "Category": sota_entry["Category"],
                    }
                )

            # Add our model
            row = {
                "System": f"{agg.model_name} (multi-action)",
                "BALROG Score": f"{agg.balrog_score_avg:.1f}%",
                "Category": "Current Evaluation",
            }

            # Add percentage of BALROG SOTA if we can map the model name
            if agg.model_name in MODEL_NAME_TO_SOTA:
                sota_name = MODEL_NAME_TO_SOTA[agg.model_name]
                if sota_name in BALROG_SOTA_SCORES["balrog_leaderboard"]:
                    balrog_sota_score = BALROG_SOTA_SCORES["balrog_leaderboard"][sota_name]
                    percentage_of_balrog_sota = (agg.balrog_score_avg / balrog_sota_score) * 100
                    row["% of BALROG SOTA"] = f"{percentage_of_balrog_sota:.1f}%"
                    row["BALROG SOTA Reference"] = f"{sota_name} ({balrog_sota_score:.1f}%)"

            comparison_data.append(row)

            # Add scores below
            for sota_entry in nearby_scores[insert_pos - start_idx :]:
                comparison_data.append(
                    {
                        "System": sota_entry["System"],
                        "BALROG Score": f"{sota_entry['Score']:.1f}%",
                        "Category": sota_entry["Category"],
                    }
                )

            balrog_nearby_comparisons.append(
                {"model": agg.model_name, "comparison": pd.DataFrame(comparison_data)}
            )

        return {
            "our_hafner_results": our_hafner_df,
            "our_balrog_results": our_balrog_df,
            "balrog_nearby_comparisons": balrog_nearby_comparisons,
            "methodology_note": "⚠️  CRITICAL: Hafner scores (log-adjusted multi-episode) and BALROG scores (simple single-episode percentage) use different methodologies and are NOT directly comparable!",
        }

    def print_report(self, report: Dict[str, Any]):
        """Print a formatted evaluation report."""

        print("\n" + "=" * 80)
        print("🎯 CRAFTER EVALUATION REPORT")
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
        for category in ["Easy", "Medium", "Hard"]:
            category_data = achievement_summary[achievement_summary["Category"] == category]
            if not category_data.empty:
                print(f"\n{category.upper()} ACHIEVEMENTS:")
                category_display = category_data.drop("Category", axis=1)
                print(category_display.to_string(index=False))

        # # Termination breakdown
        # print("\n⚰️  TERMINATION BREAKDOWN")
        # print(report["termination_breakdown_table"].to_string(index=False))

        # # Multi-action analysis
        # print("\n⚡ MULTI-ACTION ANALYSIS")
        # multi_action = report["multi_action_analysis"]

        # # Clean summary table
        # summary_clean = multi_action["summary"].copy()
        # summary_clean = summary_clean.drop(columns=["Distribution"], errors='ignore')  # Remove cluttered distribution column
        # print("Summary:")
        # print(summary_clean.to_string(index=False, max_colwidth=15))

        # # Show distribution in cleaner format
        # print("\nAction Count Distribution:")
        # dist_clean = multi_action["distribution"].copy()
        # # Only show columns with meaningful data
        # cols_to_show = ["Model", "Difficulty"] + [col for col in dist_clean.columns if "Actions" in col and not dist_clean[col].str.contains("0.0%").all()]
        # if len(cols_to_show) > 8:  # Limit to prevent overflow
        #     cols_to_show = cols_to_show[:8]
        # print(dist_clean[cols_to_show].to_string(index=False, max_colwidth=10))

        # Model comparisons
        if "note" not in report["model_comparison_tables"]:
            print("\n🔄 MODEL COMPARISONS")
            for difficulty, tables in report["model_comparison_tables"].items():
                print(f"\n{difficulty.upper()} Difficulty:")
                print(tables["comparison"].to_string(index=False))
                if tables["deltas"] is not None:
                    print(f"\nDeltas vs Baseline:")
                    print(tables["deltas"].to_string(index=False))

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
            "Termination",
        ]
        sample_df = traj_df[sample_cols].head(5)  # Show fewer rows for cleaner display
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

        print("\n🎯 BALROG vs Nearby SOTA Benchmarks (Apples-to-Apples)")
        for comparison in sota_comparison["balrog_nearby_comparisons"]:
            print(f"\n{comparison['model']} vs Nearby BALROG Scores:")
            comp_df = comparison["comparison"]
            # Clean up long reference columns
            comp_clean = comp_df.copy()
            if "BALROG SOTA Reference" in comp_clean.columns:
                comp_clean = comp_clean.drop(
                    columns=["BALROG SOTA Reference"]
                )  # Too long for display
            print(comp_clean.to_string(index=False, max_colwidth=18))

        print("\n" + "=" * 80)


# Convenience function for quick evaluations
async def run_crafter_eval(
    model_names: List[str],
    difficulties: List[str] = ["easy", "hard"],
    num_trajectories: int = 3,
    max_turns: int = 30,
) -> Dict[str, Any]:
    """Quick evaluation runner with automatic report generation."""

    framework = CrafterEvalFramework()
    report = await framework.run_evaluation(
        model_names=model_names,
        difficulties=difficulties,
        num_trajectories_per_condition=num_trajectories,
        max_turns=max_turns,
    )

    framework.print_report(report)
    return report
