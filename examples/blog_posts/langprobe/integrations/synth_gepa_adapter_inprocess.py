"""Synth GEPA adapter using in-process Python API (no HTTP polling).

This adapter uses the direct GEPAOptimizer class from the backend,
avoiding HTTP polling and providing better control and progress tracking.
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from tqdm import tqdm

from .learning_curve_tracker import LearningCurveTracker

load_dotenv()


@dataclass
class GEPAResult:
    """Result from GEPA optimization."""
    best_template: Any  # PromptTemplate
    train_score: float
    train_scores: list[float]  # Per-seed train scores
    val_score: Optional[float] = None  # Validation score if available
    job_id: str = ""
    total_rollouts: int = 0
    
    def to_dict(self, learning_curve_dict: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "job_id": self.job_id,
            "best_score": self.train_score,  # For backwards compat
            "best_prompt": self.best_template.to_dict() if self.best_template else {},
            "train_score": self.train_score,
            "train_scores": self.train_scores,
            "val_score": self.val_score,
            "total_rollouts": self.total_rollouts,
        }
        if learning_curve_dict is not None:
            result["learning_curve"] = learning_curve_dict
        return result

# Try to import backend optimizer classes
try:
    # Add monorepo backend to path if available
    REPO_ROOT = Path(__file__).resolve().parents[4]
    MONOREPO_ROOT = REPO_ROOT.parent / "monorepo"
    BACKEND_ROOT = MONOREPO_ROOT / "backend"
    if BACKEND_ROOT.exists():
        sys.path.insert(0, str(BACKEND_ROOT))
    
    from app.routes.prompt_learning.algorithm.gepa.optimizer import (
        GEPAOptimizer,
        GEPAConfig,
    )
    from app.routes.prompt_learning.core.runtime import LocalRuntime
    from app.routes.prompt_learning.core.patterns import PromptPattern, MessagePattern
    
    BACKEND_AVAILABLE = True
except ImportError as e:
    BACKEND_AVAILABLE = False
    GEPAOptimizer = None  # type: ignore
    GEPAConfig = None  # type: ignore
    LocalRuntime = None  # type: ignore
    PromptPattern = None  # type: ignore
    MessagePattern = None  # type: ignore


class JobStatus(str, Enum):
    """Job status values."""
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    
    @classmethod
    def is_terminal(cls, status: str) -> bool:
        """Check if status is a terminal state."""
        status_lower = status.lower() if isinstance(status, str) else str(status).lower()
        return status_lower in (
            cls.SUCCEEDED.value,
            cls.COMPLETED.value,
            "done",
            "finished",
            "success",
            cls.FAILED.value,
            "error",
            "crashed",
            cls.CANCELLED.value,
            "canceled",
        )


class SynthGEPAAdapterInProcess:
    """Adapter for Synth GEPA using in-process Python API."""

    def __init__(
        self,
        task_app_url: str,
        task_app_id: str,
        initial_prompt_messages: list[dict[str, Any]],
        rollout_budget: int = 400,
        train_seeds: Optional[list[int]] = None,
        val_seeds: Optional[list[int]] = None,
    ):
        """Initialize Synth GEPA adapter.

        Args:
            task_app_url: Task app URL (e.g., "http://127.0.0.1:8115")
            task_app_id: Task app ID (e.g., "iris")
            initial_prompt_messages: Initial prompt messages
            rollout_budget: Total rollout budget (~400)
            train_seeds: Training seeds (default: auto-scale based on budget)
            val_seeds: Validation seeds for held-out evaluation (default: None)
        """
        if not BACKEND_AVAILABLE:
            raise RuntimeError(
                "Backend not available. Cannot use in-process adapter.\n"
                "Make sure monorepo/backend exists and is accessible.\n"
                "Use synth_gepa_adapter.py for HTTP API approach."
            )
        
        self.task_app_url = task_app_url
        self.task_app_id = task_app_id
        self.initial_prompt_messages = initial_prompt_messages
        self.rollout_budget = rollout_budget
        self.train_seeds = train_seeds or self._get_default_train_seeds()
        self.val_seeds = val_seeds  # Validation seeds (held-out set)
        
        # Learning curve tracker
        self.learning_curve = LearningCurveTracker(
            framework="synth_gepa",
            benchmark=task_app_id,
            total_budget=rollout_budget,
        )
        
        # Progress tracking
        self._progress_bar: Optional[tqdm] = None
        self._current_rollouts = 0
        self._best_score_so_far = 0.0

    def _get_default_train_seeds(self) -> list[int]:
        """Auto-scale number of seeds based on rollout budget."""
        if self.rollout_budget < 10:
            return list(range(5))  # Ultra-minimal: 5 seeds
        elif self.rollout_budget < 50:
            return list(range(10))  # Minimal: 10 seeds
        elif self.rollout_budget < 100:
            return list(range(20))  # Small: 20 seeds
        else:
            return list(range(100))  # Full: 100 seeds

    def _get_initial_population_size(self) -> int:
        """Auto-scale initial population size based on rollout budget."""
        if self.rollout_budget < 10:
            return 1
        elif self.rollout_budget < 50:
            return 2
        elif self.rollout_budget < 100:
            return 5
        elif self.rollout_budget < 200:
            return 10
        else:
            return 20

    def _get_num_generations(self) -> int:
        """Auto-scale number of generations based on rollout budget."""
        if self.rollout_budget < 10:
            return 1
        elif self.rollout_budget < 50:
            return 2
        elif self.rollout_budget < 100:
            return 5
        elif self.rollout_budget < 200:
            return 10
        else:
            return 15

    def _get_minibatch_size(self) -> int:
        """Auto-scale minibatch size based on rollout budget."""
        if self.rollout_budget < 10:
            return 1
        elif self.rollout_budget < 50:
            return 1
        elif self.rollout_budget < 100:
            return 2
        elif self.rollout_budget < 200:
            return 4
        else:
            return 8

    def _get_children_per_generation(self) -> int:
        """Auto-scale children per generation based on rollout budget."""
        if self.rollout_budget < 10:
            return 1
        elif self.rollout_budget < 50:
            return 1
        elif self.rollout_budget < 100:
            return 4
        elif self.rollout_budget < 200:
            return 8
        else:
            return 12

    async def optimize(self) -> dict[str, Any]:
        """Run optimization using in-process API and return results.

        Returns:
            Dictionary with best_prompt, best_score, learning_curve, etc.
        """
        if not BACKEND_AVAILABLE:
            raise RuntimeError("Backend not available")
        
        # Get API key - prioritize ENVIRONMENT_API_KEY for task app authentication
        api_key = os.getenv("ENVIRONMENT_API_KEY") or os.getenv("SYNTH_API_KEY", "")
        if not api_key:
            raise ValueError(
                "ENVIRONMENT_API_KEY or SYNTH_API_KEY required. "
                "Make sure .env file exists and contains one of these keys."
            )

        # Create prompt pattern from messages
        message_patterns = [
            MessagePattern(
                role=msg["role"],
                pattern=msg.get("pattern", msg.get("content", "")),
                order=i,
            )
            for i, msg in enumerate(self.initial_prompt_messages)
        ]
        initial_pattern = PromptPattern(messages=message_patterns)

        # Create GEPA config
        config = GEPAConfig(
            task_app_url=self.task_app_url,
            task_app_api_key=api_key,
            env_name=self.task_app_id,
            rollout_budget=self.rollout_budget,
            initial_population_size=self._get_initial_population_size(),
            num_generations=self._get_num_generations(),
            minibatch_size=self._get_minibatch_size(),
            children_per_generation=self._get_children_per_generation(),
            mutation_rate=0.3,
            crossover_rate=0.5,
            pareto_set_size=64,
            feedback_fraction=0.5,
            policy_config={
                "model": "openai/gpt-oss-20b",
                "provider": "groq",
                "temperature": 1.0,
                "max_completion_tokens": 512,
            },
            mutation_llm_model="openai/gpt-oss-20b",
            mutation_llm_provider="groq",
            initial_pattern=initial_pattern,
        )

        # Create runtime and optimizer
        runtime = LocalRuntime()
        optimizer = GEPAOptimizer(config=config, runtime=runtime)
        optimizer.job_id = f"{self.task_app_id}_gepa_blog_post"

        # Create progress bar
        self._progress_bar = tqdm(
            total=self.rollout_budget,
            desc=f"Optimizing ({self.task_app_id})",
            unit="rollout",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )

        try:
            # Run optimization with optional validation seeds
            best_template, best_score = await optimizer.optimize(
                initial_pattern=initial_pattern,
                train_seeds=self.train_seeds,
                train_pool=self.val_seeds,  # Validation seeds for held-out evaluation
            )

            # Record final checkpoint
            self.learning_curve.curve.record(
                rollout_count=self.rollout_budget,
                performance=best_score,
                checkpoint_pct=1.0,
            )

            # Update progress bar
            self._progress_bar.n = self.rollout_budget
            self._progress_bar.set_postfix_str(f"best={best_score:.4f}")
            self._progress_bar.refresh()

            # Store best prompt for saving
            best_prompt_dict = best_template.to_dict() if best_template else {}
            self._best_prompt = best_prompt_dict
            
            # Extract additional score information from archive entry
            best_entry = optimizer.archive.best_by("accuracy")
            train_scores = []
            val_score = None
            
            if best_entry:
                payload, score_dict = best_entry
                instance_scores = payload.get("instance_scores", [])
                train_scores = instance_scores if instance_scores else []
            
            # Run optimization with optional validation seeds
            # NOTE: The optimizer computes validation score internally when train_pool is provided,
            # but currently only returns (best_template, train_score). Validation score is not returned.
            # Add timeout: 10 seconds per rollout budget (e.g., 100 rollouts = 1000 seconds = ~16 minutes)
            timeout_seconds = max(300.0, self.rollout_budget * 10.0)  # Minimum 5 minutes
            best_template, best_score = await asyncio.wait_for(
                optimizer.optimize(
                    initial_pattern=initial_pattern,
                    train_seeds=self.train_seeds,
                    train_pool=self.val_seeds,  # Validation seeds for held-out evaluation
                ),
                timeout=timeout_seconds,
            )

            # Record final checkpoint
            self.learning_curve.curve.record(
                rollout_count=self.rollout_budget,
                performance=best_score,
                checkpoint_pct=1.0,
            )

            # Update progress bar
            self._progress_bar.n = self.rollout_budget
            self._progress_bar.set_postfix_str(f"best={best_score:.4f}")
            self._progress_bar.refresh()

            # Store best prompt for saving
            best_prompt_dict = best_template.to_dict() if best_template else {}
            self._best_prompt = best_prompt_dict
            
            # Extract score information from archive entry
            best_entry = optimizer.archive.best_by("accuracy")
            train_scores = []
            
            if best_entry:
                payload, score_dict = best_entry
                instance_scores = payload.get("instance_scores", [])
                train_scores = instance_scores if instance_scores else []
            
            # NOTE: The optimizer computes validation score internally when train_pool is provided,
            # but it doesn't return it. It only returns (best_template, train_score).
            # To get validation scores, we'd need to modify the optimizer to return a richer result object
            # (e.g., a GEPAResult dataclass instead of a tuple).
            # For now, val_score will be None.
            val_score = None
            
            # Create result object
            result = GEPAResult(
                best_template=best_template,
                train_score=best_score,
                train_scores=train_scores,
                val_score=val_score,
                job_id=optimizer.job_id,
                total_rollouts=self.rollout_budget,
            )

            # Convert result to dict with learning curve
            result_dict = result.to_dict(learning_curve_dict=self.learning_curve.curve.to_dict())
            result_dict["status"] = "completed"

            return result_dict
        except asyncio.TimeoutError:
            # Update progress bar with timeout
            if self._progress_bar:
                self._progress_bar.set_postfix_str(f"TIMEOUT after {timeout_seconds:.0f}s")
                self._progress_bar.refresh()
            raise RuntimeError(
                f"Optimization timed out after {timeout_seconds:.0f} seconds "
                f"(budget: {self.rollout_budget} rollouts)"
            ) from None
        except Exception as e:
            # Update progress bar with error
            if self._progress_bar:
                self._progress_bar.set_postfix_str(f"ERROR: {str(e)[:50]}")
                self._progress_bar.refresh()
            
            raise RuntimeError(f"Optimization failed: {e}") from e
        finally:
            if self._progress_bar:
                self._progress_bar.close()

    def save_results(self, output_dir: Path) -> None:
        """Save results to files.

        Args:
            output_dir: Directory to save results
        """
        import json
        
        output_dir.mkdir(parents=True, exist_ok=True)
        self.learning_curve.save(output_dir)
        
        # Also save the best prompt if available
        if hasattr(self, '_best_prompt') and self._best_prompt:
            prompt_file = output_dir / f"{self.learning_curve.benchmark}_best_prompt.json"
            with open(prompt_file, "w") as f:
                json.dump(self._best_prompt, f, indent=2)


# Iris-specific runner moved to task_specific/iris/synth_iris_adapter.py

