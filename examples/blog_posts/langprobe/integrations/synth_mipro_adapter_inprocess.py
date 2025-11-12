"""Synth MIPRO adapter using in-process Python API (no HTTP polling).

This adapter uses the direct MIPROOptimizer class from the backend,
avoiding HTTP polling and providing better control and progress tracking.
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from tqdm import tqdm

from .learning_curve_tracker import LearningCurveTracker

load_dotenv()


@dataclass
class MIPROResult:
    """Result from MIPRO optimization."""
    best_template: Any  # PromptTemplate
    train_score: float  # best_full_score or best_minibatch_score
    test_score: Optional[float] = None
    baseline_score: float = 0.0
    job_id: str = ""
    total_trials: int = 0
    
    def to_dict(self, learning_curve_dict: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "job_id": self.job_id,
            "best_score": self.train_score,  # For backwards compat
            "best_prompt": self.best_template.to_dict() if self.best_template else {},
            "train_score": self.train_score,
            "test_score": self.test_score,
            "baseline_score": self.baseline_score,
            "total_trials": self.total_trials,
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
    
    from app.routes.prompt_learning.algorithm.mipro.optimizer.optimizer import (
        MIPROOptimizer,
    )
    from app.routes.prompt_learning.algorithm.mipro.config import (
        MIPROConfig,
        MIPROSeedConfig,
        MIPROModuleConfig,
        MIPROStageConfig,
        MIPROMetaConfig,
    )
    from app.routes.prompt_learning.core.runtime import LocalRuntime
    
    BACKEND_AVAILABLE = True
except ImportError as e:
    BACKEND_AVAILABLE = False
    MIPROOptimizer = None  # type: ignore
    MIPROConfig = None  # type: ignore
    MIPROSeedConfig = None  # type: ignore
    MIPROModuleConfig = None  # type: ignore
    MIPROStageConfig = None  # type: ignore
    LocalRuntime = None  # type: ignore


class SynthMIPROAdapterInProcess:
    """Adapter for Synth MIPRO using in-process Python API."""
    
    _SIMPLE_API_EXCLUDE = {"hotpotqa", "hotpotqa_pipeline", "banking77_pipeline"}

    def __init__(
        self,
        task_app_url: str,
        task_app_id: str,
        initial_prompt_messages: list[dict[str, Any]],
        rollout_budget: int = 400,
        bootstrap_seeds: Optional[list[int]] = None,
        online_seeds: Optional[list[int]] = None,
        test_seeds: Optional[list[int]] = None,
    ):
        """Initialize Synth MIPRO adapter.

        Args:
            task_app_url: Task app URL (e.g., "http://127.0.0.1:8115")
            task_app_id: Task app ID (e.g., "iris")
            initial_prompt_messages: Initial prompt messages (for single-stage)
            rollout_budget: Total rollout budget (~400)
            bootstrap_seeds: Bootstrap seeds for few-shot examples (default: auto-scale)
            online_seeds: Online pool seeds for mini-batch evaluation (default: auto-scale)
            test_seeds: Test seeds for held-out evaluation (default: None)
        """
        if not BACKEND_AVAILABLE:
            raise RuntimeError(
                "Backend not available. Cannot use in-process adapter.\n"
                "Make sure monorepo/backend exists and is accessible."
            )
        
        self.task_app_url = task_app_url
        self.task_app_id = task_app_id
        self.initial_prompt_messages = initial_prompt_messages
        self.rollout_budget = rollout_budget
        
        # Auto-scale seeds based on budget
        self.bootstrap_seeds = bootstrap_seeds or self._get_default_bootstrap_seeds()
        self.online_seeds = online_seeds or self._get_default_online_seeds()
        self.test_seeds = test_seeds
        
        # Learning curve tracker
        self.learning_curve = LearningCurveTracker(
            framework="synth_mipro",
            benchmark=task_app_id,
            total_budget=rollout_budget,
        )
        
        # Progress tracking
        self._progress_bar: Optional[tqdm] = None
        self._best_prompt: Optional[dict[str, Any]] = None

    def _get_default_bootstrap_seeds(self) -> list[int]:
        """Auto-scale bootstrap seeds based on rollout budget."""
        if self.rollout_budget < 50:
            return list(range(5))  # 5 seeds
        elif self.rollout_budget < 100:
            return list(range(10))  # 10 seeds
        elif self.rollout_budget < 200:
            return list(range(20))  # 20 seeds
        else:
            return list(range(30))  # 30 seeds

    def _get_default_online_seeds(self) -> list[int]:
        """Auto-scale online seeds based on rollout budget."""
        max_bootstrap = max(self.bootstrap_seeds) if self.bootstrap_seeds else 29
        if self.rollout_budget < 50:
            return list(range(max_bootstrap + 1, max_bootstrap + 11))  # 10 seeds
        elif self.rollout_budget < 100:
            return list(range(max_bootstrap + 1, max_bootstrap + 21))  # 20 seeds
        elif self.rollout_budget < 200:
            return list(range(max_bootstrap + 1, max_bootstrap + 31))  # 30 seeds
        else:
            return list(range(max_bootstrap + 1, max_bootstrap + 51))  # 50 seeds

    def _get_num_iterations(self) -> int:
        """Auto-scale iterations based on rollout budget."""
        if self.rollout_budget < 50:
            return 3  # Very minimal
        elif self.rollout_budget < 100:
            return 5
        elif self.rollout_budget < 200:
            return 10
        else:
            return 20

    def _get_num_evaluations_per_iteration(self) -> int:
        """Auto-scale evaluations per iteration based on rollout budget."""
        if self.rollout_budget < 50:
            return 2  # Very minimal
        elif self.rollout_budget < 100:
            return 3
        elif self.rollout_budget < 200:
            return 4
        else:
            return 5

    async def optimize(self) -> dict[str, Any]:
        """Run optimization using in-process API and return results.

        Returns:
            Dictionary with best_prompt, best_score, learning_curve, etc.
        """
        if not BACKEND_AVAILABLE:
            raise RuntimeError("Backend not available")
        
        # Get API key
        api_key = os.getenv("ENVIRONMENT_API_KEY") or os.getenv("SYNTH_API_KEY", "")
        if not api_key:
            raise ValueError(
                "ENVIRONMENT_API_KEY or SYNTH_API_KEY required. "
                "Make sure .env file exists and contains one of these keys."
            )

        # Build config via simple API when available, otherwise fall back to legacy constructor
        baseline_messages = self._normalize_initial_prompt_messages()
        config = self._build_mipro_config(api_key=api_key, baseline_messages=baseline_messages)

        # Create runtime and optimizer
        runtime = LocalRuntime()
        optimizer = MIPROOptimizer(
            job_id=f"{self.task_app_id}_mipro_blog_post",
            config=config,
            runtime=runtime,
            initial_prompt_config={
                "messages": self.initial_prompt_messages,
            },
        )

        # Create progress bar
        self._progress_bar = tqdm(
            total=self.rollout_budget,
            desc=f"Optimizing ({self.task_app_id})",
            unit="trial",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )

        try:
            # Add timeout: 10 seconds per rollout budget
            timeout_seconds = max(300.0, self.rollout_budget * 10.0)  # Minimum 5 minutes
            
            # Run optimization
            result = await asyncio.wait_for(
                optimizer.optimize(),
                timeout=timeout_seconds,
            )

            # Extract results
            best_template = optimizer.build_prompt_template(result.best_candidate)
            best_score = result.best_full_score or result.best_minibatch_score
            test_score = result.test_score
            baseline_result = optimizer.bootstrap_result()
            baseline_score = baseline_result.baseline_score if baseline_result else 0.0

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
            
            # Create result object
            mipro_result = MIPROResult(
                best_template=best_template,
                train_score=best_score,
                test_score=test_score,
                baseline_score=baseline_score,
                job_id=optimizer.job_id,
                total_trials=result.total_trials,
            )

            # Convert result to dict with learning curve
            result_dict = mipro_result.to_dict(learning_curve_dict=self.learning_curve.curve.to_dict())
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

    def _normalize_initial_prompt_messages(self) -> list[dict[str, str]]:
        """Normalize prompt messages to the {role, content} structure."""
        normalized: list[dict[str, str]] = []
        for msg in self.initial_prompt_messages or []:
            role = msg.get("role", "user")
            content = msg.get("content") or msg.get("pattern") or ""
            normalized.append({"role": role, "content": str(content)})
        return normalized

    def _build_mipro_config(self, *, api_key: str, baseline_messages: list[dict[str, str]]):
        """Build a MIPRO config, preferring the simple helper when available."""
        if self._should_use_simple_api() and hasattr(MIPROConfig, "simple"):
            try:
                return self._build_simple_mipro_config(api_key=api_key, baseline_messages=baseline_messages)
            except Exception as exc:  # pragma: no cover - defensive fallback
                print(f"[SYNTH_MIPRO] Falling back to legacy config builder: {exc}", flush=True)
        return self._build_legacy_mipro_config(api_key=api_key, baseline_messages=baseline_messages)

    def _build_simple_mipro_config(self, *, api_key: str, baseline_messages: list[dict[str, str]]):
        """Create config via MIPROConfig.simple for single-stage tasks."""
        return MIPROConfig.simple(  # type: ignore[attr-defined]
            task_app_url=self.task_app_url,
            task_app_api_key=api_key,
            task_app_id=self.task_app_id,
            env_name=self.task_app_id,
            rollout_budget=self.rollout_budget,
            initial_prompt_messages=baseline_messages,
            bootstrap_seeds=self.bootstrap_seeds,
            online_seeds=self.online_seeds,
            test_seeds=self.test_seeds or [],
            num_iterations=self._get_num_iterations(),
            num_evaluations_per_iteration=self._get_num_evaluations_per_iteration(),
            batch_size=max(1, min(32, len(self.online_seeds))),
            policy_model="openai/gpt-oss-20b",
            policy_provider="groq",
            policy_temperature=1.0,
            policy_max_completion_tokens=512,
            meta_model="gpt-4o-mini",
            meta_provider="openai",
        )

    def _build_legacy_mipro_config(self, *, api_key: str, baseline_messages: list[dict[str, str]]):
        """Legacy config builder that mirrors the previous manual constructor."""
        baseline_instruction = "Classify the input."
        for msg in baseline_messages:
            if msg.get("role") == "system" and msg.get("content"):
                baseline_instruction = msg["content"]
                break
        return MIPROConfig(
            task_app_url=self.task_app_url,
            task_app_api_key=api_key,
            env_name=self.task_app_id,
            seeds=MIPROSeedConfig(
                bootstrap=self.bootstrap_seeds,
                online=self.online_seeds,
                test=self.test_seeds or [],
            ),
            num_iterations=self._get_num_iterations(),
            num_evaluations_per_iteration=self._get_num_evaluations_per_iteration(),
            batch_size=max(1, min(32, len(self.online_seeds))),
            max_concurrent=10,
            policy_config={
                "model": "openai/gpt-oss-20b",
                "provider": "groq",
                "temperature": 1.0,
                "max_completion_tokens": 512,
            },
            meta=MIPROMetaConfig(
                model="gpt-4o-mini",
                provider="openai",
                inference_url=None,
            ),
            modules=[
                MIPROModuleConfig(
                    module_id="classifier",
                    stages=[
                        MIPROStageConfig(
                            stage_id="classifier_stage_0",
                            baseline_instruction=baseline_instruction,
                        )
                    ],
                )
            ],
        )

    def _should_use_simple_api(self) -> bool:
        """Whether the adapter should use the simple single-stage API."""
        if not hasattr(MIPROConfig, "simple"):
            return False
        if not self.task_app_id:
            return True
        return self.task_app_id not in self._SIMPLE_API_EXCLUDE

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
