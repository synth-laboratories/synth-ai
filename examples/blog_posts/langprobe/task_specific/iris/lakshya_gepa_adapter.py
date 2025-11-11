"""Lakshya Agrawal's GEPA adapter for Iris classification."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from gepa import optimize
from gepa.core.adapter import GEPAAdapter, EvaluationBatch
from tqdm import tqdm

from ...integrations.learning_curve_tracker import LearningCurveTracker
from ...integrations.task_app_client import TaskAppClient

load_dotenv()


class IrisGEPAAdapter(GEPAAdapter):
    """GEPA adapter for Iris classification using task app."""

    def __init__(
        self,
        task_app_url: str,
        task_app_id: str,
        model: str = "groq/llama-3.3-70b-versatile",
        provider: str = "groq",
        api_key: Optional[str] = None,
    ):
        """Initialize Iris GEPA adapter.

        Args:
            task_app_url: Task app URL (e.g., "http://127.0.0.1:8115")
            task_app_id: Task app ID (e.g., "iris")
            model: Model identifier
            provider: Provider name
            api_key: API key for task app (defaults to ENVIRONMENT_API_KEY)
        """
        self.task_app_url = task_app_url
        self.task_app_id = task_app_id
        self.model = model
        self.provider = provider
        self.api_key = api_key or os.getenv("ENVIRONMENT_API_KEY", "")
        self.client: Optional[TaskAppClient] = None

    def evaluate(
        self,
        batch: list[Any],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        """Evaluate a candidate prompt on a batch of examples.

        Args:
            batch: List of example objects (DataInst) with seed/features
            candidate: Dictionary with prompt components (dict[str, str])
            capture_traces: Whether to capture detailed traces

        Returns:
            EvaluationBatch with scores and traces
        """
        # Create client if needed
        if self.client is None:
            self.client = TaskAppClient(self.task_app_url, self.api_key)

        # Extract prompt components
        system_prompt = candidate.get("system", 
            "You are a botany classification assistant. Based on the flower's measurements, "
            "classify the iris species. Respond with one of: setosa, versicolor, or virginica.")
        
        user_template = candidate.get("user", 
            "Flower Measurements:\n{features}\n\nClassify this iris flower. Respond with one of: setosa, versicolor, or virginica.")

        # Evaluate synchronously (GEPA expects sync, but TaskAppClient is async)
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        results = loop.run_until_complete(self._evaluate_async(batch, system_prompt, user_template))

        # Convert to EvaluationBatch format
        # EvaluationBatch expects outputs (any type), scores (list[float]), and optional trajectories
        outputs = [result.get("trace", {}) for result in results]  # Raw outputs
        scores = [result["score"] for result in results]  # Required scores list
        trajectories = [result.get("trace", {}) for result in results] if capture_traces else None
        
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    async def _evaluate_async(
        self,
        batch: list[Any],
        system_prompt: str,
        user_template: str,
    ) -> list[dict[str, Any]]:
        """Async helper for evaluation."""
        results = []
        
        for example in batch:
            # Extract seed and features from example
            if isinstance(example, dict):
                seed = example.get("seed", example.get("index", 0))
                features = example.get("features", "")
                label = example.get("label", "")
            else:
                # DataInst object
                seed = getattr(example, "seed", getattr(example, "index", 0))
                features = getattr(example, "features", "")
                label = getattr(example, "label", "")
            
            # Format user message
            user_message = user_template.replace("{features}", str(features))
            
            # Create messages
            eval_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            try:
                response = await self.client.evaluate_prompt(
                    prompt_messages=eval_messages,
                    seed=int(seed),
                    task_app_id=self.task_app_id,
                    model=self.model,
                    provider=self.provider,
                )

                # Extract score
                metrics = response.get("metrics", {})
                score = metrics.get("outcome_score", 0.0)
                
                # Extract trace
                trace = {
                    "seed": seed,
                    "response": response.get("trajectory", {}),
                    "metrics": metrics,
                    "predicted": response.get("trajectory", {}).get("steps", [{}])[-1].get("action", {}).get("text", ""),
                    "expected": label,
                }

                results.append({
                    "score": float(score),
                    "trace": trace,
                })
            except Exception as e:
                results.append({
                    "score": 0.0,
                    "trace": {"error": str(e), "seed": seed},
                })

        return results

    def extract_traces_for_reflection(
        self,
        traces: list[dict[str, Any]],
        component_name: str,
    ) -> str:
        """Extract relevant information from traces for reflection.

        Args:
            traces: List of trace dictionaries from evaluate()
            component_name: Name of the component being optimized (e.g., "system", "user")

        Returns:
            Textual content relevant to the component for reflection
        """
        reflection_texts = []
        
        for trace in traces:
            if "error" in trace:
                reflection_texts.append(f"Error on seed {trace.get('seed', 'unknown')}: {trace['error']}")
                continue
            
            seed = trace.get("seed", "unknown")
            score = trace.get("score", 0.0)
            predicted = trace.get("predicted", "")
            expected = trace.get("expected", "")
            
            # Extract response text from trajectory
            response_text = ""
            trajectory = trace.get("response", {})
            steps = trajectory.get("steps", [])
            if steps:
                last_step = steps[-1]
                action = last_step.get("action", {})
                response_text = action.get("text", "")
            
            # Build reflection text
            if score > 0.5:
                reflection_texts.append(
                    f"Seed {seed}: Correct prediction. Predicted '{predicted}', expected '{expected}'. "
                    f"Response: {response_text[:200]}"
                )
            else:
                reflection_texts.append(
                    f"Seed {seed}: Incorrect prediction. Predicted '{predicted}', expected '{expected}'. "
                    f"Response: {response_text[:200]}"
                )
        
        return "\n".join(reflection_texts)


async def run_lakshya_gepa_iris(
    task_app_url: str = "http://127.0.0.1:8115",
    train_seeds: Optional[list[int]] = None,
    val_seeds: Optional[list[int]] = None,
    rollout_budget: int = 400,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Run Lakshya's GEPA optimization on Iris.

    Args:
        task_app_url: Task app URL
        train_seeds: Training seeds (default: 0-99)
        val_seeds: Validation seeds (default: 100-149)
        rollout_budget: Rollout budget (default: 400)
        output_dir: Output directory

    Returns:
        Results dictionary
    """
    from datasets import load_dataset

    if output_dir is None:
        output_dir = Path(__file__).parent / "results" / "lakshya_gepa"
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = load_dataset("scikit-learn/iris", split="train")
    iris_examples = []
    
    # Get label names
    label_names = dataset.features.get("label", dataset.features.get("target", None))
    if label_names:
        label_names = label_names.names if hasattr(label_names, "names") else ["setosa", "versicolor", "virginica"]
    else:
        label_names = ["setosa", "versicolor", "virginica"]
    
    for i, row in enumerate(dataset):
        features = {}
        label_idx = None
        label_name = None
        
        for key, value in row.items():
            if key in ("label", "target"):
                label_idx = int(value) if isinstance(value, (int, str)) else 0
                if label_names and 0 <= label_idx < len(label_names):
                    label_name = label_names[label_idx]
                else:
                    label_name = str(value)
            elif key not in ("species", "class"):
                features[key] = value
        
        # Format features as text
        feature_text = "\n".join([f"{key}: {value}" for key, value in features.items()])
        
        iris_examples.append({
            "seed": i,
            "index": i,
            "features": feature_text,
            "label": label_name or "setosa",
        })

    # Select training and validation seeds
    if train_seeds is None:
        train_seeds = list(range(min(100, len(iris_examples))))
    if val_seeds is None:
        max_train = max(train_seeds) if train_seeds else 99
        val_seeds = list(range(max_train + 1, min(max_train + 51, len(iris_examples))))

    # Filter examples by seeds
    train_examples = [iris_examples[i] for i in train_seeds if i < len(iris_examples)]
    val_examples = [iris_examples[i] for i in val_seeds if i < len(iris_examples)]

    # Create adapter
    adapter = IrisGEPAAdapter(
        task_app_url=task_app_url,
        task_app_id="iris",
        model="groq/llama-3.3-70b-versatile",
        provider="groq",
    )

    # Initial prompt candidate (dict[str, str] format)
    initial_candidate = {
        "system": (
            "You are a botany classification assistant. Based on the flower's measurements, "
            "classify the iris species. Respond with one of: setosa, versicolor, or virginica."
        ),
        "user": (
            "Flower Measurements:\n{features}\n\n"
            "Classify this iris flower. Respond with one of: setosa, versicolor, or virginica."
        ),
    }

    # Learning curve tracker
    learning_curve = LearningCurveTracker(
        framework="lakshya_gepa",
        benchmark="iris",
        total_budget=rollout_budget,
    )

    print(f"🚀 Lakshya GEPA (budget={rollout_budget})")

    # Configure GEPA
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY required")

    # Run optimization (gepa.optimize is async-compatible)
    try:
        # Note: gepa.optimize may be synchronous, so we'll run it in executor if needed
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: optimize(
                seed_candidate=initial_candidate,
                trainset=train_examples,
                valset=val_examples,
                adapter=adapter,
                reflection_lm=f"groq/llama-3.3-70b-versatile",  # Can be string or LM object
                max_metric_calls=rollout_budget,
                display_progress_bar=True,
            )
        )

        # Extract best candidate and score from GEPAResult
        best_candidate = result.best_candidate if hasattr(result, "best_candidate") else initial_candidate
        best_score = result.best_score if hasattr(result, "best_score") else 0.0
        val_score = getattr(result, "val_score", None)

        # Record final checkpoint
        learning_curve.curve.record(
            rollout_count=rollout_budget,
            performance=float(best_score),
            checkpoint_pct=1.0,
        )

        # Save results
        learning_curve.save(output_dir)

        results = {
            "best_score": float(best_score),
            "val_score": float(val_score) if val_score is not None else None,
            "total_rollouts": rollout_budget,
            "best_candidate": best_candidate if isinstance(best_candidate, dict) else dict(best_candidate),
        }

        # Save prompt details
        best_candidate_dict = best_candidate if isinstance(best_candidate, dict) else dict(best_candidate)
        with open(output_dir / "iris_best_prompt.json", "w") as f:
            json.dump(best_candidate_dict, f, indent=2)

        # Save human-readable prompt
        with open(output_dir / "optimized_prompt.txt", "w") as f:
            f.write("=" * 80 + "\n")
            f.write("Lakshya GEPA Optimized Prompt\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"System Prompt:\n{best_candidate_dict.get('system', 'N/A')}\n\n")
            f.write(f"User Template:\n{best_candidate_dict.get('user', 'N/A')}\n\n")
            f.write(f"Best Score: {best_score:.4f}\n")
            if val_score is not None:
                f.write(f"Val Score: {val_score:.4f}\n")

        results["prompt_file"] = str(output_dir / "optimized_prompt.txt")

        return results

    finally:
        if adapter.client:
            await adapter.client.close()

