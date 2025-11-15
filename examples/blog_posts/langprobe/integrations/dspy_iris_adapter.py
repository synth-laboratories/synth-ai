"""DSPy adapters for Iris classification using MIPROv2 and GEPA optimizers."""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

import dspy
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

from .learning_curve_tracker import LearningCurveTracker

load_dotenv()


class IrisClassification(dspy.Signature):
    """Classify iris flowers based on sepal and petal dimensions.
    
    Given the measurements of an iris flower, predict its species.
    """
    
    features: str = dspy.InputField(
        desc="Flower measurements: sepal_length, sepal_width, petal_length, petal_width"
    )
    species: str = dspy.OutputField(
        desc="Predicted species: one of setosa, versicolor, or virginica"
    )


class IrisClassifier(dspy.Module):
    """DSPy module for Iris classification."""
    
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(IrisClassification)
    
    def forward(self, features: str) -> dspy.Prediction:
        """Classify iris flower from features string.
        
        Args:
            features: Feature text (e.g., "sepal_length: 5.1\nsepal_width: 3.5\n...")
            
        Returns:
            Prediction with species field
        """
        result = self.predict(features=features)
        return result


def load_iris_dataset(split: str = "train") -> list[dict[str, Any]]:
    """Load Iris dataset from HuggingFace and convert to list of dicts.
    
    Args:
        split: Dataset split (default: "train")
        
    Returns:
        List of examples with features and label
    """
    dataset = load_dataset("scikit-learn/iris", split=split)
    examples = []
    
    # Get label names
    label_names = None
    if "label" in dataset.features:
        label_names = dataset.features["label"].names
    elif "target" in dataset.features:
        label_names = dataset.features["target"].names
    else:
        label_names = ["setosa", "versicolor", "virginica"]
    
    for row in dataset:
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
        
        # Format features as text (matching task app format)
        feature_text = "\n".join([f"{key}: {value}" for key, value in features.items()])
        
        examples.append({
            "features": feature_text,
            "label": label_name or "setosa",
            "label_idx": label_idx or 0,
            "seed": len(examples),  # Use index as seed
        })
    
    return examples


def create_dspy_examples(iris_examples: list[dict[str, Any]]) -> list[dspy.Example]:
    """Convert Iris examples to DSPy Examples.
    
    Args:
        iris_examples: List of Iris example dicts
        
    Returns:
        List of DSPy Examples
    """
    dspy_examples = []
    for ex in iris_examples:
        dspy_ex = dspy.Example(
            features=ex["features"],
            species=ex["label"],
        ).with_inputs("features")
        dspy_examples.append(dspy_ex)
    return dspy_examples


# --- helper: sanitize .env warning lines (optional, non-fatal) ---
def _warn_if_dotenv_is_messy():
    """Warn if .env file has non-standard lines (optional, non-fatal)."""
    p = Path(".env")
    if p.exists():
        bad = [
            i
            for i, line in enumerate(p.read_text().splitlines(), 1)
            if line
            and not line.lstrip().startswith("#")
            and not re.match(r"^[A-Za-z_][A-Za-z0-9_]*=.+$", line)
        ]
        if bad:
            print(f"[dotenv] Non KEY=VALUE lines at: {bad}  (ignored)")


def iris_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """Metric function for Iris classification.
    
    Args:
        example: Ground truth example
        pred: Model prediction
        trace: Optional trace (unused)
        
    Returns:
        Score (1.0 if correct, 0.0 if incorrect)
    """
    predicted = (pred.species or "").lower().strip()
    expected = (example.species or "").lower().strip()
    
    # Normalize predictions
    if "setosa" in predicted:
        predicted = "setosa"
    elif "versicolor" in predicted:
        predicted = "versicolor"
    elif "virginica" in predicted:
        predicted = "virginica"
    
    return 1.0 if predicted == expected else 0.0


def iris_metric_gepa(gold: dspy.Example, pred: dspy.Prediction, trace=None, pred_name=None, pred_trace=None) -> float:
    """GEPA-compatible metric function (5 arguments required).
    
    Args:
        gold: Ground truth example
        pred: Model prediction
        trace: Optional trace
        pred_name: Optional prediction name
        pred_trace: Optional prediction trace
        
    Returns:
        Score (1.0 for correct, 0.0 for incorrect)
    """
    return iris_metric(gold, pred, trace)


async def run_dspy_miprov2_iris(
    task_app_url: str = "http://127.0.0.1:8115",
    train_seeds: Optional[list[int]] = None,
    val_seeds: Optional[list[int]] = None,
    rollout_budget: int = 400,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Run DSPy MIPROv2 optimization on Iris.
    
    Args:
        task_app_url: Task app URL (for reference, not used directly)
        train_seeds: Training seeds (default: 0-99, limited by dataset size)
        val_seeds: Validation seeds (default: 100-149, limited by dataset size)
        rollout_budget: Rollout budget (default: 400)
        output_dir: Output directory
        
    Returns:
        Results dictionary
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results" / "dspy_mipro"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    _warn_if_dotenv_is_messy()
    
    # Configure DSPy LM
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY required")
    
    # Groq via LiteLLM (current, non-deprecated)
    lm = dspy.LM("groq/llama-3.3-70b-versatile", api_key=groq_api_key)
    dspy.configure(lm=lm)
    
    # Load dataset
    iris_examples = load_iris_dataset(split="train")
    
    # Select training and validation seeds
    if train_seeds is None:
        train_seeds = list(range(min(100, len(iris_examples))))
    if val_seeds is None:
        max_train = max(train_seeds) if train_seeds else 99
        val_seeds = list(range(max_train + 1, min(max_train + 51, len(iris_examples))))
    
    # Filter examples by seeds
    train_examples = [iris_examples[i] for i in train_seeds if i < len(iris_examples)]
    val_examples = [iris_examples[i] for i in val_seeds if i < len(iris_examples)]
    
    # Convert to DSPy Examples
    trainset = create_dspy_examples(train_examples)
    valset = create_dspy_examples(val_examples)
    
    # Create module
    module = IrisClassifier()
    
    # Initialize MIPROv2 optimizer
    from dspy.teleprompt import MIPROv2
    
    # Auto-scale based on budget
    auto_level = "light" if rollout_budget < 100 else ("medium" if rollout_budget < 200 else "heavy")
    # NOTE: When auto is set, do NOT pass num_candidates/num_trials.
    optimizer = MIPROv2(metric=iris_metric, auto=auto_level)
    
    # Learning curve tracker
    learning_curve = LearningCurveTracker(
        framework="dspy_miprov2",
        benchmark="iris",
        total_budget=rollout_budget,
    )
    
    # Optimize with progress tracking
    print(f"🚀 DSPy MIPROv2 (budget={rollout_budget}, auto={auto_level})")
    
    optimized_module = optimizer.compile(student=module, trainset=trainset, valset=valset)
    
    # Evaluate optimized module
    from dspy.evaluate import Evaluate
    
    evaluate = Evaluate(devset=trainset[:10], metric=iris_metric, num_threads=1)
    train_score = evaluate(optimized_module)
    val_score = evaluate(optimized_module, devset=valset[:10]) if valset else None
    
    # Record final checkpoint
    learning_curve.curve.record(
        rollout_count=rollout_budget,
        performance=train_score if isinstance(train_score, (int, float)) else train_score.get("accuracy", 0.0),
        checkpoint_pct=1.0,
    )
    
    # Save results
    learning_curve.save(output_dir)
    
    results = {
        "best_score": float(train_score) if isinstance(train_score, (int, float)) else float(train_score.get("accuracy", 0.0)),
        "val_score": float(val_score) if val_score is not None and isinstance(val_score, (int, float)) else (float(val_score.get("accuracy")) if val_score and isinstance(val_score, dict) else None),
        "total_rollouts": rollout_budget,
        "auto_level": auto_level,
    }
    
    # Save module info with full prompt details
    module_info = {
        "instructions": str(getattr(optimized_module.predict, "instructions", None)),
        "demos": len(getattr(optimized_module.predict, "demos", [])),
    }
    
    # Extract full prompt details for review
    prompt_details = {}
    if hasattr(optimized_module.predict, "instructions"):
        prompt_details["instructions"] = str(optimized_module.predict.instructions)
    if hasattr(optimized_module.predict, "demos"):
        demo_list = optimized_module.predict.demos
        prompt_details["num_demos"] = len(demo_list)
        prompt_details["demo_examples"] = []
        for i, demo in enumerate(demo_list[:5]):  # Save first 5 demos
            demo_dict = {}
            if hasattr(demo, "features"):
                demo_dict["input"] = str(demo.features)
            if hasattr(demo, "species"):
                demo_dict["output"] = str(demo.species)
            if hasattr(demo, "rationale"):
                demo_dict["rationale"] = str(demo.rationale)
            prompt_details["demo_examples"].append(demo_dict)
    
    # Try to get the full prompt text by running a forward pass
    try:
        sample_input = trainset[0].features if trainset and len(trainset) > 0 else "sepal_length: 5.1\nsepal_width: 3.5\npetal_length: 1.4\npetal_width: 0.2"
        sample_pred = optimized_module(sample_input)
        prompt_details["sample_prediction"] = {
            "input": sample_input,
            "output": str(sample_pred.species) if hasattr(sample_pred, "species") else str(sample_pred),
            "rationale": str(sample_pred.rationale) if hasattr(sample_pred, "rationale") else None,
        }
    except Exception as e:
        prompt_details["sample_prediction_error"] = str(e)
    
    module_info["prompt_details"] = prompt_details
    
    with open(output_dir / "iris_best_module.json", "w") as f:
        json.dump(module_info, f, indent=2)
    
    # Also save a human-readable prompt file
    framework_name = "DSPy MIPROv2" if "mipro" in str(output_dir).lower() else "DSPy GEPA"
    with open(output_dir / "optimized_prompt.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"{framework_name} Optimized Prompt\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Instructions:\n{prompt_details.get('instructions', 'None')}\n\n")
        f.write(f"Number of Few-Shot Examples: {prompt_details.get('num_demos', 0)}\n\n")
        if prompt_details.get("demo_examples"):
            f.write("Few-Shot Examples:\n")
            for i, demo in enumerate(prompt_details["demo_examples"], 1):
                f.write(f"\nExample {i}:\n")
                f.write(f"  Input: {demo.get('input', 'N/A')}\n")
                f.write(f"  Output: {demo.get('output', 'N/A')}\n")
                if demo.get("rationale"):
                    f.write(f"  Rationale: {demo.get('rationale')}\n")
        if prompt_details.get("sample_prediction"):
            f.write("\n" + "=" * 80 + "\n")
            f.write("Sample Prediction:\n")
            f.write("=" * 80 + "\n")
            sp = prompt_details["sample_prediction"]
            f.write(f"Input: {sp.get('input')}\n")
            f.write(f"Output: {sp.get('output')}\n")
            if sp.get("rationale"):
                f.write(f"Rationale: {sp.get('rationale')}\n")
    
    results["prompt_file"] = str(output_dir / "optimized_prompt.txt")


async def run_dspy_gepa_iris(
    task_app_url: str = "http://127.0.0.1:8115",
    train_seeds: Optional[list[int]] = None,
    val_seeds: Optional[list[int]] = None,
    rollout_budget: int = 400,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Run DSPy GEPA optimization on Iris.
    
    Args:
        task_app_url: Task app URL (for reference, not used directly)
        train_seeds: Training seeds (default: 0-99, limited by dataset size)
        val_seeds: Validation seeds (default: 100-149, limited by dataset size)
        rollout_budget: Rollout budget (default: 400)
        output_dir: Output directory
        
    Returns:
        Results dictionary
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results" / "dspy_gepa"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    _warn_if_dotenv_is_messy()
    
    # Configure DSPy LM
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY required")
    
    lm = dspy.LM("groq/llama-3.3-70b-versatile", api_key=groq_api_key)
    dspy.configure(lm=lm)
    
    # Load dataset
    iris_examples = load_iris_dataset(split="train")
    
    # Select training and validation seeds
    if train_seeds is None:
        train_seeds = list(range(min(100, len(iris_examples))))
    if val_seeds is None:
        max_train = max(train_seeds) if train_seeds else 99
        val_seeds = list(range(max_train + 1, min(max_train + 51, len(iris_examples))))
    
    # Filter examples by seeds
    train_examples = [iris_examples[i] for i in train_seeds if i < len(iris_examples)]
    val_examples = [iris_examples[i] for i in val_seeds if i < len(iris_examples)]
    
    # Convert to DSPy Examples
    trainset = create_dspy_examples(train_examples)
    valset = create_dspy_examples(val_examples)
    
    # Create module
    module = IrisClassifier()
    
    # Initialize GEPA optimizer
    from dspy.teleprompt.gepa import GEPA
    
    max_metric_calls = int(rollout_budget)
    # GEPA requires a reflection LM for proposing new instructions
    reflection_lm = dspy.LM("groq/llama-3.3-70b-versatile", api_key=groq_api_key)
    optimizer = GEPA(metric=iris_metric_gepa, max_metric_calls=max_metric_calls, reflection_lm=reflection_lm)
    
    # Learning curve tracker
    learning_curve = LearningCurveTracker(
        framework="dspy_gepa",
        benchmark="iris",
        total_budget=rollout_budget,
    )
    
    # Optimize with progress tracking
    print(f"🚀 DSPy GEPA (budget={rollout_budget}, max_metric_calls={max_metric_calls})")
    
    optimized_module = optimizer.compile(student=module, trainset=trainset, valset=valset)
    
    # Evaluate optimized module
    from dspy.evaluate import Evaluate
    
    evaluate = Evaluate(devset=trainset[:10], metric=iris_metric, num_threads=1)
    train_score = evaluate(optimized_module)
    val_score = evaluate(optimized_module, devset=valset[:10]) if valset else None
    
    # Record final checkpoint
    learning_curve.curve.record(
        rollout_count=rollout_budget,
        performance=train_score if isinstance(train_score, (int, float)) else train_score.get("accuracy", 0.0),
        checkpoint_pct=1.0,
    )
    
    # Save results
    learning_curve.save(output_dir)
    
    results = {
        "best_score": float(train_score) if isinstance(train_score, (int, float)) else float(train_score.get("accuracy", 0.0)),
        "val_score": float(val_score) if val_score is not None and isinstance(val_score, (int, float)) else (float(val_score.get("accuracy")) if val_score and isinstance(val_score, dict) else None),
        "total_rollouts": rollout_budget,
        "max_metric_calls": max_metric_calls,
    }
    
    # Save module info with full prompt details
    module_info = {
        "instructions": str(getattr(optimized_module.predict, "instructions", None)),
        "demos": len(getattr(optimized_module.predict, "demos", [])),
    }
    
    # Extract full prompt details for review
    prompt_details = {}
    if hasattr(optimized_module.predict, "instructions"):
        prompt_details["instructions"] = str(optimized_module.predict.instructions)
    if hasattr(optimized_module.predict, "demos"):
        demo_list = optimized_module.predict.demos
        prompt_details["num_demos"] = len(demo_list)
        prompt_details["demo_examples"] = []
        for i, demo in enumerate(demo_list[:5]):  # Save first 5 demos
            demo_dict = {}
            if hasattr(demo, "features"):
                demo_dict["input"] = str(demo.features)
            if hasattr(demo, "species"):
                demo_dict["output"] = str(demo.species)
            if hasattr(demo, "rationale"):
                demo_dict["rationale"] = str(demo.rationale)
            prompt_details["demo_examples"].append(demo_dict)
    
    # Try to get the full prompt text by running a forward pass
    try:
        sample_input = trainset[0].features if trainset and len(trainset) > 0 else "sepal_length: 5.1\nsepal_width: 3.5\npetal_length: 1.4\npetal_width: 0.2"
        sample_pred = optimized_module(sample_input)
        prompt_details["sample_prediction"] = {
            "input": sample_input,
            "output": str(sample_pred.species) if hasattr(sample_pred, "species") else str(sample_pred),
            "rationale": str(sample_pred.rationale) if hasattr(sample_pred, "rationale") else None,
        }
    except Exception as e:
        prompt_details["sample_prediction_error"] = str(e)
    
    module_info["prompt_details"] = prompt_details
    
    with open(output_dir / "iris_best_module.json", "w") as f:
        json.dump(module_info, f, indent=2)
    
    # Also save a human-readable prompt file
    framework_name = "DSPy MIPROv2" if "mipro" in str(output_dir).lower() else "DSPy GEPA"
    with open(output_dir / "optimized_prompt.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"{framework_name} Optimized Prompt\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Instructions:\n{prompt_details.get('instructions', 'None')}\n\n")
        f.write(f"Number of Few-Shot Examples: {prompt_details.get('num_demos', 0)}\n\n")
        if prompt_details.get("demo_examples"):
            f.write("Few-Shot Examples:\n")
            for i, demo in enumerate(prompt_details["demo_examples"], 1):
                f.write(f"\nExample {i}:\n")
                f.write(f"  Input: {demo.get('input', 'N/A')}\n")
                f.write(f"  Output: {demo.get('output', 'N/A')}\n")
                if demo.get("rationale"):
                    f.write(f"  Rationale: {demo.get('rationale')}\n")
        if prompt_details.get("sample_prediction"):
            f.write("\n" + "=" * 80 + "\n")
            f.write("Sample Prediction:\n")
            f.write("=" * 80 + "\n")
            sp = prompt_details["sample_prediction"]
            f.write(f"Input: {sp.get('input')}\n")
            f.write(f"Output: {sp.get('output')}\n")
            if sp.get("rationale"):
                f.write(f"Rationale: {sp.get('rationale')}\n")
    
    results["prompt_file"] = str(output_dir / "optimized_prompt.txt")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run DSPy optimizers on Iris")
    parser.add_argument(
        "--optimizer",
        choices=["miprov2", "gepa"],
        required=True,
        help="Optimizer to use",
    )
    parser.add_argument(
        "--task-app-url",
        default="http://127.0.0.1:8115",
        help="Task app URL (for reference)",
    )
    parser.add_argument(
        "--rollout-budget",
        type=int,
        default=400,
        help="Rollout budget",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    if args.optimizer == "miprov2":
        results = asyncio.run(
            run_dspy_miprov2_iris(
                task_app_url=args.task_app_url,
                rollout_budget=args.rollout_budget,
                output_dir=args.output_dir,
            )
        )
    elif args.optimizer == "gepa":
        results = asyncio.run(
            run_dspy_gepa_iris(
                task_app_url=args.task_app_url,
                rollout_budget=args.rollout_budget,
                output_dir=args.output_dir,
            )
        )
    
    print(f"\n✅ Optimization complete!")
    print(f"   Best score: {results['best_score']:.4f}")
    print(f"   Val score: {results.get('val_score', 'N/A')}")
    print(f"   Total rollouts: {results['total_rollouts']}")
