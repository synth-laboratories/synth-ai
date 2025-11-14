"""DSPy adapters for Heart Disease classification using MIPROv2 and GEPA optimizers."""

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

# Handle imports for both module and direct script execution
try:
    from ...integrations.learning_curve_tracker import LearningCurveTracker
except ImportError:
    # When run as a script, add parent directories to path
    import sys
    from pathlib import Path as PathLib
    _script_dir = PathLib(__file__).resolve().parent
    _langprobe_dir = _script_dir.parent.parent
    if str(_langprobe_dir) not in sys.path:
        sys.path.insert(0, str(_langprobe_dir))
    from integrations.learning_curve_tracker import LearningCurveTracker

load_dotenv()


class HeartDiseaseClassification(dspy.Signature):
    """Classify heart disease based on patient features.

    Given patient measurements and medical data, predict whether the patient has heart disease.
    """

    features: str = dspy.InputField(
        desc="Patient features: age, sex, chest pain type, blood pressure, cholesterol, etc."
    )
    classification: str = dspy.OutputField(
        desc="Predicted classification: '1' for heart disease, '0' for no heart disease"
    )


class HeartDiseaseClassifier(dspy.Module):
    """DSPy module for Heart Disease classification."""

    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(HeartDiseaseClassification)

    def forward(self, features: str) -> dspy.Prediction:
        """Classify heart disease from features string.

        Args:
            features: Feature text (e.g., "age: 63\nsex: 1\n...")

        Returns:
            Prediction with classification field
        """
        result = self.predict(features=features)
        return result


def load_heartdisease_dataset(split: str = "train") -> list[dict[str, Any]]:
    """Load Heart Disease dataset from HuggingFace and convert to list of dicts.

    Args:
        split: Dataset split (default: "train")

    Returns:
        List of examples with features and label
    """
    dataset = load_dataset("buio/heart-disease", split=split)
    examples = []

    for idx, row in enumerate(dataset):
        # Extract features and label
        features = {}
        label = None

        for key, value in row.items():
            if key in ("target", "label", "class", "disease"):
                # This is the label
                label = str(int(value)) if isinstance(value, (int, float)) else str(value)
            else:
                # These are features
                features[key] = value

        # Format features as text (matching task app format)
        feature_text = "\n".join([f"{key}: {value}" for key, value in features.items()])

        examples.append({
            "features": feature_text,
            "label": label or "0",
            "seed": idx,  # Use index as seed
        })

    return examples


def create_dspy_examples(heartdisease_examples: list[dict[str, Any]]) -> list[dspy.Example]:
    """Convert Heart Disease examples to DSPy Examples.

    Args:
        heartdisease_examples: List of Heart Disease example dicts

    Returns:
        List of DSPy Examples
    """
    dspy_examples = []
    for ex in heartdisease_examples:
        dspy_ex = dspy.Example(
            features=ex["features"],
            classification=ex["label"],
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


def heartdisease_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """Metric function for Heart Disease classification.

    Args:
        example: Ground truth example
        pred: Model prediction
        trace: Optional trace (unused)

    Returns:
        Score (1.0 if correct, 0.0 if incorrect)
    """
    predicted = (pred.classification or "").lower().strip()
    expected = (example.classification or "").lower().strip()

    # Normalize predictions (handle "yes/no", "positive/negative", etc.)
    if "yes" in predicted or "positive" in predicted or "disease" in predicted:
        predicted = "1"
    elif "no" in predicted or "negative" in predicted or "healthy" in predicted:
        predicted = "0"

    # Extract digits
    pred_digits = "".join([c for c in predicted if c.isdigit()])
    if pred_digits:
        predicted = pred_digits[0]  # Take first digit

    exp_digits = "".join([c for c in expected if c.isdigit()])
    if exp_digits:
        expected = exp_digits[0]  # Take first digit

    return 1.0 if predicted == expected else 0.0


def heartdisease_metric_gepa(gold: dspy.Example, pred: dspy.Prediction, trace=None, pred_name=None, pred_trace=None) -> float:
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
    return heartdisease_metric(gold, pred, trace)


async def run_dspy_miprov2_heartdisease(
    task_app_url: str = "http://127.0.0.1:8114",
    train_seeds: Optional[list[int]] = None,
    val_seeds: Optional[list[int]] = None,
    rollout_budget: int = 300,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Run DSPy MIPROv2 optimization on Heart Disease.

    Args:
        task_app_url: Task app URL (for reference, not used directly)
        train_seeds: Training seeds (default: 0-29, 30 examples)
        val_seeds: Validation seeds (default: 30-79, 50 examples)
        rollout_budget: Rollout budget (default: 300)
        output_dir: Output directory

    Returns:
        Results dictionary
    """
    import time
    start_time = time.time()

    if output_dir is None:
        output_dir = Path(__file__).parent / "results" / "dspy_mipro"

    output_dir.mkdir(parents=True, exist_ok=True)
    _warn_if_dotenv_is_messy()

    # Configure DSPy LM
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError("GROQ_API_KEY required")

    # Main LM: gpt-oss-20b via Groq
    lm = dspy.LM("groq/openai/gpt-oss-20b", api_key=groq_api_key)
    dspy.configure(lm=lm)

    # Track actual metric evaluations
    metric_calls = {"count": 0}

    def tracked_metric(gold, pred, trace=None):
        """Wrapped metric that counts calls."""
        metric_calls["count"] += 1
        return heartdisease_metric(gold, pred, trace)

    # Load dataset
    heartdisease_examples = load_heartdisease_dataset(split="train")

    # Select training and validation seeds (matching heartdisease_mipro.toml)
    if train_seeds is None:
        train_seeds = list(range(30))  # 0-29: 30 training examples
    if val_seeds is None:
        val_seeds = list(range(30, 80))  # 30-79: 50 validation examples

    # Filter examples by seeds
    train_examples = [heartdisease_examples[i] for i in train_seeds if i < len(heartdisease_examples)]
    val_examples = [heartdisease_examples[i] for i in val_seeds if i < len(heartdisease_examples)]

    # Convert to DSPy Examples
    trainset = create_dspy_examples(train_examples)
    valset = create_dspy_examples(val_examples)

    # Create module
    module = HeartDiseaseClassifier()

    # Initialize MIPROv2 optimizer
    from dspy.teleprompt import MIPROv2

    # Manual parameter control for consistent evaluation counts (~200 rollouts target)
    # num_candidates=20, num_trials=10 -> ~200 rollouts (20 candidates * 10 trials)
    log_dir = output_dir / "mipro_logs"
    log_dir.mkdir(exist_ok=True)

    optimizer = MIPROv2(
        metric=tracked_metric,
        num_candidates=20,              # Total candidates to generate (instruct + fewshot)
        max_bootstrapped_demos=10,      # Max few-shot examples to bootstrap
        max_labeled_demos=10,           # Max labeled demonstrations
        auto=None,                      # Disable auto presets, use manual params
        log_dir=str(log_dir),           # Save trial programs and logs
        track_stats=True,               # Track trial statistics
    )

    # Learning curve tracker
    learning_curve = LearningCurveTracker(
        framework="dspy_miprov2",
        benchmark="heartdisease",
        total_budget=rollout_budget,
    )

    # Evaluate baseline (before optimization)
    from dspy.evaluate import Evaluate

    # Reset counter before baseline evaluation
    metric_calls["count"] = 0
    evaluate = Evaluate(devset=valset, metric=tracked_metric, num_threads=1)
    baseline_score = evaluate(module)
    baseline_metric_calls = metric_calls["count"]

    # Extract the score (DSPy returns EvaluationResult with .score attribute)
    if isinstance(baseline_score, (int, float)):
        baseline_val = float(baseline_score) / 100.0 if baseline_score > 1 else float(baseline_score)
    elif isinstance(baseline_score, dict):
        baseline_val = baseline_score.get("accuracy", baseline_score.get("score", 0.0))
    elif hasattr(baseline_score, "score"):
        # EvaluationResult object - score is already a percentage (0-100)
        baseline_val = float(baseline_score.score) / 100.0
    else:
        baseline_val = 0.0

    print(f"📊 Baseline performance: {baseline_val:.4f}")

    # Record baseline checkpoint
    learning_curve.curve.record(
        rollout_count=0,
        performance=baseline_val,
        checkpoint_pct=0.0,
    )

    # Optimize with progress tracking
    print(f"🚀 DSPy MIPROv2 (candidates=20, trials=10, max_demos=10, auto=None)")

    # Reset counter for optimization phase (exclude baseline calls)
    metric_calls["count"] = 0
    optimized_module = optimizer.compile(student=module, trainset=trainset, valset=valset, num_trials=10)
    optimization_metric_calls = metric_calls["count"]

    # Evaluate optimized module
    val_score = evaluate(optimized_module)
    final_eval_calls = metric_calls["count"] - optimization_metric_calls

    # Extract final score
    if isinstance(val_score, (int, float)):
        val_score_pct = float(val_score) / 100.0 if val_score > 1 else float(val_score)
    elif isinstance(val_score, dict):
        val_score_pct = val_score.get("accuracy", val_score.get("score", 0.0))
    elif hasattr(val_score, "score"):
        val_score_pct = float(val_score.score) / 100.0
    else:
        val_score_pct = 0.0

    # Record final checkpoint
    learning_curve.curve.record(
        rollout_count=rollout_budget,
        performance=val_score_pct,
        checkpoint_pct=1.0,
    )

    # Calculate time taken
    total_time = time.time() - start_time

    # Save results
    learning_curve.save(output_dir)

    # Save detailed optimization results
    detailed_results_file = output_dir / "dspy_gepa_detailed_results.json"

    # Extract detailed results if available
    detailed_results = {
        "best_score": val_score_pct,
        "baseline_score": float(baseline_val),
        "total_rollouts": rollout_budget,
        "total_time": total_time,
        "candidates": [],
        "evolution": [],
    }

    if hasattr(optimized_module, "detailed_results"):
        gepa_results = optimized_module.detailed_results

        # Add actual metric calls if available
        if hasattr(gepa_results, "total_metric_calls") and gepa_results.total_metric_calls is not None:
            detailed_results["actual_rollouts"] = gepa_results.total_metric_calls

        # Add log directory if available
        if hasattr(gepa_results, "log_dir") and gepa_results.log_dir is not None:
            detailed_results["log_dir"] = gepa_results.log_dir
            detailed_results["note"] = "Candidate programs and full optimization logs are saved in the log_dir"

        # Extract candidate information
        for i, (candidate, score, discovery_count) in enumerate(zip(
            gepa_results.candidates,
            gepa_results.val_aggregate_scores,
            gepa_results.discovery_eval_counts
        )):
            candidate_info = {
                "candidate_num": i,
                "score": float(score),
                "discovery_rollout": discovery_count,
                "is_best": i == gepa_results.best_idx,
            }

            # Extract instruction text for each predictor
            candidate_info["instructions"] = {}
            if isinstance(candidate, dict):
                for pred_name, instruction in candidate.items():
                    candidate_info["instructions"][pred_name] = str(instruction)
            elif hasattr(candidate, 'named_predictors'):
                # Module object - extract instructions from predictors
                for pred_name, predictor in candidate.named_predictors():
                    if hasattr(predictor, 'signature') and hasattr(predictor.signature, 'instructions'):
                        candidate_info["instructions"][pred_name] = str(predictor.signature.instructions)

            detailed_results["candidates"].append(candidate_info)

        # Add evolution/lineage information
        if hasattr(gepa_results, "parents"):
            for i, parent_list in enumerate(gepa_results.parents):
                evolution_info = {
                    "candidate_num": i,
                    "parents": parent_list if parent_list else [],
                }
                detailed_results["evolution"].append(evolution_info)

    # Save to JSON
    with open(detailed_results_file, "w") as f:
        json.dump(detailed_results, f, indent=2)

    print(f"📊 Saved detailed results to {detailed_results_file}")

    # Also save time and rollout info to JSON for parallel runner
    stats_file = output_dir / "dspy_miprov2_heartdisease_stats.json"
    with open(stats_file, "w") as f:
        json.dump({
            "total_time": total_time,
            "total_rollouts": rollout_budget,
            "actual_rollouts": optimization_metric_calls,  # Actual metric evaluations during optimization
            "baseline_score": float(baseline_val),
            "val_score": val_score_pct,
            "val_n": len(valset),
            "train_n": len(trainset),
            "num_candidates": 20,
            "num_trials": 10,
            "max_bootstrapped_demos": 10,
            "max_labeled_demos": 10,
        }, f, indent=2)

    results = {
        "best_score": val_score_pct,
        "val_score": val_score_pct,
        "total_rollouts": rollout_budget,
        "num_candidates": 20,
        "num_trials": 10,
        "max_demos": 10,
    }

    # Save detailed optimization results
    detailed_results_file = output_dir / "dspy_miprov2_detailed_results.json"

    # Extract trial logs and candidate programs if available
    trial_logs = getattr(optimized_module, "trial_logs", {})
    candidate_programs = getattr(optimized_module, "candidate_programs", [])

    # Build detailed results with all instructions and scores
    detailed_results = {
        "best_score": val_score_pct,
        "baseline_score": float(baseline_val),
        "total_rollouts": rollout_budget,
        "actual_rollouts": optimization_metric_calls,
        "total_time": total_time,
        "num_candidates": 20,
        "num_trials": 10,
        "log_dir": str(log_dir),
        "note": "Trial programs and full optimization logs are saved in the log_dir",
        "trials": [],
        "candidate_programs": [],
    }

    # Add trial information
    for trial_num, trial_data in trial_logs.items():
        trial_info = {
            "trial_num": trial_num,
            "score": trial_data.get("full_eval_score", trial_data.get("score")),
            "instructions": {},
            "demos": {},
            "total_eval_calls": trial_data.get("total_eval_calls_so_far"),
        }

        # Extract instruction indices for each predictor
        for key, value in trial_data.items():
            if "_predictor_instruction" in key:
                predictor_idx = key.split("_")[0]
                trial_info["instructions"][f"predictor_{predictor_idx}"] = value
            elif "_predictor_demos" in key:
                predictor_idx = key.split("_")[0]
                trial_info["demos"][f"predictor_{predictor_idx}"] = value

        detailed_results["trials"].append(trial_info)

    # Add candidate program scores
    for i, candidate in enumerate(candidate_programs):
        detailed_results["candidate_programs"].append({
            "rank": i + 1,
            "score": candidate.get("score"),
            "full_eval": candidate.get("full_eval", False),
        })

    # Save to JSON
    with open(detailed_results_file, "w") as f:
        json.dump(detailed_results, f, indent=2)

    print(f"📊 Saved detailed results to {detailed_results_file}")

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
            if hasattr(demo, "classification"):
                demo_dict["output"] = str(demo.classification)
            if hasattr(demo, "rationale"):
                demo_dict["rationale"] = str(demo.rationale)
            prompt_details["demo_examples"].append(demo_dict)

    # Try to get the full prompt text by running a forward pass
    try:
        sample_input = trainset[0].features if trainset and len(trainset) > 0 else "age: 63\nsex: 1\ncp: 3\ntrestbps: 145\nchol: 233\nfbs: 1\nrestecg: 0\nthalach: 150\nexang: 0\noldpeak: 2.3\nslope: 0\nca: 0\nthal: 1"
        sample_pred = optimized_module(sample_input)
        prompt_details["sample_prediction"] = {
            "input": sample_input,
            "output": str(sample_pred.classification) if hasattr(sample_pred, "classification") else str(sample_pred),
            "rationale": str(sample_pred.rationale) if hasattr(sample_pred, "rationale") else None,
        }
    except Exception as e:
        prompt_details["sample_prediction_error"] = str(e)

    module_info["prompt_details"] = prompt_details

    with open(output_dir / "heartdisease_best_module.json", "w") as f:
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
    return results


async def run_dspy_gepa_heartdisease(
    task_app_url: str = "http://127.0.0.1:8114",
    train_seeds: Optional[list[int]] = None,
    val_seeds: Optional[list[int]] = None,
    rollout_budget: int = 300,
    output_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Run DSPy GEPA optimization on Heart Disease.

    Args:
        task_app_url: Task app URL (for reference, not used directly)
        train_seeds: Training seeds (default: 0-29, 30 examples)
        val_seeds: Validation seeds (default: 30-79, 50 examples)
        rollout_budget: Rollout budget (default: 300)
        output_dir: Output directory

    Returns:
        Results dictionary
    """
    import time
    start_time = time.time()

    if output_dir is None:
        output_dir = Path(__file__).parent / "results" / "dspy_gepa"

    output_dir.mkdir(parents=True, exist_ok=True)
    _warn_if_dotenv_is_messy()

    # Configure DSPy LM
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY required")

    # Main LM: llama-3.1-8b-instant via Groq (matching synth GEPA)
    lm = dspy.LM("groq/llama-3.1-8b-instant", api_key=groq_api_key)
    dspy.configure(lm=lm)

    # Track actual metric evaluations
    metric_calls = {"count": 0}

    def tracked_metric_gepa(gold, pred, trace=None, pred_name=None, pred_trace=None):
        """Wrapped metric that counts calls (GEPA version)."""
        metric_calls["count"] += 1
        return heartdisease_metric_gepa(gold, pred, trace, pred_name, pred_trace)

    # Load dataset
    heartdisease_examples = load_heartdisease_dataset(split="train")

    # Select training and validation seeds (matching heartdisease_gepa.toml)
    if train_seeds is None:
        train_seeds = list(range(30))  # 0-29: 30 training examples
    if val_seeds is None:
        val_seeds = list(range(30, 80))  # 30-79: 50 validation examples

    # Filter examples by seeds
    train_examples = [heartdisease_examples[i] for i in train_seeds if i < len(heartdisease_examples)]
    val_examples = [heartdisease_examples[i] for i in val_seeds if i < len(heartdisease_examples)]

    # Convert to DSPy Examples
    trainset = create_dspy_examples(train_examples)
    valset = create_dspy_examples(val_examples)

    # Create module
    module = HeartDiseaseClassifier()

    # Initialize GEPA optimizer
    from dspy.teleprompt.gepa import GEPA

    max_metric_calls = int(rollout_budget)
    # GEPA requires a reflection LM - use llama-3.3-70b-versatile
    reflection_lm = dspy.LM("groq/llama-3.3-70b-versatile", api_key=groq_api_key)
    optimizer = GEPA(metric=tracked_metric_gepa, max_metric_calls=max_metric_calls, reflection_lm=reflection_lm, track_stats=True)

    # Learning curve tracker
    learning_curve = LearningCurveTracker(
        framework="dspy_gepa",
        benchmark="heartdisease",
        total_budget=rollout_budget,
    )

    # Evaluate baseline (before optimization)
    from dspy.evaluate import Evaluate

    # Reset counter before baseline evaluation
    metric_calls["count"] = 0
    evaluate = Evaluate(devset=valset, metric=tracked_metric_gepa, num_threads=1)
    baseline_score = evaluate(module)
    baseline_metric_calls = metric_calls["count"]

    # Extract the score (DSPy returns EvaluationResult with .score attribute)
    if isinstance(baseline_score, (int, float)):
        baseline_val = float(baseline_score) / 100.0 if baseline_score > 1 else float(baseline_score)
    elif isinstance(baseline_score, dict):
        baseline_val = baseline_score.get("accuracy", baseline_score.get("score", 0.0))
    elif hasattr(baseline_score, "score"):
        # EvaluationResult object - score is already a percentage (0-100)
        baseline_val = float(baseline_score.score) / 100.0
    else:
        baseline_val = 0.0

    print(f"📊 Baseline performance: {baseline_val:.4f}")

    # Record baseline checkpoint
    learning_curve.curve.record(
        rollout_count=0,
        performance=baseline_val,
        checkpoint_pct=0.0,
    )

    # Optimize with progress tracking
    print(f"🚀 DSPy GEPA (budget={rollout_budget}, max_metric_calls={max_metric_calls})")

    optimized_module = optimizer.compile(student=module, trainset=trainset, valset=valset)

    # Evaluate optimized module
    val_score = evaluate(optimized_module)

    # Extract final score
    if isinstance(val_score, (int, float)):
        val_score_pct = float(val_score) / 100.0 if val_score > 1 else float(val_score)
    elif isinstance(val_score, dict):
        val_score_pct = val_score.get("accuracy", val_score.get("score", 0.0))
    elif hasattr(val_score, "score"):
        val_score_pct = float(val_score.score) / 100.0
    else:
        val_score_pct = 0.0

    # Record final checkpoint
    learning_curve.curve.record(
        rollout_count=rollout_budget,
        performance=val_score_pct,
        checkpoint_pct=1.0,
    )

    # Calculate time taken
    total_time = time.time() - start_time

    # Save results
    learning_curve.save(output_dir)

    # Save detailed optimization results
    detailed_results_file = output_dir / "dspy_gepa_detailed_results.json"

    # Extract detailed results if available
    detailed_results = {
        "best_score": val_score_pct,
        "baseline_score": float(baseline_val),
        "total_rollouts": rollout_budget,
        "total_time": total_time,
        "candidates": [],
        "evolution": [],
    }

    if hasattr(optimized_module, "detailed_results"):
        gepa_results = optimized_module.detailed_results

        # Add actual metric calls if available
        if hasattr(gepa_results, "total_metric_calls") and gepa_results.total_metric_calls is not None:
            detailed_results["actual_rollouts"] = gepa_results.total_metric_calls

        # Add log directory if available
        if hasattr(gepa_results, "log_dir") and gepa_results.log_dir is not None:
            detailed_results["log_dir"] = gepa_results.log_dir
            detailed_results["note"] = "Candidate programs and full optimization logs are saved in the log_dir"

        # Extract candidate information
        for i, (candidate, score, discovery_count) in enumerate(zip(
            gepa_results.candidates,
            gepa_results.val_aggregate_scores,
            gepa_results.discovery_eval_counts
        )):
            candidate_info = {
                "candidate_num": i,
                "score": float(score),
                "discovery_rollout": discovery_count,
                "is_best": i == gepa_results.best_idx,
            }

            # Extract instruction text for each predictor
            candidate_info["instructions"] = {}
            if isinstance(candidate, dict):
                for pred_name, instruction in candidate.items():
                    candidate_info["instructions"][pred_name] = str(instruction)
            elif hasattr(candidate, 'named_predictors'):
                # Module object - extract instructions from predictors
                for pred_name, predictor in candidate.named_predictors():
                    if hasattr(predictor, 'signature') and hasattr(predictor.signature, 'instructions'):
                        candidate_info["instructions"][pred_name] = str(predictor.signature.instructions)

            detailed_results["candidates"].append(candidate_info)

        # Add evolution/lineage information
        if hasattr(gepa_results, "parents"):
            for i, parent_list in enumerate(gepa_results.parents):
                evolution_info = {
                    "candidate_num": i,
                    "parents": parent_list if parent_list else [],
                }
                detailed_results["evolution"].append(evolution_info)

    # Save to JSON
    with open(detailed_results_file, "w") as f:
        json.dump(detailed_results, f, indent=2)

    print(f"📊 Saved detailed results to {detailed_results_file}")

    # Also save time and rollout info to JSON for parallel runner
    stats_file = output_dir / "dspy_gepa_heartdisease_stats.json"
    with open(stats_file, "w") as f:
        json.dump({
            "total_time": total_time,
            "total_rollouts": rollout_budget,
            "baseline_score": float(baseline_val),
            "val_score": val_score_pct,
            "val_n": len(valset),
        }, f, indent=2)

    results = {
        "best_score": val_score_pct,
        "val_score": val_score_pct,
        "total_rollouts": rollout_budget,
        "max_metric_calls": max_metric_calls,
        "total_time": total_time,
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
            if hasattr(demo, "classification"):
                demo_dict["output"] = str(demo.classification)
            if hasattr(demo, "rationale"):
                demo_dict["rationale"] = str(demo.rationale)
            prompt_details["demo_examples"].append(demo_dict)

    # Try to get the full prompt text by running a forward pass
    try:
        sample_input = trainset[0].features if trainset and len(trainset) > 0 else "age: 63\nsex: 1\ncp: 3\ntrestbps: 145\nchol: 233\nfbs: 1\nrestecg: 0\nthalach: 150\nexang: 0\noldpeak: 2.3\nslope: 0\nca: 0\nthal: 1"
        sample_pred = optimized_module(sample_input)
        prompt_details["sample_prediction"] = {
            "input": sample_input,
            "output": str(sample_pred.classification) if hasattr(sample_pred, "classification") else str(sample_pred),
            "rationale": str(sample_pred.rationale) if hasattr(sample_pred, "rationale") else None,
        }
    except Exception as e:
        prompt_details["sample_prediction_error"] = str(e)

    module_info["prompt_details"] = prompt_details

    with open(output_dir / "heartdisease_best_module.json", "w") as f:
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
    return results
