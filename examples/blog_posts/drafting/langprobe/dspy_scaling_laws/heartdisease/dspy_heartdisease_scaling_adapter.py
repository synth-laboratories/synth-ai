"""DSPy adapters for Heart Disease classification with varying numbers of LLM calls (1, 3, 5) for scaling law experiments."""

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
    # dspy_scaling_laws/heartdisease/ -> dspy_scaling_laws/ -> langprobe/
    _langprobe_dir = _script_dir.parent.parent
    if str(_langprobe_dir) not in sys.path:
        sys.path.insert(0, str(_langprobe_dir))
    from integrations.learning_curve_tracker import LearningCurveTracker

load_dotenv()


# Signatures for multi-step reasoning
class HeartDiseaseClassification(dspy.Signature):
    """Classify heart disease based on patient features."""

    features: str = dspy.InputField(desc="Patient features: age, sex, chest pain type, blood pressure, cholesterol, etc.")
    classification: str = dspy.OutputField(desc="Predicted classification: '1' for heart disease, '0' for no heart disease")


class HeartDiseaseStep1(dspy.Signature):
    """First reasoning step: Analyze patient features."""

    features: str = dspy.InputField(desc="Patient features: age, sex, chest pain type, blood pressure, cholesterol, etc.")
    analysis: str = dspy.OutputField(desc="Analysis of patient features and their medical significance")


class HeartDiseaseStep2(dspy.Signature):
    """Second reasoning step: Identify risk factors."""

    features: str = dspy.InputField(desc="Patient features")
    analysis: str = dspy.InputField(desc="Analysis from step 1")
    risk_factors: str = dspy.OutputField(desc="Identified risk factors for heart disease")


class HeartDiseaseStep3(dspy.Signature):
    """Third reasoning step: Make classification decision."""

    features: str = dspy.InputField(desc="Patient features")
    risk_factors: str = dspy.InputField(desc="Risk factors from step 2")
    classification: str = dspy.OutputField(desc="Predicted classification: '1' for heart disease, '0' for no heart disease")
    reasoning: str = dspy.OutputField(desc="Reasoning for the classification")


class HeartDiseaseStep4(dspy.Signature):
    """Fourth reasoning step: Verify classification."""

    features: str = dspy.InputField(desc="Patient features")
    classification: str = dspy.InputField(desc="Classification from step 3")
    reasoning: str = dspy.InputField(desc="Reasoning from step 3")
    verified_classification: str = dspy.OutputField(desc="Verified classification: '1' for heart disease, '0' for no heart disease")
    verification: str = dspy.OutputField(desc="Verification of classification correctness")


class HeartDiseaseStep5(dspy.Signature):
    """Fifth reasoning step: Final classification with comprehensive reasoning."""

    features: str = dspy.InputField(desc="Patient features")
    verified_classification: str = dspy.InputField(desc="Verified classification from step 4")
    verification: str = dspy.InputField(desc="Verification from step 4")
    classification: str = dspy.OutputField(desc="Final classification: '1' for heart disease, '0' for no heart disease")
    final_reasoning: str = dspy.OutputField(desc="Comprehensive reasoning for the final classification")


# Module with 1 LLM call
class HeartDiseaseClassifier1Call(dspy.Module):
    """DSPy module for Heart Disease classification with 1 LLM call."""

    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(HeartDiseaseClassification)

    def forward(self, features: str) -> dspy.Prediction:
        """Classify heart disease from features with single reasoning step."""
        result = self.predict(features=features)
        return result


# Module with 3 LLM calls
class HeartDiseaseClassifier3Calls(dspy.Module):
    """DSPy module for Heart Disease classification with 3 LLM calls."""

    def __init__(self):
        super().__init__()
        self.step1 = dspy.ChainOfThought(HeartDiseaseStep1)
        self.step2 = dspy.ChainOfThought(HeartDiseaseStep2)
        self.step3 = dspy.ChainOfThought(HeartDiseaseStep3)

    def forward(self, features: str) -> dspy.Prediction:
        """Classify heart disease from features with 3 reasoning steps."""
        step1_result = self.step1(features=features)
        step2_result = self.step2(features=features, analysis=step1_result.analysis)
        step3_result = self.step3(features=features, risk_factors=step2_result.risk_factors)
        return step3_result


# Module with 5 LLM calls
class HeartDiseaseClassifier5Calls(dspy.Module):
    """DSPy module for Heart Disease classification with 5 LLM calls."""

    def __init__(self):
        super().__init__()
        self.step1 = dspy.ChainOfThought(HeartDiseaseStep1)
        self.step2 = dspy.ChainOfThought(HeartDiseaseStep2)
        self.step3 = dspy.ChainOfThought(HeartDiseaseStep3)
        self.step4 = dspy.ChainOfThought(HeartDiseaseStep4)
        self.step5 = dspy.ChainOfThought(HeartDiseaseStep5)

    def forward(self, features: str) -> dspy.Prediction:
        """Classify heart disease from features with 5 reasoning steps."""
        step1_result = self.step1(features=features)
        step2_result = self.step2(features=features, analysis=step1_result.analysis)
        step3_result = self.step3(features=features, risk_factors=step2_result.risk_factors)
        step4_result = self.step4(
            features=features,
            classification=step3_result.classification,
            reasoning=step3_result.reasoning
        )
        step5_result = self.step5(
            features=features,
            verified_classification=step4_result.verified_classification,
            verification=step4_result.verification
        )
        return step5_result


def load_heartdisease_dataset(split: str = "train") -> list[dict[str, Any]]:
    """Load Heart Disease dataset from HuggingFace and convert to list of dicts."""
    dataset = load_dataset("buio/heart-disease", split=split)
    examples = []

    for idx, row in enumerate(dataset):
        features = {}
        label = None

        for key, value in row.items():
            if key in ("target", "label", "class", "disease"):
                label = str(int(value)) if isinstance(value, (int, float)) else str(value)
            else:
                features[key] = value

        feature_text = "\n".join([f"{key}: {value}" for key, value in features.items()])

        examples.append({
            "features": feature_text,
            "label": label or "0",
            "seed": idx,
        })

    return examples


def create_dspy_examples(heartdisease_examples: list[dict[str, Any]]) -> list[dspy.Example]:
    """Convert Heart Disease examples to DSPy Examples."""
    dspy_examples = []
    for ex in heartdisease_examples:
        dspy_ex = dspy.Example(
            features=ex["features"],
            classification=ex["label"],
        ).with_inputs("features")
        dspy_examples.append(dspy_ex)
    return dspy_examples


def heartdisease_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """Metric function for Heart Disease classification."""
    predicted = (pred.classification or "").lower().strip()
    expected = (example.classification or "").lower().strip()

    # Normalize predictions
    if "yes" in predicted or "positive" in predicted or "disease" in predicted:
        predicted = "1"
    elif "no" in predicted or "negative" in predicted or "healthy" in predicted:
        predicted = "0"

    # Extract digits
    pred_digits = "".join([c for c in predicted if c.isdigit()])
    if pred_digits:
        predicted = pred_digits[0]

    exp_digits = "".join([c for c in expected if c.isdigit()])
    if exp_digits:
        expected = exp_digits[0]

    return 1.0 if predicted == expected else 0.0


def heartdisease_metric_gepa(gold: dspy.Example, pred: dspy.Prediction, trace=None, pred_name=None, pred_trace=None) -> float:
    """GEPA-compatible metric function."""
    return heartdisease_metric(gold, pred, trace)


async def run_dspy_gepa_heartdisease_scaling(
    num_calls: int,
    task_app_url: str = "http://127.0.0.1:8114",
    train_seeds: Optional[list[int]] = None,
    val_seeds: Optional[list[int]] = None,
    rollout_budget: int = 300,
    reflection_minibatch_size: int = 3,
    output_dir: Optional[Path] = None,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """Run DSPy GEPA optimization on Heart Disease with specified number of LLM calls.

    Args:
        num_calls: Number of LLM calls (1, 3, or 5)
        task_app_url: Task app URL (for reference, not used directly)
        train_seeds: Training seeds (default: 0-29, 30 examples)
        val_seeds: Validation seeds (default: 30-79, 50 examples)
        rollout_budget: Rollout budget (default: 300)
        reflection_minibatch_size: Minibatch size for reflection evaluation (default: 3)
        output_dir: Output directory
        model: Model string (e.g., "groq/openai/gpt-oss-20b"). Defaults to "groq/openai/gpt-oss-20b"

    Returns:
        Results dictionary
    """
    import time
    start_time = time.time()

    if num_calls not in [1, 3, 5]:
        raise ValueError(f"num_calls must be 1, 3, or 5, got {num_calls}")

    if output_dir is None:
        output_dir = Path(__file__).parent / "results" / f"gepa_{num_calls}calls"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure DSPy LM
    if model is None:
        model = "groq/openai/gpt-oss-20b"

    model_lower = model.lower()
    if "groq" in model_lower:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(f"GROQ_API_KEY required for Groq models (model: {model})")
    elif "openai" in model_lower:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(f"OPENAI_API_KEY required for OpenAI models (model: {model})")
    elif "gemini" in model_lower or "google" in model_lower:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(f"GEMINI_API_KEY required for Gemini models (model: {model})")
        # LiteLLM expects gemini/gemini-* format (not google/)
        if model.startswith("google/"):
            model = model.replace("google/", "gemini/")
        elif not model.startswith("gemini/"):
            # If just "gemini-2.5-flash-lite", add gemini/ prefix
            model = f"gemini/{model}"
    else:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(f"GROQ_API_KEY required (default provider, model: {model})")

    lm = dspy.LM(model, api_key=api_key)

    # Redirect DSPy verbose logging to file
    import logging
    log_file = output_dir / "dspy_gepa.log"
    dspy_logger = logging.getLogger("dspy")
    dspy_logger.handlers.clear()
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    dspy_logger.addHandler(file_handler)
    dspy_logger.setLevel(logging.DEBUG)
    dspy_logger.propagate = False

    print(f"📝 Verbose logs redirected to: {log_file}")

    # Track actual metric evaluations
    metric_calls = {"count": 0}

    def tracked_metric_gepa(gold, pred, trace=None, pred_name=None, pred_trace=None):
        """Wrapped metric that counts calls (GEPA version)."""
        metric_calls["count"] += 1
        return heartdisease_metric_gepa(gold, pred, trace, pred_name, pred_trace)

    # Load dataset
    heartdisease_examples = load_heartdisease_dataset(split="train")

    # Select training and validation seeds
    if train_seeds is None:
        train_seeds = list(range(30))
    if val_seeds is None:
        val_seeds = list(range(30, 80))

    # Filter examples by seeds
    train_examples = [heartdisease_examples[i] for i in train_seeds if i < len(heartdisease_examples)]
    val_examples = [heartdisease_examples[i] for i in val_seeds if i < len(heartdisease_examples)]

    # Convert to DSPy Examples
    trainset = create_dspy_examples(train_examples)
    valset = create_dspy_examples(val_examples)

    # Create module based on num_calls
    if num_calls == 1:
        module = HeartDiseaseClassifier1Call()
    elif num_calls == 3:
        module = HeartDiseaseClassifier3Calls()
    else:  # num_calls == 5
        module = HeartDiseaseClassifier5Calls()

    # Learning curve tracker
    learning_curve = LearningCurveTracker(
        framework=f"dspy_gepa_{num_calls}calls",
        benchmark="heartdisease",
        total_budget=rollout_budget,
    )

    # Initialize GEPA optimizer
    from dspy.teleprompt.gepa import GEPA

    max_metric_calls = int(rollout_budget)
    # Use Gemini for reflection LM
    reflection_api_key = os.getenv("GEMINI_API_KEY")
    if not reflection_api_key:
        raise ValueError("GEMINI_API_KEY required for reflection LM")
    reflection_lm = dspy.LM("gemini/gemini-2.5-flash", api_key=reflection_api_key)

    optimizer = GEPA(
        metric=tracked_metric_gepa,
        max_metric_calls=max_metric_calls,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=reflection_minibatch_size,
        track_stats=True,
    )

    # Evaluate baseline (before optimization)
    from dspy.evaluate import Evaluate

    with dspy.context(lm=lm):
        # Reset counter before baseline evaluation
        metric_calls["count"] = 0
        evaluate = Evaluate(devset=valset, metric=tracked_metric_gepa, num_threads=1)
        baseline_score = evaluate(module)
        baseline_metric_calls = metric_calls["count"]

        # Extract the score
        if isinstance(baseline_score, (int, float)):
            baseline_val = float(baseline_score) / 100.0 if baseline_score > 1 else float(baseline_score)
        elif isinstance(baseline_score, dict):
            baseline_val = baseline_score.get("accuracy", baseline_score.get("score", 0.0))
        elif hasattr(baseline_score, "score"):
            baseline_val = float(baseline_score.score) / 100.0
        else:
            baseline_val = 0.0

        print(f"📊 Baseline performance ({num_calls} calls): {baseline_val:.4f}")

        # Record baseline checkpoint
        learning_curve.curve.record(
            rollout_count=0,
            performance=baseline_val,
            checkpoint_pct=0.0,
        )

        # Optimize with progress tracking
        print(f"🚀 DSPy GEPA ({num_calls} calls, budget={rollout_budget}, max_metric_calls={max_metric_calls})")
        print(f"   Progress updates will be printed here; detailed logs saved to {log_file.name}")

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

    detailed_results = {
        "best_score": val_score_pct,
        "baseline_score": float(baseline_val),
        "total_rollouts": rollout_budget,
        "total_time": total_time,
        "num_calls": num_calls,
        "candidates": [],
        "evolution": [],
    }

    if hasattr(optimized_module, "detailed_results"):
        gepa_results = optimized_module.detailed_results

        if hasattr(gepa_results, "total_metric_calls") and gepa_results.total_metric_calls is not None:
            detailed_results["actual_rollouts"] = gepa_results.total_metric_calls

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

            candidate_info["instructions"] = {}
            if isinstance(candidate, dict):
                for pred_name, instruction in candidate.items():
                    candidate_info["instructions"][pred_name] = str(instruction)
            elif hasattr(candidate, 'named_predictors'):
                for pred_name, predictor in candidate.named_predictors():
                    if hasattr(predictor, 'signature') and hasattr(predictor.signature, 'instructions'):
                        candidate_info["instructions"][pred_name] = str(predictor.signature.instructions)

            detailed_results["candidates"].append(candidate_info)

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

    # Save stats
    stats_file = output_dir / f"dspy_gepa_heartdisease_{num_calls}calls_stats.json"
    with open(stats_file, "w") as f:
        json.dump({
            "total_time": total_time,
            "total_rollouts": rollout_budget,
            "baseline_score": float(baseline_val),
            "val_score": val_score_pct,
            "val_n": len(valset),
            "num_calls": num_calls,
        }, f, indent=2)

    results = {
        "best_score": val_score_pct,
        "val_score": val_score_pct,
        "total_rollouts": rollout_budget,
        "max_metric_calls": max_metric_calls,
        "num_calls": num_calls,
    }

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run DSPy GEPA on Heart Disease with scaling")
    parser.add_argument("--num-calls", type=int, choices=[1, 3, 5], required=True, help="Number of LLM calls (1, 3, or 5)")
    parser.add_argument("--rollout-budget", type=int, default=300, help="Rollout budget")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--model", type=str, help="Model string")

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None

    asyncio.run(run_dspy_gepa_heartdisease_scaling(
        num_calls=args.num_calls,
        rollout_budget=args.rollout_budget,
        output_dir=output_dir,
        model=args.model,
    ))

