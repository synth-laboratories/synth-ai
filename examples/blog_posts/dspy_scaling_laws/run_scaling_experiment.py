"""Run DSPy scaling experiment across multiple benchmarks and pipeline complexities."""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Optional
import dspy
from datasets import load_dataset
from dotenv import load_dotenv
import sys

# Add langprobe to path
langprobe_dir = Path(__file__).resolve().parents[1] / "langprobe"
if str(langprobe_dir) not in sys.path:
    sys.path.insert(0, str(langprobe_dir))

from integrations.learning_curve_tracker import LearningCurveTracker

load_dotenv()


# ============================================================================
# Multi-Step DSPy Modules
# ============================================================================

class MultiStepClassifier(dspy.Module):
    """Generic multi-step classifier with configurable pipeline depth."""

    def __init__(self, signature_class, num_steps: int = 1):
        super().__init__()
        self.num_steps = num_steps
        self.predictors = [dspy.ChainOfThought(signature_class) for _ in range(num_steps)]

    def forward(self, **kwargs) -> dspy.Prediction:
        """Run multi-step prediction pipeline."""
        result = None
        for i, predictor in enumerate(self.predictors):
            # First step uses original inputs
            if i == 0:
                result = predictor(**kwargs)
            else:
                # Subsequent steps can use previous outputs
                # For now, just repeat prediction (can be enhanced with chain-of-thought)
                result = predictor(**kwargs)

        return result


# ============================================================================
# Banking77
# ============================================================================

class Banking77Classification(dspy.Signature):
    """Classify banking customer query into one of 77 intents."""
    query: str = dspy.InputField(desc="Customer query text")
    available_intents: str = dspy.InputField(desc="List of available intent labels")
    intent: str = dspy.OutputField(desc="Predicted intent label")


def load_banking77_data(split="train"):
    """Load Banking77 dataset."""
    dataset = load_dataset("banking77", split=split, trust_remote_code=False)
    label_names = dataset.features["label"].names if hasattr(dataset.features.get("label"), "names") else []
    intents_text = "\n".join([f"{i+1}. {intent}" for i, intent in enumerate(label_names)])

    examples = []
    for idx, row in enumerate(dataset):
        label_idx = int(row.get("label", 0))
        label_name = label_names[label_idx] if label_idx < len(label_names) else str(label_idx)
        examples.append(dspy.Example(
            query=str(row.get("text", "")),
            available_intents=intents_text,
            intent=label_name,
        ).with_inputs("query", "available_intents"))

    return examples, label_names


def banking77_metric(gold: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """Banking77 metric."""
    predicted = (pred.intent or "").lower().strip().replace("_", " ")
    expected = (gold.intent or "").lower().strip().replace("_", " ")
    return 1.0 if predicted == expected else 0.0


# ============================================================================
# HeartDisease
# ============================================================================

class HeartDiseaseClassification(dspy.Signature):
    """Classify heart disease based on patient features."""
    features: str = dspy.InputField(desc="Patient features")
    classification: str = dspy.OutputField(desc="'1' for disease, '0' for no disease")


def load_heartdisease_data(split="train"):
    """Load Heart Disease dataset."""
    dataset = load_dataset("buio/heart-disease", split=split)
    examples = []

    for idx, row in enumerate(dataset):
        features = {k: v for k, v in row.items() if k not in ("target", "label", "class", "disease")}
        feature_text = "\n".join([f"{k}: {v}" for k, v in features.items()])
        label = str(int(row.get("target", row.get("label", row.get("class", row.get("disease", 0))))))

        examples.append(dspy.Example(
            features=feature_text,
            classification=label,
        ).with_inputs("features"))

    return examples, None


def heartdisease_metric(gold: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """Heart disease metric."""
    predicted = (pred.classification or "").strip()
    expected = (gold.classification or "").strip()
    return 1.0 if predicted == expected else 0.0


# ============================================================================
# HotpotQA
# ============================================================================

class HotpotQAAnswering(dspy.Signature):
    """Answer multi-hop questions using context."""
    question: str = dspy.InputField(desc="The question")
    context: str = dspy.InputField(desc="Context passages")
    answer: str = dspy.OutputField(desc="The answer")


def load_hotpotqa_data(split="validation"):
    """Load HotPotQA dataset."""
    dataset = load_dataset("hotpot_qa", "distractor", split=split)
    examples = []

    for idx, row in enumerate(dataset):
        # Format context
        context_list = row.get("context", [])
        context_lines = []

        if isinstance(context_list, dict):
            titles = context_list.get("title", [])
            sentences_list = context_list.get("sentences", [])
            for title, sentences in zip(titles, sentences_list):
                context_lines.append(f"### {title}")
                context_lines.extend(sentences)
                context_lines.append("")
        elif isinstance(context_list, list):
            for title, sentences in context_list:
                context_lines.append(f"### {title}")
                context_lines.extend(sentences)
                context_lines.append("")

        context = "\n".join(context_lines)
        answer = str(row.get("answer", ""))

        examples.append(dspy.Example(
            question=str(row.get("question", "")),
            context=context,
            answer=answer,
        ).with_inputs("question", "context"))

    return examples, None


def hotpotqa_metric(gold: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """HotPotQA metric (exact match)."""
    predicted = (pred.answer or "").lower().strip()
    expected = (gold.answer or "").lower().strip()
    return 1.0 if predicted == expected else 0.0


# ============================================================================
# Experiment Runner
# ============================================================================

BENCHMARK_CONFIG = {
    "banking77": {
        "signature": Banking77Classification,
        "loader": load_banking77_data,
        "metric": banking77_metric,
        "train_seeds": list(range(50)),
        "val_seeds": list(range(50, 250)),
    },
    "heartdisease": {
        "signature": HeartDiseaseClassification,
        "loader": load_heartdisease_data,
        "metric": heartdisease_metric,
        "train_seeds": list(range(25)),
        "val_seeds": list(range(50, 150)),
    },
    "hotpotqa": {
        "signature": HotpotQAAnswering,
        "loader": load_hotpotqa_data,
        "metric": hotpotqa_metric,
        "train_seeds": list(range(25)),
        "val_seeds": list(range(50, 150)),
    },
}


async def run_single_experiment(
    benchmark: str,
    num_steps: int,
    optimizer: str,  # "gepa" or "mipro"
    output_dir: Path,
    rollout_budget: int = 200,
) -> dict[str, Any]:
    """Run a single scaling experiment."""

    print(f"\n{'='*80}")
    print(f"🚀 Running: {benchmark} | {num_steps}-step | {optimizer}")
    print(f"{'='*80}\n")

    config = BENCHMARK_CONFIG[benchmark]

    # Load data
    all_examples, metadata = config["loader"]()
    train_seeds = config["train_seeds"]
    val_seeds = config["val_seeds"]

    trainset = [all_examples[i] for i in train_seeds if i < len(all_examples)]
    valset = [all_examples[i] for i in val_seeds if i < len(all_examples)]

    print(f"📊 Train examples: {len(trainset)}, Val examples: {len(valset)}")

    # Configure LM
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY required")

    if optimizer == "mipro":
        lm = dspy.LM("groq/openai/gpt-oss-20b", api_key=groq_api_key)
    else:  # gepa
        lm = dspy.LM("groq/llama-3.1-8b-instant", api_key=groq_api_key)

    dspy.configure(lm=lm)

    # Create module
    module = MultiStepClassifier(config["signature"], num_steps=num_steps)

    # Evaluate baseline
    from dspy.evaluate import Evaluate
    evaluate = Evaluate(devset=valset, metric=config["metric"], num_threads=1)
    baseline_result = evaluate(module)

    if hasattr(baseline_result, "score"):
        baseline_score = float(baseline_result.score) / 100.0
    else:
        baseline_score = float(baseline_result) / 100.0 if baseline_result > 1 else float(baseline_result)

    print(f"✅ Baseline: {baseline_score:.4f}")

    # Learning curve tracker
    tracker = LearningCurveTracker(
        framework=f"dspy_{optimizer}",
        benchmark=f"{benchmark}_{num_steps}step",
        total_budget=rollout_budget,
    )

    tracker.curve.record(rollout_count=0, performance=baseline_score, checkpoint_pct=0.0)

    # Optimize
    if optimizer == "mipro":
        from dspy.teleprompt import MIPROv2
        opt = MIPROv2(
            metric=config["metric"],
            num_candidates=20,
            max_bootstrapped_demos=10,
            max_labeled_demos=10,
            auto=None,
        )
        optimized_module = opt.compile(student=module, trainset=trainset, valset=valset, num_trials=10, minibatch_size=100)
    else:  # gepa
        from dspy.teleprompt.gepa import GEPA
        reflection_lm = dspy.LM("groq/llama-3.3-70b-versatile", api_key=groq_api_key)
        opt = GEPA(
            metric=config["metric"],
            max_metric_calls=rollout_budget,
            reflection_lm=reflection_lm,
            reflection_minibatch_size=3,
            track_stats=True,
        )
        optimized_module = opt.compile(student=module, trainset=trainset, valset=valset)

    # Evaluate optimized
    final_result = evaluate(optimized_module)

    if hasattr(final_result, "score"):
        final_score = float(final_result.score) / 100.0
    else:
        final_score = float(final_result) / 100.0 if final_result > 1 else float(final_result)

    print(f"✅ Final: {final_score:.4f} (Δ {final_score - baseline_score:+.4f})")

    tracker.curve.record(rollout_count=rollout_budget, performance=final_score, checkpoint_pct=1.0)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    tracker.save(output_dir)

    results = {
        "benchmark": benchmark,
        "num_steps": num_steps,
        "optimizer": optimizer,
        "baseline_score": baseline_score,
        "final_score": final_score,
        "improvement": final_score - baseline_score,
        "train_n": len(trainset),
        "val_n": len(valset),
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"📁 Saved to: {output_dir}")

    return results


async def main():
    """Run all scaling experiments."""

    base_dir = Path(__file__).parent / "results"

    # Define experiments
    benchmarks = ["banking77", "heartdisease", "hotpotqa"]
    pipeline_steps = [1, 2, 3, 5]
    optimizers = ["gepa", "mipro"]

    all_results = []

    for benchmark in benchmarks:
        for steps in pipeline_steps:
            # Skip 2-step for now (only implemented for banking77 via task app)
            if steps == 2:
                continue

            for optimizer in optimizers:
                output_dir = base_dir / benchmark / f"{steps}step" / optimizer

                try:
                    result = await run_single_experiment(
                        benchmark=benchmark,
                        num_steps=steps,
                        optimizer=optimizer,
                        output_dir=output_dir,
                        rollout_budget=200,
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"❌ Failed: {benchmark} {steps}-step {optimizer}: {e}")
                    import traceback
                    traceback.print_exc()

    # Save aggregate results
    aggregate_file = base_dir / "aggregate_results.json"
    with open(aggregate_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✅ All experiments complete!")
    print(f"📁 Aggregate results: {aggregate_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    asyncio.run(main())
