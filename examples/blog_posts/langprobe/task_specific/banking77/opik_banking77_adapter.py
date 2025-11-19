"""Opik adapters for Banking77 intent classification using MetaPromptOptimizer."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import litellm
from datasets import load_dataset
from dotenv import load_dotenv
from opik_optimizer import ChatPrompt, MetaPromptOptimizer
# Note: Using custom metric function instead of ExactMatch import

# Note: LearningCurveTracker not needed for Opik adapters

load_dotenv()


def load_banking77_dataset(split: str = "train") -> list[dict[str, Any]]:
    """Load Banking77 dataset from HuggingFace and convert to list of dicts.

    Args:
        split: Dataset split (default: "train")

    Returns:
        List of examples with query and intent label
    """
    dataset = load_dataset("banking77", split=split, trust_remote_code=False)
    examples = []

    # Get label names
    label_names = dataset.features["label"].names if hasattr(dataset.features.get("label"), "names") else []

    for idx, row in enumerate(dataset):
        label_idx = int(row.get("label", 0))
        label_name = label_names[label_idx] if label_idx < len(label_names) else str(label_idx)
        query = str(row.get("text", ""))

        examples.append({
            "query": query,
            "intent": label_name,
            "label_idx": label_idx,
            "seed": idx,  # Use index as seed
        })

    return examples


def get_available_intents() -> list[str]:
    """Get list of available intents from Banking77 dataset."""
    dataset = load_dataset("banking77", split="train", trust_remote_code=False)
    if hasattr(dataset.features.get("label"), "names"):
        return dataset.features["label"].names
    return [str(i) for i in range(77)]


def banking77_metric(item: dict[str, Any], output: str) -> float:
    """Metric function for Banking77 intent classification.

    Args:
        item: Item dict with 'answer' key containing expected intent
        output: Model output string

    Returns:
        Score (1.0 if correct, 0.0 if incorrect)
    """
    expected = str(item.get("answer", "")).strip().lower()
    predicted = str(output).strip().lower()
    
    # Simple exact match (case-insensitive)
    return 1.0 if expected == predicted else 0.0


async def run_opik_hierarchical_banking77(
    task_app_url: str = "http://127.0.0.1:8102",
    train_seeds: Optional[list[int]] = None,
    val_seeds: Optional[list[int]] = None,
    rollout_budget: int = 200,
    output_dir: Optional[Path] = None,
    model: Optional[str] = None,
    max_trials: int = 5,
    n_samples: int = 100,
) -> dict[str, Any]:
    """Run Opik MetaPromptOptimizer on Banking77.

    Args:
        task_app_url: Task app URL (for reference, not used directly)
        train_seeds: Training seeds (default: 0-49, 50 examples)
        val_seeds: Validation seeds (default: 50-79, 30 examples)
        rollout_budget: Rollout budget (approximate, via max_trials and n_samples)
        output_dir: Output directory
        model: Model name (default: groq/llama-3.1-8b-instant)
        max_trials: Maximum optimization trials
        n_samples: Number of samples per trial

    Returns:
        Results dictionary
    """
    start_time = time.time()

    if output_dir is None:
        output_dir = Path(__file__).parent / "results" / "opik_hierarchical"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure model
    if model is None:
        model = "groq/llama-3.1-8b-instant"
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY required")
    
    # Set LiteLLM API key
    os.environ["GROQ_API_KEY"] = groq_api_key

    # Load dataset
    banking77_examples = load_banking77_dataset(split="train")
    available_intents = get_available_intents()

    # Select training and validation seeds (matching other adapters)
    if train_seeds is None:
        train_seeds = list(range(50))  # 0-49: 50 training examples
    if val_seeds is None:
        val_seeds = list(range(50, 80))  # 50-79: 30 validation examples

    # Filter examples by seeds
    train_examples = [banking77_examples[i] for i in train_seeds if i < len(banking77_examples)]
    val_examples = [banking77_examples[i] for i in val_seeds if i < len(banking77_examples)]

    # Create local dataset using Opik Dataset class with minimal setup
    # We need to create a Dataset object even for local-only runs
    import opik
    from opik.api_objects.dataset.dataset import Dataset
    
    available_intents_str = ", ".join(available_intents)
    items = [
        {
            "question": f"Query: {ex['query']}\n\nAvailable intents: {available_intents_str}",
            "answer": ex["intent"],
            "metadata": {"seed": ex["seed"], "label_idx": ex["label_idx"]}
        }
        for ex in train_examples
    ]
    
    # Create Opik client in offline/local mode (will warn but work)
    try:
        client = opik.Opik()
        dataset_name = f"banking77_local_{int(time.time())}"
        dataset = client.get_or_create_dataset(
            name=dataset_name,
            description="Banking77 local dataset"
        )
        dataset.insert(items)
    except Exception as e:
        # If Opik client fails, create Dataset directly with mock rest_client
        # This is a workaround for local-only runs
        from unittest.mock import MagicMock
        
        # Create a mock rest_client with all necessary attributes
        mock_rest_client = MagicMock()
        mock_datasets = MagicMock()
        # Mock the datasets.stream_dataset_items method to return our local items
        mock_stream = MagicMock()
        mock_stream.__iter__ = lambda self: iter(items)
        mock_stream.__next__ = lambda self: next(iter(items)) if items else None
        mock_datasets.stream_dataset_items = MagicMock(return_value=mock_stream)
        mock_rest_client.datasets = mock_datasets
        
        dataset = Dataset(
            name=f"banking77_local_{int(time.time())}",
            description="Banking77 local dataset",
            rest_client=mock_rest_client
        )
        # Monkey-patch get_items to return our local items directly
        def get_items_patch(self, n_samples=None):
            from opik.api_objects.dataset.dataset_item import DatasetItem
            items_to_return = items[:n_samples] if n_samples else items
            return [DatasetItem(**item) for item in items_to_return]
        dataset.get_items = get_items_patch.__get__(dataset, Dataset)

    # Create initial prompt template
    prompt = ChatPrompt(
        project_name="langprobe-opik-banking77",
        messages=[
            {
                "role": "system",
                "content": "You are a precise assistant for intent classification in banking customer service queries. Classify the intent of the given query from the available intents."
            },
            {
                "role": "user",
                "content": "{question}"
            }
        ]
    )

    # Create optimizer
    # Note: Using MetaPromptOptimizer as HierarchicalReflectiveOptimizer may not be available
    # Update to HierarchicalReflectiveOptimizer when available in opik-optimizer package
    # MetaPromptOptimizer uses 'rounds' instead of 'max_trials'
    
    # Mock Opik client creation to avoid cloud API calls
    import opik
    from unittest.mock import patch, MagicMock
    
    # Create a comprehensive mock that handles all Opik API calls
    def create_mock_opik_client():
        mock_client = MagicMock()
        mock_experiment = MagicMock()
        mock_experiment.key = "local-test-experiment"
        mock_experiment.url = "http://localhost/local"
        mock_client.create_experiment = MagicMock(return_value=mock_experiment)
        mock_client.get_or_create_dataset = MagicMock(return_value=dataset)  # Use our local dataset
        return mock_client
    
    # Patch Opik client creation globally
    original_opik_init = opik.Opik
    opik.Opik = lambda *args, **kwargs: create_mock_opik_client()
    
    try:
        optimizer = MetaPromptOptimizer(
            model=model,
            n_threads=8,
            rounds=max_trials,  # Use max_trials as rounds
            num_prompts_per_round=2,  # Small number for testing
        )
    finally:
        # Restore original
        opik.Opik = original_opik_init

    # Evaluate baseline on validation set
    print(f"📊 Evaluating baseline on {len(val_examples)} validation examples...")
    baseline_scores = []
    for ex in val_examples:
        try:
            # Format prompt with available intents (matching training format)
            user_content = f"Query: {ex['query']}\n\nAvailable intents: {available_intents_str}"
            
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": prompt.messages[0]["content"]},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.0
            )
            
            output = response.choices[0].message.content
            score = banking77_metric({"answer": ex["intent"]}, output)
            baseline_scores.append(score)
        except Exception as e:
            print(f"Warning: Baseline evaluation failed for example: {e}")
            baseline_scores.append(0.0)
    
    baseline_score = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0
    print(f"Baseline score: {baseline_score:.4f}")

    # Run optimization
    print(f"🚀 Starting Opik MetaPromptOptimizer optimization...")
    print(f"   Model: {model}")
    print(f"   Max trials: {max_trials}")
    print(f"   N samples: {n_samples}")
    
    # Patch Opik client creation during optimization to avoid cloud calls
    from unittest.mock import patch
    
    def create_mock_opik_client():
        mock_client = MagicMock()
        mock_experiment = MagicMock()
        mock_experiment.key = "local-test-experiment"
        mock_experiment.url = "http://localhost/local"
        mock_client.create_experiment = MagicMock(return_value=mock_experiment)
        return mock_client
    
    # Patch at multiple levels where Opik client might be created
    with patch('opik.Opik', side_effect=create_mock_opik_client), \
         patch('opik_optimizer.task_evaluator.opik.Opik', side_effect=create_mock_opik_client), \
         patch('opik.evaluation.evaluator.opik.Opik', side_effect=create_mock_opik_client):
        result = optimizer.optimize_prompt(
            prompt=prompt,
            dataset=dataset,
            metric=banking77_metric,
            n_samples=n_samples,  # MetaPromptOptimizer uses n_samples, not max_trials
        )

    # Evaluate optimized prompt on validation set
    print(f"📊 Evaluating optimized prompt on {len(val_examples)} validation examples...")
    best_prompt = getattr(result, "prompt", prompt)
    optimized_scores = []
    
    for ex in val_examples:
        try:
            # Format prompt with available intents (matching training format)
            user_content = f"Query: {ex['query']}\n\nAvailable intents: {available_intents_str}"
            
            # Get the user message template from optimized prompt
            user_template = best_prompt.messages[1]["content"] if len(best_prompt.messages) > 1 else "{question}"
            if "{question}" in user_template:
                user_content = user_template.format(question=user_content)
            
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": best_prompt.messages[0]["content"]},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.0
            )
            
            output = response.choices[0].message.content
            score = banking77_metric({"answer": ex["intent"]}, output)
            optimized_scores.append(score)
        except Exception as e:
            print(f"Warning: Optimized evaluation failed for example: {e}")
            optimized_scores.append(0.0)
    
    best_score = sum(optimized_scores) / len(optimized_scores) if optimized_scores else 0.0
    print(f"Optimized score: {best_score:.4f}")

    elapsed_time = time.time() - start_time

    # Save results
    results = {
        "benchmark": "banking77",
        "optimizer": "opik_metaprompt",
        "model": model,
        "baseline_score": baseline_score,
        "best_score": best_score,
        "lift": best_score - baseline_score,
        "total_rollouts": getattr(result, "n_samples", n_samples) * getattr(optimizer, "rounds", max_trials),
        "elapsed_time": elapsed_time,
        "train_seeds": train_seeds,
        "val_seeds": val_seeds,
        "max_trials": max_trials,
        "n_samples": n_samples,
    }

    # Save best prompt
    best_prompt_dict = {}
    try:
        msgs = getattr(best_prompt, "messages", None)
        if msgs is None and hasattr(best_prompt, "model_dump"):
            msgs = best_prompt.model_dump().get("messages", None)
        best_prompt_dict = {"messages": msgs if msgs is not None else str(best_prompt)}
    except Exception as e:
        best_prompt_dict = {"prompt_repr": repr(best_prompt), "error": str(e)}

    results_file = output_dir / "opik_hierarchical_banking77_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    prompt_file = output_dir / "opik_hierarchical_banking77_best_prompt.json"
    with open(prompt_file, "w") as f:
        json.dump(best_prompt_dict, f, indent=2)

    print(f"\n✅ Opik MetaPromptOptimizer complete!")
    print(f"   Baseline: {baseline_score:.4f}")
    print(f"   Best: {best_score:.4f}")
    print(f"   Lift: {results['lift']:.4f}")
    print(f"   Time: {elapsed_time:.1f}s")
    print(f"   Results saved to: {results_file}")

    return results

