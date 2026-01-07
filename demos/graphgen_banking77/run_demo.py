#!/usr/bin/env python3
"""Run the Banking77 Graph Optimization demo end-to-end.

Usage:
    uv run python demos/graphgen_banking77/run_demo.py                    # Production mode (default)
    uv run python demos/graphgen_banking77/run_demo.py --local            # Local mode (localhost:8000)
    uv run python demos/graphgen_banking77/run_demo.py --local --local-host 127.0.0.1
    uv run python demos/graphgen_banking77/run_demo.py --prod             # Explicit production mode
"""

import argparse
import os
import random
import sys

# Parse args early so we can configure before imports
parser = argparse.ArgumentParser(description="Run Banking77 Graph Optimization demo")
parser.add_argument(
    "--local",
    action="store_true",
    help="Run in local mode: use localhost:8000 backend",
)
parser.add_argument(
    "--prod",
    action="store_true",
    help="Run in production mode: use production backend (default if --local not specified)",
)
parser.add_argument(
    "--local-host",
    type=str,
    default="localhost",
    help="Hostname for local API URLs (use 'host.docker.internal' if backend runs in Docker)",
)
args = parser.parse_args()

# Determine mode: --local takes precedence, then --prod, then default to prod
LOCAL_MODE = args.local
if not LOCAL_MODE and not args.prod:
    # Default to prod if neither specified
    pass

LOCAL_HOST = args.local_host

# Imports
import json
import time
from pathlib import Path

from datasets import load_dataset

# Synth SDK imports
from synth_ai.core.env import PROD_BASE_URL, mint_demo_api_key
from synth_ai.sdk.api.train.graphgen import GraphGenJob
from synth_ai.sdk.api.train.graphgen_models import (
    GraphGenTaskSet,
    GraphGenTaskSetMetadata,
    GraphGenTask,
    GraphGenGoldOutput,
    GraphGenVerifierConfig,
)

# Backend configuration
if LOCAL_MODE:
    SYNTH_API_BASE = "http://localhost:8000"
    print("=" * 60)
    print("RUNNING IN LOCAL MODE")
    print("=" * 60)
else:
    SYNTH_API_BASE = PROD_BASE_URL
    print("=" * 60)
    print("RUNNING IN PRODUCTION MODE")
    print(f"Backend: {SYNTH_API_BASE}")
    print("=" * 60)

print(f'Backend: {SYNTH_API_BASE}')

# Check backend health
import httpx

r = httpx.get(f'{SYNTH_API_BASE}/health', timeout=30)
if r.status_code == 200:
    print(f'Backend health: {r.json()}')
else:
    print(f'WARNING: Backend returned status {r.status_code}')
    raise RuntimeError(f'Backend not healthy: status {r.status_code}')

# Get API Key
API_KEY = os.environ.get('SYNTH_API_KEY', '')
if not API_KEY:
    print('No SYNTH_API_KEY found, minting demo key...')
    API_KEY = mint_demo_api_key()
    print(f'Demo API Key: {API_KEY[:25]}...')
else:
    print(f'Using SYNTH_API_KEY: {API_KEY[:20]}...')

# Set API key in environment for SDK to use
os.environ['SYNTH_API_KEY'] = API_KEY


def build_banking77_graphgen_dataset(num_train_tasks: int = 50, num_test_tasks: int = 20) -> GraphGenTaskSet:
    """Build a GraphGenTaskSet from the Banking77 dataset.
    
    Args:
        num_train_tasks: Number of training tasks to include
        num_test_tasks: Number of test tasks to include
        
    Returns:
        GraphGenTaskSet ready for graph optimization
    """
    print(f'\nLoading Banking77 dataset...')
    train_ds = load_dataset("banking77", split="train", trust_remote_code=False)
    test_ds = load_dataset("banking77", split="test", trust_remote_code=False)
    
    # Get label names
    label_names = train_ds.features["label"].names if hasattr(train_ds.features.get("label"), "names") else []
    
    print(f'  Train size: {len(train_ds)}')
    print(f'  Test size: {len(test_ds)}')
    print(f'  Labels: {len(label_names)}')
    
    # Shuffle ALL possible seeds from the full dataset, then select from shuffled pool (like GEPA demo)
    # This ensures we get diverse classes, not just sequential examples
    TOTAL_DATASET_SIZE = min(10000, len(train_ds))  # Use up to 10k from train split (like GEPA)
    TOTAL_TEST_SIZE = min(5000, len(test_ds))  # Use up to 5k from test split
    
    # Create shuffled indices from the FULL dataset (not just the subset we need)
    all_train_indices = list(range(TOTAL_DATASET_SIZE))
    all_test_indices = list(range(TOTAL_TEST_SIZE))
    random.shuffle(all_train_indices)
    random.shuffle(all_test_indices)
    
    # Select from the shuffled pools
    selected_train_indices = all_train_indices[:num_train_tasks]
    selected_test_indices = all_test_indices[:num_test_tasks]
    
    # Create examples from shuffled indices
    all_examples = []
    for dataset_idx in selected_train_indices:
        row = train_ds[dataset_idx]
        label_idx = int(row.get("label", 0))
        label_text = label_names[label_idx] if label_idx < len(label_names) else f"label_{label_idx}"
        all_examples.append({
            "query": str(row.get("text", "")),
            "intent": label_text,
            "source": "train",
            "dataset_idx": dataset_idx,
        })
    
    for dataset_idx in selected_test_indices:
        row = test_ds[dataset_idx]
        label_idx = int(row.get("label", 0))
        label_text = label_names[label_idx] if label_idx < len(label_names) else f"label_{label_idx}"
        all_examples.append({
            "query": str(row.get("text", "")),
            "intent": label_text,
            "source": "test",
            "dataset_idx": dataset_idx,
        })
    
    # Shuffle the combined examples one more time to mix train/test
    random.shuffle(all_examples)
    
    # Build tasks and gold outputs from shuffled examples
    tasks = []
    gold_outputs = []
    
    for task_idx, example in enumerate(all_examples):
        task_id = f"task_{task_idx}"
        
        tasks.append(GraphGenTask(
            id=task_id,
            input={"query": example["query"]},
        ))
        
        gold_outputs.append(GraphGenGoldOutput(
            task_id=task_id,
            output={"intent": example["intent"]},
        ))
    
    # Problem specification for the graph proposer
    # Include ALL valid classification labels so the optimizer knows what options are valid
    problem_spec = (
        "You are building a banking intent classification system. "
        "Given a customer query, classify it into one of 77 banking intents. "
        "The system should use a single LLM call to analyze the query and return the intent label. "
        "\n\n"
        "VALID INTENT LABELS (must return exactly one of these):\n"
        + "\n".join([f"  - {label}" for label in label_names])
        + "\n\n"
        "The output must be a JSON object with an 'intent' field containing one of the above labels."
    )
    
    # Define schemas
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Customer banking query to classify"
            }
        },
        "required": ["query"]
    }
    
    output_schema = {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "description": "Predicted banking intent label",
                "enum": label_names
            }
        },
        "required": ["intent"]
    }
    
    dataset = GraphGenTaskSet(
        metadata=GraphGenTaskSetMetadata(
            name="Banking77 Intent Classification",
            description="Banking77 dataset for intent classification graph optimization",
            input_schema=input_schema,
            output_schema=output_schema,
        ),
        tasks=tasks,
        gold_outputs=gold_outputs,
        verifier_config=GraphGenVerifierConfig(
            mode="rubric",
            model="gpt-4.1-mini",
            provider="openai",
        ),
        # Also include at top level for backward compatibility
        input_schema=input_schema,
        output_schema=output_schema,
        # Extract only "intent" field from final_state for reward computation
        # This ensures output format matches expected format {"intent": "..."}
        select_output="intent",
    )
    
    print(f'\nCreated GraphGenTaskSet:')
    print(f'  Tasks: {len(tasks)} ({num_train_tasks} train, {num_test_tasks} test)')
    print(f'  Gold outputs: {len(gold_outputs)}')
    print(f'  Verifier mode: {dataset.verifier_config.mode}')
    
    return dataset


def main():
    """Main function to run the graph optimization demo."""
    
    # Timing helper
    def format_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        mins, secs = divmod(int(seconds), 60)
        return f"{mins}m {secs}s"
    
    timings: dict[str, float] = {}
    total_start = time.time()
    
    # Build dataset
    dataset = build_banking77_graphgen_dataset(num_train_tasks=50, num_test_tasks=20)
    
    # Create GraphGen job
    print('\n' + '=' * 60)
    print('CREATING GRAPH OPTIMIZATION JOB')
    print('=' * 60)
    
    # Build problem spec with all valid labels from the dataset
    label_names = dataset.metadata.output_schema.get("properties", {}).get("intent", {}).get("enum", [])
    if not label_names:
        # Fallback: extract from gold outputs if enum not in schema
        label_names = sorted(set(gold.output.get("intent") for gold in dataset.gold_outputs if gold.output.get("intent")))
    
    problem_spec = (
        "You are building a banking intent classification system. "
        "Given a customer query, classify it into one of the valid banking intent categories. "
        "The system should use a single LLM call to analyze the query and return the intent label. "
        "\n\n"
        "VALID INTENT LABELS (must return exactly one of these):\n"
        + "\n".join([f"  - {label}" for label in label_names])
        + "\n\n"
        "The output must be a JSON object with an 'intent' field containing one of the above labels."
    )
    
    job = GraphGenJob.from_dataset(
        dataset=dataset,
        policy_model="gpt-4.1-mini",
        rollout_budget=100,
        proposer_effort="medium",
        num_generations=2,
        problem_spec=problem_spec,
        backend_url=SYNTH_API_BASE,
        api_key=API_KEY,
        auto_start=True,
    )
    
    print(f'\nJob Configuration:')
    print(f'  Graph type: {job.config.graph_type}')
    print(f'  Policy model: {job.config.policy_model}')
    print(f'  Rollout budget: {job.config.rollout_budget}')
    print(f'  Proposer effort: {job.config.proposer_effort}')
    print(f'  Generations: {job.config.num_generations}')
    
    # Submit job
    print(f'\nSubmitting job...')
    job_id = job.submit()
    print(f'Job ID: {job_id}')
    
    # Stream until complete
    print(f'\n' + '=' * 60)
    print('OPTIMIZATION IN PROGRESS')
    print('=' * 60)
    print('Streaming events...\n')
    
    optimization_start = time.time()
    
    result = job.stream_until_complete(
        timeout=3600.0,
        interval=3.0,
    )
    
    timings['optimization'] = time.time() - optimization_start
    
    print(f'\n' + '=' * 60)
    print('RESULTS')
    print('=' * 60)
    
    if result.get('status') == 'succeeded':
        best_score = result.get('best_score', 0.0)
        print(f'Status: SUCCEEDED')
        print(f'Best Score: {best_score:.2%}')
        print(f'Duration: {format_duration(timings["optimization"])}')
        
        # Download optimized graph
        print(f'\nDownloading optimized graph...')
        try:
            graph_txt = job.download_graph_txt()
            print(f'\nOptimized Graph (first 500 chars):')
            print('=' * 60)
            print(graph_txt[:500] + '...' if len(graph_txt) > 500 else graph_txt)
            print('=' * 60)
        except Exception as e:
            print(f'Could not download graph: {e}')
        
        # Run inference example
        print(f'\nRunning inference example...')
        try:
            example_input = {"query": "How do I activate my card?"}
            output = job.run_inference(example_input)
            print(f'  Input: {example_input}')
            print(f'  Output: {output}')
        except Exception as e:
            print(f'  Could not run inference: {e}')
    else:
        status = result.get('status', 'unknown')
        error = result.get('error', 'Unknown error')
        print(f'Status: {status.upper()}')
        print(f'Error: {error}')
    
    # Timing summary
    timings['total'] = time.time() - total_start
    print('\n' + '=' * 60)
    print('TIMING SUMMARY')
    print('=' * 60)
    print(f"  Optimization:  {format_duration(timings['optimization'])}")
    print(f"  ─────────────────────────")
    print(f"  Total:         {format_duration(timings['total'])}")
    
    print('\nDemo complete!')


if __name__ == "__main__":
    main()

