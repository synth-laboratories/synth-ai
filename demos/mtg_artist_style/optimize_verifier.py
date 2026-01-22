#!/usr/bin/env python3
"""Optimize verifier using Graph Evolve on the calibration dataset.

Uses gpt-5-nano to optimize a verifier that:
- Takes an image + target artist style info
- Returns a score (0-1) for whether the image is by that artist
- Loss: |IS_ACTUALLY_ARTIST - VERIFIER_SCORE|

Usage:
    uv run python demos/mtg_artist_style/optimize_verifier.py
    uv run python demos/mtg_artist_style/optimize_verifier.py --local
"""

import argparse
import json
import os
import time
from pathlib import Path

import httpx
from synth_ai.sdk.optimization import GraphOptimizationJob

parser = argparse.ArgumentParser(description="Optimize MTG artist verifier with Graph Evolve")
parser.add_argument(
    "--local",
    action="store_true",
    help="Use local backend at localhost:8000",
)
parser.add_argument(
    "--dev",
    action="store_true",
    help="Use dev backend at api-dev.usesynth.ai",
)
parser.add_argument(
    "--model",
    type=str,
    default="gpt-4.1-nano",
    help="Model for the verifier (default: gpt-4.1-nano)",
)
parser.add_argument(
    "--rollout-budget",
    type=int,
    default=50,
    help="Rollout budget (default: 50)",
)
parser.add_argument(
    "--population-size",
    type=int,
    default=4,
    help="Population size (default: 4)",
)
parser.add_argument(
    "--generations",
    type=int,
    default=3,
    help="Number of generations (default: 3)",
)
parser.add_argument(
    "--initial-graph-id",
    type=str,
    default="zero_shot_verifier_rubric_rlm_v2",
    help=(
        "Preset graph ID to optimize (default: zero_shot_verifier_rubric_rlm_v2). "
        "Graph Evolve now requires a preset starter graph."
    ),
)
args = parser.parse_args()

demo_dir = Path(__file__).resolve().parent
synth_root = demo_dir.parents[1]


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("' ")
        if key:
            os.environ[key] = value


_load_env_file(synth_root / ".env")

USE_LOCAL_BACKEND = args.local
USE_DEV_BACKEND = args.dev
if USE_LOCAL_BACKEND:
    SYNTH_API_BASE = "http://127.0.0.1:8000"
elif USE_DEV_BACKEND:
    SYNTH_API_BASE = "https://api-dev.usesynth.ai"
else:
    SYNTH_API_BASE = "https://api.usesynth.ai"
os.environ["BACKEND_BASE_URL"] = SYNTH_API_BASE


def _validate_api_key(api_key: str) -> bool:
    if not api_key:
        return False
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = httpx.get(f"{SYNTH_API_BASE}/api/v1/me", headers=headers, timeout=10)
    except Exception:
        return False
    return resp.status_code == 200


print(f"Backend: {SYNTH_API_BASE}")

r = httpx.get(f"{SYNTH_API_BASE}/health", timeout=30)
if r.status_code != 200:
    raise RuntimeError(f"Backend not healthy: status {r.status_code}")
print(f"Backend health: {r.json()}")

API_KEY = os.environ.get("SYNTH_API_KEY", "").strip()
if not API_KEY or not _validate_api_key(API_KEY):
    print("SYNTH_API_KEY missing or invalid; minting demo key...")
    resp = httpx.post(f"{SYNTH_API_BASE}/api/demo/keys", json={"ttl_hours": 4}, timeout=30)
    resp.raise_for_status()
    API_KEY = resp.json()["api_key"]
    print(f"Demo API Key: {API_KEY[:25]}...")
else:
    print(f"Using SYNTH_API_KEY: {API_KEY[:20]}...")

os.environ["SYNTH_API_KEY"] = API_KEY

# Load calibration dataset
dataset_path = demo_dir / "verifier_calibration_dataset.json"
if not dataset_path.exists():
    raise FileNotFoundError(
        "Verifier calibration dataset not found. Run build_verifier_dataset.py first."
    )

with open(dataset_path) as f:
    calibration_data = json.load(f)

print(f"\nLoaded calibration dataset:")
print(f"  Train examples: {len(calibration_data['train_tasks'])} (using {min(20, len(calibration_data['train_tasks']))})")
print(f"  Validation examples: {len(calibration_data['val_tasks'])}")

# Build dataset in GraphEvolve format
# The dataset needs tasks and gold_outputs for contrastive evaluation
# Limit to 20 training examples for faster payload serialization
MAX_TRAIN_EXAMPLES = 20
train_tasks_subset = calibration_data["train_tasks"][:MAX_TRAIN_EXAMPLES]
train_gold_outputs_subset = calibration_data["train_gold_outputs"][:MAX_TRAIN_EXAMPLES]

dataset = {
    "version": "1.0",
    "metadata": calibration_data["metadata"],
    "tasks": [
        {"id": t.get("id") or t.get("task_id"), "input": t["input"]}
        for t in train_tasks_subset
    ],
    "gold_outputs": train_gold_outputs_subset,
    "input_schema": calibration_data["metadata"]["input_schema"],
    "output_schema": calibration_data["metadata"]["output_schema"],
    "initial_prompt": (
        "Analyze the image and determine if it matches the target artist's style.\n\n"
        "You will receive:\n"
        "- image_url: The image to analyze\n"
        "- target_artist_style: Description of the target artist's style\n"
        "- distinguishing_giveaways: Specific visual characteristics to look for\n\n"
        "Return a score from 0.0 to 1.0:\n"
        "- 1.0 = Definitely by this artist\n"
        "- 0.0 = Definitely NOT by this artist\n\n"
        "Look carefully for the distinguishing giveaways listed."
    ),
    "default_rubric": {
        "outcome": {
            "criteria": [
                {
                    "name": "artist_identification",
                    "description": "Correctly identify if image is by target artist",
                    "weight": 1.0,
                },
            ]
        }
    },
    "verifier_config": {
        "mode": "contrastive",
        "model": "gpt-4.1-nano",
        "provider": "openai",
    },
}

# Save as temp file for GraphEvolve
temp_dataset_path = demo_dir / "temp_verifier_dataset.json"
with open(temp_dataset_path, "w") as f:
    json.dump(dataset, f)

print(f"\nStarting Graph Evolve optimization...")
print(f"  Model: {args.model}")
print(f"  Rollout budget: {args.rollout_budget}")
print(f"  Population size: {args.population_size}")
print(f"  Generations: {args.generations}")
print(f"  Initial graph: {args.initial_graph_id}")

# Problem spec tells the graph proposer what we're optimizing
problem_spec = (
    "You are building a verifier that determines if an image matches a target MTG artist's style.\n\n"
    "INPUT (provided to the verifier):\n"
    "- image_url: Base64-encoded image to analyze\n"
    "- target_artist_style: Description of the target artist's distinctive style\n"
    "- distinguishing_giveaways: List of specific visual characteristics unique to this artist\n\n"
    "OUTPUT (verifier must return):\n"
    "- score: Float 0.0-1.0 (1.0 = definitely this artist, 0.0 = definitely not)\n\n"
    "CRITICAL INSTRUCTIONS:\n"
    "1. The verifier MUST analyze the actual image content\n"
    "2. Look for the specific giveaways - they are key differentiators between artists\n"
    "3. A high score (>0.8) requires finding MULTIPLE matching giveaways in the image\n"
    "4. A low score (<0.3) means the style clearly doesn't match or matches a different artist\n"
    "5. Be discriminating - different MTG artists have very distinct styles\n\n"
    "The loss function is |actual_label - predicted_score| where actual_label is 1.0 if "
    "the image is truly by the target artist and 0.0 if it's by a different artist."
)

# Create and run the job
job = GraphOptimizationJob.from_dataset(
    dataset=str(temp_dataset_path),
    policy_models=args.model,
    rollout_budget=args.rollout_budget,
    proposer_effort="high",
    population_size=args.population_size,
    num_generations=args.generations,
    problem_spec=problem_spec,
    graph_type="verifier",
    initial_graph_id=args.initial_graph_id,
    backend_url=SYNTH_API_BASE,
    api_key=API_KEY,
    auto_start=True,
)

# Submit the job
job_id = job.submit()
print(f"\nJob submitted: {job_id}")

# Stream events until complete for better logging/pass-through
print("\nStreaming optimization events...")
try:
    result = job.stream_until_complete(timeout=1800.0)
except Exception as exc:
    print(f"[stream] error: {exc}")
    print("Falling back to status polling...")
    result = job.poll_until_complete(timeout=1800.0, interval=5.0, progress=True)

print(f"\nOptimization complete!")
print(f"  Status: {result.status.value}")
print(f"  Best score: {result.best_score}")

if result.failed:
    print(f"  Error: {result.error}")
else:
    # Save artifact
    artifacts_dir = demo_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    artifact = {
        "job_id": result.job_id,
        "model": args.model,
        "best_score": result.best_score,
        "status": result.status.value,
        "rollout_budget": args.rollout_budget,
        "population_size": args.population_size,
        "generations": args.generations,
        "train_examples": len(calibration_data["train_tasks"]),
        "val_examples": len(calibration_data["val_tasks"]),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    
    # Try to get best yaml
    if result.succeeded:
        try:
            graph_txt = job.download_graph_txt()
            artifact["optimized_prompt"] = graph_txt
            print(f"\n{'=' * 60}")
            print("Optimized Verifier Prompt:")
            print("=" * 60)
            print(graph_txt[:2000])
            if len(graph_txt) > 2000:
                print("...")
        except Exception as e:
            print(f"  Could not download graph: {e}")
            try:
                prompt_txt = job.download_prompt()
                artifact["optimized_prompt"] = prompt_txt
                print(f"\n{'=' * 60}")
                print("Optimized Verifier Prompt (prompt snapshot):")
                print("=" * 60)
                print(prompt_txt[:2000])
                if len(prompt_txt) > 2000:
                    print("...")
            except Exception as prompt_err:
                print(f"  Could not download prompt snapshot: {prompt_err}")
    
    artifact_path = artifacts_dir / "optimized_verifier.json"
    with open(artifact_path, "w") as f:
        json.dump(artifact, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print("Verifier Optimization Complete")
    print("=" * 60)
    print(f"Job ID: {result.job_id}")
    print(f"Best Score: {result.best_score}")
    print(f"Artifact: {artifact_path}")

# Cleanup temp file
temp_dataset_path.unlink(missing_ok=True)
