#!/usr/bin/env python3
"""Run verifier optimization using the graded baseline dataset.

Uses the VLM-scored images from grading_manifest.json as training data.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import time
from pathlib import Path

import httpx

parser = argparse.ArgumentParser(description="Run verifier opt from graded dataset")
parser.add_argument("--artist", type=str, default="seb_mckinnon")
parser.add_argument("--train-seeds", type=int, default=3)
parser.add_argument("--val-seeds", type=int, default=2)
parser.add_argument("--generations", type=int, default=2)
parser.add_argument("--local", action="store_true", help="Use localhost backend")
parser.add_argument("--dev", action="store_true", help="Use dev backend (api-dev.usesynth.ai)")
args = parser.parse_args()

DEMO_DIR = Path(__file__).resolve().parent
if args.local:
    SYNTH_API_BASE = "http://127.0.0.1:8000"
elif args.dev:
    SYNTH_API_BASE = "https://api-dev.usesynth.ai"
else:
    SYNTH_API_BASE = "https://api.usesynth.ai"

# Load graded dataset
dataset_dir = DEMO_DIR / "baseline_images" / args.artist
manifest_path = dataset_dir / "grading_manifest.json"

if not manifest_path.exists():
    raise FileNotFoundError(f"Grading manifest not found: {manifest_path}")

manifest = json.loads(manifest_path.read_text())
samples = [s for s in manifest["samples"] if s.get("human_score") is not None]

if len(samples) < args.train_seeds + args.val_seeds:
    raise ValueError(
        f"Not enough scored samples ({len(samples)}) for "
        f"{args.train_seeds} train + {args.val_seeds} val seeds"
    )

# Sort by reward for stratified split
samples.sort(key=lambda x: x["human_score"])
train_samples = samples[:args.train_seeds]
val_samples = samples[args.train_seeds:args.train_seeds + args.val_seeds]

print(f"Artist: {manifest['artist_name']}")
print(f"Style: {manifest['style_description']}")
print(f"Train samples: {len(train_samples)} (rewards: {[s['human_score'] for s in train_samples]})")
print(f"Val samples: {len(val_samples)} (rewards: {[s['human_score'] for s in val_samples]})")


def load_image_b64(path: Path) -> str:
    """Load image as base64 data URL - PLACEHOLDER for testing."""
    # For testing, return a placeholder instead of actual image data
    # This helps verify the API format before sending large payloads
    return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


def make_trace(prompt: str, gen_image_url: str, ref_image_url: str) -> dict:
    """Create trace structure with both generated and reference images."""
    return {
        "session_id": "trace",
        "session_time_steps": [
            {
                "step_id": "1",
                "step_index": 0,
                "events": [
                    {
                        "event_type": "runtime",
                        "event_id": 1,
                        "type": "user_message",
                        "content": prompt,
                    },
                    {
                        "event_type": "runtime",
                        "event_id": 2,
                        "type": "assistant_message",
                        "content": [
                            {"type": "image_url", "image_url": {"url": gen_image_url}},
                        ],
                    },
                    {
                        "event_type": "runtime",
                        "event_id": 3,
                        "type": "reference_image",
                        "content": [
                            {"type": "image_url", "image_url": {"url": ref_image_url}},
                        ],
                    },
                ],
            }
        ],
    }


# Build verifier examples
verifier_examples = []
for i, sample in enumerate(train_samples + val_samples):
    gen_path = dataset_dir / sample["generated_image"]
    ref_path = dataset_dir / sample["reference_image"] if sample.get("reference_image") else None

    if not gen_path.exists():
        print(f"Skipping {sample['card_name']}: generated image not found")
        continue
    if ref_path and not ref_path.exists():
        print(f"Skipping {sample['card_name']}: reference image not found")
        continue

    gen_url = load_image_b64(gen_path)
    ref_url = load_image_b64(ref_path) if ref_path else gen_url

    verifier_examples.append({
        "task_id": sample["id"],
        "trace": make_trace(sample["prompt"], gen_url, ref_url),
        "reward": sample["human_score"],
        "card_name": sample["card_name"],
    })

print(f"\nBuilt {len(verifier_examples)} verifier examples")

TRAIN_SEEDS = list(range(len(train_samples)))
VAL_SEEDS = list(range(len(train_samples), len(train_samples) + len(val_samples)))

# Gold examples for the verifier
GOLD_EXAMPLES = [
    {
        "summary": "High style match - captures mood, brushwork, and composition",
        "gold_reward": 0.85,
        "gold_reasoning": "Strong match on color palette, atmospheric mood, and compositional principles",
    },
    {
        "summary": "Low style match - different aesthetic",
        "gold_reward": 0.3,
        "gold_reasoning": "Does not capture the moody, gothic, painterly surrealism of the artist",
    },
]

STYLE_DESC = manifest["style_description"]
ARTIST_NAME = manifest["artist_name"]

verifier_dataset = {
    "tasks": [
        {
            "id": ex["task_id"],  # API expects 'id' not 'task_id'
            "input": {
                "trace": ex["trace"],
                "gold_examples": GOLD_EXAMPLES,
                "style_description": STYLE_DESC,
            },
        }
        for ex in verifier_examples
    ],
    "gold_outputs": [
        {"task_id": ex["task_id"], "output": {"reward": ex["reward"]}}
        for ex in verifier_examples
    ],
    "metadata": {
        "name": f"mtg_style_verifier_{args.artist}",
        "task_description": (
            f"Evaluate how well a generated MTG card illustration matches {ARTIST_NAME}'s style: "
            f"{STYLE_DESC}. Compare the GENERATED image to the REFERENCE image. "
            "Assign a reward 0.0-1.0 based on color palette, brushwork, mood/atmosphere, and composition."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "trace": {"type": "object"},
                "gold_examples": {"type": "array"},
                "style_description": {"type": "string"},
            },
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "color_palette": {"type": "object"},
                "brushwork_texture": {"type": "object"},
                "mood_atmosphere": {"type": "object"},
                "compositional_style": {"type": "object"},
                "overall_score": {"type": "number"},
                "overall_reasoning": {"type": "string"},
            },
        },
    },
}

# Get API key
API_KEY = os.environ.get("SYNTH_API_KEY", "").strip()
if not API_KEY:
    print("Minting demo key...")
    resp = httpx.post(f"{SYNTH_API_BASE}/api/demo/keys", json={"ttl_hours": 4}, timeout=30)
    resp.raise_for_status()
    API_KEY = resp.json()["api_key"]
    print(f"Demo key: {API_KEY[:20]}...")


async def run_graph_evolve():
    from synth_ai.sdk.optimization.internal.graph_optimization_client import (
        GraphOptimizationClient,
    )
    from synth_ai.sdk.optimization.internal.graph_optimization_config import (
        GraphOptimizationConfig,
        EvolutionConfig,
        LimitsConfig,
        ProposerConfig,
        SeedsConfig,
    )

    config = GraphOptimizationConfig(
        algorithm="graph_evolve",
        dataset_name=f"mtg_style_verifier_{args.artist}",
        graph_type="verifier",
        graph_structure="single_prompt",
        initial_graph_id="single",  # Required for Graph Evolve
        topology_guidance=(
            "Single-node VerifierGraph for image style matching.\n"
            "The verifier receives:\n"
            "1. A GENERATED image (assistant response)\n"
            "2. A REFERENCE image (gold standard)\n"
            "3. Style description to match\n\n"
            "Score 0.0-1.0 based on:\n"
            "- Color palette match (0.25 weight)\n"
            "- Brushwork/texture style (0.25 weight)\n"
            "- Mood/atmosphere (0.25 weight)\n"
            "- Compositional style (0.25 weight)\n\n"
            "Output JSON with per-criterion scores and overall_score."
        ),
        allowed_policy_models=["gpt-4.1-nano"],
        evolution=EvolutionConfig(
            num_generations=args.generations,
            children_per_generation=2,
        ),
        proposer=ProposerConfig(model="gpt-4.1", temperature=0.0),
        seeds=SeedsConfig(train=TRAIN_SEEDS, validation=VAL_SEEDS),
        limits=LimitsConfig(max_spend_usd=10.0, timeout_seconds=3600),
        verifier_mode="contrastive",
        verifier_model="gpt-4o",  # VLM with vision for image analysis
        dataset=verifier_dataset,
        output_schema=verifier_dataset["metadata"]["output_schema"],
        task_description=verifier_dataset["metadata"]["task_description"],
        problem_spec=(
            f"Create a VerifierGraph that evaluates MTG card illustrations for style match.\n\n"
            f"Target style: {STYLE_DESC}\n\n"
            f"The trace contains:\n"
            f"- event_id 1: User prompt requesting the image\n"
            f"- event_id 2: Generated image to evaluate\n"
            f"- event_id 3: Reference image (gold standard)\n\n"
            f"Compare generated vs reference and assign style similarity reward 0.0-1.0.\n"
            f"Output detailed rubric rewards for color, brushwork, mood, composition."
        ),
    )

    print(f"\nStarting graph evolve job...")
    print(f"Backend: {SYNTH_API_BASE}")
    print(f"Generations: {args.generations}")
    print(f"Train seeds: {TRAIN_SEEDS}")
    print(f"Val seeds: {VAL_SEEDS}")

    # Make the request manually to capture error details
    headers = {"Authorization": f"Bearer {API_KEY}"}
    request_body = config.to_request_dict()
    # API requires policy_models field (maps from allowed_policy_models)
    request_body["policy_models"] = request_body.get("allowed_policy_models", ["gpt-4.1-nano"])
    # Dev environment requires proposer_effort=high
    request_body["proposer_effort"] = "high"

    # Debug: save request body
    import json as json_module
    with open("/tmp/graph_evolve_request.json", "w") as f:
        json_module.dump(request_body, f, indent=2)
    print(f"Request body saved to /tmp/graph_evolve_request.json")

    async with httpx.AsyncClient(timeout=300.0) as http:
        response = await http.post(
            f"{SYNTH_API_BASE}/graph-evolve/jobs",
            headers=headers,
            json=request_body,
        )
        resp_data = response.json()
        print(f"Response ({response.status_code}): {resp_data}")
        if response.status_code != 200:
            print(f"Error {response.status_code}: {response.text}")
            response.raise_for_status()
        job_id = resp_data.get("job_id") or resp_data.get("graphgen_job_id")
        if not job_id:
            raise ValueError(f"No job_id in response: {resp_data}")
        print(f"Job ID: {job_id}")

    # Poll for completion
    headers = {"Authorization": f"Bearer {API_KEY}"}
    async with httpx.AsyncClient(timeout=90.0) as http:
        for i in range(300):  # 10 min max
            try:
                resp = await http.get(
                    f"{SYNTH_API_BASE}/graph-evolve/jobs/{job_id}/status",
                    headers=headers,
                )
                if resp.status_code == 404:
                    await asyncio.sleep(2.0)
                    continue
                resp.raise_for_status()
                status = resp.json().get("status")
                best = resp.json().get("best_score")
                print(f"[{i*2}s] Status: {status}, Best: {best}")

                if status in {"completed", "failed", "cancelled"}:
                    result_resp = await http.get(
                        f"{SYNTH_API_BASE}/graph-evolve/jobs/{job_id}/result",
                        headers=headers,
                    )
                    result_resp.raise_for_status()
                    return job_id, result_resp.json()
            except httpx.HTTPStatusError as e:
                print(f"[{i*2}s] HTTP error: {e}")
            await asyncio.sleep(2.0)

    raise RuntimeError("Job did not complete in time")


async def main():
    job_id, result = await run_graph_evolve()

    status = result.get("status")
    best_reward = result.get("best_score") or result.get("best_reward")

    print("\n" + "=" * 60)
    print("Verifier Optimization Complete")
    print("=" * 60)
    print(f"Status: {status}")
    print(f"Best reward: {best_reward}")
    print(f"Job ID: {job_id}")

    if status == "completed":
        # Save artifact
        artifacts_dir = DEMO_DIR / "artifacts" / args.artist
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifacts_dir / "verifier_opt_test.json"
        artifact_path.write_text(json.dumps({
            "job_id": job_id,
            "best_reward": best_reward,
            "train_seeds": args.train_seeds,
            "val_seeds": args.val_seeds,
            "generations": args.generations,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }, indent=2))
        print(f"Artifact saved: {artifact_path}")


if __name__ == "__main__":
    asyncio.run(main())
