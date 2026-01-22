#!/usr/bin/env python3
"""Run verifier optimization (Graph Evolve) for MTG artist style matching.

This optimizes a verifier that:
1. Compares generated images to reference art using contrastive VLM
2. Returns 0 if the prompt contains the artist's name

Usage:
    uv run python demos/mtg_artist_style/run_verifier_opt.py --artist seb_mckinnon
    uv run python demos/mtg_artist_style/run_verifier_opt.py --artist seb_mckinnon --local
"""

import argparse
import asyncio
import base64
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
from synth_ai.sdk.api.train.graph_optimization import (
    GraphOptimizationClient,
    GraphOptimizationConfig,
)
from synth_ai.sdk.api.train.graph_optimization_config import (
    EvolutionConfig,
    LimitsConfig,
    ProposerConfig,
    SeedsConfig,
)

parser = argparse.ArgumentParser(description="Run MTG artist verifier optimization")
parser.add_argument(
    "--artist",
    type=str,
    default="seb_mckinnon",
    help="Artist key (see README for full list of 18 artists)",
)
parser.add_argument(
    "--local",
    action="store_true",
    help="Run in local mode: use localhost:8000 backend",
)
parser.add_argument(
    "--out",
    type=str,
    default=None,
    help="Path to write verifier optimization artifact JSON",
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
SYNTH_API_BASE = "http://127.0.0.1:8000" if USE_LOCAL_BACKEND else "https://api.usesynth.ai"
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
    print("SYNTH_API_KEY missing or invalid for this backend; minting demo key...")
    resp = httpx.post(f"{SYNTH_API_BASE}/api/demo/keys", json={"ttl_hours": 4}, timeout=30)
    resp.raise_for_status()
    API_KEY = resp.json()["api_key"]
    print(f"Demo API Key: {API_KEY[:25]}...")
else:
    print(f"Using SYNTH_API_KEY: {API_KEY[:20]}...")

os.environ["SYNTH_API_KEY"] = API_KEY

VERIFIER_MODEL = "gpt-4.1-nano"

# Load artist metadata
metadata_path = demo_dir / "artist_metadata.json"
if not metadata_path.exists():
    raise FileNotFoundError(
        f"Artist metadata not found. Run fetch_artist_cards.py first:\n"
        f"  uv run python demos/mtg_artist_style/fetch_artist_cards.py"
    )

with open(metadata_path) as f:
    artist_metadata = json.load(f)

artist_info = artist_metadata["artists"].get(args.artist)
if not artist_info:
    available = list(artist_metadata["artists"].keys())
    raise ValueError(f"Unknown artist '{args.artist}'. Available: {available}")

ARTIST_NAME = artist_info["name"]
STYLE_DESCRIPTION = artist_info["style_description"]
ARTIST_KEY = args.artist

# Get cards for this artist
artist_cards = [c for c in artist_metadata["cards"] if c["artist_key"] == ARTIST_KEY]

print(f"\nArtist: {ARTIST_NAME}")
print(f"Style: {STYLE_DESCRIPTION}")
print(f"Reference cards: {len(artist_cards)}")

# Build forbidden patterns
name_parts = ARTIST_NAME.lower().split()
FORBIDDEN_PATTERNS = [
    ARTIST_NAME.lower(),
    ARTIST_NAME.lower().replace(" ", ""),
    ARTIST_NAME.lower().replace(" ", "-"),
    ARTIST_NAME.lower().replace(" ", "_"),
]
if len(name_parts) > 1:
    FORBIDDEN_PATTERNS.append(name_parts[-1])

print(f"Forbidden patterns: {FORBIDDEN_PATTERNS}")


def _load_image_as_data_url(image_path: str) -> str:
    """Load an image file as a base64 data URL."""
    full_path = demo_dir / image_path
    with open(full_path, "rb") as f:
        img_data = f.read()
    ext = full_path.suffix.lower()
    mime_type = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(
        ext, "image/jpeg"
    )
    return f"data:{mime_type};base64,{base64.b64encode(img_data).decode('ascii')}"


def _make_trace(prompt: str, image_url: str) -> dict[str, Any]:
    """Create a trace structure for verifier training."""
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
                        "content": [{"type": "image_url", "image_url": {"url": image_url}}],
                    },
                ],
            }
        ],
    }


# Create verifier examples
# Good examples: prompts without artist name + good style match
# Bad examples: prompts WITH artist name (should get 0) + poor style prompts
VERIFIER_EXAMPLES = []

# Good: descriptive style prompts (no artist name)
good_prompts = [
    f"Generate {STYLE_DESCRIPTION.lower()}",
    f"Create a moody fantasy scene with painterly surrealism",
    f"Paint an ethereal figure with atmospheric depth",
    f"Illustrate a dark fantasy creature with dramatic lighting",
]

for i, (card, prompt) in enumerate(zip(artist_cards[:4], good_prompts)):
    image_url = _load_image_as_data_url(card["image_path"])
    VERIFIER_EXAMPLES.append(
        {
            "task_id": f"good_{i}",
            "trace": _make_trace(prompt, image_url),
            "score": 0.90 + (i * 0.02),  # 0.90-0.96
        }
    )

# Bad: prompts that mention artist name (MUST score 0)
bad_prompts_with_name = [
    f"Generate an image in the style of {ARTIST_NAME}",
    f"Paint like {ARTIST_NAME} would",
    f"Create art similar to {name_parts[-1]}'s work",
]

for i, prompt in enumerate(bad_prompts_with_name):
    # Use a random card image - the key is the prompt contains forbidden name
    card = artist_cards[i % len(artist_cards)]
    image_url = _load_image_as_data_url(card["image_path"])
    VERIFIER_EXAMPLES.append(
        {
            "task_id": f"bad_name_{i}",
            "trace": _make_trace(prompt, image_url),
            "score": 0.0,  # MUST be 0 - artist name in prompt
        }
    )

# Bad: generic/vague prompts (low quality)
bad_prompts_generic = [
    "Generate a fantasy image",
    "Make some art",
]

for i, prompt in enumerate(bad_prompts_generic):
    card = artist_cards[(i + 4) % len(artist_cards)]
    image_url = _load_image_as_data_url(card["image_path"])
    VERIFIER_EXAMPLES.append(
        {
            "task_id": f"bad_generic_{i}",
            "trace": _make_trace(prompt, image_url),
            "score": 0.15 + (i * 0.05),  # Low but not 0
        }
    )

VERIFIER_TRAIN_SEEDS = list(range(6))
VERIFIER_VAL_SEEDS = list(range(6, len(VERIFIER_EXAMPLES)))

# Gold examples for reference
GOLD_EXAMPLES = [
    {
        "summary": f"Descriptive style prompt matching {ARTIST_NAME}'s aesthetic",
        "gold_score": 0.95,
        "gold_reasoning": f"Captures {STYLE_DESCRIPTION} without naming the artist",
        "trace": VERIFIER_EXAMPLES[0]["trace"],
    },
    {
        "summary": "Prompt with artist name - MUST score 0",
        "gold_score": 0.0,
        "gold_reasoning": "Contains forbidden artist name - automatic 0",
        "trace": VERIFIER_EXAMPLES[4]["trace"],
    },
]

verifier_dataset = {
    "tasks": [
        {
            "task_id": ex["task_id"],
            "input": {"trace": ex["trace"], "gold_examples": GOLD_EXAMPLES},
        }
        for ex in VERIFIER_EXAMPLES
    ],
    "gold_outputs": [
        {"task_id": ex["task_id"], "output": {}, "score": ex["score"]} for ex in VERIFIER_EXAMPLES
    ],
    "metadata": {
        "name": f"mtg_artist_verifier_{ARTIST_KEY}",
        "task_description": (
            f"Score style-matching quality for {ARTIST_NAME}'s distinctive art style. "
            f"CRITICAL: Return score 0 if the prompt contains the artist's name or variations: {FORBIDDEN_PATTERNS}. "
            f"For valid prompts, evaluate how well the style matches: {STYLE_DESCRIPTION}"
        ),
        "input_schema": {
            "type": "object",
            "properties": {"trace": {"type": "object"}, "gold_examples": {"type": "array"}},
            "required": ["trace", "gold_examples"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "event_reviews": {"type": "array"},
                "outcome_review": {"type": "object"},
                "event_totals": {"type": "array", "items": {"type": "number"}},
                "score": {"type": "number"},
                "artist_name_detected": {"type": "boolean"},
            },
            "required": ["event_reviews", "outcome_review", "event_totals"],
        },
        "domain": "image",
        "artist_name": ARTIST_NAME,
        "forbidden_patterns": FORBIDDEN_PATTERNS,
    },
}

verifier_config = GraphOptimizationConfig(
    algorithm="graph_evolve",
    dataset_name=f"mtg_artist_verifier_{ARTIST_KEY}",
    graph_type="verifier",
    graph_structure="single_prompt",
    topology_guidance=(
        "Single-node VerifierGraph. The verifier MUST:\n"
        "1. FIRST check if the prompt contains any forbidden artist name patterns\n"
        f"2. If prompt contains any of {FORBIDDEN_PATTERNS}, return score 0 immediately\n"
        "3. Only if no forbidden patterns found, evaluate style match\n"
        f"4. Score based on how well output matches: {STYLE_DESCRIPTION}\n"
        "Set output_mapping to copy score, artist_name_detected to root."
    ),
    allowed_policy_models=["gpt-4.1-nano", "gpt-4o-mini"],
    evolution=EvolutionConfig(num_generations=3, children_per_generation=2),
    proposer=ProposerConfig(model="gpt-4.1", temperature=0.0),
    seeds=SeedsConfig(train=VERIFIER_TRAIN_SEEDS, validation=VERIFIER_VAL_SEEDS),
    limits=LimitsConfig(max_spend_usd=5.0, timeout_seconds=3600),
    verifier_mode="contrastive",
    verifier_model=VERIFIER_MODEL,
    dataset=verifier_dataset,
    output_schema=verifier_dataset["metadata"]["output_schema"],
    task_description=verifier_dataset["metadata"]["task_description"],
    problem_spec=(
        f"You are generating a VerifierGraph for {ARTIST_NAME}'s art style.\n\n"
        f"CRITICAL RULE: If the user prompt contains ANY of these patterns (case-insensitive), "
        f"return score 0 IMMEDIATELY: {FORBIDDEN_PATTERNS}\n\n"
        f"For valid prompts (no artist name), score based on:\n"
        f"- Style match to: {STYLE_DESCRIPTION}\n"
        f"- Artistic quality and coherence\n"
        f"- Subject accuracy\n\n"
        f"Output JSON with: event_reviews, outcome_review, event_totals, score, artist_name_detected"
    ),
)


def _get_org_id() -> str:
    headers = {"Authorization": f"Bearer {API_KEY}"}
    urls = [f"{SYNTH_API_BASE}/api/v1/me", f"{SYNTH_API_BASE}/me"]
    for url in urls:
        resp = httpx.get(url, headers=headers, timeout=30)
        if resp.status_code == 404:
            continue
        resp.raise_for_status()
        data = resp.json()
        org_id = data.get("org_id") or data.get("orgId")
        if org_id:
            return str(org_id)
    raise RuntimeError("Unable to resolve org_id from /api/v1/me or /me")


async def run_verifier_optimization() -> tuple[str, dict[str, Any]]:
    async with GraphOptimizationClient(SYNTH_API_BASE, API_KEY) as client:
        job_id = await client.start_job(verifier_config)
        print(f"Graph evolve job: {job_id}")

    async with httpx.AsyncClient(timeout=90.0) as http:
        headers = {"Authorization": f"Bearer {API_KEY}"}
        for _ in range(900):
            try:
                status_resp = await http.get(
                    f"{SYNTH_API_BASE}/graph-evolve/jobs/{job_id}/status", headers=headers
                )
                if status_resp.status_code == 404:
                    await asyncio.sleep(2.0)
                    continue
                status_resp.raise_for_status()
                status = status_resp.json().get("status")
                if status in {"completed", "failed", "cancelled"}:
                    result_resp = await http.get(
                        f"{SYNTH_API_BASE}/graph-evolve/jobs/{job_id}/result", headers=headers
                    )
                    result_resp.raise_for_status()
                    return job_id, result_resp.json()
            except httpx.HTTPStatusError:
                await asyncio.sleep(2.0)
                continue
            await asyncio.sleep(2.0)

    raise RuntimeError(f"Graph evolve job {job_id} did not complete in time")


async def save_verifier_graph(job_id: str, org_id: str) -> str:
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "name": f"mtg-{ARTIST_KEY}-verifier-{job_id[:8]}",
        "org_id": org_id,
        "kind": "verifier",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{SYNTH_API_BASE}/graph-evolve/jobs/{job_id}/save-graph",
            headers=headers,
            json=payload,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"save-graph failed ({resp.status_code}): {resp.text[:500]}")
        data = resp.json()
    graph_id = data.get("graph_id") or data.get("id")
    if not graph_id:
        raise RuntimeError(f"save-graph did not return graph_id: {data}")
    return str(graph_id)


async def main() -> None:
    org_id = _get_org_id()
    print(f"Resolved org_id: {org_id}")

    job_id, result = await run_verifier_optimization()
    status = result.get("status")
    if status != "completed":
        error_msg = result.get("error") or "unknown error"
        raise RuntimeError(f"Graph evolve job failed with status={status}: {error_msg}")

    graph_id = await save_verifier_graph(job_id, org_id)
    best_score = result.get("best_score")

    artifacts_dir = demo_dir / "artifacts" / ARTIST_KEY
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.out) if args.out else artifacts_dir / "verifier_opt.json"

    payload = {
        "artist_key": ARTIST_KEY,
        "artist_name": ARTIST_NAME,
        "forbidden_patterns": FORBIDDEN_PATTERNS,
        "graph_evolve_job_id": job_id,
        "graph_id": graph_id,
        "best_score": best_score,
        "verifier_model": VERIFIER_MODEL,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("Verifier optimization complete")
    print("=" * 60)
    print(f"Artist: {ARTIST_NAME}")
    print(f"Optimized verifier job id: {job_id}")
    print(f"Optimized verifier graph id: {graph_id}")
    print(f"Best score: {best_score}")
    print(f"Saved artifact: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
