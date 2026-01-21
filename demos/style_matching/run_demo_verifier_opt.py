#!/usr/bin/env python3
"""Run the style-matching verifier optimization (Graph Evolve) only.

Usage:
    uv run python demos/style_matching/run_demo_verifier_opt.py --local
    uv run python demos/style_matching/run_demo_verifier_opt.py
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict

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

parser = argparse.ArgumentParser(description="Run style-matching verifier optimization")
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

synth_root = Path(__file__).resolve().parents[2]


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


def _make_trace(user_text: str, assistant_text: str) -> Dict[str, Any]:
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
                        "content": user_text,
                    },
                    {
                        "event_type": "runtime",
                        "event_id": 2,
                        "type": "assistant_message",
                        "content": assistant_text,
                    },
                ],
            }
        ],
    }


GOLD_EXAMPLES = [
    {
        "summary": "Direct, builder tone with concrete examples",
        "gold_score": 0.95,
        "gold_reasoning": "Short sentences, decisive stance, concrete advice, crisp closing line.",
        "trace": _make_trace(
            "Write about shipping fast.",
            "Speed is a learning engine. Ship a small bet, watch users react, then sharpen the next move. Protect maker time, keep scope tight, and treat every release as feedback, not theater. Momentum beats motivation.",
        ),
    },
    {
        "summary": "Opinionated essay with a sharp closing line",
        "gold_score": 0.92,
        "gold_reasoning": "Clear thesis, direct claims, practical advice, and a memorable end.",
        "trace": _make_trace(
            "Write about focus.",
            "Optionality feels safe, but it dilutes learning. Pick one wedge, cut parallel bets, and ship. Small teams win by saying no early and often. Clarity is leverage.",
        ),
    },
    {
        "summary": "Concrete, tactical quality guidance with guardrails",
        "gold_score": 0.96,
        "gold_reasoning": "Direct stance, concrete guardrails, and a crisp closing line.",
        "trace": _make_trace(
            "Write about quality as a constraint.",
            "Quality is the guardrail that keeps speed from turning into chaos. Ship small, test the riskiest paths, and make rollback cheap. Fix root causes once, then automate the prevention. Quality is a habit, not a milestone.",
        ),
    },
]

VERIFIER_EXAMPLES = [
    {
        "task_id": "good_speed",
        "trace": _make_trace(
            "Write about shipping fast.",
            "Speed compounds learning. Ship small bets, learn fast, keep scope tight, and protect maker time.",
        ),
        "score": 0.95,
    },
    {
        "task_id": "good_focus",
        "trace": _make_trace(
            "Write about focus.",
            "Optionality dilutes learning. Pick one wedge, cut parallel bets, and repeat a simple story.",
        ),
        "score": 0.92,
    },
    {
        "task_id": "good_quality",
        "trace": _make_trace(
            "Write about quality as a constraint.",
            "Quality is a guardrail. Ship small, test risky paths, and make rollback cheap.",
        ),
        "score": 0.93,
    },
    {
        "task_id": "good_learning",
        "trace": _make_trace(
            "Write about learning in public.",
            "Publish drafts to accelerate feedback, build credibility, and clarify thinking.",
        ),
        "score": 0.91,
    },
    {
        "task_id": "bad_rambling",
        "trace": _make_trace(
            "Write about focus.",
            "Focus is important because focus is important. You should focus on focusing and focus on focus.",
        ),
        "score": 0.10,
    },
    {
        "task_id": "bad_vague",
        "trace": _make_trace(
            "Write about quality.",
            "Quality is good. Teams should be good and do good things to make quality good.",
        ),
        "score": 0.05,
    },
]

VERIFIER_TRAIN_SEEDS = list(range(4))
VERIFIER_VAL_SEEDS = list(range(4, len(VERIFIER_EXAMPLES)))

verifier_dataset = {
    "tasks": [
        {
            "task_id": example["task_id"],
            "input": {"trace": example["trace"], "gold_examples": GOLD_EXAMPLES},
        }
        for example in VERIFIER_EXAMPLES
    ],
    "gold_outputs": [
        {"task_id": example["task_id"], "output": {}, "score": example["score"]}
        for example in VERIFIER_EXAMPLES
    ],
    "metadata": {
        "name": "style_matching_verifier",
        "task_description": (
            "Score style-matching quality using strict verifier-style outputs. Return event_reviews, "
            "outcome_review, event_totals; use a 1.0 baseline with deductions for discrepancies "
            "(generic outputs < 0.3)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "trace": {"type": "object"},
                "gold_examples": {"type": "array"},
            },
            "required": ["trace", "gold_examples"],
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "event_reviews": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "criteria": {"type": "object"},
                            "total": {"type": "number"},
                            "summary": {"type": "string"},
                        },
                        "required": ["criteria", "total"],
                    },
                },
                "outcome_review": {
                    "type": "object",
                    "properties": {
                        "criteria": {"type": "object"},
                        "total": {"type": "number"},
                        "summary": {"type": "string"},
                    },
                    "required": ["criteria", "total"],
                },
                "event_totals": {
                    "type": "array",
                    "items": {"type": "number"},
                },
                "score": {"type": "number"},
            },
            "required": ["event_reviews", "outcome_review", "event_totals"],
        },
        "output_config": {
            "format": "json",
            "strict": True,
            "extract_from": ["parse_output_output", "judge_style_output", "(root)"],
            "schema": {
                "type": "object",
                "properties": {
                    "event_reviews": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "criteria": {"type": "object"},
                                "total": {"type": "number"},
                                "summary": {"type": "string"},
                            },
                            "required": ["criteria", "total"],
                        },
                    },
                    "outcome_review": {
                        "type": "object",
                        "properties": {
                            "criteria": {"type": "object"},
                            "total": {"type": "number"},
                            "summary": {"type": "string"},
                        },
                        "required": ["criteria", "total"],
                    },
                    "event_totals": {
                        "type": "array",
                        "items": {"type": "number"},
                    },
                    "score": {"type": "number"},
                },
                "required": ["event_reviews", "outcome_review", "event_totals"],
            },
        },
        "domain": "text",
    },
}

verifier_config = GraphOptimizationConfig(
    algorithm="graph_evolve",
    dataset_name="style_matching_verifier",
    graph_type="verifier",
    graph_structure="dag",
    topology_guidance=(
        "Two-node VerifierGraph: judge_style -> parse_output. "
        "judge_style runs the evaluator, parse_output is a schema adapter that returns strict JSON. "
        "parse_output should read judge_style_output and emit: event_reviews (list), "
        "outcome_review (object), event_totals (list), score (number). "
        "Set output_mapping on parse_output to copy these fields to root. "
        "Include verdict_weights and aggregation_policy: weighted_average."
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
    output_config=verifier_dataset["metadata"]["output_config"],
    task_description=(
        "Score style-matching quality using strict verifier-style outputs. Return event_reviews, "
        "outcome_review, event_totals; use a 1.0 baseline with deductions for discrepancies "
        "(generic outputs < 0.3)."
    ),
    problem_spec=(
        "You are generating a VerifierGraph. Use two nodes: judge_style then parse_output. "
        "parse_output MUST return ONLY valid JSON (no prose, no markdown) with this schema:\n"
        "{\n"
        '  "event_reviews": [\n'
        '    {"criteria": {"tone": 0.0, "decisiveness": 0.0, "concreteness": 0.0}, "total": 0.0, "summary": ""}\n'
        "  ],\n"
        '  "outcome_review": {"criteria": {"tone": 0.0, "decisiveness": 0.0, "concreteness": 0.0}, "total": 0.0, "summary": ""},\n'
        '  "event_totals": [0.0],\n'
        '  "score": 0.0\n'
        "}\n"
        "criteria must be an object mapping strings to numbers, total must be a number, "
        "event_totals must be a list of numbers. If unsure, output empty lists/objects and 0.0 values. "
        "Scoring policy: start at 1.0 and deduct for discrepancies vs gold examples; "
        "generic outputs should score below 0.3."
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


async def run_verifier_optimization() -> tuple[str, Dict[str, Any]]:
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
        "name": f"style-matching-verifier-{job_id[:8]}",
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

    artifacts_dir = synth_root / "demos" / "style_matching" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.out) if args.out else artifacts_dir / "verifier_opt.json"

    payload = {
        "graph_evolve_job_id": job_id,
        "graph_id": graph_id,
        "best_score": best_score,
        "verifier_model": VERIFIER_MODEL,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Verifier optimization complete")
    print(f"Optimized verifier job id: {job_id}")
    print(f"Optimized verifier graph id: {graph_id}")
    print(f"Best score: {best_score}")
    print(f"Saved artifact: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
