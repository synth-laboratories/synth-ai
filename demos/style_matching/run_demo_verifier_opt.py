#!/usr/bin/env python3
"""Run the style-matching verifier optimization (Graph Evolve) only.

Usage:
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
from synth_ai.core.urls import (
    BACKEND_URL_BASE,
    backend_health_url,
    backend_me_url,
    join_url,
)
from synth_ai.products.graph_evolve import GraphOptimizationClient, GraphOptimizationConfig
from synth_ai.products.graph_evolve.config import (
    EvolutionConfig,
    LimitsConfig,
    ProposerConfig,
    SeedsConfig,
)
from synth_ai.sdk.auth import get_or_mint_synth_api_key

parser = argparse.ArgumentParser(description="Run style-matching verifier optimization")
parser.add_argument(
    "--out",
    type=str,
    default=None,
    help="Path to write verifier optimization artifact JSON",
)
args = parser.parse_args()

SYNTH_API_BASE = BACKEND_URL_BASE
os.environ["BACKEND_BASE_URL"] = SYNTH_API_BASE


def _validate_api_key(api_key: str) -> bool:
    if not api_key:
        return False
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = httpx.get(backend_me_url(SYNTH_API_BASE), headers=headers, timeout=10)
    except Exception:
        return False
    return resp.status_code == 200


print(f"Backend: {SYNTH_API_BASE}")

r = httpx.get(backend_health_url(SYNTH_API_BASE), timeout=30)
if r.status_code != 200:
    raise RuntimeError(f"Backend not healthy: status {r.status_code}")
print(f"Backend health: {r.json()}")

API_KEY = get_or_mint_synth_api_key(backend_url=SYNTH_API_BASE, validator=_validate_api_key)
print(f"Using SYNTH_API_KEY: {API_KEY[:20]}...")

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
            "extract_from": ["(root)"],
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
    graph_structure="single_prompt",
    topology_guidance=(
        "Single-node VerifierGraph. Use one DagNode (e.g., judge_style) with template_transform. "
        "Set output_mapping to copy event_reviews, outcome_review, event_totals, score to root. "
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
        "You are generating a VerifierGraph. The final output MUST be a JSON object with: "
        "event_reviews (list of per-event review objects with criteria, total, summary), "
        "outcome_review (object with criteria, total, summary), and event_totals (list of numbers). "
        "Include a top-level score if helpful, but the verifier contract is based on outcome_review.total "
        "and event_totals. Make totals floats in [0,1]. Scoring policy must be strict: start at 1.0 and "
        "deduct for every discrepancy vs gold examples. Generic/standard outputs should score below 0.3. "
        "Deduction guide: obvious/giveaway discrepancy deduct 0.15-0.3, major discrepancy 0.08-0.15, "
        "minor 0.02-0.08."
    ),
)


def _get_org_id() -> str:
    headers = {"Authorization": f"Bearer {API_KEY}"}
    urls = [backend_me_url(SYNTH_API_BASE), join_url(SYNTH_API_BASE, "/me")]
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
                    join_url(SYNTH_API_BASE, f"/graph-evolve/jobs/{job_id}/status"),
                    headers=headers,
                )
                if status_resp.status_code == 404:
                    await asyncio.sleep(2.0)
                    continue
                status_resp.raise_for_status()
                status = status_resp.json().get("status")
                if status in {"completed", "failed", "cancelled"}:
                    result_resp = await http.get(
                        join_url(SYNTH_API_BASE, f"/graph-evolve/jobs/{job_id}/result"),
                        headers=headers,
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
            join_url(SYNTH_API_BASE, f"/graph-evolve/jobs/{job_id}/save-graph"),
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

    artifacts_dir = Path(__file__).parent / "artifacts"
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
