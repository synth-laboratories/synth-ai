#!/usr/bin/env python3
"""Run verifier optimization (Graph Evolve) for PTCG gameplay traces.

Usage:
    uv run python demos/gepa_ptcg/run_verifier_opt.py --local
"""

import argparse
import asyncio
import json
import os
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

parser = argparse.ArgumentParser(description="Run PTCG verifier optimization")
parser.add_argument("--local", action="store_true", help="Use localhost:8000 backend")
parser.add_argument(
    "--dataset",
    type=str,
    default="demos/gepa_ptcg/artifacts/ptcg_verifier_dataset.json",
    help="Path to verifier dataset JSON",
)
parser.add_argument(
    "--out",
    type=str,
    default="demos/gepa_ptcg/artifacts/verifier_opt.json",
    help="Path to write verifier optimization artifact JSON",
)
parser.add_argument("--generations", type=int, default=3, help="Number of evolution generations")
parser.add_argument(
    "--children-per-generation",
    type=int,
    default=2,
    help="Children per generation",
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


def _get_org_id() -> str:
    headers = {"Authorization": f"Bearer {API_KEY}"}
    for path in ("/api/v1/me", "/me"):
        resp = httpx.get(f"{SYNTH_API_BASE}{path}", headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            org_id = data.get("org_id") or data.get("orgId")
            if org_id:
                return str(org_id)
    raise RuntimeError("Unable to resolve org_id from /api/v1/me or /me")


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

dataset_path = Path(args.dataset)
if not dataset_path.exists():
    raise FileNotFoundError(f"Dataset not found: {dataset_path}")

verifier_dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
num_tasks = len(verifier_dataset.get("tasks", []))
if num_tasks < 2:
    raise RuntimeError("Verifier dataset must contain at least 2 tasks")

split = max(1, int(num_tasks * 0.8))
train_seeds = list(range(split))
val_seeds = list(range(split, num_tasks)) or [split - 1]

VERIFIER_MODEL = "gpt-4.1-nano"

verifier_config = GraphOptimizationConfig(
    algorithm="graph_evolve",
    dataset_name="ptcg_gameplay_verifier",
    graph_type="verifier",
    graph_structure="dag",
    topology_guidance=(
        "Two-node VerifierGraph: judge_gameplay -> parse_output. "
        "judge_gameplay scores gameplay quality against rubric guidance; "
        "parse_output normalizes to strict JSON with event_reviews, outcome_review, event_totals, score."
    ),
    allowed_policy_models=["gpt-4.1-nano", "gpt-4o-mini"],
    evolution=EvolutionConfig(
        num_generations=args.generations, children_per_generation=args.children_per_generation
    ),
    proposer=ProposerConfig(model="gpt-4.1", temperature=0.0),
    seeds=SeedsConfig(train=train_seeds, validation=val_seeds),
    limits=LimitsConfig(max_spend_usd=10.0, timeout_seconds=3600),
    verifier_mode="contrastive",
    verifier_model=VERIFIER_MODEL,
    dataset=verifier_dataset,
    output_schema=verifier_dataset["metadata"]["output_schema"],
    output_config=verifier_dataset["metadata"]["output_config"],
    task_description=verifier_dataset["metadata"]["task_description"],
    problem_spec=(
        "You are generating a VerifierGraph for Pokemon TCG gameplay traces. "
        "Use two nodes: judge_gameplay then parse_output. parse_output MUST return ONLY JSON (no prose) "
        "with keys: event_reviews (list), outcome_review (object with criteria+total), event_totals (list), "
        "score (number). Penalize illegal actions, stalling, and missed attacks; reward clean tempo, "
        "board development, and prize-taking intent."
    ),
)


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
        "name": f"ptcg-verifier-{job_id[:8]}",
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

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "graph_evolve_job_id": job_id,
        "graph_id": graph_id,
        "best_score": best_score,
        "verifier_model": VERIFIER_MODEL,
        "dataset_path": str(dataset_path),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Verifier optimization complete")
    print(f"Optimized verifier job id: {job_id}")
    print(f"Optimized verifier graph id: {graph_id}")
    print(f"Wrote artifacts to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
