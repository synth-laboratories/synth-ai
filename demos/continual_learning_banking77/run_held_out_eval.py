#!/usr/bin/env python3
"""
Held-Out Evaluation Script for Banking77 Continual Learning

For each split (1-4), evaluates 4 prompts on the held-out test set:
1. Baseline — the default system prompt (no optimization)
2. MIPRO Continual — best prompt from the MIPRO continual run
3. GEPA Cold — best prompt from GEPA trained cold (baseline init)
4. GEPA Warm — best prompt from GEPA trained warm (prev split's best)

Usage:
    # Full run (train + evaluate)
    uv run python run_held_out_eval.py --rollouts-per-split 100

    # Evaluate only with existing results
    uv run python run_held_out_eval.py --eval-only \
        --mipro-results results/mipro_continual_*.json \
        --gepa-results results/classic_gepa_*.json

    # Skip training for one method
    uv run python run_held_out_eval.py --skip-mipro --mipro-results results/mipro_continual_*.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from data_splits import (
    Banking77SplitDataset,
    format_available_intents,
    get_split_intents,
    get_split_size,
)

# Tool schema (same as other scripts)
TOOL_NAME = "banking77_classify"
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Return the predicted banking77 intent label.",
        "parameters": {
            "type": "object",
            "properties": {"intent": {"type": "string"}},
            "required": ["intent"],
        },
    },
}

BASELINE_PROMPT = (
    "You are an expert banking assistant that classifies customer queries into banking intents. "
    "Given a customer message, respond with exactly one intent label from the provided list "
    "using the `banking77_classify` tool."
)


def resolve_backend_url() -> str:
    """Resolve the backend URL."""
    for env_var in ("SYNTH_URL", "SYNTH_BACKEND_URL", "RUST_BACKEND_URL"):
        env_url = (os.environ.get(env_var) or "").strip()
        if env_url:
            return env_url.rstrip("/")
    return "https://api.usesynth.ai"


async def evaluate_on_held_out(
    *,
    prompt: str,
    intent_split: int,
    model: str,
    api_key: str,
    backend_url: str,
    max_concurrent: int = 10,
) -> dict:
    """Evaluate prompt on ALL test samples for a split.

    Returns:
        {
            "accuracy": float,
            "correct": int,
            "total": int,
            "per_sample": [{"query": str, "expected": str, "predicted": str, "correct": bool}, ...]
        }
    """
    dataset = Banking77SplitDataset()
    dataset.ensure_ready(["test"])

    split_labels = dataset.get_split_labels(intent_split)
    available_intents = format_available_intents(split_labels)

    test_size = dataset.size("test", intent_split)
    indices = list(range(test_size))

    semaphore = asyncio.Semaphore(max_concurrent)
    per_sample: List[dict] = [None] * test_size  # type: ignore[list-item]
    correct_count = 0
    total_count = 0
    lock = asyncio.Lock()

    async def eval_one(idx: int, client: httpx.AsyncClient):
        nonlocal correct_count, total_count

        sample = dataset.sample(data_split="test", intent_split=intent_split, index=idx)

        user_msg = (
            f"Customer Query: {sample['text']}\n\n"
            f"Available Intents:\n{available_intents}\n\n"
            f"Classify this query into one of the above banking intents using the tool call."
        )

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_msg},
            ],
            "tools": [TOOL_SCHEMA],
            "tool_choice": {"type": "function", "function": {"name": TOOL_NAME}},
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        predicted_intent = ""
        got_response = False

        async with semaphore:
            try:
                response = await client.post(
                    f"{backend_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                )
                got_response = True

                if response.status_code == 200:
                    data = response.json()
                    choices = data.get("choices", [])
                    if choices:
                        tool_calls = choices[0].get("message", {}).get("tool_calls", [])
                        if tool_calls:
                            args_raw = tool_calls[0].get("function", {}).get("arguments")
                            if args_raw:
                                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                                predicted_intent = args.get("intent", "")
                else:
                    print(f"  Warning: status {response.status_code} for sample {idx}")
            except Exception as e:
                print(f"  Error on sample {idx}: {e}")

        expected_intent = sample["label"]
        is_correct = (
            predicted_intent.lower().replace("_", " ").strip()
            == expected_intent.lower().replace("_", " ").strip()
        ) if predicted_intent else False

        result = {
            "query": sample["text"],
            "expected": expected_intent,
            "predicted": predicted_intent,
            "correct": is_correct,
        }

        async with lock:
            per_sample[idx] = result
            if is_correct:
                correct_count += 1
            if got_response:
                total_count += 1

    async with httpx.AsyncClient(timeout=120.0) as client:
        tasks = [eval_one(idx, client) for idx in indices]
        # Process in batches to show progress
        batch_size = max_concurrent * 5
        for batch_start in range(0, len(tasks), batch_size):
            batch = tasks[batch_start : batch_start + batch_size]
            await asyncio.gather(*batch)
            done = min(batch_start + batch_size, len(tasks))
            print(f"    Progress: {done}/{len(tasks)} samples evaluated")

    # Filter out any None entries (shouldn't happen but defensive)
    per_sample = [s for s in per_sample if s is not None]
    total_count = len(per_sample)
    correct_count = sum(1 for s in per_sample if s["correct"])

    accuracy = correct_count / total_count if total_count > 0 else 0.0
    print(f"    Result: {correct_count}/{total_count} = {accuracy:.1%}")

    return {
        "accuracy": accuracy,
        "correct": correct_count,
        "total": total_count,
        "per_sample": per_sample,
    }


def extract_prompts_from_mipro(mipro_results: dict) -> Dict[int, Optional[str]]:
    """Extract best prompt per split from MIPRO results."""
    prompts: Dict[int, Optional[str]] = {}
    checkpoints = mipro_results.get("checkpoints", [])
    for checkpoint in checkpoints:
        split_num = checkpoint.get("split")
        best_text = checkpoint.get("best_candidate_text")
        if split_num is not None:
            prompts[split_num] = best_text
    return prompts


def extract_prompts_from_gepa(gepa_results: dict) -> Dict[int, Dict[str, Optional[str]]]:
    """Extract cold/warm best prompts per split from GEPA results.

    Returns: {split_num: {"cold": prompt_or_none, "warm": prompt_or_none}}
    """
    prompts: Dict[int, Dict[str, Optional[str]]] = {}
    splits = gepa_results.get("splits", {})
    for split_key, split_data in splits.items():
        split_num = int(split_key)
        cold_prompt = None
        warm_prompt = None

        cold = split_data.get("cold_start")
        if cold and cold.get("succeeded"):
            cold_prompt = cold.get("best_prompt")

        warm = split_data.get("warm_start")
        if warm and warm.get("succeeded"):
            warm_prompt = warm.get("best_prompt")

        prompts[split_num] = {"cold": cold_prompt, "warm": warm_prompt}
    return prompts


def run_subprocess(cmd: List[str], label: str) -> str:
    """Run a subprocess and return stdout. Raises on failure."""
    print(f"\n{'='*60}")
    print(f"Running {label}...")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(
        cmd,
        capture_output=False,
        text=True,
        cwd=Path(__file__).parent,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with return code {result.returncode}")
    return ""


def find_latest_result(pattern: str, results_dir: Path) -> Optional[Path]:
    """Find the most recently modified file matching a glob pattern."""
    matches = sorted(results_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


async def main():
    parser = argparse.ArgumentParser(
        description="Held-Out Evaluation for Banking77 Continual Learning"
    )
    parser.add_argument("--rollouts-per-split", type=int, default=100, help="Rollouts for MIPRO/GEPA training")
    parser.add_argument("--model", default="gpt-4.1-nano", help="Model for training")
    parser.add_argument("--eval-model", default=None, help="Model for held-out eval (defaults to --model)")
    parser.add_argument("--train-size", type=int, default=30, help="Training seeds per split")
    parser.add_argument("--output-dir", default="results/", help="Output directory")
    parser.add_argument("--skip-mipro", action="store_true", help="Use existing MIPRO results")
    parser.add_argument("--skip-gepa", action="store_true", help="Use existing GEPA results")
    parser.add_argument("--mipro-results", type=str, default=None, help="Path to existing MIPRO results JSON")
    parser.add_argument("--gepa-results", type=str, default=None, help="Path to existing GEPA results JSON")
    parser.add_argument("--eval-only", action="store_true", help="Skip training, just evaluate")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Max concurrent eval requests")
    args = parser.parse_args()

    eval_model = args.eval_model or args.model
    backend_url = resolve_backend_url().rstrip("/")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Get API key
    api_key = os.environ.get("SYNTH_API_KEY", "").strip()
    if not api_key:
        # Add synth-ai to path for mint_demo_api_key
        ROOT = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(ROOT))
        from synth_ai.core.utils.env import mint_demo_api_key
        print("Minting demo API key...")
        api_key = mint_demo_api_key(backend_url=backend_url)

    print(f"Backend URL: {backend_url}")
    print(f"Model (train): {args.model}")
    print(f"Model (eval):  {eval_model}")
    print(f"API Key: {api_key[:20]}...")

    # ── Phase 1: Train prompts (or load existing) ──

    if args.eval_only:
        args.skip_mipro = True
        args.skip_gepa = True

    # MIPRO
    mipro_results_path: Optional[Path] = None
    if not args.skip_mipro:
        run_subprocess(
            [
                sys.executable, "run_mipro_continual.py",
                "--rollouts-per-split", str(args.rollouts_per_split),
                "--model", args.model,
                "--train-size", str(args.train_size),
            ],
            "MIPRO Continual Training",
        )
        mipro_results_path = find_latest_result("mipro_continual_*.json", results_dir)
    elif args.mipro_results:
        mipro_results_path = Path(args.mipro_results)
    else:
        mipro_results_path = find_latest_result("mipro_continual_*.json", results_dir)

    if mipro_results_path and mipro_results_path.exists():
        print(f"Loading MIPRO results from: {mipro_results_path}")
        with open(mipro_results_path) as f:
            mipro_results = json.load(f)
    else:
        print("WARNING: No MIPRO results found. MIPRO prompts will be None.")
        mipro_results = {}

    # GEPA
    gepa_results_path: Optional[Path] = None
    if not args.skip_gepa:
        run_subprocess(
            [
                sys.executable, "run_classic_gepa.py",
                "--rollouts", str(args.rollouts_per_split),
                "--model", args.model,
                "--train-size", str(args.train_size),
            ],
            "Classic GEPA Training",
        )
        gepa_results_path = find_latest_result("classic_gepa_*.json", results_dir)
    elif args.gepa_results:
        gepa_results_path = Path(args.gepa_results)
    else:
        gepa_results_path = find_latest_result("classic_gepa_*.json", results_dir)

    if gepa_results_path and gepa_results_path.exists():
        print(f"Loading GEPA results from: {gepa_results_path}")
        with open(gepa_results_path) as f:
            gepa_results = json.load(f)
    else:
        print("WARNING: No GEPA results found. GEPA prompts will be None.")
        gepa_results = {}

    # ── Phase 2: Extract prompts per split ──

    mipro_prompts = extract_prompts_from_mipro(mipro_results)
    gepa_prompts = extract_prompts_from_gepa(gepa_results)

    prompts_per_split: Dict[str, Dict[str, Optional[str]]] = {}
    for split_num in [1, 2, 3, 4]:
        prompts_per_split[str(split_num)] = {
            "baseline": BASELINE_PROMPT,
            "mipro": mipro_prompts.get(split_num),
            "gepa_cold": gepa_prompts.get(split_num, {}).get("cold"),
            "gepa_warm": gepa_prompts.get(split_num, {}).get("warm") if split_num > 1 else None,
        }

    print("\n" + "=" * 60)
    print("Prompts extracted per split:")
    for split_num in [1, 2, 3, 4]:
        sp = prompts_per_split[str(split_num)]
        print(f"  Split {split_num}:")
        for method, prompt in sp.items():
            status = f"{len(prompt)} chars" if prompt else "N/A"
            print(f"    {method}: {status}")

    # ── Phase 3: Evaluate each prompt on held-out test set ──

    print("\n" + "=" * 60)
    print("HELD-OUT TEST EVALUATION")
    print("=" * 60)

    eval_results: Dict[str, Any] = {}
    dataset = Banking77SplitDataset()
    dataset.ensure_ready(["test"])

    for split_num in [1, 2, 3, 4]:
        test_size = dataset.size("test", split_num)
        num_intents = get_split_size(split_num)
        print(f"\n--- Split {split_num} ({num_intents} intents, {test_size} test samples) ---")

        split_eval: Dict[str, Any] = {
            "num_intents": num_intents,
            "test_size": test_size,
        }

        sp = prompts_per_split[str(split_num)]
        for method in ["baseline", "mipro", "gepa_cold", "gepa_warm"]:
            prompt = sp[method]
            if prompt is None:
                print(f"  {method}: skipped (no prompt)")
                split_eval[method] = None
                continue

            print(f"  Evaluating {method}...")
            result = await evaluate_on_held_out(
                prompt=prompt,
                intent_split=split_num,
                model=eval_model,
                api_key=api_key,
                backend_url=backend_url,
                max_concurrent=args.max_concurrent,
            )
            split_eval[method] = result

        eval_results[str(split_num)] = split_eval

    # ── Phase 4: Save results ──

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    date_str = time.strftime("%Y-%m-%d")

    full_output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": eval_model,
        "results": eval_results,
        "prompts": prompts_per_split,
    }

    json_path = output_dir / f"held_out_eval_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(full_output, f, indent=2)
    print(f"\nJSON results saved to: {json_path}")

    # Build summary table
    header = f"{'Split':<7}{'Intents':<9}{'Test Size':<11}{'Baseline':<10}{'MIPRO':<9}{'GEPA Cold':<11}{'GEPA Warm':<10}"
    divider = f"{'-----':<7}{'-------':<9}{'---------':<11}{'--------':<10}{'-----':<9}{'---------':<11}{'---------':<10}"

    table_lines = []
    for split_num in [1, 2, 3, 4]:
        sr = eval_results[str(split_num)]

        def fmt(method: str) -> str:
            r = sr.get(method)
            if r is None:
                return "-"
            return f"{r['accuracy']:.1%}"

        line = (
            f"{split_num:<7}"
            f"{sr['num_intents']:<9}"
            f"{sr['test_size']:<11}"
            f"{fmt('baseline'):<10}"
            f"{fmt('mipro'):<9}"
            f"{fmt('gepa_cold'):<11}"
            f"{fmt('gepa_warm'):<10}"
        )
        table_lines.append(line)

    txt_content = (
        f"Banking77 Held-Out Test Evaluation\n"
        f"==================================\n"
        f"Model: {eval_model}\n"
        f"Date: {date_str}\n\n"
        f"{header}\n"
        f"{divider}\n"
        + "\n".join(table_lines)
        + "\n\nPer-split raw scores saved in JSON.\n"
    )

    txt_path = output_dir / f"held_out_eval_{timestamp}.txt"
    with open(txt_path, "w") as f:
        f.write(txt_content)
    print(f"TXT summary saved to: {txt_path}")

    # Print summary
    print(f"\n{txt_content}")

    # Update README
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        readme_content = readme_path.read_text()

        readme_table_header = "| Split | Intents | Test Size | Baseline | MIPRO | GEPA Cold | GEPA Warm |"
        readme_table_divider = "|-------|---------|-----------|----------|-------|-----------|-----------|"
        readme_table_rows = []
        for split_num in [1, 2, 3, 4]:
            sr = eval_results[str(split_num)]

            def fmt_md(method: str) -> str:
                r = sr.get(method)
                if r is None:
                    return "-"
                return f"{r['accuracy']:.1%}"

            row = f"| Split {split_num} | {sr['num_intents']} | {sr['test_size']} | {fmt_md('baseline')} | {fmt_md('mipro')} | {fmt_md('gepa_cold')} | {fmt_md('gepa_warm')} |"
            readme_table_rows.append(row)

        held_out_section = (
            f"\n## Held-Out Test Evaluation\n\n"
            f"Full held-out test set evaluation (all test samples per split, not just 50).\n\n"
            f"**Model:** {eval_model}  \n"
            f"**Date:** {date_str}\n\n"
            f"{readme_table_header}\n"
            f"{readme_table_divider}\n"
            + "\n".join(readme_table_rows)
            + "\n\n"
            f"Per-split raw scores saved in `{json_path.name}`.\n"
        )

        # Replace existing section or append
        section_marker = "## Held-Out Test Evaluation"
        if section_marker in readme_content:
            # Find the section and replace it (up to next ## or end of file)
            start = readme_content.index(section_marker)
            # Find next section header
            rest = readme_content[start + len(section_marker):]
            next_section = rest.find("\n## ")
            if next_section != -1:
                end = start + len(section_marker) + next_section
                readme_content = readme_content[:start] + held_out_section.lstrip("\n") + readme_content[end:]
            else:
                readme_content = readme_content[:start] + held_out_section.lstrip("\n")
        else:
            readme_content = readme_content.rstrip() + "\n" + held_out_section

        readme_path.write_text(readme_content)
        print(f"README.md updated with held-out evaluation section.")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
