"""
Example: Random Search optimizer on Banking77 using Groq gpt-oss-20b.

Requires:
- .env with GROQ_API_KEY
- datasets (`uv add datasets` if needed)

Run:
- uv run -q python -m synth_ai.learning.prompts.run_random_search_banking77
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from datasets import load_dataset
from dotenv import load_dotenv
from synth_ai.learning.prompts.random_search import random_search_compile
from synth_ai.lm.core.main_v3 import LM, build_messages
from tqdm import tqdm


def choose_label(pred: str, label_names: list[str]) -> str:
    norm = (pred or "").strip().lower()
    d = {ln.lower(): ln for ln in label_names}
    if norm in d:
        return d[norm]

    def score(cand: str) -> int:
        c = cand.lower()
        return sum(1 for w in c.split() if w in norm)

    return max(label_names, key=score)


def accuracy(pred: str, gold: str, labels: list[str]) -> float:
    return 1.0 if choose_label(pred, labels) == gold else 0.0


@dataclass
class StudentProgram:
    lm: LM
    label_names: list[str]
    instruction: str
    demos: list[tuple[str, str]]

    def reset_copy(self):
        return replace(self, instruction=self.instruction, demos=list(self.demos))

    def deepcopy(self):
        return replace(self, instruction=str(self.instruction), demos=list(self.demos))

    def with_demos(self, demos: list[tuple[str, str]]):
        return replace(self, demos=list(demos))

    def run(self, x: str) -> str:
        # Build a prompt with optional demos
        examples = "\n".join(f"Input: {a}\nLabel: {b}" for a, b in self.demos)
        sys = self.instruction or "You are an intent classifier for Banking77."
        user = (f"Examples:\n{examples}\n\n" if examples else "") + f"Message: {x}\nLabel:"
        messages = build_messages(sys, user, images_bytes=None, model_name=self.lm.model)

        # Call LM synchronously via asyncio
        async def _call():
            resp = await self.lm.respond_async(messages=messages)
            return (resp.raw_response or "").strip()

        return asyncio.run(_call())

    async def _apredict(self, x: str):
        examples = "\n".join(f"Input: {a}\nLabel: {b}" for a, b in self.demos)
        sys = self.instruction or "You are an intent classifier for Banking77."
        user = (f"Examples:\n{examples}\n\n" if examples else "") + f"Message: {x}\nLabel:"
        messages = build_messages(sys, user, images_bytes=None, model_name=self.lm.model)
        resp = await self.lm.respond_async(messages=messages)
        return (resp.raw_response or "").strip(), (resp.usage or {})


def main():
    load_dotenv()
    random.seed(0)

    model = os.getenv("MODEL", "openai/gpt-oss-20b")
    vendor = os.getenv("VENDOR", "groq")
    lm = LM(model=model, vendor=vendor, temperature=0.0)

    print("Loading Banking77 dataset (train/dev split of test for demo)...")
    ds = load_dataset("banking77")
    label_names: list[str] = ds["test"].features["label"].names  # type: ignore

    # Create small train/val from the test split for speed
    all_items = [(r["text"], label_names[int(r["label"])]) for r in ds["test"]]
    random.shuffle(all_items)
    trainset: Sequence[tuple[str, str]] = all_items[:40]
    valset: Sequence[tuple[str, str]] = all_items[40:60]  # 20 examples

    student = StudentProgram(
        lm=lm,
        label_names=label_names,
        instruction="You are an intent classifier for the Banking77 dataset. Return exactly one label.",
        demos=[],
    )

    def metric(yhat: str, y: str) -> float:
        return accuracy(yhat, y, label_names)

    total_candidates = 3 + 3  # zero-shot, labeled few-shot, bootstrapped + 3 random seeds
    print(
        f"Running Random Search optimizer ({total_candidates} candidates, parallel eval of 20 questions)..."
    )

    def eval_parallel(program: StudentProgram, dataset: Sequence[tuple[str, str]], metric_fn):
        async def _run():
            xs = [x for x, _ in dataset]
            ys = [y for _, y in dataset]
            preds: list[Optional[str]] = [None] * len(xs)
            sem = asyncio.Semaphore(int(os.getenv("CONCURRENCY", "5")))

            async def worker(i: int, x: str, y: str):
                import time

                t_start = time.monotonic()
                try:
                    async with sem:
                        pred, usage = await asyncio.wait_for(
                            program._apredict(x),
                            timeout=float(os.getenv("TIMEOUT_S", "45")),
                        )
                        t_end = time.monotonic()
                        return i, y, pred, t_start, t_end, usage or {}
                except asyncio.CancelledError:
                    # Respect cancellation but return a placeholder record so scheduler can proceed
                    t_end = time.monotonic()
                    return i, y, "", t_start, t_end, {}
                except Exception:
                    t_end = time.monotonic()
                    return i, y, "", t_start, t_end, {}

            tasks = [asyncio.create_task(worker(i, x, y)) for i, (x, y) in enumerate(zip(xs, ys, strict=False))]
            correct_sum = 0.0
            processed = 0
            import statistics
            import time

            durations: list[float] = []
            in_tok_sum = 0
            out_tok_sum = 0
            in_tok_count = 0
            out_tok_count = 0
            details: list[dict[str, Any]] = []
            t_batch_start = time.monotonic()
            deadline = float(os.getenv("BATCH_DEADLINE_S", "20"))
            with tqdm(total=len(tasks), desc="Rollouts", leave=False) as pbar:
                pending = set(tasks)
                # Process completions until all done or deadline reached
                while pending:
                    elapsed = time.monotonic() - t_batch_start
                    remaining = max(0.0, deadline - elapsed)
                    if remaining <= 0.0:
                        # Cancel any remaining
                        for t in pending:
                            t.cancel()
                        done, _ = await asyncio.wait(pending, return_when=asyncio.ALL_COMPLETED)
                        # Record canceled as zeros
                        for task in done:
                            try:
                                i, y_true, pred, t_start, t_end, usage = task.result()
                            except Exception:
                                # Unknown index: we can't recover; skip as it's canceled before start
                                continue
                            # Already processed ones shouldn't be in pending; skip
                        break
                    # Wait for at least one completion within remaining time (polling granularity <= 1s)
                    timeout = min(1.0, remaining)
                    done, pending = await asyncio.wait(
                        pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
                    )
                    import contextlib
                    for task in done:
                        try:
                            i, y_true, pred, t_start, t_end, usage = task.result()
                        except BaseException:
                            # Treat as failure/cancelled
                            continue
                        durations.append(max(0.0, t_end - t_start))
                        preds[i] = pred
                        processed += 1
                        with contextlib.suppress(Exception):
                            correct_sum += float(metric_fn(pred, y_true))
                        with contextlib.suppress(Exception):
                            pt = usage.get("prompt_tokens") or usage.get("input_tokens")
                            ct = usage.get("completion_tokens") or usage.get("output_tokens")
                            if isinstance(pt, (int, float)):
                                in_tok_sum += int(pt)
                                in_tok_count += 1
                            if isinstance(ct, (int, float)):
                                out_tok_sum += int(ct)
                                out_tok_count += 1
                        details.append(
                            {
                                "index": i,
                                "seconds": max(0.0, t_end - t_start),
                                "score": float(metric_fn(pred, y_true)),
                                "usage": {
                                    "prompt_tokens": usage.get("prompt_tokens")
                                    or usage.get("input_tokens"),
                                    "completion_tokens": usage.get("completion_tokens")
                                    or usage.get("output_tokens"),
                                },
                            }
                        )
                        pbar.update(1)
                        med = statistics.median(durations) if durations else 0.0
                        mx = max(durations) if durations else 0.0
                        avg_in = (in_tok_sum / in_tok_count) if in_tok_count else 0.0
                        avg_out = (out_tok_sum / out_tok_count) if out_tok_count else 0.0
                        pbar.set_postfix(
                            {
                                "acc": f"{(correct_sum / processed):.2f}",
                                "done": f"{processed}/{len(tasks)}",
                                "med_s": f"{med:.1f}",
                                "max_s": f"{mx:.1f}",
                                "tin": f"{avg_in:.1f}",
                                "tout": f"{avg_out:.1f}",
                            }
                        )
            # Compute score only from completed/successful rollouts (drop timeouts/cancelled)
            subs = [float(d.get("score", 0.0)) for d in details]
            result = SimpleNamespace(score=(sum(subs) / max(1, len(subs))), subscores=subs)
            result.details = details
            result.mean_in = (in_tok_sum / in_tok_count) if in_tok_count else 0.0
            result.mean_out = (out_tok_sum / out_tok_count) if out_tok_count else 0.0
            return result

        return asyncio.run(_run())

    pbar = tqdm(total=total_candidates, desc="Candidates")
    candidate_eval_details: dict[int, Any] = {}

    def on_cand(idx: int, score: float, res, intervention):
        pbar.update(1)
        pbar.set_postfix({"score": f"{score:.2f}"})
        # store per-instance details (for apples-to-apples)
        import contextlib
        with contextlib.suppress(Exception):
            candidate_eval_details[idx] = {
                "score": score,
                "mean_in": getattr(res, "mean_in", None),
                "mean_out": getattr(res, "mean_out", None),
                "instances": getattr(res, "details", None),
            }
        # visible summary line per candidate
        kind = (
            intervention.get("kind", "candidate") if isinstance(intervention, dict) else "candidate"
        )
        label = intervention.get("label") if isinstance(intervention, dict) else None
        seed = intervention.get("seed") if isinstance(intervention, dict) else None
        processed = len(getattr(res, "details", []) or [])
        from tqdm import tqdm as _tqdm

        _tqdm.write(
            f"Candidate {idx}/{total_candidates} [{kind}{'' if label is None else f', label={label}'}{'' if seed is None else f', seed={seed}'}]: "
            f"score={score:.2f} | mean tin/tout={getattr(res, 'mean_in', 0):.1f}/{getattr(res, 'mean_out', 0):.1f} | N={processed}"
        )

    best, records = random_search_compile(
        student=student,
        trainset=trainset,
        valset=valset,
        metric=metric,
        evaluate_fn=eval_parallel,
        max_bootstrapped_demos=0,
        max_labeled_demos=4,
        max_rounds=2,
        num_candidate_programs=3,
        on_candidate_evaluated=on_cand,
    )
    pbar.close()

    # Evaluate best on holdout (valset) with parallel rollouts
    print("Evaluating best program on val (parallel rollouts)...")
    best_res = eval_parallel(best, valset, metric)
    correct = int(round(best_res.score * max(1, len(best_res.subscores))))
    print(
        "Best program accuracy on val: "
        f"{correct}/{len(valset)} ({best_res.score:.2%}) "
        f"| mean tokens in/out: {getattr(best_res, 'mean_in', 0):.1f}/{getattr(best_res, 'mean_out', 0):.1f}"
    )

    # Save per-candidate scores and interventions
    out = {
        "context": {
            "model": model,
            "vendor": vendor,
            "train_size": len(trainset),
            "val_size": len(valset),
        },
        "candidates": records,
        "candidate_eval_details": candidate_eval_details,
        "best_eval_details": {
            "score": best_res.score,
            "mean_in": getattr(best_res, "mean_in", None),
            "mean_out": getattr(best_res, "mean_out", None),
            "instances": getattr(best_res, "details", None),
        },
    }
    out_dir = Path(__file__).parent
    fname = str(out_dir / f"random_search_banking77_{int(time.time())}.json")
    with open(fname, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved candidate records to {fname}")


if __name__ == "__main__":
    main()
