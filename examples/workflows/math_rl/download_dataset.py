#!/usr/bin/env python3
"""Download subsets of the MATH dataset to local JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


def extract_examples(dataset: Any, *, limit: int | None) -> list[dict[str, str]]:
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    examples: list[dict[str, str]] = []
    for item in dataset:
        problem = (item.get("problem") or "").strip()
        solution = item.get("solution") or ""
        if isinstance(solution, list):
            solution = "\n".join(str(part) for part in solution)
        examples.append(
            {
                "problem": problem,
                "solution": solution,
            }
        )
    return examples


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download MATH dataset splits to JSONL for offline use"
    )
    parser.add_argument(
        "--output-dir", default="examples/rl/data", help="Directory to write <split>.jsonl files"
    )
    parser.add_argument(
        "--dataset",
        default="nlile/hendrycks-MATH-benchmark",
        help="Hugging Face dataset identifier",
    )
    parser.add_argument(
        "--config", default="algebra", help="Hugging Face dataset config (if required)"
    )
    parser.add_argument(
        "--splits", nargs="*", default=["train", "validation", "test"], help="Splits to download"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Optional cap on examples per split"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        print(f"[INFO] Downloading {args.dataset} ({args.config}) split={split}")
        if args.config:
            dataset = load_dataset(args.dataset, args.config, split=split)
        else:
            dataset = load_dataset(args.dataset, split=split)
        rows = extract_examples(dataset, limit=args.limit)
        out_path = output_dir / f"{split}.jsonl"
        write_jsonl(out_path, rows)
        print(f"[INFO] Wrote {len(rows)} examples to {out_path}")

    print("Done. Set MATH_DATASET_LOCAL_DIR to the output directory when serving the task app.")


if __name__ == "__main__":
    main()
