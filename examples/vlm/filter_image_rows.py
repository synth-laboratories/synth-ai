#!/usr/bin/env python3
"""
Filter SFT JSONL rows to those that contain image content.

This is a convenience wrapper around `examples/warming_up_to_rl/export_trace_sft.py`
output now that each record's metadata includes `has_image`, `user_has_image`, and
`assistant_has_image`.

Usage:
    uv run python examples/vlm/filter_image_rows.py \
        --input examples/sft/ft_data/crafter_traces.jsonl \
        --output examples/vlm/output/crafter_vlm_dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source JSONL dataset")
    parser.add_argument("--output", type=Path, required=True, help="Filtered JSONL path")
    parser.add_argument(
        "--include-assistant",
        action="store_true",
        help="Require the assistant message to include an image as well",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = args.input
    dst = args.output
    dst.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    total = 0
    with src.open("r", encoding="utf-8") as reader, dst.open("w", encoding="utf-8") as writer:
        for line in reader:
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            metadata = record.get("metadata") or {}
            has_user_image = bool(metadata.get("user_has_image"))
            has_assistant_image = bool(metadata.get("assistant_has_image"))
            if not has_user_image:
                continue
            if args.include_assistant and not has_assistant_image:
                continue
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Filtered {kept} / {total} rows with user images -> {dst}")


if __name__ == "__main__":
    main()
