#!/usr/bin/env python3
"""Validate that a JSONL file parses and contains chat-like records.

Checks first N lines (default 50) for objects with `messages` including an
assistant response (role == "assistant").
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path", help="Path to JSONL file")
    p.add_argument("--n", type=int, default=50, help="Number of lines to sample")
    args = p.parse_args()

    src = Path(args.path)
    if not src.exists():
        raise SystemExit(f"No such file: {src}")

    checked = 0
    ok = 0
    with src.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if i > args.n:
                break
            line = line.strip()
            if not line:
                continue
            checked += 1
            try:
                obj = json.loads(line)
            except Exception as exc:
                raise SystemExit(f"Line {i} is not valid JSON: {exc}")
            msgs = obj.get("messages") if isinstance(obj, dict) else None
            if not isinstance(msgs, list):
                raise SystemExit(f"Line {i} missing 'messages' list")
            has_assistant = any(
                isinstance(m, dict) and m.get("role") == "assistant" and m.get("content")
                for m in msgs
            )
            if has_assistant:
                ok += 1

    if checked == 0:
        raise SystemExit("No lines checked; file empty?")
    if ok == 0:
        raise SystemExit("No assistant messages found in sampled lines")
    print(f"Validated: {ok}/{checked} sampled lines contain assistant messages")


if __name__ == "__main__":
    main()


