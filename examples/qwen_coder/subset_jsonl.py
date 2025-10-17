#!/usr/bin/env python3
"""Create a capped subset of a JSONL dataset for quick runs."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src", help="Source JSONL path")
    p.add_argument("dst", help="Destination JSONL path")
    p.add_argument("--n", type=int, default=200, help="Max examples to keep")
    args = p.parse_args()

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"No such file: {src}")
    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            if kept >= args.n:
                break
            if not line.strip():
                continue
            fout.write(line)
            kept += 1
    print(f"Wrote {dst} with {kept} lines")


if __name__ == "__main__":
    main()


