from __future__ import annotations

import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from examples.finetuning.synth_qwen_v1.util import validate_jsonl


def main() -> None:
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "training.jsonl"

    # Minimal single-example JSONL
    lines = [
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Write a short greeting."},
                    {"role": "assistant", "content": "Hello there!"},
                ]
            }
        )
    ]
    out_path.write_text("\n".join(lines) + "\n")
    validate_jsonl(out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
