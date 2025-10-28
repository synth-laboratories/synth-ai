#!/usr/bin/env python3
"""
Generate a small, synthetic SFT JSONL dataset for coder-style prompts.

Each line uses the minimal schema consumed by the SFT CLI:
  {"messages": [{"role": "user", "content": "..."}], "response": "..."}

Example:
  uv run python examples/qwen_coder/generate_dataset.py \
    --output examples/qwen_coder/ft_data/coder_sft.small.jsonl \
    --n 50 --seed 42 --lang python
"""
from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable
from pathlib import Path

PROMPT_TEMPLATES: dict[str, list[str]] = {
    "python": [
        "Write a Python function `add(a, b)` that returns the sum of two numbers.",
        "Write a Python function `reverse_string(s)` that returns the reversed string.",
        "Implement a Python function `is_palindrome(s)` that returns True if s is a palindrome.",
        "Write a Python function `fibonacci(n)` that returns a list of the first n Fibonacci numbers.",
        "Write a Python function `count_words(text)` that returns a dict of word -> count.",
    ],
    "javascript": [
        "Write a JavaScript function `add(a, b)` that returns the sum of two numbers.",
        "Write a JavaScript function `reverseString(s)` that returns the reversed string.",
        "Implement a JavaScript function `isPalindrome(s)` that returns true if s is a palindrome.",
        "Write a JavaScript function `fibonacci(n)` that returns an array of the first n Fibonacci numbers.",
        "Write a JavaScript function `countWords(text)` that returns an object mapping word -> count.",
    ],
}


SOLUTIONS: dict[str, list[str]] = {
    "python": [
        """def add(a, b):\n    return a + b\n""",
        """def reverse_string(s: str) -> str:\n    return s[::-1]\n""",
        """def is_palindrome(s: str) -> bool:\n    t = ''.join(ch.lower() for ch in s if ch.isalnum())\n    return t == t[::-1]\n""",
        """def fibonacci(n: int) -> list[int]:\n    a, b = 0, 1\n    out: list[int] = []\n    for _ in range(max(0, n)):\n        out.append(a)\n        a, b = b, a + b\n    return out\n""",
        """from collections import Counter\n\n"""
        """def count_words(text: str) -> dict[str, int]:\n    words = [w for w in text.split() if w]\n    return dict(Counter(words))\n""",
    ],
    "javascript": [
        """function add(a, b) {\n  return a + b;\n}\n""",
        """function reverseString(s) {\n  return s.split('').reverse().join('');\n}\n""",
        """function isPalindrome(s) {\n  const t = (s.match(/[a-z0-9]/gi) || []).join('').toLowerCase();\n  return t === t.split('').reverse().join('');\n}\n""",
        """function fibonacci(n) {\n  const out = [];\n  let a = 0, b = 1;\n  for (let i = 0; i < Math.max(0, n); i++) {\n    out.push(a);\n    [a, b] = [b, a + b];\n  }\n  return out;\n}\n""",
        """function countWords(text) {\n  const words = text.split(/\s+/).filter(Boolean);\n  return words.reduce((acc, w) => { acc[w] = (acc[w] || 0) + 1; return acc; }, {});\n}\n""",
    ],
}


def _iter_examples(n: int, lang: str) -> Iterable[dict]:
    prompts = PROMPT_TEMPLATES.get(lang, PROMPT_TEMPLATES["python"]).copy()
    answers = SOLUTIONS.get(lang, SOLUTIONS["python"]).copy()
    for _ in range(n):
        i = random.randrange(0, len(prompts))
        j = random.randrange(0, len(answers))
        user = prompts[i]
        assistant = answers[j]
        yield {
            "messages": [
                {"role": "user", "content": user},
            ],
            "response": assistant,
        }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic coder SFT JSONL dataset")
    ap.add_argument("--output", required=True, help="Path to write JSONL (will create parent dir)")
    ap.add_argument("--n", type=int, default=50, help="Number of examples to generate")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--lang", choices=["python", "javascript"], default="python")
    args = ap.parse_args()

    random.seed(args.seed)
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write JSONL
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in _iter_examples(max(1, int(args.n)), lang=args.lang):
            fh.write(json.dumps(rec, ensure_ascii=False))
            fh.write("\n")

    print(f"Wrote {args.n} examples to {out_path}")


if __name__ == "__main__":
    main()


