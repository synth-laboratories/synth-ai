#!/usr/bin/env python3
"""
Generate a fixed Wordle instances.json using the "wordfreq" package.

Usage:
  pip install wordfreq
  python -m synth_ai.environments.examples.wordle.helpers.generate_instances_wordfreq \
      --count 500 --min-zipf 3.0 --outfile synth_ai/environments/examples/wordle/instances.json

This script writes a deterministic list of 5-letter English words ranked by frequency.
Commit the resulting instances.json to remove runtime dependencies.
"""

from __future__ import annotations

import argparse
import json
import re

from wordfreq import top_n_list, zipf_frequency


def build_word_list(count: int, length: int, min_zipf: float, wordlist: str = "large") -> list[str]:
    n_candidates = max(count * 20, 5000)
    cands = [w.lower() for w in top_n_list("en", n_candidates, wordlist=wordlist)]
    cands = [w for w in cands if len(w) == length and re.fullmatch(r"[a-z]+", w)]
    scored = [(w, zipf_frequency(w, "en")) for w in cands]
    scored = [p for p in scored if p[1] >= float(min_zipf)]
    scored.sort(key=lambda t: (-t[1], t[0]))
    out: list[str] = []
    seen = set()
    for w, _ in scored:
        if w in seen:
            continue
        seen.add(w)
        out.append(w)
        if len(out) >= count:
            break
    if len(out) < count:
        raise RuntimeError(
            f"Insufficient {length}-letter words from wordfreq after filtering ({len(out)} < {count})."
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=500)
    ap.add_argument("--length", type=int, default=5)
    ap.add_argument("--min-zipf", type=float, default=3.0)
    ap.add_argument("--wordlist", type=str, default="large")
    ap.add_argument("--outfile", type=str, required=True)
    args = ap.parse_args()

    words = build_word_list(args.count, args.length, args.min_zipf, args.wordlist)

    data = {
        "name": f"Wordle Fixed TaskSet ({args.count} English words)",
        "description": f"{len(words)} {args.length}-letter English words ranked by frequency (wordfreq).",
        "defaults": {
            "word_length": args.length,
            "max_guesses": 6,
            "enforce_wordlist": True,
            "consume_invalid_attempts": True,
        },
        "instances": [{"target_word": w} for w in words],
    }

    with open(args.outfile, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {len(words)} words to {args.outfile}")


if __name__ == "__main__":
    main()
