"""
Script: enron_keyword_logging.py
Purpose: Iterate over a sample of Enron-QA tasks and compare the hit-rate of the
full keyword list extracted from the natural-language question with the hit-rate
when the *final* keyword is dropped (the heuristic your current agent uses).

It logs the result counts side-by-side so you can see whether the heuristic is
generally helpful or not.

Run with:
    python enron_keyword_logging.py --n 50  # test 50 random tasks
Outputs a CSV "keyword_stats.csv" for easy inspection in Excel/Sheets.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import random
import re
from pathlib import Path

from synth_ai.environments.examples.enron.art_helpers import email_search_tools  # low-level search
from synth_ai.environments.examples.enron.taskset import create_enron_taskset

# --- simple helpers ---------------------------------------------------------
STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "at",
    "in",
    "on",
    "for",
    "to",
    "with",
    "my",
    "your",
    "our",
    "did",
    "do",
    "is",
    "was",
    "were",
    "be",
    "been",
    "am",
    "when",
    "what",
    "which",
    "who",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def extract_keywords(question: str) -> list[str]:
    """Very naive keyword extractor: tokens minus stop-words."""
    tokens = [t.lower() for t in TOKEN_RE.findall(question)]
    return [t for t in tokens if t not in STOPWORDS]


# ---------------------------------------------------------------------------
async def main(n: int):
    taskset = await create_enron_taskset()
    sample = random.sample(taskset.instances, k=min(n, len(taskset.instances)))

    rows: list[dict[str, str | int]] = []
    for inst in sample:
        q = inst.impetus.instructions
        kws_full = extract_keywords(q)
        if not kws_full:
            continue

        # search using the low-level helper once so we don't need a whole env
        hits_full = email_search_tools.search_emails(inbox="user", keywords=kws_full, max_results=5)

        hits_trim = (
            email_search_tools.search_emails(inbox="user", keywords=kws_full[:-1], max_results=5)
            if len(kws_full) > 1
            else []
        )

        rows.append(
            {
                "question": q,
                "keywords_full": " ".join(kws_full),
                "hits_full": len(hits_full),
                "keywords_trim": " ".join(kws_full[:-1]),
                "hits_trim": len(hits_trim),
            }
        )

    # write CSV
    out_path = Path("keyword_stats.csv")
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="number of tasks to sample")
    args = parser.parse_args()
    asyncio.run(main(args.n))
