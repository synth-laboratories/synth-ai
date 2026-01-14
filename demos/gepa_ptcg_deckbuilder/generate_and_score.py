#!/usr/bin/env python3
"""
Generate Pokemon TCG decks with an LLM and score them via deterministic AI v4-v4 battles.

This script does NOT use the Synth interceptor. It calls the provider directly (OpenAI SDK)
and uses the Rust-parallel tcg_py scorer.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from localapi_deckbuilder import (
    CARD_POOL,
    OPPONENT_DECKS,
    evaluate_constraints,
    evaluate_deck_vs_opponents,
    get_instance,
)
from openai import OpenAI


def _build_prompt(*, challenge_id: str) -> tuple[str, str]:
    inst = get_instance(challenge_id)
    if not inst:
        raise SystemExit(
            f"Unknown challenge_id '{challenge_id}'. Options: {sorted(set(get_instance_ids()))}"
        )

    # Keep prompt short-ish but unambiguous.
    constraints = inst.get("constraints", [])
    opponent_names = inst.get("opponent_decks", list(OPPONENT_DECKS.keys()))

    sys = (
        "You are an expert Pokemon TCG deck builder. Output ONLY JSON: "
        '{"deck": ["card-id", ...]}. No markdown, no commentary.'
    )
    user = (
        f"Build a 60-card deck.\n\n"
        f"Challenge: {inst.get('name')} ({challenge_id})\n"
        f"Constraints: {json.dumps(constraints)}\n"
        f"Opponent decks for scoring: {opponent_names}\n\n"
        f"Allowed card IDs are exactly these keys:\n"
        f"{sorted(CARD_POOL.keys())}\n\n"
        f"Return ONLY JSON."
    )
    return sys, user


def get_instance_ids() -> list[str]:
    # localapi_deckbuilder prints INSTANCE_IDS at import time; use get_instance helper instead
    ids: list[str] = []
    for cid in ("basic-deck", "gardevoir-deck", "flygon-deck", "no-ex-deck", "dual-line-deck"):
        if get_instance(cid):
            ids.append(cid)
    return ids


def _parse_json_deck(text: str) -> list[str]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract first JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("LLM output is not valid JSON")
        data = json.loads(text[start : end + 1])

    deck = data.get("deck")
    if not isinstance(deck, list) or not all(isinstance(x, str) for x in deck):
        raise ValueError("JSON must be {'deck': [<string>, ...]}")
    return deck


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate and score Pokemon TCG decks (no interceptor)."
    )
    parser.add_argument("--challenge", default="basic-deck", help="Challenge ID (e.g. basic-deck).")
    parser.add_argument("--model", default="gpt-4.1-mini", help="OpenAI model name.")
    parser.add_argument("--n", type=int, default=1, help="How many decks to generate.")
    parser.add_argument(
        "--games-per-opponent",
        type=int,
        default=500,
        help="Games per opponent deck (default: 500).",
    )
    parser.add_argument(
        "--use-challenge-opponents",
        action="store_true",
        help="If set, only score against the challenge's opponent_decks (default is ALL 6 comparison decks).",
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Base seed for deterministic scoring."
    )
    parser.add_argument(
        "--max-workers", type=int, default=0, help="Thread count for Rust batch scorer (0=auto)."
    )
    args = parser.parse_args(argv)

    # Load synth-ai/.env if present (for OPENAI_API_KEY, etc.)
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        env_path = parent / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=False)
            break

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY (required for direct provider calls).")

    inst = get_instance(args.challenge)
    if not inst:
        raise SystemExit(f"Unknown challenge_id '{args.challenge}'.")

    sys, user = _build_prompt(challenge_id=args.challenge)
    client = OpenAI(api_key=api_key)

    for i in range(args.n):
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            temperature=0.7,
        )
        text = resp.choices[0].message.content or ""

        try:
            deck = _parse_json_deck(text)
        except Exception as e:
            print(f"\n=== DECK {i} ===")
            print(f"parse_error: {e}")
            print(text[:500])
            continue

        constraint_score, constraint_results = evaluate_constraints(deck, inst["constraints"])

        opponent_names = (
            inst.get("opponent_decks", list(OPPONENT_DECKS.keys()))
            if args.use_challenge_opponents
            else list(OPPONENT_DECKS.keys())
        )
        win_rate, per_opp = evaluate_deck_vs_opponents(
            deck,
            opponent_decks=opponent_names,
            num_games_per_opponent=args.games_per_opponent,
            base_seed=args.seed + i * 10_000,
            max_workers=args.max_workers,
        )

        print(f"\n=== DECK {i} ===")
        print(f"constraint_score: {constraint_score:.3f}")
        print(f"battle_win_rate: {win_rate:.3f}")
        print(
            "per_opponent:",
            [(d["opponent"], round(d["win_rate"], 3), d["wins"], d["games"]) for d in per_opp],
        )


if __name__ == "__main__":
    main()
