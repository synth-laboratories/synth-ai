"""
GEPA demo for Pokemon TCG Deck Building.

This demo trains an LLM agent to build Pokemon TCG decks that satisfy constraints
and perform well against opponent decks using deterministic AI battles.

The reward is a combination of:
1. Constraint satisfaction score (0.0-0.5)
2. Win rate against opponent decks using deterministic AI v4 battles (0.0-0.5)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
from synth_ai.sdk.localapi import LocalAPIConfig, create_local_api
from synth_ai.sdk.task.contracts import (
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
    TaskInfo,
)
from synth_ai.sdk.task.trace_correlation_helpers import extract_trace_correlation_id
from synth_ai.sdk.task.validators import normalize_inference_url

# ============================================================================
# Configuration
# ============================================================================

ENGINE_BENCH_REPO_URL = "https://github.com/JoshuaPurtell/engine-bench.git"
_LOCAL_ENGINE_BENCH = Path.home() / "Documents" / "GitHub" / "engine-bench"
ENGINE_BENCH_DIR = Path(
    os.getenv(
        "ENGINE_BENCH_DIR",
        str(
            _LOCAL_ENGINE_BENCH
            if _LOCAL_ENGINE_BENCH.exists()
            else (Path.home() / ".cache" / "engine-bench")
        ),
    )
)

# ============================================================================
# Card Pool - Available cards from Dragon Frontiers expansion
# ============================================================================

CARD_POOL = {
    # Basic Pokemon
    "df-061-ralts": {
        "name": "Ralts δ",
        "type": "pokemon",
        "stage": "basic",
        "hp": 50,
        "types": ["psychic", "fire"],
    },
    "df-070-vulpix": {
        "name": "Vulpix δ",
        "type": "pokemon",
        "stage": "basic",
        "hp": 50,
        "types": ["psychic"],
    },
    "df-068-trapinch": {
        "name": "Trapinch δ",
        "type": "pokemon",
        "stage": "basic",
        "hp": 50,
        "types": ["psychic", "fire"],
    },
    "df-050-horsea": {
        "name": "Horsea δ",
        "type": "pokemon",
        "stage": "basic",
        "hp": 40,
        "types": ["fire"],
    },
    "df-045-cyndaquil": {
        "name": "Cyndaquil δ",
        "type": "pokemon",
        "stage": "basic",
        "hp": 50,
        "types": ["lightning", "fire"],
    },
    "df-067-totodile": {
        "name": "Totodile δ",
        "type": "pokemon",
        "stage": "basic",
        "hp": 50,
        "types": ["lightning", "fire"],
    },
    "df-009-pinsir": {
        "name": "Pinsir δ",
        "type": "pokemon",
        "stage": "basic",
        "hp": 70,
        "types": ["fire"],
    },
    "df-003-heracross": {
        "name": "Heracross δ",
        "type": "pokemon",
        "stage": "basic",
        "hp": 80,
        "types": ["fire", "metal"],
    },
    "df-010-snorlax": {
        "name": "Snorlax δ",
        "type": "pokemon",
        "stage": "basic",
        "hp": 90,
        "types": ["grass", "metal"],
    },
    "df-017-jynx": {
        "name": "Jynx δ",
        "type": "pokemon",
        "stage": "basic",
        "hp": 60,
        "types": ["fire"],
    },
    # Stage 1 Pokemon
    "df-033-kirlia": {
        "name": "Kirlia δ",
        "type": "pokemon",
        "stage": "stage1",
        "hp": 70,
        "types": ["psychic", "fire"],
        "evolves_from": "df-061-ralts",
    },
    "df-008-ninetales": {
        "name": "Ninetales δ",
        "type": "pokemon",
        "stage": "stage1",
        "hp": 80,
        "types": ["psychic"],
        "evolves_from": "df-070-vulpix",
    },
    "df-024-vibrava": {
        "name": "Vibrava δ",
        "type": "pokemon",
        "stage": "stage1",
        "hp": 70,
        "types": ["psychic", "fire"],
        "evolves_from": "df-068-trapinch",
    },
    "df-022-seadra": {
        "name": "Seadra δ",
        "type": "pokemon",
        "stage": "stage1",
        "hp": 70,
        "types": ["fire"],
        "evolves_from": "df-050-horsea",
    },
    "df-036-quilava": {
        "name": "Quilava δ",
        "type": "pokemon",
        "stage": "stage1",
        "hp": 70,
        "types": ["lightning", "fire"],
        "evolves_from": "df-045-cyndaquil",
    },
    "df-027-croconaw": {
        "name": "Croconaw δ",
        "type": "pokemon",
        "stage": "stage1",
        "hp": 70,
        "types": ["lightning", "fire"],
        "evolves_from": "df-067-totodile",
    },
    # Stage 2 Pokemon / Pokemon-ex
    "df-093-gardevoir-ex": {
        "name": "Gardevoir ex δ",
        "type": "pokemon",
        "stage": "stage2",
        "hp": 150,
        "types": ["psychic", "fire"],
        "evolves_from": "df-033-kirlia",
        "is_ex": True,
    },
    "df-092-flygon-ex": {
        "name": "Flygon ex δ",
        "type": "pokemon",
        "stage": "stage2",
        "hp": 150,
        "types": ["psychic", "fire"],
        "evolves_from": "df-024-vibrava",
        "is_ex": True,
    },
    "df-094-kingdra-ex": {
        "name": "Kingdra ex δ",
        "type": "pokemon",
        "stage": "stage2",
        "hp": 140,
        "types": ["fire", "water"],
        "evolves_from": "df-022-seadra",
        "is_ex": True,
    },
    "df-012-typhlosion": {
        "name": "Typhlosion δ",
        "type": "pokemon",
        "stage": "stage2",
        "hp": 110,
        "types": ["lightning", "fire"],
        "evolves_from": "df-036-quilava",
    },
    "df-002-feraligatr": {
        "name": "Feraligatr δ",
        "type": "pokemon",
        "stage": "stage2",
        "hp": 110,
        "types": ["lightning", "fire"],
        "evolves_from": "df-027-croconaw",
    },
    # Legendary Pokemon-ex (Basic)
    "df-095-latias-ex": {
        "name": "Latias ex δ",
        "type": "pokemon",
        "stage": "basic",
        "hp": 100,
        "types": ["fire", "water"],
        "is_ex": True,
    },
    "df-096-latios-ex": {
        "name": "Latios ex δ",
        "type": "pokemon",
        "stage": "basic",
        "hp": 100,
        "types": ["lightning", "psychic"],
        "is_ex": True,
    },
    "df-097-rayquaza-ex": {
        "name": "Rayquaza ex δ",
        "type": "pokemon",
        "stage": "basic",
        "hp": 110,
        "types": ["fire", "lightning"],
        "is_ex": True,
    },
    # Trainers
    "df-082-tv-reporter": {"name": "TV Reporter", "type": "trainer", "subtype": "supporter"},
    "df-079-prof-elms-training": {
        "name": "Professor Elm's Training Method",
        "type": "trainer",
        "subtype": "supporter",
    },
    "df-072-buffer-piece": {"name": "Buffer Piece", "type": "trainer", "subtype": "tool"},
    # Energy (from Holon Phantoms set)
    "hp-109-psychic-energy": {"name": "Psychic Energy", "type": "energy", "energy_type": "psychic"},
    "hp-106-fire-energy": {"name": "Fire Energy", "type": "energy", "energy_type": "fire"},
    "hp-108-lightning-energy": {
        "name": "Lightning Energy",
        "type": "energy",
        "energy_type": "lightning",
    },
    "hp-105-grass-energy": {"name": "Grass Energy", "type": "energy", "energy_type": "grass"},
    "hp-107-water-energy": {"name": "Water Energy", "type": "energy", "energy_type": "water"},
    "hp-095-metal-energy": {"name": "Metal Energy", "type": "energy", "energy_type": "metal"},
}

# Evolution lines for constraint checking
EVOLUTION_LINES = {
    "gardevoir": ["df-061-ralts", "df-033-kirlia", "df-093-gardevoir-ex"],
    "flygon": ["df-068-trapinch", "df-024-vibrava", "df-092-flygon-ex"],
    "kingdra": ["df-050-horsea", "df-022-seadra", "df-094-kingdra-ex"],
    "typhlosion": ["df-045-cyndaquil", "df-036-quilava", "df-012-typhlosion"],
    "feraligatr": ["df-067-totodile", "df-027-croconaw", "df-002-feraligatr"],
    "ninetales": ["df-070-vulpix", "df-008-ninetales"],
}

# ============================================================================
# Opponent Decks for Battle Testing
# ============================================================================

OPPONENT_DECKS = {
    "gardevoir_control": (
        ["df-061-ralts"] * 4
        + ["df-033-kirlia"] * 3
        + ["df-093-gardevoir-ex"] * 2
        + ["df-017-jynx"] * 2
        + ["df-070-vulpix"] * 3
        + ["df-008-ninetales"] * 2
        + ["df-010-snorlax"] * 2
        + ["df-082-tv-reporter"] * 4
        + ["df-079-prof-elms-training"] * 4
        + ["df-072-buffer-piece"] * 2
        + ["hp-109-psychic-energy"] * 16
        + ["hp-106-fire-energy"] * 16
    ),
    "flygon_storm": (
        ["df-068-trapinch"] * 4
        + ["df-024-vibrava"] * 3
        + ["df-092-flygon-ex"] * 2
        + ["df-009-pinsir"] * 3
        + ["df-003-heracross"] * 2
        + ["df-082-tv-reporter"] * 4
        + ["df-079-prof-elms-training"] * 3
        + ["df-072-buffer-piece"] * 2
        + ["hp-109-psychic-energy"] * 20
        + ["hp-106-fire-energy"] * 17
    ),
    "legendary_dragons": (
        ["df-095-latias-ex"] * 2
        + ["df-096-latios-ex"] * 2
        + ["df-097-rayquaza-ex"] * 2
        + ["df-009-pinsir"] * 4
        + ["df-003-heracross"] * 4
        + ["df-010-snorlax"] * 2
        + ["df-082-tv-reporter"] * 4
        + ["df-079-prof-elms-training"] * 4
        + ["df-072-buffer-piece"] * 2
        + ["hp-106-fire-energy"] * 17
        + ["hp-108-lightning-energy"] * 17
    ),
    # Added comparison decks (so we have 6 total)
    "kingdra_splash": (
        ["df-050-horsea"] * 4
        + ["df-022-seadra"] * 3
        + ["df-094-kingdra-ex"] * 2
        + ["df-010-snorlax"] * 2
        + ["df-009-pinsir"] * 3
        + ["df-082-tv-reporter"] * 4
        + ["df-079-prof-elms-training"] * 4
        + ["df-072-buffer-piece"] * 2
        + ["hp-107-water-energy"] * 18
        + ["hp-106-fire-energy"] * 18
        + ["hp-109-psychic-energy"] * 2
    ),
    "typhlosion_blaze": (
        ["df-045-cyndaquil"] * 4
        + ["df-036-quilava"] * 3
        + ["df-012-typhlosion"] * 2
        + ["df-070-vulpix"] * 3
        + ["df-008-ninetales"] * 2
        + ["df-003-heracross"] * 2
        + ["df-010-snorlax"] * 2
        + ["df-082-tv-reporter"] * 4
        + ["df-079-prof-elms-training"] * 4
        + ["df-072-buffer-piece"] * 2
        + ["hp-106-fire-energy"] * 22
        + ["hp-108-lightning-energy"] * 12
    ),
    "feraligatr_surge": (
        ["df-067-totodile"] * 4
        + ["df-027-croconaw"] * 3
        + ["df-002-feraligatr"] * 2
        + ["df-009-pinsir"] * 3
        + ["df-010-snorlax"] * 2
        + ["df-017-jynx"] * 2
        + ["df-082-tv-reporter"] * 4
        + ["df-079-prof-elms-training"] * 4
        + ["df-072-buffer-piece"] * 2
        + ["hp-107-water-energy"] * 18
        + ["hp-108-lightning-energy"] * 16
    ),
}


DEFAULT_EVAL_WORKERS = int(os.getenv("PTCG_EVAL_WORKERS", "0") or "0")  # 0 => auto
DEFAULT_GAMES_PER_OPPONENT = int(os.getenv("PTCG_GAMES_PER_OPPONENT", "2") or "2")

# ============================================================================
# Constraint Checking
# ============================================================================


def check_constraint(deck: list[str], constraint: dict) -> tuple[bool, str]:
    """Check if a deck satisfies a constraint. Returns (satisfied, explanation)."""
    ctype = constraint["type"]

    if ctype == "exactly_60_cards":
        satisfied = len(deck) == 60
        return satisfied, f"Deck has {len(deck)} cards (need 60)"

    elif ctype == "at_least_basic":
        count = constraint["count"]
        basics = [c for c in deck if CARD_POOL.get(c, {}).get("stage") == "basic"]
        satisfied = len(basics) >= count
        return satisfied, f"Deck has {len(basics)} Basic Pokemon (need at least {count})"

    elif ctype == "no_more_than_4_copies":
        counts = Counter(deck)
        violations = []
        for card, cnt in counts.items():
            card_info = CARD_POOL.get(card, {})
            # Basic energy can have unlimited copies
            if card_info.get("type") == "energy" and not card_info.get("special"):
                continue
            if cnt > 4:
                violations.append(f"{card}: {cnt}")
        satisfied = len(violations) == 0
        return (
            satisfied,
            f"Violations: {violations}" if violations else "All cards within 4-copy limit",
        )

    elif ctype == "include_evolution_line":
        pokemon = constraint["pokemon"]
        line = EVOLUTION_LINES.get(pokemon.lower(), [])
        missing = [c for c in line if c not in deck]
        satisfied = len(missing) == 0
        return (
            satisfied,
            f"Missing from {pokemon} line: {missing}"
            if missing
            else f"Full {pokemon} line present",
        )

    elif ctype == "energy_ratio":
        min_pct, max_pct = constraint["min_pct"], constraint["max_pct"]
        energy_count = sum(1 for c in deck if CARD_POOL.get(c, {}).get("type") == "energy")
        if len(deck) == 0:
            return False, "Empty deck"
        ratio = (energy_count / len(deck)) * 100
        satisfied = min_pct <= ratio <= max_pct
        return satisfied, f"Energy ratio: {ratio:.1f}% (need {min_pct}%-{max_pct}%)"

    elif ctype == "pokemon_ratio":
        min_pct, max_pct = constraint["min_pct"], constraint["max_pct"]
        pokemon_count = sum(1 for c in deck if CARD_POOL.get(c, {}).get("type") == "pokemon")
        if len(deck) == 0:
            return False, "Empty deck"
        ratio = (pokemon_count / len(deck)) * 100
        satisfied = min_pct <= ratio <= max_pct
        return satisfied, f"Pokemon ratio: {ratio:.1f}% (need {min_pct}%-{max_pct}%)"

    elif ctype == "trainer_ratio":
        min_pct, max_pct = constraint["min_pct"], constraint["max_pct"]
        trainer_count = sum(1 for c in deck if CARD_POOL.get(c, {}).get("type") == "trainer")
        if len(deck) == 0:
            return False, "Empty deck"
        ratio = (trainer_count / len(deck)) * 100
        satisfied = min_pct <= ratio <= max_pct
        return satisfied, f"Trainer ratio: {ratio:.1f}% (need {min_pct}%-{max_pct}%)"

    elif ctype == "max_ex_pokemon":
        max_count = constraint["count"]
        ex_count = sum(1 for c in deck if CARD_POOL.get(c, {}).get("is_ex"))
        satisfied = ex_count <= max_count
        return satisfied, f"Pokemon-ex count: {ex_count} (max {max_count})"

    elif ctype == "include_card":
        card = constraint["card"]
        count = constraint["count"]
        actual = deck.count(card)
        satisfied = actual >= count
        return satisfied, f"Has {actual} copies of {card} (need at least {count})"

    elif ctype == "dual_energy_types":
        type1, type2 = constraint["type1"], constraint["type2"]
        valid_energies = [f"ENERGY-{type1.upper()}", f"ENERGY-{type2.upper()}", "ENERGY-COLORLESS"]
        invalid = [
            c
            for c in deck
            if CARD_POOL.get(c, {}).get("type") == "energy" and c not in valid_energies
        ]
        satisfied = len(invalid) == 0
        return (
            satisfied,
            f"Invalid energies: {invalid}" if invalid else f"Only {type1}/{type2} energy used",
        )

    return False, f"Unknown constraint type: {ctype}"


def evaluate_constraints(deck: list[str], constraints: list[dict]) -> tuple[float, list[dict]]:
    """Evaluate all constraints and return (score, detailed_results)."""
    results = []
    satisfied_count = 0

    for constraint in constraints:
        satisfied, explanation = check_constraint(deck, constraint)
        weight = constraint.get("weight", 1.0)
        results.append(
            {
                "type": constraint["type"],
                "satisfied": satisfied,
                "explanation": explanation,
                "weight": weight,
            }
        )
        if satisfied:
            satisfied_count += weight

    total_weight = sum(c.get("weight", 1.0) for c in constraints)
    score = satisfied_count / total_weight if total_weight > 0 else 0.0

    return score, results


# ============================================================================
# Engine Setup
# ============================================================================


def ensure_engine_bench_repo() -> None:
    """Clone or update engine-bench repo if needed."""
    if not ENGINE_BENCH_DIR.exists():
        print(f"[deckbuilder] Cloning engine-bench to {ENGINE_BENCH_DIR}...")
        subprocess.run(
            ["git", "clone", "--depth", "1", ENGINE_BENCH_REPO_URL, str(ENGINE_BENCH_DIR)],
            check=True,
        )
    else:
        # Pull latest (ignore errors)
        subprocess.run(
            ["git", "-C", str(ENGINE_BENCH_DIR), "pull", "--ff-only"],
            check=False,
            capture_output=True,
        )


def ensure_tcg_py_built() -> bool:
    """Ensure the tcg_py Python extension is built. Returns True if available."""
    # Check if already importable
    try:
        import tcg_py  # noqa: F401

        return True
    except ImportError:
        pass

    # Try to build
    tcg_py_dir = ENGINE_BENCH_DIR / "tcg_py"
    if not tcg_py_dir.exists():
        print(f"[deckbuilder] tcg_py not found at {tcg_py_dir}")
        return False

    print("[deckbuilder] Building tcg_py extension...")
    try:
        subprocess.run(
            ["maturin", "develop", "--release"],
            cwd=str(tcg_py_dir),
            check=True,
            capture_output=True,
        )
        return True
    except FileNotFoundError:
        # Common local setup issue: maturin missing
        print(
            "[deckbuilder] Failed to build tcg_py: `maturin` not found. "
            "Install it (e.g. `python -m pip install maturin`) and retry."
        )
        return False
    except Exception as e:
        print(f"[deckbuilder] Failed to build tcg_py: {e}")
        return False


# Initialize engine repo
ensure_engine_bench_repo()

# ============================================================================
# Game Simulation
# ============================================================================


def run_game_sync(
    p1_deck: list[str],
    p2_deck: list[str],
    game_seed: int,
    p1_ai_seed: int,
    p2_ai_seed: int = 456,
    max_steps: int = 1000,
) -> dict:
    """Run a single game between two decks using AI v4 for both players."""
    try:
        import tcg_py
    except ImportError:
        return {
            "winner": None,
            "turns": 0,
            "steps": 0,
            "p1_prizes": 6,
            "p2_prizes": 6,
            "end_reason": "tcg_py not available",
        }

    try:
        # Deckbuilder requires deterministic AI v4 vs AI v4.
        if not hasattr(tcg_py, "run_ai_vs_ai"):
            raise RuntimeError(
                "tcg_py is missing run_ai_vs_ai (AI v4 vs AI v4). "
                "Rebuild tcg_py against the v4 engine (overzealous) and reinstall it."
            )

        result = tcg_py.run_ai_vs_ai(
            p1_deck=p1_deck,
            p2_deck=p2_deck,
            game_seed=game_seed,
            p1_ai_seed=p1_ai_seed,
            p2_ai_seed=p2_ai_seed,
            max_steps=max_steps,
        )

        # Diagnostics: surface whether the engine actually played any meaningful game actions.
        history = list(getattr(result, "history", []) or [])
        action_count = 0
        attack_count = 0
        for turn in history:
            actions = list(getattr(turn, "actions", []) or [])
            action_count += len(actions)
            attack_count += sum(1 for a in actions if "DeclareAttack" in a)

        suspicious_no_prize_gameover = (
            result.end_reason == "GameOver"
            and result.p1_prizes_remaining == 6
            and result.p2_prizes_remaining == 6
        )

        return {
            "winner": result.winner,
            "turns": result.turns,
            "steps": result.steps,
            "p1_prizes": result.p1_prizes_remaining,
            "p2_prizes": result.p2_prizes_remaining,
            "end_reason": result.end_reason,
            "history_len": len(history),
            "action_count": action_count,
            "attack_count": attack_count,
            "suspicious_no_prize_gameover": suspicious_no_prize_gameover,
        }
    except Exception as e:
        import traceback

        traceback.print_exc()
        return {
            "winner": None,
            "turns": 0,
            "steps": 0,
            "p1_prizes": 6,
            "p2_prizes": 6,
            "end_reason": f"error: {str(e)}",
        }


def evaluate_deck_vs_opponents(
    deck: list[str],
    opponent_decks: list[str],
    num_games_per_opponent: int,
    base_seed: int,
    *,
    max_workers: int | None = None,
) -> tuple[float, list[dict]]:
    """Evaluate deck against multiple opponents. Returns (win_rate, game_results)."""
    import tcg_py

    all_results: list[dict] = []
    total_wins = 0
    total_games = 0

    if max_workers is None:
        max_workers = DEFAULT_EVAL_WORKERS

    for opp_idx, opp_name in enumerate(opponent_decks):
        opp_deck = OPPONENT_DECKS.get(opp_name)
        if not opp_deck:
            continue

        # We run 50/50 as P1 vs P2 to reduce first-player advantage.
        half = num_games_per_opponent // 2
        remainder = num_games_per_opponent - (2 * half)

        def _make_seeds(
            offset: int, count: int, _opp_idx: int = opp_idx
        ) -> tuple[list[int], list[int], list[int]]:
            game_seeds = [base_seed + _opp_idx * 1_000_000 + offset + i for i in range(count)]
            p1_ai_seeds = [
                base_seed + 10_000 + _opp_idx * 1_000_000 + offset + i for i in range(count)
            ]
            p2_ai_seeds = [
                base_seed + 20_000 + _opp_idx * 1_000_000 + offset + i for i in range(count)
            ]
            return game_seeds, p1_ai_seeds, p2_ai_seeds

        # Batch A: candidate deck as P1
        game_seeds_a, p1_ai_seeds_a, p2_ai_seeds_a = _make_seeds(0, half + remainder)
        batch_a = tcg_py.run_ai_vs_ai_batch(
            p1_deck=list(deck),
            p2_deck=list(opp_deck),
            game_seeds=game_seeds_a,
            p1_ai_seeds=p1_ai_seeds_a,
            p2_ai_seeds=p2_ai_seeds_a,
            max_steps=2000,
            max_workers=max_workers or 0,
            sample_stride=0,
            max_samples=0,
        )

        # Batch B: candidate deck as P2
        game_seeds_b, p1_ai_seeds_b, p2_ai_seeds_b = _make_seeds(500_000, half)
        batch_b = tcg_py.run_ai_vs_ai_batch(
            p1_deck=list(opp_deck),
            p2_deck=list(deck),
            game_seeds=game_seeds_b,
            p1_ai_seeds=p1_ai_seeds_b,
            p2_ai_seeds=p2_ai_seeds_b,
            max_steps=2000,
            max_workers=max_workers or 0,
            sample_stride=0,
            max_samples=0,
        )

        # Wins for candidate deck:
        # - In batch A, candidate is P1 => candidate wins == P1 wins.
        # - In batch B, candidate is P2 => candidate wins == P2 wins.
        candidate_wins = int(batch_a.p1_wins) + int(batch_b.p2_wins)
        games_here = int(batch_a.total_games) + int(batch_b.total_games)

        total_wins += candidate_wins
        total_games += games_here

        all_results.append(
            {
                "opponent": opp_name,
                "games": games_here,
                "wins": candidate_wins,
                "win_rate": (candidate_wins / games_here) if games_here else 0.0,
                "batch_a": {
                    "total": int(batch_a.total_games),
                    "p1_wins": int(batch_a.p1_wins),
                    "p2_wins": int(batch_a.p2_wins),
                    "draws": int(batch_a.draws),
                    "win_conditions": list(batch_a.win_condition_counts),
                    "mean_turns": float(batch_a.mean_turns),
                    "mean_steps": float(batch_a.mean_steps),
                },
                "batch_b": {
                    "total": int(batch_b.total_games),
                    "p1_wins": int(batch_b.p1_wins),
                    "p2_wins": int(batch_b.p2_wins),
                    "draws": int(batch_b.draws),
                    "win_conditions": list(batch_b.win_condition_counts),
                    "mean_turns": float(batch_b.mean_turns),
                    "mean_steps": float(batch_b.mean_steps),
                },
            }
        )

    win_rate = total_wins / total_games if total_games > 0 else 0.0
    return win_rate, all_results


# ============================================================================
# Task Instances - Deckbuilding challenges
# ============================================================================

DECKBUILDING_CHALLENGES = [
    {
        "id": "basic-deck",
        "name": "Basic Deck Building",
        "description": "Build a valid 60-card deck with proper structure",
        "constraints": [
            {"type": "exactly_60_cards", "weight": 2.0},
            {"type": "at_least_basic", "count": 8, "weight": 1.5},
            {"type": "no_more_than_4_copies", "weight": 1.5},
            {"type": "energy_ratio", "min_pct": 30, "max_pct": 50, "weight": 1.0},
            {"type": "pokemon_ratio", "min_pct": 20, "max_pct": 40, "weight": 1.0},
        ],
        "opponent_decks": ["gardevoir_control", "flygon_storm"],
        "num_games_per_opponent": 2,
    },
    {
        "id": "gardevoir-deck",
        "name": "Gardevoir Deck",
        "description": "Build a deck featuring the Gardevoir evolution line",
        "constraints": [
            {"type": "exactly_60_cards", "weight": 2.0},
            {"type": "at_least_basic", "count": 10, "weight": 1.0},
            {"type": "no_more_than_4_copies", "weight": 1.5},
            {"type": "include_evolution_line", "pokemon": "gardevoir", "weight": 2.0},
            {"type": "dual_energy_types", "type1": "psychic", "type2": "fire", "weight": 1.5},
            {"type": "max_ex_pokemon", "count": 3, "weight": 1.0},
        ],
        "opponent_decks": ["flygon_storm", "legendary_dragons"],
        "num_games_per_opponent": 2,
    },
    {
        "id": "flygon-deck",
        "name": "Flygon Deck",
        "description": "Build a deck featuring the Flygon evolution line with Pinsir support",
        "constraints": [
            {"type": "exactly_60_cards", "weight": 2.0},
            {"type": "at_least_basic", "count": 10, "weight": 1.0},
            {"type": "no_more_than_4_copies", "weight": 1.5},
            {"type": "include_evolution_line", "pokemon": "flygon", "weight": 2.0},
            {"type": "include_card", "card": "df-009-pinsir", "count": 2, "weight": 1.5},
            {"type": "trainer_ratio", "min_pct": 10, "max_pct": 25, "weight": 1.0},
        ],
        "opponent_decks": ["gardevoir_control", "legendary_dragons"],
        "num_games_per_opponent": 2,
    },
    {
        "id": "no-ex-deck",
        "name": "No Pokemon-ex Challenge",
        "description": "Build a competitive deck without any Pokemon-ex",
        "constraints": [
            {"type": "exactly_60_cards", "weight": 2.0},
            {"type": "at_least_basic", "count": 12, "weight": 1.0},
            {"type": "no_more_than_4_copies", "weight": 1.5},
            {"type": "max_ex_pokemon", "count": 0, "weight": 2.5},
            {"type": "pokemon_ratio", "min_pct": 25, "max_pct": 45, "weight": 1.0},
            {"type": "energy_ratio", "min_pct": 35, "max_pct": 55, "weight": 1.0},
        ],
        "opponent_decks": ["gardevoir_control", "flygon_storm", "legendary_dragons"],
        "num_games_per_opponent": 2,
    },
    {
        "id": "dual-line-deck",
        "name": "Dual Evolution Lines",
        "description": "Build a deck with both Typhlosion and Feraligatr lines",
        "constraints": [
            {"type": "exactly_60_cards", "weight": 2.0},
            {"type": "at_least_basic", "count": 12, "weight": 1.0},
            {"type": "no_more_than_4_copies", "weight": 1.5},
            {"type": "include_evolution_line", "pokemon": "typhlosion", "weight": 2.0},
            {"type": "include_evolution_line", "pokemon": "feraligatr", "weight": 2.0},
        ],
        "opponent_decks": ["gardevoir_control", "flygon_storm"],
        "num_games_per_opponent": 2,
    },
]

INSTANCE_IDS = [c["id"] for c in DECKBUILDING_CHALLENGES]
print(f"[deckbuilder] Loaded {len(INSTANCE_IDS)} challenges")


def get_instance(instance_id: str) -> dict | None:
    """Get instance by ID."""
    for c in DECKBUILDING_CHALLENGES:
        if c["id"] == instance_id:
            return c
    return None


# ============================================================================
# System Prompt
# ============================================================================

DEFAULT_SYSTEM_PROMPT = """You are an expert Pokemon TCG deck builder. Build decks that satisfy the given constraints AND win battles.

RULES:
- A deck must have exactly 60 cards
- Maximum 4 copies of any non-energy card (basic energy has no limit)
- Include enough Basic Pokemon (8-12 minimum)
- Evolution lines must be complete (Basic -> Stage 1 -> Stage 2)

STRATEGY TIPS:
- Include draw support trainers (TV Reporter)
- Match energy types to your Pokemon's attack costs
- Include multiple copies of key Pokemon for consistency
- Balance attackers with support Pokemon

OUTPUT FORMAT:
Respond with ONLY a JSON object:
{"deck": ["card-id-1", "card-id-2", ...]}

The deck array must contain exactly 60 card IDs from the available pool.
Use the exact card IDs provided (e.g., "df-061-ralts", "ENERGY-PSYCHIC").
"""


# ============================================================================
# Deck Parser
# ============================================================================


def parse_deck_response(response: str) -> tuple[list[str] | None, str]:
    """Parse deck from LLM response. Returns (deck, error_message)."""
    # Try to extract JSON with deck key
    try:
        json_match = re.search(r'\{[^{}]*"deck"\s*:\s*\[[^\]]*\][^{}]*\}', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            if "deck" in data and isinstance(data["deck"], list):
                return data["deck"], ""
    except json.JSONDecodeError:
        pass

    # Try to find any JSON array
    try:
        array_match = re.search(r'\[[^\[\]]*(?:"[^"]*"[^\[\]]*)+\]', response)
        if array_match:
            deck = json.loads(array_match.group())
            if isinstance(deck, list) and len(deck) > 0:
                return deck, ""
    except json.JSONDecodeError:
        pass

    return None, "Could not parse deck from response"


# ============================================================================
# LLM Caller
# ============================================================================


async def call_llm(
    system_prompt: str,
    user_prompt: str,
    inference_url: str,
    api_key: str | None,
    model: str = "gpt-4.1-mini",
) -> str:
    """Call the LLM through the Synth interceptor."""
    inference_url = normalize_inference_url(inference_url)
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        # Interceptor primarily uses X-API-Key, but keep Authorization for compatibility.
        headers["X-API-Key"] = api_key
        headers["Authorization"] = f"Bearer {api_key}"
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            inference_url,
            headers=headers,
            json={
                "model": model,
                "temperature": 0.7,
                "max_tokens": 4096,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


# ============================================================================
# Rollout Handler
# ============================================================================


async def run_rollout(request: RolloutRequest, fastapi_request: Any) -> RolloutResponse:
    """Execute a single rollout: build deck, check constraints, battle opponents."""
    env_config = request.env.config or {}
    policy_config = request.policy.config or {}

    # Get instance by seed
    seed = request.env.seed or 0
    instance_id = env_config.get("instance_id") or INSTANCE_IDS[seed % len(INSTANCE_IDS)]
    instance = get_instance(instance_id)

    if not instance:
        return RolloutResponse(
            trace_correlation_id=request.trace_correlation_id,
            reward_info=RolloutMetrics(
                outcome_reward=0.0, details={"error": f"Unknown instance: {instance_id}"}
            ),
        )

    # Get policy config
    system_prompt = policy_config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    model = policy_config.get("model", "gpt-4.1-mini")
    inference_url = policy_config.get("inference_url")
    api_key = policy_config.get("api_key")

    if not inference_url:
        return RolloutResponse(
            trace_correlation_id=request.trace_correlation_id,
            reward_info=RolloutMetrics(
                outcome_reward=0.0, details={"error": "No inference_url provided"}
            ),
        )

    # Extract trace_correlation_id from inference_url early (for all error paths)
    policy_cfg_for_trace = {
        key: value
        for key, value in policy_config.items()
        if key not in {"trace_correlation_id", "trace"}
    }
    trace_correlation_id = (
        extract_trace_correlation_id(
            policy_config=policy_cfg_for_trace,
            inference_url=str(inference_url or ""),
        )
        or request.trace_correlation_id
    )

    # Debug: log inference URL to help diagnose hydration failures
    print(
        f"[deckbuilder] Using inference_url: {inference_url[:100]}..."
        if len(inference_url) > 100
        else f"[deckbuilder] Using inference_url: {inference_url}"
    )
    print(f"[deckbuilder] Trace correlation ID: {trace_correlation_id}")

    print(f"[deckbuilder] Challenge: {instance['name']} (seed={seed})")

    # Build user prompt
    card_pool_lines = []
    for card_id, info in CARD_POOL.items():
        parts = [f"{card_id}: {info['name']} ({info['type']}"]
        if info.get("stage"):
            parts.append(f", {info['stage']}")
        if info.get("hp"):
            parts.append(f", HP:{info['hp']}")
        if info.get("types"):
            parts.append(f", types:{info['types']}")
        if info.get("energy_type"):
            parts.append(f", provides:{info['energy_type']}")
        if info.get("is_ex"):
            parts.append(", ex")
        parts.append(")")
        card_pool_lines.append("".join(parts))

    constraint_lines = []
    for c in instance["constraints"]:
        ctype = c["type"]
        if ctype == "exactly_60_cards":
            constraint_lines.append("- Deck must have exactly 60 cards")
        elif ctype == "at_least_basic":
            constraint_lines.append(f"- At least {c['count']} Basic Pokemon")
        elif ctype == "no_more_than_4_copies":
            constraint_lines.append("- Max 4 copies of any card (except basic energy)")
        elif ctype == "include_evolution_line":
            line = EVOLUTION_LINES.get(c["pokemon"].lower(), [])
            constraint_lines.append(f"- Must include {c['pokemon']} line: {line}")
        elif ctype == "energy_ratio":
            constraint_lines.append(f"- Energy must be {c['min_pct']}%-{c['max_pct']}% of deck")
        elif ctype == "pokemon_ratio":
            constraint_lines.append(f"- Pokemon must be {c['min_pct']}%-{c['max_pct']}% of deck")
        elif ctype == "trainer_ratio":
            constraint_lines.append(f"- Trainers must be {c['min_pct']}%-{c['max_pct']}% of deck")
        elif ctype == "max_ex_pokemon":
            constraint_lines.append(f"- Max {c['count']} Pokemon-ex cards")
        elif ctype == "include_card":
            constraint_lines.append(f"- Must include at least {c['count']}x {c['card']}")
        elif ctype == "dual_energy_types":
            constraint_lines.append(f"- Only use {c['type1']} and {c['type2']} energy")

    opponent_info = (
        f"Your deck will battle against: {', '.join(instance.get('opponent_decks', []))}"
    )

    user_prompt = f"""## Challenge: {instance["name"]}
{instance["description"]}

## Constraints (ALL must be satisfied):
{chr(10).join(constraint_lines)}

## Battle Info
{opponent_info}

## Available Cards:
{chr(10).join(card_pool_lines)}

Build the deck now. Output ONLY the JSON with the deck array."""

    try:
        # Single LLM call to build deck
        response = await call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            inference_url=inference_url,
            api_key=api_key,
            model=model,
        )

        print(f"[deckbuilder] Response length: {len(response)} chars")

        # Parse deck
        deck, parse_error = parse_deck_response(response)

        if not deck:
            print(f"[deckbuilder] Parse failed: {parse_error}")
            return RolloutResponse(
                trace_correlation_id=trace_correlation_id,
                reward_info=RolloutMetrics(
                    outcome_reward=0.0,
                    details={"error": parse_error, "response_preview": response[:500]},
                ),
            )

        # Basic structural validation: deck must be a list of non-empty strings.
        non_strings = [c for c in deck if not isinstance(c, str)]
        if non_strings:
            return RolloutResponse(
                trace_correlation_id=trace_correlation_id,
                reward_info=RolloutMetrics(
                    outcome_reward=0.0,
                    details={
                        "error": "Invalid deck format (non-string card IDs)",
                        "deck_size": len(deck),
                        "invalid_entries_preview": [repr(c)[:200] for c in non_strings[:10]],
                    },
                ),
            )
        deck = [c.strip() for c in deck]
        empty_ids = [c for c in deck if not c]
        if empty_ids:
            return RolloutResponse(
                trace_correlation_id=trace_correlation_id,
                reward_info=RolloutMetrics(
                    outcome_reward=0.0,
                    details={
                        "error": "Invalid deck format (empty card ID)",
                        "deck_size": len(deck),
                    },
                ),
            )

        # Validate card IDs
        invalid_cards = [c for c in deck if c not in CARD_POOL]
        if invalid_cards:
            print(f"[deckbuilder] Invalid cards: {invalid_cards[:5]}")
            return RolloutResponse(
                trace_correlation_id=trace_correlation_id,
                reward_info=RolloutMetrics(
                    outcome_reward=0.0,
                    details={
                        "error": "Invalid card IDs",
                        "invalid_cards": invalid_cards[:10],
                        "deck_size": len(deck),
                    },
                ),
            )

        # Evaluate constraints and fail-fast if the deck violates any task-specific requirements.
        constraint_score, constraint_results = evaluate_constraints(deck, instance["constraints"])
        print(
            f"[deckbuilder] Constraint score: {constraint_score:.2f} ({sum(1 for r in constraint_results if r['satisfied'])}/{len(constraint_results)})"
        )
        unsatisfied = [r for r in constraint_results if not r.get("satisfied")]
        if unsatisfied:
            # Treat ANY constraint failure as an invalid deck. This avoids giving partial credit
            # for structurally illegal decks (wrong size, too many copies, etc.) and enforces
            # per-task requirements (no-ex, required evo lines, ratios, etc.).
            return RolloutResponse(
                trace_correlation_id=trace_correlation_id,
                reward_info=RolloutMetrics(
                    outcome_reward=0.0,
                    details={
                        "error": "Invalid deck (failed task requirements)",
                        "deck_size": len(deck),
                        "constraint_score": constraint_score,
                        "constraint_results": constraint_results,
                        "failed_constraints": unsatisfied,
                        "deck": deck,
                    },
                ),
            )

        # Battle against opponent decks (50% of score)
        win_rate = 0.0
        game_results = []

        tcg_available = ensure_tcg_py_built()
        if tcg_available:
            win_rate, game_results = evaluate_deck_vs_opponents(
                deck=deck,
                opponent_decks=instance.get("opponent_decks", ["gardevoir_control"]),
                num_games_per_opponent=instance.get("num_games_per_opponent", 2),
                base_seed=seed * 1000,
            )
            wins_total = sum(int(r.get("wins", 0)) for r in game_results if isinstance(r, dict))
            games_total = sum(int(r.get("games", 0)) for r in game_results if isinstance(r, dict))
            print(f"[deckbuilder] Win rate: {win_rate:.2f} ({wins_total}/{games_total} games)")
        else:
            print("[deckbuilder] tcg_py not available, skipping battles")
            game_results = [{"error": "tcg_py not available"}]

        # Final reward: 50% constraints + 50% win rate
        final_reward = 0.5 * constraint_score + 0.5 * win_rate

        print(
            f"[deckbuilder] Final reward: {final_reward:.2f} (constraints={constraint_score:.2f}, wins={win_rate:.2f})"
        )

        return RolloutResponse(
            trace_correlation_id=trace_correlation_id,
            reward_info=RolloutMetrics(
                outcome_reward=final_reward,
                details={
                    "instance_id": instance_id,
                    "deck_size": len(deck),
                    "constraint_score": constraint_score,
                    "constraint_results": constraint_results,
                    "win_rate": win_rate,
                    "game_results": game_results,
                    "deck": deck,
                },
            ),
            metadata={
                "instance_id": instance_id,
                "constraint_score": constraint_score,
                "win_rate": win_rate,
                "final_reward": final_reward,
            },
        )

    except Exception as e:
        import traceback

        traceback.print_exc()

        # Use trace_correlation_id extracted earlier, or fallback to request
        _trace_correlation_id = (
            trace_correlation_id
            if "trace_correlation_id" in locals()
            else request.trace_correlation_id
        )

        return RolloutResponse(
            trace_correlation_id=_trace_correlation_id,
            reward_info=RolloutMetrics(outcome_reward=0.0, details={"error": str(e)}),
        )


# ============================================================================
# Task App
# ============================================================================


def provide_taskset_description() -> dict:
    return {
        "name": "Pokemon TCG Deck Builder",
        "description": "Build Pokemon TCG decks satisfying constraints and winning battles",
        "metrics": ["constraint_score", "win_rate", "final_reward"],
        "total_instances": len(INSTANCE_IDS),
        "opponent_decks": list(OPPONENT_DECKS.keys()),
    }


def provide_task_instances(seeds: list[int]) -> list[TaskInfo]:
    instances = []
    for seed in seeds:
        idx = seed % len(INSTANCE_IDS)
        instance_id = INSTANCE_IDS[idx]
        instance = get_instance(instance_id)
        instances.append(
            TaskInfo(
                task={"id": "deckbuilder", "name": "Pokemon TCG Deck Builder"},
                dataset={"id": "ptcg-deckbuilder", "split": "train", "index": idx},
                inference={"tool": "deck_builder"},
                limits={"max_turns": 1},
                task_metadata={
                    "instance_id": instance_id,
                    "challenge_name": instance["name"] if instance else "",
                    "opponent_decks": instance.get("opponent_decks", []) if instance else [],
                },
            )
        )
    return instances


app = create_local_api(
    LocalAPIConfig(
        app_id="ptcg_deckbuilder",
        name="Pokemon TCG Deck Builder",
        description="Build Pokemon TCG decks satisfying constraints and winning battles",
        provide_taskset_description=provide_taskset_description,
        provide_task_instances=provide_task_instances,
        rollout=run_rollout,
        cors_origins=["*"],
    )
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8018)
