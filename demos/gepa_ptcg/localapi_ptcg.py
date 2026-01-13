"""
GEPA demo for Pokemon TCG game playing.

This demo trains an LLM agent to play Pokemon TCG against the AI v4 opponent
using GEPA (Generalized Evolutionary Prompt Algorithm) to optimize the system prompt.

The agent receives game observations and must output JSON actions.
Win rate against AI v4 is the reward signal.
"""

import contextlib
import os
import subprocess
from pathlib import Path
from typing import Any

import httpx
from synth_ai.sdk.localapi.server import LocalAPIConfig, RubricBundle, create_local_api
from synth_ai.sdk.task.contracts import (
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
    TaskInfo,
)
from synth_ai.sdk.task.rubrics import Criterion, Rubric
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

# Sample decks for testing - Dragon Frontiers cards
SAMPLE_DECK_1 = (
    # Pokemon (18)
    ["df-061-ralts"] * 4  # Basic for Gardevoir line
    + ["df-033-kirlia"] * 3  # Stage 1
    + ["df-093-gardevoir-ex"] * 2  # Main attacker
    + ["df-017-jynx"] * 2  # Stages of Evolution body
    + ["df-070-vulpix"] * 3  # Basic for Ninetales
    + ["df-008-ninetales"] * 2  # Volunteer power
    + ["df-010-snorlax"] * 2  # Dozing heal
    # Trainers (10)
    + ["df-082-tv-reporter"] * 4  # Draw support
    + ["df-079-prof-elms-training"] * 4  # Evolution search
    + ["df-072-buffer-piece"] * 2  # Damage reduction
    # Energy (32)
    + ["ENERGY-PSYCHIC"] * 16
    + ["ENERGY-FIRE"] * 16
)

SAMPLE_DECK_2 = (
    # Pokemon (14)
    ["df-068-trapinch"] * 4  # Basic for Flygon line
    + ["df-024-vibrava"] * 3  # Stage 1
    + ["df-092-flygon-ex"] * 2  # Main attacker
    + ["df-009-pinsir"] * 3  # Armor body tank
    + ["df-003-heracross"] * 2  # Shining Horn body
    # Trainers (9)
    + ["df-082-tv-reporter"] * 4  # Draw support
    + ["df-079-prof-elms-training"] * 3  # Evolution search
    + ["df-072-buffer-piece"] * 2  # Damage reduction
    # Energy (37)
    + ["ENERGY-PSYCHIC"] * 20
    + ["ENERGY-GRASS"] * 17
)

# System prompt for the Pokemon TCG agent
DEFAULT_SYSTEM_PROMPT = """You are an expert Pokemon TCG player. Your goal is to win by knocking out opponent's Pokemon to take prize cards.

IMPORTANT RULES:
- You can only have 5 Pokemon on your bench. Do NOT try PlayBasic if bench is full!
- You can only attach ONE energy per turn.
- After your actions, you MUST use EndTurn.
- Check available_actions to see what you can do!
- You MUST choose an action that is currently allowed (from available_actions). If you output an illegal action, you lose immediately.

You must respond with ONLY a JSON action:
- {"action": "PlayBasic", "card_id": <id>} - Play basic to bench
- {"action": "AttachEnergy", "energy_id": <id>, "target_id": <id>} - Attach energy
- {"action": "DeclareAttack", "attack": "<name>"} - Attack with active Pokemon
- {"action": "EndTurn"} - End your turn
- {"action": "ChooseActive", "card_id": <id>} - Setup: choose active
- {"action": "ChooseBench", "card_ids": [<id>, ...]} - Setup: choose bench

Respond with ONLY the JSON action, no explanation.
"""

# ============================================================================
# Engine Setup
# ============================================================================


def ensure_engine_bench_repo() -> None:
    """Clone or update engine-bench repo if needed."""
    if not ENGINE_BENCH_DIR.exists():
        print(f"[ptcg] Cloning engine-bench to {ENGINE_BENCH_DIR}...")
        subprocess.run(
            ["git", "clone", "--depth", "1", ENGINE_BENCH_REPO_URL, str(ENGINE_BENCH_DIR)],
            check=True,
        )
    else:
        # Pull latest
        subprocess.run(
            ["git", "-C", str(ENGINE_BENCH_DIR), "pull", "--ff-only"],
            check=False,
            capture_output=True,
        )


def ensure_tcg_py_built() -> None:
    """Ensure the tcg_py Python extension is built."""
    tcg_py_dir = ENGINE_BENCH_DIR / "tcg_py"
    if not tcg_py_dir.exists():
        raise RuntimeError(f"tcg_py directory not found at {tcg_py_dir}")

    # Check if already importable
    try:
        import tcg_py  # noqa: F401

        return
    except ImportError:
        pass

    # Build with maturin
    print("[ptcg] Building tcg_py extension...")
    subprocess.run(
        ["maturin", "develop"],
        cwd=str(tcg_py_dir),
        check=True,
    )


# ============================================================================
# Task Instances
# ============================================================================

# Each "instance" is a game configuration (decks + seed)
GAME_INSTANCES = [
    {
        "id": f"ptcg-game-{i:03d}",
        "p1_deck": SAMPLE_DECK_1,
        "p2_deck": SAMPLE_DECK_2,
        "game_seed": 1000 + i,
        "ai_seed": 2000 + i,
    }
    for i in range(100)
]


def load_instance_ids() -> list[str]:
    """Load available instance IDs."""
    return [inst["id"] for inst in GAME_INSTANCES]


def get_instance(instance_id: str) -> dict | None:
    """Get instance by ID."""
    for inst in GAME_INSTANCES:
        if inst["id"] == instance_id:
            return inst
    return None


# Initialize
ensure_engine_bench_repo()

INSTANCE_IDS = load_instance_ids()
print(f"[ptcg] Loaded {len(INSTANCE_IDS)} game instances")


# ============================================================================
# LLM Caller
# ============================================================================


async def call_llm(
    system_prompt: str,
    user_prompt: str,
    inference_url: str,
    api_key: str | None,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.3,
    max_tokens: int = 512,
) -> str:
    """Call the LLM through the Synth interceptor."""
    inference_url = normalize_inference_url(inference_url)
    headers = {"Content-Type": "application/json"}
    if api_key:
        # Interceptor primarily uses X-API-Key; keep Authorization for compatibility.
        headers["X-API-Key"] = api_key
        headers["Authorization"] = f"Bearer {api_key}"
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            inference_url,
            headers=headers,
            json={
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
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
# Game Runner
# ============================================================================


async def run_game(
    instance: dict,
    system_prompt: str,
    inference_url: str,
    api_key: str,
    model: str = "gpt-4.1-mini",
    max_steps: int = 500,
) -> dict:
    """Run a single game and return the result."""
    import tcg_py

    # Create game
    game = tcg_py.PtcgGame(
        p1_deck=instance["p1_deck"],
        p2_deck=instance["p2_deck"],
        game_seed=instance["game_seed"],
        ai_seed=instance["ai_seed"],
        max_steps=max_steps,
    )

    # NOTE: tcg_py advances its own internal step counter inside run_until_agent_turn().
    # Keep a separate counter here for "decision steps" (i.e., how many times we asked the LLM).
    decision_steps = 0
    errors = 0

    while not game.is_game_over():
        # Get observation - handles AI turns automatically
        obs = game.run_until_agent_turn()

        if game.is_game_over():
            final = game.get_result()
            print(
                f"[ptcg] Game over (decision_steps={decision_steps}, game_steps={final.steps}, "
                f"end_reason={final.end_reason}, winner={final.winner})"
            )
            break

        # Log current state
        bench = getattr(obs, "my_bench_count", "?")
        # Pull internal game steps for debugging "winner=None" cases (often MaxSteps vs genuine loss).
        game_steps = getattr(game.get_result(), "steps", "?")
        print(
            f"[ptcg] Decision {decision_steps}: player={obs.current_player}, phase={obs.phase}, "
            f"bench={bench}, prizes=({obs.my_prizes},{obs.opp_prizes}), actions={obs.available_actions}, "
            f"game_steps={game_steps}"
        )

        # Skip if no actions and no prompt
        if not obs.available_actions and not obs.has_prompt:
            print("[ptcg] No actions, stepping...")
            game.step()
            decision_steps += 1
            continue

        # Auto-end turn if only EndTurn available
        if obs.available_actions == ["EndTurn"]:
            print("[ptcg] Only EndTurn, auto-ending")
            with contextlib.suppress(Exception):
                game.submit_action('{"action": "EndTurn"}')
            game.step()
            decision_steps += 1
            continue

        # Call LLM (fail-fast: invalid/unparseable action => immediate loss)
        try:
            allowed = ", ".join(obs.available_actions or [])
            user_prompt = f"{obs.game_state}\n\navailable_actions: [{allowed}]\n"
            action_json = await call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                inference_url=inference_url,
                api_key=api_key,
                model=model,
            )
            print(f"[ptcg] LLM: {action_json[:100]}...")

            # Submit action
            try:
                game.submit_action(action_json)
                print("[ptcg] Action OK")
            except Exception as e:
                print(f"[ptcg] Action FAILED: {e}")
                errors += 1
                break

        except Exception as e:
            print(f"[ptcg] LLM error: {e}")
            errors += 1
            break

        # Step the game
        game.step()
        decision_steps += 1

    # Get result
    result = game.get_result()
    # If we bailed due to errors, treat as a loss for P1.
    if errors:
        return {
            "winner": "P2",
            "turns": result.turns,
            "steps": result.steps,
            "p1_prizes": result.p1_prizes_remaining,
            "p2_prizes": result.p2_prizes_remaining,
            "end_reason": f"llm_error_or_invalid_action (errors={errors})",
            "errors": errors,
        }
    return {
        "winner": result.winner,
        "turns": result.turns,
        "steps": result.steps,
        "p1_prizes": result.p1_prizes_remaining,
        "p2_prizes": result.p2_prizes_remaining,
        "end_reason": result.end_reason,
        "errors": errors,
    }


# ============================================================================
# Rollout Handler
# ============================================================================


async def run_rollout(request: RolloutRequest, fastapi_request: Any) -> RolloutResponse:
    """Execute a single rollout (game) for GEPA evaluation."""
    env_config = request.env.config or {}
    policy_config = request.policy.config or {}

    seed = request.env.seed or 0
    instance_id = env_config.get("instance_id") or INSTANCE_IDS[seed % len(INSTANCE_IDS)]
    instance = get_instance(instance_id)

    if not instance:
        return RolloutResponse(
            trace_correlation_id=request.trace_correlation_id,
            metrics=RolloutMetrics(
                outcome_reward=0.0,
                details={"error": f"Instance not found: {instance_id}"},
            ),
        )

    # Get config from policy
    system_prompt = policy_config.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    model = policy_config.get("model", "gpt-4.1-mini")
    inference_url = policy_config.get("inference_url")
    api_key = policy_config.get("api_key")
    max_steps = int(policy_config.get("max_steps", 500))

    if not inference_url:
        return RolloutResponse(
            trace_correlation_id=request.trace_correlation_id,
            metrics=RolloutMetrics(
                outcome_reward=0.0,
                details={"error": "No inference_url provided"},
            ),
        )

    print(f"\n{'=' * 60}")
    print(f"[ptcg] Running game {instance_id}")
    print(f"  Model: {model}")
    print(f"  Interceptor: {inference_url[:50]}...")
    print(f"{'=' * 60}\n")

    try:
        result = await run_game(
            instance=instance,
            system_prompt=system_prompt,
            inference_url=inference_url,
            api_key=api_key,
            model=model,
            max_steps=max_steps,
        )

        # Calculate reward: 1.0 for win, 0.0 for loss, 0.5 for draw/max_steps
        if result["winner"] == "P1":
            reward = 1.0
        elif result["winner"] == "P2":
            reward = 0.0
        else:
            # Draw or timeout - partial reward based on prize differential
            p1_prizes = result["p1_prizes"]
            p2_prizes = result["p2_prizes"]
            if p1_prizes < p2_prizes:
                reward = 0.6  # P1 was winning
            elif p1_prizes > p2_prizes:
                reward = 0.4  # P1 was losing
            else:
                reward = 0.5  # Even

        print(f"\n{'=' * 60}")
        print(f"[ptcg] Game {instance_id} completed")
        print(f"  Winner: {result['winner']}")
        print(f"  Turns: {result['turns']}")
        print(f"  Prizes: P1={result['p1_prizes']}, P2={result['p2_prizes']}")
        print(f"  Reward: {reward}")
        print(f"{'=' * 60}\n")

        return RolloutResponse(
            trace_correlation_id=request.trace_correlation_id,
            metrics=RolloutMetrics(
                outcome_reward=reward,
                details={
                    "instance_id": instance_id,
                    "winner": result["winner"],
                    "turns": result["turns"],
                    "steps": result["steps"],
                    "p1_prizes": result["p1_prizes"],
                    "p2_prizes": result["p2_prizes"],
                    "end_reason": result["end_reason"],
                    "errors": result["errors"],
                },
            ),
            metadata={
                "instance_id": instance_id,
                "winner": result["winner"],
                "reward": reward,
            },
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return RolloutResponse(
            trace_correlation_id=request.trace_correlation_id,
            metrics=RolloutMetrics(
                outcome_reward=0.0,
                details={"error": str(e)},
            ),
        )


# ============================================================================
# Task App
# ============================================================================


def provide_taskset_description() -> dict:
    """Provide task set metadata."""
    return {
        "name": "Pokemon TCG vs AI v4",
        "description": "Play Pokemon TCG games against the deterministic AI v4 opponent",
        "metrics": ["win_rate"],
        "total_instances": len(INSTANCE_IDS),
    }


def provide_task_instances(seeds: list[int]) -> list[TaskInfo]:
    """Provide task instances for given seeds."""
    instances = []
    for seed in seeds:
        idx = seed % len(INSTANCE_IDS)
        instance_id = INSTANCE_IDS[idx]

        instances.append(
            TaskInfo(
                task={"id": "ptcg", "name": "Pokemon TCG"},
                dataset={"id": "ptcg-games", "split": "train", "index": idx},
                inference={"tool": "ptcg_action"},
                limits={"max_turns": 500},
                task_metadata={"instance_id": instance_id},
            )
        )

    return instances


# A general gameplay-quality rubric bundle for zero-shot verifier evaluation.
PTCG_GAMEPLAY_RUBRICS = RubricBundle(
    outcome=Rubric(
        version="1.0",
        goal_text="Evaluate Pokemon TCG gameplay quality at the end of the rollout",
        criteria=[
            Criterion(
                id="win_or_strong_advantage",
                description=(
                    "Did the agent win? If not, did it create a clear advantage (e.g., prize lead, board control) "
                    "by the end of the rollout?"
                ),
                weight=1.0,
                required=False,
            ),
            Criterion(
                id="avoid_blunders",
                description=(
                    "Avoid obvious blunders and self-sabotage (illegal actions, wasting turns, repeatedly missing "
                    "attacks when available)."
                ),
                weight=1.0,
                required=True,
            ),
        ],
        aggregation="weighted_sum",
    ),
    events=Rubric(
        version="1.0",
        goal_text="Evaluate action quality during the rollout",
        criteria=[
            Criterion(
                id="legality_and_prompt_following",
                description="Actions are legal and follow the current available_actions requirements.",
                weight=2.0,
                required=True,
            ),
            Criterion(
                id="progress_turn_economy",
                description="Makes progress (attach energy, evolve, attack) and avoids stalling.",
                weight=1.0,
                required=False,
            ),
            Criterion(
                id="attack_when_good",
                description="Attacks when a reasonable attack is available instead of passing unnecessarily.",
                weight=1.0,
                required=False,
            ),
        ],
        aggregation="weighted_sum",
    ),
)


# Create the LocalAPI app
app = create_local_api(
    LocalAPIConfig(
        app_id="ptcg",
        name="Pokemon TCG GEPA Demo",
        description="Train an LLM to play Pokemon TCG against AI v4",
        provide_taskset_description=provide_taskset_description,
        provide_task_instances=provide_task_instances,
        rollout=run_rollout,
        rubrics=PTCG_GAMEPLAY_RUBRICS,
        cors_origins=["*"],
    )
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8017)
