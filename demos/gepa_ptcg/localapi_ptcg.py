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

import httpx
from synth_ai.sdk.localapi import LocalAPIConfig, create_local_api
from synth_ai.sdk.task.contracts import (
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
    TaskInfo,
)

# ============================================================================
# Configuration
# ============================================================================

ENGINE_BENCH_REPO_URL = "https://github.com/JoshuaPurtell/engine-bench.git"
ENGINE_BENCH_DIR = Path(os.getenv("ENGINE_BENCH_DIR", str(Path.home() / ".cache" / "engine-bench")))

# Sample decks for testing
# These are Crystal Guardians cards that are fully implemented
SAMPLE_DECK_1 = (
    [
        "CG-001-Blaziken",
        "CG-002-Charizard",
    ]
    * 4
    + [
        "ENERGY-FIRE",
    ]
    * 20
    + [
        "CG-003-Combusken",
    ]
    * 32
)

SAMPLE_DECK_2 = (
    [
        "CG-004-Venusaur",
        "CG-005-Sceptile",
    ]
    * 4
    + [
        "ENERGY-GRASS",
    ]
    * 20
    + [
        "CG-006-Ivysaur",
    ]
    * 32
)

# System prompt for the Pokemon TCG agent
DEFAULT_SYSTEM_PROMPT = """You are an expert Pokemon TCG player. Your goal is to win the game by:
1. Taking all 6 of your opponent's prize cards by knocking out their Pokemon
2. Knocking out all of your opponent's Pokemon in play
3. Making your opponent unable to draw a card at the start of their turn

Strategy guidelines:
- Play Basic Pokemon to your bench early to have options
- Attach energy to power up attacks
- Evolve your Pokemon when possible for stronger attacks
- Attack when you can deal significant damage
- Consider type matchups (weaknesses deal double damage)

You must respond with a JSON action. Valid actions include:
- {"action": "PlayBasic", "card_id": <id>} - Play a basic Pokemon from hand
- {"action": "AttachEnergy", "energy_id": <id>, "target_id": <id>} - Attach energy to a Pokemon
- {"action": "EvolveFromHand", "card_id": <id>, "target_id": <id>} - Evolve a Pokemon
- {"action": "DeclareAttack", "attack": "<name>"} - Use an attack
- {"action": "EndTurn"} - End your turn
- {"action": "Retreat", "card_id": <id>} - Retreat active Pokemon

Always analyze the game state carefully before deciding on an action.
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
    api_key: str,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.3,
    max_tokens: int = 512,
) -> str:
    """Call the LLM through the Synth interceptor."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            inference_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
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

    steps = 0
    errors = 0
    max_errors = 10

    while not game.is_game_over() and steps < max_steps:
        # Get observation
        obs = game.run_until_agent_turn()

        if game.is_game_over():
            break

        # Call LLM
        try:
            action_json = await call_llm(
                system_prompt=system_prompt,
                user_prompt=obs.game_state,
                inference_url=inference_url,
                api_key=api_key,
                model=model,
            )

            # Submit action
            try:
                game.submit_action(action_json)
            except Exception:
                errors += 1
                if errors >= max_errors:
                    break
                # Fallback to EndTurn
                game.submit_action('{"action": "EndTurn"}')

        except Exception:
            errors += 1
            if errors >= max_errors:
                break
            # On LLM error, try EndTurn
            with contextlib.suppress(Exception):
                game.submit_action('{"action": "EndTurn"}')

        # Step the game
        game.step()
        steps += 1

    # Get result
    result = game.get_result()
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


async def run_rollout(request: RolloutRequest) -> RolloutResponse:
    """Execute a single rollout (game) for GEPA evaluation."""
    env_config = request.env.config or {}
    policy_config = request.policy.config or {}

    instance_id = env_config.get("instance_id", INSTANCE_IDS[0])
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


# Create the LocalAPI app
app = create_local_api(
    LocalAPIConfig(
        app_id="ptcg",
        name="Pokemon TCG GEPA Demo",
        description="Train an LLM to play Pokemon TCG against AI v4",
        provide_taskset_description=provide_taskset_description,
        provide_task_instances=provide_task_instances,
        rollout=run_rollout,
        cors_origins=["*"],
    )
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8017)
