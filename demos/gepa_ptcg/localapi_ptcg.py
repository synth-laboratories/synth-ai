"""
GEPA demo for Pokemon TCG game playing.

This demo trains an LLM agent to play Pokemon TCG against the AI v4 opponent
using GEPA (Generalized Evolutionary Prompt Algorithm) to optimize the system prompt.

The agent receives game observations and must output JSON actions.
Win rate against AI v4 is the reward signal.
"""

import contextlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from synth_ai.sdk.localapi import LocalAPIConfig, create_local_api
from synth_ai.sdk.task.server import RubricBundle
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

PTCG_TRACE_DIR_ENV = "PTCG_TRACE_DIR"

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

# ReAct-style variant used by prompt optimization demos. The environment still requires the
# final response to be ONLY a JSON action; any intermediate reasoning must remain internal.
PTCG_REACT_SYSTEM_PROMPT = """You are an expert Pokemon TCG player. Your goal is to win by knocking out opponent's Pokemon to take prize cards.

You MUST select only legal actions based on the current available_actions list.

Guidance (internal):
- Consider the phase, available attacks, energy attachment limits, and bench capacity.
- Prefer clear progress: develop the board, attach energy with intent, and attack when it is reasonable.
- Avoid illegal actions, wasting turns, or repeatedly passing when attacking is available.

Hard rules:
- You can only have 5 Pokemon on your bench. Do NOT try PlayBasic if bench is full.
- You can only attach ONE energy per turn.
- After your actions, you MUST use EndTurn.
- You MUST choose an action that is currently allowed (from available_actions). If you output an illegal action, you lose immediately.

You must respond with ONLY a JSON action (no extra text):
- {"action": "PlayBasic", "card_id": <id>} - Play basic to bench
- {"action": "AttachEnergy", "energy_id": <id>, "target_id": <id>} - Attach energy
- {"action": "DeclareAttack", "attack": "<name>"} - Attack with active Pokemon
- {"action": "EndTurn"} - End your turn
- {"action": "ChooseActive", "card_id": <id>} - Setup: choose active
- {"action": "ChooseBench", "card_ids": [<id>, ...]} - Setup: choose bench
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

    # If engine-bench is on sys.path (common after editable installs), it can shadow the compiled
    # extension with a namespace package. Remove it before checking/importing.
    engine_bench_path = str(ENGINE_BENCH_DIR)
    if engine_bench_path in sys.path:
        sys.path.remove(engine_bench_path)

    # Check if already importable and exposes the API we need.
    try:
        import tcg_py  # noqa: F401

        if hasattr(tcg_py, "PtcgGame"):
            return
    except ImportError:
        pass

    # Build with maturin into the *current* interpreter environment.
    # (This task app runs in-process, so tcg_py must be importable from sys.executable.)
    print("[ptcg] Building tcg_py extension (this can take a few minutes the first time)...")
    try:
        # Ensure maturin is installed for this interpreter.
        subprocess.run(
            ["uv", "pip", "install", "--python", sys.executable, "maturin"],
            check=True,
            capture_output=True,
        )
    except Exception:
        # If maturin install fails, the next command will raise with a clear error.
        pass

    # Remove any prior editable install that can shadow the compiled extension with a namespace package.
    try:
        subprocess.run(
            ["uv", "pip", "uninstall", "--python", sys.executable, "-y", "tcg_py"],
            check=False,
            capture_output=True,
        )
    except Exception:
        pass

    # Build a wheel and install it. `maturin develop` can produce an editable install that
    # doesn't reliably expose the compiled extension module for in-process imports.
    subprocess.run(
        [sys.executable, "-m", "maturin", "build", "--release", "-i", sys.executable],
        cwd=str(tcg_py_dir),
        check=True,
    )

    wheels_dir = tcg_py_dir / "target" / "wheels"
    wheels = sorted(wheels_dir.glob("tcg_py-*.whl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not wheels:
        raise RuntimeError(f"No wheels found under {wheels_dir} after build")
    wheel = wheels[0]

    subprocess.run(
        ["uv", "pip", "install", "--python", sys.executable, "--force-reinstall", str(wheel)],
        check=True,
        capture_output=True,
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
ensure_tcg_py_built()

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
    collect_trace: bool = False,
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
    trace_steps: list[dict[str, Any]] = []

    def _record_step(step: dict[str, Any]) -> None:
        if collect_trace:
            trace_steps.append(step)

    def _parse_action(action_json: str) -> tuple[dict[str, Any] | None, str | None, str | None]:
        try:
            parsed = json.loads(action_json)
        except Exception as exc:
            return None, None, f"json_parse_error: {exc}"
        action_type = parsed.get("action") if isinstance(parsed, dict) else None
        return parsed if isinstance(parsed, dict) else None, action_type, None

    trace_steps: list[dict[str, Any]] = []

    def _truncate(text: str | None, limit: int = 20_000) -> str | None:
        if text is None:
            return None
        if len(text) <= limit:
            return text
        return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"

    def _parse_action_type(action_json: str) -> str | None:
        try:
            parsed = json.loads(action_json)
        except Exception:
            return None
        if not isinstance(parsed, dict):
            return None
        action_val = parsed.get("action")
        return str(action_val) if isinstance(action_val, str) else None

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
        game_state = str(obs.game_state)
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
            trace_steps.append(
                {
                    "decision_step": decision_steps,
                    "available_actions": list(obs.available_actions or []),
                    "action_type": None,
                    "action_raw": None,
                    "action_valid": True,
                    "note": "no_actions_step",
                    "game_state": _truncate(getattr(obs, "game_state", None)),
                }
            )
            game.step()
            decision_steps += 1
            continue

        # Auto-end turn if only EndTurn available
        if obs.available_actions == ["EndTurn"]:
            print("[ptcg] Only EndTurn, auto-ending")
            trace_steps.append(
                {
                    "decision_step": decision_steps,
                    "available_actions": ["EndTurn"],
                    "action_type": "EndTurn",
                    "action_raw": '{"action": "EndTurn"}',
                    "action_valid": True,
                    "note": "auto_end_turn",
                    "game_state": _truncate(getattr(obs, "game_state", None)),
                }
            )
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
                parsed_action, action_type, parse_error = _parse_action(action_json)
                game.submit_action(action_json)
                print("[ptcg] Action OK")
                trace_steps.append(
                    {
                        "decision_step": decision_steps,
                        "available_actions": list(obs.available_actions or []),
                        "action_type": _parse_action_type(action_json),
                        "action_raw": action_json,
                        "action_valid": True,
                        "note": "llm_action",
                        "game_state": _truncate(getattr(obs, "game_state", None)),
                    }
                )
            except Exception as e:
                print(f"[ptcg] Action FAILED: {e}")
                trace_steps.append(
                    {
                        "decision_step": decision_steps,
                        "available_actions": list(obs.available_actions or []),
                        "action_type": _parse_action_type(action_json),
                        "action_raw": action_json,
                        "action_valid": False,
                        "error": str(e),
                        "note": "invalid_action",
                        "game_state": _truncate(getattr(obs, "game_state", None)),
                    }
                )
                errors += 1
                _record_step(
                    {
                        "decision_step": decision_steps,
                        "phase": str(obs.phase),
                        "current_player": obs.current_player,
                        "available_actions": list(obs.available_actions or []),
                        "action_type": "InvalidAction",
                        "action": {"raw": action_json},
                        "action_valid": False,
                        "auto_action": False,
                        "error": str(e),
                        "game_state": game_state,
                    }
                )
                break

        except Exception as e:
            print(f"[ptcg] LLM error: {e}")
            trace_steps.append(
                {
                    "decision_step": decision_steps,
                    "available_actions": list(obs.available_actions or []),
                    "action_type": None,
                    "action_raw": None,
                    "action_valid": False,
                    "error": str(e),
                    "note": "llm_error",
                    "game_state": _truncate(getattr(obs, "game_state", None)),
                }
            )
            errors += 1
            _record_step(
                {
                    "decision_step": decision_steps,
                    "phase": str(obs.phase),
                    "current_player": obs.current_player,
                    "available_actions": list(obs.available_actions or []),
                    "action_type": "LLMError",
                    "action": None,
                    "action_valid": False,
                    "auto_action": False,
                    "error": str(e),
                    "game_state": game_state,
                }
            )
            break

        # Step the game
        game.step()
        decision_steps += 1

    # Get result
    result = game.get_result()
    # If we bailed due to errors, treat as a loss for P1.
    if errors:
        result_payload = {
            "winner": "P2",
            "turns": result.turns,
            "steps": result.steps,
            "decision_steps": decision_steps,
            "p1_prizes": result.p1_prizes_remaining,
            "p2_prizes": result.p2_prizes_remaining,
            "end_reason": f"llm_error_or_invalid_action (errors={errors})",
            "errors": errors,
            "trace_steps": trace_steps,
        }
        if collect_trace:
            result_payload["trace_steps"] = trace_steps
            result_payload["decision_steps"] = decision_steps
        return result_payload
    result_payload = {
        "winner": result.winner,
        "turns": result.turns,
        "steps": result.steps,
        "decision_steps": decision_steps,
        "p1_prizes": result.p1_prizes_remaining,
        "p2_prizes": result.p2_prizes_remaining,
        "end_reason": result.end_reason,
        "errors": errors,
        "trace_steps": trace_steps,
    }
    if collect_trace:
        result_payload["trace_steps"] = trace_steps
        result_payload["decision_steps"] = decision_steps
    return result_payload


def compute_outcome_reward(result: dict[str, Any]) -> float:
    """Score the rollout outcome for P1: win=1.0, loss=0.0, draws use prize differential."""
    if result.get("winner") == "P1":
        return 1.0
    if result.get("winner") == "P2":
        return 0.0
    p1_prizes = result.get("p1_prizes", 0)
    p2_prizes = result.get("p2_prizes", 0)
    if p1_prizes < p2_prizes:
        return 0.6
    if p1_prizes > p2_prizes:
        return 0.4
    return 0.5


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
            reward_info=RolloutMetrics(
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
            reward_info=RolloutMetrics(
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
        trace_dir_value = os.getenv("PTCG_TRACE_DIR", "").strip()
        trace_dir = Path(trace_dir_value).expanduser() if trace_dir_value else None
        collect_trace = trace_dir is not None
        result = await run_game(
            instance=instance,
            system_prompt=system_prompt,
            inference_url=inference_url,
            api_key=api_key,
            model=model,
            max_steps=max_steps,
            collect_trace=collect_trace,
        )

        trace_steps = result.get("trace_steps", []) or []

        # Calculate reward: 1.0 for win, 0.0 for loss, 0.5 for draw/max_steps
        reward = compute_outcome_reward(result)

        print(f"\n{'=' * 60}")
        print(f"[ptcg] Game {instance_id} completed")
        print(f"  Winner: {result['winner']}")
        print(f"  Turns: {result['turns']}")
        print(f"  Prizes: P1={result['p1_prizes']}, P2={result['p2_prizes']}")
        print(f"  Reward: {reward}")
        print(f"{'=' * 60}\n")

        trace_dir_raw = os.getenv(PTCG_TRACE_DIR_ENV, "").strip()
        if trace_dir_raw:
            out_dir = Path(trace_dir_raw).expanduser()
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "ptcg_rollouts.jsonl"
            record = {
                "trace_id": request.trace_correlation_id,
                "seed": seed,
                "instance_id": instance_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "outcome_reward": reward,
                "result": {k: v for k, v in result.items() if k != "trace_steps"},
                "trace_steps": trace_steps,
                "policy": {"model": model},
            }
            with out_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")

        return RolloutResponse(
            trace_correlation_id=request.trace_correlation_id,
            reward_info=RolloutMetrics(
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
                    "trace_steps": trace_steps,
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
            reward_info=RolloutMetrics(
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
        goal_text="Evaluate Pokemon TCG gameplay quality and strategic play at the end of the rollout",
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
                id="prize_plan_and_tempo",
                description=(
                    "Demonstrates a coherent prize plan and tempo: prioritizes taking prizes, avoids low-impact lines "
                    "when meaningful progress (damage/prizes/board improvement) is available."
                ),
                weight=1.0,
                required=False,
            ),
            Criterion(
                id="resource_management",
                description=(
                    "Manages resources well: attaches energy with intent, avoids wasting limited resources, and makes "
                    "reasonable retreat/switch choices relative to board state."
                ),
                weight=1.0,
                required=False,
            ),
            Criterion(
                id="board_development",
                description=(
                    "Develops the board: benches basics when appropriate, evolves when possible, and maintains an "
                    "attacker pipeline rather than stalling with an empty/fragile board."
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
        goal_text="Evaluate action quality during the rollout (basic best practices)",
        criteria=[
            Criterion(
                id="legality_and_prompt_following",
                description="Actions are legal and follow the current available_actions requirements.",
                weight=2.0,
                required=True,
            ),
            Criterion(
                id="energy_attachment_discipline",
                description=(
                    "Attaches energy most turns when it enables near-term attacks or improves future tempo; avoids "
                    "obviously wasted attachments."
                ),
                weight=1.0,
                required=False,
            ),
            Criterion(
                id="evolution_when_available",
                description="Evolves Pokémon when it improves survivability/damage output and is available.",
                weight=1.0,
                required=False,
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
            Criterion(
                id="target_selection",
                description=(
                    "Selects reasonable targets: prefers taking prizes or threatening KOs over low-value attacks when "
                    "choices exist."
                ),
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
