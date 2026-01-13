"""
LocalAPI Task App - EngineBench Pokemon TCG Card Implementation

This file creates a task app that Synth AI uses to evaluate coding agents
on Pokemon TCG card implementations. The backend calls your /rollout endpoint
with different seeds (instance IDs) and aggregates the scores.

EngineBench evaluates an agent's ability to:
1. Understand a domain-specific game engine architecture
2. Implement complex game mechanics from card text descriptions
3. Follow established patterns and compose with existing code
4. Produce code that passes deterministic cargo tests
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import Request
from synth_ai.data.artifacts import Artifact
from synth_ai.data.enums import SuccessStatus
from synth_ai.sdk.localapi import LocalAPIConfig, create_local_api
from synth_ai.sdk.task.contracts import (
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
    TaskInfo,
)

# =============================================================================
# APP CONFIGURATION
# =============================================================================

APP_ID = "engine_bench"
APP_NAME = "EngineBench - Pokemon TCG Card Implementation"

# GitHub repo URL for engine-bench
ENGINE_BENCH_REPO_URL = "https://github.com/JoshuaPurtell/engine-bench.git"

# Path to engine-bench repo (will be cloned if not present)
ENGINE_BENCH_DIR = Path(os.getenv("ENGINE_BENCH_DIR", str(Path.home() / ".cache" / "engine-bench")))

DATA_DIR = ENGINE_BENCH_DIR / "data"
GOLD_DIR = ENGINE_BENCH_DIR / "gold"
SCAFFOLD_DIR = ENGINE_BENCH_DIR / "scaffold"


def ensure_engine_bench_repo() -> None:
    """Clone or update engine-bench repo if needed."""
    if not ENGINE_BENCH_DIR.exists():
        print(f"[engine_bench] Cloning engine-bench to {ENGINE_BENCH_DIR}...")
        subprocess.run(
            ["git", "clone", "--depth", "1", ENGINE_BENCH_REPO_URL, str(ENGINE_BENCH_DIR)],
            check=True,
        )
    elif not (ENGINE_BENCH_DIR / ".git").exists():
        raise RuntimeError(f"ENGINE_BENCH_DIR exists but is not a git repo: {ENGINE_BENCH_DIR}")


# Ensure repo is available at module load
ensure_engine_bench_repo()


# =============================================================================
# INSTANCE LOADING
# =============================================================================


def load_instance_ids() -> list[str]:
    """Load available instance IDs from data directory."""
    instances_dir = DATA_DIR / "instances" / "single"
    if not instances_dir.exists():
        return []
    return sorted([p.stem for p in instances_dir.glob("*.json")])


def load_instance(instance_id: str) -> dict[str, Any]:
    """Load instance specification."""
    instance_path = DATA_DIR / "instances" / "single" / f"{instance_id}.json"
    if not instance_path.exists():
        raise ValueError(f"Instance not found: {instance_id}")
    return json.loads(instance_path.read_text())


# Cache instance IDs at module load
INSTANCE_IDS = load_instance_ids()
print(f"[engine_bench] Loaded {len(INSTANCE_IDS)} instances")


def get_instance_by_seed(seed: int) -> str:
    """Get instance ID by seed (modulo number of instances)."""
    if not INSTANCE_IDS:
        raise ValueError("No instances available")
    return INSTANCE_IDS[seed % len(INSTANCE_IDS)]


# =============================================================================
# SANDBOX SETUP
# =============================================================================


async def setup_sandbox(instance_id: str, work_dir: Path) -> Path:
    """Set up a sandbox for the coding agent using the scaffold from engine-bench."""
    sandbox_dir = work_dir / "tcg_expansions"

    if not SCAFFOLD_DIR.exists():
        raise RuntimeError(f"Scaffold not found: {SCAFFOLD_DIR}")

    # Copy the scaffold (which uses crates.io dependencies)
    await asyncio.to_thread(
        shutil.copytree,
        SCAFFOLD_DIR,
        sandbox_dir,
        symlinks=True,
        ignore=shutil.ignore_patterns(".git", "target", "*.pyc", "__pycache__"),
    )

    # Load instance to get card_file path
    instance = load_instance(instance_id)
    card_file = instance.get("card_file", "")

    if card_file:
        # Use canonical stub from gold/stubs/
        stub_file = GOLD_DIR / "stubs" / f"{instance_id.replace('-', '_')}.rs"
        # Card file path is relative to tcg_expansions, e.g., "tcg_expansions/src/df/cards/df_010_snorlax.rs"
        # In the scaffold, the root IS tcg_expansions, so strip the prefix
        relative_card_file = card_file.replace("tcg_expansions/", "")
        stub_path = sandbox_dir / relative_card_file

        if stub_file.exists():
            stub_path.parent.mkdir(parents=True, exist_ok=True)
            stub_path.write_text(stub_file.read_text())

            # Setup expansion module structure if needed (e.g., for HP cards)
            expansion = instance_id.split("-")[0]
            expansion_dir = sandbox_dir / "src" / expansion

            if expansion not in ["df", "cg"]:  # DF and CG already have full structure
                card_module = instance_id.replace("-", "_")

                # Create/update cards/mod.rs
                cards_mod_path = expansion_dir / "cards" / "mod.rs"
                if cards_mod_path.exists():
                    content = cards_mod_path.read_text()
                    if f"pub mod {card_module};" not in content:
                        cards_mod_path.write_text(content + f"\npub mod {card_module};")
                else:
                    cards_mod_path.parent.mkdir(parents=True, exist_ok=True)
                    cards_mod_path.write_text(f"pub mod {card_module};\n")

                # Create expansion/mod.rs if needed
                mod_path = expansion_dir / "mod.rs"
                if not mod_path.exists():
                    expansion_dir.mkdir(parents=True, exist_ok=True)
                    mod_path.write_text(
                        "pub mod cards;\npub mod runtime;\nmod import_specs;\npub use import_specs::{attack_effect_ast, power_effect_ast, trainer_effect_ast};\n"
                    )
                elif "pub mod cards;" not in mod_path.read_text():
                    mod_path.write_text(mod_path.read_text() + "\npub mod cards;")

                # Update lib.rs if needed
                lib_path = sandbox_dir / "src" / "lib.rs"
                if lib_path.exists():
                    lib_content = lib_path.read_text()
                    if f"pub mod {expansion};" not in lib_content:
                        lib_path.write_text(lib_content + f"\npub mod {expansion};\n")

    return sandbox_dir


# =============================================================================
# EVALUATION
# =============================================================================


async def run_cargo_check(repo_dir: Path) -> tuple[bool, str]:
    """Run cargo check and return (success, error_output)."""
    proc = await asyncio.create_subprocess_exec(
        "cargo",
        "check",
        cwd=str(repo_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    success = proc.returncode == 0
    output = stderr.decode("utf-8", errors="replace")
    return success, output if not success else ""


async def inject_eval_tests(sandbox_dir: Path, instance_id: str) -> bool:
    """Inject evaluation tests into the sandbox."""
    eval_test_file = GOLD_DIR / "tests" / f"{instance_id.replace('-', '_')}_eval.rs"
    if not eval_test_file.exists():
        return False

    eval_tests = eval_test_file.read_text()
    expansion = instance_id.split("-")[0]
    # sandbox_dir IS tcg_expansions, so path is src/{expansion}/cards/{card}.rs
    card_file = sandbox_dir / "src" / expansion / "cards" / f"{instance_id.replace('-', '_')}.rs"

    if not card_file.exists():
        return False

    current_content = card_file.read_text()
    eval_module = f"""

// ============================================================================
// EVALUATION TESTS (injected after agent completion)
// ============================================================================

{eval_tests}
"""
    card_file.write_text(current_content + eval_module)
    return True


async def run_cargo_test(repo_dir: Path, instance_id: str) -> tuple[int, int, str]:
    """Run cargo test and return (passed, total, output)."""
    import re

    # Tests are in module eval_tests inside the card file
    # e.g., df::cards::df_001_ampharos::eval_tests::*
    card_module = instance_id.replace("-", "_")
    test_filter = f"{card_module}::eval"
    proc = await asyncio.create_subprocess_exec(
        "cargo",
        "test",
        "--",
        "--test-threads=1",
        test_filter,
        cwd=str(repo_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)
    except TimeoutError:
        proc.kill()
        return 0, 0, "Test timeout"

    output = stdout.decode("utf-8", errors="replace") + stderr.decode("utf-8", errors="replace")
    passed = 0
    failed = 0

    for line in output.split("\n"):
        if "test result:" in line:
            match_passed = re.search(r"(\d+) passed", line)
            match_failed = re.search(r"(\d+) failed", line)
            if match_passed:
                passed = int(match_passed.group(1))
            if match_failed:
                failed = int(match_failed.group(1))

    total = passed + failed
    return passed, total, output


def calculate_score(compile_pass: bool, tests_passed: int, tests_total: int) -> float:
    """Calculate final score (0.0-1.0)."""
    if not compile_pass:
        return 0.0

    compile_weight = 0.30
    test_weight = 0.70

    compile_score = compile_weight
    test_score = test_weight * (tests_passed / tests_total if tests_total > 0 else 0.0)

    return compile_score + test_score


# =============================================================================
# DEFAULT CONTEXT ARTIFACTS (for unified optimization)
# =============================================================================

DEFAULT_SYSTEM_PROMPT = """You are an expert Rust developer implementing Pokemon TCG cards.

Your task: Implement card effects by editing Rust files with stub functions marked with TODO comments.

Key patterns:
- Use `def_id_matches(&card.def_id, "DF", NUMBER)` or `def_id_matches(&card.def_id, "HP", NUMBER)` to identify cards
- Implement attack modifiers in the `attack_override` function
- Implement Poke-Powers/Bodies in the `power_effect` function
- Use `game.queue_prompt()` for user choices
- Return `AttackOverrides::default()` if card doesn't apply

Output requirements:
1. ACTUALLY EDIT files - replace TODO stubs with working code
2. Make sure code compiles (`cargo check`)
3. Make sure tests pass (`cargo test`)"""

DEFAULT_ARCHITECTURE_GUIDE = """# Pokemon TCG Engine Architecture

## Core Concepts

The engine uses a **hook-based** architecture where card implementations register themselves for specific game events.

### Hook System

Card effects are implemented by registering hooks for game events:
- `attack_override`: Modify attack damage/effects before resolution
- `power_effect`: Implement Poke-Powers and Poke-Bodies
- `between_turns`: Handle effects that persist between turns
- `on_damage`: React to damage events

### Pattern Matching

Cards identify themselves using the expansion prefix and card number:
```rust
if def_id_matches(&card.def_id, "DF", 10) {
    // This is Dragon Frontiers card #10 (Snorlax)
    // Implement its effects here
}
```

### State Management

- Use `game.queue_prompt()` for player choices (e.g., discarding cards, choosing Pokemon)
- Access game state through the `game` parameter
- Modify attack properties through `AttackOverrides` struct

## Common Patterns

**Attack Damage Modification:**
```rust
if def_id_matches(&attacker.def_id, "DF", 8) && attack.name == "Volunteer" {
    return AttackOverrides {
        damage_multiplier: 0.0,  // No damage
        ..Default::default()
    };
}
```

**Poke-Power/Body:**
```rust
if def_id_matches(&card.def_id, "DF", 8) && power_name == "Volunteer" {
    // Implement power logic
    return true;  // Power executed successfully
}
```

## Anti-Patterns

- ❌ Don't hardcode card IDs - use `def_id_matches`
- ❌ Don't forget to handle edge cases (no valid targets, etc.)
- ❌ Don't modify state without checking game rules first"""

DEFAULT_REFERENCE_SNIPPETS = """# Reference Implementation Examples

## Example 1: Attack Damage Modifier (Jynx - Selfish)

```rust
// DF-017 Jynx - "Stages of Evolution" Poke-Body
// Damage done to Jynx by attacks from Stage 1 or Stage 2 Evolved Pokemon is reduced by 30
if def_id_matches(&defender.def_id, "DF", 17) {
    let attacker_is_evolved = matches!(attacker_stage, Stage::Stage1 | Stage::Stage2);
    if attacker_is_evolved {
        return AttackOverrides {
            damage_bonus: -30,
            ..Default::default()
        };
    }
}
```

## Example 2: Poke-Power with Player Choice (Ninetales - Volunteer)

```rust
// DF-008 Ninetales - "Volunteer" Poke-Power
// Search your deck for a Basic Pokemon and put it onto your Bench
if def_id_matches(&card.def_id, "DF", 8) && power_name == "Volunteer" {
    // Queue prompt for player to choose a Basic Pokemon from deck
    game.queue_prompt(Prompt::ChooseCardFromDeck {
        filter: |c| matches!(c.stage, Stage::Basic),
        destination: Destination::Bench,
    });
    return true;
}
```

## Example 3: Zero-Damage Attack (Search/Draw effects)

```rust
// Many utility attacks do no damage
if def_id_matches(&attacker.def_id, "DF", 8) && attack.name == "Volunteer" {
    return AttackOverrides {
        damage_multiplier: 0.0,
        // Effect is handled elsewhere (e.g., in power_effect)
        ..Default::default()
    };
}
```

## Example 4: Conditional Damage Bonus

```rust
// Example: Extra damage if opponent is a certain type
if def_id_matches(&attacker.def_id, "DF", 9) && attack.name == "Armor Fang" {
    if defender.types.contains(&Type::Metal) {
        return AttackOverrides {
            damage_bonus: 30,
            ..Default::default()
        };
    }
}
```"""

DEFAULT_HOOKS_DOCUMENTATION = """# Runtime Hooks Reference

## attack_override

```rust
pub fn attack_override(
    game: &Game,
    attack: &Attack,
    attacker_id: CardId,
    defender_id: CardId
) -> AttackOverrides
```

**Purpose:** Modify attack damage or effects before resolution.

**When to use:**
- Reduce/increase attack damage based on card effects
- Change attack properties (e.g., disable effects, change targeting)
- Implement Poke-Bodies that affect damage calculation

**Return:** `AttackOverrides` struct with modifications, or `AttackOverrides::default()` if no modifications.

**Example use cases:**
- "Reduce damage by 30 if attacker is Stage 2"
- "This attack does no damage"
- "Double damage against Fire types"

---

## power_effect

```rust
pub fn power_effect(
    game: &mut Game,
    power_name: &str,
    source_id: CardId
) -> bool
```

**Purpose:** Implement Poke-Powers and Poke-Bodies that have active effects.

**When to use:**
- Player-activated Poke-Powers (e.g., search deck, heal, draw cards)
- Poke-Bodies with ongoing effects that need explicit handling

**Return:** `true` if power executed successfully, `false` otherwise.

**Example use cases:**
- "Search your deck for a Basic Pokemon"
- "Heal 20 damage between turns"
- "Draw a card when this Pokemon is played"

---

## Common Patterns

### Using game.queue_prompt()

For player choices:
```rust
game.queue_prompt(Prompt::ChooseCardFromDeck {
    filter: |c| matches!(c.stage, Stage::Basic),
    destination: Destination::Bench,
});
```

### Accessing game state

```rust
let attacker = game.get_card(attacker_id);
let defender = game.get_card(defender_id);
let attacker_stage = attacker.stage;
```

### Checking card properties

```rust
if def_id_matches(&card.def_id, "DF", 10) {
    // This card is DF-010 (Snorlax)
}

if attacker.types.contains(&Type::Fire) {
    // Attacker is Fire type
}
```"""


# =============================================================================
# AGENT RUNNER (simplified - uses OpenCode via subprocess)
# =============================================================================


async def run_opencode_agent(
    prompt: str,
    sandbox_dir: Path,
    model: str = "gpt-4.1-mini",
    timeout: int = 300,
    inference_url: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run OpenCode agent on the sandbox.

    Args:
        prompt: The task prompt for the agent
        sandbox_dir: Directory to run the agent in
        model: Model to use (e.g. "gpt-4.1-mini")
        timeout: Timeout in seconds
        inference_url: Synth interceptor URL to route LLM calls through
        api_key: API key for the interceptor
    """
    cmd = [
        "opencode",
        "run",
        "--model",
        model,
        "--agent",
        "build",
        "--format",
        "json",
        prompt,
    ]

    # Build environment for subprocess
    env = os.environ.copy()
    if inference_url:
        # Route opencode's LLM calls through the Synth interceptor
        env["OPENAI_BASE_URL"] = inference_url
    if api_key:
        env["OPENAI_API_KEY"] = api_key

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(sandbox_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return {
            "success": proc.returncode == 0,
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
        }
    except TimeoutError:
        proc.kill()
        return {"success": False, "stdout": "", "stderr": f"Timeout after {timeout}s"}
    except Exception as e:
        return {"success": False, "stdout": "", "stderr": str(e)}


def build_prompt(instance: dict[str, Any]) -> str:
    """Build the prompt for the coding agent (LEGACY - for backwards compat)."""
    return build_prompt_with_context(
        instance,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        architecture_guide=DEFAULT_ARCHITECTURE_GUIDE,
        reference_snippets=DEFAULT_REFERENCE_SNIPPETS,
        hooks_documentation=DEFAULT_HOOKS_DOCUMENTATION,
    )


def build_prompt_with_context(
    instance: dict[str, Any],
    system_prompt: str,
    architecture_guide: str,
    reference_snippets: str,
    hooks_documentation: str,
) -> str:
    """Build the prompt for the coding agent with context artifacts.

    This is the UNIFIED approach - combines system prompt with context artifacts
    that can be optimized by GEPA.
    """
    cards = instance.get("cards", [])
    card_file = instance.get("card_file", "").replace("tcg_expansions/", "")
    instance_id = instance.get("id", "")

    expansion = instance.get("expansion", "dragon_frontiers")
    expansion_name = "Holon Phantoms" if expansion == "holon_phantoms" else "Dragon Frontiers"

    card_specs = "\n\n".join(
        [f"### {card['name']}\n{json.dumps(card, indent=2)}" for card in cards]
    )

    tests = instance.get("tests", [])

    def format_test(t):
        desc = t.get("description")
        if desc:
            return f"- {t['name']}: {desc}"
        return f"- {t['name']}: expected={t.get('expected', '?')}"

    test_descriptions = (
        "\n".join([format_test(t) for t in tests]) if tests else "- See card specification"
    )

    # Build unified prompt from context artifacts
    return f"""{system_prompt}

---

# EXPANSION: {expansion_name}

## Cards to Implement
{card_specs}

## File to Edit
`{card_file}` - This file contains stub functions with TODO comments.

## Tests to Pass
{test_descriptions}

---

{architecture_guide}

---

{reference_snippets}

---

{hooks_documentation}

---

## Final Instructions
1. READ the stub file at `{card_file}`
2. Use the architecture guide and reference snippets above as patterns
3. EDIT `{card_file}` to replace the TODO stubs with working implementations
4. Run `cargo check` to verify compilation
5. Run `cargo test -- {instance_id.replace("-", "_")}` to run tests

You MUST edit the file and write actual code. Do not just describe what to do!
"""


# =============================================================================
# TASK APP PROVIDERS
# =============================================================================


def provide_taskset_description() -> dict:
    """Return metadata about the task set."""
    df_count = len([i for i in INSTANCE_IDS if i.startswith("df-")])
    hp_count = len([i for i in INSTANCE_IDS if i.startswith("hp-")])
    return {
        "id": APP_ID,
        "splits": ["df", "hp"],
        "sizes": {"df": df_count, "hp": hp_count, "total": len(INSTANCE_IDS)},
    }


def provide_task_instances(seeds: list[int]):
    """Yield TaskInfo for each seed."""
    for seed in seeds:
        instance_id = get_instance_by_seed(seed)
        instance = load_instance(instance_id)
        cards = instance.get("cards", [])
        card_name = cards[0].get("name", instance_id) if cards else instance_id

        yield TaskInfo(
            task={"id": APP_ID, "name": APP_NAME},
            dataset={
                "id": APP_ID,
                "split": instance_id.split("-")[0],  # df or hp
                "index": seed,
                "instance_id": instance_id,
            },
            inference={"tool": "code_edit"},
            limits={"max_turns": 30},
            task_metadata={
                "instance_id": instance_id,
                "card_name": card_name,
                "expansion": instance.get("expansion", "dragon_frontiers"),
            },
        )


# =============================================================================
# ROLLOUT HANDLER
# =============================================================================


async def run_rollout(request: RolloutRequest, fastapi_request: Request) -> RolloutResponse:
    """
    Handle a single evaluation rollout.

    This runs the full pipeline:
    1. Setup sandbox with stub files
    2. Run OpenCode agent to implement the card
    3. Inject eval tests
    4. Run cargo test
    5. Return score

    Supports UNIFIED CONTEXT ENGINEERING:
    - Extracts context artifacts from request.context_override
    - Falls back to defaults if not provided
    - GEPA can optimize all artifacts together
    """
    seed = request.env.seed or 0
    env_config = request.env.config or {}
    policy_config = request.policy.config or {}
    context_override = request.context_override or {}
    start = time.perf_counter()

    # Get instance - either from config or by seed
    instance_id = env_config.get("instance_id") or get_instance_by_seed(seed)
    instance = load_instance(instance_id)

    model = policy_config.get("model", "gpt-4.1-mini")
    timeout = int(policy_config.get("timeout", 300))
    inference_url = policy_config.get("inference_url")
    api_key = policy_config.get("api_key")

    # UNIFIED CONTEXT ENGINEERING: Extract context artifacts (or use defaults)
    system_prompt = context_override.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    architecture_guide = context_override.get("architecture_guide", DEFAULT_ARCHITECTURE_GUIDE)
    reference_snippets = context_override.get("reference_snippets", DEFAULT_REFERENCE_SNIPPETS)
    hooks_documentation = context_override.get("hooks_documentation", DEFAULT_HOOKS_DOCUMENTATION)

    print(f"\n{'=' * 60}")
    print(f"[engine_bench] Running rollout for {instance_id}")
    print(f"  Model: {model}")
    print(f"  Timeout: {timeout}s")
    print(f"  Interceptor: {inference_url or 'none (direct)'}")
    print(
        f"  Context artifacts: system_prompt={len(system_prompt)} chars, "
        f"arch_guide={len(architecture_guide)} chars, "
        f"ref_snippets={len(reference_snippets)} chars, "
        f"hooks_doc={len(hooks_documentation)} chars"
    )
    print(f"{'=' * 60}\n")

    # Create temp directory for sandbox
    with tempfile.TemporaryDirectory() as work_dir:
        work_path = Path(work_dir)

        # Setup sandbox
        sandbox_dir = await setup_sandbox(instance_id, work_path)

        # Build prompt with context artifacts (UNIFIED APPROACH)
        prompt = build_prompt_with_context(
            instance,
            system_prompt=system_prompt,
            architecture_guide=architecture_guide,
            reference_snippets=reference_snippets,
            hooks_documentation=hooks_documentation,
        )

        await run_opencode_agent(
            prompt,
            sandbox_dir,
            model=model,
            timeout=timeout,
            inference_url=inference_url,
            api_key=api_key,
        )

        card_file_path = None
        card_file_rel = instance.get("card_file", "")
        if card_file_rel:
            relative_path = card_file_rel.replace("tcg_expansions/", "")
            card_file_path = sandbox_dir / relative_path
        else:
            expansion = instance_id.split("-")[0]
            card_file_path = (
                sandbox_dir / "src" / expansion / "cards" / f"{instance_id.replace('-', '_')}.rs"
            )

        artifact_list = []
        if card_file_path and card_file_path.exists():
            content = card_file_path.read_text()
            artifact = Artifact(
                content=content,
                content_type="rust_code",
                metadata={
                    "file_path": str(card_file_path.relative_to(sandbox_dir)),
                    "instance_id": instance_id,
                },
            )
            artifact.validate_size(max_size_bytes=64 * 1024)
            artifact_list.append(artifact)

        # Evaluate
        compile_pass, compile_error = await run_cargo_check(sandbox_dir)

        tests_passed = 0
        tests_total = 0
        test_output = ""

        if compile_pass:
            await inject_eval_tests(sandbox_dir, instance_id)
            tests_passed, tests_total, test_output = await run_cargo_test(sandbox_dir, instance_id)

        score = calculate_score(compile_pass, tests_passed, tests_total)

    print(f"\n{'=' * 60}")
    print(f"[engine_bench] Result for {instance_id}")
    print(f"  Compile: {'PASS' if compile_pass else 'FAIL'}")
    print(f"  Tests: {tests_passed}/{tests_total}")
    print(f"  Score: {score:.2f}")
    print(f"{'=' * 60}\n")

    latency_ms = (time.perf_counter() - start) * 1000.0
    trace_correlation_id = policy_config.get("trace_correlation_id", request.trace_correlation_id)

    return RolloutResponse(
        trace_correlation_id=trace_correlation_id,
        metrics=RolloutMetrics(
            outcome_reward=score,
            details={
                "instance_id": instance_id,
                "compile_pass": compile_pass,
                "tests_passed": tests_passed,
                "tests_total": tests_total,
                "latency_ms": latency_ms,
            },
            outcome_objectives={"reward": score, "latency_ms": latency_ms},
            instance_objectives=[{"reward": score, "latency_ms": latency_ms}],
        ),
        artifact=artifact_list or None,
        success_status=SuccessStatus.SUCCESS,
    )


# =============================================================================
# CREATE THE APP
# =============================================================================

app = create_local_api(
    LocalAPIConfig(
        app_id=APP_ID,
        name=APP_NAME,
        description="EngineBench evaluates coding agents on Pokemon TCG card implementations in Rust.",
        provide_taskset_description=provide_taskset_description,
        provide_task_instances=provide_task_instances,
        rollout=run_rollout,
        cors_origins=["*"],
    )
)


# =============================================================================
# RUNNING LOCALLY
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    from synth_ai.sdk.localapi.auth import ensure_localapi_auth

    port = int(os.getenv("PORT", "8017"))

    # Ensure ENVIRONMENT_API_KEY is set
    env_key = ensure_localapi_auth(
        backend_base="http://localhost:8000",
        synth_api_key=None,
    )
    print(f"[engine_bench] ENVIRONMENT_API_KEY ready: {env_key[:15]}...")
    print(f"[engine_bench] Starting on port {port}")

    uvicorn.run(app, host="0.0.0.0", port=port)
