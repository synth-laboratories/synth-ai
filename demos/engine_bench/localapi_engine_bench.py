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
import difflib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from fastapi import Request

logger = logging.getLogger(__name__)
from synth_ai.data.artifacts import Artifact
from synth_ai.data.enums import SuccessStatus
from synth_ai.sdk.localapi import LocalAPIConfig, create_local_api
from synth_ai.sdk.task.contracts import (
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
    TaskInfo,
)
from synth_ai.sdk.task.rubrics.models import Criterion, Rubric
from synth_ai.sdk.task.server import RubricBundle

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

# Check if OPENAI_API_KEY is available
openai_key = os.environ.get("OPENAI_API_KEY")
if openai_key:
    print(f"[engine_bench] OPENAI_API_KEY available: {openai_key[:20]}...")
else:
    print("[engine_bench] WARNING: OPENAI_API_KEY not found in environment!")


def get_instance_by_seed(seed: int) -> str:
    """Get instance ID by seed (modulo number of instances)."""
    if not INSTANCE_IDS:
        raise ValueError("No instances available")
    return INSTANCE_IDS[seed % len(INSTANCE_IDS)]


# =============================================================================
# SANDBOX SETUP
# =============================================================================


@dataclass
class SandboxSetupResult:
    """Result from sandbox setup, includes paths and original content for diffing."""
    sandbox_dir: Path
    card_file_path: Path | None
    original_stub_content: str  # The stub with todo!() that agent starts with
    gold_implementation: str    # The reference/gold implementation for comparison


async def setup_sandbox(instance_id: str, work_dir: Path) -> SandboxSetupResult:
    """Set up a sandbox for the coding agent using the scaffold from engine-bench.
    
    Returns SandboxSetupResult with original stub content for later diffing.
    """
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
    
    card_file_path: Path | None = None
    original_stub_content = ""
    gold_implementation = ""

    if card_file:
        # Copy gold stub but convert implementations to todo!() so agent has work to do
        gold_stub_file = GOLD_DIR / "stubs" / f"{instance_id.replace('-', '_')}.rs"
        relative_card_file = card_file.replace("tcg_expansions/", "")
        stub_path = sandbox_dir / relative_card_file
        card_file_path = stub_path

        stub_path.parent.mkdir(parents=True, exist_ok=True)

        if gold_stub_file.exists():
            # Read gold stub (this is the reference implementation)
            gold_implementation = gold_stub_file.read_text()
            
            # Replace function bodies with todo!() - look for pattern like:
            # pub fn foo(...) -> Type { ... actual code ... }
            # and replace with: pub fn foo(...) -> Type { todo!() }
            import re
            original_stub_content = re.sub(
                r'(pub fn \w+\([^)]*\)\s*->\s*\w+\s*\{)[^}]+\}',
                r'\1 todo!() }',
                gold_implementation
            )
            stub_path.write_text(original_stub_content)

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

    return SandboxSetupResult(
        sandbox_dir=sandbox_dir,
        card_file_path=card_file_path,
        original_stub_content=original_stub_content,
        gold_implementation=gold_implementation,
    )


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
    print(f"[inject_eval_tests] Looking for: {eval_test_file}", flush=True)
    print(f"[inject_eval_tests] Exists: {eval_test_file.exists()}", flush=True)
    if not eval_test_file.exists():
        print(f"[inject_eval_tests] ❌ Test file does not exist!", flush=True)
        return False

    eval_tests = eval_test_file.read_text()
    print(f"[inject_eval_tests] Read {len(eval_tests)} bytes of eval tests", flush=True)
    
    expansion = instance_id.split("-")[0]
    # sandbox_dir IS tcg_expansions, so path is src/{expansion}/cards/{card}.rs
    card_file = sandbox_dir / "src" / expansion / "cards" / f"{instance_id.replace('-', '_')}.rs"
    print(f"[inject_eval_tests] Card file: {card_file}", flush=True)
    print(f"[inject_eval_tests] Card file exists: {card_file.exists()}", flush=True)

    if not card_file.exists():
        print(f"[inject_eval_tests] ❌ Card file does not exist!", flush=True)
        return False

    current_content = card_file.read_text()
    print(f"[inject_eval_tests] Current card content: {len(current_content)} bytes", flush=True)
    
    eval_module = f"""

// ============================================================================
// EVALUATION TESTS (injected after agent completion)
// ============================================================================

{eval_tests}
"""
    card_file.write_text(current_content + eval_module)
    print(f"[inject_eval_tests] ✅ Injected eval tests, new size: {len(current_content + eval_module)} bytes", flush=True)
    return True


async def run_cargo_test(repo_dir: Path, instance_id: str) -> tuple[int, int, str]:
    """Run cargo test and return (passed, total, output)."""
    import re

    # Tests are in module eval_tests inside the card file
    # e.g., df::cards::df_001_ampharos::eval_tests::*
    card_module = instance_id.replace("-", "_")
    test_filter = f"{card_module}::eval"
    print(f"[run_cargo_test] Running: cargo test -- --test-threads=1 {test_filter}", flush=True)
    print(f"[run_cargo_test] CWD: {repo_dir}", flush=True)
    
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
        print(f"[run_cargo_test] ❌ Test timeout!", flush=True)
        return 0, 0, "Test timeout"

    output = stdout.decode("utf-8", errors="replace") + stderr.decode("utf-8", errors="replace")
    print(f"[run_cargo_test] Exit code: {proc.returncode}", flush=True)
    print(f"[run_cargo_test] Output length: {len(output)} chars", flush=True)
    
    # Log relevant lines
    for line in output.split("\n"):
        if "test result:" in line or "running" in line.lower() or "error" in line.lower():
            print(f"[run_cargo_test] {line}", flush=True)
    
    # SUM all test results (unit tests + doc tests)
    # Cargo outputs multiple "test result:" lines for different test types
    passed = 0
    failed = 0

    for line in output.split("\n"):
        if "test result:" in line:
            match_passed = re.search(r"(\d+) passed", line)
            match_failed = re.search(r"(\d+) failed", line)
            if match_passed:
                passed += int(match_passed.group(1))  # SUM instead of overwrite
            if match_failed:
                failed += int(match_failed.group(1))  # SUM instead of overwrite

    total = passed + failed
    print(f"[run_cargo_test] Result: {passed}/{total}", flush=True)
    return passed, total, output


def calculate_outcome_reward(compile_pass: bool, tests_passed: int, tests_total: int) -> float:
    """Calculate final outcome reward (0.0-1.0)."""
    if not compile_pass:
        return 0.0

    compile_weight = 0.30
    test_weight = 0.70

    compile_reward = compile_weight
    test_reward = test_weight * (tests_passed / tests_total if tests_total > 0 else 0.0)

    return compile_reward + test_reward


# =============================================================================
# INTERCEPTOR URL HELPERS
# =============================================================================


def normalize_interceptor_base(inference_url: str) -> tuple[str, str | None]:
    """Normalize interceptor base URL and extract correlation ID if present."""
    parsed = urlparse(inference_url)
    base_path = parsed.path or ""
    for suffix in ["/v1/chat/completions", "/chat/completions", "/responses", "/v1/responses"]:
        if base_path.endswith(suffix):
            base_path = base_path[: -len(suffix)]
            break
    base = f"{parsed.scheme}://{parsed.netloc}{base_path}".rstrip("/")
    cid_values = parse_qs(parsed.query).get("cid", [])
    correlation_id = cid_values[0] if cid_values else None
    return base, correlation_id


# =============================================================================
# DEFAULT CONTEXT ARTIFACTS (for unified optimization)
# =============================================================================

DEFAULT_SYSTEM_PROMPT = """You are an expert Rust developer implementing Pokemon TCG cards.

CRITICAL: The stub file contains `todo!()` macros that YOU MUST REPLACE with working code.

Example - if you see:
```rust
pub fn grind_damage(attached_energy: u32) -> i32 { todo!() }
// ATTACK_1_TEXT: "Does 10 damage times the amount of Energy attached"
```

You MUST use the edit tool to change it to:
```rust
pub fn grind_damage(attached_energy: u32) -> i32 { (10 * attached_energy) as i32 }
```

REQUIRED WORKFLOW:
1. Read the stub file ONCE to find the todo!() functions
2. IMMEDIATELY use the edit tool to replace todo!() with working code
3. Run `cargo check` to verify compilation
4. Run `cargo test` to verify tests pass

DO NOT read the file multiple times. After ONE read, you must EDIT.
"""

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
# AGENT RUNNER (OpenCode + Codex CLI via subprocess)
# =============================================================================


async def run_opencode_agent(
    prompt: str,
    sandbox_dir: Path,
    model: str = "gpt-4o-mini",
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
    opencode_bin = os.environ.get("OPENCODE_BIN")
    if not opencode_bin:
        preferred = Path.home() / ".opencode" / "bin" / "opencode"
        opencode_bin = str(preferred) if preferred.exists() else shutil.which("opencode")

    if not opencode_bin:
        return {
            "success": False,
            "stdout": "",
            "stderr": "opencode binary not found in PATH",
        }

    # OpenCode reads opencode.json from the working directory or parent tree.
    # Write a scoped config in the sandbox to force interceptor routing.
    base_url = ""
    model_id = model.split("/", 1)[1] if "/" in model else model
    model_with_provider = f"openai/{model_id}"
    actual_api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not actual_api_key:
        print("  [OpenCode] WARNING: No API key available!")
        actual_api_key = "placeholder"

    if inference_url:
        base_url, correlation_id = normalize_interceptor_base(inference_url)
        original_base = base_url
        if correlation_id:
            base_url = f"{base_url}/{correlation_id}"
        logger.info(
            "[OpenCode] URL construction: inference_url=%s base_url=%s correlation_id=%s final_base=%s",
            inference_url,
            original_base,
            correlation_id,
            base_url,
        )
    else:
        logger.info("[OpenCode] No inference_url provided, using direct provider")

    opencode_config = {
        "$schema": "https://opencode.ai/config.json",
        "model": model_with_provider,
        "provider": {
            "openai": {
                # CRITICAL: Must include full provider definition for OpenCode to merge options correctly
                "name": "OpenAI",
                "npm": "@ai-sdk/openai",
                "options": {
                    "apiKey": actual_api_key,
                    "baseURL": base_url or "https://api.openai.com/v1",
                },
                "models": {
                    "gpt-5-nano": {},
                    "gpt-5.2": {},
                    "gpt-4o": {},
                    "gpt-4o-mini": {},
                }
            }
        },
        # OpenCode permissions - allow everything for non-interactive eval
        # CRITICAL: Must explicitly allow external_directory for paths outside sandbox
        "permission": {
            "*": "allow",
            "external_directory": "allow",
            "bash": "allow",
            "read": "allow",
            "write": "allow",
            "edit": "allow",
            "list": "allow",
            "glob": "allow",
            "grep": "allow",
        },
    }

    config_path = sandbox_dir / "opencode.json"
    config_path.write_text(json.dumps(opencode_config, indent=2))

    logger.info(
        "[OpenCode] Config written: path=%s model=%s baseURL=%s",
        config_path,
        model_with_provider,
        opencode_config["provider"]["openai"]["options"]["baseURL"],
    )
    print(f"  [OpenCode] Config written to: {config_path}")
    print(f"  [OpenCode] Model: {model_with_provider}")
    print(f"  [OpenCode] BaseURL: {base_url}")
    print(
        f"  [OpenCode] API key: {actual_api_key[:15]}..."
        if len(actual_api_key) > 15
        else "  [OpenCode] API key: (short)"
    )
    print(f"  [OpenCode] Working directory: {sandbox_dir}")

    cmd = [
        opencode_bin,
        "run",
        "--format",
        "json",
        "--model",
        model_with_provider,
        prompt,
    ]
    if os.environ.get("OPENCODE_DEBUG") == "1":
        cmd.extend(["--print-logs", "--log-level", "DEBUG"])

    # Build environment for subprocess
    env = os.environ.copy()
    if actual_api_key and actual_api_key != "placeholder":
        env["OPENAI_API_KEY"] = actual_api_key
        print("  [OpenCode] OPENAI_API_KEY set in subprocess environment")
    print(f"  [OpenCode] Using binary: {opencode_bin}")

    logger.info(
        "[OpenCode] Starting subprocess: cmd=%s cwd=%s timeout=%ds",
        " ".join(cmd),
        sandbox_dir,
        timeout,
    )

    try:
        print(f"[OpenCode] ⚡⚡⚡ STARTING SUBPROCESS: cmd={cmd[:3]}... cwd={sandbox_dir}", flush=True)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(sandbox_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        print(f"[OpenCode] ⚡⚡⚡ SUBPROCESS STARTED: pid={proc.pid}", flush=True)
        
        # Stream output in real-time
        stdout_chunks = []
        stderr_chunks = []
        
        async def read_stream(stream, chunks, prefix):
            try:
                while True:
                    chunk = await stream.read(1024)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    text = chunk.decode("utf-8", errors="replace")
                    # Print each line as it comes
                    for line in text.splitlines():
                        if line.strip():
                            print(f"[OpenCode] {prefix}: {line}", flush=True)
            except Exception as e:
                print(f"[OpenCode] ⚡⚡⚡ Error reading {prefix}: {e}", flush=True)
        
        # Start reading streams
        stdout_task = asyncio.create_task(read_stream(proc.stdout, stdout_chunks, "STDOUT"))
        stderr_task = asyncio.create_task(read_stream(proc.stderr, stderr_chunks, "STDERR"))
        
        try:
            # Wait for process to finish (with timeout)
            returncode = await asyncio.wait_for(proc.wait(), timeout=timeout)
            # Wait for streams to finish reading
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        except asyncio.TimeoutError:
            print(f"[OpenCode] ⚡⚡⚡ TIMEOUT after {timeout}s - killing process", flush=True)
            proc.kill()
            await proc.wait()
            stdout_task.cancel()
            stderr_task.cancel()
            return {"success": False, "stdout": "", "stderr": f"Timeout after {timeout}s"}
        
        stdout = b"".join(stdout_chunks)
        stderr = b"".join(stderr_chunks)
        
        print(f"[OpenCode] ⚡⚡⚡ SUBPROCESS COMPLETED: returncode={returncode} stdout_len={len(stdout)} stderr_len={len(stderr)}", flush=True)
        
        return {
            "success": returncode == 0,
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
        }
    except asyncio.TimeoutError:
        print(f"[OpenCode] ⚡⚡⚡ TIMEOUT after {timeout}s", flush=True)
        proc.kill()
        return {"success": False, "stdout": "", "stderr": f"Timeout after {timeout}s"}
    except Exception as e:
        print(f"[OpenCode] ⚡⚡⚡ EXCEPTION: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return {"success": False, "stdout": "", "stderr": str(e)}


async def run_codex_agent(
    prompt: str,
    sandbox_dir: Path,
    model: str = "gpt-4o-mini",
    timeout: int = 300,
    inference_url: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run Codex CLI agent on the sandbox.

    Args:
        prompt: The task prompt for the agent
        sandbox_dir: Directory to run the agent in
        model: Model to use (e.g. "gpt-4o-mini")
        timeout: Timeout in seconds
        inference_url: Synth interceptor URL to route LLM calls through
        api_key: API key for the interceptor
    """
    if not shutil.which("codex"):
        return {
            "success": False,
            "stdout": "",
            "stderr": "codex binary not found in PATH",
        }

    config_dir = Path.home() / ".codex"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.toml"

    base_url = "https://api.openai.com/v1"
    if inference_url:
        base_url, correlation_id = normalize_interceptor_base(inference_url)
        if correlation_id:
            base_url = f"{base_url}/{correlation_id}"

    config_content = f"""# Auto-generated for EngineBench local runs

model = "{model}"
model_provider = "openai"

[model_providers.openai]
name = "OpenAI"
base_url = "{base_url}"
wire_api = "responses"
env_key = "OPENAI_API_KEY"
requires_openai_auth = false
request_max_retries = 4
stream_max_retries = 5
stream_idle_timeout_ms = 300000

[mcp]
enabled = false
"""
    config_file.write_text(config_content)

    cmd = [
        "codex",
        "exec",
        "--yolo",
        "--skip-git-repo-check",
        "-m",
        model,
        prompt,
    ]

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = api_key or os.environ.get("OPENAI_API_KEY", "")
    env["OPENAI_MODEL"] = model
    if inference_url:
        env["OPENAI_BASE_URL"] = base_url

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


async def run_codex_agent(
    prompt: str,
    sandbox_dir: Path,
    model: str = "gpt-4.1-mini",
    timeout: int = 300,
    inference_url: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run Codex CLI agent on the sandbox.

    Args:
        prompt: The task prompt for the agent
        sandbox_dir: Directory to run the agent in
        model: Model to use (e.g. "gpt-4.1-mini")
        timeout: Timeout in seconds
        inference_url: Synth interceptor URL to route LLM calls through
        api_key: API key for the interceptor
    """
    # Setup codex config for pure API mode
    config_dir = Path.home() / ".codex"
    config_dir.mkdir(exist_ok=True)
    config_file = config_dir / "config.toml"
    auth_file = config_dir / "auth.json"

    # CRITICAL: Remove auth.json to prevent ChatGPT account from overriding API config
    if auth_file.exists():
        backup_file = config_dir / "auth.json.bak"
        if not backup_file.exists():
            auth_file.rename(backup_file)
        else:
            auth_file.unlink()

    # Determine wire_api based on model
    # Responses API models: gpt-5*, o3*, o1*, codex-*
    # Chat completions: everything else
    if any(model.startswith(prefix) for prefix in ["gpt-5", "o3", "o1", "codex-"]):
        wire_api = "responses"
    else:
        wire_api = "chat"

    base_url = "https://api.openai.com/v1"
    if inference_url:
        # Extract base URL from inference_url (remove /openai/v1 or /v1/chat/completions)
        base_url = inference_url
        if base_url.endswith("/openai/v1"):
            base_url = base_url[:-10]
        elif "/v1/chat/completions" in base_url:
            base_url = base_url.split("/v1/chat/completions")[0]
        elif "/chat/completions" in base_url:
            base_url = base_url.split("/chat/completions")[0]

    config_content = f'''# Auto-generated config for engine_bench
# Pure API mode (no Codex Cloud login required)

model = "{model}"
model_provider = "openai"

[model_providers.openai]
name = "OpenAI"
base_url = "{base_url}"
wire_api = "{wire_api}"
env_key = "OPENAI_API_KEY"
requires_openai_auth = false
request_max_retries = 4
stream_max_retries = 5
stream_idle_timeout_ms = 300000

[mcp]
enabled = false
'''
    config_file.write_text(config_content)

    # Build command
    cmd = [
        "codex",
        "exec",
        "--yolo",  # Zero prompts, fully autonomous
        "--skip-git-repo-check",
        "-m",
        model,
        prompt,
    ]

    # Build environment for subprocess
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = api_key or os.environ.get("OPENAI_API_KEY", "")
    env["OPENAI_MODEL"] = model
    if inference_url:
        # Route codex's LLM calls through the Synth interceptor
        env["OPENAI_BASE_URL"] = inference_url

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
1. READ the stub file at `{card_file}` using the `read` tool
2. Use the architecture guide and reference snippets above as patterns
3. USE THE `edit` OR `write` TOOL to modify `{card_file}` and replace the TODO stubs with working implementations
4. Run `cargo check` using the `bash` tool to verify compilation
5. Run `cargo test -- {instance_id.replace("-", "_")}` using the `bash` tool to run tests

CRITICAL: You MUST use the `edit` or `write` tool to actually modify the file. Reading the file is not enough - you must write code!
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
    2. Run OpenCode or Codex CLI agent to implement the card
    3. Inject eval tests
    4. Run cargo test
    5. Return outcome reward

    Agent selection:
    - Set policy_config["agent"] = "codex" to use Codex CLI
    - Defaults to "opencode" for backward compatibility

    Supports UNIFIED CONTEXT ENGINEERING:
    - Extracts context artifacts from request.context_override
    - Falls back to defaults if not provided
    - GEPA can optimize all artifacts together
    """
    seed = request.env.seed or 0
    env_config = request.env.config or {}
    policy_config = request.policy.config or {}
    context_override = getattr(request, "context_override", None) or {}
    start = time.perf_counter()

    # Get instance - either from config or by seed
    instance_id = env_config.get("instance_id") or get_instance_by_seed(seed)
    instance = load_instance(instance_id)

    model = policy_config.get("model", "gpt-4o-mini")
    timeout = int(policy_config.get("timeout", 300))
    inference_url = policy_config.get("inference_url")
    # Use Synth API key for interceptor auth
    api_key = os.environ.get("SYNTH_API_KEY")
    agent_type = policy_config.get("agent", "opencode")
    
    logger.info(
        "[engine_bench] Agent config: type=%s model=%s inference_url=%s timeout=%ds",
        agent_type,
        model,
        inference_url or "none (direct)",
        timeout,
    )

    # UNIFIED CONTEXT ENGINEERING: Extract context artifacts (or use defaults)
    system_prompt = context_override.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    architecture_guide = context_override.get("architecture_guide", DEFAULT_ARCHITECTURE_GUIDE)
    reference_snippets = context_override.get("reference_snippets", DEFAULT_REFERENCE_SNIPPETS)
    hooks_documentation = context_override.get("hooks_documentation", DEFAULT_HOOKS_DOCUMENTATION)

    print(f"\n{'=' * 60}", flush=True)
    print(f"[engine_bench] ⚡⚡⚡ Running rollout for {instance_id}", flush=True)
    print(f"  Agent: {agent_type}", flush=True)
    print(f"  Model: {model}", flush=True)
    print(f"  Timeout: {timeout}s", flush=True)
    print(f"  Interceptor: {inference_url or 'none (direct)'}", flush=True)
    api_key_status = f"✓ present ({api_key[:20]}...)" if api_key else "✗ missing"
    print(f"  API Key: {api_key_status}", flush=True)
    print(
        f"  Context artifacts: system_prompt={len(system_prompt)} chars, "
        f"arch_guide={len(architecture_guide)} chars, "
        f"ref_snippets={len(reference_snippets)} chars, "
        f"hooks_doc={len(hooks_documentation)} chars", flush=True
    )
    print(f"{'=' * 60}\n", flush=True)

    # Create temp directory for sandbox
    agent_result: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as work_dir:
        work_path = Path(work_dir)

        # Setup sandbox - returns original stub content for diffing
        setup_result = await setup_sandbox(instance_id, work_path)
        sandbox_dir = setup_result.sandbox_dir

        # HACK: Create empty AGENTS.md to stop OpenCode from searching for it in an infinite loop
        # OpenCode seems to have default behavior to search for this file
        agents_md_path = sandbox_dir / "AGENTS.md"
        agents_md_path.write_text("# Agent Instructions\n\nSee the task prompt for instructions.\n")
        print(f"[engine_bench] ⚡⚡⚡ Created AGENTS.md hack file: {agents_md_path}", flush=True)

        # Build prompt with context artifacts (UNIFIED APPROACH)
        prompt = build_prompt_with_context(
            instance,
            system_prompt=system_prompt,
            architecture_guide=architecture_guide,
            reference_snippets=reference_snippets,
            hooks_documentation=hooks_documentation,
        )

        if agent_type == "codex":
            agent_result = await run_codex_agent(
                prompt,
                sandbox_dir,
                model=model,
                timeout=timeout,
                inference_url=inference_url,
                api_key=api_key,
            )
            if not agent_result["success"]:
                print("  [Codex] FAILED")
                print(f"  [Codex] stdout: {agent_result.get('stdout', '')[:500]}")
                print(f"  [Codex] stderr: {agent_result.get('stderr', '')[:1000]}")
            else:
                print("  [Codex] Completed successfully")
                if agent_result.get("stderr"):
                    stderr_full = agent_result["stderr"]
                    if stderr_full.strip():
                        print(f"  [Codex] stderr (full):\n{stderr_full}")
        else:
            print(f"[engine_bench] ⚡⚡⚡ CALLING run_opencode_agent: model={model} timeout={timeout} inference_url={inference_url}", flush=True)
            agent_result = await run_opencode_agent(
                prompt,
                sandbox_dir,
                model=model,
                timeout=timeout,
                inference_url=inference_url,
                api_key=api_key,
            )
            print(f"[engine_bench] ⚡⚡⚡ run_opencode_agent RETURNED: success={agent_result.get('success')}", flush=True)

            # Log OpenCode output for debugging
            if not agent_result["success"]:
                print("  [OpenCode] ⚡⚡⚡ FAILED", flush=True)
                print(f"  [OpenCode] ⚡⚡⚡ stdout: {agent_result.get('stdout', '')[:500]}", flush=True)
                print(f"  [OpenCode] ⚡⚡⚡ stderr: {agent_result.get('stderr', '')[:1000]}", flush=True)
            else:
                print("  [OpenCode] ⚡⚡⚡ Completed successfully", flush=True)
                if agent_result.get("stderr"):
                    stderr_full = agent_result["stderr"]
                    if stderr_full.strip():
                        print(f"  [OpenCode] ⚡⚡⚡ stderr (full):\n{stderr_full}", flush=True)

        # Use card_file_path from setup_result if available, otherwise derive it
        card_file_path = setup_result.card_file_path
        if not card_file_path:
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
        final_code = ""
        if card_file_path and card_file_path.exists():
            final_code = card_file_path.read_text()
            
            # Artifact 1: The final code the agent produced
            artifact = Artifact(
                content=final_code,
                content_type="rust_code",
                metadata={
                    "file_path": str(card_file_path.relative_to(sandbox_dir)),
                    "instance_id": instance_id,
                    "artifact_type": "final_code",
                },
            )
            artifact.validate_size(max_size_bytes=64 * 1024)
            artifact_list.append(artifact)
            
            # Artifact 2: Unified diff showing what the agent changed
            if setup_result.original_stub_content:
                diff_lines = list(difflib.unified_diff(
                    setup_result.original_stub_content.splitlines(keepends=True),
                    final_code.splitlines(keepends=True),
                    fromfile="original_stub.rs",
                    tofile="agent_output.rs",
                    lineterm="",
                ))
                if diff_lines:
                    diff_content = "".join(diff_lines)
                    diff_artifact = Artifact(
                        content=diff_content,
                        content_type="unified_diff",
                        metadata={
                            "instance_id": instance_id,
                            "artifact_type": "agent_diff",
                            "description": "Changes made by the agent (original stub -> final code)",
                        },
                    )
                    diff_artifact.validate_size(max_size_bytes=64 * 1024)
                    artifact_list.append(diff_artifact)
                    print(f"[engine_bench] Created diff artifact: {len(diff_content)} chars", flush=True)
            
            # Artifact 3: Gold reference implementation for verifier comparison
            if setup_result.gold_implementation:
                gold_artifact = Artifact(
                    content=setup_result.gold_implementation,
                    content_type="rust_code_gold",
                    metadata={
                        "instance_id": instance_id,
                        "artifact_type": "gold_reference",
                        "description": "Reference implementation for comparison",
                    },
                )
                gold_artifact.validate_size(max_size_bytes=64 * 1024)
                artifact_list.append(gold_artifact)
                print(f"[engine_bench] Created gold reference artifact: {len(setup_result.gold_implementation)} chars", flush=True)

        # Evaluate
        print(f"[engine_bench] Running cargo check in {sandbox_dir}...", flush=True)
        compile_pass, compile_error = await run_cargo_check(sandbox_dir)
        print(f"[engine_bench] Cargo check: {'PASS' if compile_pass else 'FAIL'}", flush=True)
        if compile_error:
            print(f"[engine_bench] Compile error: {compile_error[:500]}", flush=True)

        tests_passed = 0
        tests_total = 0
        test_output = ""

        if compile_pass:
            print(f"[engine_bench] Injecting eval tests...", flush=True)
            inject_success = await inject_eval_tests(sandbox_dir, instance_id)
            print(f"[engine_bench] Inject result: {inject_success}", flush=True)
            
            print(f"[engine_bench] Running cargo tests...", flush=True)
            tests_passed, tests_total, test_output = await run_cargo_test(sandbox_dir, instance_id)
            print(f"[engine_bench] Test result: {tests_passed}/{tests_total}", flush=True)
        else:
            print(f"[engine_bench] Skipping tests (compile failed)", flush=True)

        outcome_reward_value = calculate_outcome_reward(compile_pass, tests_passed, tests_total)

    print(f"\n{'=' * 60}")
    print(f"[engine_bench] Result for {instance_id}")
    print(f"  Compile: {'PASS' if compile_pass else 'FAIL'}")
    print(f"  Tests: {tests_passed}/{tests_total}")
    print(f"  Reward: {outcome_reward_value:.2f}")
    print(f"{'=' * 60}\n")

    latency_ms = (time.perf_counter() - start) * 1000.0
    trace_correlation_id = policy_config.get("trace_correlation_id", request.trace_correlation_id)

    # Return RolloutResponse - let validation hydrate traces from interceptor
    # OpenCode's LLM calls go through interceptor (via OPENAI_BASE_URL env var)
    # so traces should be captured in Redis for validation
    return RolloutResponse(
        trace_correlation_id=trace_correlation_id,
        metrics=RolloutMetrics(
            outcome_reward=outcome_reward_value,
            details={
                "instance_id": instance_id,
                "compile_pass": compile_pass,
                "tests_passed": tests_passed,
                "tests_total": tests_total,
                "latency_ms": latency_ms,
                "agent": agent_type,
                "agent_success": agent_result.get("success"),
                "agent_stdout_tail": agent_result.get("stdout", "")[-1500:],
                "agent_stderr_tail": agent_result.get("stderr", "")[-2000:],
            },
            outcome_objectives={"reward": outcome_reward_value, "latency_ms": latency_ms},
            instance_objectives=[{"reward": outcome_reward_value, "latency_ms": latency_ms}],
        ),
        trace=None,  # Let validation hydrate from interceptor
        artifact=artifact_list or None,
        success_status=SuccessStatus.SUCCESS,
    )


# =============================================================================
# RUBRIC DEFINITION
# =============================================================================
# This rubric is exposed via GET /info for verifier graphs to use.
# The verifier graph ID is configured in the eval job, not here.

ENGINE_BENCH_OUTCOME_RUBRIC = Rubric(
    version="1.0",
    goal_text="""Evaluate Pokemon TCG card implementation quality in Rust.

ARTIFACTS PROVIDED:
1. 'rust_code' (final_code) - Agent's final implementation
2. 'unified_diff' (agent_diff) - What changed from todo!() stub to final
3. 'rust_code_gold' (gold_reference) - Correct reference implementation

SCORING: Each criterion scored 0.0-1.0. Compare agent output to gold reference.""",
    criteria=[
        Criterion(
            id="compilation",
            description="""Does the code compile?
- 1.0: Compiles with no errors (cargo check passes)
- 0.5: Minor warnings but compiles
- 0.0: Compilation errors""",
            weight=2.0,
            required=True,
        ),
        Criterion(
            id="correctness_vs_gold",
            description="""Does the implementation match the gold reference logic?
Compare 'rust_code' (agent) to 'rust_code_gold' (reference):
- 1.0: Logic is equivalent to gold reference (exact match or correct alternative)
- 0.7: Mostly correct with minor logic differences
- 0.4: Partially correct, some abilities work
- 0.0: Logic is wrong or missing""",
            weight=3.0,
            required=False,
        ),
        Criterion(
            id="completeness",
            description="""Are all todo!() stubs replaced with implementations?
Check the 'unified_diff' artifact:
- 1.0: All todo!() replaced with working code
- 0.5: Some todo!() remain or partial implementations
- 0.0: Most/all todo!() still present""",
            weight=2.5,
            required=False,
        ),
        Criterion(
            id="pattern_adherence",
            description="""Does code follow game engine patterns?
Compare to gold reference for pattern usage:
- 1.0: Uses correct hooks, effect handlers, card registration
- 0.5: Patterns partially correct
- 0.0: Ignores established patterns""",
            weight=1.5,
            required=False,
        ),
        Criterion(
            id="code_quality",
            description="""Is the code clean and idiomatic Rust?
- 1.0: Clean, idiomatic, good naming, proper error handling
- 0.5: Functional but messy or non-idiomatic
- 0.0: Poor quality, hard to read""",
            weight=1.0,
            required=False,
        ),
    ],
    aggregation="weighted_sum",
)

ENGINE_BENCH_EVENT_RUBRIC = Rubric(
    version="1.0",
    goal_text="Evaluate intermediate agent actions during implementation.",
    criteria=[
        Criterion(
            id="file_reading",
            description="Agent reads relevant files before making changes. Shows understanding of existing code structure.",
            weight=1.0,
        ),
        Criterion(
            id="incremental_progress",
            description="Agent makes incremental progress toward solution. Avoids going in circles or repeating actions.",
            weight=1.0,
        ),
        Criterion(
            id="tool_usage",
            description="Agent uses appropriate tools (read, edit, write, bash). Efficiently navigates codebase and makes changes.",
            weight=1.0,
        ),
        Criterion(
            id="verification",
            description="Agent verifies work with cargo check/test. Doesn't assume code is correct without testing.",
            weight=1.5,
        ),
    ],
    aggregation="weighted_sum",
)

ENGINE_BENCH_RUBRICS = RubricBundle(
    outcome=ENGINE_BENCH_OUTCOME_RUBRIC,
    events=ENGINE_BENCH_EVENT_RUBRIC,
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
        rubrics=ENGINE_BENCH_RUBRICS,
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
