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
from pathlib import Path
from typing import Any

from fastapi import Request
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

    test_filter = instance_id.replace("-", "_")
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
    """Build the prompt for the coding agent."""
    cards = instance.get("cards", [])
    # Card file path is relative to tcg_expansions in the instance, strip the prefix
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

    return f"""You are implementing Pokemon TCG cards for the {expansion_name} expansion.

## Task
EDIT the file `{card_file}` to implement the card below. You MUST:
1. Actually WRITE code to the file - replace the TODO stubs with working implementations
2. Make sure it compiles without errors
3. Make sure all tests pass

## Cards to Implement
{card_specs}

## File to Edit
`{card_file}` - This file contains stub functions with TODO comments.

## Tests to Pass
{test_descriptions}

## Instructions
1. READ the stub file at `{card_file}`
2. Look at Crystal Guardians (src/cg/) for reference implementations
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
    """
    seed = request.env.seed or 0
    env_config = request.env.config or {}
    policy_config = request.policy.config or {}

    # Get instance - either from config or by seed
    instance_id = env_config.get("instance_id") or get_instance_by_seed(seed)
    instance = load_instance(instance_id)

    model = policy_config.get("model", "gpt-4.1-mini")
    timeout = int(policy_config.get("timeout", 300))
    inference_url = policy_config.get("inference_url")
    api_key = policy_config.get("api_key")

    print(f"\n{'=' * 60}")
    print(f"[engine_bench] Running rollout for {instance_id}")
    print(f"  Model: {model}")
    print(f"  Timeout: {timeout}s")
    print(f"  Interceptor: {inference_url or 'none (direct)'}")
    print(f"{'=' * 60}\n")

    # Create temp directory for sandbox
    with tempfile.TemporaryDirectory() as work_dir:
        work_path = Path(work_dir)

        # Setup sandbox
        sandbox_dir = await setup_sandbox(instance_id, work_path)

        # Build prompt and run agent
        prompt = build_prompt(instance)
        await run_opencode_agent(
            prompt,
            sandbox_dir,
            model=model,
            timeout=timeout,
            inference_url=inference_url,
            api_key=api_key,
        )

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
            },
        ),
        metadata={
            "instance_id": instance_id,
            "model": model,
            "compile_pass": compile_pass,
            "tests_passed": tests_passed,
            "tests_total": tests_total,
        },
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
