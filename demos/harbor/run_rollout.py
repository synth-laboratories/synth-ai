#!/usr/bin/env python3
"""
Harbor Runner Script for EngineBench evaluations.

This script follows the Harbor runner contract:
- Input:  JSON file with trace_correlation_id, seed, prompt_template, inference_url, limits
- Output: JSON file with trace_correlation_id, metrics.reward_mean, success

Usage:
    run_rollout --input /tmp/rollout.json --output /tmp/result.json

Or via stdin/stdout:
    echo '{"seed": 42, ...}' | run_rollout --stdio
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import shutil
from urllib.parse import parse_qs, urlparse

# =============================================================================
# INSTANCE LOADING (dynamic from engine-bench repo, fallback to hardcoded)
# =============================================================================

# Default hardcoded tasks (fallback if instance files not found)
_FALLBACK_TASKS = [
    {"id": "aerodactyl", "card": "Aerodactyl", "set": "Fossil"},
    {"id": "alakazam", "card": "Alakazam", "set": "Base"},
    {"id": "blastoise", "card": "Blastoise", "set": "Base"},
    {"id": "charizard", "card": "Charizard", "set": "Base"},
    {"id": "clefairy", "card": "Clefairy", "set": "Base"},
    {"id": "dragonite", "card": "Dragonite", "set": "Fossil"},
    {"id": "gengar", "card": "Gengar", "set": "Fossil"},
    {"id": "gyarados", "card": "Gyarados", "set": "Base"},
    {"id": "machamp", "card": "Machamp", "set": "Base"},
    {"id": "mewtwo", "card": "Mewtwo", "set": "Base"},
    {"id": "nidoking", "card": "Nidoking", "set": "Base"},
    {"id": "ninetales", "card": "Ninetales", "set": "Base"},
    {"id": "pikachu", "card": "Pikachu", "set": "Base"},
    {"id": "poliwrath", "card": "Poliwrath", "set": "Base"},
    {"id": "raichu", "card": "Raichu", "set": "Base"},
    {"id": "venusaur", "card": "Venusaur", "set": "Base"},
    {"id": "zapdos", "card": "Zapdos", "set": "Fossil"},
    {"id": "articuno", "card": "Articuno", "set": "Fossil"},
    {"id": "moltres", "card": "Moltres", "set": "Fossil"},
    {"id": "snorlax", "card": "Snorlax", "set": "Jungle"},
]

ENGINE_BENCH_DATA_DIR = Path("/engine-bench/data")


def load_instance_ids() -> List[str]:
    """Load available instance IDs from the engine-bench data directory."""
    instances_dir = ENGINE_BENCH_DATA_DIR / "instances" / "single"
    if not instances_dir.exists():
        return []
    return sorted([p.stem for p in instances_dir.glob("*.json")])


def load_instance(instance_id: str) -> Dict[str, Any]:
    """Load a full instance specification from disk."""
    instance_path = ENGINE_BENCH_DATA_DIR / "instances" / "single" / f"{instance_id}.json"
    if not instance_path.exists():
        raise ValueError(f"Instance not found: {instance_id}")
    return json.loads(instance_path.read_text())


# Load instance IDs at module level (empty list if not in container)
INSTANCE_IDS = load_instance_ids()

# Legacy task list for backward compatibility
TASKS = _FALLBACK_TASKS


# =============================================================================
# DEFAULT CONTEXT ARTIFACTS
# =============================================================================

DEFAULT_SYSTEM_PROMPT = """You are an expert Rust developer implementing Pokemon TCG cards.

CRITICAL: The stub file contains `todo!()` macros that YOU MUST REPLACE with working code.

Example - if you see:
```rust
pub fn grind_damage(attached_energy: u32) -> i32 { todo!() }
```
You must replace `todo!()` with the actual implementation.

Your task: Implement card effects by editing Rust files with stub functions marked with TODO comments.

Key patterns:
- Use `def_id_matches(&card.def_id, "DF", NUMBER)` to identify cards
- Implement attack modifiers in the `attack_override` function
- Use `game.queue_prompt()` for user choices
- Return `AttackOverrides::default()` if card doesn't apply

Output requirements:
1. EDIT files - replace TODO stubs with working code
2. Make code compile (`cargo check`)
3. Make tests pass (`cargo test`)"""

DEFAULT_ARCHITECTURE_GUIDE = """# Pokemon TCG Engine Architecture

## Core Concepts

The engine uses a hook-based architecture where card implementations register themselves for specific game events.

### Hook System

Card effects are implemented by registering hooks for game events:
- `attack_override`: Modify attack damage or effects
- `defend_override`: Modify incoming damage
- `poke_power`: Implement Pokemon Powers
- `poke_body`: Implement Pokemon Bodies (passive effects)

### Card Identification

Cards are identified by their set prefix and number:
- `def_id_matches(&card.def_id, "DF", 1)` matches Dragon Frontiers card #1
- `def_id_matches(&card.def_id, "HP", 15)` matches Holon Phantoms card #15"""

DEFAULT_REFERENCE_SNIPPETS = ""

DEFAULT_HOOKS_DOCUMENTATION = ""


# =============================================================================
# INTERCEPTOR URL HELPERS
# =============================================================================


def normalize_interceptor_base(inference_url: str) -> tuple[str, str | None]:
    """Normalize interceptor base URL and extract correlation ID if present.
    
    Returns a base URL compatible with OpenAI SDK conventions:
    - SDK expects baseURL like https://api.openai.com/v1
    - SDK calls {baseURL}/chat/completions and {baseURL}/models
    """
    parsed = urlparse(inference_url)
    base_path = parsed.path or ""
    
    # Strip endpoint suffix but KEEP /v1 prefix (OpenAI SDK convention)
    for suffix in ["/chat/completions", "/responses"]:
        if base_path.endswith(suffix):
            base_path = base_path[: -len(suffix)]
            break
    
    base = f"{parsed.scheme}://{parsed.netloc}{base_path}".rstrip("/")
    cid_values = parse_qs(parsed.query).get("cid", [])
    correlation_id = cid_values[0] if cid_values else None
    return base, correlation_id


def resolve_interceptor_api_key(
    *, inference_url: str | None, interceptor_key: str | None, openai_key: str | None
) -> str:
    """Choose the API key for agent calls."""
    if inference_url and interceptor_key:
        return interceptor_key
    if openai_key:
        return openai_key
    return interceptor_key or ""


@dataclass
class RolloutInput:
    """Input payload for a rollout."""
    trace_correlation_id: str
    seed: int
    prompt_template: Dict[str, Any]
    inference_url: str
    limits: Dict[str, Any]
    params: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RolloutInput":
        return cls(
            trace_correlation_id=data.get("trace_correlation_id", "unknown"),
            seed=data.get("seed", 0),
            prompt_template=data.get("prompt_template", {}),
            inference_url=data["inference_url"],  # Required
            limits=data.get("limits", {"timeout_s": 300}),
            params=data.get("params", {}),
        )


@dataclass
class RolloutResult:
    """Output payload for a rollout."""
    trace_correlation_id: str
    metrics: Dict[str, Any]
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_correlation_id": self.trace_correlation_id,
            "metrics": self.metrics,
            "success": self.success,
            "error": self.error,
        }


def get_task_for_seed(seed: int) -> Dict[str, Any]:
    """Get the task for a given seed (deterministic mapping).

    If instance files are available (running inside container with /engine-bench),
    loads the full instance spec. Otherwise falls back to hardcoded task list.
    """
    if INSTANCE_IDS:
        instance_id = INSTANCE_IDS[seed % len(INSTANCE_IDS)]
        return load_instance(instance_id)
    return TASKS[seed % len(TASKS)]


def setup_workspace(task: Dict[str, Any], workspace_path: Path) -> Path:
    """Set up the workspace for a task.

    Copies the engine-bench repo and prepares it for the agent.
    """
    # Copy engine-bench repo
    engine_bench_src = Path("/engine-bench")
    if not engine_bench_src.exists():
        raise RuntimeError("Engine-bench repo not found at /engine-bench")

    # Create workspace
    workspace_path.mkdir(parents=True, exist_ok=True)

    # Copy the repo
    shutil.copytree(engine_bench_src, workspace_path / "engine-bench", dirs_exist_ok=True)

    return workspace_path / "engine-bench"


def build_prompt(
    task: Dict[str, Any],
    prompt_template: Dict[str, Any],
    context_overrides: Optional[Dict[str, Any]] = None,
) -> str:
    """Build the agent prompt from the task, template, and context overrides.

    Supports two modes:
    1. Legacy mode (hardcoded tasks): simple card name/set prompt
    2. Instance mode (loaded from /engine-bench): full instance-aware prompt
       matching localapi_engine_bench.py's build_prompt_with_context()

    Context overrides (from GEPA) can replace the default system_prompt,
    architecture_guide, reference_snippets, and hooks_documentation.
    """
    context_overrides = context_overrides or {}

    # Extract context artifacts, falling back to defaults
    system_prompt = context_overrides.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    architecture_guide = context_overrides.get("architecture_guide", DEFAULT_ARCHITECTURE_GUIDE)
    reference_snippets = context_overrides.get("reference_snippets", DEFAULT_REFERENCE_SNIPPETS)
    hooks_documentation = context_overrides.get("hooks_documentation", DEFAULT_HOOKS_DOCUMENTATION)

    # Instance mode: full instance spec loaded from /engine-bench
    if "cards" in task:
        return _build_instance_prompt(
            task,
            system_prompt=system_prompt,
            architecture_guide=architecture_guide,
            reference_snippets=reference_snippets,
            hooks_documentation=hooks_documentation,
        )

    # Legacy mode: simple card name/set
    card_name = task.get("card", "Unknown")
    card_set = task.get("set", "Unknown")

    default_prompt = f"""
Implement the Pokemon TCG card "{card_name}" from the {card_set} set in Rust.

The card implementation should be in the engine-bench repository.
Look at existing card implementations for reference (e.g., Crystal Guardians cards).

Requirements:
1. Implement all abilities and attacks for the card
2. Follow the existing code patterns
3. Ensure the code compiles with `cargo check`
4. Pass all tests with `cargo test`

Start by reading the existing card implementations to understand the patterns.
"""

    # Use template sections if provided
    if prompt_template and prompt_template.get("sections"):
        sections = sorted(
            prompt_template["sections"],
            key=lambda s: s.get("order", 0)
        )
        parts = []
        for section in sections:
            pattern = section.get("pattern", "")
            content = pattern.replace("{task}", default_prompt)
            content = content.replace("{card_name}", card_name)
            content = content.replace("{card_set}", card_set)
            parts.append(content)
        return "\n\n".join(parts)

    return default_prompt


def _build_instance_prompt(
    instance: Dict[str, Any],
    system_prompt: str,
    architecture_guide: str,
    reference_snippets: str,
    hooks_documentation: str,
) -> str:
    """Build prompt from a full instance spec (matches localapi_engine_bench.py).

    This uses the same unified context engineering approach as the local runner.
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
    prompt = f"""{system_prompt}

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
"""

    if reference_snippets:
        prompt += f"\n---\n\n{reference_snippets}\n"

    if hooks_documentation:
        prompt += f"\n---\n\n{hooks_documentation}\n"

    prompt += f"""
---

## Final Instructions
1. READ the stub file at `{card_file}` using the `read` tool
2. Use the architecture guide and reference snippets above as patterns
3. USE THE `edit` OR `write` TOOL to modify `{card_file}` and replace the TODO stubs with working implementations
4. Run `cargo check` using the `bash` tool to verify compilation
5. Run `cargo test -- {instance_id.replace("-", "_")}` using the `bash` tool to run tests

CRITICAL: You MUST use the `edit` or `write` tool to actually modify the file. Reading the file is not enough - you must write code!
"""
    return prompt


def run_agent(
    prompt: str,
    workspace_path: Path,
    inference_url: str,
    timeout_s: int,
    agent_type: str = "opencode",
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """Run the coding agent on the task.

    Args:
        prompt: The task prompt
        workspace_path: Path to the workspace
        inference_url: URL for LLM calls (must be used for all API calls)
        timeout_s: Timeout in seconds
        agent_type: Type of agent (opencode, codex, claude)

    Returns:
        Dict with agent output and metadata
    """
    start_time = time.time()

    # Set up environment for agent
    env = os.environ.copy()

    if not inference_url:
        return {"success": False, "error": "inference_url is required for agent execution"}

    # CRITICAL: Route all LLM calls through inference_url
    base_url, correlation_id = normalize_interceptor_base(inference_url)
    if correlation_id:
        base_url = f"{base_url}/{correlation_id}"

    # Set OpenAI-compatible env vars to use interceptor
    env["OPENAI_BASE_URL"] = base_url
    env["OPENAI_API_BASE"] = base_url

    # For OpenCode
    env["OPENCODE_API_BASE"] = base_url

    # Resolve API key (prefer Synth key when routing via interceptor)
    interceptor_key = env.get("SYNTH_API_KEY") or env.get("INTERCEPTOR_API_KEY")
    api_key = resolve_interceptor_api_key(
        inference_url=inference_url,
        interceptor_key=interceptor_key,
        openai_key=env.get("OPENAI_API_KEY"),
    )
    if not api_key:
        return {"success": False, "error": "No API key available for interceptor routing"}
    env["OPENAI_API_KEY"] = api_key

    # Run the agent
    try:
        if agent_type == "opencode":
            result = _run_opencode(prompt, workspace_path, env, timeout_s, base_url, api_key, model)
        elif agent_type == "codex":
            result = _run_codex(prompt, workspace_path, env, timeout_s, base_url, api_key, model)
        elif agent_type == "claude_code":
            result = _run_claude_code(prompt, workspace_path, env, timeout_s, base_url, api_key, model)
        else:
            result = {"success": False, "error": f"Unknown agent type: {agent_type}"}
    except subprocess.TimeoutExpired:
        result = {"success": False, "error": f"Agent timed out after {timeout_s}s"}
    except Exception as e:
        result = {"success": False, "error": str(e)}

    result["duration_s"] = time.time() - start_time
    return result


def _run_opencode(
    prompt: str,
    workspace_path: Path,
    env: Dict,
    timeout_s: int,
    base_url: str,
    api_key: str,
    model: str,
) -> Dict:
    """Run OpenCode agent."""
    # Write opencode.json to enforce interceptor routing
    model_id = model.split("/", 1)[1] if "/" in model else model
    model_with_provider = f"openai/{model_id}"
    opencode_config = {
        "$schema": "https://opencode.ai/config.json",
        "model": model_with_provider,
        "provider": {
            "openai": {
                "name": "OpenAI",
                "npm": "@ai-sdk/openai",
                "options": {
                    "apiKey": api_key,
                    "baseURL": base_url or "https://api.openai.com/v1",
                },
            }
        },
        "permission": {"*": "allow"},
    }
    config_path = workspace_path / "opencode.json"
    config_path.write_text(json.dumps(opencode_config, indent=2))
    
    # Set OPENCODE_CONFIG_CONTENT for highest priority (inline config)
    # This ensures OpenCode uses our config regardless of other config sources
    env["OPENCODE_CONFIG_CONTENT"] = json.dumps(opencode_config)
    env["OPENCODE_CONFIG"] = str(config_path)

    # OpenCode CLI: use 'run' subcommand with message directly
    cmd = [
        "opencode",
        "run",
        prompt,
    ]

    proc = subprocess.run(
        cmd,
        cwd=workspace_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )

    return {
        "success": proc.returncode == 0,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "exit_code": proc.returncode,
    }


def _run_codex(
    prompt: str,
    workspace_path: Path,
    env: Dict,
    timeout_s: int,
    base_url: str,
    api_key: str,
    model: str,
) -> Dict:
    """Run Codex agent."""
    config_dir = Path.home() / ".codex"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.toml"
    # Use wire_api = "chat" for chat completions API (instead of "responses")
    # This uses the standard /v1/chat/completions endpoint
    config_content = f"""# Auto-generated for Harbor runs

model = "{model}"
model_provider = "openai"

[model_providers.openai]
name = "OpenAI"
base_url = "{base_url or "https://api.openai.com/v1"}"
wire_api = "chat"
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

    env["OPENAI_API_KEY"] = api_key
    env["OPENAI_MODEL"] = model
    if base_url:
        env["OPENAI_BASE_URL"] = base_url

    proc = subprocess.run(
        cmd,
        cwd=workspace_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )

    return {
        "success": proc.returncode == 0,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "exit_code": proc.returncode,
    }


def _run_claude_code(
    prompt: str,
    workspace_path: Path,
    env: Dict,
    timeout_s: int,
    base_url: str,
    api_key: str,
    model: str,
) -> Dict:
    """Run Claude Code agent."""
    claude_bin = os.environ.get("CLAUDE_BIN") or shutil.which("claude")
    if not claude_bin:
        return {
            "success": False,
            "stderr": "claude binary not found. Install Claude Code or set CLAUDE_BIN.",
            "stdout": "",
        }

    # Claude Code uses ANTHROPIC_BASE_URL and ANTHROPIC_AUTH_TOKEN
    if base_url:
        env["ANTHROPIC_BASE_URL"] = base_url
        env["ANTHROPIC_AUTH_TOKEN"] = api_key

    cmd = [
        claude_bin,
        "--print",
        "--model",
        model,
        "--dangerously-skip-permissions",
        prompt,
    ]

    proc = subprocess.run(
        cmd,
        cwd=workspace_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )

    return {
        "success": proc.returncode == 0,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "exit_code": proc.returncode,
    }


def evaluate_result(workspace_path: Path) -> Dict[str, Any]:
    """Evaluate the agent's result.

    Runs cargo check and cargo test to determine success.
    """
    metrics = {
        "compilation": False,
        "tests_passed": 0,
        "tests_total": 0,
        "reward_mean": 0.0,
    }

    # Run cargo check
    try:
        check_result = subprocess.run(
            ["cargo", "check"],
            cwd=workspace_path,
            capture_output=True,
            text=True,
            timeout=120,
        )
        metrics["compilation"] = check_result.returncode == 0
        metrics["check_stderr"] = check_result.stderr[:1000] if check_result.stderr else ""
    except Exception as e:
        metrics["check_error"] = str(e)

    # Run cargo test
    try:
        test_result = subprocess.run(
            ["cargo", "test", "--", "--test-threads=1"],
            cwd=workspace_path,
            capture_output=True,
            text=True,
            timeout=300,
        )

        # Parse test output
        output = test_result.stdout + test_result.stderr

        # Look for test summary line: "test result: ok. X passed; Y failed; Z ignored"
        import re
        match = re.search(r"(\d+) passed.*?(\d+) failed", output)
        if match:
            passed = int(match.group(1))
            failed = int(match.group(2))
            total = passed + failed
            metrics["tests_passed"] = passed
            metrics["tests_total"] = total

        metrics["test_output"] = output[:2000]
    except Exception as e:
        metrics["test_error"] = str(e)

    # Calculate reward_mean
    if metrics["compilation"]:
        # Base reward for compilation
        reward = 0.3

        # Additional reward for passing tests
        if metrics["tests_total"] > 0:
            test_ratio = metrics["tests_passed"] / metrics["tests_total"]
            reward += 0.7 * test_ratio

        metrics["reward_mean"] = reward
    else:
        metrics["reward_mean"] = 0.0

    return metrics


def _extract_context_overrides(prompt_template: Dict[str, Any]) -> Dict[str, Any]:
    """Extract context_overrides from prompt_template.

    The Harbor backend stores context_overrides inside prompt_template.
    GEPA sends them as a list of override objects with file_artifacts.
    We also support the legacy dict format (context_override with direct keys).
    """
    overrides: Dict[str, Any] = {}

    # Check for context_overrides list (new format from GEPA)
    ctx_overrides = prompt_template.get("context_overrides")
    if ctx_overrides and isinstance(ctx_overrides, list):
        for override in ctx_overrides:
            file_artifacts = override.get("file_artifacts") or []
            for artifact in file_artifacts:
                artifact_type = artifact.get("type") or artifact.get("artifact_type", "")
                content = artifact.get("content", "")
                if artifact_type == "system_prompt" and content:
                    overrides["system_prompt"] = content
                elif artifact_type == "architecture_guide" and content:
                    overrides["architecture_guide"] = content
                elif artifact_type == "reference_snippets" and content:
                    overrides["reference_snippets"] = content
                elif artifact_type == "hooks_documentation" and content:
                    overrides["hooks_documentation"] = content

    # Check for legacy context_override dict (direct keys)
    ctx_override = prompt_template.get("context_override")
    if ctx_override and isinstance(ctx_override, dict):
        for key in ("system_prompt", "architecture_guide", "reference_snippets", "hooks_documentation"):
            if key in ctx_override and ctx_override[key]:
                overrides.setdefault(key, ctx_override[key])

    # Check for direct keys on prompt_template itself
    for key in ("system_prompt", "architecture_guide", "reference_snippets", "hooks_documentation"):
        if key in prompt_template and prompt_template[key]:
            overrides.setdefault(key, prompt_template[key])

    return overrides


def run_rollout(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single rollout.

    Args:
        input_data: Rollout input payload

    Returns:
        Rollout result payload
    """
    rollout_input = RolloutInput.from_dict(input_data)

    # Extract context overrides from prompt_template (Harbor backend stores them there)
    context_overrides = _extract_context_overrides(rollout_input.prompt_template)

    # Get task for this seed
    task = get_task_for_seed(rollout_input.seed)

    # Create workspace
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace_path = Path(tmpdir) / "workspace"

        try:
            # Setup workspace
            repo_path = setup_workspace(task, workspace_path)

            # Build prompt with context overrides from GEPA
            prompt = build_prompt(task, rollout_input.prompt_template, context_overrides)

            # Get agent type/model from params or prompt template
            agent_type = rollout_input.params.get("agent") or rollout_input.params.get("agent_type") or "opencode"
            model = rollout_input.params.get("model") or rollout_input.prompt_template.get("model") or "gpt-4o-mini"

            # Run agent
            agent_result = run_agent(
                prompt=prompt,
                workspace_path=repo_path,
                inference_url=rollout_input.inference_url,
                timeout_s=rollout_input.limits.get("timeout_s", 300),
                agent_type=agent_type,
                model=model,
            )

            # Evaluate result
            eval_metrics = evaluate_result(repo_path)

            # Build result with debugging info
            task_id = task.get("id", "unknown")
            task_card = task.get("card") or task.get("cards", [{}])[0].get("name", "unknown")
            details = {
                "task_id": task_id,
                "card": task_card,
                "compilation": eval_metrics.get("compilation", False),
                "tests_passed": eval_metrics.get("tests_passed", 0),
                "tests_total": eval_metrics.get("tests_total", 0),
                "agent_duration_s": agent_result.get("duration_s", 0),
            }

            # Include agent output for debugging (truncated)
            if agent_result.get("stdout"):
                details["agent_stdout"] = agent_result["stdout"][:8000]
            if agent_result.get("stderr"):
                details["agent_stderr"] = agent_result["stderr"][:8000]
            if agent_result.get("exit_code") is not None:
                details["agent_exit_code"] = agent_result["exit_code"]

            # Build error message if agent failed
            error_msg = agent_result.get("error")
            if not agent_result.get("success") and not error_msg:
                error_msg = f"Agent exited with code {agent_result.get('exit_code', 'unknown')}"

            result = RolloutResult(
                trace_correlation_id=rollout_input.trace_correlation_id,
                metrics={
                    "reward_mean": eval_metrics["reward_mean"],
                    "details": details,
                },
                success=eval_metrics["reward_mean"] > 0,
                error=error_msg if not agent_result.get("success") else None,
            )

        except Exception as e:
            result = RolloutResult(
                trace_correlation_id=rollout_input.trace_correlation_id,
                metrics={"reward_mean": 0.0, "details": {"error": str(e)}},
                success=False,
                error=str(e),
            )

    return result.to_dict()


def main():
    parser = argparse.ArgumentParser(description="Harbor Runner for EngineBench")
    parser.add_argument("--input", "-i", type=str, help="Input JSON file path")
    parser.add_argument("--output", "-o", type=str, help="Output JSON file path")
    parser.add_argument("--stdio", action="store_true", help="Use stdin/stdout mode")
    args = parser.parse_args()

    # Read input
    if args.stdio:
        input_data = json.load(sys.stdin)
    elif args.input:
        with open(args.input) as f:
            input_data = json.load(f)
    else:
        # Default to file mode
        input_path = Path("/tmp/rollout.json")
        if not input_path.exists():
            print(json.dumps({"error": "No input file found at /tmp/rollout.json"}))
            sys.exit(1)
        with open(input_path) as f:
            input_data = json.load(f)

    # Run rollout
    result = run_rollout(input_data)

    # Write output
    if args.stdio:
        print(json.dumps(result))
    elif args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
    else:
        # Default to file mode
        with open("/tmp/result.json", "w") as f:
            json.dump(result, f, indent=2)
        # Also print to stdout
        print(json.dumps(result))


if __name__ == "__main__":
    main()
