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

# EngineBench task definitions (Pokemon TCG cards to implement)
TASKS = [
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
    """Get the task for a given seed (deterministic mapping)."""
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


def build_prompt(task: Dict[str, Any], prompt_template: Dict[str, Any]) -> str:
    """Build the agent prompt from the task and template."""
    card_name = task["card"]
    card_set = task["set"]

    # Default prompt if no template sections
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

    # Use template if provided
    if prompt_template and prompt_template.get("sections"):
        sections = sorted(
            prompt_template["sections"],
            key=lambda s: s.get("order", 0)
        )
        parts = []
        for section in sections:
            pattern = section.get("pattern", "")
            # Replace placeholders
            content = pattern.replace("{task}", default_prompt)
            content = content.replace("{card_name}", card_name)
            content = content.replace("{card_set}", card_set)
            parts.append(content)
        return "\n\n".join(parts)

    return default_prompt


def run_agent(
    prompt: str,
    workspace_path: Path,
    inference_url: str,
    timeout_s: int,
    agent_type: str = "opencode",
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

    # CRITICAL: Route all LLM calls through inference_url
    # Parse the inference URL to extract base and API key if present
    if "/v1/" in inference_url:
        base_url = inference_url.rsplit("/v1/", 1)[0] + "/v1"
    else:
        base_url = inference_url

    # Set OpenAI-compatible env vars to use interceptor
    env["OPENAI_BASE_URL"] = base_url
    env["OPENAI_API_BASE"] = base_url

    # For OpenCode
    env["OPENCODE_API_BASE"] = base_url

    # Run the agent
    try:
        if agent_type == "opencode":
            result = _run_opencode(prompt, workspace_path, env, timeout_s)
        elif agent_type == "codex":
            result = _run_codex(prompt, workspace_path, env, timeout_s)
        else:
            result = {"success": False, "error": f"Unknown agent type: {agent_type}"}
    except subprocess.TimeoutExpired:
        result = {"success": False, "error": f"Agent timed out after {timeout_s}s"}
    except Exception as e:
        result = {"success": False, "error": str(e)}

    result["duration_s"] = time.time() - start_time
    return result


def _run_opencode(prompt: str, workspace_path: Path, env: Dict, timeout_s: int) -> Dict:
    """Run OpenCode agent."""
    # Write prompt to file
    prompt_file = workspace_path / ".opencode_prompt.txt"
    prompt_file.write_text(prompt)

    cmd = [
        "opencode",
        "--prompt", str(prompt_file),
        "--non-interactive",
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


def _run_codex(prompt: str, workspace_path: Path, env: Dict, timeout_s: int) -> Dict:
    """Run Codex agent."""
    cmd = [
        "codex",
        "--approval-mode", "full-auto",
        "-m", prompt,
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


def run_rollout(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single rollout.

    Args:
        input_data: Rollout input payload

    Returns:
        Rollout result payload
    """
    rollout_input = RolloutInput.from_dict(input_data)

    # Get task for this seed
    task = get_task_for_seed(rollout_input.seed)

    # Create workspace
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace_path = Path(tmpdir) / "workspace"

        try:
            # Setup workspace
            repo_path = setup_workspace(task, workspace_path)

            # Build prompt
            prompt = build_prompt(task, rollout_input.prompt_template)

            # Get agent type from params
            agent_type = rollout_input.params.get("agent", "opencode")

            # Run agent
            agent_result = run_agent(
                prompt=prompt,
                workspace_path=repo_path,
                inference_url=rollout_input.inference_url,
                timeout_s=rollout_input.limits.get("timeout_s", 300),
                agent_type=agent_type,
            )

            # Evaluate result
            eval_metrics = evaluate_result(repo_path)

            # Build result
            result = RolloutResult(
                trace_correlation_id=rollout_input.trace_correlation_id,
                metrics={
                    "reward_mean": eval_metrics["reward_mean"],
                    "details": {
                        "task_id": task["id"],
                        "card": task["card"],
                        "compilation": eval_metrics.get("compilation", False),
                        "tests_passed": eval_metrics.get("tests_passed", 0),
                        "tests_total": eval_metrics.get("tests_total", 0),
                        "agent_duration_s": agent_result.get("duration_s", 0),
                    },
                },
                success=eval_metrics["reward_mean"] > 0,
                error=agent_result.get("error"),
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
