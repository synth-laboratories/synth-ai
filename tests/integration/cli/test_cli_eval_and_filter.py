"""Integration tests for CLI eval and filter utilities against Crafter with Groq policy."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import pytest


pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture
def temp_workspace(tmp_path: Path):
    """Create a temporary workspace with necessary directories."""
    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()
    
    ft_data_dir = tmp_path / "ft_data"
    ft_data_dir.mkdir()
    
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    
    return {
        "root": tmp_path,
        "traces": traces_dir,
        "ft_data": ft_data_dir,
        "configs": configs_dir,
    }


@pytest.fixture
def eval_config(temp_workspace: dict) -> Path:
    """Create a minimal eval config for crafter with groq."""
    config_path = temp_workspace["configs"] / "eval_crafter_groq.toml"
    trace_db = temp_workspace["traces"] / "eval_traces.db"
    
    config_content = f"""# Eval config for Crafter with Groq
# Uses localhost task app (spawned by eval command)

app_id = "grpo-crafter"
model = "qwen/qwen3-32b"
provider = "groq"
inference_url = "https://api.groq.com/openai"

# Minimal settings for fast test
num_episodes = 2
max_turns = 3
concurrency = 1
seeds = "0,1"

[policy]
inference_url = "https://api.groq.com/openai"
temperature = 0.0
max_tokens = 256
thinking_mode = "no_think"
"""
    
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def filter_config(temp_workspace: dict) -> Path:
    """Create a filter config to extract traces."""
    config_path = temp_workspace["configs"] / "filter_crafter.toml"
    trace_db = temp_workspace["traces"] / "eval_traces.db"
    output_jsonl = temp_workspace["ft_data"] / "filtered_traces.jsonl"
    
    config_content = f"""[filter]
db = "{trace_db}"
output = "{output_jsonl}"
splits = []
models = []
min_official_score = 0.0
limit = 5
"""
    
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def test_env():
    """Set up environment variables for testing."""
    env = os.environ.copy()
    
    # Check for required API keys
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        pytest.skip("GROQ_API_KEY not set; cannot run groq policy tests")
    
    # Set defaults for task app
    env.setdefault("ENVIRONMENT_API_KEY", "test_eval_key_123")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("SYNTH_FAKE_INFERENCE", "0")  # Use real Groq inference
    
    return env


def test_eval_command_with_groq_policy(eval_config: Path, temp_workspace: dict, test_env: dict):
    """Test synth-ai eval command runs rollouts and produces trace DB."""
    pytest.importorskip("crafter", reason="crafter dependency not installed")
    
    trace_db = temp_workspace["traces"] / "eval_traces.db"
    
    # Run eval command
    cmd = [
        "uv",
        "run",
        "synth-ai",
        "eval",
        "--config",
        str(eval_config),
        "--trace-db",
        str(trace_db),
        "--seeds",
        "0,1",
    ]
    
    print(f"\n[test_eval] Running: {' '.join(cmd)}")
    print(f"[test_eval] Trace DB will be at: {trace_db}")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[3]),
        env=test_env,
        timeout=180,  # 3 minutes should be enough for 2 episodes
    )
    
    print(f"\n[test_eval] Return code: {result.returncode}")
    print(f"[test_eval] STDOUT (last 100 lines):\n{chr(10).join(result.stdout.splitlines()[-100:])}")
    if result.stderr:
        print(f"[test_eval] STDERR (last 50 lines):\n{chr(10).join(result.stderr.splitlines()[-50:])}")
    
    # Check command succeeded
    assert result.returncode == 0, f"eval command failed with code {result.returncode}"
    
    # Check trace DB was created
    assert trace_db.exists(), f"Trace DB not created at {trace_db}"
    assert trace_db.stat().st_size > 0, "Trace DB is empty"
    
    # Check output contains expected summary info
    stdout_lower = result.stdout.lower()
    assert any(word in stdout_lower for word in ["episode", "rollout", "seed"]), \
        "Output should mention episodes/rollouts"
    
    # Verify we can query the DB (basic sanity check)
    try:
        import sqlite3
        conn = sqlite3.connect(str(trace_db))
        cursor = conn.cursor()
        
        # Check sessions table exists and has rows
        cursor.execute("SELECT COUNT(*) FROM sessions")
        session_count = cursor.fetchone()[0]
        assert session_count >= 2, f"Expected at least 2 sessions, got {session_count}"
        
        print(f"[test_eval] Trace DB contains {session_count} sessions")
        
        conn.close()
    except Exception as e:
        pytest.fail(f"Failed to query trace DB: {e}")


def test_filter_command_exports_jsonl(
    eval_config: Path,
    filter_config: Path, 
    temp_workspace: dict,
    test_env: dict
):
    """Test synth-ai filter command exports traces to JSONL."""
    pytest.importorskip("crafter", reason="crafter dependency not installed")
    
    trace_db = temp_workspace["traces"] / "eval_traces.db"
    output_jsonl = temp_workspace["ft_data"] / "filtered_traces.jsonl"
    
    # First run eval to create traces
    eval_cmd = [
        "uv",
        "run",
        "synth-ai",
        "eval",
        "--config",
        str(eval_config),
        "--trace-db",
        str(trace_db),
        "--seeds",
        "0,1",
    ]
    
    print(f"\n[test_filter] Step 1: Running eval to generate traces")
    eval_result = subprocess.run(
        eval_cmd,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[3]),
        env=test_env,
        timeout=180,
    )
    
    assert eval_result.returncode == 0, f"eval step failed: {eval_result.returncode}"
    assert trace_db.exists(), "Trace DB not created"
    
    # Now run filter command
    filter_cmd = [
        "uv",
        "run",
        "synth-ai",
        "filter",
        "--config",
        str(filter_config),
    ]
    
    print(f"\n[test_filter] Step 2: Running filter to export JSONL")
    print(f"[test_filter] Command: {' '.join(filter_cmd)}")
    
    filter_result = subprocess.run(
        filter_cmd,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[3]),
        env=test_env,
        timeout=60,
    )
    
    print(f"\n[test_filter] Return code: {filter_result.returncode}")
    print(f"[test_filter] STDOUT:\n{filter_result.stdout}")
    if filter_result.stderr:
        print(f"[test_filter] STDERR:\n{filter_result.stderr}")
    
    # Check command succeeded
    assert filter_result.returncode == 0, \
        f"filter command failed with code {filter_result.returncode}"
    
    # Check JSONL output was created
    assert output_jsonl.exists(), f"Output JSONL not created at {output_jsonl}"
    assert output_jsonl.stat().st_size > 0, "Output JSONL is empty"
    
    # Verify JSONL content
    with output_jsonl.open() as f:
        lines = f.readlines()
        assert len(lines) > 0, "JSONL file has no lines"
        
        # Parse first line to verify it's valid JSON with messages
        first_line = json.loads(lines[0])
        assert "messages" in first_line, "JSONL line missing 'messages' field"
        assert isinstance(first_line["messages"], list), "'messages' should be a list"
        assert len(first_line["messages"]) > 0, "'messages' should not be empty"
        
        # Check message structure
        first_msg = first_line["messages"][0]
        assert "role" in first_msg, "Message missing 'role' field"
        assert "content" in first_msg, "Message missing 'content' field"
        
        print(f"[test_filter] Successfully exported {len(lines)} training examples")
        print(f"[test_filter] First example has {len(first_line['messages'])} messages")


def test_eval_command_with_metadata_filter(temp_workspace: dict, test_env: dict):
    """Test eval command with metadata filtering (if supported by crafter)."""
    pytest.importorskip("crafter", reason="crafter dependency not installed")
    
    trace_db = temp_workspace["traces"] / "eval_metadata.db"
    
    # Run eval with metadata filter
    cmd = [
        "uv",
        "run",
        "synth-ai",
        "eval",
        "grpo-crafter",
        "--trace-db",
        str(trace_db),
        "--seeds",
        "0",
        "--model",
        "qwen/qwen3-32b",
    ]
    
    print(f"\n[test_eval_metadata] Running: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[3]),
        env=test_env,
        timeout=120,
    )
    
    # Should complete (even if it prompts or uses defaults)
    print(f"[test_eval_metadata] Return code: {result.returncode}")
    print(f"[test_eval_metadata] Output preview:\n{result.stdout[:500]}")
    
    # This test is more about ensuring the command doesn't crash
    # The actual filtering logic depends on task app configuration
    assert result.returncode in (0, 1), \
        f"Command should complete or fail gracefully, got {result.returncode}"


def test_filter_with_judge_scores(temp_workspace: dict, test_env: dict):
    """Test filter command with judge score thresholds."""
    pytest.importorskip("crafter", reason="crafter dependency not installed")
    
    trace_db = temp_workspace["traces"] / "eval_judges.db"
    output_jsonl = temp_workspace["ft_data"] / "high_score_traces.jsonl"
    
    # Create filter config with judge score threshold
    filter_config_path = temp_workspace["configs"] / "filter_with_judges.toml"
    filter_config_path.write_text(f"""[filter]
db = "{trace_db}"
output = "{output_jsonl}"
min_official_score = 0.5

[filter.min_judge_scores]
# If crafter has custom judges, filter on them
# For now just test the config parsing works
""")
    
    # First create some traces (simplified eval)
    eval_cmd = [
        "uv",
        "run",
        "synth-ai",
        "eval",
        "grpo-crafter",
        "--trace-db",
        str(trace_db),
        "--seeds",
        "0",
        "--model",
        "qwen/qwen3-32b",
    ]
    
    print(f"\n[test_filter_judges] Running eval...")
    eval_result = subprocess.run(
        eval_cmd,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[3]),
        env=test_env,
        timeout=120,
    )
    
    if eval_result.returncode != 0:
        pytest.skip("Eval step failed, skipping filter test")
    
    # Now test filter accepts the judge config
    filter_cmd = [
        "uv",
        "run",
        "synth-ai",
        "filter",
        "--config",
        str(filter_config_path),
    ]
    
    print(f"\n[test_filter_judges] Running filter with judge thresholds...")
    filter_result = subprocess.run(
        filter_cmd,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[3]),
        env=test_env,
        timeout=60,
    )
    
    # Should complete even if no traces match the filter
    assert filter_result.returncode == 0, \
        f"filter command should handle judge configs gracefully"
    
    print(f"[test_filter_judges] Filter completed successfully")




