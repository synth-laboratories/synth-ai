"""Integration test for Verilog rollouts via /rollout endpoint."""
import os
import pytest

requests = pytest.importorskip("requests")

# Use the actual ENVIRONMENT_API_KEY from .env
AUTH_HEADER = {"Authorization": "Bearer sk_env_30c78a787bac223c716918181209f263"}


def test_verilog_manual_rollout(verilog_server: str) -> None:
    """Test a manual Verilog rollout with explicit write/compile/simulate/submit actions."""
    if "GROQ_API_KEY" not in os.environ:
        pytest.skip("GROQ_API_KEY required for this test")
    
    # Create a simple adder module
    verilog_code = """module adder(
    input [3:0] a,
    input [3:0] b,
    output [4:0] sum
);
    assign sum = a + b;
endmodule
"""
    
    rollout_payload = {
        "run_id": "test_manual_verilog",
        "env": {"seed": 0},
        "ops": [
            {"type": "write_file", "path": "adder.v", "content": verilog_code},
            {"type": "compile", "files": ["adder.v"]},
            {"type": "write_file", "path": "adder_tb.v", "content": "// Simple testbench"},
            {"type": "simulate", "testbench": "adder_tb.v"},
            {"type": "submit"},
        ],
        "policy": {
            "policy_name": "manual",
            "config": {"provider": "noop"},
        },
    }
    
    resp = requests.post(
        f"{verilog_server}/rollout",
        json=rollout_payload,
        headers=AUTH_HEADER,
        timeout=60.0,
    )
    
    assert resp.status_code == 200, f"Rollout failed: {resp.status_code} {resp.text}"
    data = resp.json()
    
    # Verify response structure
    assert "trajectories" in data
    assert len(data["trajectories"]) > 0
    assert "metrics" in data
    assert "trace" in data
    
    # Check that trace is present
    assert data["trace"] is not None
    assert "session_trace" in data["trace"]


def test_verilog_policy_rollout(verilog_server: str) -> None:
    """Test a Verilog rollout using Groq policy."""
    if "GROQ_API_KEY" not in os.environ:
        pytest.skip("GROQ_API_KEY required for this test")
    
    rollout_payload = {
        "run_id": "test_policy_verilog",
        "env": {"seed": 0},
        "ops": [],  # Empty ops means use policy for all steps
        "policy": {
            "policy_name": "qwen-groq",
            "config": {
                "provider": "groq",
                "model": "qwen/qwen3-32b",
                "max_steps": 5,  # Limit steps for test
            },
        },
    }
    
    resp = requests.post(
        f"{verilog_server}/rollout",
        json=rollout_payload,
        headers=AUTH_HEADER,
        timeout=120.0,
    )
    
    assert resp.status_code == 200, f"Rollout failed: {resp.status_code} {resp.text}"
    data = resp.json()
    
    # Verify response structure
    assert "trajectories" in data
    assert "metrics" in data
    assert "trace" in data
    
    trajectory = data["trajectories"][0]
    assert "steps" in trajectory
    
    # Check that at least one step was taken
    assert len(trajectory["steps"]) > 0
    
    # Verify metrics
    metrics = data["metrics"]
    assert "episode_returns" in metrics or "mean_return" in metrics

