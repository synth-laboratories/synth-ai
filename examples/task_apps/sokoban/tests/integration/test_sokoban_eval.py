"""Integration tests for Sokoban task app with evaluation."""
import pytest

requests = pytest.importorskip("requests")

# sokoban_server fixture is in conftest.py


def test_sokoban_server_health(sokoban_server: str) -> None:
    """Test that the Sokoban server health endpoint works."""
    resp = requests.get(f"{sokoban_server}/health", timeout=5.0)
    assert resp.status_code in (200, 400), f"Unexpected status: {resp.status_code}"


def test_sokoban_task_info(sokoban_server: str) -> None:
    """Test that the Sokoban server returns valid task_info."""
    resp = requests.get(f"{sokoban_server}/task_info", timeout=5.0)
    assert resp.status_code == 200
    data = resp.json()
    assert "task" in data
    assert data["task"]["id"] == "sokoban"


def test_sokoban_manual_rollout(sokoban_server: str) -> None:
    """Test a manual Sokoban rollout with explicit actions."""
    # Try explicit action rollout (no LLM required)
    rollout_payload = {
        "run_id": "test_manual",
        "env": {"seed": 0, "config": {"difficulty": "easy", "max_steps": 50}},
        "ops": [0, 2, 2, 3],  # left, right, right, down
        "policy": {"config": {"provider": "noop"}},
    }
    
    resp = requests.post(
        f"{sokoban_server}/rollout",
        json=rollout_payload,
        headers={"Authorization": "Bearer sk_env_test"},
        timeout=30.0,
    )
    
    assert resp.status_code == 200
    data = resp.json()
    assert "trajectories" in data
    assert len(data["trajectories"]) > 0
    assert "metrics" in data

