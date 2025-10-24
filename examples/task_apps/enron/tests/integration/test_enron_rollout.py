"""Integration test for Enron rollouts via /rollout endpoint."""
import os
import pytest

requests = pytest.importorskip("requests")

# Use the actual ENVIRONMENT_API_KEY from .env
AUTH_HEADER = {"Authorization": "Bearer sk_env_30c78a787bac223c716918181209f263"}


@pytest.mark.slow
def test_enron_manual_rollout(enron_server: str) -> None:
    """Test a manual Enron rollout with explicit search/read/answer actions."""
    rollout_payload = {
        "run_id": "test_manual_enron",
        "env": {"seed": 0},
        "ops": [
            {
                "tool": "search_emails",
                "args": {
                    "inbox": "test@enron.com",
                    "keywords": ["test", "question"],
                    "max_results": 5,
                },
            },
            {
                "tool": "answer_question",
                "args": {"answer": "This is a test answer"},
            },
        ],
        "policy": {
            "policy_name": "manual",
            "config": {"provider": "noop"},
        },
    }
    
    resp = requests.post(
        f"{enron_server}/rollout",
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
    
    trajectory = data["trajectories"][0]
    assert "steps" in trajectory
    
    # Should have at least initial observation
    assert len(trajectory["steps"]) > 0


@pytest.mark.slow
def test_enron_policy_rollout(enron_server: str) -> None:
    """Test an Enron rollout using Groq policy."""
    if "GROQ_API_KEY" not in os.environ:
        pytest.skip("GROQ_API_KEY required for this test")
    
    rollout_payload = {
        "run_id": "test_policy_enron",
        "env": {"seed": 0},
        "ops": [],  # Empty ops means use policy
        "policy": {
            "policy_name": "qwen-groq",
            "config": {
                "provider": "groq",
                "model": "qwen/qwen3-32b",
                "temperature": 0.2,
                "max_tokens": 1024,
            },
        },
    }
    
    resp = requests.post(
        f"{enron_server}/rollout",
        json=rollout_payload,
        headers=AUTH_HEADER,
        timeout=180.0,  # Enron can be slow with multiple tool calls
    )
    
    assert resp.status_code == 200, f"Rollout failed: {resp.status_code} {resp.text}"
    data = resp.json()
    
    # Verify response structure
    assert "trajectories" in data
    assert "metrics" in data
    assert "trace" in data
    
    trajectory = data["trajectories"][0]
    assert "steps" in trajectory
    
    # Check that steps were taken
    assert len(trajectory["steps"]) > 0
    
    # Verify metrics
    metrics = data["metrics"]
    assert "episode_returns" in metrics or "mean_return" in metrics
    
    # Check that we got some reward (could be negative for search penalty)
    if "episode_returns" in metrics and len(metrics["episode_returns"]) > 0:
        # Just verify it's a number
        assert isinstance(metrics["episode_returns"][0], (int, float))


@pytest.mark.fast
def test_enron_rollout_with_auth(enron_server: str) -> None:
    """Test that Enron rollout requires proper authentication."""
    rollout_payload = {
        "run_id": "test_auth",
        "env": {"seed": 0},
        "ops": [],
        "policy": {"config": {"provider": "noop"}},
    }
    
    # Try without auth header
    resp = requests.post(
        f"{enron_server}/rollout",
        json=rollout_payload,
        timeout=10.0,
    )
    
    # Should fail without auth (400 or 401)
    assert resp.status_code in (400, 401, 403), f"Expected auth error, got {resp.status_code}"

