"""Verilog backend judge that calls the Synth judge API with inline rubric."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, TypedDict

import httpx


class TraceMetadata(TypedDict, total=False):
    """Metadata for the trace."""
    env_id: str
    policy_id: str
    length: int


class JudgeTracePayload(TypedDict):
    """Trace payload sent to backend judge."""
    event_history: List[Dict[str, Any]]
    markov_blanket_message_history: List[Dict[str, Any]]
    metadata: TraceMetadata


class JudgeOptions(TypedDict, total=False):
    """Options for judge scoring."""
    model: str
    timeout_s: int
    event: bool
    outcome: bool


class TaskApp(TypedDict):
    """Task application metadata."""
    id: str


class JudgeScoreRequest(TypedDict):
    """Request to backend judge API."""
    policy_name: str
    task_app: TaskApp
    trace: JudgeTracePayload
    rubric: Dict[str, Any]
    options: JudgeOptions


# Load rubric from file (cached at module level)
_RUBRIC_PATH = Path(__file__).parent.parent / "rubrics" / "verilog_backend_judge.json"
_RUBRIC: Dict[str, Any] | None = None


def _load_rubric() -> Dict[str, Any]:
    """Load rubric from file with fallback to inline default."""
    global _RUBRIC
    if _RUBRIC is None:
        try:
            with open(_RUBRIC_PATH, 'r') as f:
                _RUBRIC = json.load(f)
            assert isinstance(_RUBRIC, dict), "Rubric must be a dict"
            assert "outcome" in _RUBRIC, "Rubric must have 'outcome' key"
            assert isinstance(_RUBRIC["outcome"], list), "Rubric 'outcome' must be a list"
        except Exception as e:
            print(f"[verilog_backend_judge] Warning: Failed to load rubric from {_RUBRIC_PATH}: {e}")
            # Fallback inline rubric (matching RubricCriteriaBlock format)
            _RUBRIC = {
                "event": [],
                "outcome": [
                    {"id": "correctness.tests_pass", "description": "Tests pass", "weight": 0.5, "scale": "bounded"},
                    {"id": "efficiency.code_quality", "description": "Code quality", "weight": 0.3, "scale": "bounded"},
                    {"id": "efficiency.solution_steps", "description": "Solution efficiency", "weight": 0.2, "scale": "bounded"}
                ]
            }
    return _RUBRIC


def judge(payload: Dict[str, Any], **kwargs: Any) -> float:
    """
    Call the Synth backend judge API to score a Verilog rollout.

    Args:
        payload: Dict with keys: seed, prompt, completion, metrics, response, trace
        **kwargs: Additional config (backend_url, model, timeout_s, etc.)

    Returns:
        float: Aggregate score from 0.0 to 1.0
    """
    try:
        # Extract configuration
        backend_url = kwargs.get("backend_url", "http://localhost:8000/api")
        model = kwargs.get("model", "openai/gpt-oss-120b")
        timeout = kwargs.get("timeout_s", 45)
        
        assert isinstance(backend_url, str), "backend_url must be a string"
        assert isinstance(model, str), "model must be a string"
        assert isinstance(timeout, (int, float)), "timeout_s must be numeric"
        
        # Extract trajectory from response
        response_data = payload.get("response", {})
        assert isinstance(response_data, dict), "response must be a dict"
        
        trajectories = response_data.get("trajectories", [])
        assert isinstance(trajectories, list), "trajectories must be a list"
        
        if not trajectories:
            print("[verilog_backend_judge] No trajectories in response")
            return 0.0
        
        trajectory = trajectories[0]  # First trajectory
        assert isinstance(trajectory, dict), "trajectory must be a dict"
        
        # Load rubric
        rubric = _load_rubric()
        
        # Transform trajectory into JudgeTracePayload format
        # The backend expects: event_history, markov_blanket_message_history, metadata
        steps = trajectory.get("steps", [])
        assert isinstance(steps, list), "trajectory steps must be a list"
        
        event_history: List[Dict[str, Any]] = []
        for idx, step in enumerate(steps):
            assert isinstance(step, dict), f"step {idx} must be a dict"
            # Each step becomes an event with obs, tool_calls, reward, done, info
            event_history.append({
                "observation": step.get("obs", {}),
                "tool_calls": step.get("tool_calls", []),
                "reward": step.get("reward", 0.0),
                "done": step.get("done", False),
                "truncated": step.get("truncated", False),
                "info": step.get("info", {}),
            })
        
        # Add final observation - backend will extract this as outcome context
        final_data = trajectory.get("final", {})
        if final_data:
            assert isinstance(final_data, dict), "final data must be a dict"
            final_obs = final_data.get("observation", {})
            assert isinstance(final_obs, dict), "final observation must be a dict"
            
            event_history.append({
                "observation": final_obs,
                "reward": final_data.get("reward", 0.0),
                "done": final_data.get("done", True),
                "truncated": final_data.get("truncated", False),
                "info": final_data.get("info", {}),
            })
        
        # Build trace metadata
        metadata: TraceMetadata = {
            "env_id": trajectory.get("env_id", "verilog"),
            "policy_id": trajectory.get("policy_id", "verilog-designer"),
            "length": trajectory.get("length", len(steps)),
        }
        
        # Build judge request with rubric included
        judge_request: JudgeScoreRequest = {
            "policy_name": "verilog-designer",
            "task_app": {"id": "grpo-verilog"},
            "trace": {
                "event_history": event_history,
                "markov_blanket_message_history": [],
                "metadata": metadata,
            },
            "rubric": rubric,
            "options": {
                "model": model,
                "timeout_s": timeout,
                "event": False,  # Not scoring per-event
                "outcome": True,  # Score the final outcome
            }
        }
        
        # Call backend judge API
        with httpx.Client(timeout=timeout) as client:
            # Get API key from env
            api_key = os.environ.get("SYNTH_API_KEY") or os.environ.get("OPENAI_API_KEY")
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            url = f"{backend_url.rstrip('/')}/judge/v1/score"
            
            # Debug: print request details
            print(f"\n[verilog_backend_judge] REQUEST DEBUG:")
            print(f"  URL: {url}")
            print(f"  Request body keys: {list(judge_request.keys())}")
            rubric_data = judge_request.get('rubric', {})
            print(f"  Rubric event criteria: {len(rubric_data.get('event', []))}")
            print(f"  Rubric outcome criteria: {len(rubric_data.get('outcome', []))}")
            trace_data = judge_request.get('trace', {})
            event_hist = trace_data.get('event_history', [])
            print(f"  Trace event_history count: {len(event_hist)}")
            if event_hist:
                last_event = event_hist[-1]
                last_obs = last_event.get('observation', {})
                print(f"  Last event done: {last_event.get('done', False)}")
                print(f"  Last obs keys: {list(last_obs.keys())}")
                print(f"  Task completed: {last_obs.get('task_completed', 'N/A')}")
            
            response = client.post(url, json=judge_request, headers=headers)
            
            # Debug: print response details
            print(f"\n[verilog_backend_judge] RESPONSE DEBUG:")
            print(f"  Status: {response.status_code}")
            print(f"  Response body: {response.text[:500]}")  # First 500 chars
            
            response.raise_for_status()
            result = response.json()
            assert isinstance(result, dict), "Response must be a dict"
            
            # Extract aggregate score
            aggregate_score = result.get("aggregate_score", 0.0)
            
            # Try outcome_review.total if aggregate_score not found
            if aggregate_score == 0.0 and "outcome_review" in result:
                outcome_review = result["outcome_review"]
                if isinstance(outcome_review, dict):
                    aggregate_score = outcome_review.get("total", 0.0)
            
            print(f"  Aggregate score: {aggregate_score}\n")
            return float(aggregate_score)
            
    except httpx.HTTPStatusError as e:
        print(f"\n[verilog_backend_judge] HTTP ERROR:")
        print(f"  Status: {e.response.status_code}")
        print(f"  Response body: {e.response.text}\n")
        return 0.0
    except AssertionError as e:
        print(f"[verilog_backend_judge] Assertion error: {e}")
        return 0.0
    except Exception as e:
        print(f"[verilog_backend_judge] Unexpected error: {e}")
        return 0.0
