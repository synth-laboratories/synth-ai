"""Helper for tracking experiment status during execution."""

from __future__ import annotations

import contextlib
import json
import re
import time
from pathlib import Path
from typing import Any

from .config_utils import _load_toml
from .progress_info import ProgressInfo
from .status import ExperimentStatusTracker
from .validation import validate_path


def extract_config_info(config_path: Path) -> tuple[str | None, str | None]:
    """Extract policy and environment from GEPA config file.
    
    Args:
        config_path: Path to TOML config file
        
    Returns:
        Tuple of (policy_model, environment_name) or (None, None) if not found
        
    Raises:
        AssertionError: If config_path is invalid
    """
    # Validate input
    assert config_path is not None, "config_path cannot be None"
    path = validate_path(config_path, "config_path", must_exist=True)
    
    try:
        config = _load_toml(path)
        assert isinstance(config, dict), (
            f"Config must be dict, got {type(config).__name__}"
        )
        
        pl_config = config.get("prompt_learning", {})
        assert isinstance(pl_config, dict), (
            f"prompt_learning section must be dict, got {type(pl_config).__name__}"
        )
        
        # Extract environment name from gepa.env_name
        gepa_config = pl_config.get("gepa", {})
        assert isinstance(gepa_config, dict), (
            f"gepa section must be dict, got {type(gepa_config).__name__}"
        )
        environment = gepa_config.get("env_name")
        if environment is not None:
            assert isinstance(environment, str), (
                f"env_name must be str, got {type(environment).__name__}: {environment}"
            )
        
        # Extract policy model - try gepa.policy first, then policy
        policy = None
        gepa_policy_config = gepa_config.get("policy", {})
        if gepa_policy_config and isinstance(gepa_policy_config, dict):
            policy = gepa_policy_config.get("model")
            if policy is not None:
                assert isinstance(policy, str), (
                    f"policy.model must be str, got {type(policy).__name__}: {policy}"
                )
        
        if not policy:
            policy_config = pl_config.get("policy", {})
            if isinstance(policy_config, dict):
                policy = policy_config.get("model")
                if policy is not None:
                    assert isinstance(policy, str), (
                        f"policy.model must be str, got {type(policy).__name__}: {policy}"
                    )
        
        return (policy, environment)
    except AssertionError:
        raise
    except Exception:
        return (None, None)


def parse_progress_from_output(output: str) -> ProgressInfo:
    """Parse progress information from subprocess stdout/stderr.
    
    Looks for JSON events like:
    - prompt.learning.progress events with rollouts_completed
    - gepa.start events
    - gepa.complete events
    
    Also parses text patterns from backend logs and CLI output:
    - Backend log: "GEPA progress: overall=X% rollouts=(Y/Z rem=W)..."
    - CLI format: "Progress: rollouts=X/Y (Z%) elapsed=Xs eta=Ys"
    
    Args:
        output: Combined stdout/stderr from training command
        
    Returns:
        ProgressInfo instance with parsed progress data
        
    Raises:
        AssertionError: If output is invalid or parsed data is inconsistent
    """
    # Validate input
    assert isinstance(output, str), (
        f"output must be str, got {type(output).__name__}"
    )
    
    progress_data: dict[str, Any] = {
        "rollouts_completed": None,
        "total_rollouts": None,
        "best_score": None,
        "trials_completed": None,
    }
    
    # Look for JSON lines (events are often logged as JSON)
    json_lines = []
    for line in output.split("\n"):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            with contextlib.suppress(json.JSONDecodeError):
                json_lines.append(json.loads(line))
    
    # Also look for progress patterns in text output
    # Backend log format: "GEPA progress: overall=X% rollouts=(Y/Z rem=W) transforms=(A/B) elapsed=Xs eta=Ys"
    # CLI format: "Progress: rollouts=X/Y (Z%) elapsed=Xs eta=Ys"
    rollout_patterns = [
        # Backend log format: rollouts=(15/200 rem=185)
        r"rollouts=\((\d+)/(\d+)\s+rem=\d+\)",
        # CLI format: rollouts=15/200 or rollouts=15/200 (7.5%)
        r"rollouts=(\d+)/(\d+)",
        # Generic: "15/200 rollouts" or "Completed 15/200 rollouts"
        r"Completed\s+(\d+)\s*/\s*(\d+)\s+rollouts?",
        r"(\d+)\s*/\s*(\d+)\s+rollouts?",
        # Just completed count: "rollouts_completed: 15"
        r"rollouts_completed[:\s]+(\d+)",
    ]
    
    for pattern in rollout_patterns:
        matches = re.finditer(pattern, output, re.IGNORECASE)
        for match in matches:
            if len(match.groups()) == 2:
                # Two groups: completed/total
                try:
                    completed = int(match.group(1))
                    total = int(match.group(2))
                    assert completed >= 0, f"rollouts_completed must be >= 0, got {completed}"
                    assert total > 0, f"total_rollouts must be > 0, got {total}"
                    assert completed <= total, (
                        f"rollouts_completed ({completed}) > total_rollouts ({total})"
                    )
                    progress_data["rollouts_completed"] = completed
                    progress_data["total_rollouts"] = total
                except (ValueError, AssertionError):
                    # Skip invalid matches
                    continue
            elif len(match.groups()) == 1:
                # One group: just completed
                try:
                    completed = int(match.group(1))
                    assert completed >= 0, f"rollouts_completed must be >= 0, got {completed}"
                    progress_data["rollouts_completed"] = completed
                except (ValueError, AssertionError):
                    # Skip invalid matches
                    continue
    
    # Extract from JSON events
    for event in json_lines:
        event_type = event.get("type", "")
        event_data = event.get("data", {})
        
        assert isinstance(event_data, dict), (
            f"event.data must be dict, got {type(event_data).__name__}"
        )
        
        if event_type == "prompt.learning.progress":
            if "rollouts_completed" in event_data:
                val = event_data["rollouts_completed"]
                if isinstance(val, int | float):
                    val_int = int(val)
                    assert val_int >= 0, f"rollouts_completed must be >= 0, got {val_int}"
                    progress_data["rollouts_completed"] = val_int
            if "rollouts_total" in event_data:
                val = event_data["rollouts_total"]
                if isinstance(val, int | float):
                    val_int = int(val)
                    assert val_int > 0, f"rollouts_total must be > 0, got {val_int}"
                    progress_data["total_rollouts"] = val_int
            elif "total_rollouts" in event_data:
                val = event_data["total_rollouts"]
                if isinstance(val, int | float):
                    val_int = int(val)
                    assert val_int > 0, f"total_rollouts must be > 0, got {val_int}"
                    progress_data["total_rollouts"] = val_int
            if "best_score" in event_data:
                val = event_data["best_score"]
                if isinstance(val, int | float):
                    val_float = float(val)
                    assert 0 <= val_float <= 1, f"best_score must be in [0, 1], got {val_float}"
                    progress_data["best_score"] = val_float
            if "trials_completed" in event_data:
                val = event_data["trials_completed"]
                if isinstance(val, int | float):
                    val_int = int(val)
                    assert val_int >= 0, f"trials_completed must be >= 0, got {val_int}"
                    progress_data["trials_completed"] = val_int
        
        # Look for best score in other events too
        if "best_score" in event_data and progress_data["best_score"] is None:
            val = event_data["best_score"]
            if isinstance(val, int | float):
                val_float = float(val)
                assert 0 <= val_float <= 1, f"best_score must be in [0, 1], got {val_float}"
                progress_data["best_score"] = val_float
        
        # Also check gepa.start event for rollout budget
        if event_type == "prompt.learning.gepa.start" and "rollout_budget" in event_data:
                val = event_data["rollout_budget"]
                if isinstance(val, int | float):
                    val_int = int(val)
                    assert val_int > 0, f"rollout_budget must be > 0, got {val_int}"
                    progress_data["total_rollouts"] = val_int
    
    # Look for best score in text output
    if progress_data["best_score"] is None:
        score_patterns = [
            r"best[_\s]score[:\s]+([\d.]+)",
            r"best[:\s]+([\d.]+)",
            # Backend log format: "Best: 0.85" or "(Best: 0.85)"
            r"\(?Best[:\s]+([\d.]+)\)?",
        ]
        for pattern in score_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    assert 0 <= score <= 1, f"best_score must be in [0, 1], got {score}"
                    progress_data["best_score"] = score
                    break
                except (ValueError, AssertionError):
                    pass
    
    # Also try to extract rollout budget from config if we see gepa.start but no total yet
    if progress_data["total_rollouts"] is None:
        # Look for "rollout_budget" or "rollout_limit" in output
        budget_patterns = [
            r"rollout[_\s]budget[:\s]+(\d+)",
            r"rollout[_\s]limit[:\s]+(\d+)",
        ]
        for pattern in budget_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                try:
                    total = int(match.group(1))
                    assert total > 0, f"total_rollouts must be > 0, got {total}"
                    progress_data["total_rollouts"] = total
                    break
                except (ValueError, AssertionError):
                    pass
    
    # Validate consistency before creating ProgressInfo
    if (
        progress_data["rollouts_completed"] is not None
        and progress_data["total_rollouts"] is not None
    ):
        assert progress_data["rollouts_completed"] <= progress_data["total_rollouts"], (
            f"rollouts_completed ({progress_data['rollouts_completed']}) > "
            f"total_rollouts ({progress_data['total_rollouts']})"
        )
    
    # Create and return ProgressInfo (will validate in __post_init__)
    return ProgressInfo.from_dict(progress_data)


def update_status_from_output(
    tracker: ExperimentStatusTracker,
    output: str,
    policy: str | None = None,
    environment: str | None = None,
    start_time: float | None = None,
) -> None:
    """Update experiment status based on parsed output.
    
    Always updates status_json to show at least elapsed time, even if no progress
    data is found. This ensures status is visible even for fast jobs that complete
    before progress events are emitted.
    
    Args:
        tracker: ExperimentStatusTracker instance
        output: Subprocess output to parse
        policy: Policy model (from config)
        environment: Environment name (from config)
        start_time: Job start time for ETA calculation
        
    Raises:
        AssertionError: If inputs are invalid or progress data is inconsistent
    """
    # Validate inputs
    assert tracker is not None, "tracker cannot be None"
    assert isinstance(output, str), (
        f"output must be str, got {type(output).__name__}"
    )
    if start_time is not None:
        assert isinstance(start_time, int | float), (
            f"start_time must be int | float, got {type(start_time).__name__}"
        )
        assert start_time > 0, f"start_time must be > 0, got {start_time}"
    
    progress = parse_progress_from_output(output)
    assert isinstance(progress, ProgressInfo), (
        f"parse_progress_from_output must return ProgressInfo, got {type(progress).__name__}"
    )
    
    # Calculate ETA and rollouts/min if we have progress info
    eta_seconds = None
    rollouts_per_minute = None
    if (
        progress.rollouts_completed is not None
        and progress.total_rollouts is not None
        and progress.total_rollouts > 0
        and start_time is not None
    ):
        completed = progress.rollouts_completed
        total = progress.total_rollouts
        assert completed > 0, "rollouts_completed must be > 0 to calculate ETA"
        elapsed = time.time() - start_time
        assert elapsed > 0, f"elapsed time must be > 0, got {elapsed}"
        rate = completed / elapsed  # rollouts per second
        rollouts_per_minute = rate * 60.0  # Convert to rollouts per minute
        remaining = total - completed
        assert remaining >= 0, f"remaining rollouts must be >= 0, got {remaining}"
        if rate > 0:
            eta_seconds = remaining / rate
            assert eta_seconds >= 0, f"eta_seconds must be >= 0, got {eta_seconds}"
    
    # Calculate progress percentage
    progress_pct = None
    if (
        progress.rollouts_completed is not None
        and progress.total_rollouts is not None
        and progress.total_rollouts > 0
    ):
        progress_pct = (progress.rollouts_completed / progress.total_rollouts) * 100
        assert 0 <= progress_pct <= 100, (
            f"progress_pct must be in [0, 100], got {progress_pct}"
        )
    
    # Always update status_json, even if we only have elapsed time
    # This ensures status is visible even for fast jobs
    tracker.update(
        policy=policy,
        environment=environment,
        rollouts_completed=progress.rollouts_completed,
        total_rollouts=progress.total_rollouts,
        trials_completed=progress.trials_completed,
        best_score=progress.best_score,
        eta_seconds=eta_seconds,
        progress_pct=progress_pct,
        rollouts_per_minute=rollouts_per_minute,
    )

