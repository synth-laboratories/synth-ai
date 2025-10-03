from __future__ import annotations

import logging
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Request, status
import os
import time as _time
from pydantic import BaseModel

from .registry import registry

logger = logging.getLogger(__name__)

# --- Seeding utilities (robust, optional deps) ---
def _set_global_seed(seed_value: int) -> Dict[str, Any]:
    """Set global RNG seeds across common libraries; return details for logging/restoration.

    Returns a dict containing which libraries were seeded and prior states if obtainable.
    """
    seeded: Dict[str, Any] = {"seed": int(seed_value), "libs": []}
    try:
        import random as _random  # type: ignore
        _random.seed(seed_value)
        seeded["libs"].append("random")
    except Exception:
        pass
    try:
        import numpy as _np  # type: ignore
        _np.random.seed(seed_value)
        seeded["libs"].append("numpy")
    except Exception:
        pass
    try:
        import torch as _torch  # type: ignore
        if hasattr(_torch, "manual_seed"):
            _torch.manual_seed(seed_value)
            seeded["libs"].append("torch")
        # Make CUDA deterministic if present (best-effort)
        try:
            if getattr(_torch, "cuda", None) and _torch.cuda.is_available():
                _torch.cuda.manual_seed_all(seed_value)
                seeded.setdefault("cuda", True)
        except Exception:
            pass
        # CUDNN deterministic flags (optional)
        try:
            if getattr(_torch, "backends", None) and getattr(_torch.backends, "cudnn", None):
                _torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
                _torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception:
        pass
    return seeded

def _clear_seed_side_effects() -> None:
    """Best-effort cleanup to avoid global deterministic side-effects between requests."""
    # We cannot truly restore prior RNG states without capturing them; we just avoid
    # leaving aggressive deterministic flags enabled where it matters.
    try:
        import torch as _torch  # type: ignore
        try:
            if getattr(_torch, "backends", None) and getattr(_torch.backends, "cudnn", None):
                # Re-enable cudnn.benchmark default True only if it was True; safest is False -> leave as is.
                # We'll keep deterministic False to avoid global impact; benchmark left False for stability.
                _torch.backends.cudnn.deterministic = False  # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception:
        pass

router = APIRouter()


class RolloutEnvSpec(BaseModel):
    env_id: Optional[str] = None
    env_name: Optional[str] = None
    config: Dict[str, Any] = {}
    seed: Optional[int] = None


class RolloutPolicySpec(BaseModel):
    policy_id: Optional[str] = None
    policy_name: Optional[str] = None
    config: Dict[str, Any] = {}


class RolloutBranchConfig(BaseModel):
    branch_every_n_steps: int = 0
    branch_on_condition: Optional[str] = None
    max_branches: int = 0
    branch_policy: bool = False
    branch_env: bool = False


class RolloutRecordConfig(BaseModel):
    trajectories: bool = True
    logprobs: bool = False
    value: bool = False


class RolloutSafetyConfig(BaseModel):
    max_ops: int = 100000
    max_time_s: float = 3600.0


class RolloutRequest(BaseModel):
    run_id: str
    env: RolloutEnvSpec
    policy: RolloutPolicySpec
    ops: List[str]  # ["agent", "env", ...]
    record: RolloutRecordConfig = RolloutRecordConfig()
    on_done: str = "reset"  # "reset" | "terminate"
    branch: Optional[RolloutBranchConfig] = None
    safety: RolloutSafetyConfig = RolloutSafetyConfig()
    # Optional run/session context
    training_session_id: Optional[str] = None
    synth_base_url: Optional[str] = None


class RolloutStep(BaseModel):
    obs: Dict[str, Any]
    tool_calls: List[Dict[str, Any]]
    reward: Optional[float] = None
    done: bool = False
    truncated: Optional[bool] = None
    logprob: Optional[float] = None
    value: Optional[float] = None
    info: Optional[Dict[str, Any]] = None


class RolloutTrajectory(BaseModel):
    env_id: str
    policy_id: str
    steps: List[RolloutStep]
    final: Optional[Dict[str, Any]] = None
    length: int
    decision_samples: Optional[List[Dict[str, Any]]] = None


def compute_stepwise_reward(
    prev_achievements: Dict[str, bool],
    new_achievements: Dict[str, bool],
    decision_index: int,
    actions_summary: List[Dict[str, Any]],
    indicator_lambda: float,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, float]]:
    """Compute stepwise reward metadata given achievement states before/after a decision."""

    prev_map = prev_achievements or {}
    next_map = new_achievements or {}

    unlocked = [
        name
        for name, value in next_map.items()
        if value and not prev_map.get(name, False)
    ]
    indicator = 1 if unlocked else 0
    reward_value = float(indicator_lambda) * indicator

    stepwise_info = {
        "decision_index": decision_index,
        "indicator": indicator,
        "new_achievements": unlocked,
        "reward": reward_value,
    }
    decision_sample = {
        "decision_index": decision_index,
        "indicator": indicator,
        "r_i": reward_value,
        "actions": actions_summary,
    }
    stats = {
        "indicator": float(indicator),
        "reward": reward_value,
        "new_achievements_count": float(len(unlocked)),
    }
    return stepwise_info, decision_sample, stats


class RolloutMetrics(BaseModel):
    episode_returns: List[float]
    mean_return: float
    num_steps: int
    num_episodes: int = 0


class RolloutResponse(BaseModel):
    run_id: str
    trajectories: List[RolloutTrajectory]
    branches: Dict[str, List[str]] = {}
    metrics: RolloutMetrics
    aborted: bool = False
    ops_executed: int = 0
def _summarize_observation_for_storage(env_handle: Any, observation: Dict[str, Any]) -> Dict[str, Any]:
    """Return a compact dict for trajectory storage instead of the raw observation.

    - For Crafter, use the same summary used for the policy user prompt
    - For others, keep a minimal subset or plain text preview
    """
    # Try Crafter-specific formatter
    try:
        from .envs.crafter.environment import CrafterEnvironmentWrapper as _CrafterWrapper  # type: ignore
    except Exception:
        _CrafterWrapper = None  # type: ignore

    if _CrafterWrapper is not None and isinstance(getattr(env_handle, "env", None), _CrafterWrapper):
        try:
            from .envs.crafter.shared import format_observation as _fmt  # type: ignore
            text = _fmt(observation or {})
            return {"text": text}
        except Exception:
            pass

    # Generic fallback: extract a few small fields if present; avoid huge arrays
    try:
        inv = observation.get("inventory") if isinstance(observation, dict) else None
        ach = observation.get("achievements_status") if isinstance(observation, dict) else None
        pos = observation.get("player_position") if isinstance(observation, dict) else None
        health = None
        if isinstance(inv, dict):
            health = inv.get("health")
        summary = {
            "position": pos,
            "health": health,
            "inventory_keys": sorted([k for k, v in (inv or {}).items() if v])[:10] if isinstance(inv, dict) else None,
            "achievements_unlocked": sorted([k for k, v in (ach or {}).items() if v])[:10] if isinstance(ach, dict) else None,
        }
        return {"text": json.dumps(summary, ensure_ascii=False)}
    except Exception:
        pass

    # Last resort: plain string preview
    try:
        return {"text": str(observation)[:10000]}
    except Exception:
        return {"text": ""}



class RunAbortRequest(BaseModel):
    run_id: str


class RunAbortResponse(BaseModel):
    ok: bool
    run_id: str


class RunStatusResponse(BaseModel):
    run_id: str
    status: str
    started_at: datetime
    finished_at: Optional[datetime] = None


@router.post("/rollout", response_model=RolloutResponse)
async def execute_rollout(
    request: RolloutRequest,
    req: Request,
) -> RolloutResponse:
    """Execute a rollout with coordinated environment and policy steps."""
    # Simple API key auth for inbound rollout
    header_key = req.headers.get("x-api-key")
    env_key = os.getenv("ENVIRONMENT_API_KEY")
    dev_key = os.getenv("dev_environment_api_key")
    # Accept either ENVIRONMENT_API_KEY or dev_environment_api_key
    expected_keys = [k for k in (env_key, dev_key) if k]
    if not expected_keys:
        missing = []
        if not env_key:
            missing.append("ENVIRONMENT_API_KEY")
        if not dev_key:
            missing.append("dev_environment_api_key")
        msg = f"Auth not configured: missing {', '.join(missing)} in task service environment"
        logger.error(msg)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=msg)
    if not header_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key: X-API-Key header not provided",
        )
    if header_key not in expected_keys:
        # Do not leak secrets; include short prefix for diagnostics
        exp_src = env_key if env_key else (dev_key or "")
        exp_prefix = (exp_src[:7] + "…") if len(exp_src) >= 7 else "set"
        got_prefix = (header_key[:7] + "…") if len(header_key) >= 7 else "set"
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid API key: header does not match expected (got={got_prefix}, expected_prefix={exp_prefix})",
        )

    # Log contextual fields for traceability
    if request.training_session_id:
        logger.info(f"ROLL_OUT: training_session_id={request.training_session_id}")
    if request.synth_base_url:
        logger.info(f"ROLL_OUT: synth_base_url={request.synth_base_url}")

    # Log masked OpenAI API key presence for diagnostics
    try:
        _oa = os.getenv("OPENAI_API_KEY")
        if _oa:
            _pref = (_oa[:6] + "…") if len(_oa) >= 6 else "set"
            logger.info(f"ROLL_OUT: OPENAI_API_KEY present (prefix={_pref})")
        else:
            logger.warning("ROLL_OUT: OPENAI_API_KEY missing")
    except Exception:
        pass

    # Make synth_base_url available for outbound calls in this app
    try:
        task_app = req.app.state.task_app
        if request.synth_base_url:
            setattr(task_app, "synth_base_url", request.synth_base_url)
    except Exception:
        pass

    # Register run
    registry.register_run(request.run_id)

    # Track resources created during this rollout so we can guarantee cleanup
    created_env_id: str | None = None
    created_policy_id: str | None = None
    env_seed_used: int | None = None

    try:
        # Initialize deterministic seed early for the entire rollout
        seed_value: Optional[int] = None
        try:
            if request.env and request.env.seed is not None:
                seed_value = int(request.env.seed)
            else:
                # Derive a stable seed from run_id
                import hashlib as _hashlib  # local import to avoid global deps

                _digest = _hashlib.sha256(request.run_id.encode("utf-8")).hexdigest()
                # Use lower 32 bits to fit common RNG ranges
                seed_value = int(_digest[:8], 16)
        except Exception:
            # Fallback to time-based seed if anything goes wrong
            try:
                seed_value = int((_time.time_ns() // 1_000_000) % (2**31 - 1))
            except Exception:
                seed_value = 42

        _seed_info = _set_global_seed(int(seed_value))
        try:
            logger.info(
                "ROLL_OUT: RNG seeded seed=%s libs=%s",
                str(_seed_info.get("seed")),
                ",".join(_seed_info.get("libs", [])),
            )
        except Exception:
            pass
        # Resolve or create environment
        if request.env.env_id:
            env_handle = registry.get_env(request.env.env_id)
            if not env_handle:
                raise HTTPException(
                    status_code=404,
                    detail=f"Environment {request.env.env_id} not found",
                )
            env_id = request.env.env_id
        else:
            # Create new environment
            from .environment_routes import create_environment, EnvCreateRequest

            if not request.env.env_name:
                raise ValueError("FATAL: env_name is required - NO FALLBACKS!")

            # Propagate training_session_id via env config for downstream usage
            _env_config = dict(request.env.config or {})
            if request.training_session_id is not None:
                _env_config.setdefault(
                    "training_session_id", request.training_session_id
                )
            env_response = await create_environment(
                EnvCreateRequest(
                    env_name=request.env.env_name,
                    config=_env_config,
                    seed=request.env.seed,
                    rl_run_id=request.run_id,
                )
            )
            env_id = env_response.env_id
            env_handle = registry.get_env(env_id)
            created_env_id = env_id

        # Resolve or create policy
        if request.policy.policy_id:
            policy_handle = registry.get_policy(request.policy.policy_id)
            if not policy_handle:
                raise HTTPException(
                    status_code=404,
                    detail=f"Policy {request.policy.policy_id} not found",
                )
            policy_id = request.policy.policy_id
        else:
            # Create new policy
            from .policy_routes import create_policy, PolicyCreateRequest

            if not request.policy.policy_name:
                raise ValueError("FATAL: policy_name is required - NO FALLBACKS!")

            # Propagate training_session_id and synth_base_url via policy config
            _policy_config = dict(request.policy.config or {})
            if request.training_session_id is not None:
                _policy_config.setdefault(
                    "training_session_id", request.training_session_id
                )
            if request.synth_base_url is not None:
                _policy_config.setdefault("synth_base_url", request.synth_base_url)
            policy_response = await create_policy(
                PolicyCreateRequest(
                    policy_name=request.policy.policy_name,
                    config=_policy_config,
                    rl_run_id=request.run_id,
                    bound_env_id=env_id,
                ),
                req,
            )
            policy_id = policy_response.policy_id
            policy_handle = registry.get_policy(policy_id)
            created_policy_id = policy_id

        # Bind policy to environment if not already bound
        if policy_handle and not policy_handle.bound_env_id:
            policy_handle.bound_env_id = env_id

        # Record seed bound to environment for end-of-rollout verification/logging
        try:
            env_seed_used = int(getattr(env_handle, "seed", 0) or 0)
        except Exception:
            env_seed_used = None

        # Initialize trajectory
        trajectory_steps = []
        pending_tool_calls = None
        current_obs = env_handle.last_observation
        total_reward = 0.0
        ops_executed = 0
        last_agent_response_ts: float | None = None
        last_policy_meta: Dict[str, Any] | None = None
        last_env_step_ms: float | None = None
        last_env_step_completed_ts: float | None = None

        # Stepwise reward configuration (Crafter shaping; gate on explicit enable)
        step_rewards_cfg_raw: Dict[str, Any] = {}
        try:
            if isinstance(request.policy.config, dict):
                step_rewards_cfg_raw = dict(request.policy.config.get("step_rewards") or {})
        except Exception:
            step_rewards_cfg_raw = {}
        if not step_rewards_cfg_raw:
            try:
                if isinstance(request.env.config, dict):
                    step_rewards_cfg_raw = dict(request.env.config.get("step_rewards") or {})
            except Exception:
                step_rewards_cfg_raw = {}

        step_rewards_enabled = bool(step_rewards_cfg_raw.get("enabled", False))
        step_rewards_mode = str(step_rewards_cfg_raw.get("mode") or "off").lower()
        try:
            step_rewards_indicator_lambda = float(
                step_rewards_cfg_raw.get("indicator_lambda") or 0.0
            )
        except Exception:
            step_rewards_indicator_lambda = 0.0
        try:
            step_rewards_beta = float(step_rewards_cfg_raw.get("step_beta") or 0.0)
        except Exception:
            step_rewards_beta = 0.0
        step_rewards_active = step_rewards_enabled and step_rewards_mode == "decision_stepwise"

        def _extract_achievements(obs: Any) -> Dict[str, bool]:
            if not isinstance(obs, dict):
                return {}
            ach = obs.get("achievements_status")
            if isinstance(ach, dict):
                return {str(k): bool(v) for k, v in ach.items()}
            return {}

        def _summarize_tool_calls(tool_calls: Any) -> List[Dict[str, Any]]:
            if not tool_calls:
                return []
            try:
                items = (
                    tool_calls
                    if isinstance(tool_calls, list)
                    else list(tool_calls)  # tolerates tuples or pydantic lists
                )
            except Exception:
                return []
            summary: List[Dict[str, Any]] = []
            for tc in items:
                tool_name = None
                args: Any = {}
                if isinstance(tc, dict):
                    tool_name = tc.get("tool") or tc.get("tool_name") or tc.get("name")
                    raw_args = tc.get("arguments") or tc.get("args") or {}
                else:
                    tool_name = getattr(tc, "tool", None) or getattr(tc, "tool_name", None)
                    raw_args = getattr(tc, "arguments", None) or getattr(tc, "args", None) or {}
                args = raw_args
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                    except Exception:
                        args = raw_args
                summary.append({"tool": tool_name, "args": args})
            return summary

        decision_samples: List[Dict[str, Any]] = []
        decision_index = 0
        prev_achievements = _extract_achievements(current_obs)
        # Track episode-level achievements that have been seen as true at any point so far
        episode_seen_achievements: set[str] = set(
            [k for k, v in (prev_achievements or {}).items() if bool(v)]
        )
        stepwise_indicator_sum = 0.0
        stepwise_reward_sum = 0.0
        stepwise_new_achievements_total = 0
        final_achievement_count = sum(1 for v in prev_achievements.values() if v)

        # Execute ops sequence
        for op_idx, op in enumerate(request.ops):
            # Check for abort
            if registry.is_run_aborted(request.run_id):
                logger.info(f"Run {request.run_id} aborted at op {op_idx}")
                break

            # Check safety limits
            if ops_executed >= request.safety.max_ops:
                logger.warning(f"Reached max_ops limit ({request.safety.max_ops})")
                break

            if op == "agent":
                # Policy step
                from .policy_routes import step_policy, PolicyStepRequest

                agent_request_start = _time.perf_counter()
                if last_agent_response_ts is not None and last_policy_meta is not None:
                    try:
                        timing_prev = last_policy_meta.setdefault("timing", {})
                        decision_ms = max(
                            0.0,
                            (agent_request_start - float(last_agent_response_ts)) * 1000.0,
                        )
                        # Update timing on prior policy meta (kept by previous env step)
                        timing_prev["decision_ms"] = decision_ms
                        if last_env_step_ms is not None:
                            timing_prev["env_step_ms"] = float(last_env_step_ms)
                            timing_prev["overhead_ms"] = max(
                                0.0, decision_ms - float(last_env_step_ms)
                            )
                        else:
                            timing_prev.setdefault("overhead_ms", 0.0)
                        timing_prev["decision_ready_s"] = agent_request_start
                        # Also backfill the last appended trajectory step so the trainer
                        # can always see decision_ms without relying on shared dict refs.
                        if trajectory_steps:
                            try:
                                _last = trajectory_steps[-1]
                                _info = dict(_last.info or {})
                                _meta = dict(_info.get("meta") or {})
                                _timing = dict(_meta.get("timing") or {})
                                _timing["decision_ms"] = decision_ms
                                if last_env_step_ms is not None:
                                    _timing.setdefault("env_step_ms", float(last_env_step_ms))
                                    _timing.setdefault("overhead_ms", max(0.0, decision_ms - float(last_env_step_ms)))
                                else:
                                    _timing.setdefault("overhead_ms", 0.0)
                                _meta["timing"] = _timing
                                _info["meta"] = _meta
                                _last.info = _info
                            except Exception:
                                pass
                    except Exception:
                        pass
                last_env_step_ms = None
                last_env_step_completed_ts = None

                # Build metadata for policy (carry previous tool_calls and env result)
                metadata = {}
                if pending_tool_calls:
                    metadata["prev_tool_calls"] = pending_tool_calls
                if len(trajectory_steps) > 0:
                    last_step = trajectory_steps[-1]
                    # Prefer the last executed tool calls to seed history
                    if last_step.tool_calls:
                        metadata["prev_tool_calls"] = last_step.tool_calls
                    # Provide a compact env result snapshot
                    metadata["prev_env_result"] = {
                        "observation": last_step.obs,
                        "reward": last_step.reward,
                        "done": last_step.done,
                        "truncated": last_step.truncated,
                        "info": last_step.info,
                    }

                # Log compact metadata summary to confirm history threading
                try:
                    _prev_calls = (
                        metadata["prev_tool_calls"]
                        if isinstance(metadata, dict) and "prev_tool_calls" in metadata
                        else None
                    )
                    _count = len(_prev_calls) if isinstance(_prev_calls, list) else 0
                    _first_guess = None
                    if _count > 0 and isinstance(_prev_calls[0], dict):
                        _args = (
                            _prev_calls[0]["arguments"]
                            if "arguments" in _prev_calls[0]
                            else None
                        )
                        if isinstance(_args, str):
                            import json as _json

                            try:
                                _args = _json.loads(_args)
                            except Exception:
                                _args = {}
                        if isinstance(_args, dict):
                            _first_guess = (
                                _args["guess"] if "guess" in _args else None
                            ) or (_args["word"] if "word" in _args else None)
                    logger.info(
                        "POLICY_METADATA: prev_tool_calls=%d first_guess=%r has_prev_env_result=%s",
                        _count,
                        _first_guess,
                        str("prev_env_result" in metadata),
                    )
                except Exception:
                    pass

                try:
                    policy_response = await step_policy(
                        PolicyStepRequest(
                            policy_id=policy_id,
                            observation=current_obs,
                            metadata=metadata,
                        ),
                        req,
                    )
                except Exception as _pe:
                    # Do not 500 the rollout; finalize with partial trajectory
                    try:
                        logger.warning(
                            "POLICY_STEP_FAIL: terminating episode early run_id=%s op_idx=%s err=%s",
                            request.run_id,
                            str(op_idx),
                            str(_pe),
                        )
                    except Exception:
                        pass

                    # Build partial trajectory and return HTTP 200
                    trajectory = RolloutTrajectory(
                        env_id=env_id,
                        policy_id=policy_id,
                        steps=trajectory_steps,
                        final={
                            "observation": current_obs,
                            "rollout_status": "partial_policy_error",
                            "error": str(_pe),
                            "at_op": op,
                        },
                        length=len(trajectory_steps),
                        decision_samples=decision_samples if step_rewards_active else None,
                    )
                    metrics = RolloutMetrics(
                        episode_returns=[total_reward],
                        mean_return=total_reward,
                        num_steps=len(trajectory_steps),
                        num_episodes=1,
                    )
                    aborted = registry.is_run_aborted(request.run_id)
                    if not aborted:
                        registry.complete_run(request.run_id)
                    return RolloutResponse(
                        run_id=request.run_id,
                        trajectories=[trajectory],
                        branches={},
                        metrics=metrics,
                        aborted=aborted,
                        ops_executed=ops_executed,
                    )

                agent_response_ts = _time.perf_counter()
                if isinstance(policy_response.meta, dict):
                    try:
                        timing_cur = policy_response.meta.setdefault("timing", {})
                        timing_cur["agent_request_start_s"] = agent_request_start
                        timing_cur["agent_response_s"] = agent_response_ts
                        if "inference_ms" in policy_response.meta:
                            try:
                                timing_cur.setdefault(
                                    "inference_ms",
                                    float(policy_response.meta["inference_ms"]),
                                )
                                timing_cur.setdefault(
                                    "inference_s",
                                    float(policy_response.meta["inference_ms"]) / 1000.0,
                                )
                            except Exception:
                                pass
                    except Exception:
                        pass
                    last_policy_meta = policy_response.meta
                else:
                    last_policy_meta = None
                last_agent_response_ts = agent_response_ts

                pending_tool_calls = policy_response.tool_calls
                ops_executed += 1

            elif op == "env":
                if not pending_tool_calls:
                    # Treat absence of tool calls as a soft terminal condition; yield partial trajectory
                    try:
                        logger.warning(
                            "NO_TOOL_CALLS: terminating episode early run_id=%s op_idx=%s",
                            request.run_id,
                            str(op_idx),
                        )
                    except Exception:
                        pass
                    term_step = RolloutStep(
                        obs=current_obs,
                        tool_calls=[],
                        reward=None,
                        done=True,
                        truncated=False,
                        info={
                            "terminated": True,
                            "reason": "no_tool_calls",
                        },
                    )
                    trajectory_steps.append(term_step)
                    trajectory = RolloutTrajectory(
                        env_id=env_id,
                        policy_id=policy_id,
                        steps=trajectory_steps,
                        final={
                            "observation": current_obs,
                            "rollout_status": "partial_no_tool_calls",
                            "at_op": op,
                        },
                        length=len(trajectory_steps),
                        decision_samples=decision_samples if step_rewards_active else None,
                    )
                    metrics = RolloutMetrics(
                        episode_returns=[total_reward],
                        mean_return=total_reward,
                        num_steps=len(trajectory_steps),
                        num_episodes=1,
                    )
                    aborted = registry.is_run_aborted(request.run_id)
                    if not aborted:
                        registry.complete_run(request.run_id)
                    return RolloutResponse(
                        run_id=request.run_id,
                        trajectories=[trajectory],
                        branches={},
                        metrics=metrics,
                        aborted=aborted,
                        ops_executed=ops_executed,
                    )

                # Environment step
                from .environment_routes import step_environment, EnvStepRequest

                env_step_error: Exception | None = None
                env_response = None
                env_step_start = _time.perf_counter()
                try:
                    env_response = await step_environment(
                        EnvStepRequest(
                            env_id=env_id,
                            tool_calls=pending_tool_calls,
                        )
                    )
                except Exception as _ee:
                    env_step_error = _ee
                env_step_end = _time.perf_counter()
                env_step_duration_ms = (env_step_end - env_step_start) * 1000.0
                last_env_step_ms = env_step_duration_ms
                last_env_step_completed_ts = env_step_end
                if last_policy_meta is not None:
                    try:
                        timing_env = last_policy_meta.setdefault("timing", {})
                        timing_env["env_step_ms"] = env_step_duration_ms
                        timing_env["env_step_end_s"] = env_step_end
                    except Exception:
                        pass

                if env_step_error is not None:
                    # Invalid action or environment rejection — terminate episode early with partial trajectory
                    try:
                        logger.warning(
                            "ENV_STEP_FAIL: terminating episode early run_id=%s op_idx=%s err=%s",
                            request.run_id,
                            str(op_idx),
                            str(env_step_error),
                        )
                    except Exception:
                        pass

                    term_step = RolloutStep(
                        obs=current_obs,
                        tool_calls=pending_tool_calls,
                        reward=None,
                        done=True,
                        truncated=False,
                        info={
                            "terminated": True,
                            "reason": "invalid_action",
                            "error": str(env_step_error),
                        },
                    )
                    trajectory_steps.append(term_step)
                    # Build partial response
                    trajectory = RolloutTrajectory(
                        env_id=env_id,
                        policy_id=policy_id,
                        steps=trajectory_steps,
                        final={
                            "observation": current_obs,
                            "rollout_status": "partial_invalid_action",
                            "error": str(env_step_error),
                            "at_op": op,
                        },
                        length=len(trajectory_steps),
                        decision_samples=decision_samples if step_rewards_active else None,
                    )
                    metrics = RolloutMetrics(
                        episode_returns=[total_reward],
                        mean_return=total_reward,
                        num_steps=len(trajectory_steps),
                        num_episodes=1,
                    )
                    aborted = registry.is_run_aborted(request.run_id)
                    if not aborted:
                        registry.complete_run(request.run_id)
                    if (
                        last_policy_meta is not None
                        and last_agent_response_ts is not None
                        and "decision_ms" not in last_policy_meta.get("timing", {})
                    ):
                        try:
                            timing_last = last_policy_meta.setdefault("timing", {})
                            decision_ms = max(
                                0.0,
                                (env_step_end - float(last_agent_response_ts)) * 1000.0,
                            )
                            timing_last["decision_ms"] = decision_ms
                            timing_last.setdefault("overhead_ms", max(0.0, decision_ms - env_step_duration_ms))
                        except Exception:
                            pass
                    return RolloutResponse(
                        run_id=request.run_id,
                        trajectories=[trajectory],
                        branches={},
                        metrics=metrics,
                        aborted=aborted,
                        ops_executed=ops_executed,
                    )

                # Reaching here means env step succeeded
                assert env_response is not None

                # Record step, including policy meta if present for timing/tokens observability
                _info = env_response.info if isinstance(env_response.info, dict) else {}
                # Attach policy meta from the immediately preceding agent step
                try:
                    prev_meta = {}
                    if "policy_response" in locals() and isinstance(
                        policy_response.meta, dict
                    ):  # type: ignore[name-defined]
                        prev_meta = policy_response.meta
                    if prev_meta:
                        _info = dict(_info)
                        _info["meta"] = prev_meta
                except Exception:
                    pass

                decision_index += 1
                next_obs = env_response.observation
                new_achievement_state = _extract_achievements(next_obs)
                final_achievement_count = sum(
                    1 for _, unlocked in new_achievement_state.items() if unlocked
                )
                indicator_val = 0
                reward_stepwise = 0.0
                if step_rewards_active:
                    decision_actions = _summarize_tool_calls(pending_tool_calls)
                    stepwise_info, decision_record, stats = compute_stepwise_reward(
                        prev_achievements or {},
                        new_achievement_state,
                        decision_index,
                        decision_actions,
                        step_rewards_indicator_lambda,
                    )
                    indicator_val = int(stats.get("indicator", 0.0))
                    reward_stepwise = float(stats.get("reward", 0.0))
                    stepwise_indicator_sum += float(stats.get("indicator", 0.0))
                    stepwise_reward_sum += reward_stepwise
                    stepwise_new_achievements_total += int(
                        stats.get("new_achievements_count", 0.0)
                    )
                    if not isinstance(_info, dict):
                        _info = {}
                    else:
                        _info = dict(_info)
                    _info["stepwise"] = stepwise_info
                    # Compute decision-level rewards (absolute vs unique) and attach to metadata
                    try:
                        turned_true = set(stepwise_info.get("new_achievements") or [])
                        seen_before = set(episode_seen_achievements)
                        new_unique = sorted(list(turned_true - seen_before))
                        ach_delta = int(len(turned_true))
                        unique_delta = int(len(new_unique))
                        # Prepare stable lists for logging/metadata
                        all_list = sorted(list(turned_true))
                        # Ensure nested meta exists
                        meta_block = _info.get("meta") if isinstance(_info.get("meta"), dict) else {}
                        decision_rewards = {
                            "turn": int(decision_index),
                            "ach_delta": ach_delta,
                            "unique_delta": unique_delta,
                            "all": all_list,
                            "unique": new_unique,
                        }
                        meta_block["decision_rewards"] = decision_rewards
                        _info["meta"] = meta_block
                        # Update episode-level seen set after attributing uniqueness to this decision
                        episode_seen_achievements.update(turned_true)
                    except Exception:
                        # Best-effort; do not block rollout on metadata computation
                        pass
                    decision_samples.append(decision_record)
                prev_achievements = new_achievement_state

                step = RolloutStep(
                    obs=_summarize_observation_for_storage(env_handle, current_obs),
                    tool_calls=pending_tool_calls,
                    reward=env_response.reward,
                    done=env_response.done,
                    truncated=env_response.truncated,
                    info=_info,
                )
                trajectory_steps.append(step)

                if env_response.reward is not None:
                    total_reward += env_response.reward

                # Update state
                current_obs = next_obs
                pending_tool_calls = None
                ops_executed += 1

                # Handle episode end
                if env_response.done:
                    if request.on_done == "reset":
                        # Reset environment
                        from .environment_routes import (
                            reset_environment,
                            EnvResetRequest,
                        )

                        reset_response = await reset_environment(
                            EnvResetRequest(env_id=env_id)
                        )
                        current_obs = reset_response.observation
                    elif request.on_done == "terminate":
                        break

            else:
                logger.warning(f"Unknown op: {op}")

        if (
            last_policy_meta is not None
            and last_agent_response_ts is not None
            and "timing" in last_policy_meta
            and isinstance(last_policy_meta["timing"], dict)
            and "decision_ms" not in last_policy_meta["timing"]
        ):
            try:
                final_now = last_env_step_completed_ts or _time.perf_counter()
                final_decision_ms = max(
                    0.0, (final_now - float(last_agent_response_ts)) * 1000.0
                )
                timing_final = last_policy_meta.setdefault("timing", {})
                timing_final["decision_ms"] = final_decision_ms
                if last_env_step_ms is not None:
                    timing_final.setdefault(
                        "env_step_ms", float(last_env_step_ms)
                    )
                    timing_final.setdefault(
                        "overhead_ms",
                        max(0.0, final_decision_ms - float(last_env_step_ms)),
                    )
                else:
                    timing_final.setdefault("overhead_ms", 0.0)
            except Exception:
                pass

        # Build trajectory
        trajectory = RolloutTrajectory(
            env_id=env_id,
            policy_id=policy_id,
            steps=trajectory_steps,
            final={"observation": _summarize_observation_for_storage(env_handle, current_obs)},
            length=len(trajectory_steps),
            decision_samples=decision_samples if step_rewards_active else None,
        )

        # Build metrics
        metrics = RolloutMetrics(
            episode_returns=[total_reward],
            mean_return=total_reward,
            num_steps=len(trajectory_steps),
            num_episodes=1,
        )

        # Environment-specific: Log summary if available
        try:
            # Check if this is a Wordle environment and use Wordle helpers (lazy import)
            try:
                from .envs.wordle.environment import WordleEnvironmentWrapper as _WordleWrapper
                from .envs.wordle.helpers import (
                    get_wordle_rollout_summary,
                    log_wordle_rollout_summary,
                )
            except Exception:
                _WordleWrapper = None  # type: ignore
                get_wordle_rollout_summary = None  # type: ignore
                log_wordle_rollout_summary = None  # type: ignore

            is_wordle = _WordleWrapper is not None and isinstance(env_handle.env, _WordleWrapper)
            if is_wordle:
                # Convert trajectory steps to expected format
                formatted_steps = []
                for step in trajectory_steps:
                    formatted_steps.append({"tool_calls": step.tool_calls or []})

                if get_wordle_rollout_summary is not None and log_wordle_rollout_summary is not None:
                    summary = get_wordle_rollout_summary(
                        formatted_steps, current_obs, env_handle
                    )
                    log_wordle_rollout_summary(request.run_id, summary)
        except ImportError:
            # Wordle helpers not available, skip Wordle-specific logging
            pass
        except Exception as e:
            logger.warning(f"Failed to generate environment-specific summary: {e}")

        # Mark run as completed
        aborted = registry.is_run_aborted(request.run_id)
        if not aborted:
            registry.complete_run(request.run_id)

        return RolloutResponse(
            run_id=request.run_id,
            trajectories=[trajectory],
            branches={},
            metrics=metrics,
            aborted=aborted,
            ops_executed=ops_executed,
        )

    except Exception as e:
        logger.error(f"Rollout failed for run {request.run_id}: {e}")
        registry.abort_run(request.run_id)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Ensure any environment created for this rollout is terminated (no reuse across rollouts)
        try:
            if created_env_id:
                from .environment_routes import terminate_environment, EnvTerminateRequest

                await terminate_environment(EnvTerminateRequest(env_id=created_env_id))
                logger.info(
                    "ROLL_OUT: terminated environment env_id=%s seed=%s",
                    str(created_env_id),
                    str(env_seed_used) if env_seed_used is not None else "unknown",
                )
                # Verify removal from registry
                try:
                    _post = registry.get_env(created_env_id)
                    logger.info(
                        "ROLL_OUT: env_killed=%s (post_lookup=%s)",
                        str(_post is None),
                        str(_post),
                    )
                except Exception:
                    pass
        except Exception as _te:
            logger.warning(
                f"ROLL_OUT: failed to terminate environment {created_env_id}: {_te}"
            )

        # Best-effort policy cleanup if we created one (avoid reuse across rollouts)
        try:
            if created_policy_id:
                from .policy_routes import terminate_policy, PolicyTerminateRequest

                await terminate_policy(PolicyTerminateRequest(policy_id=created_policy_id))
                logger.info("ROLL_OUT: terminated policy policy_id=%s", str(created_policy_id))
        except Exception:
            pass

        try:
            _clear_seed_side_effects()
            logger.info("ROLL_OUT: RNG seed terminated/cleared before conclusion")
        except Exception:
            pass


@router.post("/run/abort", response_model=RunAbortResponse)
async def abort_run(request: RunAbortRequest) -> RunAbortResponse:
    """Abort a running rollout."""
    success = registry.abort_run(request.run_id)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Run {request.run_id} not found",
        )

    return RunAbortResponse(
        ok=True,
        run_id=request.run_id,
    )


@router.get("/run/status/{run_id}", response_model=RunStatusResponse)
async def get_run_status(run_id: str) -> RunStatusResponse:
    """Get the status of a run."""
    run_handle = registry.get_run(run_id)

    if not run_handle:
        raise HTTPException(
            status_code=404,
            detail=f"Run {run_id} not found",
        )

    return RunStatusResponse(
        run_id=run_id,
        status=run_handle.status,
        started_at=run_handle.started_at,
        finished_at=run_handle.finished_at,
    )
