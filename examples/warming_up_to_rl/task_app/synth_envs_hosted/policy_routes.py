from __future__ import annotations

import contextlib
import logging
import os
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from .envs.crafter.policy import CrafterPolicy
from .inference.openai_client import create_inference_client
from .registry import registry
from .storage.volume import storage

# Token budgeting (shared logic with inference server)
try:
    from ..core.algorithms.gspo.inference.token_limits import (
        clamp_effective_max_ctx,
    )
except Exception:  # pragma: no cover - defensive import path fallback
    clamp_effective_max_ctx = None  # type: ignore

try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover
    tiktoken = None  # type: ignore

logger = logging.getLogger(__name__)

router = APIRouter()


class PolicyCreateRequest(BaseModel):
    policy_name: str
    config: dict[str, Any] = {}
    parent_policy_id: str | None = None
    rl_run_id: str
    bound_env_id: str | None = None


class PolicyCreateResponse(BaseModel):
    policy_id: str


class PolicyStepRequest(BaseModel):
    policy_id: str
    observation: dict[str, Any]
    state: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    dry_run: bool = False


class PolicyStepResponse(BaseModel):
    tool_calls: list[dict[str, Any]]
    meta: dict[str, Any]


class PolicySnapshotRequest(BaseModel):
    policy_id: str


class PolicySnapshotResponse(BaseModel):
    snapshot_id: str
    path: str
    rl_run_id: str
    size: int


class PolicyRestoreRequest(BaseModel):
    snapshot_id: str


class PolicyRestoreResponse(BaseModel):
    policy_id: str


class PolicyTerminateRequest(BaseModel):
    policy_id: str


class PolicyTerminateResponse(BaseModel):
    ok: bool


@router.post("/create", response_model=PolicyCreateResponse)
async def create_policy(
    request: PolicyCreateRequest,
    req: Request,
) -> PolicyCreateResponse:
    """Create a new policy instance."""
    try:
        task_app = getattr(req.app.state, "task_app", None)

        # Set defaults from TaskApp / environment if not provided
        config = dict(request.config or {})
        if "inference_url" not in config and task_app is not None:
            base_url = getattr(task_app, "vllm_base_url", None)
            if base_url:
                config["inference_url"] = base_url
        if "model" not in config and task_app is not None:
            default_model = getattr(task_app, "default_model", None)
            if default_model:
                config["model"] = default_model
        if "inference_url" not in config or "model" not in config:
            raise HTTPException(
                status_code=422,
                detail="Policy configuration must include 'inference_url' and 'model'.",
            )

        # Create policy instance based on name
        pname = request.policy_name.lower()
        if pname in ["crafter-react", "crafter"]:
            policy = CrafterPolicy(
                inference_url=config["inference_url"],
                model=config["model"],
            )
            await policy.initialize(config)
        elif pname in ["wordle-react", "wordle"]:
            try:
                from .envs.wordle.policy import WordlePolicy
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Wordle policy unavailable: {e}"
                ) from e

            policy = WordlePolicy(
                inference_url=config["inference_url"],
                model=config["model"],
                word_length=int(config["word_length"]),
                max_guesses=int(config["max_guesses"]),
            )
            await policy.initialize(config)
        elif pname in ["sokoban-react", "sokoban"]:
            try:
                from .envs.sokoban.policy import SokobanPolicy
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Sokoban policy unavailable: {e}"
                ) from e

            policy = SokobanPolicy(
                inference_url=config["inference_url"],
                model=config["model"],
            )
            await policy.initialize(config)
        elif pname in ["math-react", "math"]:
            try:
                from .envs.math.policy import MathPolicy
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Math policy unavailable: {e}") from e

            policy = MathPolicy(
                inference_url=config["inference_url"],
                model=config["model"],
            )
            await policy.initialize(config)
        else:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown policy name: {request.policy_name}",
            )

        # Register in memory
        policy_id = registry.register_policy(
            policy=policy,
            rl_run_id=request.rl_run_id,
            bound_env_id=request.bound_env_id,
        )

        return PolicyCreateResponse(policy_id=policy_id)

    except Exception as e:
        logger.error(f"Failed to create policy: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/step", response_model=PolicyStepResponse)
async def step_policy(
    request: PolicyStepRequest,
    req: Request,
) -> PolicyStepResponse:
    """Execute a policy step to generate actions."""
    handle = registry.get_policy(request.policy_id)
    if not handle:
        raise HTTPException(status_code=404, detail=f"Policy {request.policy_id} not found")

    try:
        task_app = req.app.state.task_app
        policy = handle.policy
        tracing_context = getattr(req.state, "rollout_tracing", None)

        obs_text = request.observation
        if isinstance(request.observation, dict):
            if isinstance(policy, CrafterPolicy):
                from .envs.crafter.shared import format_observation as format_crafter

                obs_text = format_crafter(request.observation)
            else:
                formatted: str | None = None

                # Wordle formatting
                try:
                    from .envs.wordle.policy import WordlePolicy
                except Exception:
                    wordle_policy_cls = None  # type: ignore[assignment]
                else:
                    wordle_policy_cls = WordlePolicy

                if formatted is None and wordle_policy_cls is not None and isinstance(
                    policy, wordle_policy_cls
                ):
                    from .envs.wordle.shared import format_observation_wordle

                    # ASSERTION: Validate observation structure
                    assert request.observation is not None, "request.observation cannot be None"
                    assert isinstance(request.observation, dict), (
                        f"request.observation must be dict, got {type(request.observation)}"
                    )

                    required_keys = {
                        "text",
                        "status",
                        "remaining_guesses",
                        "guesses",
                        "feedback",
                        "reward_last",
                        "total_reward",
                        "terminated",
                    }
                    missing_keys = required_keys - set(request.observation.keys())
                    assert (
                        not missing_keys
                    ), f"Wordle observation missing required keys: {missing_keys}"

                    print("DEBUG POLICY_ROUTES: About to format Wordle observation")
                    print(f"DEBUG POLICY_ROUTES: Observation type: {type(request.observation)}")
                    print(
                        f"DEBUG POLICY_ROUTES: Observation keys: {list(request.observation.keys())}"
                    )
                    feedback_val = request.observation["feedback"]
                    print(f"DEBUG POLICY_ROUTES: Observation feedback: {feedback_val}")
                    print(
                        f"DEBUG POLICY_ROUTES: Observation guesses: {request.observation['guesses']}"
                    )
                    print(
                        "DEBUG POLICY_ROUTES: Observation text length: "
                        f"{len(request.observation['text'])}"
                    )

                    guesses = request.observation["guesses"]
                    feedback = request.observation["feedback"]
                    assert isinstance(guesses, list), f"guesses must be list, got {type(guesses)}"
                    assert isinstance(
                        feedback, list
                    ), f"feedback must be list, got {type(feedback)}"

                    formatted = format_observation_wordle(request.observation)

                    assert isinstance(formatted, str), (
                        f"obs_text must be string, got {type(formatted)}"
                    )
                    assert len(formatted) > 0, "obs_text cannot be empty"
                    assert "WORDLE" in formatted, "obs_text must contain 'WORDLE' header"
                    assert "Respond with a single tool call" in formatted, (
                        "obs_text must contain instruction text"
                    )

                    print(
                        f"DEBUG POLICY_ROUTES: Formatted obs_text length: {len(formatted)}"
                    )
                    print(
                        "DEBUG POLICY_ROUTES: Formatted obs_text contains 🟩: "
                        f"{'🟩' in formatted}"
                    )
                    print(
                        "DEBUG POLICY_ROUTES: Formatted obs_text contains 🟨: "
                        f"{'🟨' in formatted}"
                    )
                    print(
                        "DEBUG POLICY_ROUTES: Formatted obs_text contains ⬛: "
                        f"{'⬛' in formatted}"
                    )
                    print(
                        "DEBUG POLICY_ROUTES: Formatted obs_text first 200 chars: "
                        f"{formatted[:200]}"
                    )

                # Sokoban formatting
                try:
                    from .envs.sokoban.policy import SokobanPolicy
                except Exception:
                    sokoban_policy_cls = None  # type: ignore[assignment]
                else:
                    sokoban_policy_cls = SokobanPolicy

                if formatted is None and sokoban_policy_cls is not None and isinstance(
                    policy, sokoban_policy_cls
                ):
                    from .envs.sokoban.shared import format_observation_sokoban

                    formatted = format_observation_sokoban(request.observation)

                # Math formatting
                try:
                    from .envs.math.policy import MathPolicy
                except Exception:
                    math_policy_cls = None  # type: ignore[assignment]
                else:
                    math_policy_cls = MathPolicy

                if formatted is None and math_policy_cls is not None and isinstance(
                    policy, math_policy_cls
                ):
                    try:
                        formatted = str(
                            request.observation.get("problem_text") or request.observation
                        )
                    except Exception:
                        formatted = str(request.observation)

                if formatted is None:
                    formatted = str(request.observation)

                obs_text = formatted

        # Merge metadata with raw observation for multimodal policies
        step_metadata: dict[str, Any] = dict(request.metadata or {})
        step_metadata["raw_observation"] = request.observation

        # Execute policy step to get inference request
        tool_calls, meta = await policy.step(
            observation_text=obs_text,
            state=request.state,
            metadata=step_metadata,
        )
        # Compact tool call summary
        with contextlib.suppress(Exception):
            _summary: list[dict[str, Any]] = []
            _tc = tool_calls or []
            for _item in (_tc if isinstance(_tc, list) else []):
                if isinstance(_item, dict):
                    _tool = _item.get("tool")
                    _args = _item.get("args")
                    _keys = list(_args.keys()) if isinstance(_args, dict) else []
                    _summary.append({"tool": _tool, "args_keys": _keys})
            logger.info(
                "POLICY_STEP: tool_calls=%d summary=%s",
                len(_tc),
                _summary,
            )

        # If not dry run, perform inference
        if not request.dry_run and "inference_request" in meta:
            # CRITICAL: Validate that the inference request contains the correct prompts for the policy
            inf_req = meta["inference_request"]
            msgs = inf_req["messages"]
            model_name = inf_req.get("model") or getattr(policy, "model", None) or ""
            system_messages: list[str] = []
            user_messages: list[str] = []
            if msgs and len(msgs) > 0 and msgs[0]["role"] == "system":
                sys_text = msgs[0]["content"]
                policy_name = getattr(policy, "name", "") or type(policy).__name__.lower()

                # Assert environment-specific prompts match the policy
                if policy_name in ("wordle-react", "wordle"):
                    if "Wordle" not in sys_text:
                        raise ValueError(
                            f"PROMPT MISMATCH: Wordle policy {policy_name} received system prompt without 'Wordle' keyword: {sys_text[:200]}..."
                        )
                    if "Crafter" in sys_text:
                        raise ValueError(
                            f"PROMPT MISMATCH: Wordle policy {policy_name} received Crafter system prompt: {sys_text[:200]}..."
                        )

                elif policy_name in ("crafter-react", "crafter") or isinstance(
                    policy, CrafterPolicy
                ):
                    if "Crafter" not in sys_text:
                        raise ValueError(
                            f"PROMPT MISMATCH: Crafter policy {policy_name} received system prompt without 'Crafter' keyword: {sys_text[:200]}..."
                        )
                    if "Wordle" in sys_text:
                        raise ValueError(
                            f"PROMPT MISMATCH: Crafter policy {policy_name} received Wordle system prompt: {sys_text[:200]}..."
                        )
                elif policy_name in ("sokoban-react", "sokoban"):
                    if "Sokoban" not in sys_text:
                        raise ValueError(
                            f"PROMPT MISMATCH: Sokoban policy {policy_name} received system prompt without 'Sokoban' keyword: {sys_text[:200]}..."
                        )
                    if "Crafter" in sys_text or "Wordle" in sys_text:
                        raise ValueError(
                            f"PROMPT MISMATCH: Sokoban policy {policy_name} received wrong environment system prompt: {sys_text[:200]}..."
                        )

                logger.info(
                    f"✅ PROMPT VALIDATION: {policy_name} policy has correct system prompt containing expected environment keywords"
                )
            else:
                logger.warning(
                    f"⚠️ PROMPT VALIDATION: No system message found in inference request for policy {getattr(policy, 'name', type(policy).__name__)}"
                )

            # Emit full system/user prompts for observability (no secrets included)
            try:

                def _as_text(content: object) -> str:
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list):
                        # Concatenate any dict segments that resemble OpenAI content parts
                        parts: list[str] = []
                        for seg in content:
                            try:
                                if isinstance(seg, dict):
                                    txt = seg.get("text") or seg.get("content") or ""
                                    if isinstance(txt, str):
                                        parts.append(txt)
                            except Exception:
                                continue
                        return "".join(parts)
                    return str(content)

                system_prompt_records: list[dict[str, Any]] = []
                user_prompt_records: list[dict[str, Any]] = []
                for message in msgs:
                    role = message.get("role")
                    raw_content = message.get("content")
                    content = _as_text(raw_content)
                    record = {"role": role, "text": content, "content": raw_content}
                    if role == "system":
                        system_prompt_records.append(record)
                    elif role == "user":
                        user_prompt_records.append(record)

                logger.info(
                    "PROMPTS: system_msgs=%d user_msgs=%d last_user_chars=%d",
                    len(system_prompt_records),
                    len(user_prompt_records),
                    len(user_prompt_records[-1].get("text", "")) if user_prompt_records else 0,
                )

                if system_prompt_records:
                    logger.info("PROMPT_DUMP_SYSTEM_BEGIN")
                    for idx, rec in enumerate(system_prompt_records):
                        smsg = rec.get("text", "")
                        logger.info(f"SYSTEM[{idx}]\n{smsg}")
                    logger.info("PROMPT_DUMP_SYSTEM_END")

                if user_prompt_records:
                    logger.info("PROMPT_DUMP_USER_BEGIN")
                    for idx, rec in enumerate(user_prompt_records):
                        umsg = rec.get("text", "")
                        logger.info(f"USER[{idx}]\n{umsg}")
                    logger.info("PROMPT_DUMP_USER_END")
                    # Print concise preview for visibility in standard logs
                    with contextlib.suppress(Exception):
                        last_user = (
                            user_prompt_records[-1].get("text", "")
                            if user_prompt_records
                            else ""
                        )
                        print(f"[task:crafter] user prompt: {last_user}", flush=True)
            except Exception as e:
                logger.warning(f"PROMPT_DUMP_FAILED: {e}")

            if tracing_context is not None:
                try:
                    await tracing_context.record_policy_prompts(
                        system_prompt_records, user_prompt_records
                    )
                except Exception as exc:
                    logger.debug(f"TRACING_PROMPTS_FAIL: {exc}")

            # Create inference client (choose API key by target provider)
            # Require inference_url to be set explicitly by the rollout policy config.
            target_url = (
                meta.get("inference_url")
                or getattr(policy, "inference_url", None)
                or getattr(task_app, "vllm_base_url", None)
            )

            # Ensure meta carries the final target URL for downstream logging/clients
            with contextlib.suppress(Exception):
                meta["inference_url"] = target_url

            # Select API key based on resolved target URL
            api_key_override = None
            try:
                import os as _os

                if isinstance(target_url, str):
                    low_url = target_url.lower()
                    # Proxy endpoints should not receive a bearer; the server-side proxy holds the vendor key
                    if "/proxy/groq" in low_url or "/proxy/openai" in low_url:
                        api_key_override = None
                    elif "openai.com" in low_url:
                        api_key_override = _os.getenv("OPENAI_API_KEY") or getattr(
                            task_app, "openai_api_key", None
                        )
                    elif "groq.com" in low_url or "/proxy/groq" in low_url:
                        api_key_override = _os.getenv("GROQ_API_KEY")
                    else:
                        api_key_override = (
                            _os.getenv("SYNTH_API_KEY")
                            or _os.getenv("OPENAI_API_KEY")
                            or getattr(task_app, "openai_api_key", None)
                        )
                else:
                    api_key_override = (
                        _os.getenv("SYNTH_API_KEY")
                        or _os.getenv("OPENAI_API_KEY")
                        or getattr(task_app, "openai_api_key", None)
                    )
            except Exception:
                api_key_override = None

            if api_key_override:
                try:
                    masked = f"{api_key_override[:6]}…{api_key_override[-4:]}"
                except Exception:
                    masked = "<masked>"
                logger.debug(f"INFERENCE_AUTH: Using bearer key {masked}")
            else:
                logger.warning(
                    "INFERENCE_AUTH: No API key resolved for inference request; downstream may 401"
                )

            client = create_inference_client(task_app, api_key=api_key_override)

            # Add policy identification header for observability
            policy_name = getattr(policy, "name", "") or type(policy).__name__.lower()
            extra_headers = {"X-Policy-Name": policy_name}

            # Apply input truncation to avoid 422 from inference server
            try:
                model_name = inf_req.get("model") or getattr(policy, "model", None) or ""
                env_max_ctx = None
                try:
                    _env_max = int(os.getenv("CHAT_MAX_MODEL_LEN", "0") or 0)
                    env_max_ctx = _env_max if _env_max > 0 else None
                except Exception:
                    env_max_ctx = None
                # Compute effective max context and safety margin
                eff_ctx = None
                if clamp_effective_max_ctx is not None:
                    eff_ctx = clamp_effective_max_ctx(
                        model_name=model_name,
                        configured_max_model_len=None,
                        env_max_model_len=env_max_ctx,
                    )
                # Hard lower-only chat input cap if provided
                try:
                    hard_input_cap = int(os.getenv("CHAT_MAX_INPUT_TOKENS", "0") or 0)
                    hard_input_cap = hard_input_cap if hard_input_cap > 0 else None
                except Exception:
                    hard_input_cap = None
                try:
                    safety_margin = int(os.getenv("CHAT_BUDGET_SAFETY", "64").strip() or 64)
                except Exception:
                    safety_margin = 64

                # Determine budget
                budget = None
                if isinstance(eff_ctx, int) and eff_ctx > 0:
                    budget = max(256, eff_ctx - safety_margin)
                if isinstance(hard_input_cap, int) and hard_input_cap > 0:
                    budget = min(budget, hard_input_cap) if budget is not None else hard_input_cap

                if budget is not None and budget > 0 and isinstance(msgs, list):
                    # Choose tokenizer
                    enc = None
                    if tiktoken is not None:
                        try:
                            if model_name:
                                enc = tiktoken.encoding_for_model(model_name)
                            else:
                                enc = tiktoken.get_encoding("cl100k_base")
                        except Exception:
                            try:
                                enc = tiktoken.get_encoding("cl100k_base")
                            except Exception:
                                enc = None

                    def _content_to_text(content: object) -> str:
                        if isinstance(content, str):
                            return content
                        if isinstance(content, list):
                            parts: list[str] = []
                            for seg in content:
                                try:
                                    if isinstance(seg, dict):
                                        txt = seg.get("text") or seg.get("content") or ""
                                        if isinstance(txt, str):
                                            parts.append(txt)
                                except Exception:
                                    continue
                            return "".join(parts)
                        try:
                            return str(content)
                        except Exception:
                            return ""

                    def _count_tokens(text: str) -> int:
                        if enc is None:
                            # Fall back to character count heuristic (~4 chars per token)
                            try:
                                return max(1, int(len(text) / 4))
                            except Exception:
                                return len(text)
                        try:
                            return len(enc.encode(text))
                        except Exception:
                            return max(1, int(len(text) / 4))

                    def _count_messages_tokens(messages: list[dict[str, Any]]) -> int:
                        total = 0
                        for m in messages:
                            total += _count_tokens(_content_to_text(m.get("content")))
                        return total

                    def _truncate_messages_to_budget(
                        messages: list[dict[str, Any]],
                        max_tokens: int,
                    ) -> tuple[list[dict[str, Any]], int, int, int]:
                        before = _count_messages_tokens(messages)
                        if before <= max_tokens:
                            return messages, before, before, len(messages)
                        # Always try to preserve the first system message if present
                        system_msg = None
                        start_idx = 0
                        if messages and messages[0].get("role") == "system":
                            system_msg = messages[0]
                            start_idx = 1
                        kept_rev: list[dict[str, Any]] = []
                        total = _count_messages_tokens([system_msg] if system_msg else [])
                        # Walk from the end keeping most recent messages
                        for m in reversed(messages[start_idx:]):
                            t = _count_tokens(_content_to_text(m.get("content")))
                            if total + t <= max_tokens:
                                kept_rev.append(m)
                                total += t
                            else:
                                # Try to keep a truncated version of this message if we have some budget left
                                remaining = max_tokens - total
                                if remaining > 16:  # keep at least a little context
                                    txt = _content_to_text(m.get("content"))
                                    # Binary search-ish trim by tokens
                                    low, high = 0, len(txt)
                                    best = None
                                    while low <= high:
                                        mid = (low + high) // 2
                                        candidate = txt[-mid:]
                                        if _count_tokens(candidate) <= remaining:
                                            best = candidate
                                            low = mid + 1
                                        else:
                                            high = mid - 1
                                    if best is not None and best:
                                        m2 = dict(m)
                                        m2["content"] = best
                                        kept_rev.append(m2)
                                        total += _count_tokens(best)
                                break
                        kept = list(reversed(kept_rev))
                        if system_msg is not None:
                            kept = [system_msg] + kept
                        after = _count_messages_tokens(kept)
                        return kept, before, after, len(kept)

                    new_msgs, before_toks, after_toks, kept_count = _truncate_messages_to_budget(
                        msgs, int(budget)
                    )
                    if new_msgs is not msgs:
                        inf_req["messages"] = new_msgs
                        with contextlib.suppress(Exception):
                            logger.info(
                                {
                                    "chat_truncated": True,
                                    "token_budget": int(budget),
                                    "before_tokens": int(before_toks),
                                    "after_tokens": int(after_toks),
                                    "kept_msgs": int(kept_count),
                                }
                            )
            except Exception as _trunc_e:
                logger.warning(f"CHAT_TRUNCATION_FAILED: {type(_trunc_e).__name__}: {_trunc_e}")

            # Formal assertion: If tools are expected, ensure tool_choice and tools are set
            if policy_name in (
                "wordle-react",
                "sokoban-react",
                "crafter-react",
            ) and getattr(policy, "use_tools", True):
                req_tools = meta["inference_request"]["tools"]
                req_tool_choice = meta["inference_request"]["tool_choice"]
                req_stop_after = meta["inference_request"]["stop_after_tool_calls"]
                logger.info(
                    f"TOOLCALL_CONFIG: policy={policy_name} tools_present={bool(req_tools)} tool_choice={req_tool_choice} stop_after={req_stop_after}"
                )
                if not req_tools or req_tool_choice != "required":
                    raise HTTPException(
                        status_code=500,
                        detail=f"TOOLCALL_ASSERTION_FAIL: Missing tools or tool_choice!=required for policy {policy_name}",
                    )

            # Call inference service with retries for Flash cold-start (503)
            import time as _t

            # Prompt diagnostics before sending to inference: build chat template locally,
            # count tokens, and log the first 10k tokens if oversized. Also stash a
            # compact preview in meta so the trainer can surface it.
            with contextlib.suppress(Exception):
                req_for_diag = meta.get("inference_request", {})
                model_for_diag = req_for_diag.get("model") or getattr(policy, "model", None) or ""
                messages_for_diag = req_for_diag.get("messages") or []
                if model_for_diag and messages_for_diag:
                    from transformers import AutoTokenizer

                    tok = AutoTokenizer.from_pretrained(model_for_diag)
                    prompt_preview = tok.apply_chat_template(
                        messages_for_diag,
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    ids = tok.encode(prompt_preview, add_special_tokens=False)
                    max_len = getattr(tok, "model_max_length", None)
                    over_limit = False
                    with contextlib.suppress(Exception):
                        over_limit = (
                            isinstance(max_len, int) and max_len > 0 and len(ids) > int(max_len)
                        )
                    if over_limit or len(ids) > 10000:
                        preview_ids = ids[:10000]
                        preview_text = tok.decode(
                            preview_ids,
                            skip_special_tokens=False,
                        )
                        with contextlib.suppress(Exception):
                            logger.warning(
                                {
                                    "prompt_token_overflow_local": True,
                                    "model": str(model_for_diag),
                                    "token_count": int(len(ids)),
                                    "model_max_length": int(max_len)
                                    if isinstance(max_len, int)
                                    else None,
                                    "preview_tokens_logged": int(len(preview_ids)),
                                    "prompt_preview_first_10k_tokens": preview_text,
                                }
                            )
                        with contextlib.suppress(Exception):
                            meta["prompt_debug"] = {
                                "token_count": int(len(ids)),
                                "model_max_length": int(max_len)
                                if isinstance(max_len, int)
                                else None,
                                "preview_first_10k_tokens": preview_text,
                            }

            # Emit the exact prompt/messages and tools before calling the LLM (bounded preview)
            with contextlib.suppress(Exception):
                req_dump = meta.get("inference_request", {})
                msgs = req_dump.get("messages")
                tools_dump = req_dump.get("tools")
                if isinstance(msgs, list):
                    # Print compact messages structure and tool schema with bounded length
                    import json as _json

                    msgs_compact = _json.dumps(msgs)[:20000]
                    tools_compact = (
                        _json.dumps(tools_dump)[:8000] if tools_dump is not None else None
                    )
                    print(
                        {
                            "llm.call": True,
                            "policy": str(policy_name),
                            "messages_preview": msgs_compact,
                            "tools_preview": tools_compact,
                        }
                    )

            # Normalize request for non-OpenAI endpoints (strict schemas)
            with contextlib.suppress(Exception):
                base = str(target_url or "")
                is_openai_dotcom = "openai.com" in base.lower()
                if not is_openai_dotcom:
                    req_body = meta.get("inference_request", {})
                    if isinstance(req_body, dict):
                        # Force structured tool_choice if a bare "required" is present
                        if req_body.get("tool_choice") == "required":
                            func_name = "interact_many"
                            with contextlib.suppress(Exception):
                                tools_arr = req_body.get("tools") or []
                                if isinstance(tools_arr, list) and tools_arr:
                                    f = (
                                        tools_arr[0].get("function")
                                        if isinstance(tools_arr[0], dict)
                                        else None
                                    )
                                    cand = (f or {}).get("name") if isinstance(f, dict) else None
                                    if isinstance(cand, str) and cand:
                                        func_name = cand
                            req_body["tool_choice"] = {
                                "type": "function",
                                "function": {"name": func_name},
                            }
                            req_body["parallel_tool_calls"] = False
                            req_body.setdefault("function_call", {"name": func_name})
                        # Inject extra_body for thinking controls expected by Modal service
                        with contextlib.suppress(Exception):
                            tb = req_body.get("thinking_budget")
                            tm = str(req_body.get("thinking_mode") or "").lower()
                            enable_thinking = bool(tb) or tm == "think"
                            extra = dict(req_body.get("extra_body") or {})
                            chat_kwargs = dict(extra.get("chat_template_kwargs") or {})
                            if enable_thinking:
                                chat_kwargs["enable_thinking"] = True
                            if isinstance(tb, int | float | str) and str(tb).strip():
                                with contextlib.suppress(Exception):
                                    chat_kwargs["thinking_budget"] = int(tb)
                            if chat_kwargs:
                                extra["chat_template_kwargs"] = chat_kwargs
                            # Ensure stop_after_tool_calls honored via extra_body for stricter servers
                            extra.setdefault("stop_after_tool_calls", 1)
                            if extra:
                                req_body["extra_body"] = extra
                        # Provide a conservative default temperature if missing
                        if "temperature" not in req_body:
                            req_body["temperature"] = 0.1
                        meta["inference_request"] = req_body

                # Strip image parts: Crafter policy currently only uses text prompts.
                # Some providers reject image_url payloads entirely, so always flatten to plain text.
                req_body2 = meta.get("inference_request", {})
                if isinstance(req_body2, dict):
                    msgs = req_body2.get("messages")
                    if isinstance(msgs, list):
                        new_msgs = []
                        changed = False
                        for m in msgs:
                            try:
                                if isinstance(m, dict):
                                    content = m.get("content")
                                    if isinstance(content, list):
                                        parts: list[str] = []
                                        for seg in content:
                                            if isinstance(seg, dict):
                                                txt = seg.get("text") or seg.get("content")
                                                if isinstance(txt, str) and txt:
                                                    parts.append(txt)
                                        m2 = dict(m)
                                        m2["content"] = "\n".join(parts)
                                        new_msgs.append(m2)
                                        changed = True
                                    else:
                                        new_msgs.append(m)
                                else:
                                    new_msgs.append(m)
                            except Exception:
                                new_msgs.append(m)
                        if changed:
                            req_body2["messages"] = new_msgs
                            meta["inference_request"] = req_body2

            _t_start = _t.time()
            call_started_at = datetime.utcnow()
            inference_response = await client.generate_with_retries(
                request=meta["inference_request"],
                base_url=meta["inference_url"],
                max_retries=12,
                backoff_factor=2.0,
                extra_headers=extra_headers,
            )
            meta["inference_ms"] = int((_t.time() - _t_start) * 1000)
            call_completed_at = datetime.utcnow()

            provider_url = str(meta.get("inference_url") or "")
            low_url = provider_url.lower()
            if "groq" in low_url:
                provider_name = "groq"
            elif "openai" in low_url:
                provider_name = "openai"
            else:
                provider_name = "custom"

            # Parse response to tool calls
            tool_calls = policy.parse_response_to_tool_calls(
                response=inference_response,
                use_tools=getattr(policy, "use_tools", True),
            )

            # Debug logging (echo tool calls)
            if not tool_calls:
                # Structured error log with small preview; avoid dumping full response repeatedly
                preview = str(inference_response)[:400]
                logger.error(
                    f"TOOLCALL_PARSE_FAIL: policy={policy_name} parsed=0 preview={preview}"
                )
            else:
                try:
                    import json as _json

                    print(
                        {
                            "tool_calls_parsed": int(len(tool_calls)),
                            "tool_calls_preview": _json.dumps(tool_calls)[:20000],
                        }
                    )
                except Exception:
                    logger.info(f"Parsed {len(tool_calls)} tool calls: {tool_calls}")

            # Add response to metadata
            # Parse tool calls from model response using policy-specific parser
            try:
                if hasattr(policy, "parse_response_to_tool_calls"):
                    parsed = policy.parse_response_to_tool_calls(
                        inference_response, getattr(policy, "use_tools", True)
                    )
                else:
                    parsed = policy.parse_model_response(inference_response, request.observation)
                # Replace tool_calls with parsed result
                if isinstance(parsed, list):
                    tool_calls = parsed
                with contextlib.suppress(Exception):
                    logger.info(
                        "TOOLCALL_PARSE: parsed=%d has_tools=%s example=%r",
                        len(tool_calls) if isinstance(tool_calls, list) else -1,
                        bool(getattr(policy, "use_tools", True)),
                        (tool_calls[0] if isinstance(tool_calls, list) and tool_calls else None),
                    )
            except Exception as _pe:
                logger.warning(f"Failed to parse tool calls: {str(_pe)}")
            # Attach raw response + usage for observability
            meta["raw_response"] = inference_response
            if "usage" in inference_response:
                meta["usage"] = inference_response["usage"]

            if tracing_context is not None:
                try:
                    await tracing_context.record_llm_call(
                        inference_request=meta["inference_request"],
                        inference_response=inference_response,
                        tool_calls=tool_calls,
                        provider=provider_name,
                        model_name=model_name,
                        started_at=call_started_at,
                        completed_at=call_completed_at,
                        latency_ms=meta.get("inference_ms"),
                    )
                except Exception as exc:
                    logger.debug(f"TRACING_LLM_FAIL: {exc}")

        return PolicyStepResponse(
            tool_calls=tool_calls,
            meta=meta,
        )

    except Exception as e:
        logger.error(f"Failed to step policy {request.policy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/snapshot", response_model=PolicySnapshotResponse)
async def snapshot_policy(request: PolicySnapshotRequest) -> PolicySnapshotResponse:
    """Create a snapshot of the policy state."""
    handle = registry.get_policy(request.policy_id)
    if not handle:
        raise HTTPException(status_code=404, detail=f"Policy {request.policy_id} not found")

    try:
        # Serialize policy state
        state_dict = await handle.policy.serialize()

        # Save to volume
        snapshot_id, path, size = storage.save_snapshot(
            rl_run_id=handle.rl_run_id,
            kind="policy",
            state_dict=state_dict,
        )

        # Register snapshot
        registry.register_snapshot(
            kind="policy",
            rl_run_id=handle.rl_run_id,
            size=size,
            path=path,
        )

        return PolicySnapshotResponse(
            snapshot_id=snapshot_id,
            path=path,
            rl_run_id=handle.rl_run_id,
            size=size,
        )

    except Exception as e:
        logger.error(f"Failed to snapshot policy {request.policy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/restore", response_model=PolicyRestoreResponse)
async def restore_policy(request: PolicyRestoreRequest) -> PolicyRestoreResponse:
    """Restore a policy from a snapshot."""
    snapshot = registry.get_snapshot(request.snapshot_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail=f"Snapshot {request.snapshot_id} not found")

    if snapshot.kind != "policy":
        raise HTTPException(
            status_code=422,
            detail=f"Snapshot {request.snapshot_id} is not a policy snapshot",
        )

    try:
        # Load snapshot from volume
        state_dict, meta = storage.load_snapshot(
            rl_run_id=snapshot.rl_run_id,
            kind="policy",
            snapshot_id=request.snapshot_id,
        )

        # Recreate policy
        policy_name = state_dict["name"]
        low = policy_name.lower()
        if low in ["crafter-react", "crafter"]:
            policy = await CrafterPolicy.deserialize(state_dict)
        elif low in ["wordle-react", "wordle"]:
            try:
                from .envs.wordle.policy import WordlePolicy
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Wordle policy unavailable: {e}"
                ) from e
            policy = await WordlePolicy.deserialize(state_dict)
        elif low in ["sokoban-react", "sokoban"]:
            try:
                from .envs.sokoban.policy import SokobanPolicy
            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Sokoban policy unavailable: {e}"
                ) from e
            policy = await SokobanPolicy.deserialize(state_dict)
        else:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown policy name in snapshot: {policy_name}",
            )

        # Register new instance
        policy_id = registry.register_policy(
            policy=policy,
            rl_run_id=snapshot.rl_run_id,
        )

        return PolicyRestoreResponse(policy_id=policy_id)

    except Exception as e:
        logger.error(f"Failed to restore policy from snapshot {request.snapshot_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/terminate", response_model=PolicyTerminateResponse)
async def terminate_policy(request: PolicyTerminateRequest) -> PolicyTerminateResponse:
    """Terminate a policy and clean up resources."""
    handle = registry.get_policy(request.policy_id)
    if not handle:
        raise HTTPException(status_code=404, detail=f"Policy {request.policy_id} not found")

    try:
        # Call terminate on the policy
        await handle.policy.terminate()

        # Remove from registry
        registry.remove_policy(request.policy_id)

        return PolicyTerminateResponse(ok=True)

    except Exception as e:
        logger.error(f"Failed to terminate policy {request.policy_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
