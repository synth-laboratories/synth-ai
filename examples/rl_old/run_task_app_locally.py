#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List


def add_example_path() -> None:
    here = Path(__file__).resolve()
    ex_rl = here.parent
    if str(ex_rl) not in sys.path:
        sys.path.append(str(ex_rl))


add_example_path()

# Local imports from examples/rl package (canonical helpers name)
from crafter_task_app_helpers import EnvRegistry, CrafterPolicy  # type: ignore


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Crafter env locally and print policy prompts")
    p.add_argument("--steps", type=int, default=30, help="Max total env steps to execute")
    p.add_argument("--seed", type=int, default=0, help="Environment seed")
    p.add_argument("--actions", type=str, nargs="*", default=[], help="(Optional) seed actions to start the rollout")
    p.add_argument("--model", type=str, default="gpt-5-mini", help="Model label (for payload display)")
    p.add_argument("--openai-url", type=str, default="https://api.openai.com", help="Base URL for OpenAI API")
    return p.parse_args()


def _format_obs(obs: dict[str, Any]) -> str:
    """Format observation exactly like the task app does (no raw matrices)."""
    if not isinstance(obs, dict):
        return "no salient state; explore to gather context"
    inv = obs.get("inventory") or {}
    pos = obs.get("player_position")
    steps = obs.get("num_steps_taken")
    direction = obs.get("player_direction")
    ach = obs.get("achievements_status") or {}
    inv_lines = ", ".join(f"{k}:{v}" for k, v in inv.items() if v)
    ach_on = [k for k, v in ach.items() if v]
    lines: List[str] = []
    if pos is not None:
        px, py = int(pos[0]), int(pos[1])
        lines.append(f"position: (x={px}, y={py})")
    if direction is not None:
        dx, dy = int(direction[0]), int(direction[1])
        dir_label = {
            (1, 0): "→ east/right",
            (-1, 0): "← west/left",
            (0, 1): "↓ south/down",
            (0, -1): "↑ north/up",
            (0, 0): "• idle",
        }.get((dx, dy), f"({dx},{dy})")
        lines.append(f"direction: {dir_label}")
    if steps is not None:
        lines.append(f"steps: {int(steps)}")
    if inv_lines:
        lines.append(f"inventory: {inv_lines}")
    if ach:
        all_achievements = list(ach.keys())
        lines.append(f"achievements_available: {', '.join(all_achievements)}")
        if ach_on:
            lines.append(f"achievements_unlocked: {', '.join(ach_on)}")
            lines.append(f"achievements_progress: {len(ach_on)}/{len(all_achievements)}")

    # Local surroundings (7x7) using semantic_map with id->name mapping
    smap = obs.get("semantic_map")
    if smap is not None and pos is not None:
        try:
            import numpy as np  # lazy import
            import crafter as _crafter  # type: ignore
            import itertools as _it

            px, py = int(pos[0]), int(pos[1])
            view_size = 7
            half = view_size // 2
            # build dynamic id->name map
            dummy = _crafter.Env()
            try:
                max_id = max(max(dummy._world._mat_ids.values()), max(dummy._sem_view._obj_ids.values())) + 1
                id_to_item = ["void"] * max_id
                for name, ind in _it.chain(dummy._world._mat_ids.items(), dummy._sem_view._obj_ids.items()):
                    if name is None:
                        clean = "none"
                    elif hasattr(name, "__name__"):
                        clean = name.__name__
                    else:
                        clean = str(name)
                    id_to_item[ind] = clean.lower()
            finally:
                try:
                    dummy.close()
                except Exception:
                    pass

            sm = np.asarray(smap)
            matrix: List[List[str]] = []
            for dy in range(-half, half + 1):
                row: List[str] = []
                for dx in range(-half, half + 1):
                    x, y = px + dx, py + dy
                    if not (0 <= x < sm.shape[0] and 0 <= y < sm.shape[1]):
                        row.append("void")
                    elif dx == 0 and dy == 0:
                        row.append("player")
                    else:
                        idx = int(sm[x, y])
                        name = id_to_item[idx] if 0 <= idx < len(id_to_item) else str(idx)
                        row.append(name)
                matrix.append(row)
            transposed = list(zip(*matrix))
            grid_rows = [" ".join(r) for r in transposed]
            if grid_rows:
                lines.append("Local Map View (7x7):\n" + "\n".join(grid_rows))
        except Exception:
            pass
    if not lines:
        lines.append("no salient state; explore to gather context")
    return "\n".join(lines)


async def main() -> None:
    args = build_args()
    import os
    import httpx
    reg = EnvRegistry()
    cfg: dict[str, Any] = {"seed": int(args.seed)}
    env_id, obs = await reg.initialize(cfg)

    policy = CrafterPolicy(inference_url="https://api.openai.com", model=args.model)

    # Local sanitizer mirroring task_app prepare_inference_payload_for_model
    OPENAI_MAX_COMPLETION_TOKENS_MIN = 16000
    OPENAI_REMOVE_FIELDS = ("stop_after_tool_calls", "thinking_mode", "thinking_budget", "reasoning")
    OPENAI_REMOVE_SAMPLING_FIELDS = ("temperature", "top_p")

    def sanitize_payload(model: str | None, payload: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(payload)
        # Always remove unsupported fields
        for k in OPENAI_REMOVE_FIELDS:
            if k in out:
                out.pop(k)
        if model and "gpt-5" in model:
            if "max_completion_tokens" not in out and "max_tokens" in out:
                out["max_completion_tokens"] = out.pop("max_tokens")
            if "max_tokens" in out:
                out.pop("max_tokens")
            for k in OPENAI_REMOVE_SAMPLING_FIELDS:
                if k in out:
                    out.pop(k)
            mct = out.get("max_completion_tokens")
            if not isinstance(mct, int) or mct < OPENAI_MAX_COMPLETION_TOKENS_MIN:
                out["max_completion_tokens"] = OPENAI_MAX_COMPLETION_TOKENS_MIN
            out["tool_choice"] = {"type": "function", "function": {"name": "interact"}}
            out["parallel_tool_calls"] = False
        return out

    prev_tool_calls: List[dict] = []

    steps_executed = 0
    while steps_executed < int(args.steps):
        obs_text = _format_obs(obs)
        context_lines = [f"- {tc.get('tool_name')}: {tc.get('arguments')}" for tc in prev_tool_calls[-3:]]
        context_text = "Previous tool calls (most recent first):\n" + ("\n".join(reversed(context_lines)) if context_lines else "- none")
        combined_text = f"Current observation:\n{obs_text}\n\n{context_text}"
        payload = policy.build_inference_request(combined_text, history=[], turn=steps_executed)
        messages = payload.get("messages", [])
        print("\n========== PROMPT #", steps_executed + 1, "==========", sep="")
        for m in messages:
            role = m.get("role")
            content = m.get("content")
            if role in ("system", "user"):
                print(f"\n[{role.upper()}]\n{content}")
        # Call OpenAI locally (no Modal), sanitize payload
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Missing OPENAI_API_KEY in environment for local rollout")
        to_send = sanitize_payload(args.model, payload)
        headers = {"Authorization": f"Bearer {key}"}
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(f"{args.openai_url.rstrip('/')}/v1/chat/completions", json=to_send, headers=headers)
            if resp.status_code >= 400:
                print("[local] LLM ERR:", resp.text[:800])
                break
            data = resp.json()
        # Parse tool calls and print them fully
        parsed = CrafterPolicy.parse_response_to_tool_calls(data, use_tools=True) or []
        import json as _json
        for idx, tc in enumerate(parsed):
            try:
                print(f"[local] tool_call[{idx}] ", _json.dumps(tc, separators=(",", ":")))
            except Exception:
                print(f"[local] tool_call[{idx}] ", tc)
            print(f"[DEBUG] tool_call keys: {list(tc.keys())}")
            print(f"[DEBUG] tool_name: {tc.get('tool_name')}, arguments: {tc.get('arguments')}")
        # Execute tool calls
        if not parsed:
            break

        # Add successful tool calls to history for next iteration
        for tc in parsed:
            prev_tool_calls.append(tc)

        # Helper: normalize action strings to env names
        def _normalize_action_name(name: str) -> str:
            n = str(name).strip()
            # Strict allowed set
            allowed = {"noop","move_left","move_right","move_up","move_down","do","sleep"}
            if n in allowed:
                return n
            # common LLM aliases
            alias = {
                "south": "move_down",
                "north": "move_up",
                "east": "move_right",
                "west": "move_left",
            }
            return alias.get(n, "noop")

        for tc in parsed:
            tool_name = (tc.get("tool_name") or tc.get("name") or "").strip()
            if tool_name == "interact":
                # Parse arguments (may be JSON string)
                args_obj = tc.get("arguments")
                if isinstance(args_obj, str):
                    try:
                        args_obj = _json.loads(args_obj)
                    except Exception as e:
                        raise AssertionError(f"tool_call.arguments is string but not JSON: {args_obj}") from e
                assert isinstance(args_obj, dict), f"tool_call.arguments must be dict, got {type(args_obj)}"
                actions = args_obj.get("actions") or []
                reasoning = args_obj.get("reasoning", "")
                print(f"[local] reasoning: {reasoning}")
                assert isinstance(actions, list), f"actions must be a list, got {type(actions)}"
                assert 1 <= len(actions) <= 5, f"actions length out of bounds: {len(actions)}"
                # Execute normalized actions individually
                norm_actions = [_normalize_action_name(a) for a in actions]
                print(f"[local] executing actions: {norm_actions}")
                for act in norm_actions:
                    assert isinstance(act, str) and act, f"invalid action: {act}"
                    obs, reward, done, _info = await reg.step(env_id, act)
                    steps_executed += 1
                    print(f"\n[action] {act} -> r={float(reward)} done={bool(done)}")

                    # Check for achievement-based termination
                    if isinstance(obs, dict):
                        current_achievements = obs.get("achievements_status", {})
                        achieved_count = sum(1 for v in current_achievements.values() if v)
                        total_achievements = len(current_achievements)

                        # Terminate if we've achieved a significant portion of available achievements
                        if total_achievements > 0 and achieved_count >= max(3, total_achievements // 2):
                            print(f"[local] achievement_termination: {achieved_count}/{total_achievements} achievements reached")
                            print(f"[local] achieved: {[k for k, v in current_achievements.items() if v]}")
                            return

                    if steps_executed >= int(args.steps) or done:
                        break
                if steps_executed >= int(args.steps) or done:
                    break
            elif tool_name == "terminate":
                print(f"[local] Agent requested termination")
                break
            else:
                print(f"[local][warn] unsupported tool '{tool_name}', skipping")

    await reg.terminate(env_id)


if __name__ == "__main__":
    asyncio.run(main())


