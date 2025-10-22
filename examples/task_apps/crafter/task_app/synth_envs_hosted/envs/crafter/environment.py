from __future__ import annotations

import base64
import logging
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image
from synth_ai.environments.environment.tools import EnvToolCall
from synth_ai.environments.stateful.core import StatefulEnvironment

from ...utils import convert_numpy_to_python
from .shared import CRAFTER_ACTIONS, _format_semantic_map_view
from .tools import TOOLS_SCHEMA

logger = logging.getLogger(__name__)


def _encode_image_to_base64(image_array: Any) -> dict[str, Any] | None:
    """Encode an RGB ndarray into a base64 PNG payload with metadata."""

    if not isinstance(image_array, np.ndarray):
        return None
    if image_array.ndim != 3 or image_array.shape[-1] not in (1, 3, 4):
        return None
    try:
        # Ensure uint8 for PIL compatibility
        array_uint8 = (
            image_array.astype("uint8")
            if image_array.dtype != np.uint8
            else image_array  # pragma: no cover - fast path
        )
        mode = "L" if array_uint8.shape[-1] == 1 else "RGB"
        if array_uint8.shape[-1] == 4:
            mode = "RGBA"
        img = Image.fromarray(array_uint8, mode=mode)
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        width = int(array_uint8.shape[1])
        height = int(array_uint8.shape[0])
        return {
            "format": "png",
            "width": width,
            "height": height,
            "data": encoded,
            "data_url": f"data:image/png;base64,{encoded}",
        }
    except Exception:
        return None


class CrafterEnvironmentWrapper:
    """Host-side environment wrapper matching the sketch contract.

    Bridges our HTTP routes to a synth-ai `StatefulEnvironment` instance.

    Contract (see sketch.txt):
      - initialize() -> observation dict
      - step(tool_calls: List[EnvToolCall]) -> observation dict plus optional done/reward/truncated/info
      - snapshot()/restore() handled at route level; this wrapper exposes checkpoint via synth-ai
    """

    def __init__(self, env: StatefulEnvironment, seed: int | None = None) -> None:
        self.env = env
        self.seed = seed
        self.step_idx = 0
        self.last_observation: dict[str, Any] | None = None
        self.last_info: dict[str, Any] | None = None

    async def initialize(self) -> dict[str, Any]:
        obs = await self.env.initialize()
        # synth-ai InternalObservation expected to expose .observation (dict-like)
        self.step_idx = 0
        self.last_observation = getattr(obs, "observation", obs)  # tolerate dict-like
        self.last_info = getattr(obs, "info", None)
        out_obs = self._prepare_observation(self.last_observation)
        # Attach a 7x7 semantic map patch centered on player for client-side rendering
        try:
            pub = self.env.engine._get_public_state_from_env()  # type: ignore[attr-defined]
            sem = pub.semantic_map
            px, py = list(pub.player_position)
            size = 7
            half = size // 2
            patch = []
            height = len(sem) if hasattr(sem, "__len__") else 0
            width = len(sem[0]) if height and hasattr(sem[0], "__len__") else 0
            for dy in range(-half, half + 1):
                row = []
                for dx in range(-half, half + 1):
                    x, y = int(px) + dx, int(py) + dy
                    if 0 <= x < height and 0 <= y < width:
                        row.append(int(sem[x][y]))
                    else:
                        row.append(0)
                patch.append(row)
            if isinstance(out_obs, dict):
                out_obs["semantic_map_patch7"] = patch
        except Exception:
            pass
        return {
            "observation": out_obs,
            "info": convert_numpy_to_python(self.last_info) if self.last_info else None,
            "step_idx": self.step_idx,
        }

    async def step(self, tool_calls: list[dict[str, Any]] | list[EnvToolCall]) -> dict[str, Any]:
        # Normalize JSON tool_calls into EnvToolCall instances if needed
        # Underlying synth-ai environment expects only tool="interact" with args={"action": <action_name>}.
        # LLM may emit:
        # - interact_many with {actions: [...]}
        # - direct tool names like "make_wood_pickaxe" or "do"
        # - or even tool_name "do" with arguments {"action": "make_wood_pickaxe"}
        # We normalize all these into a sequence of EnvToolCall(tool="interact", args={"action": <resolved_action>}).
        allowed_actions = set(
            TOOLS_SCHEMA[0]["function"]["parameters"]["properties"]["actions"]["items"]["enum"]
        )
        normalized: list[EnvToolCall] = []

        def _action_to_int(action: Any) -> int | None:
            # Handle invalid actions gracefully instead of failing
            if isinstance(action, int):
                return action
            action_str = str(action)
            if action_str not in CRAFTER_ACTIONS:
                logger.warning("Unknown Crafter action: %s - ignoring", action_str)
                return None  # Signal to skip this action
            return CRAFTER_ACTIONS[action_str]

        for tc in tool_calls:
            if isinstance(tc, EnvToolCall):
                # Expand interact_many; otherwise coerce non-interact tools into interact(action=tool)
                if tc.tool == "interact_many":
                    actions = tc.args.get("actions", [])
                    for action in actions:
                        action_int = _action_to_int(action)
                        if action_int is not None:  # Skip invalid actions
                            normalized.append(
                                EnvToolCall(tool="interact", args={"action": action_int})
                            )
                elif tc.tool != "interact":
                    candidate_action = tc.args.get("action") if isinstance(tc.args, dict) else None
                    resolved_action = (
                        candidate_action if candidate_action in allowed_actions else tc.tool
                    )
                    action_int = _action_to_int(resolved_action)
                    if action_int is not None:  # Skip invalid actions
                        normalized.append(EnvToolCall(tool="interact", args={"action": action_int}))
                else:
                    normalized.append(tc)
            else:
                # Dict input: handle both "tool" and "tool_name" keys
                tool_name = tc.get("tool") or tc.get("tool_name")
                if not tool_name:
                    raise ValueError(f"Tool call missing tool name: {tc}")
                # Extract/parse args (may be JSON string from some clients)
                args = tc.get("arguments") or tc.get("args") or {}
                if isinstance(args, str):
                    import json as _json

                    try:
                        args = _json.loads(args)
                    except Exception:
                        args = {}
                # Expand interact_many into multiple interacts
                if tool_name == "interact_many":
                    for action in args.get("actions") or []:
                        action_int = _action_to_int(action)
                        if action_int is not None:  # Skip invalid actions
                            normalized.append(
                                EnvToolCall(tool="interact", args={"action": action_int})
                            )
                else:
                    # For any non-interact tool, resolve to an interact action.
                    # Support a packed list of actions under 'actions' for convenience.
                    if (
                        isinstance(args, dict)
                        and isinstance(args.get("actions"), list)
                        and args.get("actions")
                    ):
                        for action in args.get("actions"):
                            action_int = _action_to_int(action)
                            if action_int is not None:
                                normalized.append(
                                    EnvToolCall(tool="interact", args={"action": action_int})
                                )
                    else:
                        candidate_action = None
                        if isinstance(args, dict) and "action" in args:
                            candidate_action = args["action"]
                        # If the caller provided a numeric action id, accept it directly
                        action_int: int | None
                        if isinstance(candidate_action, int) or (
                            isinstance(candidate_action, str)
                            and candidate_action in allowed_actions
                        ):
                            action_int = _action_to_int(candidate_action)
                        else:
                            # Fallback: interpret the tool name itself as the action label
                            action_int = _action_to_int(tool_name)
                        if action_int is not None:
                            normalized.append(
                                EnvToolCall(tool="interact", args={"action": action_int})
                            )

        # Ensure we have at least one valid action; default to noop if none provided
        if not normalized:
            logger.info("No valid actions provided, defaulting to noop")
            normalized.append(EnvToolCall(tool="interact", args={"action": 0}))  # noop action

        # Pre-step logging: capture current public state and print concise summary
        before_state: dict[str, Any] | None = None
        try:
            pub_before = self.env.engine._get_public_state_from_env()  # type: ignore[attr-defined]
            before_state = {
                "inventory": pub_before.inventory,
                "achievements_status": pub_before.achievements_status,
                "player_position": list(pub_before.player_position),
                "player_direction": pub_before.player_direction,
                "semantic_map": pub_before.semantic_map,
            }
            actions_printable = [
                (tc.args.get("action") if isinstance(tc.args, dict) else None)
                if isinstance(tc, EnvToolCall)
                else None
                for tc in normalized
            ]
            logger.info(
                "Crafter BEFORE seed=%s step_idx=%s pos=%s inv=%s ach=%s actions=%s",
                str(self.seed),
                self.step_idx,
                before_state.get("player_position"),
                {k: v for k, v in before_state["inventory"].items() if v},
                [k for k, v in before_state["achievements_status"].items() if v],
                actions_printable,
            )
            logger.info(
                "Surroundings BEFORE (seed=%s):\n%s",
                str(self.seed),
                _format_semantic_map_view(before_state),
            )
        except Exception as _:
            # Logging should not interfere with stepping; fail-fast elsewhere
            pass

        if not normalized:
            raise ValueError("No valid actions provided to CrafterEnvironmentWrapper.step()")

        # Execute actions sequentially so multi-action tool calls actually advance the world
        last_obs: Any = None
        for single_call in normalized:
            last_obs = await self.env.step(single_call)
            self.step_idx += 1

        obs = last_obs
        observation = getattr(obs, "observation", obs)
        info = getattr(obs, "info", None)
        done = getattr(obs, "done", False)  # Default to False if None
        reward = getattr(obs, "reward", None)
        truncated = getattr(obs, "truncated", None)
        self.last_observation = observation
        self.last_info = info

        # Post-step logging: capture new public state and print concise summary
        ach_added_latest: list[str] | None = None
        try:
            pub_after = self.env.engine._get_public_state_from_env()  # type: ignore[attr-defined]
            after_dict: dict[str, Any] = {
                "inventory": pub_after.inventory,
                "achievements_status": pub_after.achievements_status,
                "player_position": list(pub_after.player_position),
                "player_direction": pub_after.player_direction,
                "semantic_map": pub_after.semantic_map,
            }
            logger.info(
                "Crafter AFTER seed=%s step_idx=%s pos=%s inv=%s ach=%s done=%s reward=%s",
                str(self.seed),
                self.step_idx,
                after_dict.get("player_position"),
                {k: v for k, v in after_dict["inventory"].items() if v},
                [k for k, v in after_dict["achievements_status"].items() if v],
                bool(done) if done is not None else False,
                reward,
            )

            # Changes/diff summary (position and inventory)
            if before_state is not None:
                try:
                    # Position delta
                    pb = before_state.get("player_position", [0, 0])
                    pa = after_dict.get("player_position", [0, 0])
                    pb_t = (int(pb[0]), int(pb[1])) if isinstance(pb, list | tuple) else (0, 0)
                    pa_t = (int(pa[0]), int(pa[1])) if isinstance(pa, list | tuple) else (0, 0)
                    delta = (pa_t[0] - pb_t[0], pa_t[1] - pb_t[1])

                    # Inventory changes
                    inv_b = before_state.get("inventory", {}) or {}
                    inv_a = after_dict.get("inventory", {}) or {}
                    changed_items = []
                    all_keys = set(inv_b.keys()) | set(inv_a.keys())
                    for key in sorted(all_keys):
                        vb = int(inv_b.get(key, 0) or 0)
                        va = int(inv_a.get(key, 0) or 0)
                        if vb != va:
                            changed_items.append(f"{key}:{vb}->{va}(Δ{va - vb})")
                    inv_changes = ", ".join(changed_items) if changed_items else "none"

                    # Achievements gained/lost
                    ach_b = {
                        k
                        for k, v in (before_state.get("achievements_status", {}) or {}).items()
                        if v
                    }
                    ach_a = {
                        k for k, v in (after_dict.get("achievements_status", {}) or {}).items() if v
                    }
                    ach_added = sorted(ach_a - ach_b)
                    ach_added_latest = ach_added
                    ach_removed = sorted(ach_b - ach_a)

                    logger.info(
                        "Changes: pos %s->%s Δ=%s | inv %s | ach +%s -%s",
                        pb_t,
                        pa_t,
                        delta,
                        inv_changes,
                        ach_added if ach_added else [],
                        ach_removed if ach_removed else [],
                    )
                    # Reward shaping immediately so logs and response reflect it
                    if reward is None and ach_added_latest:
                        try:
                            reward = float(len(ach_added_latest))
                            logger.info(
                                "Reward shaping applied: +%s (achievements added)",
                                len(ach_added_latest),
                            )
                        except Exception:
                            pass
                except Exception:
                    pass
            logger.info(
                "Surroundings AFTER (seed=%s):\n%s",
                str(self.seed),
                _format_semantic_map_view(after_dict),
            )
        except Exception as _:
            pass
        result: dict[str, Any] = {
            "observation": self._prepare_observation(observation),
            "step_idx": self.step_idx,
            "done": bool(done) if done is not None else False,  # Ensure boolean
        }
        # Attach a 7x7 semantic map patch centered on player for client-side rendering
        try:
            sem = after_dict.get("semantic_map")
            pos = after_dict.get("player_position") or [0, 0]
            px, py = int(pos[0]), int(pos[1])
            size = 7
            half = size // 2
            patch = []
            height = len(sem) if hasattr(sem, "__len__") else 0
            width = len(sem[0]) if height and hasattr(sem[0], "__len__") else 0
            for dy in range(-half, half + 1):
                row = []
                for dx in range(-half, half + 1):
                    x, y = px + dx, py + dy
                    if 0 <= x < height and 0 <= y < width:
                        row.append(int(sem[x][y]))
                    else:
                        row.append(0)
                patch.append(row)
            obs_out = result.get("observation")
            if isinstance(obs_out, dict):
                obs_out["semantic_map_patch7"] = patch
        except Exception:
            pass
        result_info = convert_numpy_to_python(info) if info is not None else {}
        # Attach achievements delta for downstream metrics if useful
        if ach_added_latest is not None:
            try:
                if not isinstance(result_info, dict):
                    result_info = {"_raw_info": result_info}
                result_info["achievements_added"] = ach_added_latest
            except Exception:
                pass
        if result_info:
            result["info"] = result_info
        if reward is not None:
            result["reward"] = convert_numpy_to_python(reward)
            # Also expose last-step reward inside observation for stepwise consumers
            try:
                obs_out = result.get("observation")
                if isinstance(obs_out, dict):
                    obs_out.setdefault("reward_last_step", convert_numpy_to_python(reward))
            except Exception:
                pass
        if truncated is not None:
            result["truncated"] = truncated

        # Aggregated step summary: action frequencies and achievement stats
        try:
            # Build reverse action map for readability
            int_to_action = {v: k for k, v in CRAFTER_ACTIONS.items()}
            from collections import Counter

            action_ids = []
            for tc in normalized:
                if isinstance(tc, EnvToolCall) and isinstance(tc.args, dict):
                    a = tc.args.get("action")
                    if isinstance(a, int):
                        action_ids.append(a)
            action_names = [int_to_action.get(a, str(a)) for a in action_ids]
            action_freq = Counter(action_names)

            # Public achievements after step
            pub_after = self.env.engine._get_public_state_from_env()  # type: ignore[attr-defined]
            unlocked = [name for name, on in pub_after.achievements_status.items() if on]
            ach_freq = Counter(unlocked)

            # Private achievement values (means)
            priv_after = self.env.engine._get_private_state_from_env(0.0, False, False)  # type: ignore[attr-defined]
            values = list((priv_after.achievements_current_values or {}).values())
            mean_all = (sum(values) / len(values)) if values else 0.0
            nonzero = [v for v in values if v]
            mean_nonzero = (sum(nonzero) / len(nonzero)) if nonzero else 0.0

            logger.info(
                "Step summary: seed=%s | actions=%s | achievements=%s | mean_ach_all=%.3f mean_ach_nonzero=%.3f",
                str(self.seed),
                dict(action_freq),
                dict(ach_freq),
                mean_all,
                mean_nonzero,
            )
        except Exception:
            pass

        return result

    def _prepare_observation(self, observation: Any) -> dict[str, Any]:
        """Convert raw observation into a JSON-serializable dict with encoded image."""

        obs_dict: dict[str, Any]
        image_payload: dict[str, Any] | None = None

        if isinstance(observation, dict):
            image_payload = _encode_image_to_base64(observation.get("observation_image"))
            # Work on a shallow copy to avoid mutating engine state
            sanitized = dict(observation)
            sanitized.pop("observation_image", None)
            obs_dict = convert_numpy_to_python(sanitized) or {}
        else:
            obs_dict = convert_numpy_to_python(observation) or {}

        if not isinstance(obs_dict, dict):
            obs_dict = {"value": obs_dict}

        if image_payload:
            obs_dict["observation_image_base64"] = image_payload["data"]
            obs_dict["observation_image_format"] = image_payload["format"]
            obs_dict["observation_image_width"] = image_payload["width"]
            obs_dict["observation_image_height"] = image_payload["height"]
            obs_dict["observation_image_data_url"] = image_payload["data_url"]

        return obs_dict

    async def checkpoint(self) -> dict[str, Any]:
        obs = await self.env.checkpoint()
        observation = getattr(obs, "observation", obs)
        info = getattr(obs, "info", None)
        return {
            "observation": convert_numpy_to_python(observation),
            "info": convert_numpy_to_python(info) if info else None,
            "step_idx": self.step_idx,
        }

    async def terminate(self) -> dict[str, Any]:
        obs = await self.env.terminate()
        observation = getattr(obs, "observation", obs)
        info = getattr(obs, "info", None)
        return {
            "observation": convert_numpy_to_python(observation),
            "info": convert_numpy_to_python(info) if info else None,
            "step_idx": self.step_idx,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "step_idx": self.step_idx,
            "last_observation": self.last_observation,
            "last_info": self.last_info,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.seed = state["seed"]
        self.step_idx = int(state["step_idx"])
        self.last_observation = state["last_observation"]
        self.last_info = state["last_info"]

    async def serialize(self) -> dict[str, Any]:
        return {
            "name": "crafter",
            "config": {"seed": self.seed},
            "state": self.state_dict(),
        }

    @classmethod
    async def deserialize(
        cls,
        payload: dict[str, Any],
        env: StatefulEnvironment,
    ) -> CrafterEnvironmentWrapper:
        seed = payload["config"]["seed"]
        wrapper = cls(env=env, seed=seed)
        wrapper.load_state_dict(payload["state"])
        return wrapper


__all__ = ["CrafterEnvironmentWrapper"]
