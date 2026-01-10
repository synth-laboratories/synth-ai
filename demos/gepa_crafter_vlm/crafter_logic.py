from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import crafter
except ImportError as exc:  # pragma: no cover - demo-only dependency
    raise RuntimeError(
        "Crafter demo requires the 'crafter' package (pip install crafter>=1.8.3)."
    ) from exc


CRAFTER_ACTIONS = [
    "noop",
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "do",
    "sleep",
    "place_stone",
    "place_table",
    "place_furnace",
    "place_plant",
    "make_wood_pickaxe",
    "make_stone_pickaxe",
    "make_iron_pickaxe",
    "make_wood_sword",
    "make_stone_sword",
    "make_iron_sword",
    "make_wood_axe",
    "make_stone_axe",
    "make_iron_axe",
    "make_wood_wall",
    "make_stone_wall",
    "make_iron_wall",
    "make_wood_door",
    "make_stone_door",
    "make_iron_door",
]

CRAFTER_ALLOWED_ACTIONS = [
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "do",
    "sleep",
    "place_stone",
    "place_table",
    "place_furnace",
    "place_plant",
    "make_wood_pickaxe",
    "make_stone_pickaxe",
    "make_iron_pickaxe",
    "make_wood_sword",
    "make_stone_sword",
    "make_iron_sword",
    "noop",
]
CRAFTER_ALLOWED_ACTIONS_SET = set(CRAFTER_ALLOWED_ACTIONS)

ACTION_STRING_TO_INT = {name: idx for idx, name in enumerate(CRAFTER_ACTIONS)}
INT_TO_ACTION_STRING = {idx: name for name, idx in ACTION_STRING_TO_INT.items()}


def normalize_action_name(action: str) -> Optional[str]:
    if not isinstance(action, str):
        return None
    action_norm = action.strip().lower()
    if not action_norm:
        return None
    if action_norm in CRAFTER_ALLOWED_ACTIONS_SET:
        return action_norm

    tokens = set(re.findall(r"[a-z]+", action_norm))
    if {"sleep", "rest"} & tokens:
        return "sleep" if "sleep" in CRAFTER_ALLOWED_ACTIONS_SET else None

    if "table" in tokens and {"place", "build"} & tokens:
        return "place_table" if "place_table" in CRAFTER_ALLOWED_ACTIONS_SET else None
    if "furnace" in tokens and {"place", "build"} & tokens:
        return "place_furnace" if "place_furnace" in CRAFTER_ALLOWED_ACTIONS_SET else None
    if "plant" in tokens and {"place", "build", "grow"} & tokens:
        return "place_plant" if "place_plant" in CRAFTER_ALLOWED_ACTIONS_SET else None
    if "stone" in tokens and {"place", "build"} & tokens:
        return "place_stone" if "place_stone" in CRAFTER_ALLOWED_ACTIONS_SET else None

    if "pickaxe" in tokens:
        if "wood" in tokens:
            return "make_wood_pickaxe"
        if "stone" in tokens:
            return "make_stone_pickaxe"
        if "iron" in tokens:
            return "make_iron_pickaxe"
    if "sword" in tokens:
        if "wood" in tokens:
            return "make_wood_sword"
        if "stone" in tokens:
            return "make_stone_sword"
        if "iron" in tokens:
            return "make_iron_sword"

    if tokens & {
        "interact",
        "collect",
        "gather",
        "mine",
        "chop",
        "attack",
        "fight",
        "harvest",
        "do",
        "pick",
        "grab",
    }:
        return "do"

    if {"left", "west"} & tokens:
        return "move_left"
    if {"right", "east"} & tokens:
        return "move_right"
    if {"up", "north", "forward"} & tokens:
        return "move_up"
    if {"down", "south", "back", "backward"} & tokens:
        return "move_down"

    if tokens & {"move", "walk", "go", "explore"}:
        return "move_up"

    return None


def _encode_image_to_data_url(image: Any) -> Optional[str]:
    if not isinstance(image, np.ndarray):
        return None
    if image.ndim != 3 or image.shape[-1] not in (1, 3, 4):
        return None
    array_uint8 = image.astype("uint8") if image.dtype != np.uint8 else image
    mode = "L" if array_uint8.shape[-1] == 1 else "RGB"
    if array_uint8.shape[-1] == 4:
        mode = "RGBA"
    img = Image.fromarray(array_uint8, mode=mode)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _extract_observation_image(obs: Any) -> Any:
    if isinstance(obs, np.ndarray):
        return obs
    if isinstance(obs, dict):
        for key in ("image", "observation", "obs", "pixels", "frame"):
            if key in obs:
                return obs.get(key)
    return None


def _extract_achievements(info: Dict[str, Any]) -> List[str]:
    achievements = info.get("achievements")
    if isinstance(achievements, dict):
        return [k for k, v in achievements.items() if v]
    if isinstance(achievements, list):
        return [str(item) for item in achievements]
    return []


def _extract_health(info: Dict[str, Any]) -> float:
    for key in ("health", "player_health", "hp"):
        value = info.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _extract_inventory(info: Dict[str, Any]) -> Dict[str, int]:
    inventory = info.get("inventory")
    if isinstance(inventory, dict):
        return {str(k): int(v) for k, v in inventory.items() if isinstance(v, (int, float))}
    return {}


@dataclass
class CrafterEnvironmentWrapper:
    """Wrapper for Crafter environment that provides image observations."""

    seed: int
    max_steps: int = 200

    def __post_init__(self) -> None:
        self._env = crafter.Env()
        self._step_count = 0
        self._last_info: Dict[str, Any] = {}

    def _reset_env(self) -> Tuple[Any, Dict[str, Any]]:
        try:
            obs, info = self._env.reset(seed=self.seed)
        except TypeError:
            obs = self._env.reset()
            info = {}
        return obs, info

    async def reset(self) -> Dict[str, Any]:
        obs, info = self._reset_env()
        self._step_count = 0
        self._last_info = info if isinstance(info, dict) else {}
        image_url = _encode_image_to_data_url(_extract_observation_image(obs))
        return {
            "observation_image_data_url": image_url,
            "seed": self.seed,
            "step_count": self._step_count,
            "terminated": False,
            "truncated": False,
        }

    async def step(self, action: int) -> Dict[str, Any]:
        result = self._env.step(action)
        if isinstance(result, tuple) and len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            obs, reward, done, info = result
            terminated = bool(done)
            truncated = False

        self._step_count += 1
        info = info if isinstance(info, dict) else {}
        self._last_info = info
        image_url = _encode_image_to_data_url(_extract_observation_image(obs))

        return {
            "observation_image_data_url": image_url,
            "reward": float(reward) if isinstance(reward, (int, float)) else 0.0,
            "step_count": self._step_count,
            "terminated": bool(terminated) or self._step_count >= self.max_steps,
            "truncated": bool(truncated) or self._step_count >= self.max_steps,
            "achievements": _extract_achievements(info),
            "health": _extract_health(info),
            "inventory": _extract_inventory(info),
        }

    def get_achievements(self) -> List[str]:
        return _extract_achievements(self._last_info)


class CrafterVLMReActPolicy:
    """VLM ReAct policy for Crafter - image-only mode."""

    def __init__(
        self, system_prompt: str, *, use_vision: bool = True, image_only_mode: bool = True
    ) -> None:
        self.system_prompt = system_prompt
        self.use_vision = use_vision
        self.image_only_mode = image_only_mode
        self.tools = [self._build_crafter_interact_tool()]

    def _build_crafter_interact_tool(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "crafter_interact",
                "description": "Execute actions in Crafter environment",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "actions_list": {
                            "type": "array",
                            "items": {"type": "string", "enum": CRAFTER_ALLOWED_ACTIONS},
                            "minItems": 2,
                            "maxItems": 5,
                            "description": "List of 2-5 action names from the allowed action set.",
                        },
                        "reasoning": {"type": "string", "description": "Why these actions"},
                    },
                    "required": ["actions_list", "reasoning"],
                },
            },
        }

    def build_messages(
        self,
        observation: Dict[str, Any],
        history: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        messages.extend(history)

        user_content: List[Dict[str, Any]] = []
        image_url = observation.get("observation_image_data_url")
        if self.image_only_mode:
            if image_url:
                user_content.append({"type": "image_url", "image_url": {"url": image_url}})
        else:
            summary = self._format_text_summary(observation)
            user_content.append({"type": "text", "text": summary})
            if image_url:
                user_content.append({"type": "image_url", "image_url": {"url": image_url}})

        if user_content:
            messages.append({"role": "user", "content": user_content})

        return messages

    def _format_text_summary(self, observation: Dict[str, Any]) -> str:
        inventory = observation.get("inventory") or {}
        achievements = observation.get("achievements") or []
        return (
            f"step={observation.get('step_count', 0)} "
            f"health={observation.get('health', 0)} "
            f"inventory={inventory} "
            f"achievements={achievements} "
            f"terminated={observation.get('terminated', False)} "
            f"truncated={observation.get('truncated', False)}"
        )


class CrafterScorer:
    """Score Crafter episode performance."""

    @staticmethod
    def score_episode(
        final_observation: Dict[str, Any],
        episode_length: int,
        max_steps: int,
    ) -> Tuple[float, Dict[str, Any]]:
        achievements = final_observation.get("achievements", [])
        health = final_observation.get("health", 0.0)
        terminated = final_observation.get("terminated", False)
        truncated = final_observation.get("truncated", False)

        achievement_score = min(len(achievements) / 15.0, 1.0)
        survival_score = 1.0 if (terminated or truncated) and health > 0 else 0.5
        health_score = min(float(health) / 100.0, 1.0) if isinstance(health, (int, float)) else 0.0
        efficiency_score = min(float(episode_length) / float(max_steps or 1), 1.0)

        total_score = (
            0.4 * achievement_score
            + 0.3 * survival_score
            + 0.2 * health_score
            + 0.1 * efficiency_score
        )

        details = {
            "achievements": achievements,
            "achievement_count": len(achievements),
            "health": health,
            "episode_length": episode_length,
            "terminated": terminated,
            "truncated": truncated,
        }

        return total_score, details
