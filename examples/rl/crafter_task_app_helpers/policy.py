from __future__ import annotations

from typing import Any, List, Dict
from synth_ai.environments.examples.crafter_classic.engine import (
    CRAFTER_ACTION_MAP,  # map of action name to int
)

# # Strict Crafter action space (match env mapping in helpers/env.py)
# ACTIONS: List[str] = [
#     "noop",
#     "move_left",
#     "move_right",
#     "move_up",
#     "move_down",
#     "do",
#     "sleep",
# ]
ACTIONS = list(CRAFTER_ACTION_MAP.keys())


class CrafterPolicy:
    def __init__(self, *, inference_url: str, model: str | None = None) -> None:
        self.inference_url = inference_url.rstrip("/")
        self.model = model

    def build_inference_request(self, obs_text: str, *, history: List[Dict] | None, turn: int) -> Dict[str, Any]:
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "interact",
                    "description": "Perform actions in the Crafter environment.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "actions": {
                                "type": "array",
                                "items": {"type": "string", "enum": ACTIONS},
                                "description": "List of actions to perform in sequence (e.g., ['move_right', 'move_right', 'do']). Available actions: " + ", ".join(ACTIONS),
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Reasoning for these actions",
                            },
                        },
                        "required": ["actions", "reasoning"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "terminate",
                    "description": "End the episode when finished or no progress can be made.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {"type": "string", "description": "Reason for termination"}
                        },
                        "required": ["reason"],
                    },
                },
            }
        ]
        messages = [
            {"role": "system", "content": "You are an AI agent playing Crafter. Your goal is to unlock achievements by collecting resources and completing tasks.\n\nCRITICAL RULES:\n- You MUST provide MULTIPLE actions (2-5) in EVERY interact() tool call!\n- Monitor your achievements_progress and use the terminate() tool when you've achieved significant progress (at least 3 achievements)\n- Focus on exploration and resource collection to unlock achievements\n\nChoose actions from: " + ", ".join(ACTIONS)},
            {"role": "user", "content": obs_text},
        ]
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": {"type": "function", "function": {"name": "interact"}},
            "stop_after_tool_calls": 1,
            "temperature": 0.2,
            "max_tokens": 256,
        }
        return body

    @staticmethod
    def parse_response_to_tool_calls(resp: Dict[str, Any], *, use_tools: bool = True) -> List[Dict[str, Any]]:
        try:
            # Validate response structure
            if not isinstance(resp, dict):
                raise ValueError("Response must be a dictionary")

            choices = resp.get("choices") or []
            if not choices:
                raise ValueError("Response contains no choices")

            msg = choices[0].get("message") or {}
            if not isinstance(msg, dict):
                raise ValueError("Message must be a dictionary")

            if use_tools:
                tcs = msg.get("tool_calls") or []
                if not isinstance(tcs, list):
                    raise ValueError("tool_calls must be a list")

                out = []
                for i, tc in enumerate(tcs):
                    if not isinstance(tc, dict):
                        raise ValueError(f"Tool call {i} must be a dictionary")

                    # Validate tool call structure
                    f = tc.get("function")
                    if f is None:
                        raise ValueError(f"Tool call {i} missing 'function' key")
                    if not isinstance(f, dict):
                        raise ValueError(f"Tool call {i} 'function' must be a dictionary")

                    # Validate required function keys
                    tool_name = f.get("name")
                    if tool_name is None:
                        raise ValueError(f"Tool call {i} function missing 'name' key")
                    if not isinstance(tool_name, str) or not tool_name.strip():
                        raise ValueError(f"Tool call {i} function 'name' must be a non-empty string")

                    arguments = f.get("arguments")
                    if arguments is None:
                        raise ValueError(f"Tool call {i} function missing 'arguments' key")
                    if not isinstance(arguments, str):
                        raise ValueError(f"Tool call {i} function 'arguments' must be a string")

                    # Parse and validate arguments as JSON
                    try:
                        import json
                        parsed_args = json.loads(arguments)
                        if not isinstance(parsed_args, dict):
                            raise ValueError(f"Tool call {i} arguments must parse to a dictionary")
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Tool call {i} arguments is not valid JSON: {e}")

                    out.append({"tool_name": tool_name, "arguments": arguments})

                return out

            # fallback: parse free-form text for actions
            content = msg.get("content") or ""
            if not isinstance(content, str):
                raise ValueError("Message content must be a string")

            actions = []
            action_keywords = [
                "move_up", "move_down", "move_left", "move_right",
                "do", "sleep", "place_stone", "place_table", "place_furnace", "place_plant",
                "make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe",
                "make_wood_sword", "make_stone_sword", "make_iron_sword"
            ]

            for keyword in action_keywords:
                if keyword in content.lower():
                    actions.append(keyword)

            # Ensure we have at least one action
            if not actions:
                actions = ["do"]

            return [{"tool_name": "interact", "arguments": {"actions": actions, "reasoning": "Parsed from response"}}]

        except Exception as e:
            # Re-raise with more context
            raise ValueError(f"Failed to parse tool calls: {e}") from e


