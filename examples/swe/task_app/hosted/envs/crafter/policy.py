from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .react_agent import CrafterReActAgent
from .tools import TOOLS_SCHEMA


# Define Policy base class here to avoid circular import
class Policy(ABC):
    """Base class for environment-specific policies."""

    @abstractmethod
    def prepare_inference_request(
        self, observation: dict[str, Any], history: list[dict[str, Any]] = None
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        """Prepare an inference request."""
        pass

    @abstractmethod
    def parse_model_response(
        self, response: str, observation: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Parse model response into tool calls."""
        pass


# (imports moved to top of file to satisfy linter)


class CrafterPolicy(Policy):
    """Thin policy scaffold for Crafter using the ReAct agent prompts.

    This class does not run inference itself. It prepares an inference request
    (messages and optional tools schema) that the Task App can send to the
    inference service, and provides helpers to parse the model response into
    environment tool calls.
    """

    name: str = "crafter-react"

    def __init__(self, inference_url: str, model: str | None = None) -> None:
        self.inference_url = inference_url
        self.model = model
        self.use_tools = True
        # Sampling parameters (populated via initialize(config))
        self.temperature: float | None = None
        self.top_p: float | None = None
        self.max_tokens: int | None = None
        # Thinking controls (populated via initialize(config))
        self.thinking_mode: str | None = None
        self.thinking_budget: int | None = None
        # Rolling conversation and action history for non-Markov policies
        self.history_messages: list[dict[str, str]] = []  # chat-style without system
        self.turn_index: int = 0
        self.trajectory_history: list[dict[str, Any]] = []  # env/policy step records

    async def initialize(self, config: dict[str, Any]) -> None:
        if "inference_url" in config:
            self.inference_url = config["inference_url"]
        if "model" in config:
            self.model = config["model"]
        if "use_tools" in config:
            self.use_tools = bool(config["use_tools"])
        # Adopt sampling params from policy config (trainer passes these through)
        if "temperature" in config:
            self.temperature = float(config["temperature"])  # fail fast on bad types
        if "top_p" in config:
            self.top_p = float(config["top_p"])  # fail fast on bad types
        if "max_tokens" in config:
            self.max_tokens = int(config["max_tokens"])  # fail fast on bad types
        # Thinking mode/budget forwarded into vLLM request (mirrors Wordle policy)
        if "thinking_mode" in config:
            self.thinking_mode = str(config["thinking_mode"])  # expect "think" or "no_think"
        if "thinking_budget" in config and config["thinking_budget"] is not None:
            self.thinking_budget = int(config["thinking_budget"])  # number of tokens inside <think>
        if self.thinking_budget is None:
            try:
                if "openai.com" not in (self.inference_url or "").lower():
                    self.thinking_budget = 1028
            except Exception:
                self.thinking_budget = 1028
        # Reset state on (re)initialize
        self.history_messages = []
        self.turn_index = 0
        self.trajectory_history = []

    def _append_user_observation(self, observation_text: str) -> None:
        self.history_messages.append({"role": "user", "content": observation_text})
        self.turn_index += 1

    def _append_assistant_turn(
        self,
        assistant_text: str | None,
        tool_calls: list[dict[str, Any]] | None,
        env_result: dict[str, Any] | None,
    ) -> None:
        # Record assistant content (if any)
        if assistant_text is not None:
            self.history_messages.append({"role": "assistant", "content": assistant_text})
        # Keep structured step record for training/analysis
        record: dict[str, Any] = {"turn": self.turn_index}
        if tool_calls is not None:
            record["tool_calls"] = tool_calls
        if env_result is not None:
            record["env_result"] = env_result
        self.trajectory_history.append(record)

    def build_inference_request(
        self,
        observation_text: str,
        history: list[dict[str, Any]] | None = None,
        turn: int | None = None,
        image_parts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        messages = CrafterReActAgent.build_messages(
            observation=observation_text,
            history=history,
            turn=turn,
            image_parts=image_parts,
        )
        payload: dict[str, Any] = {
            "messages": messages,
        }
        if self.model is not None:
            payload["model"] = self.model
        # Thinking controls
        if self.thinking_mode is None and "openai.com" not in (self.inference_url or "").lower():
            self.thinking_mode = "think"
        if self.thinking_mode is not None:
            payload["thinking_mode"] = self.thinking_mode
        if self.thinking_budget is None and "openai.com" not in (self.inference_url or "").lower():
            self.thinking_budget = 1028
        if self.thinking_budget is not None:
            payload["thinking_budget"] = self.thinking_budget
        # Inject sampling parameters if set via initialize(config)
        if self.max_tokens is not None:
            # Use max_completion_tokens for newer models, max_tokens for older ones
            if self.model and ("gpt-5" in self.model):
                payload["max_completion_tokens"] = self.max_tokens
            else:
                payload["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.use_tools:
            payload["tools"] = TOOLS_SCHEMA
            payload["tool_choice"] = "required"
            # Ensure the inference server injects family-specific stop sequences
            # to terminate immediately after the first tool call for compliance.
            payload["stop_after_tool_calls"] = 1
        return payload

    @staticmethod
    def parse_response_to_tool_calls(
        response: dict[str, Any],
        use_tools: bool = True,
    ) -> list[dict[str, Any]]:
        """Turn an inference response into environment tool calls.

        - If tools were used, expect tool_calls-compatible output and forward as-is
          in our simple JSON format: {"tool_name": str, "arguments": {...}}.
        - If no tools, parse plain-text actions using CrafterReActAgent parser and
          wrap them into a single interact_many tool call.
        """
        # First check if we got actual tool calls
        choices = response.get("choices", [])
        tool_calls: list[dict[str, Any]] = []

        for choice in choices:
            msg = choice.get("message", {})
            if "tool_calls" in msg and msg["tool_calls"] is not None:
                for tc in msg["tool_calls"]:
                    if tc is None:
                        continue
                    # Handle both OpenAI format and simplified format
                    if "function" in tc:
                        # Standard OpenAI format
                        tool_calls.append(
                            {
                                "tool_name": tc["function"]["name"],
                                "arguments": tc["function"]["arguments"],
                            }
                        )
                    elif "name" in tc:
                        # Simplified format from our vLLM service
                        tool_calls.append(
                            {
                                "tool_name": tc["name"],
                                "arguments": tc["arguments"],
                            }
                        )

        # If we got tool calls, return them
        if tool_calls:
            # Normalize common degenerate pattern ["move_right", "do"] when nothing is nearby.
            # If previous env_result indicates no interaction target, drop trailing 'do'.
            normalized: list[dict[str, Any]] = []
            for tc in tool_calls:
                if tc and isinstance(tc, dict) and tc.get("tool_name") == "interact_many":
                    args = tc.get("arguments")
                    if isinstance(args, str):
                        try:
                            import json

                            args = json.loads(args)
                        except (json.JSONDecodeError, ValueError):
                            args = {}
                    actions = []
                    if isinstance(args, dict):
                        maybe_actions = args.get("actions")
                        if isinstance(maybe_actions, list):
                            actions = maybe_actions
                    # Simple heuristic: avoid repeating same pair; avoid 'do' with no context
                    if len(actions) == 2 and actions[0] == "move_right" and actions[1] == "do":
                        actions = ["move_right"]
                    normalized.append(
                        {"tool_name": "interact_many", "arguments": {"actions": actions or []}}
                    )
                else:
                    normalized.append(tc)
            return normalized

        # Otherwise, parse plain text content for actions
        text = ""
        for choice in choices:
            msg = choice.get("message", {})
            content = msg.get("content", "")
            if content:
                text = content
                break

        if text:
            # Try to parse actions from the text
            from .shared import parse_actions

            actions = parse_actions(text)
            if actions:
                # Wrap actions in interact_many tool call
                return [{"tool_name": "interact_many", "arguments": {"actions": actions}}]

        # No actions found
        return []

    async def step(
        self,
        observation_text: str,
        state: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Stateful step: update policy history and prepare inference request.

        Inputs (via metadata, optional):
        - "prev_assistant_text": str — assistant text from prior turn
        - "prev_tool_calls": List[Dict] — tool calls executed last turn
        - "prev_env_result": Dict — env step result for prior tool calls
        - "prev_inference_response": Dict — raw LLM response; if present and
          use_tools=False, we record assistant_text parsed from content.

        Returns (tool_calls, meta):
        - tool_calls: empty list; coordinator should call inference and then use
          parse_response_to_tool_calls() to derive tool_calls
        - meta: { inference_url, inference_request, turn_index, history_len }
        """
        # If caller provided results from previous cycle, record them first
        if metadata is not None:
            prev_assistant_text: str | None = None
            prev_tool_calls: list[dict[str, Any]] | None = None
            prev_env_result: dict[str, Any] | None = None
            if "prev_assistant_text" in metadata:
                prev_assistant_text = metadata["prev_assistant_text"]
            if "prev_tool_calls" in metadata:
                prev_tool_calls = metadata["prev_tool_calls"]
            if "prev_env_result" in metadata:
                prev_env_result = metadata["prev_env_result"]
            if (
                prev_assistant_text is not None
                or prev_tool_calls is not None
                or prev_env_result is not None
            ):
                self._append_assistant_turn(prev_assistant_text, prev_tool_calls, prev_env_result)

        # Append current observation as the next user message (internal history only)
        self._append_user_observation(observation_text)

        # Build user message by combining the current observation text
        # (formatted surroundings/inventory) with the previous 3 tool calls as context.
        # Most recent first.
        lines: list[str] = []

        def _format_tool_call_line_for_context(
            tool_name: str, arguments: Any, max_chars: int = 500
        ) -> str:
            import json as _json

            # Render arguments compactly, then clip to max_chars
            if isinstance(arguments, dict | list):
                try:
                    rendered = _json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
                except Exception:
                    rendered = str(arguments)
            elif isinstance(arguments, str):
                rendered = arguments
            else:
                rendered = str(arguments)
            if isinstance(rendered, str) and len(rendered) > max_chars:
                rendered = rendered[:max_chars]
            return f"- {tool_name}: {rendered}"

        # Prefer pulling from trajectory_history (accumulates over turns)
        for record in reversed(self.trajectory_history):
            if len(lines) >= 3:
                break
            tc_list = record.get("tool_calls")
            if not tc_list:
                continue
            # Use the first tool call for that turn if multiple exist
            tc = tc_list[0] if isinstance(tc_list, list) and tc_list else None
            if not isinstance(tc, dict):
                continue
            name = tc.get("tool_name") or tc.get("name") or "unknown"
            args = tc.get("arguments")
            lines.append(_format_tool_call_line_for_context(name, args))

        # If trajectory history is empty (first few turns), fall back to metadata once
        if not lines and metadata is not None and metadata.get("prev_tool_calls"):
            calls: list[dict[str, Any]] = metadata["prev_tool_calls"]
            for call in reversed(calls):
                if len(lines) >= 3:
                    break
                if not isinstance(call, dict):
                    continue
                name = call.get("tool_name") or call.get("name") or "unknown"
                args = call.get("arguments")
                lines.append(_format_tool_call_line_for_context(name, args))

        context_text = "Previous tool calls (most recent first):\n" + (
            "\n".join(lines) if lines else "- none"
        )

        # Combine observation with context so the model always sees surroundings/inventory
        combined_text = f"{observation_text}\n\n{context_text}"

        raw_observation: dict[str, Any] | None = None
        if metadata is not None:
            raw_candidate = metadata.get("raw_observation")
            if isinstance(raw_candidate, dict):
                raw_observation = raw_candidate
        image_parts = self._extract_image_parts(raw_observation)

        payload = self.build_inference_request(
            combined_text,
            history=[],  # no prior user/assistant history
            turn=self.turn_index,
            image_parts=image_parts,
        )
        # print("Debugging only:; ", payload)
        meta_out = {
            "inference_url": self.inference_url,
            "inference_request": payload,
            "turn_index": self.turn_index,
            "history_len": len(self.history_messages),
        }
        return [], meta_out

    def state_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "history_messages": self.history_messages,
            "trajectory_history": self.trajectory_history,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.turn_index = int(state["turn_index"])
        self.history_messages = state["history_messages"]
        self.trajectory_history = state["trajectory_history"]

    async def serialize(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "config": {
                "inference_url": self.inference_url,
                "model": self.model,
                "use_tools": self.use_tools,
            },
            "state": self.state_dict(),
        }

    @classmethod
    async def deserialize(cls, payload: dict[str, Any]) -> CrafterPolicy:
        config = payload["config"]
        state = payload["state"]
        policy = cls(
            inference_url=config["inference_url"],
            model=config.get("model"),
        )
        policy.use_tools = bool(config["use_tools"])
        policy.load_state_dict(state)
        return policy

    async def terminate(self) -> None:
        return None

    def prepare_inference_request(
        self, observation: dict[str, Any], history: list[dict[str, Any]] = None
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        """Prepare an inference request (implementing abstract method)."""
        # Format observation with rich contextual information
        observation_text = self._format_observation_for_llm(observation)
        image_parts = self._extract_image_parts(observation)

        # Build messages (observation_text already formatted; no raw matrices)
        messages = CrafterReActAgent.build_messages(
            observation=observation_text,
            history=history,
            turn=self.turn_index,
            image_parts=image_parts,
        )

        # Return messages and tools schema
        tools = TOOLS_SCHEMA if self.use_tools else None
        return messages, tools

    def _format_observation_for_llm(self, observation: dict[str, Any]) -> str:
        """Format observation with rich contextual information for the LLM using the shared formatter."""
        from .shared import format_observation

        # Get the observation data (could be nested)
        obs_data = observation.get("observation", observation)

        # Ensure obs_data is a dict for safe access
        if not isinstance(obs_data, dict):
            return f"Observation: {str(observation)}"

        # Use the shared format_observation function with step information
        step_idx = observation.get("step_idx", 0)
        max_steps = 100  # Default max steps, could be made configurable

        # Get additional info from the observation wrapper
        info = observation.get("info", {})
        if isinstance(info, dict) and "health" in info and "health" not in obs_data:
            obs_data = dict(obs_data)  # Make a copy
            obs_data["health"] = info["health"]

        return format_observation(obs_data, step_count=step_idx, max_steps=max_steps)

    def _extract_image_parts(
        self, observation: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        """Crafter policy uses text-only prompts; do not attach image parts."""

        return []

    def parse_model_response(
        self, response: str, observation: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Parse model response into tool calls (implementing abstract method).

        Note: Despite the type hint, vLLM actually returns a dict response,
        not a string. We handle both cases.
        """
        # Handle dict response from vLLM (the actual case)
        if isinstance(response, dict):
            return self.parse_response_to_tool_calls(response, self.use_tools)

        # Handle string response (fallback case for raw text)
        if isinstance(response, str):
            actions = CrafterReActAgent.parse_actions_from_response(response)
            if actions:
                return [{"tool_name": "interact_many", "arguments": {"actions": actions}}]

        # Default empty response
        return []


__all__ = ["CrafterPolicy"]
