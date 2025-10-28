from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from jinja2 import StrictUndefined, Template
from minisweagent.agents.default import FormatError

from .shared import (
    DEFAULT_ACTION_TEMPLATE,
    DEFAULT_INSTANCE_TEMPLATE,
    DEFAULT_SYSTEM_TEMPLATE,
)
from .tools import RUN_COMMAND_TOOL, SUBMIT_TOOL, TOOLS_SCHEMA

logger = logging.getLogger(__name__)

COMMAND_PATTERN = re.compile(r"```(?:bash)?\s*\n(.*?)\n```", re.DOTALL)


def _render_template(source: str, **kwargs: Any) -> str:
    return Template(source, undefined=StrictUndefined).render(**kwargs)


@dataclass
class MiniSwePolicyConfig:
    system_template: str = DEFAULT_SYSTEM_TEMPLATE
    instance_template: str = DEFAULT_INSTANCE_TEMPLATE
    action_template: str = DEFAULT_ACTION_TEMPLATE
    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_completion_tokens: int | None = None
    tool_choice: str = "required"
    use_tools: bool = True
    step_limit: int = 0
    cost_limit: float = 3.0
    extra_template_vars: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> MiniSwePolicyConfig:
        base = MiniSwePolicyConfig()
        for key in (
            "system_template",
            "instance_template",
            "action_template",
            "model",
            "temperature",
            "top_p",
            "max_completion_tokens",
            "tool_choice",
            "use_tools",
            "step_limit",
            "cost_limit",
        ):
            if key in payload:
                setattr(base, key, payload[key])
        extra = payload.get("extra_template_vars") or {}
        if isinstance(extra, dict):
            base.extra_template_vars = dict(extra)
        return base


class MiniSwePolicy:
    """Mini-SWE policy that mirrors the default agent prompt loop."""

    name = "swe-mini"

    def __init__(self, *, inference_url: str | None = None, model: str | None = None) -> None:
        self.inference_url = inference_url
        self.config = MiniSwePolicyConfig(model=model)
        self.system_template = Template(self.config.system_template, undefined=StrictUndefined)
        self.instance_template = Template(self.config.instance_template, undefined=StrictUndefined)
        self.action_template = Template(self.config.action_template, undefined=StrictUndefined)

        self.messages: list[dict[str, Any]] = []
        self.turn_index = 0
        self.history_messages: list[dict[str, Any]] = []
        self.trajectory_history: list[dict[str, Any]] = []
        self.task: dict[str, Any] | None = None
        self.template_vars: dict[str, Any] = {}

    async def initialize(self, payload: dict[str, Any]) -> None:
        cfg = MiniSwePolicyConfig.from_payload(payload or {})
        self.config = cfg
        self.system_template = Template(cfg.system_template, undefined=StrictUndefined)
        self.instance_template = Template(cfg.instance_template, undefined=StrictUndefined)
        self.action_template = Template(cfg.action_template, undefined=StrictUndefined)
        if cfg.model:
            self.config.model = cfg.model
        self.template_vars = dict(cfg.extra_template_vars or {})
        logger.info("Mini-swe policy initialized with model=%s", self.config.model)
        self._reset_state()

    def _reset_state(self) -> None:
        self.messages = []
        self.history_messages = []
        self.trajectory_history = []
        self.turn_index = 0

    def _append_user(self, content: str) -> None:
        msg = {"role": "user", "content": content}
        self.messages.append(msg)
        self.history_messages.append(msg)
        self.turn_index += 1

    def _append_assistant(self, content: str) -> None:
        msg = {"role": "assistant", "content": content}
        self.messages.append(msg)
        self.history_messages.append(msg)

    def _apply_previous_cycle(self, metadata: dict[str, Any] | None) -> None:
        if not metadata:
            return
        prev_tool_calls = metadata.get("prev_tool_calls")
        prev_response = metadata.get("prev_inference_response")
        prev_env_result = metadata.get("prev_env_result")
        prev_assistant_text = metadata.get("prev_assistant_text")

        if prev_assistant_text:
            self._append_assistant(prev_assistant_text)
        elif prev_response:
            text = self._extract_response_text(prev_response)
            if text:
                self._append_assistant(text)

        if prev_tool_calls or prev_env_result:
            record = {
                "turn": self.turn_index,
                "tool_calls": prev_tool_calls,
                "env_result": prev_env_result,
            }
            self.trajectory_history.append(record)

    def _ensure_task_context(self, observation: dict[str, Any] | None) -> None:
        if self.task is not None:
            return
        task = (observation or {}).get("task") or {}
        self.task = dict(task)
        render_vars = dict(self.template_vars)
        render_vars.setdefault("task", task)
        render_vars.setdefault("problem_statement", task.get("problem_statement", ""))
        render_vars.setdefault("instructions", task.get("instructions", ""))
        render_vars.setdefault("metadata", task.get("metadata", {}))
        rendered_system = self.system_template.render(**render_vars)
        rendered_user = self.instance_template.render(**render_vars)
        self.messages.append({"role": "system", "content": rendered_system})
        self.history_messages.append({"role": "system", "content": rendered_system})
        self._append_user(rendered_user)

    def _render_action_observation(self, observation: dict[str, Any]) -> str:
        last = observation.get("last") or {}
        output = {
            "stdout": last.get("stdout", ""),
            "returncode": last.get("returncode", 0),
        }
        template_input = {"output": output, "observation": observation}
        return self.action_template.render(**template_input)

    def _extract_response_text(self, response: dict[str, Any]) -> str:
        try:
            choices = response.get("choices") or []
            for choice in choices:
                msg = choice.get("message") or {}
                content = msg.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts = []
                    for entry in content:
                        if isinstance(entry, dict):
                            txt = entry.get("text") or entry.get("content")
                            if isinstance(txt, str):
                                parts.append(txt)
                    if parts:
                        return "".join(parts)
        except Exception:
            pass
        return ""

    def _build_inference_request(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"messages": self.messages}
        if self.config.model:
            payload["model"] = self.config.model
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if self.config.top_p is not None:
            payload["top_p"] = self.config.top_p
        if self.config.max_completion_tokens is not None:
            payload["max_completion_tokens"] = self.config.max_completion_tokens
        if self.config.use_tools:
            model_name = str(self.config.model or "").lower()
            if "gpt-5" in model_name:
                # GPT-5 models insist on a single tool; keep run_command to avoid shim calls.
                tool_list: list[dict[str, Any]] = [RUN_COMMAND_TOOL]
                payload["tools"] = tool_list
                payload["tool_choice"] = {
                    "type": "function",
                    "function": {"name": "run_command"},
                }
                payload["parallel_tool_calls"] = False
            else:
                # Groq/Qwen and other OpenAI-compatible models handle both tools under auto mode.
                tool_list = [RUN_COMMAND_TOOL, SUBMIT_TOOL]
                payload["tools"] = tool_list
                payload["tool_choice"] = "auto"
                payload["parallel_tool_calls"] = False
        return payload

    async def step(
        self,
        observation_text: str,
        state: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        raw_observation: dict[str, Any] | None = None
        if metadata is not None:
            candidate = metadata.get("raw_observation")
            if isinstance(candidate, dict):
                raw_observation = candidate

        self._ensure_task_context(raw_observation)
        self._apply_previous_cycle(metadata)

        message_text = observation_text or ""
        if raw_observation and raw_observation.get("last"):
            rendered = self._render_action_observation(raw_observation)
            message_text = f"{message_text}\n\n{rendered}" if message_text else rendered
        elif not message_text:
            message_text = (
                "Observation: repository ready. Begin by inspecting files and planning next steps."
            )

        self._append_user(message_text)

        inference_request = self._build_inference_request()
        meta = {
            "inference_request": inference_request,
            "turn_index": self.turn_index,
            "history_len": len(self.history_messages),
            "tool_schema": TOOLS_SCHEMA,
        }
        if self.inference_url:
            meta["inference_url"] = self.inference_url

        return [], meta

    @staticmethod
    def _parse_command_from_text(text: str) -> str:
        matches = COMMAND_PATTERN.findall(text or "")
        if len(matches) != 1:
            raise FormatError(
                "Please provide exactly one bash command enclosed in a single ```bash``` block."
            )
        command = matches[0].strip()
        if not command:
            raise FormatError("Command block was empty. Provide a valid shell command.")
        return command

    def parse_response_to_tool_calls(
        self,
        response: dict[str, Any],
        use_tools: bool = True,
    ) -> list[dict[str, Any]]:
        if use_tools:
            # Prefer structured tool calls if available.
            for choice in response.get("choices", []):
                msg = choice.get("message") or {}
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    parsed: list[dict[str, Any]] = []
                    for tool in tool_calls:
                        if not isinstance(tool, dict):
                            continue
                        name = tool.get("name")
                        args = tool.get("arguments")
                        if "function" in tool:
                            name = tool["function"].get("name")
                            args = tool["function"].get("arguments")
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {"command": args}
                        parsed.append({"tool_name": name, "arguments": args})
                    if parsed:
                        return parsed

        text = self._extract_response_text(response)
        if not text:
            logger.warning("Model response missing content; defaulting to echo NOOP")
            return [{"tool_name": "run_command", "arguments": {"command": "echo NOOP"}}]

        try:
            command = self._parse_command_from_text(text)
        except FormatError as err:
            logger.warning("Format error parsing command: %s; defaulting to echo NOOP", err)
            return [{"tool_name": "run_command", "arguments": {"command": "echo NOOP"}}]

        return [{"tool_name": "run_command", "arguments": {"command": command}}]

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "messages": self.messages,
            "history_messages": self.history_messages,
            "trajectory_history": self.trajectory_history,
            "turn_index": self.turn_index,
            "task": self.task,
            "template_vars": self.template_vars,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.config = MiniSwePolicyConfig.from_payload(state.get("config", {}))
        self.system_template = Template(
            self.config.system_template, undefined=StrictUndefined
        )
        self.instance_template = Template(
            self.config.instance_template, undefined=StrictUndefined
        )
        self.action_template = Template(self.config.action_template, undefined=StrictUndefined)
        self.messages = state.get("messages", [])
        self.history_messages = state.get("history_messages", [])
        self.trajectory_history = state.get("trajectory_history", [])
        self.turn_index = int(state.get("turn_index", 0))
        self.task = state.get("task")
        self.template_vars = state.get("template_vars", {})

    async def serialize(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "config": asdict(self.config),
            "state": self.state_dict(),
        }

    @classmethod
    async def deserialize(cls, payload: dict[str, Any]) -> MiniSwePolicy:
        config = payload.get("config") or {}
        state = payload.get("state") or {}
        policy = cls(
            inference_url=config.get("inference_url"),
            model=config.get("model"),
        )
        await policy.initialize(config)
        policy.load_state_dict(state)
        return policy

    async def terminate(self) -> None:
        return None


__all__ = ["MiniSwePolicy"]
