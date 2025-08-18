# environment.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from synth_ai.environments.environment.shared_engine import (
    GetObservationCallable,
    InternalObservation,
)
from synth_ai.environments.environment.tools import (
    TOOL_REGISTRY,
    EnvToolCall,
    ToolResult,
    register_tool,
)
from synth_ai.environments.examples.enron.engine import (
    ACTION_ANSWER,
    ACTION_READ,
    ACTION_SEARCH,
    EnronEngine,
)
from synth_ai.environments.examples.enron.taskset import EnronTaskInstance
from synth_ai.environments.stateful.core import StatefulEnvironment


# -------- pydantic schemas (used by agent / LLM function calls)
class SearchEmailsArgs(BaseModel):
    inbox: str = Field(..., description="Email address performing the search (used by tool logic)")
    keywords: List[str] = Field(..., description="Keywords to AND-search for")
    from_addr: Optional[str] = None
    to_addr: Optional[str] = None
    sent_after: Optional[str] = None
    sent_before: Optional[str] = None
    max_results: int = Field(10, le=10)


class ReadEmailArgs(BaseModel):
    message_id: str


class AnswerQuestionArgs(BaseModel):
    answer: str


# --------------------------------------------------------------------------- tool wrappers
class SearchEmails(EnvToolCall):
    def __init__(self, **kwargs):
        self.action = (ACTION_SEARCH, kwargs)


class ReadEmail(EnvToolCall):
    def __init__(self, message_id: str):
        self.action = (ACTION_READ, message_id)


class AnswerQuestion(EnvToolCall):
    def __init__(self, answer: str):
        self.action = (ACTION_ANSWER, answer)


# -- terminate wrapper (maps to an empty-answer ACTION_ANSWER) --------------
class Terminate(EnvToolCall):
    def __init__(self):
        self.action = (ACTION_ANSWER, "")


# -------- observation callable (optional for formatted observations)
class SynthEnronObservationCallable(GetObservationCallable):
    async def get_observation(
        self, pub: Dict[str, Any], priv: Dict[str, Any]
    ) -> InternalObservation:
        """Format observation as a human-readable string."""
        q = pub.get("question")
        rwd = priv.get("reward_last")
        return f"Q: {q}\nTools: {pub.get('tools')}\nAnswered: {pub.get('already_answered')}\nSearch Res: {len(pub.get('search_results', []))} items\nEmail Loaded: {pub.get('email') is not None}\nTool Error: {pub.get('tool_error')}\nReward Δ: {rwd}"


# --------------------------------------------------------------------------- environment
class EnronEnvironment(StatefulEnvironment):
    def __init__(
        self,
        task_instance: EnronTaskInstance,
        custom_obs: Optional[GetObservationCallable] = None,
    ):
        self.engine = EnronEngine(task_instance)
        self.custom_obs = custom_obs or SynthEnronObservationCallable()
        self.name = "Enron-QA-Env"

        # Store tool instances on self for reliable access
        self._tools_instances = {
            "search_emails": SearchEmailsTool(self.engine),
            "read_email": ReadEmailTool(self.engine),
            "answer_question": AnswerQuestionTool(self.engine),
            "terminate": TerminateTool(self.engine),
        }
        for tool_name, tool_instance in self._tools_instances.items():
            if tool_name not in TOOL_REGISTRY:
                register_tool(tool_instance)
            elif TOOL_REGISTRY[tool_name].engine is not self.engine:
                register_tool(tool_instance)

    async def initialize(self) -> InternalObservation:
        priv, pub = await self.engine._reset_engine()
        return await self._obs(priv, pub)

    async def step(
        self,
        calls: Union[EnvToolCall, List[EnvToolCall], List[List[EnvToolCall]]],
    ) -> InternalObservation:
        # normalise → always [[EnvToolCall]]
        if isinstance(calls, EnvToolCall):
            calls = [[calls]]
        elif calls and isinstance(calls[0], EnvToolCall):
            calls = [calls]

        if not isinstance(calls[0][0], EnvToolCall):
            raise TypeError(f"Processed call is not EnvToolCall: {type(calls[0][0])}")

        tool_name = calls[0][0].tool
        tool_to_execute = self._tools_instances.get(tool_name)

        if not tool_to_execute:
            tool_to_execute = TOOL_REGISTRY.get(tool_name)
            if not tool_to_execute:
                raise ValueError(f"Tool '{tool_name}' not found.")

        tool_result: ToolResult = await tool_to_execute(calls[0][0])

        public_payload_for_engine = (
            tool_result.payload if tool_result.ok and tool_result.payload else {}
        )
        if not tool_result.ok:
            public_payload_for_engine["tool_error"] = tool_result.error

        priv, pub = await self.engine._step_engine(public_payload_for_engine)
        return await self._obs(priv, pub)

    async def terminate(self) -> InternalObservation:
        self.engine.close_db()
        priv_state_on_terminate = {
            "reward_last": 0,
            "total_reward": self.engine.total_reward,
            "terminated": True,
            "truncated": False,
            "gold_answer": self.engine._sample()["answer"],
        }
        pub_state_on_terminate = {
            "question": self.engine._sample()["question"],
            "tools": [],
            "already_answered": self.engine.answered,
            "status": "terminated_by_env",
        }
        return await self._obs(priv_state_on_terminate, pub_state_on_terminate)

    async def checkpoint(self) -> InternalObservation:
        snapshot = await self.engine._serialize_engine()
        return {
            "engine_snapshot": snapshot.model_dump(),
            "message": "Checkpoint created",
        }

    async def _obs(self, priv: Dict[str, Any], pub: Dict[str, Any]):
        if self.custom_obs:
            return await self.custom_obs.get_observation(pub, priv)
        return {**pub, **priv}
