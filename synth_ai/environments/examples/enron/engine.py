# engine.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, Tuple, Optional, List
from pydantic import BaseModel
from pathlib import Path

from synth_ai.environments.examples.enron.art_helpers.types_enron import Email
from synth_ai.environments.examples.enron.art_helpers.email_search_tools import (
    search_emails as helper_search_emails,
    read_email as helper_read_email,
    SearchResult,
)

# SQLite-backed helpers
from synth_ai.environments.stateful.engine import StatefulEngine, StatefulEngineSnapshot
from synth_ai.environments.examples.enron.taskset import EnronTaskInstance
from synth_ai.zyk import LM  # Import LM class

from synth_ai.environments.environment.db.sqlite import SQLiteManager
from synth_ai.environments.environment.rewards.core import RewardStack, RewardComponent
from synth_ai.environments.examples.enron.art_helpers.local_email_db import (
    DEFAULT_DB_PATH,
    generate_database,
)

# --------------------------------------------------------------------------- actions
ACTION_SEARCH = "search"
ACTION_READ = "read"
ACTION_ANSWER = "answer"


# --------------------------------------------------------------------------- snapshot
@dataclass
class EnronEngineSnapshot(StatefulEngineSnapshot):
    idx: int
    answered: bool
    total_reward: float
    partial_rewards: List[float]


# --------------------------------------------------------------------------- engine
class EnronEngine(StatefulEngine):
    """
    Minimal state-machine around the corbt/enron_emails_sample_questions dataset.
    Action is a tuple(kind, arg):

        (ACTION_SEARCH,  query: str)      → returns {"search_results": [message_ids]}
        (ACTION_READ,    message_id: str) → returns {"email_body": str}
        (ACTION_ANSWER,  answer: str)     → rewards +1 / -1 and terminates
    """

    # ----------------------------- init / helpers
    def __init__(self, task_instance: EnronTaskInstance):
        # Use the provided TaskInstance snapshot for this episode
        self.instance = task_instance
        self.answered = False
        self.total_reward = 0.0
        self.idx = 0
        # List to track each step's reward
        self.rewards_history: List[float] = []

        db_file_path = Path(DEFAULT_DB_PATH)
        if not db_file_path.exists():
            generate_database(overwrite=False)  # Ensure DB exists
        self.sqlite_manager = SQLiteManager(db_path=db_file_path, read_only=True)

        # RewardStack is an attribute of the engine; its calculations update private_state fields
        self.reward_stack = RewardStack(
            components=[
                EnronAnswerCorrectnessComponent(),
                EnronStepPenaltyComponent(penalty=-0.05),
            ]
        )
        # This will hold the specific arguments/details of the current agent action
        # for the reward components to inspect.
        self._current_action_details_for_reward: Optional[Dict[str, Any]] = None

    def _sample(self) -> Dict[str, Any]:
        # Return the snapshot dict from the TaskInstance
        return self.instance.initial_engine_snapshot

    # ----------------------------- step / reset
    async def _step_engine(
        self, tool_output_payload: Optional[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        r = await self._calculate_and_apply_reward()

        # Determine termination: if an answer was attempted, task terminates.
        # The 'answered' flag is set by answer_question_action.
        term = self.answered

        s = self._sample()
        priv = {
            "reward_last": r,
            "total_reward": self.total_reward,
            "terminated": term,
            "truncated": False,
            "gold_answer": s["answer"],
        }

        # Public state combines static elements with dynamic ones from tool_output_payload
        pub = {
            "question": s["question"],
            "tools": [
                "search_emails",
                "read_email",
                "answer_question",
                "terminate",
            ],  # Available tools
            "already_answered": self.answered,
            "query_date": s.get("query_date", "<unknown date>"),
            "inbox_address": s.get("inbox_address", "<unknown_inbox>"),
            # Default empty values, to be overwritten by tool_output_payload if present
            "search_results": [],
            "email": None,
            **(tool_output_payload if tool_output_payload else {}),
        }

        return priv, pub

    async def _reset_engine(
        self, *, seed: Optional[int] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Advance to the next Q-A pair and emit an initial observation **without**
        issuing an empty-keyword DB search (which would raise).
        """
        # Reset answered status and total reward for this instance
        self.answered = False
        self.total_reward = 0.0
        self.rewards_history = []
        self._current_action_details_for_reward = None
        # self.sqlite_manager.reset() # Enron DB is read-only; reset usually not needed unless switching DB files.

        s = self._sample()
        priv = {
            "reward_last": 0.0,
            "total_reward": 0.0,
            "terminated": False,
            "truncated": False,
            "gold_answer": s["answer"],
        }
        pub = {
            "question": s["question"],
            "tools": ["search_emails", "read_email", "answer_question", "terminate"],
            "already_answered": False,
            "query_date": s.get("query_date", "<unknown date>"),
            "inbox_address": s.get("inbox_address", "<unknown_inbox>"),
            "search_results": [],
            "email": None,
        }
        # No index advancement needed when using a single TaskInstance
        return priv, pub

    # ----------------------------- serialization helpers
    async def _serialize_engine(self) -> EnronEngineSnapshot:
        # Include partial rewards history in the snapshot
        return EnronEngineSnapshot(
            self.idx,
            self.answered,
            self.total_reward,
            self.rewards_history,
        )

    @classmethod
    async def _deserialize_engine(
        cls, snap: EnronEngineSnapshot, task_instance: EnronTaskInstance
    ) -> "EnronEngine":
        eng = cls(task_instance)
        eng.idx = snap.idx
        eng.answered = snap.answered
        eng.total_reward = snap.total_reward
        eng.rewards_history = (
            snap.partial_rewards
        )  # Ensure this is correctly typed in Pydantic model if not List[float]
        # Note: SQLiteManager is re-initialized in __init__ based on DEFAULT_DB_PATH.
        # If the db path could change per instance/snapshot, that would need to be part of the snapshot.
        return eng

    def close_db(self):
        self.sqlite_manager.close()

    async def _calculate_and_apply_reward(self) -> float:
        s = self._sample()
        reward_context_state = {  # State snapshot for reward calculation
            "question": s["question"],
            "gold_answer": s["answer"],
            **(
                self._current_action_details_for_reward
                if self._current_action_details_for_reward
                else {}
            ),
        }

        # The 'action' param for score can be the conceptual action type or detailed args
        action_param_for_score = (
            self._current_action_details_for_reward
            if self._current_action_details_for_reward
            else {}
        )

        reward = await self.reward_stack.step_reward(
            state=reward_context_state, action=action_param_for_score
        )

        self.total_reward += reward
        self.rewards_history.append(reward)
        self._current_action_details_for_reward = None  # Reset after use
        return reward

    async def search_emails_action(self, search_args: Dict[str, Any]) -> List[Dict[str, Any]]:
        res: List[SearchResult] = helper_search_emails(self.sqlite_manager, **search_args)
        self._current_action_details_for_reward = {"type": "search", **search_args}
        return [asdict(item) for item in res]

    async def read_email_action(self, message_id: str) -> Optional[Dict[str, Any]]:
        email: Optional[Email] = helper_read_email(self.sqlite_manager, message_id)
        self._current_action_details_for_reward = {
            "type": "read",
            "message_id": message_id,
        }
        return email.dict() if email else None

    async def answer_question_action(self, agent_answer: str) -> None:
        # This method now primarily sets up state for reward calculation.
        # The actual reward value and termination status are determined by _get_reward_and_update_state.
        s = self._sample()
        self._current_action_details_for_reward = {
            "type": "answer",
            "is_answer_action": True,  # Signal for reward component
            "question": s["question"],
            "gold_answer": s["answer"],
            "agent_answer": agent_answer,
        }
        self.answered = True  # Mark as answered, termination decided by reward logic


# ----------------------------- LLM Judge for answers
async def determine_if_answer_is_correct(
    question: str, gold_answer: str, agent_answer: str
) -> bool:
    # Instantiate LM for the judge
    llm = LM(model_name="gpt-4.1-nano", formatting_model_name="gpt-4.1-nano", temperature=0.0)

    system_prompt = (
        "You will be given a question and two different answers to the question, "
        "the correct answer and the answer given by an AI. Your job is to determine "
        "if the answer given by the AI is correct."
    )
    user_message_content = (
        f"Question: {question}\nCorrect answer: {gold_answer}\nAI answer: {agent_answer}"
    )

    class CorrectnessResponse(BaseModel):
        correct: bool

    # Use LM.respond_async
    response = await llm.respond_async(
        system_message=system_prompt,
        user_message=user_message_content,
        response_model=CorrectnessResponse,
        # Caching is typically handled within the LM class or its underlying setup
    )
    return response.structured_output.correct


# --- Placeholder Reward Components (ideally defined elsewhere and imported) ---
# (These would typically live in a shared rewards components file or alongside the engine if very specific)
class EnronAnswerCorrectnessComponent(RewardComponent):
    async def score(self, state: Dict[str, Any], action: Any) -> float:
        if state.get("is_answer_action") and state.get("agent_answer") is not None:
            # determine_if_answer_is_correct should be part of the engine or accessible
            # For now, assuming it's available in this scope.
            correct = await determine_if_answer_is_correct(
                state["question"], state["gold_answer"], state["agent_answer"]
            )
            return 1.0 if correct else -1.0
        return 0.0


class EnronStepPenaltyComponent(RewardComponent):
    def __init__(self, penalty: float = -0.01):
        self.penalty = penalty

    async def score(self, state: Dict[str, Any], action: Any) -> float:
        # Apply penalty for any action that isn't a final answer, or just every step.
        # For simplicity, apply if not a "correct" answer action.
        if not state.get("is_answer_action"):
            return self.penalty
        return 0.0
