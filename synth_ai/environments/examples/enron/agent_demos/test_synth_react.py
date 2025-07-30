# react_agent.py  ── minimal ReAct agent for the new tools (LLM wiring identical to Sokoban pattern)
# Combined with tests_eval_enron.py
import asyncio
import json
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List
import textwrap
import os

import pytest
from pydantic import BaseModel

from synth_ai.zyk import LM
from synth_sdk.tracing.abstractions import Dataset, RewardSignal, TrainingQuestion
from synth_ai.environments.environment.tools import EnvToolCall

from synth_ai.environments.examples.enron.engine import ACTION_ANSWER
from synth_ai.environments.examples.enron.environment import (
    AnswerQuestion,
    AnswerQuestionArgs,
    EnronEnvironment,
    ReadEmail,
    ReadEmailArgs,
    SearchEmails,
    SearchEmailsArgs,
    Terminate,
)
from synth_ai.environments.examples.enron.taskset import create_enron_taskset
from synth_ai.environments.examples.enron.art_helpers import local_email_db, email_search_tools

# ensure SQLite email database exists in dataset directory
# align database path with HF dataset cache folder
DATASET_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
os.makedirs(DATASET_DIR, exist_ok=True)
DB_PATH = os.path.join(DATASET_DIR, "enron_emails.db")
local_email_db.DEFAULT_DB_PATH = DB_PATH
email_search_tools.DEFAULT_DB_PATH = DB_PATH
if not os.path.exists(DB_PATH):
    local_email_db.generate_database(overwrite=False)


# ---- schemas for function-calling LLM
class TerminateArgs(BaseModel):
    reason: str


# ---- ReAct Agent
class ReActEnronAgent:
    def __init__(self, llm: LM, max_steps: int = 8, tool_window: int = 12):
        self.llm, self.max_steps = llm, max_steps
        self.history: Deque[Dict[str, Any]] = deque(maxlen=20)
        self.system_name, self.system_instance_id = "enron-react", str(uuid.uuid4())
        self.tool_window = tool_window
        self.tool_history: Deque[Dict[str, Any]] = deque(maxlen=self.tool_window)

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_emails",
                    "description": (
                        "Full-text search over the inbox. "
                        "`keywords` **must** be a list of individual words "
                        '— e.g. ["Jeff","Skilling","Enron","stock"]. '
                        "Do NOT wrap whole sentences or use quotes."
                    ),
                    "parameters": SearchEmailsArgs.model_json_schema(),
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_email",
                    "description": "Read a single email by message-id",
                    "parameters": ReadEmailArgs.model_json_schema(),
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "answer_question",
                    "description": "Final answer to the user's question",
                    "parameters": AnswerQuestionArgs.model_json_schema(),
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "terminate",
                    "description": "Stop the episode",
                    "parameters": TerminateArgs.model_json_schema(),
                },
            },
        ]

    async def act(self, observation: Dict[str, Any]) -> EnvToolCall:
        # --- build prompt -------------------------------------------------
        # ① never leak evaluation labels to the LLM
        obs_filtered = {k: v for k, v in observation.items() if k != "gold_answer"}
        self.history.append({"obs": obs_filtered})

        # ─── dynamic context pulled from the latest env observation ───
        user_email = observation.get("inbox_address", "<unknown>")
        user_query = observation.get("question", "")
        today_string = observation.get("query_date", "<unknown date>")

        # ----- expose the *current* search hits (max 5) to the LLM -----
        sr = observation.get("search_results", [])
        # ② keep hit list in-sync; clear it on 0-hit searches
        self.last_hits = sr
        hits = getattr(self, "last_hits", [])
        hits_block = (
            "\n".join(
                f"{i + 1}. {h.get('message_id', 'N/A')} : "
                f"{(h.get('snippet', '') or '')[:120].replace(chr(10), ' ')}…"
                for i, h in enumerate(hits[:10])
            )
            if hits
            else "No search results yet."
        )

        # ----- expose a short excerpt of the last-opened email ----------
        em = observation.get("email")
        if em and isinstance(em, dict) and em.get("body"):
            email_excerpt = em["body"][:10000].replace("\n", " ") + "…"
        else:
            email_excerpt = "No email opened yet."

        # system prompt: role, tool rules *and* context -------------------
        history_block = self._format_tool_history()
        system_message = textwrap.dedent(f'''
            You are an email-search agent.

            • When calling **search_emails** pass *individual* words in `keywords`.
              Example → `search_emails(keywords=["Jeff","Skilling","sell","Enron","stock"])`
              (never a whole sentence or use quotes).

            • If a search returns 0 results, try different terms or read a promising
              message-id.  

            You may take up to {self.max_steps} turns; finish with
            `answer_question` once confident.

            If an email already contains the answer, IMMEDIATELY finish with  
                `answer_question(answer="…")`.
            • When calling `answer_question`, return only the exact answer sentence verbatim as it appears in the source; do not add any extra explanation or text.

            Recent tool history:
            {history_block}

            Context  
            ────────  
            • Inbox you can query: **{user_email}**  
            • Today's date: **{today_string}**  

            Original user question:  
            """{user_query}"""  

            Latest search hits:
            {hits_block}

            Latest email excerpt:
            {email_excerpt}
        ''').strip()

        user_message = json.dumps({"history": list(self.history)})

        resp = await self.llm.respond_async(
            system_message=system_message,
            user_message=user_message,
            tools=self.tools,
        )
        if not resp.tool_calls:
            self.history.append({"tool": "no_op", "args": "LLM returned no tool calls."})
            return AnswerQuestion("")

        primary_action_to_execute = None

        for i, tc in enumerate(resp.tool_calls):
            if isinstance(tc, dict):
                # Response from a model that returns dicts (e.g. some OSS models)
                fc = tc.get("function", {})
                name, args_json_str = fc.get("name"), fc.get("arguments")
            elif hasattr(tc, "function"):
                # Response from OpenAI, Anthropic (object with .function attribute)
                name = tc.function.name
                args_json_str = tc.function.arguments
            else:
                self.history.append(
                    {
                        "tool": "unknown_format",
                        "raw_tool_call": str(tc),
                        "error": "Unknown tool call format",
                    }
                )
                if i == 0:
                    primary_action_to_execute = AnswerQuestion(
                        ""
                    )  # Fallback if first call is bad format
                continue

            if not name or args_json_str is None:
                print(
                    f"Tool call {i}: Missing name or arguments. Name: '{name}', Args: '{args_json_str}'. Skipping."
                )
                self.history.append(
                    {
                        "tool": name or "unknown_name",
                        "args_str": args_json_str,
                        "error": "Missing name or arguments",
                    }
                )
                if i == 0:
                    primary_action_to_execute = AnswerQuestion(
                        ""
                    )  # Fallback if first call is malformed
                continue

            try:
                args = json.loads(args_json_str)
            except json.JSONDecodeError as e:
                print(f"Tool call {i} ({name}): JSON decode error: {e}. Args: '{args_json_str}'")
                self.history.append(
                    {
                        "tool": name,
                        "args_str": args_json_str,
                        "error": "JSONDecodeError",
                        "detail": str(e),
                    }
                )
                if i == 0:
                    primary_action_to_execute = AnswerQuestion("")
                continue

            current_tool_env_call = None
            history_entry_for_this_tool = {"tool": name, "args": args}

            if name == "search_emails":
                try:
                    parsed = SearchEmailsArgs(**args)
                    if parsed.max_results is None or parsed.max_results < 10:
                        parsed.max_results = 10
                    history_entry_for_this_tool["args"] = parsed.model_dump()
                    current_tool_env_call = SearchEmails(**parsed.model_dump())
                except Exception as e:
                    print(f"Tool call {i} ({name}): Args parsing error: {e}")
                    history_entry_for_this_tool["error"] = (
                        f"SearchEmailsArgs parsing error: {str(e)}"
                    )
                    if i == 0:
                        primary_action_to_execute = AnswerQuestion("")

            elif name == "read_email":
                msg_id = args.get("message_id")
                if msg_id and not msg_id.startswith("<") and not msg_id.endswith(">"):
                    msg_id = f"<{msg_id}>"

                if msg_id is None:
                    print(f"Tool call {i} ({name}): message_id is missing.")
                    history_entry_for_this_tool["error"] = "message_id missing"
                    if i == 0:
                        primary_action_to_execute = AnswerQuestion("")
                else:
                    history_entry_for_this_tool["args"] = {"message_id": msg_id}
                    current_tool_env_call = ReadEmail(message_id=msg_id)

            elif name == "answer_question":
                try:
                    parsed = AnswerQuestionArgs(**args)
                    history_entry_for_this_tool["args"] = parsed.model_dump()
                    current_tool_env_call = AnswerQuestion(parsed.answer)
                except Exception as e:
                    print(f"Tool call {i} ({name}): Args parsing error: {e}")
                    history_entry_for_this_tool["error"] = (
                        f"AnswerQuestionArgs parsing error: {str(e)}"
                    )
                    if i == 0:
                        primary_action_to_execute = AnswerQuestion("")

            elif name == "terminate":
                try:
                    parsed = TerminateArgs(**args)
                    history_entry_for_this_tool["args"] = parsed.model_dump()
                    current_tool_env_call = Terminate()
                except Exception as e:
                    print(
                        f"Tool call {i} ({name}): Args parsing error (TerminateArgs): {e}. Proceeding with Terminate()."
                    )
                    history_entry_for_this_tool["args"] = (
                        args  # Log raw args if TerminateArgs parsing fails
                    )
                    history_entry_for_this_tool["error"] = (
                        f"TerminateArgs parsing error: {str(e)}, but Terminate() called"
                    )
                    current_tool_env_call = Terminate()

            else:
                print(f"Tool call {i}: Unknown tool name '{name}'")
                history_entry_for_this_tool["error"] = "Unknown tool name"
                if i == 0:
                    primary_action_to_execute = AnswerQuestion("")

            self.history.append(history_entry_for_this_tool)

            if i == 0 and primary_action_to_execute is None:
                primary_action_to_execute = current_tool_env_call

        if primary_action_to_execute is not None:
            return primary_action_to_execute
        else:
            # Fallback if primary_action_to_execute is still None after the loop
            # (e.g., first tool had an error but didn't set a fallback, or all tools had issues)
            print(
                "Fallback: No valid primary action determined from tool calls after processing all."
            )
            self.history.append(
                {
                    "tool": "no_op",
                    "args": "No valid primary action derived from LLM tools after loop.",
                }
            )
            return AnswerQuestion("")

    def _format_tool_history(self) -> str:
        lines = []
        if not self.tool_history:
            return "No calls yet."
        for h in list(self.tool_history):
            args_str = ""
            action_name = h.get("name", "")
            action_args = h.get("args", "")

            if action_name == "search_emails" and isinstance(action_args, dict):
                args_str = str(action_args.get("keywords", []))
            elif isinstance(action_args, (str, int, float, bool)):
                args_str = str(action_args)
            elif (
                isinstance(action_args, dict) and "keywords" in action_args
            ):  # Cater for SearchEmails direct args
                args_str = str(action_args.get("keywords", []))
            else:
                args_str = str(action_args)  # Fallback

            detail = h.get("result_detail", "")
            lines.append(
                f"{h.get('turn', 0)}. {action_name}({args_str}) → {h.get('result', '')}; {detail}"
            )
        return "\n".join(lines)


# ------------------------ helpers ----------------------------------------- #
async def run_episode(env: EnronEnvironment, agent: ReActEnronAgent) -> bool:
    obs = await env.initialize()
    for _ in range(agent.max_steps):
        call = await agent.act(obs)
        if isinstance(call, AnswerQuestion) and call.action[1]:  # answered
            obs = await env.step(call)
            # Minimal logging for AnswerQuestion, as per user prompt (no extra detail)
            tool_entry_answer = {
                "turn": len(agent.tool_history) + 1,
                "name": call.action[0],
                "args": call.action[1],
                "result": "Question answered",
            }
            agent.tool_history.append(tool_entry_answer)
            break
        obs = await env.step(call)

        tool_entry = {
            "turn": len(agent.tool_history) + 1,
            "name": call.action[0],
            "args": call.action[1],
        }

        if isinstance(call, SearchEmails):
            sr = obs.get("search_results", [])
            tool_entry["result"] = f"{len(sr)} hits"
            result_details_list = []
            if sr:  # If there are search results
                for idx, result_item in enumerate(sr):
                    message_id_val = result_item.get("message_id", "N/A")
                    snippet_from_db = result_item.get(
                        "snippet", ""
                    )  # Raw snippet from the search result

                    # --- Logging for defaults or empty values ---
                    if message_id_val == "N/A":
                        print(
                            f"WARNING: SearchEmails - Result {idx + 1} - Message ID is 'N/A'. Search result item: {result_item}"
                        )
                    if not snippet_from_db:
                        print(
                            f"WARNING: SearchEmails - Result {idx + 1} - Snippet from DB is empty for Message ID '{message_id_val}'. Search result item: {result_item}"
                        )

                    snippet_for_display = snippet_from_db.replace("\n", " ")[:80]

                    if not snippet_for_display.strip() and snippet_from_db.strip():
                        print(
                            f"WARNING: SearchEmails - Result {idx + 1} - Processed snippet for display ('{snippet_for_display}') is effectively empty for Message ID '{message_id_val}', original DB snippet ('{snippet_from_db[:40]}...'). Item: {result_item}"
                        )

                    result_details_list.append(
                        f"  {idx + 1}. {message_id_val} : {snippet_for_display}..."
                    )
                tool_entry["result_detail"] = "\n".join(result_details_list)
            else:
                tool_entry["result_detail"] = "  (No specific details for 0 hits)"
        elif isinstance(call, ReadEmail):
            email_data = obs.get("email")  # This is a dict or None
            email_txt = ""
            if email_data and isinstance(email_data, dict):
                email_txt = email_data.get("body", "")[:120]
            tool_entry["result"] = "email_read"
            tool_entry["result_detail"] = (
                email_txt + "..." if email_txt else "Email not found or empty."
            )
        elif isinstance(call, Terminate):
            tool_entry["result"] = "Session terminated"
            # No result_detail needed for Terminate as per user prompt

        agent.tool_history.append(tool_entry)

        if obs["terminated"]:
            break
    return obs["terminated"] and obs["reward_last"] > 0


# ------------------------ unit-style sanity -------------------------------- #
@pytest.mark.asyncio
async def test_react_agent_enron(tmp_path: Path):
    taskset = await create_enron_taskset()
    inst = taskset.instances[0]  # pick first QA pair
    env = EnronEnvironment(inst)
    llm = LM(model_name="gpt-4.1", formatting_model_name="gpt-4.1", temperature=0.0)
    agent = ReActEnronAgent(llm)
    solved = await run_episode(env, agent)
    # Retrieve and print final total_reward from the engine snapshot
    snapshot = await env.checkpoint()
    print(f"Total Reward: {snapshot.total_reward}")
    print(f"Partial Rewards: {snapshot.partial_rewards}")

    ds = Dataset(
        questions=[TrainingQuestion(id="enron_ep", intent="answer", criteria="correct")],
        reward_signals=[
            RewardSignal(
                question_id="enron_ep",
                run_id=agent.system_instance_id,
                system_instance_id=agent.system_instance_id,
                reward=1 if solved else 0,
                error_message="",
                metadata={"history": list(agent.history)},
            )
        ],
    )
    # upload(ds)  # optional
    assert isinstance(solved, bool)


# ------------------------ quick eval over 10 test instances ---------------- #
async def eval_react_enron(n: int = 2) -> None:
    ts = await create_enron_taskset()
    test_insts = [i for i in ts.instances if i.metadata.split == "test"][:n]

    rows: List[Dict[str, Any]] = []

    async def _run(instance):  # wrapper to build env/agent per instance
        env = EnronEnvironment(instance)
        llm = LM(
            model_name="gpt-4.1-nano",
            formatting_model_name="gpt-4.1-nano",
            temperature=0.0,
        )
        agent = ReActEnronAgent(llm)
        solved = await run_episode(env, agent)
        # Retrieve and print final total_reward for this instance
        snapshot = await env.checkpoint()
        # print(f"  Total Reward: {snapshot.total_reward}")
        # print(f"  Partial Rewards: {snapshot.partial_rewards}")

        agent_answer = "Agent did not attempt to answer."
        # Search agent.tool_history for the last 'answer_question' call
        for tool_call in reversed(agent.tool_history):
            if tool_call.get("name") == ACTION_ANSWER:
                # For AnswerQuestion, 'args' in tool_history holds the answer string
                agent_answer = tool_call.get(
                    "args",
                    "Agent called answer_question, but answer was not logged correctly.",
                )
                break

        gold_answer = instance.intent.gold_state_diff["answer"]
        question = instance.impetus.instructions

        # print(f"\nQuestion: {question}")
        # print(f"  Gold Answer  : {gold_answer}")
        # print(f"  Agent Answer : {agent_answer}")
        # print(f"  Solved       : {solved}")
        # print("-" * 40)
        # collect summary row
        rows.append(
            {
                "gold": gold_answer,
                "agent": agent_answer,
                "score": snapshot.total_reward,
                "partials": snapshot.partial_rewards,
            }
        )
        return solved

    solved_results = await asyncio.gather(*(_run(i) for i in test_insts))
    print(
        f"Overall Solved: {sum(solved_results)}/{len(test_insts)} ({sum(solved_results) / len(test_insts):.0%})"
    )
    # Print summary table
    print("\nSummary Table:")
    print(f"{'Gold Answer':<40} | {'Agent Answer':<40} | {'Score':<5} | Partial Rewards")
    print("-" * 100)
    for r in rows:
        gold = r["gold"][:40]
        agent = r["agent"][:40]
        score = r["score"]
        partials = ",".join(str(x) for x in r["partials"])
        print(f"{gold:<40} | {agent:<40} | {score:<5.1f} | {partials}")


if __name__ == "__main__":
    experiment_params = {"model": "gpt-4.1-mini", "n_questions": 5}
    asyncio.run(eval_react_enron(n=experiment_params["n_questions"]))

    # gpt-4.1 Overall Solved: 6/15 (40%)
    # gpt-4.1-mini Overall Solved: 8/15 (53%)
    # gpt-4.1-nano Overall Solved: 3/15 (20%)
