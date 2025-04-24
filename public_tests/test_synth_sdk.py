# # from dev.testing.hendryks import HendryksMathBenchmark, TrivialHendryksMathAgent
# import asyncio
# import json
# import logging
# import os
# import re
# import sys
# import time
# import uuid
# from typing import List, Dict
# from pydantic import BaseModel
# import pytest
# from dotenv import load_dotenv
# from synth_sdk.tracing.abstractions import (
#     Dataset,
#     RewardSignal,
#     TrainingQuestion,
# )
# from synth_sdk.tracing.client_manager import ClientManager
# from synth_sdk.tracing.decorators import get_tracing_config, trace_system_async
# from synth_sdk.tracing.upload import upload
# from synth_sdk.tracing.utils import get_system_id

# from datasets import load_dataset
# from zyk import LM


# class HendryksMathBenchmark:
#     def __init__(self):
#         self.name = "hendryks_math"
#         self.temp_dir = "temp"
#         os.makedirs(self.temp_dir, exist_ok=True)
#         os.makedirs("datasets/competition_math", exist_ok=True)

#     def load_data(self):
#         cache_path = "datasets/competition_math/dataset.json"

#         # Try to load from cache first
#         if os.path.exists(cache_path):
#             with open(cache_path, "r") as f:
#                 dataset = json.load(f)
#                 problems = []
#                 for item in dataset["train"]:  # Using train split for consistency
#                     problem = {
#                         "question": item["problem"],
#                         "answer": item["solution"],
#                         "subject": item.get("type", "unknown"),
#                         "level": "competition",  # All problems are competition level
#                     }
#                     problems.append(problem)
#                 return problems

#         # If not cached, load from HF and cache
#         dataset = load_dataset("competition_math", "main")
#         with open(cache_path, "w") as f:
#             json.dump(
#                 {"train": list(dataset["train"]), "test": list(dataset["test"])}, f
#             )

#         # Convert to our format
#         problems = []
#         for item in dataset["train"]:
#             problem = {
#                 "question": item["problem"],
#                 "answer": item["solution"],
#                 "subject": item.get("type", "unknown"),
#                 "level": "competition",
#             }
#             problems.append(problem)

#         return problems

#     def get_problems(self):
#         temp_path = os.path.join(self.temp_dir, "hendryks_math.json")

#         # Load from temp file if it exists
#         if os.path.exists(temp_path):
#             with open(temp_path, "r") as f:
#                 return json.load(f)

#         # Otherwise load from dataset and save
#         problems = self.load_data()
#         with open(temp_path, "w") as f:
#             json.dump(problems, f)
#         return problems

#     def score_answer(self, question: str, proposed_answer: str) -> bool:
#         """Score a proposed answer against the correct answer for a given question."""
#         # Find the problem that matches the question
#         problems = self.get_problems()
#         matching_problem = next(
#             (p for p in problems if p["question"] == question), None
#         )

#         if not matching_problem:
#             raise ValueError("Question not found in benchmark")

#         # Extract answer from proposed solution's \boxed{} format
#         proposed_match = re.search(r"\\boxed{((?:[^{}]|{[^{}]*})*)}", proposed_answer)
#         if not proposed_match:
#             return False

#         # Extract answer from correct solution's \boxed{} format
#         correct_match = re.search(
#             r"\\boxed{((?:[^{}]|{[^{}]*})*)}", matching_problem["answer"]
#         )
#         if not correct_match:
#             return False

#         extracted_proposed = proposed_match.group(1).strip()
#         extracted_correct = correct_match.group(1).strip()

#         # print(f"Proposed answer: {extracted_proposed}")
#         # print(f"Correct answer: {extracted_correct}")

#         return extracted_proposed == extracted_correct


# class TrivialHendryksMathAgent:
#     def __init__(self):
#         self.lm = LM(  # gemini-1.5-flash
#             model_name="gpt-4o-mini",
#             formatting_model_name="gpt-4o-mini",
#             temperature=0.1,
#             synth_logging=True,
#         )
#         self.system_name = "HendryksMathAgent"
#         self.system_id = get_system_id(self.system_name)
#         self.system_instance_id = str(uuid.uuid4())

#     @trace_system_async(
#         origin="agent",
#         event_type="plan",
#         manage_event="create_and_end",
#         increment_partition=True,
#         verbose=True,
#     )
#     async def plan(self, math_question: str) -> str:
#         logger.debug("Starting plan method with trace decorator")
#         try:
#             class Plan(BaseModel):
#                 content: str
#             response = await self.lm.respond_async(
#                 system_message="""You are an AI assisting a colleague in completing a mathematics problem.
# You will be given a mathematics problem statement. Your task is to create a detailed plan to solve the problem, 
# breaking it down into clear, logical steps.""",
#                 user_message=f"""Please provide a detailed, step-by-step plan to solve this math problem:
# {math_question}

# Your plan should include:
# 1. A clear statement of the given information and problem to be solved
# 2. Identification of relevant mathematical concepts and techniques
# 3. Definition of variables and known relationships
# 4. A step-by-step approach to solving the problem
# 5. Explanation of the reasoning behind each step""",
#                 response_model=Plan
#             )
#             logger.debug("Successfully got response from LM in plan method")
#             return response.content
#         except Exception as e:
#             logger.error(f"Error in plan method: {str(e)}", exc_info=True)
#             raise

#     @trace_system_async(
#         origin="agent",
#         event_type="execute",
#         manage_event="create_and_end",
#         increment_partition=True,
#         verbose=True,
#     )
#     async def execute(self, plan: str) -> str:
#         logger.debug("Starting execute method with trace decorator")
#         try:
#             class Solution(BaseModel):
#                 content: str
#             response = await self.lm.respond_async(
#                 system_message="""You are an AI mathematical problem-solving assistant.
# You will be given a solution plan. Your task is to implement this plan,
# showing all work and verifying correctness at each step.""",
#                 user_message=f"""
# Plan:
# {plan}

# Please solve this problem by carefully following the provided plan. Show all your work and calculations.
# Leave your final answer at the very end in the format \\boxed{{answer}}.""",
#                 response_model=Solution,
#             )
#             logger.debug("Successfully got response from LM in execute method")
#             return response.content
#         except Exception as e:
#             logger.error(f"Error in execute method: {str(e)}", exc_info=True)
#             raise

#     async def run(self, math_question: str) -> str:
#         logger.debug("Starting run method")
#         plan = await self.plan(math_question)
#         logger.debug("Completed plan method")
#         solution = await self.execute(plan)
#         logger.debug("Completed execute method")
#         return solution


# # Configure logging
# logging.basicConfig(
#     level=logging.ERROR, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger(__name__)

# # Add logging for trace decorator
# trace_logger = logging.getLogger("synth_sdk.tracing.decorators")
# trace_logger.setLevel(logging.ERROR)

# # Add logging for client manager
# client_logger = logging.getLogger("synth_sdk.tracing.client_manager")
# client_logger.setLevel(logging.ERROR)

# load_dotenv()


# async def setup_synth_config():
#     """Setup synth configuration for deferred logging."""
#     logger.info("Setting up synth configuration for deferred logging")
#     os.environ["SYNTH_LOGGING_MODE"] = "deferred"
#     os.environ["SYNTH_ENDPOINT_OVERRIDE"] = "https://agent-learning.onrender.com"
#     config = get_tracing_config()
#     ClientManager.initialize(config)
#     logger.info("Synth config:")
#     logger.info(f"  Mode: {config.mode}")
#     logger.info(f"  API Key present: {bool(config.api_key)}")
#     logger.info(f"  Base URL: {config.base_url}")


# @pytest.mark.asyncio
# async def test_deferred_logging():
#     """Test deferred logging with both pytest and regular assertions."""
#     logger.info("=== STARTING DEFERRED LOGGING TEST ===")
#     start_time = time.time()
#     logger.info(f"Test start time: {start_time}")

#     # Determine if running under pytest
#     is_pytest = "pytest" in sys.modules

#     try:
#         await setup_synth_config()

#         # Initialize and run agent
#         benchmark = HendryksMathBenchmark()
#         agent = TrivialHendryksMathAgent()
#         logger.info(f"Agent system ID: {agent.system_id}")
#         logger.info(f"Agent system instance ID: {agent.system_instance_id}")

#         problems = benchmark.get_problems()
#         test_problem = problems[0]["question"]
#         logger.info(f"Using test problem: {test_problem}")

#         # Run the agent
#         logger.info("Running agent...")
#         solution = await agent.run(test_problem)
#         logger.info(f"Agent solution: {solution}")

#         # Create dataset and upload results
#         logger.info("Creating dataset and uploading results...")
#         dataset = Dataset(
#             questions=[
#                 TrainingQuestion(
#                     id="q0",
#                     intent="Test math problem",
#                     criteria="Testing deferred tracing and upload functionality",
#                 )
#             ],
#             reward_signals=[
#                 RewardSignal(
#                     question_id="q0",
#                     system_instance_id=agent.system_instance_id,
#                     reward=1.0,
#                     annotation="Test reward",
#                 )
#             ],
#         )

#         # Upload the dataset and traces
#         logger.info("Starting upload process...")
#         upload_id, questions_json, reward_signals_json, traces_json = upload(
#             dataset=dataset
#         )

#         logger.info(f"Upload completed with ID: {upload_id}")
#         logger.debug(f"Number of traces: {len(traces_json)}")
#         print(traces_json)

#         # Verify upload results
#         if is_pytest:
#             assert upload_id
#             assert questions_json
#             assert reward_signals_json
#             assert traces_json
#         else:
#             assert upload_id, "Upload ID should not be empty"
#             assert questions_json, "Questions JSON should not be empty"
#             assert reward_signals_json, "Reward signals JSON should not be empty"
#             assert traces_json, "Traces JSON should not be empty"

#         # Verify trace content
#         for i, trace in enumerate(traces_json):
#             logger.debug(f"Verifying trace {i}:")
#             verify_trace_content(trace, is_pytest)

#         logger.info("All traces verified successfully!")
#         return True

#     except AssertionError as e:
#         logger.error(f"Test failed: {str(e)}")
#         if is_pytest:
#             raise
#         return False
#     except Exception as e:
#         logger.error(f"Unexpected error: {str(e)}", exc_info=True)
#         if is_pytest:
#             raise
#         return False


# def verify_trace_content(trace: dict, is_pytest: bool = False) -> None:
#     """Verify the content of a trace."""
#     if is_pytest:
#         assert trace["system_instance_id"]
#     else:
#         assert trace["system_instance_id"], "Trace missing system_instance_id"

#     # Verify events were captured
#     has_events = False
#     for partition in trace["partition"]:
#         if len(partition["events"]) > 0:
#             has_events = True
#             for event in partition["events"]:
#                 logger.debug(f"Checking event: {json.dumps(event, indent=2)}")
#                 if "agent_compute_step" in event:
#                     step = event["agent_compute_step"]
#                     logger.debug(f"Checking compute step: {json.dumps(step, indent=2)}")
#                     if is_pytest:
#                         assert step.get("model_name") is not None
#                         assert step.get("model_name") != ""
#                     else:
#                         assert (
#                             step.get("model_name") is not None
#                         ), "Model name is missing"
#                         assert step.get("model_name") != "", "Model name is empty"

#                     if step.get("compute_input"):
#                         for input_item in step["compute_input"]:
#                             if is_pytest:
#                                 assert "messages" in input_item, input_item.keys()
#                             else:
#                                 assert "messages" in input_item, (
#                                     f"Input must have 'messages' key, but found keys: {list(input_item.keys())}"
#                                     f"\nFull input: {json.dumps(input_item, indent=2)}"
#                                 )
#                             messages = input_item["messages"]
#                             if is_pytest:
#                                 assert isinstance(messages, list)
#                                 assert len(messages) == 2
#                             else:
#                                 assert isinstance(
#                                     messages, list
#                                 ), "Messages must be a list"
#                                 assert len(messages) == 2, (
#                                     f"Expected exactly 2 messages (system and user), but found {len(messages)}"
#                                     f"\nMessages: {json.dumps(messages, indent=2)}"
#                                 )
#             break

#     if is_pytest:
#         assert has_events
#     else:
#         assert (
#             has_events
#         ), f"At least one partition should contain events - {trace['partition']}"


# if __name__ == "__main__":
#     # Remove the pytest check so the test always runs
#     success = asyncio.run(test_deferred_logging())
#     print("✅ All tests passed!" if success else "❌ Tests failed!")
#     exit(0 if success else 1)
