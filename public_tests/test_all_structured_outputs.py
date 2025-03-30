from typing import Any, Dict, Optional

import pytest
from pydantic import BaseModel

from synth_ai.zyk import LM, BaseLMResponse

class StateUpdate(BaseModel):
    """Response model for state updates from LLM"""

    short_term_plan: Optional[str] = None
    objective: Optional[str] = None
    final_results: Optional[Dict[str, Any]] = None

    def model_post_init(self, __context):
        super().model_post_init(__context)
        # Ensure no protected fields are present
        protected_fields = ["message_history", "step_summaries"]
        for field in protected_fields:
            if hasattr(self, field):
                raise ValueError(f"Cannot modify protected field: {field}")


@pytest.fixture(scope="module")
def models():
    """Initialize LMs for different vendors"""
    return {
        "gpt-4o-mini": LM(
            model_name="gpt-4o-mini",
            formatting_model_name="gpt-4o-mini",
            temperature=0.1,
            structured_output_mode="forced_json",
        ),
        "o3-mini": LM(
            model_name="o3-mini",
            formatting_model_name="gpt-4o-mini",
            temperature=0.1,
            structured_output_mode="forced_json",
        ),
        "gemini-1.5-flash": LM(
            model_name="gemini-1.5-flash",
            formatting_model_name="gpt-4o-mini",
            temperature=0.1,
            structured_output_mode="stringified_json",
        ),
        "claude-3-haiku-20240307": LM(
            model_name="claude-3-haiku-20240307",
            formatting_model_name="gpt-4o-mini",
            temperature=0.1,
            structured_output_mode="stringified_json",
        ),
        "deepseek-chat": LM(
            model_name="deepseek-chat",
            formatting_model_name="gpt-4o-mini",
            temperature=0.1,
            structured_output_mode="stringified_json",
        ),
        "deepseek-reasoner": LM(
            model_name="deepseek-reasoner",
            formatting_model_name="gpt-4o-mini",
            temperature=1,
            structured_output_mode="stringified_json",
        ),
        "llama-3.1-8b-instant": LM(
            model_name="llama-3.1-8b-instant",
            formatting_model_name="gpt-4o-mini",
            temperature=0.1,
            structured_output_mode="stringified_json",
        ),
        "mistral-small-latest": LM(
            model_name="mistral-small-latest",
            formatting_model_name="gpt-4o-mini",
            temperature=0.1,
            structured_output_mode="stringified_json",
        ),
    }


@pytest.fixture
def system_message():
    """System message for state updates"""
    return """You are helping update the agent's state. Look at the current state and state_delta_instructions and update the state.

Available fields you can modify:
{
    "short_term_plan": "str",
    "objective": "str",
    "final_results": "Dict[str, Any]"
}

Protected fields (do not modify):
{
    "message_history": "Cannot directly edit message history - it is managed internally",
    "step_summaries": "Cannot directly edit step summaries - they are generated automatically"
}

Please be brief, the state ought not be too long."""


@pytest.fixture
def current_state():
    """Initial state for testing"""
    return {
        "short_term_plan": "Current plan: Review code changes",
        "objective": "Review pull request",
        "final_results": {
            "findings": [],
            "recommendations": [],
            "analysis": {},
            "status": "IN_PROGRESS",
        },
    }


@pytest.mark.timeout(15)
@pytest.mark.parametrize(
    "model_name",
    [
       "gpt-4o-mini",
       "gemini-1.5-flash",
        "claude-3-haiku-20240307",
        "deepseek-chat",
        "llama-3.1-8b-instant",
    ],
)
def test_state_delta_handling(
    model_name: str, models: Dict[str, LM], system_message: str, current_state: Dict
):
    """Test that each model correctly handles state updates"""

    state_delta_instructions = """Update the final_results to include findings about code quality issues. Add a recommendation to improve error handling."""
    user_message = f"Current state: {current_state}\nState delta instructions: {state_delta_instructions}\n\nHow should the state be updated?"

    #try:
    result: BaseLMResponse = models[model_name].respond_sync(
        system_message=system_message,
        user_message=user_message,
        response_model=StateUpdate,
    )
    print("Result", result)
    # Verify response structure
    assert isinstance(result, BaseLMResponse)
    assert isinstance(result.structured_output, StateUpdate)

    # Verify only allowed fields are present and have correct types
    if result.structured_output.short_term_plan is not None:
        assert isinstance(result.structured_output.short_term_plan, str)
    if result.structured_output.objective is not None:
        assert isinstance(result.structured_output.objective, str)
    if result.structured_output.final_results is not None:
        assert isinstance(result.structured_output.final_results, dict)

    # except Exception as e:
    #     pytest.fail(f"Model {model_name} failed: {str(e)}")


@pytest.mark.timeout(15)
@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-4o-mini",
        "gemini-1.5-flash",
        "claude-3-haiku-20240307",
        "deepseek-chat",
        "llama-3.1-8b-instant",
    ],
)
def test_state_delta_protected_fields(
    model_name: str, models: Dict[str, LM], system_message: str
):
    """Test that models respect protected fields"""

    current_state = {
        "short_term_plan": "Current plan: Review code changes",
        "objective": "Review pull request",
        "message_history": ["Previous message 1", "Previous message 2"],
        "step_summaries": ["Step 1 summary", "Step 2 summary"],
        "final_results": {
            "findings": [],
            "recommendations": [],
            "analysis": {},
            "status": "IN_PROGRESS",
        },
    }

    state_delta_instructions = """Update the message history to include new findings and update step summaries with recent progress."""
    user_message = f"Current state: {current_state}\nState delta instructions: {state_delta_instructions}\n\nHow should the state be updated?"

    #try:
    result = models[model_name].respond_sync(
        system_message=system_message,
        user_message=user_message,
        response_model=StateUpdate,
    )
    # except Exception as e:
    #     pytest.fail(f"Model {model_name} failed: {str(e)}")
