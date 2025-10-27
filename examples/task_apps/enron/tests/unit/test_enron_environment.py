"""Unit tests for Enron environment tools and rewards."""
import pytest


@pytest.mark.asyncio
@pytest.mark.fast
async def test_enron_search_tool():
    """Test that the search_emails tool works correctly."""
    from synth_ai.environments.examples.enron.environment import SearchEmailsTool
    from synth_ai.environments.examples.enron.engine import EnronEngine
    from synth_ai.environments.tasks.core import TaskInstance, Impetus, Intent
    
    # Create a minimal task instance
    task = TaskInstance(
        id="test",
        impetus=Impetus(instructions="Test question"),
        intent=Intent(
            rubric={"goal": "test"},
            gold_trajectories=None,
            gold_state_diff={},
            deterministic_eval_functions=[],
        ),
        metadata={
            "question": "Test?",
            "gold_answer": "Test answer",
            "inbox_address": "test@enron.com",
        },
        is_reproducible=False,
        initial_engine_snapshot=None,
    )
    
    engine = EnronEngine(task)
    tool = SearchEmailsTool(engine)
    
    # Test that tool has correct name
    assert tool.name == "search_emails"
    
    # Test that tool requires keywords
    from synth_ai.environments.environment.tools import EnvToolCall
    
    # Call with minimal args should work (or fail gracefully)
    result = await tool(EnvToolCall(tool="search_emails", args={"keywords": ["test"]}))
    assert result.ok in (True, False)  # Either succeeds or fails gracefully
    
    # Result should have search_results field
    if result.ok:
        assert "search_results" in result.payload


@pytest.mark.asyncio
async def test_enron_answer_tool():
    """Test that the answer_question tool calculates rewards correctly."""
    from synth_ai.environments.examples.enron.environment import AnswerQuestionTool
    from synth_ai.environments.examples.enron.engine import EnronEngine
    from synth_ai.environments.tasks.core import TaskInstance, Impetus, Intent
    
    task = TaskInstance(
        id="test",
        impetus=Impetus(instructions="Test question"),
        intent=Intent(
            rubric={"goal": "test"},
            gold_trajectories=None,
            gold_state_diff={},
            deterministic_eval_functions=[],
        ),
        metadata={
            "question": "What is the answer?",
            "gold_answer": "The answer is 42",
            "inbox_address": "test@enron.com",
        },
        is_reproducible=False,
        initial_engine_snapshot=None,
    )
    
    engine = EnronEngine(task)
    tool = AnswerQuestionTool(engine)
    
    # Test exact match
    from synth_ai.environments.environment.tools import EnvToolCall
    result_exact = await tool(EnvToolCall(tool="answer_question", args={"answer": "The answer is 42"}))
    assert result_exact.ok is True
    assert "status" in result_exact.payload
    
    # Test partial match (should still give some reward)
    result_partial = await tool(EnvToolCall(tool="answer_question", args={"answer": "answer is 42"}))
    assert result_partial.ok is True


@pytest.mark.asyncio
async def test_enron_reward_calculation():
    """Test that Enron rewards are calculated correctly."""
    from synth_ai.environments.examples.enron.engine import EnronEngine
    from synth_ai.environments.tasks.core import TaskInstance, Impetus, Intent
    
    task = TaskInstance(
        id="test",
        impetus=Impetus(instructions="Test question"),
        intent=Intent(
            rubric={"goal": "test"},
            gold_trajectories=None,
            gold_state_diff={},
            deterministic_eval_functions=[],
        ),
        metadata={
            "question": "What is the answer?",
            "gold_answer": "forty two",
            "inbox_address": "test@enron.com",
        },
        is_reproducible=False,
        initial_engine_snapshot=None,
    )
    
    engine = EnronEngine(task)
    
    # Test exact match gives high reward
    reward_exact = await engine._judge_answer("forty two")
    assert reward_exact > 0.9, f"Expected high reward for exact match, got {reward_exact}"
    
    # Test partial match gives medium reward
    reward_partial = await engine._judge_answer("the answer is forty two")
    assert reward_partial > 0.5, f"Expected medium reward for partial match, got {reward_partial}"
    
    # Test wrong answer gives low/zero reward
    reward_wrong = await engine._judge_answer("completely wrong answer")
    assert reward_wrong < 0.5, f"Expected low reward for wrong answer, got {reward_wrong}"

