"""Unit tests for Verilog scoring and rewards."""
import pytest

from synth_ai.environments.examples.verilog.engine import (
    VerilogCompileSuccessComponent,
    VerilogSimulationPassComponent,
    VerilogSubmitSuccessComponent,
    VerilogPublicState,
)


@pytest.mark.asyncio
async def test_compile_success_reward():
    """Test that successful compilation awards 0.1 reward."""
    component = VerilogCompileSuccessComponent()
    state = VerilogPublicState(files={}, build_dir="/tmp", task_completed=False)
    
    # Successful compile (returncode 0)
    action = {"type": "compile", "returncode": 0}
    reward = await component.score(state, action)
    assert reward == 0.1
    
    # Failed compile (returncode != 0)
    action_fail = {"type": "compile", "returncode": 1}
    reward_fail = await component.score(state, action_fail)
    assert reward_fail == 0.0
    
    # Non-compile action
    action_other = {"type": "write_file"}
    reward_other = await component.score(state, action_other)
    assert reward_other == 0.0


@pytest.mark.asyncio
async def test_simulation_pass_reward():
    """Test that passing simulation awards 1.0 reward."""
    component = VerilogSimulationPassComponent()
    state = VerilogPublicState(files={}, build_dir="/tmp", task_completed=False)
    
    # Passing simulation
    action = {"type": "simulate", "passed": True}
    reward = await component.score(state, action)
    assert reward == 1.0
    
    # Failing simulation
    action_fail = {"type": "simulate", "passed": False}
    reward_fail = await component.score(state, action_fail)
    assert reward_fail == 0.0
    
    # Non-simulate action
    action_other = {"type": "compile"}
    reward_other = await component.score(state, action_other)
    assert reward_other == 0.0


@pytest.mark.asyncio
async def test_submit_success_reward():
    """Test that successful submission awards 10.0 reward."""
    component = VerilogSubmitSuccessComponent()
    state = VerilogPublicState(files={}, build_dir="/tmp", task_completed=False)
    
    # Successful submission (tests passed)
    action = {"type": "submit", "passed": True}
    reward = await component.score(state, action)
    assert reward == 10.0
    
    # Failed submission (tests didn't pass)
    action_fail = {"type": "submit", "passed": False}
    reward_fail = await component.score(state, action_fail)
    assert reward_fail == 0.0
    
    # Non-submit action
    action_other = {"type": "compile"}
    reward_other = await component.score(state, action_other)
    assert reward_other == 0.0


@pytest.mark.asyncio
async def test_submit_checks_simulation_output():
    """Test that submit() correctly checks the last simulation output."""
    from synth_ai.environments.examples.verilog.engine import VerilogEngine
    from synth_ai.environments.tasks.core import TaskInstance, Impetus, Intent
    
    # Create a minimal task instance
    task = TaskInstance(
        id="test",
        impetus=Impetus(instructions="Test"),
        intent=Intent(
            rubric={"goal": "test"},
            gold_trajectories=None,
            gold_state_diff={},
            deterministic_eval_functions=[],
        ),
        metadata=None,
        is_reproducible=False,
        initial_engine_snapshot=None,
    )
    task.snapshot_dir = None  # Will be set by engine
    
    engine = VerilogEngine(task)
    
    # Test 1: No simulation run yet
    result = await engine.submit()
    assert result["passed"] is False
    assert "No simulation run yet" in result["detail"]
    
    # Test 2: Simulate with passing output
    engine._last_simulate_output = "Mismatches: 0 in 100 samples\nALL_TESTS_PASSED"
    result_pass = await engine.submit()
    assert result_pass["passed"] is True
    assert "All tests passed" in result_pass["detail"]
    
    # Test 3: Simulate with failing output
    engine._last_simulate_output = "Mismatches: 5 in 100 samples\nErrors detected"
    result_fail = await engine.submit()
    assert result_fail["passed"] is False
    assert "Tests failed" in result_fail["detail"]

