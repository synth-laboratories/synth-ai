import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add timeout to all async tests
pytestmark = pytest.mark.timeout(15)

from synth_ai.environments.examples.verilog.engine import (
    VerilogEngine,
    VerilogPublicState,
    VerilogPrivateState,
    VerilogCompileSuccessComponent,
    VerilogSimulationPassComponent,
    VerilogStepPenaltyComponent,
)
from synth_ai.environments.examples.verilog.taskset import (
    VerilogTaskInstance,
    VerilogTaskInstanceMetadata,
)
from synth_ai.environments.tasks.core import Impetus, Intent
from uuid import uuid4


@pytest.fixture
def mock_task_instance():
    """Create a mock task instance for testing."""
    temp_dir = tempfile.mkdtemp(prefix="test_verilog_")
    pristine_dir = Path(temp_dir) / "pristine"
    snapshot_dir = Path(temp_dir) / "snapshot"

    pristine_dir.mkdir(parents=True)
    snapshot_dir.mkdir(parents=True)

    # Create test files
    (pristine_dir / "TopModule.v").write_text("""module TopModule(
    output zero
);
    assign zero = 1'b0;
endmodule""")

    (pristine_dir / "test_tb.v").write_text("""`timescale 1ns/1ps
module test_tb;
    wire zero;
    TopModule dut(.zero(zero));
    
    initial begin
        #10;
        if (zero !== 1'b0) $fatal(1, "Test failed");
        $display("ALL_TESTS_PASSED");
        $finish;
    end
endmodule""")

    metadata = VerilogTaskInstanceMetadata(
        problem_name="test_problem",
        difficulty="easy",
        description="Test problem",
        files_provided=["TopModule.v", "test_tb.v"],
    )

    task = VerilogTaskInstance(
        id=uuid4(),
        impetus=Impetus(instructions="Test task"),
        intent=Intent(rubric="Test goal", gold_trajectories=None, gold_state_diff={}),
        metadata=metadata,
        is_reproducible=True,
        initial_engine_snapshot=None,
        pristine_dir=str(pristine_dir),
        snapshot_dir=str(snapshot_dir),
    )

    yield task

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def engine(mock_task_instance):
    """Create a VerilogEngine instance for testing."""
    return VerilogEngine(mock_task_instance)


class TestVerilogEngine:
    """Test suite for VerilogEngine class."""

    @pytest.mark.asyncio
    async def test_engine_initialization(self, engine):
        """Test engine initialization."""
        assert engine.task_instance is not None
        assert engine._total_reward == 0.0
        assert engine.reward_stack is not None
        assert len(engine.reward_stack.components) == 3

    @pytest.mark.asyncio
    async def test_reset_engine(self, engine):
        """Test engine reset functionality."""
        priv, pub = await engine._reset_engine()

        assert isinstance(priv, VerilogPrivateState)
        assert isinstance(pub, VerilogPublicState)
        assert priv.reward_last == 0.0
        assert priv.total_reward == 0.0
        assert not priv.terminated
        assert not priv.truncated
        assert len(pub.files) >= 1
        assert engine.snapshot_dir.exists()
        assert engine.build_dir.exists()

    @pytest.mark.asyncio
    async def test_write_file(self, engine):
        """Test file writing functionality."""
        await engine._reset_engine()

        result = await engine.write_file("test.v", "module test(); endmodule")

        assert result["ok"] is True
        assert result["type"] == "write_file"
        assert (engine.snapshot_dir / "test.v").exists()
        assert (engine.snapshot_dir / "test.v").read_text() == "module test(); endmodule"

    @pytest.mark.asyncio
    async def test_write_file_nested_path(self, engine):
        """Test writing file with nested directory structure."""
        await engine._reset_engine()

        result = await engine.write_file("subdir/nested.v", "module nested(); endmodule")

        assert result["ok"] is True
        nested_file = engine.snapshot_dir / "subdir" / "nested.v"
        assert nested_file.exists()
        assert nested_file.read_text() == "module nested(); endmodule"

    @pytest.mark.asyncio
    async def test_get_file_contents(self, engine):
        """Test file content retrieval."""
        await engine._reset_engine()

        # Write test file
        await engine.write_file("new_test.v", "module new_test(); endmodule")

        files = engine._get_file_contents()
        assert "new_test.v" in files
        assert "module new_test();" in files["new_test.v"]

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_compile_success(self, mock_run, engine):
        """Test successful compilation."""
        await engine._reset_engine()

        # Mock successful compilation
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        result = await engine.compile(sources=["TopModule.v"])

        assert result["ok"] is True
        assert result["type"] == "compile"
        assert result["returncode"] == 0
        assert "binary" in result

        # Verify iverilog was called with correct flags
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "iverilog" in args
        assert "-g2012" in args
        assert "-o" in args

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_compile_failure(self, mock_run, engine):
        """Test compilation failure."""
        await engine._reset_engine()

        # Mock failed compilation
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = ""
        mock_proc.stderr = "Error: syntax error"
        mock_run.return_value = mock_proc

        result = await engine.compile(sources=["invalid.v"])

        assert result["ok"] is False
        assert result["type"] == "compile"
        assert result["returncode"] == 1
        assert "syntax error" in result["stderr"]
        assert result["binary"] is None

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_simulate_success(self, mock_run, engine):
        """Test successful simulation."""
        await engine._reset_engine()

        # Mock successful simulation
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "Simulation output\nMismatches: 0 in 10 samples\n"
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        result = await engine.simulate()

        assert result["ok"] is True
        assert result["type"] == "simulate"
        assert result["returncode"] == 0
        assert result["passed"] is True
        assert "Mismatches: 0" in result["stdout"]

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_simulate_all_tests_passed(self, mock_run, engine):
        """Test simulation with ALL_TESTS_PASSED indicator."""
        await engine._reset_engine()

        # Mock simulation with ALL_TESTS_PASSED
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "Simulation running\nALL_TESTS_PASSED\n"
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        result = await engine.simulate()

        assert result["ok"] is True
        assert result["passed"] is True

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_simulate_failure(self, mock_run, engine):
        """Test simulation failure."""
        await engine._reset_engine()

        # Mock failed simulation
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "Simulation output\nMismatches: 5 in 10 samples\n"
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        result = await engine.simulate()

        assert result["ok"] is True
        assert result["passed"] is False
        assert "Mismatches: 5" in result["stdout"]

    @pytest.mark.asyncio
    async def test_submit(self, engine):
        """Test submission functionality."""
        await engine._reset_engine()

        result = await engine.submit()

        assert result["ok"] is True
        assert result["type"] == "submit"
        assert result["submitted"] is True

    @pytest.mark.asyncio
    async def test_step_engine_compile_success(self, engine):
        """Test engine stepping with successful compilation."""
        await engine._reset_engine()

        action_result = {
            "ok": True,
            "type": "compile",
            "returncode": 0,
            "stdout": "Compilation successful",
        }

        priv, pub = await engine._step_engine(action_result)

        assert priv.reward_last > 0  # Should get compile success reward
        assert pub.last_compile_output == "Compilation successful"
        assert not pub.task_completed

    @pytest.mark.asyncio
    async def test_step_engine_simulate_success(self, engine):
        """Test engine stepping with successful simulation."""
        await engine._reset_engine()

        action_result = {
            "ok": True,
            "type": "simulate",
            "returncode": 0,
            "stdout": "ALL_TESTS_PASSED",
            "passed": True,
        }

        priv, pub = await engine._step_engine(action_result)

        assert priv.reward_last > 0.5  # Should get large simulation success reward
        assert pub.last_simulate_output == "ALL_TESTS_PASSED"
        assert pub.task_completed is True
        assert priv.terminated is True

    @pytest.mark.asyncio
    async def test_step_penalty(self, engine):
        """Test that each step incurs a small penalty."""
        await engine._reset_engine()

        action_result = {"ok": True, "type": "write_file"}

        priv, pub = await engine._step_engine(action_result)

        assert priv.reward_last < 0  # Should be negative due to step penalty
        assert priv.total_reward < 0


class TestVerilogRewardComponents:
    """Test suite for Verilog reward components."""

    @pytest.mark.asyncio
    async def test_compile_success_component(self):
        """Test compile success reward component."""
        component = VerilogCompileSuccessComponent()
        state = VerilogPublicState(files={}, build_dir="", task_completed=False)

        # Test successful compilation
        action = {"type": "compile", "returncode": 0}
        reward = await component.score(state, action)
        assert reward == 0.1

        # Test failed compilation
        action = {"type": "compile", "returncode": 1}
        reward = await component.score(state, action)
        assert reward == 0.0

        # Test non-compile action
        action = {"type": "write_file"}
        reward = await component.score(state, action)
        assert reward == 0.0

    @pytest.mark.asyncio
    async def test_simulation_pass_component(self):
        """Test simulation pass reward component."""
        component = VerilogSimulationPassComponent()
        state = VerilogPublicState(files={}, build_dir="", task_completed=False)

        # Test successful simulation
        action = {"type": "simulate", "passed": True}
        reward = await component.score(state, action)
        assert reward == 1.0

        # Test failed simulation
        action = {"type": "simulate", "passed": False}
        reward = await component.score(state, action)
        assert reward == 0.0

        # Test non-simulate action
        action = {"type": "write_file"}
        reward = await component.score(state, action)
        assert reward == 0.0

    @pytest.mark.asyncio
    async def test_step_penalty_component(self):
        """Test step penalty reward component."""
        penalty = -0.05
        component = VerilogStepPenaltyComponent(penalty=penalty)
        state = VerilogPublicState(files={}, build_dir="", task_completed=False)

        # Any action should incur penalty
        action = {"type": "write_file"}
        reward = await component.score(state, action)
        assert reward == penalty

        action = {"type": "compile"}
        reward = await component.score(state, action)
        assert reward == penalty


class TestEngineIntegration:
    """Integration tests for the full engine workflow."""

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_full_workflow_success(self, mock_run, engine):
        """Test complete workflow from reset to successful completion."""

        # Setup mock subprocess calls
        def mock_subprocess(cmd, **kwargs):
            mock_proc = MagicMock()
            if "iverilog" in cmd:
                # Mock successful compilation
                mock_proc.returncode = 0
                mock_proc.stdout = ""
                mock_proc.stderr = ""
            elif "vvp" in cmd:
                # Mock successful simulation
                mock_proc.returncode = 0
                mock_proc.stdout = "ALL_TESTS_PASSED\n"
                mock_proc.stderr = ""
            return mock_proc

        mock_run.side_effect = mock_subprocess

        # Initialize engine
        priv, pub = await engine._reset_engine()
        assert priv.total_reward == 0.0

        # Write file
        write_result = await engine.write_file(
            "TopModule.v",
            """module TopModule(
    output zero
);
    assign zero = 1'b0;
endmodule""",
        )
        assert write_result["ok"] is True

        # Compile
        compile_result = await engine.compile()
        assert compile_result["ok"] is True

        priv, pub = await engine._step_engine(compile_result)
        compile_reward = priv.reward_last
        assert compile_reward > 0  # Should get compile success reward

        # Simulate
        simulate_result = await engine.simulate()
        assert simulate_result["ok"] is True
        assert simulate_result["passed"] is True

        priv, pub = await engine._step_engine(simulate_result)
        simulate_reward = priv.reward_last
        assert simulate_reward > 0.5  # Should get large simulation reward
        assert pub.task_completed is True
        assert priv.terminated is True

        # Total reward should be positive (compile + simulate - step penalties)
        assert priv.total_reward > 0

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_compilation_failure_workflow(self, mock_run, engine):
        """Test workflow with compilation failure."""
        # Mock failed compilation
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = ""
        mock_proc.stderr = "Error: syntax error"
        mock_run.return_value = mock_proc

        # Initialize engine
        await engine._reset_engine()

        # Write invalid file
        await engine.write_file("invalid.v", "invalid verilog code")

        # Attempt compilation
        compile_result = await engine.compile()
        assert compile_result["ok"] is False

        priv, pub = await engine._step_engine(compile_result)

        # Should only get step penalty, no compile success reward
        assert priv.reward_last < 0
        assert not pub.task_completed
        assert not priv.terminated
