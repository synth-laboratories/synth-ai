import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from uuid import uuid4

# Add timeout to all async tests
pytestmark = pytest.mark.timeout(15)

from synth_ai.environments.examples.verilog.environment import (
    VerilogEnvironment,
    VerilogWriteFileTool,
    VerilogCompileTool,
    VerilogSimulateTool,
    VerilogSubmitTool,
    VerilogObservationCallable,
    WriteFileInput,
    CompileInput,
    SimulateInput,
    SubmitInput,
)
from synth_ai.environments.examples.verilog.engine import (
    VerilogEngine,
    VerilogPublicState,
    VerilogPrivateState,
)
from synth_ai.environments.examples.verilog.taskset import (
    VerilogTaskInstance,
    VerilogTaskInstanceMetadata,
)
from synth_ai.environments.environment.tools import EnvToolCall, ToolResult
from synth_ai.environments.tasks.core import Impetus, Intent


@pytest.fixture
def mock_task_instance():
    """Create a mock task instance for testing."""
    temp_dir = tempfile.mkdtemp(prefix="test_verilog_env_")
    pristine_dir = Path(temp_dir) / "pristine"
    snapshot_dir = Path(temp_dir) / "snapshot"

    pristine_dir.mkdir(parents=True)
    snapshot_dir.mkdir(parents=True)

    # Create test files
    (pristine_dir / "TopModule.v").write_text("""module TopModule();
    // TODO: Implement module
endmodule""")

    (pristine_dir / "RefModule.v").write_text("""module RefModule(
    output zero
);
    assign zero = 1'b0;
endmodule""")

    (pristine_dir / "test_tb.v").write_text("""`timescale 1ns/1ps
module test_tb;
    wire zero;
    TopModule dut(.zero(zero));
    RefModule ref(.zero(zero_ref));
    
    initial begin
        #10;
        if (zero !== zero_ref) $fatal(1, "Test failed");
        $display("Mismatches: 0 in 10 samples");
        $finish;
    end
endmodule""")

    metadata = VerilogTaskInstanceMetadata(
        problem_name="test_problem",
        difficulty="easy",
        description="Test problem",
        files_provided=["TopModule.v", "RefModule.v", "test_tb.v"],
    )

    task = VerilogTaskInstance(
        id=uuid4(),
        impetus=Impetus(instructions="Test task instructions"),
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
def verilog_env(mock_task_instance):
    """Create a VerilogEnvironment instance for testing."""
    return VerilogEnvironment(mock_task_instance)


class TestVerilogEnvironment:
    """Test suite for VerilogEnvironment class."""

    @pytest.mark.asyncio
    async def test_environment_initialization(self, verilog_env):
        """Test environment initialization."""
        assert verilog_env.name == "VerilogEval"
        assert verilog_env.task_instance is not None
        assert isinstance(verilog_env.engine, VerilogEngine)
        assert len(verilog_env._tools_instances) == 4
        assert "write_file" in verilog_env._tools_instances
        assert "compile" in verilog_env._tools_instances
        assert "simulate" in verilog_env._tools_instances
        assert "submit" in verilog_env._tools_instances

    @pytest.mark.asyncio
    async def test_environment_initialize(self, verilog_env):
        """Test environment initialization method."""
        obs = await verilog_env.initialize()

        assert isinstance(obs, dict)
        assert "files" in obs
        assert "build_dir" in obs
        assert "files_summary" in obs
        assert "task_completed" in obs
        assert "reward_last" in obs
        assert "total_reward" in obs
        assert "terminated" in obs
        assert "compile_status" in obs
        assert "simulate_status" in obs

        assert len(obs["files"]) >= 3  # TopModule.v, RefModule.v, test_tb.v
        assert obs["task_completed"] is False
        assert obs["terminated"] is False
        assert obs["reward_last"] == 0.0
        assert obs["total_reward"] == 0.0

    @pytest.mark.asyncio
    async def test_environment_terminate(self, verilog_env):
        """Test environment termination."""
        await verilog_env.initialize()
        obs = await verilog_env.terminate()

        assert obs["terminated"] is True
        assert "message" in obs
        assert obs["message"] == "Environment terminated."

    def test_validate_tool_calls_dict(self, verilog_env):
        """Test tool call validation with dictionary input."""
        tool_call_dict = {
            "tool": "write_file",
            "args": {"path": "test.v", "content": "module test(); endmodule"},
        }

        validated = verilog_env.validate_tool_calls(tool_call_dict)

        assert isinstance(validated, EnvToolCall)
        assert validated.tool == "write_file"
        assert validated.args["path"] == "test.v"

    def test_validate_tool_calls_list(self, verilog_env):
        """Test tool call validation with list input."""
        tool_call_list = [{"tool": "compile", "args": {"sources": ["test.v"]}}]

        validated = verilog_env.validate_tool_calls(tool_call_list)

        assert isinstance(validated, EnvToolCall)
        assert validated.tool == "compile"
        assert validated.args["sources"] == ["test.v"]

    def test_validate_tool_calls_env_tool_call(self, verilog_env):
        """Test tool call validation with EnvToolCall input."""
        original_call = EnvToolCall(tool="simulate", args={})

        validated = verilog_env.validate_tool_calls(original_call)

        assert validated is original_call

    def test_validate_tool_calls_invalid_tool(self, verilog_env):
        """Test tool call validation with invalid tool name."""
        tool_call_dict = {"tool": "invalid_tool", "args": {}}

        with pytest.raises(ValueError, match="Unknown tool: invalid_tool"):
            verilog_env.validate_tool_calls(tool_call_dict)

    def test_validate_tool_calls_empty_list(self, verilog_env):
        """Test tool call validation with empty list."""
        with pytest.raises(ValueError, match="Received empty list"):
            verilog_env.validate_tool_calls([])

    @pytest.mark.asyncio
    async def test_step_write_file(self, verilog_env):
        """Test environment step with write_file tool."""
        await verilog_env.initialize()

        tool_call = EnvToolCall(
            tool="write_file",
            args={"path": "test.v", "content": "module test(); endmodule"},
        )

        obs = await verilog_env.step(tool_call)

        assert "test.v" in obs["files"]
        assert "module test();" in obs["files"]["test.v"]
        assert obs["reward_last"] < 0  # Step penalty

    @pytest.mark.asyncio
    @patch("src.examples.verilog.engine.subprocess.run")
    async def test_step_compile_success(self, mock_run, verilog_env):
        """Test environment step with successful compilation."""
        await verilog_env.initialize()

        # Mock successful compilation
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ""
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        tool_call = EnvToolCall(tool="compile", args={})
        obs = await verilog_env.step(tool_call)

        assert "Last compile: Success" in obs["compile_status"]
        assert obs["reward_last"] > 0  # Compile success reward minus step penalty

    @pytest.mark.asyncio
    @patch("src.examples.verilog.engine.subprocess.run")
    async def test_step_compile_failure(self, mock_run, verilog_env):
        """Test environment step with compilation failure."""
        await verilog_env.initialize()

        # Mock failed compilation
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = ""
        mock_proc.stderr = "Error: syntax error"
        mock_run.return_value = mock_proc

        tool_call = EnvToolCall(tool="compile", args={})
        obs = await verilog_env.step(tool_call)

        assert "Last compile: Failed" in obs["compile_status"]
        assert obs["reward_last"] < 0  # Only step penalty

    @pytest.mark.asyncio
    @patch("src.examples.verilog.engine.subprocess.run")
    async def test_step_simulate_success(self, mock_run, verilog_env):
        """Test environment step with successful simulation."""
        await verilog_env.initialize()

        # Mock successful simulation
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "Mismatches: 0 in 10 samples"
        mock_proc.stderr = ""
        mock_run.return_value = mock_proc

        tool_call = EnvToolCall(tool="simulate", args={})
        obs = await verilog_env.step(tool_call)

        assert "Last simulation: Passed" in obs["simulate_status"]
        assert obs["task_completed"] is True
        assert obs["terminated"] is True
        assert obs["reward_last"] > 0.5  # Large simulation success reward

    @pytest.mark.asyncio
    async def test_step_submit(self, verilog_env):
        """Test environment step with submit tool."""
        await verilog_env.initialize()

        tool_call = EnvToolCall(tool="submit", args={})
        obs = await verilog_env.step(tool_call)

        assert obs["terminated"] is True

    @pytest.mark.asyncio
    async def test_checkpoint(self, verilog_env):
        """Test environment checkpoint functionality."""
        await verilog_env.initialize()

        obs = await verilog_env.checkpoint()

        assert "engine_snapshot_data" in obs
        assert isinstance(obs["engine_snapshot_data"], dict)


class TestVerilogTools:
    """Test suite for Verilog tool implementations."""

    @pytest.fixture
    def mock_engine(self):
        """Create a mock engine for tool testing."""
        engine = MagicMock()
        return engine

    @pytest.mark.asyncio
    async def test_write_file_tool(self, mock_engine):
        """Test VerilogWriteFileTool."""
        tool = VerilogWriteFileTool(mock_engine)

        # Mock async method properly
        async def mock_write_file(*args, **kwargs):
            return {"ok": True, "type": "write_file"}

        mock_engine.write_file = mock_write_file

        call = EnvToolCall(
            tool="write_file",
            args={"path": "test.v", "content": "module test(); endmodule"},
        )

        result = await tool(call)

        assert isinstance(result, ToolResult)
        assert result.ok is True

    @pytest.mark.asyncio
    async def test_write_file_tool_error(self, mock_engine):
        """Test VerilogWriteFileTool with error."""
        tool = VerilogWriteFileTool(mock_engine)

        async def mock_write_file_error(*args, **kwargs):
            raise Exception("Write error")

        mock_engine.write_file = mock_write_file_error

        call = EnvToolCall(tool="write_file", args={"path": "test.v", "content": "test"})

        result = await tool(call)

        assert result.ok is False
        assert "Write error" in result.error

    @pytest.mark.asyncio
    async def test_compile_tool(self, mock_engine):
        """Test VerilogCompileTool."""
        tool = VerilogCompileTool(mock_engine)

        async def mock_compile(*args, **kwargs):
            return {"ok": True, "type": "compile", "returncode": 0}

        mock_engine.compile = mock_compile

        call = EnvToolCall(tool="compile", args={"sources": ["test.v"], "testbench": "test_tb.v"})

        result = await tool(call)

        assert result.ok is True

    @pytest.mark.asyncio
    async def test_compile_tool_no_args(self, mock_engine):
        """Test VerilogCompileTool with no arguments."""
        tool = VerilogCompileTool(mock_engine)

        async def mock_compile(*args, **kwargs):
            return {"ok": True, "type": "compile"}

        mock_engine.compile = mock_compile

        call = EnvToolCall(tool="compile", args={})

        result = await tool(call)

        assert result.ok is True

    @pytest.mark.asyncio
    async def test_simulate_tool(self, mock_engine):
        """Test VerilogSimulateTool."""
        tool = VerilogSimulateTool(mock_engine)

        async def mock_simulate(*args, **kwargs):
            return {"ok": True, "type": "simulate", "passed": True}

        mock_engine.simulate = mock_simulate

        call = EnvToolCall(tool="simulate", args={"binary": "test.out"})

        result = await tool(call)

        assert result.ok is True

    @pytest.mark.asyncio
    async def test_submit_tool(self, mock_engine):
        """Test VerilogSubmitTool."""
        tool = VerilogSubmitTool(mock_engine)

        async def mock_submit(*args, **kwargs):
            return {"ok": True, "type": "submit", "submitted": True}

        mock_engine.submit = mock_submit

        call = EnvToolCall(tool="submit", args={})

        result = await tool(call)

        assert result.ok is True


class TestVerilogObservationCallable:
    """Test suite for VerilogObservationCallable."""

    @pytest.mark.asyncio
    async def test_get_observation_basic(self):
        """Test basic observation generation."""
        callable_obj = VerilogObservationCallable()

        pub = VerilogPublicState(
            files={"test.v": "module test(); endmodule"},
            build_dir="/tmp/build",
            task_completed=False,
        )

        priv = VerilogPrivateState(
            reward_last=0.1, total_reward=0.5, terminated=False, truncated=False
        )

        obs = await callable_obj.get_observation(pub, priv)

        assert obs["files"] == pub.files
        assert obs["build_dir"] == pub.build_dir
        assert obs["files_summary"] == "1 Verilog files available: test.v"
        assert obs["task_completed"] is False
        assert obs["reward_last"] == 0.1
        assert obs["total_reward"] == 0.5
        assert obs["terminated"] is False
        assert obs["compile_status"] == ""
        assert obs["simulate_status"] == ""

    @pytest.mark.asyncio
    async def test_get_observation_with_compile_status(self):
        """Test observation with compile status."""
        callable_obj = VerilogObservationCallable()

        pub = VerilogPublicState(
            files={},
            build_dir="/tmp/build",
            task_completed=False,
            last_compile_output="Compilation successful",
        )

        priv = VerilogPrivateState(
            reward_last=0.0, total_reward=0.0, terminated=False, truncated=False
        )

        obs = await callable_obj.get_observation(pub, priv)

        assert obs["compile_status"] == "Last compile: Success"

    @pytest.mark.asyncio
    async def test_get_observation_with_compile_error(self):
        """Test observation with compile error."""
        callable_obj = VerilogObservationCallable()

        pub = VerilogPublicState(
            files={},
            build_dir="/tmp/build",
            task_completed=False,
            last_compile_output="Error: syntax error",
        )

        priv = VerilogPrivateState(
            reward_last=0.0, total_reward=0.0, terminated=False, truncated=False
        )

        obs = await callable_obj.get_observation(pub, priv)

        assert obs["compile_status"] == "Last compile: Failed"

    @pytest.mark.asyncio
    async def test_get_observation_with_simulate_status_passed(self):
        """Test observation with successful simulation."""
        callable_obj = VerilogObservationCallable()

        pub = VerilogPublicState(
            files={},
            build_dir="/tmp/build",
            task_completed=True,
            last_simulate_output="Mismatches: 0 in 10 samples",
        )

        priv = VerilogPrivateState(
            reward_last=1.0, total_reward=1.0, terminated=True, truncated=False
        )

        obs = await callable_obj.get_observation(pub, priv)

        assert obs["simulate_status"] == "Last simulation: Passed"
        assert obs["task_completed"] is True
        assert obs["terminated"] is True

    @pytest.mark.asyncio
    async def test_get_observation_with_simulate_status_failed(self):
        """Test observation with failed simulation."""
        callable_obj = VerilogObservationCallable()

        pub = VerilogPublicState(
            files={},
            build_dir="/tmp/build",
            task_completed=False,
            last_simulate_output="Mismatches: 5 in 10 samples",
        )

        priv = VerilogPrivateState(
            reward_last=0.0, total_reward=0.0, terminated=False, truncated=False
        )

        obs = await callable_obj.get_observation(pub, priv)

        assert obs["simulate_status"] == "Last simulation: Failed"

    @pytest.mark.asyncio
    async def test_get_observation_multiple_files(self):
        """Test observation with multiple files."""
        callable_obj = VerilogObservationCallable()

        pub = VerilogPublicState(
            files={
                "TopModule.v": "module TopModule(); endmodule",
                "RefModule.v": "module RefModule(); endmodule",
                "test_tb.v": "module test_tb(); endmodule",
            },
            build_dir="/tmp/build",
            task_completed=False,
        )

        priv = VerilogPrivateState(
            reward_last=0.0, total_reward=0.0, terminated=False, truncated=False
        )

        obs = await callable_obj.get_observation(pub, priv)

        expected_summary = "3 Verilog files available: TopModule.v, RefModule.v, test_tb.v"
        assert obs["files_summary"] == expected_summary


class TestInputSchemas:
    """Test suite for tool input schemas."""

    def test_write_file_input_valid(self):
        """Test WriteFileInput with valid data."""
        data = {"path": "test.v", "content": "module test(); endmodule"}
        input_obj = WriteFileInput(**data)

        assert input_obj.path == "test.v"
        assert input_obj.content == "module test(); endmodule"

    def test_write_file_input_missing_required(self):
        """Test WriteFileInput with missing required fields."""
        with pytest.raises(ValueError):
            WriteFileInput(path="test.v")  # Missing content

    def test_compile_input_valid(self):
        """Test CompileInput with valid data."""
        data = {"sources": ["test.v"], "testbench": "test_tb.v"}
        input_obj = CompileInput(**data)

        assert input_obj.sources == ["test.v"]
        assert input_obj.testbench == "test_tb.v"

    def test_compile_input_optional_fields(self):
        """Test CompileInput with optional fields."""
        input_obj = CompileInput()

        assert input_obj.sources is None
        assert input_obj.testbench is None

    def test_simulate_input_valid(self):
        """Test SimulateInput with valid data."""
        data = {"binary": "test.out"}
        input_obj = SimulateInput(**data)

        assert input_obj.binary == "test.out"

    def test_simulate_input_optional(self):
        """Test SimulateInput with optional binary."""
        input_obj = SimulateInput()

        assert input_obj.binary is None

    def test_submit_input(self):
        """Test SubmitInput (no fields)."""
        input_obj = SubmitInput()

        # Should create successfully with no fields
        assert input_obj is not None
