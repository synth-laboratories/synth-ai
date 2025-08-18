from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from synth_ai.environments.environment.shared_engine import (
    GetObservationCallable,
    InternalObservation,
)
from synth_ai.environments.environment.tools import (
    TOOL_REGISTRY,
    AbstractTool,
    EnvToolCall,
    ToolResult,
    register_tool,
)
from synth_ai.environments.examples.verilog.engine import (
    VerilogEngine,
    VerilogEngineSnapshot,
    VerilogPrivateState,
    VerilogPublicState,
)
from synth_ai.environments.stateful.core import StatefulEnvironment
from synth_ai.environments.tasks.core import TaskInstance


# Tool Input Schemas
class WriteFileInput(BaseModel):
    path: str
    content: str


class CompileInput(BaseModel):
    sources: Optional[List[str]] = None
    testbench: Optional[str] = None


class SimulateInput(BaseModel):
    binary: Optional[str] = None


class SubmitInput(BaseModel):
    pass  # No arguments needed for submit


# Tool Implementations
class VerilogWriteFileTool(AbstractTool):
    name = "write_file"
    description = "Write content to a Verilog file"
    call_schema = WriteFileInput
    result_schema = ToolResult

    def __init__(self, engine: VerilogEngine):
        self.engine = engine

    async def __call__(self, call: EnvToolCall) -> ToolResult:
        try:
            validated_args = self.call_schema(**call.args)
            result = await self.engine.write_file(validated_args.path, validated_args.content)
            return ToolResult(ok=result["ok"], payload=result)
        except Exception as e:
            return ToolResult(ok=False, error=str(e))


class VerilogCompileTool(AbstractTool):
    name = "compile"
    description = "Compile Verilog sources with iverilog"
    call_schema = CompileInput
    result_schema = ToolResult

    def __init__(self, engine: VerilogEngine):
        self.engine = engine

    async def __call__(self, call: EnvToolCall) -> ToolResult:
        try:
            validated_args = self.call_schema(**call.args)
            result = await self.engine.compile(validated_args.sources, validated_args.testbench)
            return ToolResult(ok=result["ok"], payload=result)
        except Exception as e:
            return ToolResult(ok=False, error=str(e))


class VerilogSimulateTool(AbstractTool):
    name = "simulate"
    description = "Run vvp on compiled binary"
    call_schema = SimulateInput
    result_schema = ToolResult

    def __init__(self, engine: VerilogEngine):
        self.engine = engine

    async def __call__(self, call: EnvToolCall) -> ToolResult:
        try:
            validated_args = self.call_schema(**call.args)
            result = await self.engine.simulate(validated_args.binary)
            return ToolResult(ok=result["ok"], payload=result)
        except Exception as e:
            return ToolResult(ok=False, error=str(e))


class VerilogSubmitTool(AbstractTool):
    name = "submit"
    description = "Submit solution for grading"
    call_schema = SubmitInput
    result_schema = ToolResult

    def __init__(self, engine: VerilogEngine):
        self.engine = engine

    async def __call__(self, call: EnvToolCall) -> ToolResult:
        try:
            result = await self.engine.submit()
            return ToolResult(ok=result["ok"], payload=result)
        except Exception as e:
            return ToolResult(ok=False, error=str(e))


class VerilogObservationCallable(GetObservationCallable):
    async def get_observation(
        self, pub: VerilogPublicState, priv: VerilogPrivateState
    ) -> InternalObservation:
        files_summary = f"{len(pub.files)} Verilog files available"
        if pub.files:
            files_summary += f": {', '.join(pub.files.keys())}"

        compile_status = ""
        if pub.last_compile_output is not None:
            # Check for common error indicators in compile output
            output_lower = pub.last_compile_output.lower()
            is_success = not any(
                indicator in output_lower for indicator in ["error", "failed", "syntax"]
            )
            if is_success:
                compile_status = "Last compile: Success"
            else:
                # Include the actual error message to help the agent debug
                compile_status = f"Last compile: Failed\n{pub.last_compile_output}"

        simulate_status = ""
        if pub.last_simulate_output:
            # Use same success detection logic as in engine
            stdout = pub.last_simulate_output
            passed = (
                "ALL_TESTS_PASSED" in stdout
                or ("Mismatches: 0 " in stdout and "samples" in stdout)
                or ("no mismatches" in stdout.lower() and "errors" not in stdout.lower())
            )
            simulate_status = f"Last simulation: {'Passed' if passed else 'Failed'}"

        observation: Dict[str, Any] = {
            "files": pub.files,
            "build_dir": pub.build_dir,
            "files_summary": files_summary,
            "task_completed": pub.task_completed,
            "reward_last": priv.reward_last,
            "total_reward": priv.total_reward,
            "terminated": priv.terminated,
            "compile_status": compile_status,
            "simulate_status": simulate_status,
        }
        return observation  # type: ignore[return-value]


class VerilogEnvironment(StatefulEnvironment):
    def __init__(
        self,
        task_instance: TaskInstance,
        custom_obs: Optional[GetObservationCallable] = None,
    ):
        self.name = "VerilogEval"
        self.task_instance = task_instance
        self.custom_observation_callable = custom_obs or VerilogObservationCallable()
        self.engine: VerilogEngine = VerilogEngine(task_instance)

        # Initialize tools
        self._tools_instances = {
            "write_file": VerilogWriteFileTool(self.engine),
            "compile": VerilogCompileTool(self.engine),
            "simulate": VerilogSimulateTool(self.engine),
            "submit": VerilogSubmitTool(self.engine),
        }

        # Register tools
        for tool_name, tool_instance in self._tools_instances.items():
            if tool_name not in TOOL_REGISTRY:
                register_tool(tool_instance)

    async def initialize(self) -> InternalObservation:
        priv, pub = await self.engine._reset_engine()
        return await self._to_observation(priv, pub)

    async def terminate(self) -> InternalObservation:
        # Get current state and mark as terminated
        try:
            # Try to get current state from engine
            current_files = self.engine._get_file_contents()
            build_dir = str(self.engine.build_dir) if self.engine.build_dir else ""

            priv = VerilogPrivateState(
                reward_last=0.0,
                total_reward=self.engine._total_reward,
                terminated=True,
                truncated=False,
            )

            pub = VerilogPublicState(files=current_files, build_dir=build_dir, task_completed=False)
        except Exception:
            # Fallback if engine state is not accessible
            priv = VerilogPrivateState(
                reward_last=0.0, total_reward=0.0, terminated=True, truncated=False
            )

            pub = VerilogPublicState(files={}, build_dir="", task_completed=False)

        obs = await self._to_observation(priv, pub)
        if isinstance(obs, dict):
            obs["terminated"] = True
            obs["message"] = "Environment terminated."
        return obs

    def validate_tool_calls(
        self,
        tool_calls: Union[
            EnvToolCall,
            List[Dict[str, Any]],
            List[List[Dict[str, Any]]],
            Dict[str, Any],
        ],
    ) -> EnvToolCall:
        """Normalize and validate tool calls to a single EnvToolCall."""
        raw_call_data: Dict[str, Any]

        if isinstance(tool_calls, list):
            if not tool_calls:
                raise ValueError("Received empty list of tool calls.")
            first_item = tool_calls[0]
            if isinstance(first_item, list):
                if not first_item:
                    raise ValueError("Received empty inner list of tool calls.")
                raw_call_data = first_item[0]
            elif isinstance(first_item, dict):
                raw_call_data = first_item
            elif isinstance(first_item, EnvToolCall):
                return first_item
            else:
                raise TypeError(f"Unexpected type in tool_calls list: {type(first_item)}")
        elif isinstance(tool_calls, dict):
            raw_call_data = tool_calls
        elif isinstance(tool_calls, EnvToolCall):
            return tool_calls
        else:
            raise TypeError(f"Unexpected type for tool_calls: {type(tool_calls)}")

        if not isinstance(raw_call_data, dict):
            raise TypeError(f"Processed call data is not a dict: {type(raw_call_data)}")

        # Convert dict to EnvToolCall instance
        tool_name = raw_call_data.get("tool")
        tool_args = raw_call_data.get("args", {})

        valid_tools = {"write_file", "compile", "simulate", "submit"}
        if tool_name not in valid_tools:
            raise ValueError(f"Unknown tool: {tool_name}. Expected one of: {valid_tools}")

        return EnvToolCall(tool=tool_name, args=tool_args)

    async def step(
        self,
        tool_calls: Union[
            EnvToolCall,
            List[Dict[str, Any]],
            List[List[Dict[str, Any]]],
            Dict[str, Any],
        ],
    ) -> InternalObservation:
        agent_call = self.validate_tool_calls(tool_calls)

        # Get the appropriate tool
        tool_instance = self._tools_instances.get(agent_call.tool)
        if not tool_instance:
            tool_instance = TOOL_REGISTRY.get(agent_call.tool)
            if not tool_instance:
                raise ValueError(f"Tool '{agent_call.tool}' not found.")

        # Execute the tool
        tool_result: ToolResult = await tool_instance(agent_call)

        # Update engine state with tool result
        if tool_result.payload:
            action_result = tool_result.payload
        elif not tool_result.ok:
            action_result = {
                "ok": False,
                "error": tool_result.error,
                "type": agent_call.tool,
            }
        else:
            action_result = {}

        priv_state, pub_state = await self.engine._step_engine(action_result)

        return await self._to_observation(priv_state, pub_state)

    async def checkpoint(self) -> InternalObservation:
        engine_snapshot: VerilogEngineSnapshot = await self.engine._serialize_engine()

        # Get current state for observation
        try:
            current_files = self.engine._get_file_contents()
            build_dir = str(self.engine.build_dir) if self.engine.build_dir else ""

            priv = VerilogPrivateState(
                reward_last=0.0,
                total_reward=self.engine._total_reward,
                terminated=False,
                truncated=False,
            )

            pub = VerilogPublicState(files=current_files, build_dir=build_dir, task_completed=False)

            obs_data = await self._to_observation(priv, pub)
        except Exception:
            obs_data = {"message": "Checkpoint created"}

        if isinstance(obs_data, dict):
            obs_data["engine_snapshot_data"] = engine_snapshot.model_dump()

        return obs_data

    async def _to_observation(
        self,
        priv: VerilogPrivateState,
        pub: VerilogPublicState,
        extra_obs: Optional[Dict[str, Any]] = None,
    ) -> InternalObservation:
        observation = await self.custom_observation_callable.get_observation(pub, priv)
        if extra_obs and isinstance(observation, dict):
            observation.update(extra_obs)
        return observation

    async def _serialize_engine(self) -> VerilogEngineSnapshot:
        return await self.engine._serialize_engine()

    @classmethod
    async def _deserialize_engine(
        cls, snapshot: VerilogEngineSnapshot, task_instance: TaskInstance
    ) -> "VerilogEnvironment":
        eng = await VerilogEngine._deserialize_engine(snapshot)
        env = cls(task_instance)
        env.engine = eng
        return env
