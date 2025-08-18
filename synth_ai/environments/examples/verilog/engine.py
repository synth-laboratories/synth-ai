from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from synth_ai.environments.environment.rewards.core import RewardComponent, RewardStack
from synth_ai.environments.stateful.engine import StatefulEngine, StatefulEngineSnapshot
from synth_ai.environments.tasks.core import TaskInstance


@dataclass
class VerilogEngineSnapshot(StatefulEngineSnapshot):
    task_instance_dict: Dict
    engine_snapshot: Dict

    def model_dump(self) -> Dict:
        """Convert dataclass to dictionary for compatibility with Pydantic models."""
        return {
            "task_instance_dict": self.task_instance_dict,
            "engine_snapshot": self.engine_snapshot,
        }


@dataclass
class VerilogPublicState:
    files: Dict[str, str]
    build_dir: str
    task_completed: bool = False
    last_compile_output: Optional[str] = None
    last_simulate_output: Optional[str] = None


@dataclass
class VerilogPrivateState:
    reward_last: float
    total_reward: float
    terminated: bool
    truncated: bool


class VerilogCompileSuccessComponent(RewardComponent):
    async def score(self, state: VerilogPublicState, action: Any) -> float:
        if hasattr(action, "get") and action.get("type") == "compile":
            # Check if compilation was successful (returncode 0)
            if action.get("returncode") == 0:
                return 0.1
        return 0.0


class VerilogSimulationPassComponent(RewardComponent):
    async def score(self, state: VerilogPublicState, action: Any) -> float:
        if hasattr(action, "get") and action.get("type") == "simulate":
            # Check if simulation passed
            if action.get("passed", False):
                return 1.0
        return 0.0


class VerilogStepPenaltyComponent(RewardComponent):
    def __init__(self, penalty: float = -0.01):
        self.penalty = penalty

    async def score(self, state: Any, action: Any) -> float:
        return self.penalty


class VerilogEngine(StatefulEngine):
    """
    Stateful Verilog evaluation engine with persistent artifact snapshots.
    """

    def __init__(self, task_instance: TaskInstance):
        self.task_instance = task_instance
        self._total_reward = 0.0
        self._current_action_for_reward: Optional[Dict[str, Any]] = None

        self.reward_stack = RewardStack(
            components=[
                VerilogCompileSuccessComponent(),
                VerilogSimulationPassComponent(),
                VerilogStepPenaltyComponent(penalty=-0.01),
            ]
        )

        # Initialize paths - will be set properly in _reset_engine
        self.snapshot_dir: Optional[Path] = None
        self.build_dir: Optional[Path] = None

        # Track last compile/simulate outputs
        self._last_compile_output: Optional[str] = None
        self._last_simulate_output: Optional[str] = None

    async def _reset_engine(
        self, *, seed: Optional[int] = None
    ) -> Tuple[VerilogPrivateState, VerilogPublicState]:
        """Initialize the Verilog environment with task files."""
        self._total_reward = 0.0
        self._current_action_for_reward = None
        self._last_compile_output = None
        self._last_simulate_output = None

        # Initialize snapshot from task instance
        self._init_snapshot()

        priv = VerilogPrivateState(
            reward_last=0.0, total_reward=0.0, terminated=False, truncated=False
        )

        pub = VerilogPublicState(
            files=self._get_file_contents(),
            build_dir=str(self.build_dir),
            task_completed=False,
        )

        return priv, pub

    async def _step_engine(
        self, action_result: Dict[str, Any]
    ) -> Tuple[VerilogPrivateState, VerilogPublicState]:
        """Process an action result and update engine state."""
        self._current_action_for_reward = action_result

        # Update last outputs if this is a compile or simulate action
        if action_result.get("type") == "compile":
            stdout = action_result.get("stdout", "")
            stderr = action_result.get("stderr", "")
            # Combine stdout and stderr for compile output, stderr has the error info
            self._last_compile_output = stderr if stderr else stdout
        elif action_result.get("type") == "simulate":
            self._last_simulate_output = action_result.get("stdout")

        # Calculate reward using RewardStack
        current_pub_state = VerilogPublicState(
            files=self._get_file_contents(),
            build_dir=str(self.build_dir),
            task_completed=action_result.get("passed", False),
        )

        reward_from_stack = await self.reward_stack.step_reward(
            state=current_pub_state, action=self._current_action_for_reward
        )
        self._current_action_for_reward = None

        self._total_reward += reward_from_stack

        # Check termination conditions
        terminated = action_result.get("passed", False) or action_result.get("submitted", False)

        priv = VerilogPrivateState(
            reward_last=reward_from_stack,
            total_reward=self._total_reward,
            terminated=terminated,
            truncated=False,
        )

        pub = VerilogPublicState(
            files=self._get_file_contents(),
            build_dir=str(self.build_dir),
            task_completed=action_result.get("passed", False),
            last_compile_output=self._last_compile_output,
            last_simulate_output=self._last_simulate_output,
        )

        return priv, pub

    def _init_snapshot(self) -> None:
        """Initialize snapshot directory from task instance data."""
        if not hasattr(self.task_instance, "snapshot_dir"):
            raise ValueError("Task instance must have a snapshot_dir attribute")

        self.snapshot_dir = Path(self.task_instance.snapshot_dir)

        if self.snapshot_dir.exists() and any(self.snapshot_dir.iterdir()):
            # Already initialized
            self.build_dir = self.snapshot_dir / "build"
            self.build_dir.mkdir(exist_ok=True)
            return

        # Copy pristine files from task data
        pristine_dir = getattr(self.task_instance, "pristine_dir", None)
        if pristine_dir and Path(pristine_dir).exists():
            shutil.copytree(pristine_dir, self.snapshot_dir, dirs_exist_ok=True)
        else:
            # Create basic structure if no pristine dir
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        self.build_dir = self.snapshot_dir / "build"
        self.build_dir.mkdir(exist_ok=True)

    def _get_file_contents(self) -> Dict[str, str]:
        """Get contents of all Verilog files in the snapshot directory."""
        if not self.snapshot_dir:
            return {}

        files = {}
        for p in self.snapshot_dir.rglob("*.v"):
            try:
                relative_path = p.relative_to(self.snapshot_dir)
                files[str(relative_path)] = p.read_text()
            except Exception:
                continue
        return files

    async def write_file(self, path: str, content: str) -> Dict[str, Any]:
        """Write content to a file in the snapshot directory."""
        if not self.snapshot_dir:
            return {"ok": False, "error": "Snapshot directory not initialized"}

        file_path = self.snapshot_dir / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return {"ok": True, "type": "write_file"}

    async def compile(
        self, sources: Optional[list] = None, testbench: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compile Verilog sources with iverilog."""
        if not self.snapshot_dir or not self.build_dir:
            return {"ok": False, "error": "Directories not initialized"}

        # Default to all .v files if no sources specified
        if sources is None:
            sources = [str(p.relative_to(self.snapshot_dir)) for p in self.snapshot_dir.glob("*.v")]

        src_paths = [self.snapshot_dir / src for src in sources]

        # Add testbench if specified
        if testbench:
            tb_path = self.snapshot_dir / testbench
            if tb_path.exists():
                src_paths.append(tb_path)

        binary = self.build_dir / "a.out"
        cmd = ["iverilog", "-g2012", "-o", str(binary)] + [str(p) for p in src_paths]

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return {
                "ok": proc.returncode == 0,
                "type": "compile",
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "returncode": proc.returncode,
                "binary": str(binary) if proc.returncode == 0 else None,
            }
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "Compilation timeout", "type": "compile"}
        except Exception as e:
            return {"ok": False, "error": str(e), "type": "compile"}

    async def simulate(self, binary: Optional[str] = None) -> Dict[str, Any]:
        """Run vvp on compiled binary."""
        if not self.build_dir:
            return {"ok": False, "error": "Build directory not initialized"}

        bin_path = binary if binary else str(self.build_dir / "a.out")

        try:
            proc = subprocess.run(["vvp", bin_path], capture_output=True, text=True, timeout=30)

            # Check for various success indicators
            stdout = proc.stdout
            passed = (
                "ALL_TESTS_PASSED" in stdout
                or ("Mismatches: 0 " in stdout and "samples" in stdout)
                or ("no mismatches" in stdout.lower() and "errors" not in stdout.lower())
            )

            return {
                "ok": True,
                "type": "simulate",
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "returncode": proc.returncode,
                "passed": passed,
            }
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "Simulation timeout", "type": "simulate"}
        except Exception as e:
            return {"ok": False, "error": str(e), "type": "simulate"}

    async def submit(self) -> Dict[str, Any]:
        """Submit solution for grading."""
        # For now, simple check based on last simulation
        # In a full implementation, this would call the task's verify method
        return {
            "ok": True,
            "type": "submit",
            "passed": True,  # Placeholder
            "detail": "Submission processed",
            "submitted": True,
        }

    async def _serialize_engine(self) -> VerilogEngineSnapshot:
        """Serialize engine state to a snapshot."""
        engine_data = {
            "total_reward": self._total_reward,
            "snapshot_dir": str(self.snapshot_dir) if self.snapshot_dir else None,
            "build_dir": str(self.build_dir) if self.build_dir else None,
        }

        task_instance_dict = await self.task_instance.serialize()

        return VerilogEngineSnapshot(
            task_instance_dict=task_instance_dict, engine_snapshot=engine_data
        )

    @classmethod
    async def _deserialize_engine(cls, snapshot: VerilogEngineSnapshot) -> "VerilogEngine":
        """Deserialize engine from snapshot."""
        # This would need proper task instance deserialization
        # For now, create a minimal implementation
        from synth_ai.environments.examples.verilog.taskset import VerilogTaskInstance

        task_instance = await VerilogTaskInstance.deserialize(snapshot.task_instance_dict)
        engine = cls(task_instance)

        engine_data = snapshot.engine_snapshot
        engine._total_reward = engine_data.get("total_reward", 0.0)

        if engine_data.get("snapshot_dir"):
            engine.snapshot_dir = Path(engine_data["snapshot_dir"])
        if engine_data.get("build_dir"):
            engine.build_dir = Path(engine_data["build_dir"])

        return engine
