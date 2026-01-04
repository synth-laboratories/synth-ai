"""Verilog business logic: dataset, workspace, and tools (no synth-ai dependencies)."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any


# Dataset configuration
DATASET_NAME = os.getenv("VERILOG_DATASET_NAME", "dakies/nvlabs-verilogeval-v2-spec-to-rtl")
DEFAULT_SPLIT = "test"
AVAILABLE_SPLITS: tuple[str, ...] = ("test",)  # VerilogEval v2 only has test split

# Tool names
TOOL_WRITE_FILE = "write_file"
TOOL_OPEN_FILE = "open_file"
TOOL_COMPILE = "compile"
TOOL_SIMULATE = "simulate"
TOOL_SUBMIT = "submit"

# Max agentic steps
MAX_STEPS = 10


def _compute_repo_root() -> Path:
    """Compute the repository root path."""
    p = Path(__file__).resolve()
    parents = list(p.parents)
    if len(parents) >= 4:
        return parents[3]
    if "/opt/synth_ai_repo" in os.getenv("PYTHONPATH", "") or Path("/opt/synth_ai_repo/synth_ai").exists():
        return Path("/opt/synth_ai_repo")
    return Path.cwd()


REPO_ROOT = _compute_repo_root()


class VerilogEvalDataset:
    """Lazy Hugging Face dataset loader for VerilogEval v2."""

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    def _load_split(self, split: str):
        if split not in AVAILABLE_SPLITS:
            raise ValueError(f"Unknown split: {split}. Available: {AVAILABLE_SPLITS}")
        if split not in self._cache:
            try:
                from datasets import load_dataset as _load_dataset

                print(
                    f"[VerilogEvalDataset] Loading dataset '{DATASET_NAME}' split '{split}'",
                    flush=True,
                )

                ds = _load_dataset(
                    DATASET_NAME,
                    split=split,
                    trust_remote_code=True,
                )

                self._cache[split] = ds
                print(
                    f"[VerilogEvalDataset] Successfully loaded {len(ds)} examples from '{DATASET_NAME}' split '{split}'",
                    flush=True,
                )
            except Exception as exc:
                import traceback
                error_details = traceback.format_exc()
                print(
                    f"[VerilogEvalDataset] Dataset load failed: {exc}\n{error_details}",
                    flush=True,
                )
                raise RuntimeError(
                    f"Dataset preparation failed: {split}: Failed to load VerilogEval dataset. "
                    f"Dataset: {DATASET_NAME} | Split: {split} | Error: {exc}"
                ) from exc
        return self._cache[split]

    def ensure_ready(self, splits: Sequence[str]) -> None:
        """Preload dataset splits."""
        for split in splits:
            self._load_split(split)

    def size(self, split: str) -> int:
        """Get the number of examples in a split."""
        dataset = self._load_split(split)
        return len(dataset)

    def sample(self, *, split: str, index: int) -> dict[str, Any]:
        """Get a sample from the dataset by index."""
        dataset = self._load_split(split)
        size = len(dataset)
        if size == 0:
            raise RuntimeError(f"VerilogEval split '{split}' is empty")
        idx = int(index) % size
        row = dataset[int(idx)]

        return {
            "index": idx,
            "split": split,
            "problem_id": str(row.get("problem_id", f"problem_{idx}")),
            "prompt": str(row.get("prompt", "")),
            "test": str(row.get("test", "")),  # testbench
            "ref": str(row.get("ref", "")),    # reference solution
        }


class VerilogWorkspace:
    """Manages a temporary workspace for Verilog compilation and simulation."""

    def __init__(self, problem_id: str, prompt: str, testbench: str, ref_solution: str):
        self.problem_id = problem_id
        self.prompt = prompt
        self.testbench = testbench
        self.ref_solution = ref_solution
        self.workspace_dir = Path(tempfile.mkdtemp(prefix=f"verilog_{problem_id}_"))
        self.files: dict[str, str] = {}
        self.last_compile_output: str | None = None
        self.last_simulate_output: str | None = None
        self.submitted = False
        self.passed = False

        # Write initial files
        self._setup_workspace()

    def _setup_workspace(self):
        """Set up the workspace with initial files."""
        # Create incomplete module template
        module_content = f"""module TopModule();
    // TODO: Implement the module based on the specification below
    /*
    Specification:
    {self.prompt.strip()}
    */
endmodule"""

        # Write files
        (self.workspace_dir / "TopModule.v").write_text(module_content)
        (self.workspace_dir / f"{self.problem_id}_tb.v").write_text(self.testbench)
        (self.workspace_dir / "RefModule.v").write_text(self.ref_solution)

        self.files = {
            "TopModule.v": module_content,
            f"{self.problem_id}_tb.v": self.testbench,
            "RefModule.v": self.ref_solution,
        }

    def write_file(self, path: str, content: str) -> dict[str, Any]:
        """Write content to a file in the workspace."""
        try:
            file_path = self.workspace_dir / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            self.files[path] = content
            return {"ok": True, "message": f"Wrote {len(content)} bytes to {path}"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def read_file(self, path: str) -> dict[str, Any]:
        """Read content from a file in the workspace."""
        try:
            file_path = self.workspace_dir / path
            if not file_path.exists():
                return {"ok": False, "error": f"File '{path}' not found"}
            content = file_path.read_text()
            return {"ok": True, "content": content}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def compile(self, sources: list[str] | None = None, testbench: str | None = None) -> dict[str, Any]:
        """Compile Verilog sources with iverilog."""
        try:
            # Default sources
            if sources is None:
                sources = ["TopModule.v"]
            if testbench is None:
                testbench = f"{self.problem_id}_tb.v"

            # Build compile command
            all_sources = sources + [testbench]
            source_paths = [str(self.workspace_dir / s) for s in all_sources]
            output_path = str(self.workspace_dir / "a.out")

            cmd = ["iverilog", "-g2012", "-o", output_path] + source_paths

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.workspace_dir),
            )

            self.last_compile_output = result.stdout + result.stderr

            if result.returncode == 0:
                return {"ok": True, "output": self.last_compile_output, "binary": "a.out"}
            else:
                return {"ok": False, "output": self.last_compile_output, "error": "Compilation failed"}
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "Compilation timed out"}
        except FileNotFoundError:
            return {"ok": False, "error": "iverilog not found - ensure Icarus Verilog is installed"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def simulate(self, binary: str | None = None) -> dict[str, Any]:
        """Run vvp on compiled binary."""
        try:
            if binary is None:
                binary = "a.out"

            binary_path = self.workspace_dir / binary
            if not binary_path.exists():
                return {"ok": False, "error": f"Binary '{binary}' not found. Run compile first."}

            result = subprocess.run(
                ["vvp", str(binary_path)],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(self.workspace_dir),
            )

            self.last_simulate_output = result.stdout + result.stderr

            # Check for pass/fail patterns
            stdout = self.last_simulate_output
            passed = (
                "ALL_TESTS_PASSED" in stdout
                or ("Mismatches: 0 " in stdout and "samples" in stdout)
                or ("no mismatches" in stdout.lower() and "errors" not in stdout.lower())
            )

            return {
                "ok": True,
                "output": self.last_simulate_output,
                "passed": passed,
            }
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "Simulation timed out"}
        except FileNotFoundError:
            return {"ok": False, "error": "vvp not found - ensure Icarus Verilog is installed"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def submit(self) -> dict[str, Any]:
        """Submit solution for grading."""
        self.submitted = True

        # Check if simulation passed
        if self.last_simulate_output:
            stdout = self.last_simulate_output
            self.passed = (
                "ALL_TESTS_PASSED" in stdout
                or ("Mismatches: 0 " in stdout and "samples" in stdout)
                or ("no mismatches" in stdout.lower() and "errors" not in stdout.lower())
            )
        else:
            self.passed = False

        return {
            "ok": True,
            "submitted": True,
            "passed": self.passed,
            "message": "Tests passed!" if self.passed else "Tests failed",
        }

    def cleanup(self):
        """Clean up the workspace directory."""
        try:
            shutil.rmtree(self.workspace_dir)
        except Exception:
            pass


def build_verilog_tools() -> list[dict[str, Any]]:
    """Build the tool schemas for Verilog operations."""
    return [
        {
            "type": "function",
            "function": {
                "name": TOOL_WRITE_FILE,
                "description": "Write content to a Verilog file in the workspace",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path (e.g., TopModule.v)"},
                        "content": {"type": "string", "description": "File content to write"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": TOOL_OPEN_FILE,
                "description": "Read content from a Verilog file in the workspace",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path (e.g., TopModule.v)"},
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": TOOL_COMPILE,
                "description": "Compile Verilog sources with iverilog",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sources": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of source files to compile (default: [TopModule.v])",
                        },
                        "testbench": {"type": "string", "description": "Testbench file (optional)"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": TOOL_SIMULATE,
                "description": "Run vvp simulation on compiled binary",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "binary": {"type": "string", "description": "Binary file to simulate (default: a.out)"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": TOOL_SUBMIT,
                "description": "Submit solution for final grading",
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
    ]


def get_system_message() -> str:
    """Get the system message for Verilog tasks."""
    return """You are an expert digital design engineer implementing Verilog spec-to-RTL tasks.

Tools available:
- write_file: Write content to a Verilog file
- open_file: Read content from a Verilog file
- compile: Compile sources with iverilog
- simulate: Run simulation with vvp
- submit: Submit solution for grading

Implement the TopModule according to the specification. Use the tools to write your implementation, compile, simulate to verify, and submit when ready."""


def format_user_message(problem_id: str, prompt: str, files: list[str]) -> str:
    """Format the user message for a Verilog task."""
    return f"""Problem: {problem_id}

Specification:
{prompt}

Available files: {', '.join(files)}

Please implement the Verilog module. Start by writing the TopModule.v file."""



