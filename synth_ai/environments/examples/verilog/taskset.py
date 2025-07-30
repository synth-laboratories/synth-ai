from synth_ai.environments.tasks.core import (
    Task,
    TaskInstance,
    TaskInstanceMetadata,
    TaskInstanceSet,
    SplitInfo,
    Impetus,
    Intent,
)
from uuid import uuid4, UUID
from dataclasses import dataclass, asdict, fields
from typing import Optional
from pathlib import Path
import tempfile
import os
import shutil
import atexit
from datasets import load_dataset

# Global list to track temp directories for cleanup
_temp_dirs = []


def _cleanup_temp_dirs():
    """Clean up all temporary directories created during task instances."""
    for temp_dir in _temp_dirs:
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception:
            pass  # Ignore cleanup errors
    _temp_dirs.clear()


# Register cleanup function to run at exit
atexit.register(_cleanup_temp_dirs)

verilog_task = Task(
    global_premises="Implement and verify Verilog hardware designs",
    global_constraints="Must pass testbench verification",
    global_objectives="Write correct Verilog code that passes all tests",
    shared_env_params={},
)


@dataclass
class VerilogTaskInstanceMetadata(TaskInstanceMetadata):
    problem_name: str
    difficulty: str
    description: str
    files_provided: list[str]


@dataclass
class VerilogTaskInstance(TaskInstance):
    pristine_dir: Optional[str] = None
    snapshot_dir: Optional[str] = None

    async def serialize(self) -> dict:
        data = asdict(self)
        if "id" in data and isinstance(data["id"], UUID):
            data["id"] = str(data["id"])
        if "intent" in data and data["intent"] is not None:
            if "deterministic_eval_functions" in data["intent"]:
                data["intent"]["deterministic_eval_functions"] = []
        return data

    @classmethod
    async def deserialize(cls, data: dict) -> "VerilogTaskInstance":
        """Gracefully accept non-UUID ids and rebuild required objects."""
        if "id" in data:
            try:
                data["id"] = UUID(str(data["id"]))
            except (ValueError, TypeError, AttributeError):
                pass  # keep original string

        if "impetus" in data and isinstance(data["impetus"], dict):
            impetus_data = data["impetus"]
            # Ensure instructions field exists with default if missing
            if "instructions" not in impetus_data:
                impetus_data["instructions"] = "Implement the Verilog module"
            data["impetus"] = Impetus(**impetus_data)

        if "intent" in data and isinstance(data["intent"], dict):
            intent_data = data["intent"]
            if "deterministic_eval_functions" not in intent_data:
                intent_data["deterministic_eval_functions"] = []
            # Provide default values for required fields if missing
            if "rubric" not in intent_data:
                intent_data["rubric"] = {"goal": "Pass all testbench tests"}
            if "gold_trajectories" not in intent_data:
                intent_data["gold_trajectories"] = None
            if "gold_state_diff" not in intent_data:
                intent_data["gold_state_diff"] = {}
            data["intent"] = Intent(**intent_data)

        if "metadata" in data and isinstance(data["metadata"], dict):
            metadata_data = data["metadata"]
            # Ensure required fields exist with defaults if missing
            if "problem_name" not in metadata_data:
                metadata_data["problem_name"] = "unknown"
            if "difficulty" not in metadata_data:
                metadata_data["difficulty"] = "medium"
            if "description" not in metadata_data:
                metadata_data["description"] = "Verilog implementation task"
            if "files_provided" not in metadata_data:
                metadata_data["files_provided"] = []
            data["metadata"] = VerilogTaskInstanceMetadata(**metadata_data)

        constructor_field_names = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in constructor_field_names}

        # Add default values for required fields if missing
        if "is_reproducible" not in filtered_data:
            filtered_data["is_reproducible"] = True
        if "initial_engine_snapshot" not in filtered_data:
            filtered_data["initial_engine_snapshot"] = None

        return cls(**filtered_data)


async def create_verilog_taskset(max_instances: int = 10) -> TaskInstanceSet:
    """Create a Verilog task set from HuggingFace VerilogEval v2 dataset."""
    # Load VerilogEval v2 dataset from HuggingFace
    ds = load_dataset("dakies/nvlabs-verilogeval-v2-spec-to-rtl", split="test")

    instances = []

    # Limit the number of instances for faster testing
    dataset_size = min(max_instances, len(ds))  # type: ignore[arg-type]

    # Convert each dataset item to VerilogTaskInstance
    for i in range(dataset_size):
        item = ds[i]
        instance = _create_hf_task_instance(item, i)
        instances.append(instance)

    # Create split info - use first 80% for validation, last 20% for test
    total_instances = len(instances)
    val_split = int(0.8 * total_instances)

    val_ids = {inst.id for inst in instances[:val_split]}
    test_ids = {inst.id for inst in instances[val_split:]}

    split_info = SplitInfo(
        val_instance_ids=val_ids,
        test_instance_ids=test_ids,
        _is_split_defined=True,
    )

    return TaskInstanceSet(
        name="VerilogEval v2 TaskSet",
        description="VerilogEval v2 spec-to-RTL tasks from HuggingFace",
        instances=instances,
        split_info=split_info,
    )


def _create_hf_task_instance(item, index: int) -> VerilogTaskInstance:
    """Create a VerilogTaskInstance from a HuggingFace dataset item."""
    instance_id = uuid4()

    # Create temporary directory for this task
    temp_dir = tempfile.mkdtemp(prefix=f"verilog_hf_{index}_{instance_id}_")
    _temp_dirs.append(temp_dir)  # Track for cleanup
    pristine_dir = Path(temp_dir)
    pristine_dir.mkdir(exist_ok=True)

    # Extract information from dataset item
    problem_id = item["problem_id"]
    prompt = item["prompt"]
    testbench = item["test"]
    ref_solution = item["ref"]

    # Create incomplete module template (TopModule is the expected name in tests)
    module_content = (
        """module TopModule();
    // TODO: Implement the module based on the specification below
    /*
    Specification:
    """
        + prompt.strip()
        + """
    */
endmodule"""
    )

    # Write files to pristine directory
    module_file = "TopModule.v"
    testbench_file = f"{problem_id}_tb.v"
    ref_file = "RefModule.v"

    (pristine_dir / module_file).write_text(module_content)
    (pristine_dir / testbench_file).write_text(testbench)
    (pristine_dir / ref_file).write_text(ref_solution)  # Include reference module

    files_provided = [module_file, testbench_file, ref_file]

    # Create task components
    impetus = Impetus(
        instructions=f"Problem: {problem_id}\n\n{prompt.strip()}\n\nImplement the TopModule according to the specification. The testbench will verify your implementation."
    )

    intent = Intent(
        rubric={
            "goal": f"Implement correct TopModule for {problem_id} that passes testbench verification"
        },
        gold_trajectories=None,
        gold_state_diff={},
    )

    metadata = VerilogTaskInstanceMetadata(
        problem_name=problem_id,
        difficulty="medium",  # VerilogEval doesn't specify difficulty levels
        description=prompt.strip(),  # Full description
        files_provided=files_provided,
    )

    # Create snapshot directory and track for cleanup
    snapshot_dir = tempfile.mkdtemp(prefix=f"verilog_snapshot_{instance_id}_")
    _temp_dirs.append(snapshot_dir)

    return VerilogTaskInstance(
        id=instance_id,
        impetus=impetus,
        intent=intent,
        metadata=metadata,
        is_reproducible=True,
        initial_engine_snapshot=None,
        pristine_dir=str(pristine_dir),
        snapshot_dir=snapshot_dir,
    )


def _create_adder_task() -> VerilogTaskInstance:
    """Create a simple 4-bit adder task."""
    instance_id = uuid4()

    # Create temporary directory for this task
    temp_dir = tempfile.mkdtemp(prefix=f"verilog_adder_{instance_id}_")
    _temp_dirs.append(temp_dir)  # Track for cleanup

    # Write adder testbench
    adder_tb_content = """`timescale 1ns/1ps
module adder4_tb;
    reg [3:0] a, b;
    wire [4:0] sum;
    
    adder4 dut(.a(a), .b(b), .sum(sum));
    
    initial begin
        a = 4'b0000; b = 4'b0000; #10;
        if (sum != 5'b00000) $fatal(1, "Test failed: 0 + 0 != 0");
        
        a = 4'b0001; b = 4'b0001; #10;
        if (sum != 5'b00010) $fatal(1, "Test failed: 1 + 1 != 2");
        
        a = 4'b1111; b = 4'b0001; #10;
        if (sum != 5'b10000) $fatal(1, "Test failed: 15 + 1 != 16");
        
        $display("ALL_TESTS_PASSED");
        $finish;
    end
endmodule"""

    # Write incomplete adder module (for student to complete)
    adder_content = """module adder4(
    input [3:0] a,
    input [3:0] b,
    output [4:0] sum
);
    // TODO: Implement 4-bit adder
    // assign sum = ?;
endmodule"""

    pristine_dir = Path(temp_dir)
    pristine_dir.mkdir(exist_ok=True)

    (pristine_dir / "adder4_tb.v").write_text(adder_tb_content)
    (pristine_dir / "adder4.v").write_text(adder_content)

    impetus = Impetus(
        instructions="Implement a 4-bit adder module that takes two 4-bit inputs 'a' and 'b' and produces a 5-bit output 'sum'."
    )

    intent = Intent(
        rubric="Implement correct 4-bit adder that passes testbench",
        gold_trajectories=None,
        gold_state_diff={},
    )

    metadata = VerilogTaskInstanceMetadata(
        problem_name="adder4",
        difficulty="easy",
        description="4-bit adder implementation",
        files_provided=["adder4.v", "adder4_tb.v"],
    )

    return VerilogTaskInstance(
        id=instance_id,
        impetus=impetus,
        intent=intent,
        metadata=metadata,
        is_reproducible=True,
        initial_engine_snapshot=None,
        pristine_dir=str(pristine_dir),
        snapshot_dir=(
            lambda: (
                _temp_dirs.append(d := tempfile.mkdtemp(prefix=f"verilog_snapshot_{instance_id}_")),
                d,
            )[1]
        )(),
    )


def _create_and_gate_task() -> VerilogTaskInstance:
    """Create a simple AND gate task."""
    instance_id = uuid4()

    # Create temporary directory for this task
    temp_dir = tempfile.mkdtemp(prefix=f"verilog_and_{instance_id}_")
    _temp_dirs.append(temp_dir)  # Track for cleanup

    # Write AND gate testbench
    and_tb_content = """`timescale 1ns/1ps
module and_gate_tb;
    reg a, b;
    wire y;
    
    and_gate dut(.a(a), .b(b), .y(y));
    
    initial begin
        a = 0; b = 0; #10;
        if (y != 0) $fatal(1, "Test failed: 0 AND 0 != 0");
        
        a = 0; b = 1; #10;
        if (y != 0) $fatal(1, "Test failed: 0 AND 1 != 0");
        
        a = 1; b = 0; #10;
        if (y != 0) $fatal(1, "Test failed: 1 AND 0 != 0");
        
        a = 1; b = 1; #10;
        if (y != 1) $fatal(1, "Test failed: 1 AND 1 != 1");
        
        $display("ALL_TESTS_PASSED");
        $finish;
    end
endmodule"""

    # Write incomplete AND gate module
    and_content = """module and_gate(
    input a,
    input b,
    output y
);
    // TODO: Implement AND gate
    // assign y = ?;
endmodule"""

    pristine_dir = Path(temp_dir)
    pristine_dir.mkdir(exist_ok=True)

    (pristine_dir / "and_gate_tb.v").write_text(and_tb_content)
    (pristine_dir / "and_gate.v").write_text(and_content)

    impetus = Impetus(
        instructions="Implement an AND gate module that takes two inputs 'a' and 'b' and produces output 'y'."
    )

    intent = Intent(
        rubric="Implement correct AND gate that passes testbench",
        gold_trajectories=None,
        gold_state_diff={},
    )

    metadata = VerilogTaskInstanceMetadata(
        problem_name="and_gate",
        difficulty="easy",
        description="Basic AND gate implementation",
        files_provided=["and_gate.v", "and_gate_tb.v"],
    )

    return VerilogTaskInstance(
        id=instance_id,
        impetus=impetus,
        intent=intent,
        metadata=metadata,
        is_reproducible=True,
        initial_engine_snapshot=None,
        pristine_dir=str(pristine_dir),
        snapshot_dir=(
            lambda: (
                _temp_dirs.append(d := tempfile.mkdtemp(prefix=f"verilog_snapshot_{instance_id}_")),
                d,
            )[1]
        )(),
    )


# Example usage
if __name__ == "__main__":
    import asyncio

    async def main():
        taskset = await create_verilog_taskset()

        serialized = await asyncio.gather(*(inst.serialize() for inst in taskset.instances))

        print(f"Created {len(serialized)} Verilog task instances")

        # Print summary
        for i, inst in enumerate(taskset.instances):
            print(f"Task {i + 1}: {inst.metadata.problem_name} ({inst.metadata.difficulty})")
            print(f"  Description: {inst.metadata.description}")
            print(f"  Files: {inst.metadata.files_provided}")
            print()

    asyncio.run(main())
