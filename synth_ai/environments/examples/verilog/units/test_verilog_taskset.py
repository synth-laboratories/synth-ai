import pytest
import asyncio
from pathlib import Path
from unittest.mock import patch

# Add timeout to all async tests
pytestmark = pytest.mark.timeout(15)

from synth_ai.environments.examples.verilog.taskset import (
    create_verilog_taskset,
    _create_hf_task_instance,
    VerilogTaskInstance,
    VerilogTaskInstanceMetadata,
    _cleanup_temp_dirs,
    _temp_dirs,
)
from synth_ai.environments.tasks.core import TaskInstanceSet, SplitInfo, Impetus, Intent
from uuid import uuid4
from typing import cast


class TestVerilogTaskset:
    """Test suite for Verilog taskset creation."""

    @pytest.mark.asyncio
    @patch("src.examples.verilog.taskset.load_dataset")
    async def test_create_verilog_taskset_basic(self, mock_load_dataset):
        """Test basic taskset creation."""
        # Mock dataset
        mock_dataset = [
            {
                "problem_id": "test_001",
                "prompt": "Implement a simple AND gate with inputs a, b and output y.",
                "test": "`timescale 1ns/1ps\nmodule test_tb;\n  // testbench code\nendmodule",
                "ref": "module RefModule(input a, b, output y);\n  assign y = a & b;\nendmodule",
            },
            {
                "problem_id": "test_002",
                "prompt": "Implement a simple OR gate with inputs a, b and output y.",
                "test": "`timescale 1ns/1ps\nmodule test_tb2;\n  // testbench code\nendmodule",
                "ref": "module RefModule(input a, b, output y);\n  assign y = a | b;\nendmodule",
            },
        ]
        mock_load_dataset.return_value = mock_dataset

        taskset = await create_verilog_taskset(max_instances=2)

        assert isinstance(taskset, TaskInstanceSet)
        assert taskset.name == "VerilogEval v2 TaskSet"
        assert taskset.description == "VerilogEval v2 spec-to-RTL tasks from HuggingFace"
        assert len(taskset.instances) == 2

        # Check split info
        assert isinstance(taskset.split_info, SplitInfo)
        assert taskset.split_info._is_split_defined is True

        # Check instance properties
        instance = taskset.instances[0]
        assert isinstance(instance, VerilogTaskInstance)
        metadata = cast(VerilogTaskInstanceMetadata, instance.metadata)
        assert metadata.problem_name == "test_001"
        assert "AND gate" in metadata.description
        assert len(metadata.files_provided) == 3  # TopModule.v, testbench, RefModule.v

    @pytest.mark.asyncio
    @patch("src.examples.verilog.taskset.load_dataset")
    async def test_create_verilog_taskset_max_instances(self, mock_load_dataset):
        """Test taskset creation with max_instances limit."""
        # Mock larger dataset
        mock_dataset = [
            {
                "problem_id": f"test_{i:03d}",
                "prompt": f"Test {i}",
                "test": "",
                "ref": "",
            }
            for i in range(20)
        ]
        mock_load_dataset.return_value = mock_dataset

        taskset = await create_verilog_taskset(max_instances=5)

        assert len(taskset.instances) == 5
        # Should only create instances for first 5 items
        metadata0 = cast(VerilogTaskInstanceMetadata, taskset.instances[0].metadata)
        metadata4 = cast(VerilogTaskInstanceMetadata, taskset.instances[4].metadata)
        assert metadata0.problem_name == "test_000"
        assert metadata4.problem_name == "test_004"

    @pytest.mark.asyncio
    @patch("src.examples.verilog.taskset.load_dataset")
    async def test_create_verilog_taskset_split_info(self, mock_load_dataset):
        """Test that split info is correctly calculated."""
        mock_dataset = [
            {
                "problem_id": f"test_{i:03d}",
                "prompt": f"Test {i}",
                "test": "",
                "ref": "",
            }
            for i in range(10)
        ]
        mock_load_dataset.return_value = mock_dataset

        taskset = await create_verilog_taskset(max_instances=10)

        # Should have 80% val (8 instances) and 20% test (2 instances)
        assert len(taskset.split_info.val_instance_ids) == 8
        assert len(taskset.split_info.test_instance_ids) == 2

        # Check that all instance IDs are accounted for
        all_ids = set(inst.id for inst in taskset.instances)
        split_ids = taskset.split_info.val_instance_ids | taskset.split_info.test_instance_ids
        assert all_ids == split_ids

    def test_create_hf_task_instance(self):
        """Test creation of task instance from HuggingFace dataset item."""
        item = {
            "problem_id": "Prob001_zero",
            "prompt": "I would like you to implement a module named TopModule with output zero that always outputs LOW.",
            "test": "`timescale 1 ps/1 ps\nmodule tb();\n  // testbench\nendmodule",
            "ref": "module RefModule(output zero);\n  assign zero = 1'b0;\nendmodule",
        }

        instance = _create_hf_task_instance(item, 0)

        assert isinstance(instance, VerilogTaskInstance)
        metadata = cast(VerilogTaskInstanceMetadata, instance.metadata)
        assert metadata.problem_name == "Prob001_zero"
        assert "TopModule" in instance.impetus.instructions
        assert "always outputs LOW" in metadata.description
        assert metadata.difficulty == "medium"
        assert len(metadata.files_provided) == 3

        # Check that files were created
        pristine_dir = Path(instance.pristine_dir)
        assert (pristine_dir / "TopModule.v").exists()
        assert (pristine_dir / "Prob001_zero_tb.v").exists()
        assert (pristine_dir / "RefModule.v").exists()

        # Check file contents
        topmodule_content = (pristine_dir / "TopModule.v").read_text()
        assert "module TopModule();" in topmodule_content
        assert "TODO: Implement" in topmodule_content
        assert "always outputs LOW" in topmodule_content

        ref_content = (pristine_dir / "RefModule.v").read_text()
        assert "module RefModule" in ref_content
        assert "assign zero = 1'b0" in ref_content

    @pytest.mark.asyncio
    async def test_task_instance_serialization(self):
        """Test task instance serialization and deserialization."""
        item = {
            "problem_id": "test_serial",
            "prompt": "Test serialization",
            "test": "module test_tb(); endmodule",
            "ref": "module RefModule(); endmodule",
        }

        instance = _create_hf_task_instance(item, 0)

        # Test serialization
        serialized = await instance.serialize()
        assert isinstance(serialized, dict)
        assert serialized["metadata"]["problem_name"] == "test_serial"
        assert "id" in serialized
        assert isinstance(serialized["id"], str)  # UUID should be converted to string

        # Test deserialization
        deserialized = await VerilogTaskInstance.deserialize(serialized)
        assert isinstance(deserialized, VerilogTaskInstance)
        deserialized_metadata = cast(VerilogTaskInstanceMetadata, deserialized.metadata)
        instance_metadata = cast(VerilogTaskInstanceMetadata, instance.metadata)
        assert deserialized_metadata.problem_name == instance_metadata.problem_name
        assert deserialized.impetus.instructions == instance.impetus.instructions


class TestVerilogTaskInstanceMetadata:
    """Test suite for VerilogTaskInstanceMetadata."""

    def test_metadata_creation(self):
        """Test metadata creation with all fields."""
        metadata = VerilogTaskInstanceMetadata(
            problem_name="test_problem",
            difficulty="hard",
            description="A test problem for unit testing",
            files_provided=["TopModule.v", "test_tb.v", "RefModule.v"],
        )

        assert metadata.problem_name == "test_problem"
        assert metadata.difficulty == "hard"
        assert metadata.description == "A test problem for unit testing"
        assert len(metadata.files_provided) == 3
        assert "TopModule.v" in metadata.files_provided


class TestVerilogTaskInstance:
    """Test suite for VerilogTaskInstance class."""

    def test_task_instance_creation(self):
        """Test basic task instance creation."""
        metadata = VerilogTaskInstanceMetadata(
            problem_name="test",
            difficulty="easy",
            description="Test description",
            files_provided=["test.v"],
        )

        instance = VerilogTaskInstance(
            id=uuid4(),
            impetus=Impetus(instructions="Test instructions"),
            intent=Intent(rubric={"goal": "Test goal"}, gold_trajectories=None, gold_state_diff={}),
            metadata=metadata,
            is_reproducible=True,
            initial_engine_snapshot=None,
            pristine_dir="/tmp/pristine",
            snapshot_dir="/tmp/snapshot",
        )

        metadata_check = cast(VerilogTaskInstanceMetadata, instance.metadata)
        assert metadata_check.problem_name == "test"
        assert instance.impetus.instructions == "Test instructions"
        assert instance.intent.rubric == "Test goal"
        assert instance.pristine_dir == "/tmp/pristine"
        assert instance.snapshot_dir == "/tmp/snapshot"

    @pytest.mark.asyncio
    async def test_serialization_with_uuid(self):
        """Test serialization properly handles UUID conversion."""

        metadata = VerilogTaskInstanceMetadata(
            problem_name="test",
            difficulty="easy",
            description="Test",
            files_provided=["test.v"],
        )

        instance = VerilogTaskInstance(
            id=uuid4(),
            impetus=Impetus(instructions="Test"),
            intent=Intent(rubric={"goal": "Test"}, gold_trajectories=None, gold_state_diff={}),
            metadata=metadata,
            is_reproducible=True,
            initial_engine_snapshot=None,
        )

        serialized = await instance.serialize()
        assert isinstance(serialized["id"], str)

        # Test deserialization can handle string ID
        deserialized = await VerilogTaskInstance.deserialize(serialized)
        assert deserialized is not None

    @pytest.mark.asyncio
    async def test_deserialization_graceful_id_handling(self):
        """Test deserialization gracefully handles various ID formats."""
        metadata = VerilogTaskInstanceMetadata(
            problem_name="test",
            difficulty="easy",
            description="Test",
            files_provided=["test.v"],
        )

        # Test with string ID
        data = {
            "id": "some-string-id",
            "impetus": {"instructions": "Test"},
            "intent": {"rubric": {"goal": "Test"}, "deterministic_eval_functions": []},
            "metadata": {
                "problem_name": "test",
                "difficulty": "easy",
                "description": "Test",
                "files_provided": ["test.v"],
            },
        }

        instance = await VerilogTaskInstance.deserialize(data)
        metadata_check = cast(VerilogTaskInstanceMetadata, instance.metadata)
        assert metadata_check.problem_name == "test"

    @pytest.mark.asyncio
    async def test_deserialization_filters_constructor_fields(self):
        """Test deserialization only uses valid constructor fields."""
        data = {
            "id": "test-id",
            "impetus": {"instructions": "Test"},
            "intent": {"rubric": {"goal": "Test"}, "deterministic_eval_functions": []},
            "metadata": {
                "problem_name": "test",
                "difficulty": "easy",
                "description": "Test",
                "files_provided": ["test.v"],
            },
            "extra_field": "should_be_ignored",
            "another_extra": 123,
        }

        instance = await VerilogTaskInstance.deserialize(data)
        metadata_check = cast(VerilogTaskInstanceMetadata, instance.metadata)
        assert metadata_check.problem_name == "test"
        # Extra fields should be filtered out and not cause errors


class TestTempDirectoryCleanup:
    """Test suite for temporary directory cleanup functionality."""

    def test_temp_dirs_tracking(self):
        """Test that temporary directories are tracked."""
        initial_count = len(_temp_dirs)

        item = {
            "problem_id": "cleanup_test",
            "prompt": "Test cleanup",
            "test": "module test(); endmodule",
            "ref": "module ref(); endmodule",
        }

        instance = _create_hf_task_instance(item, 0)

        # Should have added 2 directories (pristine and snapshot)
        assert len(_temp_dirs) == initial_count + 2

        # Verify directories exist
        pristine_dir = Path(instance.pristine_dir)
        snapshot_dir = Path(instance.snapshot_dir)
        assert pristine_dir.exists()
        assert snapshot_dir.exists()

    def test_cleanup_temp_dirs(self):
        """Test manual cleanup of temporary directories."""
        # Create some temp directories through task creation
        item = {
            "problem_id": "cleanup_test2",
            "prompt": "Test cleanup",
            "test": "module test(); endmodule",
            "ref": "module ref(); endmodule",
        }

        instance = _create_hf_task_instance(item, 0)
        pristine_dir = Path(instance.pristine_dir)
        snapshot_dir = Path(instance.snapshot_dir)

        # Verify they exist
        assert pristine_dir.exists()
        assert snapshot_dir.exists()

        # Clean up
        _cleanup_temp_dirs()

        # Verify they're removed
        assert not pristine_dir.exists()
        assert not snapshot_dir.exists()
        assert len(_temp_dirs) == 0


class TestTasksetIntegration:
    """Integration tests for the complete taskset workflow."""

    @pytest.mark.asyncio
    @patch("src.examples.verilog.taskset.load_dataset")
    async def test_full_taskset_workflow(self, mock_load_dataset):
        """Test complete workflow from dataset loading to task creation."""
        # Mock realistic VerilogEval dataset items
        mock_dataset = [
            {
                "problem_id": "Prob001_zero",
                "prompt": "I would like you to implement a module named TopModule with the following interface. All input and output ports are one bit unless otherwise specified.\n\n - output zero\n\nThe module should always outputs a LOW.",
                "test": '`timescale 1 ps/1 ps\n`define OK 12\n`define INCORRECT 13\n\nmodule stimulus_gen (\n\tinput clk,\n\toutput reg[511:0] wavedrom_title,\n\toutput reg wavedrom_enable\n);\n\ntask wavedrom_start(input[511:0] title = "");\nendtask\n\nendmodule\n\nmodule tb();\n\nreg clk=0;\ninitial forever\n\t#5 clk = ~clk;\n\nlogic zero_ref;\nlogic zero_dut;\n\nRefModule good1 (\n\t.zero(zero_ref) );\n\t\nTopModule top_module1 (\n\t.zero(zero_dut) );\n\nendmodule',
                "ref": "module RefModule (\n  output zero\n);\n\n  assign zero = 1'b0;\n\nendmodule",
            },
            {
                "problem_id": "Prob002_and_gate",
                "prompt": "Implement an AND gate with inputs a, b and output y.",
                "test": "`timescale 1ns/1ps\nmodule test_tb;\n  reg a, b;\n  wire y;\n  TopModule dut(.a(a), .b(b), .y(y));\n  RefModule ref(.a(a), .b(b), .y(y_ref));\nendmodule",
                "ref": "module RefModule(input a, b, output y);\n  assign y = a & b;\nendmodule",
            },
        ]
        mock_load_dataset.return_value = mock_dataset

        # Create taskset
        taskset = await create_verilog_taskset(max_instances=2)

        # Verify taskset structure
        assert len(taskset.instances) == 2
        assert len(taskset.split_info.val_instance_ids) == 1  # 80% of 2 = 1.6 -> 1
        assert len(taskset.split_info.test_instance_ids) == 1  # 20% of 2 = 0.4 -> 1

        # Verify first instance (zero module)
        zero_instance = taskset.instances[0]
        zero_metadata = cast(VerilogTaskInstanceMetadata, zero_instance.metadata)
        assert zero_metadata.problem_name == "Prob001_zero"
        assert "output zero" in zero_instance.impetus.instructions
        assert "always outputs a LOW" in zero_metadata.description

        # Check files were created properly
        pristine_dir = Path(zero_instance.pristine_dir)
        assert (pristine_dir / "TopModule.v").exists()
        assert (pristine_dir / "Prob001_zero_tb.v").exists()
        assert (pristine_dir / "RefModule.v").exists()

        # Verify TopModule template
        topmodule_content = (pristine_dir / "TopModule.v").read_text()
        assert "module TopModule();" in topmodule_content
        assert "TODO: Implement" in topmodule_content
        assert "output zero" in topmodule_content

        # Verify RefModule content
        ref_content = (pristine_dir / "RefModule.v").read_text()
        assert "module RefModule" in ref_content
        assert "assign zero = 1'b0" in ref_content

        # Verify second instance (AND gate)
        and_instance = taskset.instances[1]
        and_metadata = cast(VerilogTaskInstanceMetadata, and_instance.metadata)
        assert and_metadata.problem_name == "Prob002_and_gate"
        assert "AND gate" in and_instance.impetus.instructions

        # Test serialization of entire taskset
        serialized_instances = await asyncio.gather(
            *(inst.serialize() for inst in taskset.instances)
        )
        assert len(serialized_instances) == 2
        assert all(isinstance(s, dict) for s in serialized_instances)

    @pytest.mark.asyncio
    @patch("src.examples.verilog.taskset.load_dataset")
    async def test_empty_dataset_handling(self, mock_load_dataset):
        """Test handling of empty dataset."""
        mock_load_dataset.return_value = []

        taskset = await create_verilog_taskset(max_instances=5)

        assert len(taskset.instances) == 0
        assert len(taskset.split_info.val_instance_ids) == 0
        assert len(taskset.split_info.test_instance_ids) == 0

    @pytest.mark.asyncio
    @patch("src.examples.verilog.taskset.load_dataset")
    async def test_single_instance_split(self, mock_load_dataset):
        """Test split calculation with single instance."""
        mock_dataset = [
            {
                "problem_id": "single_test",
                "prompt": "Single test",
                "test": "module test(); endmodule",
                "ref": "module ref(); endmodule",
            }
        ]
        mock_load_dataset.return_value = mock_dataset

        taskset = await create_verilog_taskset(max_instances=1)

        # With 1 instance: 80% = 0.8 -> 0, 20% = 0.2 -> 0
        # But we need at least one instance somewhere, so it should go to val
        assert len(taskset.instances) == 1
        # The split calculation should handle edge cases gracefully
