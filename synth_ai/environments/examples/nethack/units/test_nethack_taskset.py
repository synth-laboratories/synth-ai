"""Unit tests for NetHack taskset."""

import pytest
import asyncio

from synth_ai.environments.examples.nethack.taskset import (
    create_nethack_taskset,
    NetHackTaskInstance,
    NetHackTaskInstanceMetadata,
    CHARACTER_ROLES,
    SPECIAL_OBJECTIVES,
)


class TestNetHackTaskSet:
    """Test cases for NetHack taskset generation."""

    @pytest.mark.asyncio
    async def test_taskset_creation(self):
        """Test basic taskset creation."""
        taskset = await create_nethack_taskset()

        assert taskset.name == "NetHack TaskSet"
        assert len(taskset.instances) == 100  # Sum of all difficulty counts
        assert taskset.split_info._is_split_defined is True

    @pytest.mark.asyncio
    async def test_task_instance_properties(self):
        """Test properties of generated task instances."""
        taskset = await create_nethack_taskset()

        for instance in taskset.instances[:10]:  # Check first 10
            assert isinstance(instance, NetHackTaskInstance)
            assert isinstance(instance.metadata, NetHackTaskInstanceMetadata)

            # Check required fields
            assert instance.id is not None
            assert instance.impetus.instructions != ""
            assert instance.intent.rubric is not None
            assert instance.is_reproducible is True

            # Check metadata
            meta = instance.metadata
            assert meta.character_role in CHARACTER_ROLES
            assert meta.starting_level == 1
            assert meta.target_depth > 0
            assert meta.time_limit > 0
            assert meta.difficulty in [
                "tutorial",
                "beginner",
                "intermediate",
                "advanced",
                "expert",
            ]
            assert isinstance(meta.special_objectives, list)
            assert meta.seed is not None

    @pytest.mark.asyncio
    async def test_difficulty_distribution(self):
        """Test that difficulties are properly distributed."""
        taskset = await create_nethack_taskset()

        difficulty_counts = {
            "tutorial": 0,
            "beginner": 0,
            "intermediate": 0,
            "advanced": 0,
            "expert": 0,
        }

        for instance in taskset.instances:
            difficulty_counts[instance.metadata.difficulty] += 1

        assert difficulty_counts["tutorial"] == 20
        assert difficulty_counts["beginner"] == 30
        assert difficulty_counts["intermediate"] == 25
        assert difficulty_counts["advanced"] == 15
        assert difficulty_counts["expert"] == 10

    @pytest.mark.asyncio
    async def test_character_role_assignment(self):
        """Test character role assignment by difficulty."""
        taskset = await create_nethack_taskset()

        # Check tutorial only has tourist
        tutorial_instances = [i for i in taskset.instances if i.metadata.difficulty == "tutorial"]
        for inst in tutorial_instances:
            assert inst.metadata.character_role == "tourist"

        # Check expert has all roles
        expert_instances = [i for i in taskset.instances if i.metadata.difficulty == "expert"]
        expert_roles = set(inst.metadata.character_role for inst in expert_instances)
        assert len(expert_roles) > 1  # Should have multiple roles

    @pytest.mark.asyncio
    async def test_objective_assignment(self):
        """Test special objectives assignment."""
        taskset = await create_nethack_taskset()

        # Check objectives are from valid categories
        all_valid_objectives = []
        for category in SPECIAL_OBJECTIVES.values():
            all_valid_objectives.extend(category)

        for instance in taskset.instances:
            for obj in instance.metadata.special_objectives:
                assert obj in all_valid_objectives

        # Check objective count by difficulty
        tutorial_inst = next(i for i in taskset.instances if i.metadata.difficulty == "tutorial")
        assert len(tutorial_inst.metadata.special_objectives) == 1

        expert_inst = next(i for i in taskset.instances if i.metadata.difficulty == "expert")
        assert len(expert_inst.metadata.special_objectives) == 4

    @pytest.mark.asyncio
    async def test_instruction_content(self):
        """Test that instructions contain necessary information."""
        taskset = await create_nethack_taskset()

        for instance in taskset.instances[:5]:  # Check first 5
            instructions = instance.impetus.instructions

            # Check key elements are present
            assert instance.metadata.character_role in instructions
            assert str(instance.metadata.target_depth) in instructions
            assert str(instance.metadata.time_limit) in instructions
            assert "Additional objectives:" in instructions
            assert "Character strengths:" in instructions
            assert "Character weaknesses:" in instructions
            assert "Tips:" in instructions

    @pytest.mark.asyncio
    async def test_rubric_structure(self):
        """Test intent rubric structure."""
        taskset = await create_nethack_taskset()

        for instance in taskset.instances[:5]:
            rubric = instance.intent.rubric

            assert "goal" in rubric
            assert "success_criteria" in rubric
            assert "evaluation_metrics" in rubric

            # Check success criteria
            assert "primary" in rubric["success_criteria"]
            assert "secondary" in rubric["success_criteria"]

            # Check evaluation metrics
            metrics = rubric["evaluation_metrics"]
            assert metrics["depth_reached"] == instance.metadata.target_depth
            assert metrics["time_limit"] == instance.metadata.time_limit
            assert metrics["objectives_completed"] == len(instance.metadata.special_objectives)

    @pytest.mark.asyncio
    async def test_split_info(self):
        """Test train/val/test split."""
        taskset = await create_nethack_taskset()

        total_instances = len(taskset.instances)
        val_size = len(taskset.split_info.val_instance_ids)
        test_size = len(taskset.split_info.test_instance_ids)

        # Check split sizes (should be ~10% each)
        assert val_size == total_instances // 10
        assert test_size == total_instances // 10

        # Check no overlap
        assert len(taskset.split_info.val_instance_ids & taskset.split_info.test_instance_ids) == 0

        # Check all split IDs are valid
        all_ids = {inst.id for inst in taskset.instances}
        assert taskset.split_info.val_instance_ids.issubset(all_ids)
        assert taskset.split_info.test_instance_ids.issubset(all_ids)

    @pytest.mark.asyncio
    async def test_task_serialization(self):
        """Test task instance serialization."""
        taskset = await create_nethack_taskset()
        instance = taskset.instances[0]

        # Serialize
        serialized = await instance.serialize()

        assert isinstance(serialized, dict)
        assert "id" in serialized
        assert "impetus" in serialized
        assert "intent" in serialized
        assert "metadata" in serialized

        # Check metadata fields
        meta = serialized["metadata"]
        assert meta["character_role"] == instance.metadata.character_role
        assert meta["target_depth"] == instance.metadata.target_depth
        assert meta["time_limit"] == instance.metadata.time_limit

        # Deserialize
        restored = await NetHackTaskInstance.deserialize(serialized)

        assert restored.metadata.character_role == instance.metadata.character_role
        assert restored.metadata.target_depth == instance.metadata.target_depth
        assert restored.metadata.time_limit == instance.metadata.time_limit
        assert restored.metadata.special_objectives == instance.metadata.special_objectives

    @pytest.mark.asyncio
    async def test_reproducibility(self):
        """Test that tasks are marked as reproducible."""
        taskset = await create_nethack_taskset()

        for instance in taskset.instances:
            assert instance.is_reproducible is True
            assert instance.metadata.seed is not None
            assert 0 <= instance.metadata.seed < 2**31
