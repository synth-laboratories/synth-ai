"""Unit tests for MiniGrid taskset."""

import asyncio
import pytest
from uuid import UUID

from synth_ai.environments.examples.minigrid.taskset import (
    MiniGridTaskInstance,
    MiniGridTaskInstanceMetadata,
    create_minigrid_taskset,
    DEFAULT_MINIGRID_TASK,
    ENVIRONMENTS,
)


@pytest.mark.asyncio
async def test_default_task():
    """Test the default MiniGrid task."""
    task = DEFAULT_MINIGRID_TASK

    # Check task properties
    assert isinstance(task.id, UUID)
    assert task.impetus.instructions == "Navigate the 5x5 grid to reach the goal marked with 'G'."
    assert (
        task.intent.rubric["goal"]
        == "Successfully reach the goal tile in the MiniGrid-Empty-5x5-v0 environment."
    )
    assert task.metadata.env_name == "MiniGrid-Empty-5x5-v0"
    assert task.metadata.grid_size == (5, 5)
    assert task.metadata.difficulty == "easy"
    assert task.metadata.seed == 42
    assert task.is_reproducible is True


@pytest.mark.asyncio
async def test_task_serialization():
    """Test task instance serialization and deserialization."""
    task = DEFAULT_MINIGRID_TASK

    # Serialize
    serialized = await task.serialize()

    # Check serialized data
    assert "id" in serialized
    assert "impetus" in serialized
    assert "intent" in serialized
    assert "metadata" in serialized
    assert serialized["metadata"]["env_name"] == "MiniGrid-Empty-5x5-v0"
    assert serialized["metadata"]["grid_size"] == [5, 5]

    # Deserialize
    deserialized = await MiniGridTaskInstance.deserialize(serialized)

    # Check deserialized task
    assert deserialized.impetus.instructions == task.impetus.instructions
    assert deserialized.metadata.env_name == task.metadata.env_name
    assert deserialized.metadata.grid_size == task.metadata.grid_size


@pytest.mark.asyncio
async def test_create_taskset():
    """Test taskset creation."""
    taskset = await create_minigrid_taskset(
        num_tasks_per_difficulty={"easy": 5, "medium": 3, "hard": 2}, seed=42
    )

    # Check taskset properties
    assert taskset.name == "MiniGrid TaskSet"
    assert len(taskset.instances) == 10  # 5 + 3 + 2

    # Check splits
    assert taskset.split_info._is_split_defined
    assert len(taskset.split_info.val_instance_ids) >= 1
    assert len(taskset.split_info.test_instance_ids) >= 1

    # Check no overlap between splits
    assert taskset.split_info.val_instance_ids.isdisjoint(taskset.split_info.test_instance_ids)

    # Check all instances are valid
    for instance in taskset.instances:
        assert isinstance(instance, MiniGridTaskInstance)
        assert instance.metadata.env_name in [
            env[0] for envs in ENVIRONMENTS.values() for env in envs
        ]


@pytest.mark.asyncio
async def test_task_metadata():
    """Test task metadata properties."""
    taskset = await create_minigrid_taskset(
        num_tasks_per_difficulty={"easy": 2, "medium": 2, "hard": 2}, seed=123
    )

    easy_tasks = [t for t in taskset.instances if t.metadata.difficulty == "easy"]
    medium_tasks = [t for t in taskset.instances if t.metadata.difficulty == "medium"]
    hard_tasks = [t for t in taskset.instances if t.metadata.difficulty == "hard"]

    # Check counts
    assert len(easy_tasks) == 2
    assert len(medium_tasks) == 2
    assert len(hard_tasks) == 2

    # Check metadata properties
    for task in medium_tasks:
        if "DoorKey" in task.metadata.env_name:
            assert task.metadata.has_key is True
            assert task.metadata.has_door is True

    for task in hard_tasks:
        if "Lava" in task.metadata.env_name:
            assert task.metadata.has_lava is True


@pytest.mark.asyncio
async def test_task_instructions():
    """Test that task instructions are properly generated."""
    taskset = await create_minigrid_taskset(num_tasks_per_difficulty={"medium": 5}, seed=456)

    for task in taskset.instances:
        # Check instructions exist
        assert task.impetus.instructions
        assert len(task.impetus.instructions) > 0

        # Check instructions match environment type
        if task.metadata.has_lava:
            assert "avoiding lava" in task.impetus.instructions
        if task.metadata.has_key:
            assert "key" in task.impetus.instructions

        # Check rubric
        assert "goal" in task.intent.rubric
        assert "success_criteria" in task.intent.rubric
        assert isinstance(task.intent.rubric["success_criteria"], list)


@pytest.mark.asyncio
async def test_environment_configurations():
    """Test that environment configurations are valid."""
    # Check all predefined environments
    for difficulty, env_list in ENVIRONMENTS.items():
        assert difficulty in ["easy", "medium", "hard"]
        for env_name, grid_size in env_list:
            assert isinstance(env_name, str)
            assert "MiniGrid" in env_name
            assert isinstance(grid_size, tuple)
            assert len(grid_size) == 2
            assert all(isinstance(x, int) for x in grid_size)


@pytest.mark.asyncio
async def test_reproducibility():
    """Test that taskset generation is reproducible with same seed."""
    seed = 789

    # Generate two tasksets with same seed
    taskset1 = await create_minigrid_taskset(
        num_tasks_per_difficulty={"easy": 3, "medium": 3}, seed=seed
    )

    taskset2 = await create_minigrid_taskset(
        num_tasks_per_difficulty={"easy": 3, "medium": 3}, seed=seed
    )

    # Check that they have the same tasks
    assert len(taskset1.instances) == len(taskset2.instances)

    for t1, t2 in zip(taskset1.instances, taskset2.instances):
        assert t1.metadata.env_name == t2.metadata.env_name
        assert t1.metadata.seed == t2.metadata.seed
        assert t1.metadata.difficulty == t2.metadata.difficulty


@pytest.mark.asyncio
async def test_empty_taskset():
    """Test creating an empty taskset."""
    taskset = await create_minigrid_taskset(num_tasks_per_difficulty={}, seed=42)

    assert len(taskset.instances) == 0
    assert taskset.split_info.val_instance_ids == set()
    assert taskset.split_info.test_instance_ids == set()


@pytest.mark.asyncio
async def test_task_instance_fields():
    """Test all required fields are present in task instances."""
    task = DEFAULT_MINIGRID_TASK

    # Check all required fields
    assert hasattr(task, "id")
    assert hasattr(task, "impetus")
    assert hasattr(task, "intent")
    assert hasattr(task, "metadata")
    assert hasattr(task, "is_reproducible")
    assert hasattr(task, "initial_engine_snapshot")

    # Check metadata fields
    metadata = task.metadata
    assert hasattr(metadata, "env_name")
    assert hasattr(metadata, "grid_size")
    assert hasattr(metadata, "difficulty")
    assert hasattr(metadata, "has_key")
    assert hasattr(metadata, "has_door")
    assert hasattr(metadata, "has_lava")
    assert hasattr(metadata, "num_objects")
    assert hasattr(metadata, "optimal_path_length")
    assert hasattr(metadata, "seed")


if __name__ == "__main__":
    asyncio.run(pytest.main([__file__, "-v"]))
