import logging
import os
from dataclasses import asdict, dataclass, fields
from typing import List, Tuple
from uuid import UUID, uuid4

from synth_ai.environments.examples.sokoban.puzzle_loader import (
    SokobanPuzzle,
    get_puzzle_loader,
)
from synth_ai.environments.tasks.core import (
    Impetus,
    Intent,
    SplitInfo,
    Task,
    TaskInstance,
    TaskInstanceMetadata,
    TaskInstanceMetadataFilter,
    TaskInstanceSet,
)

logger = logging.getLogger(__name__)

sokoban_task = Task(
    global_premises="Procedural Sokoban task generation",
    global_constraints="",
    global_objectives="Push all boxes onto target locations",
    shared_env_params={},
)

# Configuration parameters
NUM_INSTANCES_PER_DIFFICULTY = 10  # Number of puzzles to include per difficulty in the taskset
DIFFICULTY_CONFIGS = {
    "ultra_easy": {
        "impetus_prompt": "Solve this very simple Sokoban puzzle by pushing the box onto the target.",
    },
    "easy": {
        "impetus_prompt": "Solve this simple Sokoban puzzle by pushing the box onto the target.",
    },
    "medium": {
        "impetus_prompt": "Solve this Sokoban puzzle by pushing the 2 boxes onto the targets.",
    },
    "hard": {
        "impetus_prompt": "Solve this challenging Sokoban puzzle by pushing the 3 boxes onto the targets.",
    },
}


@dataclass
class SokobanTaskInstanceMetadata(TaskInstanceMetadata):
    difficulty: str
    num_boxes: int
    dim_room: Tuple[int, int]
    max_steps: int
    shortest_path_length: int
    seed: int
    generation_params: str


@dataclass
class SokobanTaskInstance(TaskInstance):
    async def serialize(self) -> dict:
        data = asdict(self)
        if "id" in data and isinstance(data["id"], UUID):
            data["id"] = str(data["id"])
        if "intent" in data and data["intent"] is not None:
            if "deterministic_eval_functions" in data["intent"]:
                data["intent"]["deterministic_eval_functions"] = []
        return data

    @classmethod
    async def deserialize(cls, data: dict) -> "SokobanTaskInstance":
        """Gracefully accept non-UUID ids (e.g. 'demo-mcts')."""
        if "id" in data:
            try:
                data["id"] = UUID(str(data["id"]))
            except (ValueError, TypeError, AttributeError):
                pass  # keep original string

        if "impetus" in data and isinstance(data["impetus"], dict):
            data["impetus"] = Impetus(**data["impetus"])

        if "intent" in data and isinstance(data["intent"], dict):
            intent_data = data["intent"]
            intent_data["deterministic_eval_functions"] = []
            if "gold_trajectories" in intent_data and intent_data["gold_trajectories"] is not None:
                pass
            data["intent"] = Intent(**intent_data)

        if "metadata" in data and isinstance(data["metadata"], dict):
            data["metadata"] = SokobanTaskInstanceMetadata(**data["metadata"])

        constructor_field_names = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in constructor_field_names}

        return cls(**filtered_data)


async def create_sokoban_taskset() -> TaskInstanceSet:
    """Generates Sokoban task instances from pre-generated verified puzzles."""
    instances = []

    # Load pre-generated puzzles
    try:
        puzzle_loader = get_puzzle_loader()
        logger.info("Loading pre-generated Sokoban puzzles...")
        puzzle_loader.load_puzzles()
        logger.info(f"Loaded {puzzle_loader.get_total_puzzle_count()} total puzzles")
    except Exception as e:
        logger.error(f"Failed to load pre-generated puzzles: {e}")
        logger.info("Falling back to empty taskset. Run generate_verified_puzzles.py first.")
        return TaskInstanceSet(
            name="Sokoban Verified TaskSet",
            description="Verified pre-generated Sokoban tasks with guaranteed solvability.",
            instances=[],
            split_info=SplitInfo(
                val_instance_ids=set(), test_instance_ids=set(), _is_split_defined=True
            ),
        )

    for difficulty, config in DIFFICULTY_CONFIGS.items():
        available_puzzles = puzzle_loader.get_puzzles_by_difficulty(difficulty)

        if not available_puzzles:
            logger.warning(f"No puzzles found for difficulty {difficulty}")
            continue

        # Take up to NUM_INSTANCES_PER_DIFFICULTY puzzles
        puzzles_to_use = available_puzzles[:NUM_INSTANCES_PER_DIFFICULTY]
        logger.info(f"Using {len(puzzles_to_use)} puzzles for {difficulty} difficulty")

        for puzzle in puzzles_to_use:
            instance_id = uuid4()

            impetus = Impetus(instructions=config["impetus_prompt"])
            intent = Intent(
                rubric={"goal": "Push all boxes onto target locations."},
                gold_trajectories=None,
                gold_state_diff={},
            )
            metadata = SokobanTaskInstanceMetadata(
                difficulty=difficulty,
                num_boxes=puzzle.num_boxes,
                dim_room=puzzle.dim_room,
                max_steps=puzzle.max_steps,
                shortest_path_length=puzzle.solution_length,
                seed=puzzle.generation_seed,
                generation_params=f"verified_puzzle_id={puzzle.id}",
            )

            # Use the puzzle data as the initial engine snapshot
            initial_engine_snapshot = puzzle.to_engine_snapshot()

            task_instance = SokobanTaskInstance(
                id=instance_id,
                impetus=impetus,
                intent=intent,
                metadata=metadata,
                is_reproducible=True,
                initial_engine_snapshot=initial_engine_snapshot,
            )
            instances.append(task_instance)

    class NumBoxesFilter(TaskInstanceMetadataFilter):
        def __init__(self, num_boxes):
            self.num_boxes = num_boxes

        def __call__(self, instance):
            if hasattr(instance.metadata, "num_boxes"):
                return instance.metadata.num_boxes == self.num_boxes
            return False

    class DimRoomFilter(TaskInstanceMetadataFilter):
        def __init__(self, dim_room):
            self.dim_room = dim_room

        def __call__(self, instance):
            if hasattr(instance.metadata, "dim_room"):
                return instance.metadata.dim_room == self.dim_room
            return False

    class PathLengthFilter(TaskInstanceMetadataFilter):
        def __init__(self, min_length=None, max_length=None):
            self.min_length = min_length
            self.max_length = max_length

        def __call__(self, instance):
            if not hasattr(instance.metadata, "shortest_path_length"):
                return False
            length = instance.metadata.shortest_path_length
            if self.min_length is not None and length < self.min_length:
                return False
            if self.max_length is not None and length > self.max_length:
                return False
            return True

    val_filter = NumBoxesFilter(2)
    test_filter = PathLengthFilter(max_length=10)
    val_ids = {inst.id for inst in instances if val_filter(inst)}
    # remove anything already tagged as validation
    test_ids = {inst.id for inst in instances if test_filter(inst) and inst.id not in val_ids}
    split_info = SplitInfo(
        val_instance_ids=val_ids,
        test_instance_ids=test_ids,
        _is_split_defined=True,
    )

    return TaskInstanceSet(
        name="Sokoban Verified TaskSet",
        description="Verified pre-generated Sokoban tasks with guaranteed solvability.",
        instances=instances,
        split_info=split_info,
    )


async def create_easy_sokoban_taskset(num_instances: int = 50) -> TaskInstanceSet:
    """Create a taskset with only easy difficulty puzzles."""
    return await create_filtered_sokoban_taskset(
        difficulties=["easy"], num_instances_per_difficulty=num_instances
    )


async def create_filtered_sokoban_taskset(
    difficulties: List[str], num_instances_per_difficulty: int = 10
) -> TaskInstanceSet:
    """
    Create a taskset with only specified difficulties.

    Args:
        difficulties: List of difficulty levels to include
        num_instances_per_difficulty: Number of instances per difficulty

    Returns:
        TaskInstanceSet with only the specified difficulties
    """
    instances = []

    # Load pre-generated puzzles
    try:
        puzzle_loader = get_puzzle_loader()
        logger.info("Loading pre-generated Sokoban puzzles...")
        puzzle_loader.load_puzzles()
        logger.info(f"Loaded {puzzle_loader.get_total_puzzle_count()} total puzzles")
    except Exception as e:
        logger.error(f"Failed to load pre-generated puzzles: {e}")
        return TaskInstanceSet(
            name="Sokoban Filtered TaskSet",
            description=f"Filtered Sokoban tasks for difficulties: {', '.join(difficulties)}",
            instances=[],
            split_info=SplitInfo(
                val_instance_ids=set(), test_instance_ids=set(), _is_split_defined=True
            ),
        )

    for difficulty in difficulties:
        if difficulty not in DIFFICULTY_CONFIGS:
            logger.warning(f"Unknown difficulty '{difficulty}', skipping")
            continue

        config = DIFFICULTY_CONFIGS[difficulty]
        available_puzzles = puzzle_loader.get_puzzles_by_difficulty(difficulty)

        if not available_puzzles:
            logger.warning(f"No puzzles found for difficulty {difficulty}")
            continue

        # Take up to num_instances_per_difficulty puzzles
        puzzles_to_use = available_puzzles[:num_instances_per_difficulty]
        logger.info(f"Using {len(puzzles_to_use)} puzzles for {difficulty} difficulty")

        for puzzle in puzzles_to_use:
            instance_id = uuid4()

            impetus = Impetus(instructions=config["impetus_prompt"])
            intent = Intent(
                rubric={"goal": "Push all boxes onto target locations."},
                gold_trajectories=None,
                gold_state_diff={},
            )
            metadata = SokobanTaskInstanceMetadata(
                difficulty=difficulty,
                num_boxes=puzzle.num_boxes,
                dim_room=puzzle.dim_room,
                max_steps=puzzle.max_steps,
                shortest_path_length=puzzle.solution_length,
                seed=puzzle.generation_seed,
                generation_params=f"verified_puzzle_id={puzzle.id}",
            )

            # Use the puzzle data as the initial engine snapshot
            initial_engine_snapshot = puzzle.to_engine_snapshot()

            task_instance = SokobanTaskInstance(
                id=instance_id,
                impetus=impetus,
                intent=intent,
                metadata=metadata,
                is_reproducible=True,
                initial_engine_snapshot=initial_engine_snapshot,
            )
            instances.append(task_instance)

    # Create simple split info for filtered set
    val_ids = {inst.id for inst in instances[::3]}  # Every 3rd instance for validation
    test_ids = {inst.id for inst in instances[1::3]}  # Every 3rd starting from 1 for test
    split_info = SplitInfo(
        val_instance_ids=val_ids,
        test_instance_ids=test_ids,
        _is_split_defined=True,
    )

    return TaskInstanceSet(
        name="Sokoban Filtered TaskSet",
        description=f"Filtered Sokoban tasks for difficulties: {', '.join(difficulties)}",
        instances=instances,
        split_info=split_info,
    )


async def create_task_instance_from_seed(difficulty: str, seed: int) -> SokobanTaskInstance:
    """
    Create a single task instance from a specific seed.
    Uses modular arithmetic to deterministically select a puzzle.

    Args:
        difficulty: The difficulty level
        seed: Seed for deterministic puzzle selection

    Returns:
        Single SokobanTaskInstance
    """
    from synth_ai.environments.examples.sokoban.puzzle_loader import get_puzzle_by_seed

    puzzle = get_puzzle_by_seed(difficulty, seed)
    if not puzzle:
        raise ValueError(f"No puzzles available for difficulty '{difficulty}'")

    config = DIFFICULTY_CONFIGS.get(difficulty)
    if not config:
        raise ValueError(f"Unknown difficulty '{difficulty}'")

    instance_id = uuid4()

    impetus = Impetus(instructions=config["impetus_prompt"])
    intent = Intent(
        rubric={"goal": "Push all boxes onto target locations."},
        gold_trajectories=None,
        gold_state_diff={},
    )
    metadata = SokobanTaskInstanceMetadata(
        difficulty=difficulty,
        num_boxes=puzzle.num_boxes,
        dim_room=puzzle.dim_room,
        max_steps=puzzle.max_steps,
        shortest_path_length=puzzle.solution_length,
        seed=seed,  # Use the input seed, not the puzzle's generation seed
        generation_params=f"verified_puzzle_id={puzzle.id}_from_seed={seed}",
    )

    # Use the puzzle data as the initial engine snapshot
    initial_engine_snapshot = puzzle.to_engine_snapshot()

    task_instance = SokobanTaskInstance(
        id=instance_id,
        impetus=impetus,
        intent=intent,
        metadata=metadata,
        is_reproducible=True,
        initial_engine_snapshot=initial_engine_snapshot,
    )

    return task_instance


# Example usage
if __name__ == "__main__":
    import asyncio
    import json
    import os

    NUM_INSTANCES_PER_DIFFICULTY = 2
    # Updated path to examples/sokoban/dataset/instances.json
    OUTPUT_FILE_PATH = "dataset/instances.json"

    async def main():
        taskset = await create_sokoban_taskset()

        serialized = await asyncio.gather(*(inst.serialize() for inst in taskset.instances))

        output_dir = os.path.dirname(OUTPUT_FILE_PATH)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(OUTPUT_FILE_PATH, "w") as f:
            json.dump(serialized, f, indent=2)
        print(f"Serialized {len(serialized)} instances to {OUTPUT_FILE_PATH}")

        with open(OUTPUT_FILE_PATH, "r") as f:
            read_serialized_data = json.load(f)

        deserialized = await asyncio.gather(
            *(SokobanTaskInstance.deserialize(data) for data in read_serialized_data)
        )
        print(f"Deserialized {len(deserialized)} instances.")

        if any(inst is None for inst in deserialized):
            print("Error: Deserialization returned None for some instances.")
            for i, inst in enumerate(deserialized):
                if inst is None:
                    print(
                        f"Instance at index {i} is None. Serialized data: {read_serialized_data[i]}"
                    )
            return

        val_ids = taskset.split_info.val_instance_ids
        test_ids = taskset.split_info.test_instance_ids
        all_ids = {inst.id for inst in deserialized}
        train_ids = all_ids - val_ids - test_ids

        train = [inst for inst in deserialized if inst.id in train_ids]
        val = [inst for inst in deserialized if inst.id in val_ids]
        test = [inst for inst in deserialized if inst.id in test_ids]

        print(f"Train set ({len(train)} instances): {[str(i.id) for i in train]}")
        print(f"Val set ({len(val)} instances): {[str(i.id) for i in val]}")
        print(f"Test set ({len(test)} instances): {[str(i.id) for i in test]}")

    asyncio.run(main())
