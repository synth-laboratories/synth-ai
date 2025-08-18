#!/usr/bin/env python3
"""
Dataset builder for Crafter Custom Environments
"""

import gzip
import json
import pickle
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


@dataclass
class CrafterTask:
    """Defines the global task parameters for Crafter"""

    global_premises: str = "You are playing Crafter, a survival game where you collect resources, craft items, and survive."
    global_constraints: str = "You can only perform actions available in the action space. You must manage health and hunger."
    global_objectives: str = (
        "Achieve as many unique achievements as possible to maximize your score."
    )
    shared_env_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.shared_env_params is None:
            self.shared_env_params = {"render_mode": None, "size": (64, 64)}


@dataclass
class CrafterImpetus:
    """Instructions for a specific Crafter instance"""

    instructions: str
    achievement_focus: Optional[List[str]] = None  # Optional list of achievements to prioritize


@dataclass
class CrafterIntent:
    """Success criteria and evaluation for Crafter"""

    rubric: Dict[str, Any]
    target_achievements: Optional[List[str]] = None
    minimum_score: Optional[int] = None
    gold_trajectories: Optional[List[Dict[str, Any]]] = None


@dataclass
class CrafterMetadata:
    """Metadata for a Crafter instance"""

    difficulty: str  # easy, normal, hard, peaceful, resource_rich
    world_seed: int
    spawn_position: Optional[tuple] = None
    initial_resources_nearby: Optional[Dict[str, int]] = None
    initial_entities: Optional[Dict[str, int]] = None
    config_params: Optional[Dict[str, Any]] = None


@dataclass
class SplitInfo:
    """Train/val/test split information"""

    val_instance_ids: Set[str]
    test_instance_ids: Set[str]
    _is_split_defined: bool = True


@dataclass
class CrafterInstance:
    """A single Crafter environment instance"""

    id: uuid.UUID
    impetus: CrafterImpetus
    intent: CrafterIntent
    metadata: CrafterMetadata
    is_reproducible: bool = True
    initial_engine_snapshot: Optional[bytes] = None  # Serialized engine state


@dataclass
class CrafterDataset:
    """Collection of Crafter instances"""

    name: str
    description: str
    instances: List[CrafterInstance]
    split_info: SplitInfo
    task: CrafterTask


class CrafterDatasetBuilder:
    """Builds datasets of Crafter instances with various difficulties"""

    DIFFICULTIES = ["easy", "normal", "hard", "peaceful", "resource_rich"]

    # Achievement categories for focused tasks
    ACHIEVEMENT_CATEGORIES = {
        "basic_survival": ["collect_coal", "collect_wood", "eat_cow", "collect_sapling"],
        "tools": ["make_wood_pickaxe", "make_stone_pickaxe", "make_iron_pickaxe"],
        "combat": ["defeat_zombie", "defeat_skeleton", "make_wood_sword", "make_stone_sword"],
        "advanced": ["make_iron_sword", "collect_diamond", "place_table", "place_plant"],
        "exploration": ["collect_stone", "collect_iron", "collect_coal", "drink_water"],
    }

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("dataset")
        self.output_dir.mkdir(exist_ok=True)

    def create_instance(
        self,
        difficulty: str,
        seed: int,
        impetus_type: str = "general",
        achievement_focus: Optional[List[str]] = None,
    ) -> CrafterInstance:
        """Create a single Crafter instance"""

        # Create impetus based on type
        if impetus_type == "general":
            impetus = CrafterImpetus(
                instructions="Survive and thrive in the Crafter world. Collect resources, craft tools, and achieve as many accomplishments as possible.",
                achievement_focus=None,
            )
            intent = CrafterIntent(
                rubric={
                    "description": "General survival and achievement",
                    "criteria": {
                        "achievements": "Number of unique achievements unlocked",
                        "survival_time": "Number of steps survived",
                        "resource_efficiency": "Efficient use of collected resources",
                    },
                },
                minimum_score=1 if difficulty == "easy" else (3 if difficulty == "normal" else 5),
            )
        elif impetus_type == "focused":
            focus_category = random.choice(list(self.ACHIEVEMENT_CATEGORIES.keys()))
            achievements = self.ACHIEVEMENT_CATEGORIES[focus_category]
            impetus = CrafterImpetus(
                instructions=f"Focus on {focus_category.replace('_', ' ')} achievements. Try to complete: {', '.join(achievements[:3])}",
                achievement_focus=achievements,
            )
            intent = CrafterIntent(
                rubric={
                    "description": f"Complete {focus_category} achievements",
                    "criteria": {
                        "target_achievements": f"Complete at least one of: {', '.join(achievements[:3])}",
                        "efficiency": "Complete target achievements quickly",
                    },
                },
                target_achievements=achievements[:3],
                minimum_score=1,
            )
        elif impetus_type == "speedrun":
            target = random.choice(["place_table", "make_iron_pickaxe", "collect_diamond"])
            impetus = CrafterImpetus(
                instructions=f"Speedrun challenge: {target} as quickly as possible!",
                achievement_focus=[target],
            )
            intent = CrafterIntent(
                rubric={
                    "description": f"Complete {target} speedrun",
                    "criteria": {
                        "completion": f"Successfully {target}",
                        "speed": "Complete in minimal steps",
                    },
                },
                target_achievements=[target],
                minimum_score=1,
            )
        else:
            raise ValueError(f"Unknown impetus type: {impetus_type}")

        # Create metadata
        metadata = CrafterMetadata(
            difficulty=difficulty,
            world_seed=seed,
            config_params=self._get_config_params(difficulty),
        )

        # Create instance
        instance = CrafterInstance(
            id=uuid.uuid4(),
            impetus=impetus,
            intent=intent,
            metadata=metadata,
            is_reproducible=True,
            initial_engine_snapshot=None,
        )

        return instance

    def _get_config_params(self, difficulty: str) -> Dict[str, Any]:
        """Get key config parameters for a difficulty"""
        # These are simplified summaries - actual configs are in JSON files
        params = {
            "easy": {
                "resource_multiplier": "~2x",
                "enemy_density": "low",
                "coal_probability": 0.25,
                "iron_probability": 0.35,
                "diamond_probability": 0.02,
            },
            "normal": {
                "resource_multiplier": "1x",
                "enemy_density": "normal",
                "coal_probability": 0.15,
                "iron_probability": 0.25,
                "diamond_probability": 0.006,
            },
            "hard": {
                "resource_multiplier": "~0.5x",
                "enemy_density": "high",
                "coal_probability": 0.08,
                "iron_probability": 0.15,
                "diamond_probability": 0.002,
            },
            "peaceful": {
                "resource_multiplier": "~3x",
                "enemy_density": "none",
                "coal_probability": 0.30,
                "iron_probability": 0.40,
                "diamond_probability": 0.03,
            },
            "resource_rich": {
                "resource_multiplier": "~8x",
                "enemy_density": "minimal",
                "coal_probability": 0.60,
                "iron_probability": 0.70,
                "diamond_probability": 0.80,
            },
        }
        return params.get(difficulty, {})

    def build_dataset(
        self,
        name: str,
        instances_per_difficulty: Dict[str, int],
        impetus_distribution: Dict[str, float] = None,
        val_split: float = 0.2,
        test_split: float = 0.2,
    ) -> CrafterDataset:
        """Build a complete dataset"""

        if impetus_distribution is None:
            impetus_distribution = {"general": 0.6, "focused": 0.3, "speedrun": 0.1}

        instances = []

        # Create instances for each difficulty
        for difficulty, count in instances_per_difficulty.items():
            if difficulty not in self.DIFFICULTIES:
                raise ValueError(f"Unknown difficulty: {difficulty}")

            for i in range(count):
                # Generate unique seed
                seed = random.randint(0, 1000000)

                # Choose impetus type based on distribution
                impetus_type = random.choices(
                    list(impetus_distribution.keys()), weights=list(impetus_distribution.values())
                )[0]

                instance = self.create_instance(
                    difficulty=difficulty, seed=seed, impetus_type=impetus_type
                )
                instances.append(instance)

        # Create splits
        random.shuffle(instances)
        n_val = int(len(instances) * val_split)
        n_test = int(len(instances) * test_split)

        val_ids = {str(inst.id) for inst in instances[:n_val]}
        test_ids = {str(inst.id) for inst in instances[n_val : n_val + n_test]}

        split_info = SplitInfo(
            val_instance_ids=val_ids, test_instance_ids=test_ids, _is_split_defined=True
        )

        # Create dataset
        dataset = CrafterDataset(
            name=name,
            description=f"Crafter dataset with {len(instances)} instances across {len(instances_per_difficulty)} difficulties",
            instances=instances,
            split_info=split_info,
            task=CrafterTask(),
        )

        return dataset

    def save_dataset(self, dataset: CrafterDataset, format: str = "json"):
        """Save dataset to disk"""
        dataset_dir = self.output_dir / dataset.name
        dataset_dir.mkdir(exist_ok=True)

        if format == "json":
            # Save metadata
            metadata = {
                "name": dataset.name,
                "description": dataset.description,
                "num_instances": len(dataset.instances),
                "task": asdict(dataset.task),
                "split_info": {
                    "val_instance_ids": list(dataset.split_info.val_instance_ids),
                    "test_instance_ids": list(dataset.split_info.test_instance_ids),
                },
            }
            with open(dataset_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            # Save instances
            instances_data = []
            for inst in dataset.instances:
                inst_data = {
                    "id": str(inst.id),
                    "impetus": asdict(inst.impetus),
                    "intent": asdict(inst.intent),
                    "metadata": asdict(inst.metadata),
                    "is_reproducible": inst.is_reproducible,
                }
                instances_data.append(inst_data)

            with open(dataset_dir / "instances.json", "w") as f:
                json.dump(instances_data, f, indent=2)

        print(f"Dataset saved to {dataset_dir}")
        print(f"Total instances: {len(dataset.instances)}")
        print(f"Val instances: {len(dataset.split_info.val_instance_ids)}")
        print(f"Test instances: {len(dataset.split_info.test_instance_ids)}")
        print(
            f"Train instances: {len(dataset.instances) - len(dataset.split_info.val_instance_ids) - len(dataset.split_info.test_instance_ids)}"
        )


def main():
    """Example usage"""
    builder = CrafterDatasetBuilder()

    # Build a balanced dataset
    dataset = builder.build_dataset(
        name="crafter_balanced_v1",
        instances_per_difficulty={
            "easy": 20,
            "normal": 20,
            "hard": 20,
            "peaceful": 10,
            "resource_rich": 10,
        },
        impetus_distribution={"general": 0.6, "focused": 0.3, "speedrun": 0.1},
    )

    builder.save_dataset(dataset)

    # Build a difficulty progression dataset
    dataset2 = builder.build_dataset(
        name="crafter_progression_v1",
        instances_per_difficulty={
            "easy": 30,
            "normal": 25,
            "hard": 15,
            "peaceful": 5,
            "resource_rich": 5,
        },
    )

    builder.save_dataset(dataset2)


if __name__ == "__main__":
    main()
