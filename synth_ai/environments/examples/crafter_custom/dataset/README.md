# Crafter Custom Datasets

This directory contains structured datasets of Crafter environment instances with varying difficulties and objectives.

## Dataset Structure

Each dataset contains:
- `metadata.json`: Dataset metadata including splits and task information
- `instances.json`: Individual environment instances with:
  - Unique ID
  - Difficulty setting (easy/normal/hard/peaceful/resource_rich)
  - World seed for reproducibility
  - Impetus (instructions/objectives)
  - Intent (success criteria)
  - Metadata (config parameters)

## Available Datasets

### crafter_balanced_v1
- 80 instances total (20 per main difficulty)
- Balanced mix of general exploration, focused achievements, and speedrun challenges
- 60% general, 30% focused, 10% speedrun tasks

### crafter_progression_v1
- 80 instances with difficulty progression
- More easy instances (30) gradually decreasing to hard (15)
- Designed for curriculum learning

## Usage

### Running instances from command line:
```bash
# Run 10 random instances
python run_dataset.py crafter_balanced_v1 -n 10

# Run only easy instances
python run_dataset.py crafter_balanced_v1 -n 5 -d easy

# Run validation set
python run_dataset.py crafter_balanced_v1 -s val

# Run speedrun challenges with rendering
python run_dataset.py crafter_balanced_v1 -t speedrun --render
```

### Using in code:
```python
from run_dataset import CrafterDatasetRunner

runner = CrafterDatasetRunner()
results = runner.run_batch(
    dataset_name="crafter_balanced_v1",
    num_instances=10,
    difficulties=["easy", "normal"],
    max_steps=1000
)
```

## Difficulty Settings

- **easy**: ~2x resources, fewer enemies
- **normal**: Standard Crafter experience  
- **hard**: ~0.5x resources, many enemies
- **peaceful**: ~3x resources, no enemies
- **resource_rich**: ~8x resources, minimal enemies

## Instance Types

- **General**: Open-ended survival and achievement
- **Focused**: Target specific achievement categories (tools, combat, etc.)
- **Speedrun**: Complete specific achievements as fast as possible

## Creating New Datasets

```python
from dataset_builder import CrafterDatasetBuilder

builder = CrafterDatasetBuilder()
dataset = builder.build_dataset(
    name="my_dataset",
    instances_per_difficulty={
        "easy": 10,
        "normal": 10,
        "hard": 5
    }
)
builder.save_dataset(dataset)
```