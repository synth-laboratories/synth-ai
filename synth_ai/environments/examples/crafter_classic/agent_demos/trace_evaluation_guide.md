# Crafter Trace Evaluation System

## Overview

The trace evaluation system provides a quantitative scoring mechanism for Crafter agent episodes based on achievements earned and invalid actions taken.

## Scoring System

### Weights
- **Easy Achievement**: +1.0 points
- **Medium Achievement**: +2.5 points  
- **Hard Achievement**: +5.0 points
- **Invalid Action**: -0.05 points

The weights are designed so that 50 invalid actions approximately equal the penalty of missing 1 medium achievement.

### Achievement Categories

**Easy Achievements** (require no prerequisites):
- collect_wood, collect_stone, collect_sapling, collect_drink
- place_stone, place_table, wake_up, eat_plant

**Medium Achievements** (require some prerequisites):
- make_wood_pickaxe, make_wood_sword, place_furnace, place_plant
- collect_coal, collect_iron, eat_cow

**Hard Achievements** (require many prerequisites):
- make_stone_pickaxe, make_stone_sword, make_iron_pickaxe, make_iron_sword
- collect_diamond, defeat_skeleton, defeat_zombie

### Invalid Actions
Actions that had no effect on the game state, such as:
- Movement blocked by walls/edges
- Attempting to craft without required materials
- Invalid placement attempts

## Usage

### Command Line

```bash
# Evaluate all traces in a directory
python trace_eval.py traces/

# Show detailed evaluation for each trace
python trace_eval.py traces/ --verbose

# Filter by pattern
python trace_eval.py traces/ --pattern "*episode_1*.json"
```

### Programmatic Usage

```python
from trace_eval import evaluate_trace, evaluate_all_traces

# Evaluate single trace
result = evaluate_trace(Path("trace.json"))
print(f"Score: {result['total_score']}")
print(f"Trajectory: {result['trajectory']}")

# Evaluate multiple traces
results = evaluate_all_traces(Path("traces/"))
for r in results:
    print(f"{r['trace_file']}: {r['total_score']:.2f}")
```

## Trajectory Visualization

Each event in a trace is represented by a symbol:
- `+` : Positive event (achievement unlocked)
- `-` : Negative event (invalid action)
- `0` : Neutral event (would show if we had neutral scored events)
- No symbol: Regular step with no scoring impact

Example trajectory: `-----++----` shows 5 invalid actions, 2 achievements, then 4 more invalid actions.

## Example Output

```
📊 Trace: session_crafter_episode_1.json
   Score: 1.55
   Events: 60
   Breakdown:
     Easy achievements: 2 × 1.0 = 2.00
     Invalid actions: 9 × -0.05 = -0.45
   Trajectory: -----++----
```

## Interpreting Results

- **Score > 2.0**: Excellent performance with multiple achievements
- **Score 1.0-2.0**: Good performance with some achievements
- **Score 0-1.0**: Limited success, few achievements
- **Score < 0**: Poor performance dominated by invalid actions

## Implementation Details

The evaluation system:
1. Reads trace files containing SessionEvent history
2. Examines event_metadata added by hooks (achievement and invalid action hooks)
3. Calculates scores based on hook detections
4. Generates trajectory strings for visualization
5. Provides detailed breakdowns and statistics

## Files

- `trace_eval.py` - Core evaluation functions
- `eval_example.py` - Example usage scripts
- `eval_by_difficulty.py` - Evaluation grouped by difficulty level
- `trace_hooks.py` - Hook implementations that generate the metadata