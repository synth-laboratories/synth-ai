# Continual Learning Comparison: Banking77

This demo compares classic GEPA (non-continual) vs MIPRO continual learning on Banking77 dataset with progressive data splits.

## Hypothesis

**Continual learning will outperform restart-from-scratch on later splits**, because:
1. Earlier learned prompts transfer knowledge to new intents
2. The ontology accumulates useful patterns over time
3. Restart loses all prior learning at each split

## Data Splits

We create 4 progressive data splits where each is a superset of the previous:

| Split | Intents | Description |
|-------|---------|-------------|
| Split 1 | 2 | `card_arrival`, `lost_or_stolen_card` |
| Split 2 | 7 | Split 1 + 5 more card-related intents |
| Split 3 | 27 | Split 2 + 20 more common banking intents |
| Split 4 | 77 | All Banking77 intents (complete dataset) |

## Approaches

### Classic GEPA (Non-Continual)
- Run GEPA on Split 1, score, save best prompt
- Run GEPA on Split 2 with two conditions:
  - **Warm start**: Initialize with Split 1's best prompt
  - **Cold start**: Initialize with baseline prompt
- Repeat for Splits 3 and 4
- Compare warm start vs cold start at each stage

### MIPRO Continual Learning
- Run MIPRO in online mode continuously across all splits
- Stream data from Split 1 → 2 → 3 → 4 sequentially
- Track prompt evolution and ontology growth at checkpoints
- No restarts - learning persists throughout

## Results

*(Results will be populated after running the comparison)*

| Split | Cold Start | Warm Start | MIPRO Continual | Best Method |
|-------|------------|------------|-----------------|-------------|
| Split 1 (2 intents) | - | - | - | - |
| Split 2 (7 intents) | - | - | - | - |
| Split 3 (27 intents) | - | - | - | - |
| Split 4 (77 intents) | - | - | - | - |

## Usage

### Run Full Comparison
```bash
# Run all experiments
uv run python run_comparison.py

# With custom settings
uv run python run_comparison.py --rollouts-per-split 100 --model gpt-4.1-nano
```

### Run Individual Experiments
```bash
# Classic GEPA approach
uv run python run_classic_gepa.py --split 1

# MIPRO continual approach  
uv run python run_mipro_continual.py --rollouts 400
```

### Analyze Results
```bash
# Generate comparison table
uv run python analyze_results.py
```

## Environment Setup

Requires `SYNTH_API_KEY` environment variable:
```bash
export SYNTH_API_KEY=sk_live_...
```

## Files

- `data_splits.py` - Defines the 4 progressive data splits
- `run_classic_gepa.py` - Classic GEPA with warm/cold starts
- `run_mipro_continual.py` - MIPRO continual learning
- `run_comparison.py` - Full comparison runner
- `analyze_results.py` - Results analysis and table generation
