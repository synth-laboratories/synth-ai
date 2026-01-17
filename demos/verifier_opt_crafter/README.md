# Crafter Verifier Optimization Demo

This demo optimizes a verifier graph that scores fixed Crafter traces using GraphGen (Graph-GEPA).

## Overview

The demo:
1. Loads fixed Crafter execution traces with gold scores
2. Uses GraphGen to optimize a verifier graph that predicts scores correlating with gold labels
3. Evaluates the optimized verifier on held-out traces
4. Compares baseline vs optimized verifier performance

## Prerequisites

- Backend running (default: `http://localhost:8000` for local mode)
- `SYNTH_API_KEY` environment variable set
- Crafter trace dataset (see below)

## Dataset

The demo expects a Crafter trace dataset in one of these formats:

1. **ADAS format** (preferred): JSON file with `tasks` and `gold_outputs` arrays
   - Location: `demos/verifier_opt_crafter/crafter_judge_adas_dataset.json`
   - Or: `demos/verifier_opt_crafter/crafter_verifier_graph_opt_dataset.json`
   - Or: `demos/verifier_opt_crafter/data/crafter_verifier_graph_opt_dataset.json`

2. **JSONL format**: One trace per line with `trace_id`, `trace`, and `gold_score` fields

Each trace should contain:
- `trace`: V3/V4 trace object with event_history or session timesteps, events, rewards
- `trace_id`: Unique identifier
- `gold_score`: Float in [0, 1] (achievement-based label)

## Usage

### Local Mode (recommended for development)

```bash
uv run python demos/verifier_opt_crafter/run_demo.py --local
```

### Production Mode

```bash
uv run python demos/verifier_opt_crafter/run_demo.py
```

### Options

```bash
--local              Use localhost:8000 backend (no tunnels)
--dataset-path PATH  Path to trace dataset (if not in default location)
--max-traces N       Maximum traces to use (default: 30)
--generations N      Number of optimization generations (default: 3)
--children N         Children per generation (default: 3)
--rollout-budget N  Rollout budget for optimization (default: 100)
```

## Example Output

```
============================================================
LOADING CRAFTER TRACES
============================================================
Loading dataset from: /path/to/crafter_judge_adas_dataset.json
Loaded 30 traces from ADAS dataset
Train: 24 traces, Val: 6 traces

============================================================
RUNNING VERIFIER GRAPH OPTIMIZATION
============================================================
GraphGen Job ID: graphgen_abc123...
Graph Evolve Job ID: graph_evolve_xyz789...

Streaming optimization progress...
------------------------------------------------------------
gen 1/3 started (budget 0/100)
candidate abc123 score=0.65 seeds=24
candidate def456 score=0.72 seeds=24
gen 1 done best=0.72 avg=0.68 sec=45
...

============================================================
OPTIMIZATION COMPLETE
============================================================
Status: succeeded
Best Score: 0.85
Best Snapshot ID: snapshot_abc123...
Duration: 2m 30s
```

## How It Works

1. **Dataset Loading**: Loads fixed Crafter traces with gold scores
2. **GraphGen Optimization**: Uses GraphGen (Graph-GEPA) to evolve verifier graphs
3. **Scoring**: Each candidate graph is evaluated on fixed traces, scored by correlation with gold labels
4. **Validation**: Best graph is evaluated on held-out traces

## Related Demos

- `demos/gepa_banking77/` - Prompt optimization with GEPA
- `demos/gepa_crafter_vlm/` - Crafter VLM agent optimization
- `demos/image_style_matching/` - GraphGen for image style matching

## Research Scripts

For more advanced usage (in-process optimization, custom scoring strategies), see:
- `research/graph_opt_verifiers/crafter/run_graph_gepa_crafter_verifier_benchmark.py`
- `cookbooks/code/training/graph_learning/crafter_verifier_graphgen/run_crafter_verifier_graphgen.py`







