# Banking77 GEPA Benchmark - LangProBe Dec 31 2024

Benchmark comparing GEPA prompt optimization across different model sizes on Banking77.

## Experiment Design

| Model | Runs | Rollout Budget | Training Seeds | Validation Seeds |
|-------|------|----------------|----------------|------------------|
| gpt-4.1-nano | 3 | 1000 | 100 (0-99) | 100 (100-199) |
| gpt-5-nano | 3 | 1000 | 100 (0-99) | 100 (100-199) |
| gpt-4o-mini | 3 | 1000 | 100 (0-99) | 100 (100-199) |

**Total: 9 experiments**

### GEPA Parameters
- Initial population: 10
- Generations: 5
- Children per generation: 4
- Crossover rate: 0.5
- Mutation rate: 0.3
- Pareto set size: 20

## Running the Benchmark

### Prerequisites

```bash
# Ensure synth-ai is installed
cd /path/to/synth-ai
pip install -e .

# Set API key (optional - will mint demo key if not set)
export SYNTH_API_KEY=sk_live_...
```

### Run All Experiments

```bash
cd /path/to/cookbooks/dev/blog_posts/langprobe_dec31/banking77
python run_benchmark.py
```

### Run Single Model

```bash
python run_benchmark.py --model gpt-4.1-nano
```

### Run Single Experiment

```bash
python run_benchmark.py --model gpt-4.1-nano --run 1
```

### Dry Run

```bash
python run_benchmark.py --dry-run
```

## Results

### Benchmark Results (Dec 31, 2024)

| Model | N | Baseline | GEPA Run 1 | GEPA Run 2 | GEPA Run 3 | GEPA Avg | Improvement |
|-------|---|----------|------------|------------|------------|----------|-------------|
| **gpt-4.1-nano** | 100 | 70% | 90% | 100% | 95% | **95%** | **+25%** |
| **gpt-5-nano** | 100 | 54% | 80% | 55% | 80% | **72%** | **+18%** |
| **gpt-4o-mini** | 100 | 44% | 55% | 70% | 70% | **65%** | **+21%** |

**Key findings:**
- **gpt-4.1-nano**: Best performer - GEPA improved from 70% to 95% (+25 percentage points)
- **gpt-4o-mini**: GEPA improved from 44% to 65% (+21 percentage points)
- **gpt-5-nano**: GEPA improved from 54% to 72% (+18 percentage points), with higher variance across runs

**Notes:**
- N = number of validation seeds (100-199)
- gpt-5-nano is a reasoning model requiring temperature=1.0 and ~700 reasoning tokens per call
- Baseline uses a simple system prompt; GEPA optimizes the prompt through evolutionary search

### Output Files

Results are saved to `results/`:
- `banking77_{model}_run{N}_result.json` - Individual run results
- `benchmark_summary.json` - Full benchmark summary

Each result JSON contains:
```json
{
  "model": "gpt-4.1-nano",
  "run": 1,
  "job_id": "pl_...",
  "status": "succeeded",
  "best_score": 0.85,
  "elapsed_seconds": 180.5,
  "timestamp": "2024-12-31T..."
}
```

## Analysis

To analyze results:

```python
import json
from pathlib import Path

results_dir = Path("results")
results = []
for f in results_dir.glob("*_result.json"):
    with open(f) as fp:
        results.append(json.load(fp))

# Group by model
from collections import defaultdict
by_model = defaultdict(list)
for r in results:
    if r.get("best_score"):
        by_model[r["model"]].append(r["best_score"])

# Print summary
for model, scores in sorted(by_model.items()):
    mean = sum(scores) / len(scores)
    print(f"{model}: {mean:.1%} (n={len(scores)})")
```

## Configs

All TOML configs are in `configs/`:
- `banking77_gpt41nano_run{1,2,3}.toml`
- `banking77_gpt5nano_run{1,2,3}.toml`
- `banking77_gpt4omini_run{1,2,3}.toml`
