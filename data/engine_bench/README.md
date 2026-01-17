# EngineBench Eval Data

This directory contains traces, artifacts, and results from EngineBench evaluation jobs.

## Directory Structure

```
data/engine_bench/
├── opencode/
│   └── {timestamp}_{job_id}/
│       ├── traces/          # Downloaded trace files
│       ├── eval_results.json # Full eval results with metadata
│       └── summary.txt       # Human-readable summary
└── codex/
    └── {timestamp}_{job_id}/
        ├── traces/
        ├── eval_results.json
        └── summary.txt
```

## Usage

Traces and results are automatically saved here when running:
```bash
cd demos/engine_bench
uv run python run_eval.py --local --seeds 3 --model gpt-4o-mini --agent opencode
```

Or use the helper script:
```bash
cd demos/engine_bench
LOCAL=true ./run_both_agents.sh
```

## File Descriptions

- **traces/**: Contains trace JSON files for each seed, capturing all LLM calls and agent interactions
- **eval_results.json**: Complete evaluation results including:
  - Job metadata (job_id, agent, model, seeds)
  - Status and mean reward
  - Per-seed results with rewards and latencies
  - Configuration used (timeout, verifier settings, etc.)
- **summary.txt**: Human-readable summary for quick reference

## Accessing Traces Later

To load and analyze traces:
```python
import json
from pathlib import Path

# Load eval results
results_file = Path("data/engine_bench/opencode/20250114_120000_job123/eval_results.json")
with open(results_file) as f:
    results = json.load(f)

# Access traces
traces_dir = results_file.parent / "traces"
# Traces are organized by seed in the traces directory
```
