# Iris Task-Specific Adapters

This directory contains Iris-specific adapters and runners for different prompt optimization frameworks.

## Files

- `dspy_iris_adapter.py` - DSPy adapters for Iris classification (MIPROv2 and GEPA)
- `lakshya_gepa_adapter.py` - Lakshya Agrawal's GEPA adapter for Iris
- `run_dspy_miprov2_iris.py` - Runner script for DSPy MIPROv2
- `run_dspy_gepa_iris.py` - Runner script for DSPy GEPA
- `run_lakshya_gepa_iris.py` - Runner script for Lakshya's GEPA
- `run_dspy_miprov2.sh` - Bash wrapper for MIPROv2 runner
- `run_dspy_gepa.sh` - Bash wrapper for GEPA runner
- `run_lakshya_gepa.sh` - Bash wrapper for Lakshya's GEPA runner

## Usage

### DSPy MIPROv2
```bash
python3 -m examples.blog_posts.langprobe.task_specific.iris.run_dspy_miprov2_iris --rollout-budget 100
```

### DSPy GEPA
```bash
python3 -m examples.blog_posts.langprobe.task_specific.iris.run_dspy_gepa_iris --rollout-budget 100
```

### Lakshya's GEPA
```bash
python3 -m examples.blog_posts.langprobe.task_specific.iris.run_lakshya_gepa_iris --rollout-budget 100
```

## Results

Results are saved to:
- `iris/results/dspy_mipro/` (MIPROv2)
- `iris/results/dspy_gepa/` (DSPy GEPA)
- `iris/results/lakshya_gepa/` (Lakshya's GEPA)

Each run produces:
- `optimized_prompt.txt` - Human-readable optimized prompt
- `iris_best_module.json` or `iris_best_prompt.json` - Detailed JSON with prompt details
- Learning curve CSV files

