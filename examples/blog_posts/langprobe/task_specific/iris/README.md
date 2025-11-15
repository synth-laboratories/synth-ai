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

## Synth MIPRO (Simple API)

The in-process Synth adapters now take advantage of `MIPROConfig.simple()` for single-stage tasks like Iris. If you want to construct configs directly (e.g., in a notebook), you can do:

```python
from synth_ai.api.train.configs.prompt_learning import MIPROConfig

iris_mipro = MIPROConfig.simple(
    task_app_url="http://127.0.0.1:8115",
    task_app_api_key="ENVIRONMENT_API_KEY",
    env_name="iris",
    rollout_budget=100,
    initial_prompt_messages=[
        {"role": "system", "content": "Classify iris flowers."},
        {"role": "user", "content": "Flower Measurements:\n{features}\n\nPredict the species."},
    ],
)
```

`iris_mipro` can then be attached to `PromptLearningConfig` when submitting jobs through the SDK, or passed directly to the in-process optimizer if you're experimenting locally.

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
