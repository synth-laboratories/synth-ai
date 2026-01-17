---
name: synth-train
description: Run GEPA prompt optimization using Synth AI platform
---

# GEPA Prompt Optimization with Synth AI

GEPA (Generative Evolutionary Prompt Amplification) is Synth's approach to automatic prompt optimization.

## How GEPA Works

1. **Generation**: Creates initial prompt candidates
2. **Evaluation**: Scores each candidate against your task
3. **Evolution**: Mutates and combines promising candidates
4. **Selection**: Returns the best-performing prompt

## Basic Usage

```bash
# Run GEPA optimization with polling
synth-ai train --config gepa_config.toml --poll

# Specify number of rollouts
synth-ai train --config gepa_config.toml --rollouts 50 --poll

# Override task app URL
synth-ai train --config gepa_config.toml --app-url http://localhost:8000 --poll
```

## Config File Format

Create a `gepa_config.toml` file:

```toml
[task]
name = "my_task"
app_url = "http://localhost:8000"

[optimization]
method = "gepa"
rollouts = 50                       # Number of optimization iterations
population_size = 10                # Candidates per generation
mutation_rate = 0.3                 # How aggressively to mutate prompts

[eval]
num_samples = 20                    # Samples per candidate evaluation
```

## Monitoring Optimization

1. Use `--poll` flag to see real-time progress
2. Watch the TUI Jobs panel for status updates
3. View job events ('e' key) to see:
   - Candidate scores
   - Best prompt so far
   - Generation progress

## Example Workflow

```bash
# 1. First, run a baseline eval
synth-ai eval --config eval_config.toml --poll

# 2. Then run GEPA optimization
synth-ai train --config gepa_config.toml --rollouts 30 --poll

# 3. The optimized prompt will be in the job results
```

## Rollout Guidelines

| Use Case | Rollouts | Time Estimate |
|----------|----------|---------------|
| Quick test | 10-20 | ~5-10 min |
| Development | 30-50 | ~15-30 min |
| Production | 50-100 | ~30-60 min |

## Tips

- Start with fewer rollouts (20-30) to validate your setup
- Increase rollouts for production-quality optimization
- Check intermediate results in job events
- The best prompt is saved in the job completion event

