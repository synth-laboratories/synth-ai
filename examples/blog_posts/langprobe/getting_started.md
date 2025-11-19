# LangProbe Experiments: Getting Started

This guide shows how to rerun LangProbe experiments across the main stacks:
- Synth GEPA (prod backend, auto-tunnel)
- Synth MiPROv2
- GEPA-AI (library) adapters
- DSPy MiPROv2
- DSPy GEPA

## Prereqs
- Python 3.11 with `.venv` activated (`source .venv/bin/activate`)
- `.env` with `SYNTH_API_KEY`, `ENVIRONMENT_API_KEY` (task apps), `GROQ_API_KEY`
- `cloudflared` installed (for tunnels)
- Optional: `HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1` if running offline

## Synth GEPA (prod, auto-tunnel)
Parallel runner auto-starts task apps, opens a CF tunnel, rewrites TOML, and submits to prod.
Example: Banking77 only
```
USE_TASK_APP_TUNNEL=1 \
START_TASK_APP=1 \
GEPA_TASKS=Banking77 \
BACKEND_BASE_URL=https://agent-learning.onrender.com \
EXTERNAL_BACKEND_URL=https://agent-learning.onrender.com \
BANKING77_TASK_APP_PORT=8102 \
REDIS_URL=redis://localhost:6379/0 \
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
.venv/bin/python -m examples.blog_posts.langprobe.comparisons.run_gepa_parallel
```
Swap `GEPA_TASKS`/port for: Heart Disease (8114), HotPotQA (8110), Crafter (8116), Hover (8112), IfBench (8111), Pupa (8113), Sokoban (8117), Verilog (8115).

## Synth MiPROv2
DSPy MiPROv2 comparison runner:
```
GEPA_TASKS=Banking77 \  # or other mapped tasks
.venv/bin/python -m examples.blog_posts.langprobe.comparisons.run_dspy_miprov2
```
Configs live under `examples/blog_posts/langprobe/task_specific/<task>/`.

## GEPA-AI (library) adapters
Adapters live under `examples/blog_posts/langprobe/task_specific/*/gepa_ai_*_adapter.py`.
Example (Banking77):
```
.venv/bin/python -m examples.blog_posts.langprobe.task_specific.banking77.run_gepa_ai_banking77 \
  --rollout-budget 100
```
Requires the `gepa` package.

## DSPy MiPROv2 (per-task)
Scripts: `examples/blog_posts/langprobe/task_specific/*/run_dspy_miprov2_*.py`
Example (Banking77):
```
.venv/bin/python -m examples.blog_posts.langprobe.task_specific.banking77.run_dspy_miprov2_banking77 \
  --rollout-budget 100
```

## DSPy GEPA (per-task)
Scripts: `examples/blog_posts/langprobe/task_specific/*/run_dspy_gepa_*.py`
Example (Heart Disease):
```
.venv/bin/python -m examples.blog_posts.langprobe.task_specific.heartdisease.run_dspy_gepa_heartdisease \
  --rollout-budget 100
```

## Tips
- `GEPA_TASKS` can be a comma list (e.g., `Banking77,Heart Disease`) for multi-task runs in the parallel runner.
- Override ports with `TASK_APP_PORT` or `<TASK>_TASK_APP_PORT` if needed.
- Results/logs live under each task’s `results/` folder.
