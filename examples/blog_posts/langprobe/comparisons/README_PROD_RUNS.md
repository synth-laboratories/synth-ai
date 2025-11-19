# LangProbe GEPA Prod Runs (agent-learning) with Auto-Tunnel

One-command prod runs with auto-started task apps and Cloudflare tunnels. The runner:
- Starts the task app in-process (optional, but default when tunneling)
- Opens a Cloudflare tunnel for localhost task apps
- Rewrites the TOML to the tunneled URL
- Submits to prod (`https://agent-learning.onrender.com`)

## Supported Task Apps (auto-start/tunnel map)
- Banking77 → `examples/task_apps/banking77/banking77_task_app.py` (8102)
- Heart Disease → `examples/task_apps/other_langprobe_benchmarks/heartdisease_task_app.py` (8114)
- HotPotQA → `examples/task_apps/other_langprobe_benchmarks/hotpotqa_task_app.py` (8110)
- Crafter → `examples/task_apps/gepa_benchmarks/crafter_task_app.py` (8116)
- Hover → `examples/task_apps/gepa_benchmarks/hover_task_app.py` (8112)
- IfBench → `examples/task_apps/gepa_benchmarks/ifbench_task_app.py` (8111)
- Pupa → `examples/task_apps/gepa_benchmarks/pupa_task_app.py` (8113)
- Sokoban → `examples/task_apps/gepa_benchmarks/sokoban_task_app.py` (8117)
- Verilog → `examples/task_apps/gepa_benchmarks/verilog_task_app.py` (8115)

Configs live under `examples/blog_posts/langprobe/task_specific/<task>/<task>_gepa.toml`.

## One-task prod run (example: Banking77)
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

## Notes
- `USE_TASK_APP_TUNNEL=1` auto-creates a CF tunnel; `START_TASK_APP=1` auto-starts the task app in-process (recommended).
- For multiple tasks, set `GEPA_TASKS` to a comma list (only mapped tasks above auto-start/tunnel).
- Prod defaults are in `synth_gepa_config.yaml` (rollout 200, max_trials 30, max_cost $5, num_generations 1, local_backend=false).
- Results/logs save under each task’s `results/` directory (e.g., `examples/blog_posts/langprobe/task_specific/banking77/results/`).
- Experimental/draft runners, OPiK/DSPy comparisons, and scaling-law scripts live in `examples/blog_posts/drafting/langprobe/` to keep the prod flow clean.
