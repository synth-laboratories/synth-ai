# Crafter DEO hillclimb (1 candidate)

Self-contained Crafter code-policy directed-effort run. **Edit setup in one place:**
[`crafter_deo_run.py`](crafter_deo_run.py) → `CrafterDeoRunConfig` (notes, instructions,
seeds, models, timeboxes). Runnable code stays in [`lane/`](lane/).

| Path | Purpose |
|------|---------|
| [`crafter_deo_run.py`](crafter_deo_run.py) | **SDK driver + run config** (edit here first) |
| [`lane/`](lane/) | Baseline policy, sweep runner, hillclimb script, contracts |
| [`runs/`](runs/) | Per-run artifacts (`summary.json`, `workspace.tar.gz`, …) |

## Run

```bash
cd ~/Documents/GitHub/synth-ai
bash scripts/run_crafter_deo_hillclimb_1cand_slot1.sh
```

Or directly:

```bash
uv run python crafter_runs/crafter_deo_run.py --use-default-slot1
```

Override output: `OUTPUT_ROOT=/path bash scripts/run_crafter_deo_hillclimb_1cand_slot1.sh`

## Customize instructions

Open `crafter_deo_run.py` and edit `CrafterDeoRunConfig`:

| Field | What it controls |
|-------|------------------|
| `parallel_worker_count` / `candidate_ids` | 3 parallel workers → `attempt_1`…`attempt_3` |
| `project_notes()` | Orchestrator: `plan_tasks` all 3 at once, review first improver, submit |
| `worker_instructions()` | Per-worker brief (one candidate dir each) |
| `task_instructions_markdown()` | Staged `TASK_INSTRUCTIONS.md` |
| `extra_worker_instructions` / `extra_orchestrator_instructions` | Append freeform text |
| `train_seeds` | Eval seed list |
| `orchestrator_profile_id` / `worker_profile_id` | Codex profiles |
| `run_timebox_seconds` / `poll_timebox_seconds` | Launch + poll limits |

Change policy/runner code under `lane/containers/` and `lane/workspace/` when needed.
