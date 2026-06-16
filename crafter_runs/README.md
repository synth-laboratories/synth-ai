# Crafter DEO parallel race

Self-contained Crafter code-policy directed-effort run. **Edit setup in one place:**
[`crafter_deo_run.py`](crafter_deo_run.py) → `CrafterDeoRunConfig` (playbook, worker briefs,
seeds, models, timeboxes). Runnable code stays in [`lane/`](lane/).

Uses **guidance-only kickoff**: empty `kickoff.tasks`, rich `project_notes_framing` +
`task_briefs`; orchestrator must **create the directed-effort objective** (from lane
`[smr.objective]` spec), publish milestones, then `plan_tasks`. Nothing is pre-created
at launch (`sdk_precreate = false`; driver strips any bundle `primary_parent`).

| Path | Purpose |
|------|---------|
| [`crafter_deo_run.py`](crafter_deo_run.py) | **SDK driver + run config** (edit here first) |
| [`lane/`](lane/) | Baseline policy, sweep runner, hillclimb script, contracts |
| [`runs/`](runs/) | Per-run artifacts (`summary.json`, `workspace.tar.gz`, …) |

## Prerequisites: Crafter actor runtime image

Workers need the Open Research Crafter actor image locally. Build it once from
the workspace root (requires `synth-local-smr-runtime:latest`, normally present
after `synth-dev` slot bring-up):

```bash
docker buildx build \
  -f ~/Documents/GitHub/backend/infra/Dockerfile.actor_runtime_images \
  --target open_research_crafter \
  --build-arg BASE_IMAGE=synth-local-smr-runtime:latest \
  -t synth-local-open-research-crafter:latest \
  --load ~/Documents/GitHub
```

Verify: `docker image inspect synth-local-open-research-crafter:latest`

The driver checks this image before launch (`crafter_deo_run.py`). Pass
`--skip-docker-image-check` only when the backend resolves the image elsewhere.

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
| `objective_authoring_playbook()` | Turn-0 create objective + milestones + progress claims (from lane spec) |
| `project_notes()` | Full orchestrator playbook: objective bootstrap, plan_tasks, closeout WP |
| `worker_planning_briefs()` | Per-worker templates copied into planned task `input` |
| `worker_instructions()` | Core worker prose inside each brief |
| `task_instructions_markdown()` | Staged `TASK_INSTRUCTIONS.md` |
| `extra_worker_instructions` / `extra_orchestrator_instructions` | Append freeform text |
| `train_seeds` | Eval seed list |
| `orchestrator_profile_id` / `worker_profile_id` | Default: `opencode_deepseek_v4_flash` (DeepSeek V4 Flash via OpenRouter) |
| `run_timebox_seconds` / `poll_timebox_seconds` | Launch + poll limits |

Change policy/runner code under `lane/containers/` and `lane/workspace/` when needed.

## Verification gates

**Bundle (local, no slot):** `build_launch_bundle` emits guidance-only kickoff on all four
copies — `kickoff.tasks=[]`, `task_brief_count=3`, `kickoff_guidance_mode=guidance_only`.

**Slot1 E2E:** run `uv run python crafter_runs/crafter_deo_run.py --use-default-slot1` and
confirm in `runs/<stamp>/summary.json`:

| Gate | Expected |
|------|----------|
| T+0 | No kickoff task seed; no launch `primary_parent`; `[objective] objectives=0` |
| T+60s | Orchestrator creates objective + `plan_tasks` → 3 `running` workers |
| Poll | `run.log` / `summary.json` show `[objective]` lines every ~30s |
| Worker prompts | Worker task requires a report WorkProduct before `set_task_state(done)` |
| Staged kickoff | `tasks: []` in launch bundle / summary |
| Archive | `candidates/crafter/attempt_*`, `eval_summary.json` after push |
| Closeout | ≥1 WorkProduct; terminal success |
