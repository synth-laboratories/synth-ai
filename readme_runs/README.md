# README smoke runs

ReportBench README smoke driver and local run artifacts.

Uses **guidance-only kickoff**: empty `kickoff.tasks`, rich `project_notes_framing` +
one `task_brief`. Orchestrator must `plan_tasks` exactly one repo task; the worker
owns report WorkProduct publication.

| Path | Purpose |
|------|---------|
| [`readme_smoke.py`](readme_smoke.py) | Research SDK driver (launch, poll, archive, Codex verifier) |
| [`kickoff_guidance.py`](kickoff_guidance.py) | Shared guidance-only kickoff bundle patching |
| [`runs/`](runs/) | One directory per smoke execution |

## Run

```bash
cd ~/Documents/GitHub/synth-ai
bash scripts/run_readme_smoke_slot1.sh
```

Without `--output-root`, each run creates `runs/<UTC>_<target>/` (for example
`runs/20260604T214400Z_slot1/`).

After a run:

- **`runs/latest`** — symlink to the newest run folder
- **`runs/index.jsonl`** — one JSON line per run (ids, exit code, verifier score)
- Inside each run dir: `summary.json`, `run.log`, `README.md`, `workspace.tar.gz`, `artifacts/`

Override output location: `--output-root /path` or `OUTPUT_ROOT=/path`.

## Customize guidance

Edit `ReadmeSmokeRunConfig` in [`readme_smoke.py`](readme_smoke.py):

| Field | What it controls |
|-------|------------------|
| `extra_orchestrator_notes` | Appended to `project_notes_framing` |
| `extra_worker_notes` | Appended to the single worker planning brief |

## Verification gates

**Bundle (local):** guidance-only kickoff — `tasks=[]`, `task_brief_count=1`.

**Slot1 E2E (T1 regression):**

| Gate | Expected |
|------|----------|
| T+0 | `tasks=0` before orchestrator plans |
| Worker | Single planned task; README proof marker; report WP by worker |
| Archive | `workspace.tar.gz` with proof marker |
| Verifier | Codex verifier pass |
