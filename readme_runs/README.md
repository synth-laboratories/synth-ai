# README smoke runs

ReportBench README smoke driver and local run artifacts.

| Path | Purpose |
|------|---------|
| [`readme_smoke.py`](readme_smoke.py) | Research SDK driver (launch, poll, archive, Codex verifier) |
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
