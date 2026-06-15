# Per-run README smoke artifacts

Each subdirectory is one end-to-end smoke execution.

**Quick open:** `runs/latest/` (symlink to the most recent run).

**Typical files in a run folder:**

| File | What it is |
|------|------------|
| `summary.json` | Full driver summary (transcript, o11y, verifier, validation) |
| `run.log` | Timestamped driver log |
| `README.md` | Worker-authored README extracted from the workspace archive |
| `workspace.tar.gz` | Downloaded workspace bundle |
| `artifacts/full_trace.json` | Transcript + task events + o11y for the Codex judge |
| `artifacts/verifier_review.json` | Codex `gpt-5.3-codex-spark` rubric scores |
| `evals_summary.json` | Short pass/fail record for eval tooling |

Run folders are gitignored (large JSON). This README and `index.jsonl` stay in git.
