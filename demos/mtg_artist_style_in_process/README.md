# MTG Artist Style (In-Process)

This demo runs a **text-only** MTG artist style matching task app entirely in-process and
uses **SynthTunnel by default** for GEPA/eval jobs. It is designed to be fast, minimal,
and easy to integrate inside your Python app.

## What It Demonstrates

- In-process task app (no external server)
- SynthTunnel by default for backend connectivity
- Programmatic GEPA run with per-customer prompt customization

## Quickstart

```bash
export SYNTH_API_KEY=sk_live_...
# Optional: override backend
# export SYNTH_BACKEND_URL=https://api.usesynth.ai

# Quick validation (~30-60 seconds):
uv run python demos/mtg_artist_style_in_process/run_in_process_gepa.py --quick

# Full optimization run (10-30 minutes):
uv run python demos/mtg_artist_style_in_process/run_in_process_gepa.py
```

### Quick Mode vs Full Mode

| Mode | Seeds | Generations | Budget | Timeout | Use Case |
|------|-------|-------------|--------|---------|----------|
| `--quick` | 3 | 1 | 2 | 2 min | CI/CD, smoke tests, quick validation |
| (default) | 13 | 1 | 6 | **none** | Production optimization (waits until done) |

### Options

```bash
# Verbose output for debugging:
uv run python demos/mtg_artist_style_in_process/run_in_process_gepa.py --quick --verbose

# Custom timeout:
uv run python demos/mtg_artist_style_in_process/run_in_process_gepa.py --timeout 3600

# Submit without polling (returns immediately):
uv run python demos/mtg_artist_style_in_process/run_in_process_gepa.py --no-poll
```

### Customize Per Customer

```bash
uv run python demos/mtg_artist_style_in_process/run_in_process_gepa.py --quick \
  --artist seb_mckinnon \
  --customer-note "Lean into high-contrast, painterly brushwork with muted palettes."
```

## Eval (Optional)

The same config file includes an `[eval]` section. Run from the demo directory:

```python
from pathlib import Path
from synth_ai.sdk import run_in_process_job_sync
from mtg_in_process_task_app import build_config

config_path = Path("gepa_mtg_in_process.toml")
result = run_in_process_job_sync(
    job_type="eval",
    config_path=config_path,
    config_factory=build_config,
)
print(result.status)
```

## Local-Only Override

To force localhost (no tunnel):

```bash
export SYNTH_TUNNEL_MODE=local
uv run python demos/mtg_artist_style_in_process/run_in_process_gepa.py --quick
```

## Files

- `mtg_in_process_task_app.py`: Task app (text-only scoring)
- `gepa_mtg_in_process.toml`: GEPA + eval config
- `run_in_process_gepa.py`: Demo runner

## Troubleshooting

**Job takes a long time:**
- Use `--quick` for faster validation runs (~30-60s)
- Full mode has no timeout by default - it waits until the job completes
- Use `--timeout 600` to set an explicit limit if needed

**404 on job status:**
- This usually means the job wasn't persisted correctly
- Try with `--verbose` to see the full error
- Check that `SYNTH_API_KEY` is valid for the target backend
