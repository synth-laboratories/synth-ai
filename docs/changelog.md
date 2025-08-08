# Release v[0;34m[INFO][0m Current version: 0.2.1.dev1
[0;34m[INFO][0m Bumping version type: dev
[0;34m[INFO][0m New version: 0.2.1.dev2
[0;32m[SUCCESS][0m Version bumped to 0.2.1.dev2
0.2.2.dev0 - 2025-07-30

## What's New

- **Environment Registration API**: Third-party packages can now register custom environments dynamically via REST API, CLI, or entry points (e.g., `curl -X POST localhost:8901/registry/environments -d '{"name":"MyEnv-v1","module_path":"my_env","class_name":"MyEnv"}'`)
- **Turso Database Integration**: Added Turso/sqld daemon support with local-first database replication via `uvx synth-ai serve` (replicas sync every 2 seconds by default)
- **Environment Service Daemon**: The `uvx synth-ai serve` command now starts both the Turso database daemon (port 8080) and environment service API (port 8901) for complete local development setup

## Breaking Changes

- [List any breaking changes]

## Bug Fixes

- [List bug fixes]

## Documentation

- [List documentation updates]

---

# 0.2.2.dev1 - 2025-08-07

## Highlights

- New backend balance APIs and CLI for quick account visibility (USD balance + token/GPU spend windows)
- New CLI utilities and manual: compact, one-off commands with `uvx synth-ai <cmd>` and `man`
- Traces inventory view showing per-DB and per-system counts, plus on-disk size (GB)
- Inference and SFT API routes consolidated and documented for local and Modal deployments

## Backend (FastAPI)

- Added `GET /api/v1/balance/current` to return current org balance (cents and USD) with proper Decimal handling
- Added `GET /api/v1/balance/usage/windows?hours=24,168` to summarize token/GPU spend over the past N hours
- Improved error logging (`logger.exception`) for balance routes so stack traces appear in logs
- Normalized `DATABASE_URL` for asyncpg pool creation (strip `+psycopg`/`+asyncpg`) to prevent connection errors
- Confirmed Inference + SFT endpoints are available via learning services (v1 and v2) for chat/completions, files, fine-tuning jobs, warmup, and job status

## CLI

- Added `balance`: prints minimal balance in USD and a compact spend table for the last 24h and 7d
  - Flags: `--base-url`, `--api-key`, `--usage`; sources `.env` automatically; guards against Modal URLs for account endpoints
- Added `traces`: lists local trace DBs under `./synth_ai.db/dbs`, shows traces, experiments, last activity, and size (GB), plus aggregated per-system counts
- Added `man`: human-friendly command reference with options, env vars, and examples
# Release \e[0;34m[INFO]\e[0m Current version: 0.2.1.dev1
- Standardized one-off usage: `uvx synth-ai <command>` (removed legacy interactive `watch`)
- Improved `.env` loading and API key resolution (`SYNTH_BACKEND_API_KEY` → `SYNTH_API_KEY` → `DEFAULT_DEV_API_KEY`)
- Existing commands remain available: `experiments`, `experiment <id>`, `usage [--model]`, `status`, `calc`, and `env` (list/register/unregister)

## Demo

- Local end-to-end demo: start backend (`uv run uvicorn app.services.main:app --reload --port 8000`), set `SYNTH_BACKEND_API_KEY`, then:
  - `uvx . balance` → shows USD balance + 24h/7d spend
  - `uvx . traces` → inventories DBs and per-system counts with storage footprint
  - `uvx . experiments` and `uvx . experiment <id>` → explore local trace data

## Breaking Changes

- Removed `watch` (interactive TUI) in favor of one-off CLI commands

## Notes

- Publish a new package release to enable `uvx synth-ai man` and other commands without `.` prefix.

---

# Release v[0;34m[INFO][0m Current version: 0.2.1.dev0
[0;34m[INFO][0m Bumping version type: dev
[0;34m[INFO][0m New version: 0.2.1.dev1
[0;32m[SUCCESS][0m Version bumped to 0.2.1.dev1
0.2.1.dev1 - 2025-07-29

## What's New

- [Add your changes here]

## Breaking Changes

- [List any breaking changes]

## Bug Fixes

- [List bug fixes]

## Documentation

- [List documentation updates]

---
