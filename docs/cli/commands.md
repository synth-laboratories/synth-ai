# CLI Command Guide

Use `uvx synth-ai …` to run the CLI without installing the package globally. All commands are Click subcommands defined in `synth_ai/cli` and `synth_ai/api/train`.

## Common Flags

Most commands honour the following environment helpers:

- `.env` discovery: `uvx synth-ai setup` writes keys to `.env` files. `deploy` and `train` load those files automatically (`synth_ai/cli/commands/deploy/core.py`, `synth_ai/api/train/cli.py:159`).
- `SYNTH_API_KEY`: backend authentication for SFT/RL jobs.
- `ENVIRONMENT_API_KEY`: task app authentication; forwarded as `X-API-Key`.

## `uvx synth-ai deploy`

The consolidated entrypoint for task apps. Use `--runtime` to choose between local uvicorn and Modal execution. Omitting `APP_ID` triggers discovery across registered task apps, demos, and downstream configs.

### Local development (uvicorn runtime)

```bash
# Interactive discovery
uvx synth-ai deploy --runtime uvicorn --port 8001 --env-file examples/warming_up_to_rl/.env

# Explicit app id with tracing enabled
uvx synth-ai deploy --runtime uvicorn grpo-crafter \
  --port 8001 \
  --env-file examples/warming_up_to_rl/.env \
  --trace traces/v3 \
  --trace-db traces/v3/task_app_traces.sqlite
```

Key uvicorn flags (same as the legacy `serve` command):
- `app_id` (optional) – skip discovery and run a specific app (`grpo-crafter`, `math-single-step`, …).
- `--host` (default `0.0.0.0`).
- `--port` (default `8001`).
- `--env-file PATH` (repeatable) – additional env files to load.
- `--reload/--no-reload` – uvicorn autoreload (default off).
- `--force` – kill processes already bound to the chosen port.
- `--trace DIR` / `--trace-db PATH` – enable tracing outputs.

### Modal preview (`--modal-mode serve`)

```bash
uvx synth-ai deploy \
  --runtime modal \
  --modal-mode serve \
  --env-file examples/warming_up_to_rl/.env
```

This proxies the Modal CLI to spin up a temporary container without performing a full deploy. Flags mirror the uvicorn runtime plus:
- `--modal-cli PATH` – alternate Modal binary (defaults to `modal` on PATH).
- `--name` – override the Modal app name.
- `--env-file` – secrets to mount inside the container (required if they are not registered with Modal).
- `--dry-run` – print the generated Modal command (only valid when `--modal-mode deploy`).

### Modal deployment (`--modal-mode deploy`, default)

```bash
uvx synth-ai deploy grpo-crafter --runtime modal --name grpo-crafter-task-app --env-file examples/warming_up_to_rl/.env
```

The command packages the task app, uploads inline secrets, and invokes `modal deploy`. It preflights `ENVIRONMENT_API_KEY` by encrypting the value with Synth’s backend key when both `SYNTH_API_KEY` and the env key are present.

> **Note:** The legacy `uvx synth-ai serve` and `uvx synth-ai modal-serve` shims still exist for backwards compatibility but simply call `deploy --runtime uvicorn` and `deploy --runtime modal --modal-mode serve` under the hood. Prefer the new flags in scripts and documentation.

## `uvx synth-ai setup`

Alias of the RL demo setup routine (`synth_ai/cli/root.py:146`). It guides you through the dashboard handshake, writes `SYNTH_API_KEY` and `ENVIRONMENT_API_KEY` to your repo `.env`, and prepares the math demo config.

```bash
uvx synth-ai setup
```

## `uvx synth-ai turso`

Verify the Turso `sqld` binary needed for tracing v3 and install it through the Turso CLI, bootstrapping that CLI with Homebrew when available (`synth_ai/cli/turso.py`). If `brew` is missing, the command prints manual instructions (e.g. install the Turso CLI via `brew install tursodatabase/tap/turso` or `curl -sSfL https://get.tur.so/install.sh | bash`, then run `turso dev` once to download `sqld`).

```bash
# Check for sqld and install ~/.local/bin/sqld when absent
uvx synth-ai turso

# Re-run after installing manually to verify detection
uvx synth-ai turso
```

## `uvx synth-ai train`

Interactive launcher for SFT (FFT, QLoRA) and RL jobs (`synth_ai/api/train/cli.py:95`).

### Quick start

```bash
# Launch RL job from a TOML config
uvx synth-ai train --config examples/warming_up_to_rl/configs/rl_from_base_qwen4b.toml --type rl

# Submit an FFT job with manual dataset selection
uvx synth-ai train --config examples/warming_up_to_rl/configs/crafter_fft.toml --type sft --dataset /path/to/data.jsonl
```

During execution the command:
1. Discovers candidate `.env` files and prompts for selection.
2. Ensures required keys are present (`SYNTH_API_KEY`, `ENVIRONMENT_API_KEY`, `TASK_APP_URL` for RL).
3. Validates /health and /task_info on the task app (RL only).
4. Builds the job payload via `build_rl_payload` or `build_sft_payload`.
5. Uploads datasets when training SFT.
6. Polls job status until completion unless `--no-poll` is supplied.

Notable flags:
- `--type {auto,rl,sft}` – override the detected config type.
- `--task-url` – supply task app URL instead of reading from TOML.
- `--backend` – override backend base URL (`BACKEND_BASE_URL` env fallback).
- `--model` – change the fine-tune or RL starting model.
- `--idempotency` – extra safety for RL job creation.
- `--dry-run` – preview the payload without submitting.
- `--examples N` – limit SFT dataset rows (helpful for smoke tests).

### QLoRA support

Pass `training.use_qlora = true` in the TOML (see `synth_ai/api/train/builders.py:112`) to trigger QLoRA-friendly payloads. Pair with smaller base models when extending the Crafter FFT example.

## Math Demo (`uvx synth-ai demo` and `uvx synth-ai run`)

The math demo CLI wraps the same task app infrastructure with curated prompts (`synth_ai/cli/root.py:75`, `synth_ai/cli/rl_demo.py:13`).

```bash
# Prepare secrets via the demo group (alias of setup)
uvx synth-ai demo setup

# Deploy the math task app to Modal
uvx synth-ai demo deploy --name synth-math-demo

# Kick off the curated RL job
uvx synth-ai demo run --batch-size 4 --group-size 16 --model Qwen/Qwen3-0.6B

# Alternate entry: top-level run alias
uvx synth-ai run --config demo_config.toml
```
# Bootstrap a demo template (math modal or crafter local)
uvx synth-ai demo init

# Deploy the math task app to Modal
uvx synth-ai demo deploy --name synth-math-demo

# Kick off the curated RL job
uvx synth-ai demo run --batch-size 4 --group-size 16 --model Qwen/Qwen3-0.6B

# Alternate entry: top-level run alias
uvx synth-ai run --config demo_config.toml
```

The demo commands invoke helpers from `synth_ai.demos.core.cli` directly, which continue to print the next recommended step after each action (e.g., `uvx synth-ai run` once deployment is complete).
