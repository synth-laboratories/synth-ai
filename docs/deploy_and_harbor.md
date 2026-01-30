# Deploy & Harbor: Cloud-Hosted LocalAPI

Run your LocalAPI in the cloud without tunnels. Deploy uploads your task app to Harbor (Synth's hosted infrastructure) and returns a stable URL you can use directly in GEPA, MiPRO, and eval jobs.

## When to Use Deploy vs Tunnels

| | `synth localapi serve` (Tunnel) | `synth localapi deploy` (Harbor) |
|---|---|---|
| **Where it runs** | Your local machine | Synth cloud (Harbor) |
| **URL lifetime** | Temporary (per session) | Persistent (until deleted) |
| **Docker required** | No | Yes |
| **Setup** | Zero config | Write a Dockerfile |
| **Scaling** | Limited by your machine | Scales with Harbor |
| **Network** | Requires tunnel (SynthTunnel or Cloudflare) | No tunnel needed |
| **Best for** | Local dev, quick experiments | Production, CI/CD, team sharing |

**Use tunnels** when you're iterating locally and want the fastest path to running a GEPA job.

**Use deploy** when you want a stable, shareable task app URL that doesn't depend on your laptop being open.

## Quick Start

### 1. Write Your LocalAPI

Create a Python file with a FastAPI app using `create_local_api`:

```python
# my_localapi.py
from synth_ai.sdk.localapi import create_local_api, LocalAPIConfig

app = create_local_api(LocalAPIConfig(
    app_id="banking77",
    name="Banking77 Classification",
    description="Classify customer queries into banking intents.",
    provide_taskset_description=provide_taskset_description,
    provide_task_instances=provide_task_instances,
    rollout=run_rollout,
))
```

Your app must expose `/health` and `/rollout` endpoints (handled automatically by `create_local_api`).

### 2. Write a Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Harbor will start the server via the entrypoint command
```

### 3. Deploy

```bash
export SYNTH_API_KEY=sk_live_...

synth localapi deploy \
  --name my-banking77 \
  --app my_localapi:app \
  --dockerfile ./Dockerfile \
  --context . \
  --wait
```

Output:

```
Deploying LocalAPI 'my-banking77'...
  Dockerfile: ./Dockerfile
  Context: .
  Entrypoint: python -m uvicorn my_localapi:app --host 0.0.0.0 --port 8000
Deployment created!
  Deployment ID: dep_abc123
  Task app URL: https://api.usesynth.ai/api/harbor/deployments/my-banking77
  Task app API key: SYNTH_API_KEY

LocalAPI is ready to accept /rollout traffic.
```

### 4. Use in a Job

Pass the `task_app_url` to your optimization or eval job:

```python
from synth_ai.sdk.optimization.policy.job import PromptLearningJob

job = PromptLearningJob.from_dict(
    config,
    task_app_url="https://api.usesynth.ai/api/harbor/deployments/my-banking77",
    task_app_api_key=os.environ["SYNTH_API_KEY"],
)
result = await job.run()
```

No tunnel setup, no local server to keep running.

## CLI Reference

### `synth localapi deploy`

```
synth localapi deploy [OPTIONS]
```

**Required:**

| Option | Description |
|--------|-------------|
| `--name` / `-n` | Deployment name (must be unique within your org) |
| `--app` or `--entrypoint` | How to start the server (pick one) |

**App vs Entrypoint:**

- `--app my_module:app` generates `python -m uvicorn my_module:app --host 0.0.0.0 --port 8000`
- `--entrypoint "python run.py"` uses a custom command

**Optional:**

| Option | Default | Description |
|--------|---------|-------------|
| `--dockerfile` / `-d` | `./Dockerfile` | Path to Dockerfile |
| `--context` / `-c` | `.` | Build context directory |
| `--port` | `8000` | Server port (used with `--app`) |
| `--timeout` | `300` | Rollout timeout in seconds (30-3600) |
| `--cpu` | `2` | CPU cores (1-8) |
| `--memory` | `4096` | Memory in MB (512-32768) |
| `--disk` | `10240` | Disk in MB (1024-102400) |
| `--env KEY=VALUE` | | Environment variables (repeatable) |
| `--wait` / `--no-wait` | `--no-wait` | Wait for build to complete |
| `--build-timeout` | `600` | Max wait for build (seconds) |
| `--json-output` | | Machine-readable JSON output |

### `synth harbor status`

Check deployment build status:

```bash
synth harbor status my-banking77 --wait
```

## Harbor Architecture

```
Job (GEPA / MiPRO / Eval)
  |
  | POST /api/harbor/deployments/{name}/rollout
  v
Harbor API (Synth backend)
  |
  | maps deployment -> Daytona snapshot
  v
Daytona Sandbox (fresh container per rollout)
  |
  | runs entrypoint, serves /rollout
  v
Your LocalAPI code
  |
  | LLM calls via inference_url
  v
Synth Interceptor (captures traces)
```

Each rollout gets a fresh container from your deployment's snapshot. Harbor manages the full sandbox lifecycle: provisioning, execution, cleanup.

## Comparison: All Connectivity Options

| Feature | SynthTunnel | Cloudflare Tunnel | Harbor Deploy |
|---------|-------------|-------------------|---------------|
| **Command** | `synth localapi serve --backend synth-tunnel` | `synth localapi serve --backend cloudflare` | `synth localapi deploy` |
| **Runs on** | Your machine | Your machine | Synth cloud |
| **URL** | `https://st.usesynth.ai/s/rt_...` | `https://<random>.trycloudflare.com` | `https://api.usesynth.ai/api/harbor/deployments/...` |
| **URL lifetime** | Session | Session | Persistent |
| **Auth param** | `task_app_worker_token` | `task_app_api_key` | `task_app_api_key` (= `SYNTH_API_KEY`) |
| **Setup** | Zero config | Requires `cloudflared` binary | Requires Dockerfile |
| **Concurrency** | Up to 128 in-flight | Cloudflare limits | Scales with Harbor |
| **Dependencies** | None | `brew install cloudflared` | Docker |
| **Network** | Outbound-only (NAT-safe) | Outbound-only | No local network needed |
| **Best for** | Local dev | Long-lived local servers | Production, CI/CD |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `SYNTH_API_KEY` | Yes | Synth API key for authentication |
| `SYNTH_BACKEND_URL` | No | Override backend URL (default: `https://api.usesynth.ai`) |

Harbor forbids passing LLM provider keys (e.g., `OPENAI_API_KEY`) via `--env`. All LLM calls go through the Synth interceptor, which injects credentials automatically.

## Troubleshooting

**Build stuck or failing**
- Check status: `synth harbor status <name>`
- Common cause: Dockerfile errors (missing dependencies, wrong base image)

**Rollouts timing out**
- Increase `--timeout` (default 300s, max 3600s)
- Check resource limits (`--cpu`, `--memory`)

**429 Rate Limited**
- Harbor enforces capacity limits per org
- Reduce job concurrency or wait and retry
