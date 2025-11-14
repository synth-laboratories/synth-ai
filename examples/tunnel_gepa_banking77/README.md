# Banking77 GEPA Optimization via Cloudflare Tunnel

This example demonstrates how to use Cloudflare Tunnel to expose a local Banking77 task app to Synth's production backend for GEPA prompt optimization.

## Overview

Instead of deploying to Modal or running a local backend, this example:
1. Deploys the Banking77 task app locally
2. Exposes it via Cloudflare Tunnel (free quick tunnels)
3. Runs GEPA optimization against Synth's production backend

## Prerequisites

- `cloudflared` installed ([install guide](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/local-management/create-local-tunnel/))
- `SYNTH_API_KEY` set (get from [Synth Dashboard](https://app.usesynth.ai/api-keys))
- `ENVIRONMENT_API_KEY` set (or auto-generated)
- `GROQ_API_KEY` set (optional, for LLM-guided mutations)

## Quick Start

```bash
# Run the example
./examples/tunnel_gepa_banking77/run_gepa_with_tunnel.sh
```

The script will:
1. ✅ Check prerequisites
2. 🌐 Deploy Banking77 via Cloudflare Tunnel (quick mode)
3. 📝 Create GEPA config with tunnel URL
4. 🎯 Run GEPA optimization against prod backend
5. 📊 Show results

## What Happens

1. **Tunnel Deployment**: Creates an ephemeral `*.trycloudflare.com` URL
2. **Config Generation**: Creates `banking77_gepa_tunnel.toml` with the tunnel URL
3. **GEPA Training**: Runs prompt optimization using Synth's production backend
4. **Results**: View results in the Synth dashboard

## Manual Steps

If you prefer to run steps manually:

### Step 1: Deploy Tunnel

```bash
uvx synth-ai deploy \
    --task-app examples/task_apps/banking77/banking77_task_app.py \
    --runtime tunnel \
    --tunnel-mode quick \
    --port 8102 \
    --env .env.tunnel
```

This will write `TASK_APP_URL` to `.env.tunnel`.

### Step 2: Run GEPA

```bash
export BACKEND_BASE_URL="https://api.usesynth.ai"
export TASK_APP_URL=$(grep TASK_APP_URL .env.tunnel | cut -d'=' -f2)

uvx synth-ai train \
    --type prompt_learning \
    --config examples/tunnel_gepa_banking77/banking77_gepa_tunnel.toml \
    --backend https://api.usesynth.ai \
    --poll
```

## Configuration

The generated config (`banking77_gepa_tunnel.toml`) includes:
- **Task App URL**: Cloudflare Tunnel URL (e.g., `https://abc123.trycloudflare.com`)
- **Backend**: Production backend (`https://api.usesynth.ai`)
- **GEPA Settings**: Standard Banking77 optimization parameters

## Notes

- **Quick Tunnels**: Ephemeral, free, no account needed
- **Tunnel Lifetime**: Tunnel closes when deployment process stops
- **Production Backend**: Uses Synth's hosted backend (no local setup needed)
- **Credentials**: Saved to `.env.tunnel` for reuse

## Troubleshooting

### Tunnel not accessible
- Check `cloudflared` is installed: `which cloudflared`
- Verify tunnel process is running
- Check firewall/network settings

### Backend connection failed
- Verify `SYNTH_API_KEY` is set correctly
- Check network connectivity to `https://api.usesynth.ai`
- Ensure API key has proper permissions

### Task app health check fails
- Verify `ENVIRONMENT_API_KEY` matches the one used in deployment
- Check task app logs for errors
- Ensure task app has `/health` endpoint

## Related Examples

- Local deployment: `examples/blog_posts/gepa/deploy_banking77_task_app.sh`
- Modal deployment: `examples/blog_posts/gepa/run_gepa_banking77_pipeline.sh`
- GEPA configs: `examples/blog_posts/gepa/configs/`



