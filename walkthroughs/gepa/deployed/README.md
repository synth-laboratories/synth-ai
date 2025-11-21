# GEPA Deployment Walkthrough

This directory contains an interactive script to run GEPA optimization on Banking77 using a deployed task app via Cloudflare Tunnel.

## Overview

This walkthrough demonstrates how to:
1. Generate and register an `ENVIRONMENT_API_KEY` with the backend
2. Deploy the Banking77 task app via Cloudflare Tunnel
3. Configure and run GEPA optimization against the production backend
4. Retrieve optimized prompts and results

## Prerequisites

- `SYNTH_API_KEY` in `.env` (for backend authentication)
- `uv` installed (for running Python commands)
- `cloudflared` binary (will be auto-installed if missing)

## Usage

Run the interactive script:

```bash
cd walkthroughs/gepa/deployed
bash commands.sh
```

The script will:
- Prompt you at each step with a description of what it's doing
- Wait for your confirmation before proceeding
- Show you the commands it's executing
- Display results and next steps

## What It Does

1. **Step 1**: Generates `ENVIRONMENT_API_KEY` using synth-ai's minting function and registers it with the backend
2. **Step 2**: Kills any existing processes on port 8102, then deploys the Banking77 task app via Cloudflare Tunnel
3. **Step 3**: Extracts the tunnel URL from the environment file
4. **Step 4**: Creates a GEPA config file with the tunnel URL and increased rollout budget (2000)

## Output

After completion, you'll get:
- Job ID for tracking
- Best score achieved
- Results file location
- Optimized prompts

## Files Created

- `/tmp/gepa_walkthrough/cli_env.txt` - Environment file with API key and tunnel URL
- `/tmp/gepa_walkthrough/banking77_gepa_prod.toml` - GEPA config with tunnel URL
- `/tmp/gepa_walkthrough/results/` - Results directory with logs and outputs

## Troubleshooting

- **Port 8102 in use**: The script automatically kills existing processes, but if issues persist, manually kill them: `lsof -ti :8102 | xargs kill -9`
- **Tunnel fails**: Check that `cloudflared` is installed and network connectivity is working
- **API key errors**: Ensure `SYNTH_API_KEY` is set in your `.env` file

