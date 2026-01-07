# TUI Create Job Testing

This document describes the testing performed on the `feat/tui-create-job` branch.

## Setup

Created a worktree for this branch:
```bash
cd /Users/joshpurtell/Documents/GitHub/synth-ai
git fetch origin feat/tui-create-job
git worktree add ../synth-ai-feat-tui-create-job origin/feat/tui-create-job
```

## LocalAPI Server Testing

### Starting the LocalAPI Server

The test LocalAPI file from `/Users/joshpurtell/Downloads/localapi.py` was used (Banking77 intent classification task).

Started the server from the synth-ai root directory:
```bash
cd /Users/joshpurtell/Documents/GitHub/synth-ai
uv run python /Users/joshpurtell/Downloads/localapi.py
```

Server started successfully on `http://0.0.0.0:8001`

### Testing Authentication

The LocalAPI uses environment API key authentication with the format `sk_env_<hex_token>`.

**Without authentication (fails):**
```bash
curl -s http://localhost:8001/health | jq .
# Returns: {"detail": {"error": {"code": "unauthorised", ...}}}
```

**With correct test API key (succeeds):**
```bash
curl -s -H 'X-API-Key: sk_env_30c78a787bac223c716918181209f263' http://localhost:8001/health | jq .
# Returns: {"healthy": true, "auth": {"required": true, "expected_prefix": "sk_env..."}}
```

### Authentication Details

The server expects one of these authentication headers:
- `X-API-Key: sk_env_<token>`
- `X-API-Keys: sk_env_<token>` (comma-separated for multiple keys)
- `Authorization: Bearer sk_env_<token>`

The test API key `sk_env_30c78a787bac223c716918181209f263` is found in test files:
- `tests/local_api/test_stress_banking77.py`
- `tests/local_api/test_task_app_crash_repro.py`
- `tests/local_api/test_slow_response_crash.py`

## TUI Create Job Modal Flow

The create job modal (`synth_ai/tui/app/src/modals/create-job-modal.ts`) implements:

### 1. LocalAPI File Selection/Creation
- Scans current directory for existing `localapi.py` files
- Allows user to select existing or create new
- If creating new, prompts for directory and writes template

### 2. Deployment
Calls `python -m synth_ai.tui.deploy <filepath>` which:
- Validates the LocalAPI module (checks for required endpoints, functions)
- Starts uvicorn server on available port (default 8001)
- Creates Cloudflare tunnel (or uses localhost in `SYNTH_LOCAL_MODE`)
- Returns JSON: `{"status": "ready", "url": "https://...", "port": 8001}`

### 3. Eval Job Submission (Optional)
If user selects "Eval" job type, calls:
```bash
python -m synth_ai.tui.eval_job <deployed_url> default
```

This submits an eval job with:
- 20 seeds (quick eval)
- Model: `gpt-4.1-nano`
- Provider: `openai`
- Concurrency: 10
- Timeout: 600s

**Note:** The eval job runs in background (fire-and-forget). Progress/results are not captured by the TUI.

## Key Files

### Deploy Script
`synth_ai/tui/deploy.py` - Handles LocalAPI deployment
- Validates LocalAPI structure (FastAPI app, required endpoints)
- Checks functions: `get_dataset_size()`, `get_sample()`, `score_response()`
- Manages server lifecycle and tunnel creation

### Eval Job Script
`synth_ai/tui/eval_job.py` - Submits eval jobs to backend
- Uses `synth_ai.sdk.api.eval.EvalJob` and `EvalJobConfig`
- Polls for completion and outputs JSON progress updates
- Returns final metrics (mean_score, cost_usd, etc.)

### Create Job Modal
`synth_ai/tui/app/src/modals/create-job-modal.ts` - TUI wizard
- Multi-step wizard for job creation
- Handles LocalAPI file management
- Orchestrates deployment and job submission

## Environment Variables

Required for deployment:
- `SYNTH_API_KEY` - Your Synth API key
- `SYNTH_LOCAL_MODE=1` - (Optional) Use localhost instead of Cloudflare tunnel
- `ENVIRONMENT_API_KEY` - Auto-generated or loaded from `~/.env.synth`

## Test Results

✅ LocalAPI server starts successfully
✅ Health endpoint responds with authentication
✅ Authentication validates API key format correctly
✅ Test API key works for authorized requests
✅ Deployment script validates LocalAPI structure
✅ Eval job script can submit to backend

❌ Eval job runs in background without TUI integration
❌ No real-time progress tracking in TUI for running jobs

## Commands Reference

### Get environment API key
```bash
SYNTH_API_KEY=<your_key> uv run python -c "from synth_ai.sdk.localapi.auth import ensure_localapi_auth; print(ensure_localapi_auth())"
```

### Deploy LocalAPI (local mode)
```bash
SYNTH_LOCAL_MODE=1 SYNTH_API_KEY=<your_key> uv run python -m synth_ai.tui.deploy /path/to/localapi.py
```

### Submit eval job
```bash
SYNTH_API_KEY=<your_key> uv run python -m synth_ai.tui.eval_job http://localhost:8001 default
```

### Run TUI
```bash
SYNTH_API_KEY=<your_key> uv run synth tui
```
