RL Example (provider-agnostic)

Prereqs
- Set `PROD_BACKEND_URL` and `SYNTH_API_KEY` (or pass via flags)
- Deploy the task app (`examples/rl/task_app.py`) to your Modal account and set `TASK_APP_BASE_URL`
  - SDK app name: `grpo-task-service-sdk`
  - SDK secret name: `crafter-environment-sdk`
- Ensure your organization has an uploaded `ENVIRONMENT_API_KEY` (see below). The backend decrypts and injects it at trainer start.
- Optionally edit `examples/rl/crafter_online.toml` for model/batch/group and runner defaults; scripts default to this path when `--config-path` is not provided

Working prod flow (tested)
```bash
# 0) Load env and set prod URLs/keys
set -a; source /Users/you/Documents/GitHub/synth-ai/.env; set +a
export SYNTH_BACKEND_URL_OVERRIDE=prod && export PROD_BACKEND_URL=https://agent-learning.onrender.com/api
export SYNTH_API_KEY="${PROD_SYNTH_API_KEY:-$SYNTH_API_KEY}"
export ENVIRONMENT_API_KEY="${PROD_ENVIRONMENT_API_KEY:-$ENVIRONMENT_API_KEY}"
export OPENAI_API_KEY="${PROD_OPENAI_API_KEY:-$OPENAI_API_KEY}"

# 1) Upload org ENVIRONMENT_API_KEY to backend (sealed-box)
uv run python -c "import os; from synth_ai.rl.env_keys import setup_environment_api_key as s; r=s('https://agent-learning.onrender.com', os.environ['SYNTH_API_KEY']); print('uploaded');"

# 2) Deploy the Task App (Modal) that hosts the env rollout endpoints
bash /Users/you/Documents/GitHub/synth-ai/examples/rl/deploy_task_app.sh

# 3) Sanity check backend + task app health
uv run python /Users/you/Documents/GitHub/synth-ai/examples/rl/check.py \
  --backend-url "$PROD_BACKEND_URL" --api-key "$SYNTH_API_KEY" \
  --task-app-url "$TASK_APP_BASE_URL"

# 4) Optional: OpenAI-direct rollout workflow (backend orchestrates task app → OpenAI)
uv run python /Users/you/Documents/GitHub/synth-ai/examples/rl/openai_in_task_app.py \
  --mode prod --backend-url https://agent-learning.onrender.com/api \
  --task-app-url "$TASK_APP_BASE_URL" \
  --model gpt-5-nano --num-rollouts 2 --max-steps-each 10 --timeout-seconds 200

# 5) Start clustered RL job (Qwen baseline)
uv run python /Users/you/Documents/GitHub/synth-ai/examples/rl/run_rl_job.py \
  --backend-url https://agent-learning.onrender.com/api \
  --api-key "$SYNTH_API_KEY" \
  --task-app-url "$TASK_APP_BASE_URL" \
  --model "Qwen/Qwen3-0.6B" \
  --batch-size 2 --group-size 4 \
  --stream-seconds 10 --timeout 200

# 6) Use the trained checkpoint for inference (replace RL_JOB_ID)
export RL_JOB_ID='PASTE_JOB_ID_FROM_PREVIOUS_STEP'
uv run python /Users/you/Documents/GitHub/synth-ai/examples/rl/hello_rl_completion.py \
  --backend-url "$PROD_BACKEND_URL" --api-key "$SYNTH_API_KEY" \
  --model "rl:Qwen-Qwen3-0.6B:$RL_JOB_ID:checkpoint-epoch-1" --timeout 180
```

Setting ENVIRONMENT_API_KEY (secure upload)
- Use the SDK helper to mint, encrypt (sealed box), and upload the token to the backend. The helper prints the token once; store it securely. The backend persists only ciphertext and injects the token at trainer start.
```python
import os
from synth_ai.rl.env_keys import setup_environment_api_key

setup_environment_api_key(
    backend_base="https://agent-learning.onrender.com",
    synth_api_key=os.environ["SYNTH_API_KEY"],
)
# The helper prints the token once to stdout. Keep it safe; it is not retrievable later.
```
CLI one-liner (optional):
```bash
DEV_BACKEND_URL=https://agent-learning.onrender.com \
SYNTH_API_KEY=sk_your_org_key \
uv run python - <<'PY'
import os
from synth_ai.rl.env_keys import setup_environment_api_key
setup_environment_api_key(os.environ["DEV_BACKEND_URL"], os.environ["SYNTH_API_KEY"])
PY
```
Notes
- Do not send `ENVIRONMENT_API_KEY` from public clients. The backend decrypts and injects it into trainer containers.
- Endpoints used: `GET /api/v1/crypto/public-key`, `POST /api/v1/env-keys`.
- Details: see `examples/rl/env_api_key_crypto.txt` and `examples/rl/env_api_crypto_plan.txt`.

Notes
- Trainer endpoints are resolved server-side via trainer_id; no provider URLs in the SDK/example.
- Status, events, and metrics use learning/* endpoints; SSE uses rl/ or learning/ where available.
- For health-only validation: run with the above flags; the script prints backend/task_app health before creating the job.
- Task App auth uses ENVIRONMENT_API_KEY; pass it via Modal secret and use X-API-Key on /health/rollout.

Resources
- See `examples/rl/crafter_online.toml` for an example multi-GPU layout:
  - 8x H100 total
  - 6 GPUs for inference (tensor-parallel) and 2 for training
- The SDK/examples don’t allocate GPUs directly; resource placement is resolved by the backend/trainer.

Local config defaults and precedence
- Scripts read `examples/rl/config.toml` by default (override with `--config-path`).
- Modest smoke-test settings:
  - `[trainer]` `batch_size=2`, `group_size=4`
  - `[env]` `max_steps_per_episode=3`
  - `[job]` `model=...`
  - `[runner]` `stream_seconds`, `empty_polls_threshold`, `startup_deadline_s`
- Precedence: CLI flags > config file > built-in defaults

Other scripts
- check.py: basic diagnostics
```bash
python examples/rl/check.py --backend-url "$PROD_BACKEND_URL" --api-key "$SYNTH_API_KEY" --task-app-url "$TASK_APP_BASE_URL"
```

- openai_in_task_app.py: Task App calls OpenAI directly
```bash
uv run python examples/rl/openai_in_task_app.py --model gpt-5-nano --num-rollouts 2 --max-steps-each 7 --timeout-seconds 1200
```

- full_training.py: end-to-end clustered training
```bash
python examples/rl/full_training.py --backend-url "$PROD_BACKEND_URL" --api-key "$SYNTH_API_KEY" --task-app-url "$TASK_APP_BASE_URL"  --model Qwen/Qwen3-0.6B --stream-seconds 10
```

- inference iwth rl'ed models
```
 RL_WEIGHTS_PATH="models/Qwen-Qwen3-0.6B/rl-job_1993769d63c7506d485/checkpoint-epoch-1.tar.gz" uv run python examples/rl/hello_rl_completion.py --model "rl:Qwen-Qwen3-0 6B:job_1993769d63c7506d485:checkpoint-epoch-1"
```
