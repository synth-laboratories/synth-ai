RL Example (provider-agnostic)

Prereqs
- Set PROD_BACKEND_URL and SYNTH_API_KEY (or pass via flags)
- Deploy the task app (examples/rl/task_app.py) to your Modal account and set TASK_APP_BASE_URL
  - SDK app name: grpo-task-service-sdk
  - SDK secret name: crafter-environment-sdk
- Obtain TRAINER_ID from your backend (no provider URL exposed)
- Optionally edit examples/rl/config.toml for model/batch/group and runner defaults; scripts default to this path when --config-path is not provided

Quickstart
```bash
python examples/rl/run_rl_job.py \
  --backend-url "$PROD_BACKEND_URL" \
  --api-key "$SYNTH_API_KEY" \
  --task-app-url "$TASK_APP_BASE_URL" \
  --trainer-id "$TRAINER_ID" \
  --model "Qwen/Qwen3-0.6B" \
  --batch-size 2 \
  --group-size 4 \
  --stream-seconds 0
```

Setting ENVIRONMENT_API_KEY via SDK
- Use `synth_ai.rl.setup_environment_api_key` to mint, encrypt, and upload the token to the backend.
- The helper fetches the sealed-box public key, encrypts the token client-side, and returns the minted value so you can store it securely.
```python
from synth_ai.rl import setup_environment_api_key

result = setup_environment_api_key(
    backend_base="https://your-backend",
    synth_api_key="sk_your_org_key",
)
print("ENVIRONMENT_API_KEY:", result["token"])
```

Notes
- Trainer endpoints are resolved server-side via trainer_id; no provider URLs in the SDK/example.
- Status, events, and metrics use learning/* endpoints; SSE uses rl/ or learning/ where available.
- For health-only validation: run with the above flags; the script prints backend/task_app health before creating the job.
- Task App auth uses ENVIRONMENT_API_KEY; pass it via Modal secret and use X-API-Key on /health/rollout.

Resources
- See `examples/rl/config.toml` for an example multi-GPU layout:
  - 8x H100 total
  - 2 GPUs for inference (tensor-parallel) and 6 for training
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
