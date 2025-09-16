## Synth-Qwen v1 Finetuning Demo (Qwen3 0.6B)

Prereqs
- Python 3.11+ and uv installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Local Env Service is provided by this repo; no `sqld` required
- One of the following ways to provide backend creds:
  - Set `MONOREPO_BACKEND` to your monorepo backend path (defaults to `../monorepo/backend`) and ensure it has `.env.dev` with at least:
    - `DEV_BACKEND_URL` (e.g., `http://localhost:8000`)
    - `TESTING_LOCAL_SYNTH_API_KEY` (or `SYNTH_API_KEY`)
  - OR export these directly in your shell before running:
    - `LOCAL_BACKEND_URL` (e.g., `http://localhost:8000/api`)
    - `SYNTH_API_KEY` (local dev key)
  - Optional for prod: `.env` in repo root with
    - `PROD_BACKEND_URL=https://agent-learning.onrender.com`
    - `TESTING_PROD_SYNTH_API_KEY=...`

Steps
```bash
# 0) Go to repo root so traces and logs land in the right place
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
# Note: commands below resolve backend URL per-call using examples/common/backend.py

# 1) Start local services (sqld + Env Service) in background
uvx synth-ai serve --no-sqld --env-port 8901

# 3) Rollout base Qwen to generate v3 traces (Crafter via Env Service)
set -a; MONOREPO_BACKEND=${MONOREPO_BACKEND:-../monorepo/backend}; source "$MONOREPO_BACKEND/.env.dev"; set +a; export SYNTH_BASE_URL="$(uv run python -c 'from examples.common.backend import resolve_backend_url;print(resolve_backend_url())')"; export SYNTH_API_KEY="${DEV_SYNTH_API_KEY:-${SYNTH_API_KEY:-${SYNTH_API_KEY_TEST:-sk-local}}}"; uv run python examples/finetuning/synth_qwen/react_agent_lm.py --model "Qwen/Qwen3-0.6B" --episodes 10 --max-steps 10 --quiet --no-daemon

# 4) Convert traces → SFT JSONL (writes training.jsonl) [use single-script alternative below]
printf "[filter]\nrequired_achievements=[]\n" > /tmp/crater_filter.toml && CRAFTER_DB_URL=sqlite+aiosqlite:///$PWD/traces/v3/synth_ai.db CRAFTER_CONFIG=/tmp/crater_filter.toml WINDOW_MODE=1 MIN_TOTAL_REWARD=1 MIN_ACHIEVEMENTS=0 OUTPUT_JSONL=$PWD/examples/finetuning/synth_qwen_v1/data/training_crafter.jsonl uv run python examples/finetuning/synth_qwen/filter_traces_achievements.py

# ALT: Single-script E2E run (prepare → upload → create/start → poll → infer)
set -a; MONOREPO_BACKEND=${MONOREPO_BACKEND:-../monorepo/backend}; source "$MONOREPO_BACKEND/.env.dev"; set +a; SYNTH_BACKEND_URL_OVERRIDE=prod DEV_BACKEND_URL="$(uv run python -c 'from examples.common.backend import resolve_backend_url;print(resolve_backend_url())')" uv run python examples/finetuning/synth_qwen_v1/run_ft_job.py --mode dev

# Test model
set -a; MONOREPO_BACKEND=${MONOREPO_BACKEND:-../monorepo/backend}; source "$MONOREPO_BACKEND/.env.dev"; set +a; MODE=dev DEV_BACKEND_URL="$(uv run python -c 'from examples.common.backend import resolve_backend_url;print(resolve_backend_url())')" uv run python examples/finetuning/synth_qwen_v1/hello_ft_model.py | cat

# 8) Rollout agent again using the fine-tuned model from state.json (env service already on 8901, no sqld)
set -a; MONOREPO_BACKEND=${MONOREPO_BACKEND:-../monorepo/backend}; source "$MONOREPO_BACKEND/.env.dev"; set +a; FT_MODEL=$(uv run python - <<'PY'
import json, os
print(json.load(open(os.path.join(os.getcwd(),'examples/finetuning/synth_qwen_v1/state.json')))['fine_tuned_model'])
PY
); SYNTH_BACKEND_URL_OVERRIDE=prod SYNTH_BASE_URL="$(uv run python -c 'from examples.common.backend import resolve_backend_url;print(resolve_backend_url())')" SYNTH_API_KEY=${TESTING_LOCAL_SYNTH_API_KEY:-${SYNTH_API_KEY:-sk-local}} uv run python examples/finetuning/synth_qwen/react_agent_lm.py --model "$FT_MODEL" --episodes 10 --max-steps 10 --quiet --no-daemon --no-traces
```











export LOCAL_BACKEND_URL=http://localhost:8000/api
export SYNTH_BACKEND_URL_OVERRIDE=local
uv run python examples/finetuning/synth_qwen_v1/run_ft_job.py --mode local

HATCHET_ENV_OVERRIDE=prod python -u -m app.orchestration.hatchet.workflows

export LOCAL_BACKEND_URL=http://localhost:8000/api                       
export SYNTH_BACKEND_URL_OVERRIDE=dev  
uv run python examples/finetuning/synth_qwen_v1/run_ft_job.py --mode dev

export PROD_BACKEND_URL=https://agent-learning.onrender.com/api
export SYNTH_BACKEND_URL_OVERRIDE=prod
uv run python examples/finetuning/synth_qwen_v1/run_ft_job.py --mode prod