

Crafter

cd /Users/joshpurtell/Documents/GitHub/synth-ai && uvx synth-ai modal-serve grpo-crafter-task-app --name grpo-crafter-task-app --env-file /Users/joshpurtell/Documents/GitHub/monorepo/environments/crafter/.env

cd /Users/joshpurtell/Documents/GitHub/monorepo && uv run modal deploy backend/app/routes/clustered_training/core/algorithms/gspo/app.py --env dev

uvx synth-ai train \
  --type rl \
  --config /Users/joshpurtell/Documents/GitHub/synth-ai/examples/multi_step/configs/crafter_rl_stepwise_hosted_judge.toml \
  --task-url https://synth-laboratories--grpo-crafter-task-app-fastapi-app-dev.modal.run \
  --backend https://synth-backend-dev-docker.onrender.com/api \
  --env-file /Users/joshpurtell/Documents/GitHub/monorepo/environments/crafter/.env


---

Verilog

# 1. Deploy Verilog task app
cd /Users/joshpurtell/Documents/GitHub/synth-ai && uvx synth-ai modal-serve grpo-verilog --name grpo-verilog-task-app --env-file /Users/joshpurtell/Documents/GitHub/monorepo/environments/verilog/.env

# 2. Test with quick eval (update URL in config first!)
uvx synth-ai eval --config /Users/joshpurtell/Documents/GitHub/synth-ai/examples/multi_step/configs/verilog_eval_groq_qwen32b.toml

# 3. Deploy training backend
cd /Users/joshpurtell/Documents/GitHub/monorepo && uv run modal deploy backend/app/routes/clustered_training/core/algorithms/gspo/app.py --env dev

# 4. Run RL training
uvx synth-ai train \
  --type rl \
  --config /Users/joshpurtell/Documents/GitHub/synth-ai/examples/multi_step/configs/verilog_rl_lora.toml \
  --task-url https://synth-laboratories--grpo-verilog-task-app-fastapi-app-dev.modal.run \
  --backend https://synth-backend-dev-docker.onrender.com/api \
  --env-file /Users/joshpurtell/Documents/GitHub/monorepo/environments/verilog/.env
