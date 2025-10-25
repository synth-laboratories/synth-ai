

Crafter

cd /Users/joshpurtell/Documents/GitHub/synth-ai && uvx synth-ai modal-serve grpo-crafter-task-app --name grpo-crafter-task-app --env-file /Users/joshpurtell/Documents/GitHub/monorepo/environments/crafter/.env

cd /Users/joshpurtell/Documents/GitHub/monorepo && uv run modal deploy backend/app/routes/clustered_training/core/algorithms/gspo/app.py --env dev

uvx synth-ai eval --config /Users/joshpurtell/Documents/GitHub/synth-ai/examples/multi_step/configs/crafter_eval_text_only_groq_qwen32b.toml


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

# 2. Baseline eval using Synth backend (pre-training)
uvx synth-ai eval --config /Users/joshpurtell/Documents/GitHub/synth-ai/examples/multi_step/configs/verilog_eval_synth_qwen4b.toml

# 3. (Optional) External reference eval using Groq Qwen 32B
uvx synth-ai eval --config /Users/joshpurtell/Documents/GitHub/synth-ai/examples/multi_step/configs/verilog_eval_groq_qwen32b.toml

# 4. Deploy training backend
cd /Users/joshpurtell/Documents/GitHub/monorepo && uv run modal deploy backend/app/routes/clustered_training/core/algorithms/gspo/app.py --env dev

# 5. Run RL training
uvx synth-ai train \
  --type rl \
  --config /Users/joshpurtell/Documents/GitHub/synth-ai/examples/multi_step/configs/verilog_rl_lora.toml \
  --task-url https://synth-laboratories--grpo-verilog-task-app-fastapi-app-dev.modal.run \
  --backend https://synth-backend-dev-docker.onrender.com/api \
  --env-file /Users/joshpurtell/Documents/GitHub/monorepo/environments/verilog/.env

# 6. Post-training eval (update job_id in config first!)
# After training, note the job_id from logs (e.g., job_19a1823e56303de604f)
# Update verilog_eval_synth_trained_qwen8b.toml with your job_id
uvx synth-ai eval --config /Users/joshpurtell/Documents/GitHub/synth-ai/examples/multi_step/configs/verilog_eval_synth_trained_qwen8b.toml
