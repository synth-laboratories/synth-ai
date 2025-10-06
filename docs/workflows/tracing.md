# Tracing & Dataset Export

Tracing v3 captures every interaction between policies, environments, and reward hooks. Use it to build SFT datasets or debug rollouts.

## Enabling tracing

- **Serve locally**: `uvx synth-ai serve grpo-crafter --trace traces/v3 --trace-db traces/v3/synth_ai.db` sets `TASKAPP_TRACING_ENABLED=1`, writes JSONL SFT snippets, and points the task app at a dedicated sqlite database (`synth_ai/cli/task_apps.py:189`).
- **Ad-hoc scripts**: `examples/warming_up_to_rl/run_local_rollout_traced.py` explains the same environment variables (`TASKAPP_TRACING_ENABLED`, `TASKAPP_SFT_OUTPUT_DIR`, `SQLD_DB_PATH`) for scripted rollouts (`examples/warming_up_to_rl/run_local_rollout_traced.py:350`).

Traces are stored under `traces/v3/<db-name>/data` by default. Use `uvx synth-ai traces` to inspect the databases and per-system counts (`synth_ai/cli/traces.py:16`).

## Exporting SFT datasets

Once you accumulate traces, convert them to JSONL with the exporter:

```bash
python examples/warming_up_to_rl/export_trace_sft.py \
  --db traces/v3/synth_ai.db \
  --min-achievements 3 \
  --output ft_data/crafter_traces.jsonl
```

`export_trace_sft.py` pulls:
- **Event rewards**: incremental `unique_achievement_delta` and `achievement_delta` rows (`fetch_achievement_data`).
- **Outcome rewards**: final achievement lists (`fetch_outcome_rewards`).
- **Model metadata**: provider/model usage per session to segment datasets.

The script emits JSONL ready for FFT/QLoRA jobs (pair with `uvx synth-ai train --type sft`).

## Dataset best practices

- Filter sessions by minimum achievements or reward thresholds to avoid noisy trajectories.
- Preserve `session_id` in the JSONL to keep traceability back to the sqlite DB.
- Version your datasets alongside the task app so retraining is reproducible.

