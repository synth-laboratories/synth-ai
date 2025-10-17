# mini-SWE Task App

This directory contains an example task app that exposes the
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) workflow through
the Synth AI task app interface. The goal is to provide a Crafter-like workflow
for SWE tasks: you can serve the task app locally, point RL training at it,
collect rollouts with tracing, and run vendor inference via the standard proxy
endpoints.

> **Status:** The implementation focuses on a minimal, hackable integration.
> It supports local/docker environments, step-wise command execution, tracing
> hooks, and rollouts. By default it streams SWE-Bench Verified tasks from
> Hugging Face; you can point the loader at your own dataset (or the bundled
> sample JSONL) via environment variables (see the docs at the end of this
> file).

## Layout

- `grpo_swe_mini.py` – main task-app configuration (dataset, rollout executor,
  tracing, Modal metadata, registration).
- `grpo_swe_mini_task_app.py` – backwards-compatible FastAPI wrapper that
  allows running the module directly (mirrors `grpo_crafter_task_app.py`).
- `hosted/envs/mini_swe` – environment/policy adapters that wrap `mini-swe-agent`
  inside a hosted FastAPI service.
- `data/sample_instances.json` – optional curated subset for quick smoke tests
  (no longer the default dataset).

## Using the task app

```
uvx synth-ai serve swe-mini --port 8020
```

### Recommended: non-interactive serve + .env

```bash
uvx synth-ai serve swe-mini \
  --port 8020 \
  --env-file .env \
  --trace traces/v3 \
  --trace-db traces/v3/synth_ai.db
```

This avoids interactive prompts (useful for CI) and loads `ENVIRONMENT_API_KEY`, `OPENAI_API_KEY`, etc. from `.env`.

### Configure dataset and execution

Set `SWE_MINI_DATASET` to control what tasks the environment loads (defaults to
`hf://princeton-nlp/SWE-Bench_Verified:test`):

- `file://path/to/tasks.jsonl` – each line should contain an object with
  `instance_id`, `problem_statement`, and optional docker metadata
  (`image_name`, `repo`, …).
- `hf://namespace/dataset:split` – lazily stream from Hugging Face (requires
  `datasets` and network).
  For quick local smoke tests you can point at
  `file://$REPO/examples/swe/task_app/data/sample_instances.json`.

Execution is handled by mini-swe's environment classes. Configure execution via
`SWE_MINI_ENVIRONMENT_CLASS` (`local`, `docker`, `singularity`, …) and pass
additional keyword arguments with `SWE_MINI_ENVIRONMENT_KWARGS` (JSON).

### Tracing & SFT

Tracing works the same as Crafter; pass `--trace` / `--trace-db` to the CLI or
set `TASKAPP_TRACING_ENABLED=1`. The task app writes JSONL snippets for SFT and
records decision traces in the configured SQLite/Postgres database.

## Next steps

- `docs/examples/swe/mini_swe_task_app.md` – end-to-end walkthrough
- `examples/swe/task_app/grpo_swe_mini.py` – main entrypoint
- `examples/swe/task_app/hosted` – shared host scaffolding for the Mini-SWE task app

Pull requests welcome – especially for better dataset loaders, richer metrics,
and robust docker support.

### Example rollout configs

- OpenAI gpt-4o-mini (works out-of-the-box):

```json
{
  "run_id": "example-$(date +%s)",
  "policy": {
    "policy_name": "swe-mini-react",
    "config": {
      "model": "gpt-4o-mini",
      "inference_url": "https://api.openai.com",
      "temperature": 0.0,
      "max_completion_tokens": 256,
      "use_tools": false,
      "response_format": { "type": "text" },
      "system_template": "You are participating in a software engineering evaluation. Provide exactly one bash command enclosed in a single ```bash``` block. No THOUGHT. No extra text. If unsure, output ```bash\necho NOOP\n```.",
      "instance_template": "{{problem_statement}}\n\n{{instructions}}",
      "action_template": "{{ output.stdout }}"
    }
  },
  "env": { "env_name": "swe-mini" },
  "ops": ["agent","env","agent","env","agent","env"],
  "record": {"trajectories": true, "return_trace": true, "trace_format": "compact"}
}
```

- OpenAI gpt-5-mini (experimental): remove reasoning flags and constrain output. If responses are empty, retry without `stop` and consider switching to `gpt-4o-mini`.
