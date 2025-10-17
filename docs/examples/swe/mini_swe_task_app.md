## mini-SWE Task App

The mini-SWE task app exposes the
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) workflow through
the Synth AI task-app interface. It mirrors the Crafter integration: the server
hosts an environment + policy pair and exposes rollouts, per-env lifecycle
endpoints, and vendor inference proxies.

### Quick start

```bash
uvx synth-ai serve swe-mini --port 8020 \
  --env-file examples/swe/task_app/.env.local
```

The command above:

- loads the task app registered by `examples/swe/task_app/grpo_swe_mini.py`
- serves FastAPI routes under `http://localhost:8020`
- enables the OpenAI and Groq proxy endpoints (pass your API keys via env vars)

> By default the task app streams SWE-Bench Verified from Hugging Face so you
> can evaluate real issues immediately. You can swap in your own dataset (or
> the bundled sample JSONL) via environment variables shown below.

### Dataset selection

The environment needs a set of SWE task descriptors (instance id, instructions,
docker image metadata). Choose one of the following sources:

| Variable | Example | Notes |
| --- | --- | --- |
| `SWE_MINI_DATASET=hf://princeton-nlp/SWE-Bench_Verified:test` | (default) | streams SWE-Bench Verified via `datasets` |
| `SWE_MINI_DATASET=file:///path/to/tasks.jsonl` | local dataset | JSON or JSONL; each record must include `instance_id`, `problem_statement`, optional `metadata` (docker image etc) |
| `SWE_MINI_DATASET=file:///.../sample_instances.json` | optional sample | quick smoke test bundled with the repo |

Records are normalised into:

```jsonc
{
  "instance_id": "SWE-bench_verified/test__django__... ",
  "problem_statement": "... user-facing description ...",
  "instructions": "... additional hints (optional) ...",
  "metadata": {
    "image_name": "docker.io/swebench/...",
    "repo": "...",
    "...": "..."
  }
}
```

If `image_name` is missing we derive the standard SWE-Bench docker tag.

### Execution environment

mini-swe-agent supports multiple execution backends. Configure via:

- `SWE_MINI_ENVIRONMENT_CLASS` (`local`, `docker`, `singularity`, …)
- `SWE_MINI_ENVIRONMENT_KWARGS` (JSON of extra keyword args)

Example for docker:

```bash
export SWE_MINI_ENVIRONMENT_CLASS=docker
export SWE_MINI_ENVIRONMENT_KWARGS='{"timeout": 90, "forward_env": ["OPENAI_API_KEY"]}'
```

For local execution, set `cwd` to a prepared workspace that mirrors the target
repository layout.

### Rollouts and RL

- Use `uvx synth-ai rollout swe-mini --config ...` to trigger a traced rollout.
- RL training can point at the hosted app exactly like the Crafter example;
  the task app advertises `supports_env_lifecycle` and `supports_rollout`.
- Command/reward traces are captured by the shared hosted backend in
  `examples/swe/task_app/hosted`, enabling SFT JSONL dumps and v3 tracing when
  `TASKAPP_TRACING_ENABLED=1`.

### Tracing and SFT

Enable tracing just like other task apps:

```bash
uvx synth-ai serve swe-mini --trace traces/v3 --trace-db traces/v3/swe_mini.db
```

The environment emits:

- decision traces (commands + outputs)
- optional SFT snippets in `traces/v3/*.jsonl`
- rollout metadata in SQLite/Postgres (configured via `--trace-db`)

### Modal deployment

The modal spec in `grpo_swe_mini.py` bundles all required packages
(`mini-swe-agent`, `datasets`, `litellm`, etc.) and mounts the example
directories so the hosted service can load dataset and policy adapters.

### Related files

- `examples/swe/task_app/grpo_swe_mini.py` – task app registration
- `examples/swe/task_app/hosted/envs/mini_swe` – environment & policy wrappers
- `examples/swe/task_app/hosted` – shared host scaffolding
- `examples/swe/task_app/README.md` – repository-level overview
