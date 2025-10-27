# Task App Standards

## Case Study: Agora EX
- Dataset loader enforces schema for JSONL prompts (`customers/agora_ex/task_app/agora_ex_task_app.py:81`), guaranteeing rubric fields, IDs, and splits before seeds are exposed.  
- `TaskDatasetRegistry` registration plus cached loader (`customers/agora_ex/task_app/agora_ex_task_app.py:724`) ensures a single authoritative dataset instance for rollouts, eval, and filtering tools.  
- `TaskInfo` advertises text observations, free-text action space, dataset metadata, rubric placeholders, inference limits, and one-turn constraints (`customers/agora_ex/task_app/agora_ex_task_app.py:288`).  
- `provide_task_instances` merges base rubric data with prompt-specific rubric overrides so RL/eval agents always see the active scoring contract (`customers/agora_ex/task_app/agora_ex_task_app.py:332`).  
- Rollout executor implements the shared `RolloutRequest` contract, resolves policy config (temperature, max tokens), and returns structured metrics with trace payloads (`customers/agora_ex/task_app/agora_ex_task_app.py:414`).  
- Modal deployment mounts include repo root, `synth_ai`, and the task directory, easing remote execution (`customers/agora_ex/agora_ex.py:50`).  
- Helper scripts (`customers/agora_ex/run_rollouts.py`, `customers/agora_ex/run_rollouts.sh`) demonstrate the `/info` handshake, environment key flow, and compatible payloads for eval + RL pipelines.

## Case Study: GRPO Crafter
- Repository discovery utilities catch local vs Modal mounts and populate `sys.path`, avoiding brittle relative imports (`examples/warming_up_to_rl/task_app/grpo_crafter.py:17`).  
- `TaskDatasetSpec` describes procedural seeds and default configurations, while `CrafterDataset` exposes seed metadata for tooling (`examples/warming_up_to_rl/task_app/grpo_crafter.py:300`).  
- `TaskInfo` captures discrete action space, multi-modal observations, inference proxy endpoints, and long-horizon limits suited for RL training (`examples/warming_up_to_rl/task_app/grpo_crafter.py:306`).  
- `provide_task_instances` injects per-seed traits, inventory snapshots, and difficulty, so trainers can stratify experience collection (`examples/warming_up_to_rl/task_app/grpo_crafter.py:356`).  
- Rollout executor downgrades legacy math-style requests into Crafter-compatible ones (ops aliasing, policy defaults) and bridges to hosted environments before packaging a canonical `RolloutResponse` (`examples/warming_up_to_rl/task_app/grpo_crafter.py:420`).  
- Proxy endpoints for OpenAI and Groq plus tracing hooks (SessionTracer, SFT sinks) illustrate optional capabilities wired through `TaskAppConfig.proxy` and `app_state` (`examples/warming_up_to_rl/task_app/grpo_crafter.py:575`).  
- Modal configuration mirrors Agora by mounting repo roots and environment assets under `/opt/synth_ai_repo`, confirming portability expectations (`examples/warming_up_to_rl/task_app/grpo_crafter.py:634`).

## Shared Abstractions (Current)
- `TaskAppConfig` (synth_ai/task/server.py:33) is the central declarative object: it wires metadata, dataset registry, rollout executor, optional routers, proxy config, and lifecycle hooks.  
- `TaskInfo` (synth_ai/task/contracts.py:79) is the minimum metadata envelope RL tooling consumes; both task apps fill all top-level keys (task, environments, action_space, observation, dataset, rubric, inference, capabilities, limits).  
- `TaskDatasetRegistry` (synth_ai/task/datasets.py:31) provides caching, split validation, and seed normalization; both apps register datasets and rely on registry lookups inside `TaskAppConfig`.  
- `RolloutRequest` / `RolloutResponse` / `RolloutMetrics` (synth_ai/task/contracts.py:37) define the payload shape reused by CLI rollouts, evaluators, and hosted services.  
- `RubricBundle` (synth_ai/task/server.py:24) keeps outcome/events rubric references consistent for downstream scoring.  
- Proxy helpers (`synth_ai/task/server.py:68`) and auth requirements (`synth_ai/task/auth.py`) enforce consistent handling of API keys and vendor routing.

## Recommended Standards & Additions
- **Dataset Validation**: require every task app dataset loader to perform record-level validation (like Agora’s `_validate_record`) and expose an automated `validate_dataset.py` entrypoint. Encourage standard fields: `id`, `seed`, `split`, rubric metadata, and schema version. Consider augmenting `TaskDatasetSpec` with a `schema_version` and optional `record_schema` callable.  
- **Task Info Guarantees**: codify that `TaskInfo` must include: observation description, action space semantics, dataset descriptor (with version + selection info), rubric summary, inference endpoints/limits, capabilities flags (`supports_rollout`, `supports_env_lifecycle`), and limits (max turns/tokens or ops/time). Validation could live in a helper that asserts required keys before `TaskAppConfig` is accepted.  
- **Rollout Compatibility**: mandate support for both single-call `/rollout` and step-wise env lifecycle. Rollout executors should: (1) respect `ops` ordering, (2) surface `metrics.mean_return`, `metrics.details`, `outcome_score`, (3) embed a trace compatible with tracing_v3 (when enabled), and (4) echo back policy/env identifiers. Provide a shared mixin or utility for building `RolloutResponse` objects to reduce drift.  
- **Task Metadata Endpoint**: ensure `/info` and `/task_info` return dataset size, version, system/user prompt versions (if applicable), capabilities, and rubric hashes. Add a reusable serializer that consumes `TaskAppConfig` to populate these responses uniformly.  
- **Proxy and Inference Metadata**: standardize `TaskInfo.inference` keys (`providers`, `endpoints`, `default_model`, `temperature_bounds`) so trainers can programmatically configure policies. Provide helpers to auto-populate these from `ProxyConfig`.  
- **Tracing & SFT Hooks**: encourage use of `app_state` keys like `session_tracer_factory` and `sft_output_dir`, and document the minimal contract for these hooks. Consider formalizing them within `TaskAppConfig` instead of ad-hoc keys.  
- **Environment Keys & Health**: require `/health` and `/health/rollout` to check `ENVIRONMENT_API_KEY` and log expected prefixes (as Agora does). Provide a shared FastAPI dependency that handles this flow.  
- **Script Conventions**: recommend each task app ship `run_rollouts.py`, `{run_eval,run_rl}.sh`, and `.env.template` files wired to the shared rollout contract, ensuring eval and RL scripts behave identically across tasks.  
- **Dataset-to-SFT Automation**: standardize on the existing `synth-ai filter` CLI (available via `uvx synth-ai filter`) by shipping per-task configs and extending it with optional schema validation hooks, closing the loop between rollouts and supervised data prep.  
- **Modal Deployments**: document required `extra_local_dirs` mounts (repo root, `synth_ai`, task directory) and encourage constants/helpers to generate them, reducing per-task copy-paste.  
- **Testing Hooks**: propose lightweight smoke tests that call `/info`, `/rollout` (single seed), and dataset validator as part of CI to guarantee task apps remain deployable.
