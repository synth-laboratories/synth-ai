# Environment Reference

Synth environments are packaged under `synth_ai/environments`. They serve two purposes:

1. **Task app integration** – referenced by `TaskAppConfig` implementations (e.g., Crafter).
2. **Standalone services** – the legacy environment registry still powers internal tooling.

## Packages

- `environment/` – core FastAPI service that exposes environment lifecycle routes for legacy stacks.
- `examples/` – sample environments and tasksets used in research.
- `service/` – helpers for running the environment microservice.
- `tasks/` – schema definitions and adapters for specific tasks.
- `reproducibility/` – utilities for deterministic seeds and world state management.

The manual registry guide (`synth_ai/environments/manual_registry.md`) walks through registering new environments with the service. When building task apps, prefer the newer `TaskAppConfig` path documented in [Task Apps](../task_apps.md).

## Seeds & determinism

- Use `TaskDatasetRegistry` (`synth_ai/task/datasets.py`) to advertise reproducible seeds.
- Surface seed ranges and default values in `TaskInfo.dataset` so the CLI and UI can sample and display instances (`synth_ai/task/apps/grpo_crafter.py:178`).

## Environment API keys

Whether you use the legacy environment service or the task app harness, protect environment endpoints with `ENVIRONMENT_API_KEY`. Secret management is handled by the CLI (`uvx synth-ai setup`, `uvx synth-ai serve --env-file`).

