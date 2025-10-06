# Synth AI Documentation

Welcome to the refreshed Synth AI guides. This documentation focuses on the workflows that ship with the public SDK today:

- Task apps built with the shared FastAPI harness
- Command line tooling exposed through `uvx synth-ai`
- Finetuning (FFT/QLoRA) and reinforcement learning (RL) pipelines
- Tracing v3, dataset generation, and evaluation loops

The docs are organized by role:

| Where to start | What you will find |
| --- | --- |
| [Task Apps](task_apps.md) | API contracts for the task harness, rollouts, rewards, seeds, and proxy support |
| [CLI Command Guide](cli/commands.md) | Reference for `uvx synth-ai` commands (`serve`, `modal-serve`, `deploy`, `train`, `setup`, `demo`, `run`) |
| [Workflows](workflows/overview.md) | Step-by-step guides for evaluation, finetuning (FFT + QLoRA), RL (FFT-first), tracing, and the math demo |
| [Examples](examples/warming_up_to_rl.md) | Crafter "Warming Up to RL" pipeline and roadmap for upcoming example refreshes |
| [References](references/environments.md) | Environment registry, LM helper modules, and additional APIs |

Each section links back to code in this repository so you can trace behaviour directly. If you are migrating from earlier versions of the docs, note that the legacy Mintlify export now lives under `docs_old/` for reference.

