# Synth-AI

[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PyPI](https://img.shields.io/badge/PyPI-0.2.4.dev9-orange)](https://pypi.org/project/synth-ai/)
![Coverage](https://img.shields.io/badge/coverage-0.0%25-red)
![Tests](https://img.shields.io/badge/tests-17%2F17%20passing-brightgreen)

Docs: [Synth‑AI Documentation](https://docs.usesynth.ai/synth-ai/introduction)

Fast and effective reinforcement learning for agents, via an API

## Highlights

- Easily scale gpu topologies - train on 3 a10gs or 8 H100s (multi-node available upon request)
- Requires only a thin fastapi wrapper to integrate with existing agent software.
- Supports the best OSS models like Qwen3. (gpt-oss available upon request, GA soon)
- Own your trained models

## Getting Started

synth-ai comes with a built-in RL example tailored for training Qwen/Qwen3-0.6B to succeed at Math.

Please create an account at [Synth](https://usesynth.ai) and [Modal](https://modal.com) for the Math hello‑world test run. Then run:

```bash
uvx synth-ai rl_demo check
uvx synth-ai rl_demo deploy
uvx synth-ai rl_demo configure
uvx synth-ai rl_demo run
```

To walk through kicking off your first RL run, see the [Synth‑AI Documentation](https://docs.usesynth.ai/synth-ai/introduction).
