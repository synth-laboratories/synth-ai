# Synth-AI

[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PyPI](https://img.shields.io/badge/PyPI-0.2.10-orange)](https://pypi.org/project/synth-ai/)
![Coverage](https://img.shields.io/badge/coverage-9.09%25-red)
![Tests](https://img.shields.io/badge/tests-37%2F38%20passing-brightgreen)
![Blacksmith CI](https://img.shields.io/badge/CI-Blacksmith%20Worker-blue)

Docs: [Synth‑AI Documentation](https://docs.usesynth.ai/welcome/introduction)

Fast and effective reinforcement learning for agents, via an API

> Latest: 0.2.10 published to PyPI (uv publish)

## Highlights

- Easily scale gpu topologies - train on 3 a10gs or 8 H100s (multi-node available upon request)
- Requires only a thin fastapi wrapper to integrate with existing agent software.
- Supports the best OSS models like Qwen3. (gpt-oss available upon request, GA soon)
- Own your trained models

## Getting Started

synth-ai comes with a built-in RL example tailored for training Qwen/Qwen3-0.6B to succeed at Math.

Please create an account at [Synth](https://usesynth.ai) and [Modal](https://modal.com) for the Math hello‑world test run. Then run:

```bash
uvx synth-ai demo
uvx synth-ai setup
uvx synth-ai deploy
uvx synth-ai run
```

To walk through kicking off your first RL run, see the [Synth‑AI Documentation](https://docs.usesynth.ai/welcome/introduction).

### What `setup` does now

When you run `uvx synth-ai setup` (or the legacy `uvx synth-ai rl_demo setup`), the SDK opens your browser to the Synth dashboard for a one‑time pairing (handshake) with your signed‑in session. The SDK will automatically:

- Detect your current user and organization
- Ensure both API keys exist for that user+org
- Write the keys to your project’s `.env` file as `SYNTH_API_KEY` and `ENVIRONMENT_API_KEY`

No keys are printed or requested interactively. You’ll see a confirmation like:

```
Connecting SDK to your browser session…
Connected to Acme Labs via browser.
```

If your browser isn’t already signed in, sign in when prompted and the pairing completes automatically. The dashboard’s welcome modal will reflect a successful pairing.

Environment variables:

- `SYNTH_CANONICAL_ORIGIN` (optional): override the dashboard base URL the SDK uses for the handshake (defaults to `https://www.usesynth.ai/dashboard`).
- `SYNTH_CANONICAL_DEV` (optional): set to `1`, `true`, `yes`, or `on` to target the local dashboard at `http://localhost:3000`.
- Keys are stored only in your project’s `.env` file, not exported to your shell.
