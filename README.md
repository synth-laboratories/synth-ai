# Synth-AI SDK

[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PyPI](https://img.shields.io/badge/PyPI-0.2.10-orange)](https://pypi.org/project/synth-ai/)
![Coverage](https://img.shields.io/badge/coverage-9.09%25-red)
![Tests](https://img.shields.io/badge/tests-37%2F38%20passing-brightgreen)
![Blacksmith CI](https://img.shields.io/badge/CI-Blacksmith%20Worker-blue)

> **Synth-AI** — Reinforcement Learning-as-a-Service for agents.  
> **Docs:** [Get Started →](https://docs.usesynth.ai/sdk/get-started)

---

## 🚀 Install

```bash
pip install synth-ai
# or
uv add synth-ai
```

**Import:**

```python
import synth_ai
```

**CLI (with uvx):**

```bash
uvx synth-ai setup
uvx synth-ai demo
uvx synth-ai deploy
uvx synth-ai run
```

> Full quickstart: [https://docs.usesynth.ai/sdk/get-started](https://docs.usesynth.ai/sdk/get-started)

---

## 🧩 About

Fast and effective reinforcement learning for agents, via an API.  
Easily scale GPU topologies, train multi-node, and integrate with existing agent software.

### Highlights

- Scale GPU topologies (A10Gs, H100s, multi-node available on request)
- Thin FastAPI wrapper integration
- Supports OSS models like **Qwen3** (GPT-OSS GA soon)
- Own your trained models

---

## ⚙️ Getting Started

Synth-AI ships with a built-in RL example: training **Qwen3-0.6B** on math reasoning.

1. Create accounts at [Synth](https://usesynth.ai) and [Modal](https://modal.com)
2. Then run:

   ```bash
   uvx synth-ai demo
   uvx synth-ai setup
   uvx synth-ai deploy
   uvx synth-ai run
   ```

3. To walk through your first RL run, see  
   👉 [Synth-AI SDK Docs](https://docs.usesynth.ai/sdk/get-started)

---

## 🔐 SDK → Dashboard Pairing

When you run `uvx synth-ai setup` (or legacy `uvx synth-ai rl_demo setup`):

- The SDK opens your browser to the Synth dashboard to pair your SDK with your signed-in session.
- Automatically detects your **user + organization**
- Ensures both **API keys** exist
- Writes them to your project’s `.env` as:

  ```
  SYNTH_API_KEY=
  ENVIRONMENT_API_KEY=
  ```

✅ No keys printed or requested interactively — all handled via browser pairing.

### Environment overrides

- `SYNTH_CANONICAL_ORIGIN` → override dashboard base URL (default: https://www.usesynth.ai/dashboard)
- `SYNTH_CANONICAL_DEV` → `1|true|on` to use local dashboard (http://localhost:3000)

---

## 📚 Documentation

- **SDK Docs:** [https://docs.usesynth.ai/sdk/get-started](https://docs.usesynth.ai/sdk/get-started)
- **CLI Reference:** [https://docs.usesynth.ai/cli](https://docs.usesynth.ai/cli)
- **API Reference:** [https://docs.usesynth.ai/api](https://docs.usesynth.ai/api)
- **Changelog:** [https://docs.usesynth.ai/changelog](https://docs.usesynth.ai/changelog)

---

## 🧠 Meta

- Package: [`synth-ai`](https://pypi.org/project/synth-ai)
- Import: `synth_ai`
- Source: [github.com/synth-laboratories/synth-ai](https://github.com/synth-laboratories/synth-ai)
- License: MIT
