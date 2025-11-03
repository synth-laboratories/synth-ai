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

## 🚀 Install version 0.2.16

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
uvx synth-ai baseline  # For coding agents: get baseline scores
```

> Full quickstart: [https://docs.usesynth.ai/sdk/get-started](https://docs.usesynth.ai/sdk/get-started)

---

When you run `uvx synth-ai setup`, the SDK opens your browser to the Synth dashboard for a one‑time pairing (handshake) with your signed‑in session. The SDK will automatically:


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

## 🤖 For Coding Agents: Get Started with Baselines

**Baselines** are the fastest way for coding agents to evaluate changes and measure improvement on Synth tasks.

### Why Use Baselines?

Baselines provide a **self-contained evaluation system** that:
- ✅ **No infrastructure required** — runs locally, no deployed task app needed
- ✅ **Quick feedback loop** — get task-by-task results in seconds
- ✅ **Compare changes** — establish a baseline score before making modifications
- ✅ **Auto-discoverable** — finds baseline files automatically in your codebase

### Quick Start for Coding Agents

```bash
# 1. List available baselines
uvx synth-ai baseline list

# 2. Run a quick 3-task baseline to get started
uvx synth-ai baseline banking77 --split train --seeds 0,1,2

# 3. Get your baseline score (full train split)
uvx synth-ai baseline banking77 --split train

# 4. Make your changes to the code...

# 5. Re-run to compare performance
uvx synth-ai baseline banking77 --split train --output results_after.json
```

### Available Baselines

```bash
# Filter by task type
uvx synth-ai baseline list --tag rl          # RL tasks
uvx synth-ai baseline list --tag nlp         # NLP tasks
uvx synth-ai baseline list --tag vision      # Vision tasks

# Run specific baselines
uvx synth-ai baseline warming_up_to_rl       # Crafter survival game
uvx synth-ai baseline pokemon_vl             # Pokemon Red (vision)
uvx synth-ai baseline gepa                   # Banking77 classification
```

### Baseline Results

Each baseline run provides:
- **Task-by-task results** — see exactly which seeds succeed/fail
- **Aggregate metrics** — success rate, mean/std rewards, total tasks
- **Serializable output** — save to JSON with `--output results.json`
- **Model comparison** — test different models with `--model`

Example output:
```
============================================================
Baseline Evaluation: Banking77 Intent Classification
============================================================
Split(s): train
Tasks: 10
Success: 8/10
Execution time: 12.34s

Aggregate Metrics:
  mean_outcome_reward: 0.8000
  success_rate: 0.8000
  total_tasks: 10
```

### Creating Custom Baselines

Coding agents can create new baseline files to test custom tasks:

```python
# my_task_baseline.py
from synth_ai.baseline import BaselineConfig, BaselineTaskRunner, DataSplit, TaskResult

class MyTaskRunner(BaselineTaskRunner):
    async def run_task(self, seed: int) -> TaskResult:
        # Your task logic here
        return TaskResult(...)

my_baseline = BaselineConfig(
    baseline_id="my_task",
    name="My Custom Task",
    description="Evaluate my custom task",
    task_runner=MyTaskRunner,
    splits={
        "train": DataSplit(name="train", seeds=list(range(10))),
    },
)
```

Place this file in `examples/baseline/` or name it `*_baseline.py` for auto-discovery.

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
