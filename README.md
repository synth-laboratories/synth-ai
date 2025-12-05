# Synth

[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/synth-ai.svg)](https://pypi.org/project/synth-ai/)
[![PyPI Dev](https://img.shields.io/badge/dev-0.3.2.dev3-orange)](https://pypi.org/project/synth-ai/0.3.2.dev3/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
![Coverage](https://img.shields.io/badge/coverage-28.65%25-yellow)
![Tests](https://img.shields.io/badge/tests-847%20passing-brightgreen)

Serverless Posttraining APIs for Developers

<p align="center">
  <picture align="center">
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/synth-laboratories/synth-ai/main/assets/langprobe_v2_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/synth-laboratories/synth-ai/main/assets/langprobe_v2_light.png">
    <img alt="Shows a bar chart comparing prompt optimization performance across Synth GEPA, Synth MIPRO, GEPA (lib), DSPy MIPRO, and DSPy GEPA with baseline vs optimized." src="https://raw.githubusercontent.com/synth-laboratories/synth-ai/main/assets/langprobe_v2_light.png">
  </picture>
</p>

<p align="center">
  <i>Average accuracy on <a href="https://arxiv.org/abs/2502.20315">LangProBe</a> prompt optimization benchmarks.</i>
</p>

## Highlights

- 🚀 Train across sft, RL, and prompt opt by standing up a single cloudflared Fastapi wrapper around your code. No production code churn.
- ⚡️ Parallelize training and achieve 80% GPU util. via PipelineRL
- 🗂️ Train prompts and models across multiple experiments
- 🛠️ Spin up experiment queues and datastores locally for dev work
- 🔩 Run serverless training via cli or programmatically
- 🏢 Scales gpu-based model training to 64 H100s seemlessly
- 💾 Use GEPA-calibrated judges for fast, accurate rubric scoring
- 🖥️ Supports HTTP-based training across all programming languages
- 🤖 CLI utilities tuned for use with Claude Code, Codex, Opencode

## Getting Started

```bash
# Use with OpenAI Codex
uvx synth-ai codex
```

```bash
# Use with Opencode
uvx synth-ai opencode
```

Synth is maintained by devs behind the [MIPROv2](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=jauNVA8AAAAJ&citation_for_view=jauNVA8AAAAJ:u5HHmVD_uO8C) prompt optimizer.

## Documentation

**[docs.usesynth.ai](https://docs.usesynth.ai)**

## In-Process Runner (SDK)

Run GEPA/MIPRO/RL jobs against a tunneled task app without the CLI:

```python
import asyncio
import os
from synth_ai.sdk.task import run_in_process_job

result = asyncio.run(
    run_in_process_job(
        job_type="prompt_learning",
        config_path="configs/style_matching_gepa.toml",
        task_app_path="task_apps/style_matching_task_app.py",
        overrides={"prompt_learning.gepa.rollout.budget": 4},
        backend_url=os.getenv("TARGET_BACKEND_BASE_URL"),  # resolves envs automatically
    )
)
print(result.job_id, result.status.get("status"))
```
