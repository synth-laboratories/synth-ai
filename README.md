# Synth

[![synth](https://img.shields.io/endpoint?url=PLACEHOLDER_BADGE_ENDPOINT_URL)](https://github.com/PLACEHOLDER_ORG/synth)
[![image](https://img.shields.io/pypi/v/synth-ai.svg)](https://pypi.python.org/pypi/synth-ai)
[![image](https://img.shields.io/pypi/l/synth-ai.svg)](https://pypi.python.org/pypi/synth-ai)
[![image](https://img.shields.io/pypi/pyversions/synth-ai.svg)](https://pypi.python.org/pypi/synth-ai)
[![Actions status](https://github.com/PLACEHOLDER_ORG/synth/actions/workflows/ci.yml/badge.svg)](https://github.com/PLACEHOLDER_ORG/synth/actions)
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white)](https://discord.gg/PLACEHOLDER_DISCORD_LINK)

Serverless Posttraining APIs for Developers

<p align="center">
  <picture align="center">
    <source media="(prefers-color-scheme: dark)" srcset="benchmark_performance_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="benchmark_performance_light.png">
    <img alt="Shows a bar chart comparing prompt optimization performance across DSPy GEPA, GEPA-AI, and Synth-AI on four benchmarks." src="benchmark_performance_light.png">
  </picture>
</p>

<p align="center">
  <i>Prompt optimization performance comparison across Banking77, HeartDisease, HotpotQA, and Pupa benchmarks.</i>
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

Synth is maintained by devs behind the [MIPROv2](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=jauNVA8AAAAJ&citation_for_view=jauNVA8AAAAJ:u5HHmVD_uO8C) prompt optimizer.
