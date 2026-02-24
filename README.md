# Synth

[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)
[![PyPI](https://img.shields.io/badge/PyPI-0.9.0-orange)](https://pypi.org/project/synth-ai/)
[![Crates.io](https://img.shields.io/crates/v/synth-ai?label=crates.io)](https://crates.io/crates/synth-ai)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Prompt Optimization, Graphs, and Agent Infrastructure

Use the sdk in Python (`uv add synth-ai`) and Rust (beta) (`cargo add synth-ai`), or hit our serverless endpoints in any language

## Synth Style

For engineering principles and coding standards, follow:
`specs/README.md`.

<p align="center">
  <picture align="center">
    <source media="(prefers-color-scheme: dark)" srcset="assets/langprobe_v2_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="assets/langprobe_v2_light.png">
    <img alt="Shows a bar chart comparing prompt optimization performance across GPT-4.1 Nano, GPT-4o Mini, and GPT-5 Nano with baseline vs GEPA optimized." src="assets/langprobe_v2_light.png">
  </picture>
</p>

<p align="center">
  <i>Average accuracy on <a href="https://arxiv.org/abs/2502.20315">LangProBe</a> prompt optimization benchmarks.</i>
</p>

## Demo Walkthroughs

- [GEPA Banking77 Prompt Optimization](https://docs.usesynth.ai/cookbooks/banking77-colab)
- [GEPA Crafter VLM Verifier Optimization](https://docs.usesynth.ai/cookbooks/verifier-optimization)
- [GraphGen Image Style Matching](https://docs.usesynth.ai/cookbooks/graphs/overview)

Benchmark and demo runner source files have moved to the `Benchmarking` repo (`../Benchmarking` in a sibling checkout).

## Highlights

- 🎯 **GEPA Prompt Optimization** - Automatically improve prompts with evolutionary search. See 70%→95% accuracy gains on Banking77, +62% on critical game achievements
- 🔍 **Zero-Shot Verifiers** - Fast, accurate rubric-based evaluation with configurable scoring criteria
- 🧬 **GraphGen** - Train custom verifier graphs optimized for your specific workflows. Train custom pipelines for other tasks
- 🧰 **Environment Pools** - Managed sandboxes and browser pools for coding and computer-use agents
- 🚀 **No Code Changes** - Wrap existing code in a FastAPI app and optimize via HTTP. Works with any language or framework
- ⚡️ **Local Development** - Run experiments locally with tunneled containers. No cloud setup required
- 🗂️ **Multi-Experiment Management** - Track and compare prompts/models across runs with built-in experiment queues

## Getting Started

### SDK (Python)

```bash
pip install synth-ai==0
```
