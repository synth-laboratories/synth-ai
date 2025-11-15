# LangProBe Task Apps

This directory contains planning and implementation for task apps related to the **LangProBe (Language Programs Benchmark)**.

## Overview

LangProBe is a comprehensive benchmark evaluating language models across multiple categories:
- **Knowledge**: MMLU, HoVer, IReRa, HotpotQA, RAG, and more
- **Math**: MATH, GSM8K
- **Classification**: HeartDisease, Iris, CoTBasedVote, and more
- **Coding/Software Engineering**: HumanEval, SWE
- **Agentic/Multi-step Tasks**: AppWorld

## Current Status

### Already Implemented
- ✅ HotpotQA (in `examples/task_apps/gepa_benchmarks/`)
- ✅ HoVer (in `examples/task_apps/gepa_benchmarks/`)
- ✅ IFBench (in `examples/task_apps/gepa_benchmarks/`)
- ✅ PUPA (in `examples/task_apps/gepa_benchmarks/`)

### Planned Implementations
See [`langprobe.md`](../langprobe.md) for detailed implementation plan covering all 15+ LangProBe datasets.

**Implementation Location:** `examples/task_apps/other_langprobe_benchmarks/`

## Reference Papers

- **LangProBe**: [arXiv:2502.20315v1](https://arxiv.org/html/2502.20315v1)
- **GEPA Baseline**: [arXiv:2507.19457](https://arxiv.org/pdf/2507.19457) (provides rollout budgets and baseline scores)

## Quick Links

- [Planning Document](../langprobe.md) - Detailed implementation plan
- [Existing GEPA Benchmarks](../examples/task_apps/gepa_benchmarks/) - Reference implementations
- [Task App Standards](../docs/task_app_standards.md) - Implementation guidelines

Here are the dataset tasks included in LangProBe (the Language Programs Benchmark). ([arXiv][1])

| Category                    | Datasets                                                                                                                   |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Knowledge                   | MMLU, HoVer, IReRa, HotpotQA, RAG, RAG-Based Rank, MultiHopSummarize, HotpotQA Conditional, Simplified Baleen ([arXiv][1]) |
| Math                        | MATH, GSM8K ([arXiv][1])                                                                                                   |
| Classification              | HeartDisease, Iris, CoTBasedVote, GeneratorCriticRanker, GeneratorCriticFuser ([arXiv][1])                                 |
| Coding/Software Engineering | HumanEval

Alfworld

If you like, I can pull out **all 15 datasets** with full names, splits, and brief descriptions from the paper.

[1]: https://arxiv.org/html/2502.20315v1?utm_source=chatgpt.com "LangProBe: a Language Programs Benchmark"
[2]: https://www.themoonlight.io/en/review/langprobe-a-language-programs-benchmark?utm_source=chatgpt.com "[Literature Review] LangProBe: a Language Programs ..."
[3]: https://www.alphaxiv.org/overview/2502.20315v1?utm_source=chatgpt.com "LangProBe: a Language Programs Benchmark"
