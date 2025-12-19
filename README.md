# Synth

[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/synth-ai.svg)](https://pypi.org/project/synth-ai/)
[![PyPI Main](https://img.shields.io/badge/main-0.4.1-blue)](https://pypi.org/project/synth-ai/0.4.1/)
[![PyPI Nightly](https://img.shields.io/badge/nightly-0.4.0-orange)](https://pypi.org/project/synth-ai/)
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
- 💾 Use GEPA-calibrated verifiers for fast, accurate rubric scoring
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

## Zero-Shot Verifiers (SDK)

Run a built-in verifier graph with rubric criteria passed at runtime:

```python
import asyncio
import os
from synth_ai.sdk.graphs import VerifierClient

async def run_verifier():
    client = VerifierClient(
        base_url=os.environ["SYNTH_BACKEND_BASE"],
        api_key=os.environ["SYNTH_API_KEY"],
    )
    result = await client.evaluate(
        job_id="zero_shot_verifier_single",
        session_trace={"session_id": "s", "event_history": []},
        rubric={
            "event": [{"id": "accuracy", "weight": 1.0, "description": "Correctness"}],
            "outcome": [{"id": "task_completion", "weight": 1.0, "description": "Completed task"}],
        },
        options={"event": True, "outcome": True, "model": "gpt-5-nano"},
        policy_name="my_policy",
        task_app_id="my_task",
    )
    return result

asyncio.run(run_verifier())
```

You can also call arbitrary graphs directly:

```python
from synth_ai.sdk.graphs import GraphCompletionsClient

client = GraphCompletionsClient(base_url="https://api.usesynth.ai", api_key="...")
resp = await client.run(
    graph={"kind": "zero_shot", "verifier_type": "zero_shot_verifier_mapreduce"},
    input_data={"session_trace": {"session_id": "s", "event_history": []}, "rubric": {"event": [], "outcome": []}},
)
```

## GraphGen: Train Custom Verifier and RLM Graphs

Train custom verifier and RLM graphs using GraphGen:

```python
from synth_ai.sdk.api.train.graphgen import GraphGenJob

# Train a verifier graph (judge/scorer)
verifier_job = GraphGenJob.from_dataset(
    dataset="verifier_dataset.json",
    graph_type="verifier",
    policy_models=["gpt-4.1"],
    proposer_effort="medium",  # Use "medium" (gpt-4.1) or "high" (gpt-5.2)
    rollout_budget=200,
)
verifier_job.submit()
result = verifier_job.stream_until_complete(timeout=3600.0)

# Run inference with trained verifier
judgment = verifier_job.run_verifier(
    session_trace=my_trace,
    context={"rubric": my_rubric},
)
print(f"Score: {judgment.score}, Reasoning: {judgment.reasoning}")
```

```python
# Train an RLM graph (massive context via tools)
rlm_job = GraphGenJob.from_dataset(
    dataset="rlm_dataset.json",
    graph_type="rlm",
    configured_tools=[
        {"name": "materialize_context", "kind": "rlm_materialize", "stateful": True},
        {"name": "local_grep", "kind": "rlm_local_grep", "stateful": False},
        {"name": "codex_exec", "kind": "daytona_exec", "stateful": True},
    ],
    policy_models=["gpt-4.1"],
    proposer_effort="medium",
    rollout_budget=100,
)
rlm_job.submit()
result = rlm_job.stream_until_complete(timeout=3600.0)

# Run inference with trained RLM graph
output = rlm_job.run_inference({"query": "Find relevant sections", "context": large_document})
```

**Graph Types:**
- **`verifier`**: Trains a judge/scorer that evaluates traces and returns structured rewards
- **`rlm`**: Trains a graph optimized for massive contexts (1M+ tokens) using tool-based search
- **`policy`**: Trains a standard input→output graph (default)

**RLM Tools:**
- `materialize_context` - Store input fields for fast searching (~1ms local)
- `local_grep` - Regex search on materialized content (~1ms)
- `local_search` - Substring search (~1ms)
- `query_lm` - Sub-LM calls for processing chunks
- `codex_exec` - Shell execution for complex operations

**When to use RLM:**
- Context exceeds ~100K tokens (too large for prompt)
- You need to search/filter large datasets
- RAG-style workflows over massive corpora
