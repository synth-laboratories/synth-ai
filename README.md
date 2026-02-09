# Synth

[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)
[![PyPI](https://img.shields.io/badge/PyPI-0.7.15-orange)](https://pypi.org/project/synth-ai/)
[![Crates.io](https://img.shields.io/crates/v/synth-ai?label=crates.io)](https://crates.io/crates/synth-ai)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Prompt Optimization, Graphs, and Agent Infrastructure

Use the sdk in Python (`uv add synth-ai`) and Rust (beta) (`cargo add synth-ai`), or hit our serverless endpoints in any language

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

## Highlights

- 🎯 **GEPA Prompt Optimization** - Automatically improve prompts with evolutionary search. See 70%→95% accuracy gains on Banking77, +62% on critical game achievements
- 🔍 **Zero-Shot Verifiers** - Fast, accurate rubric-based evaluation with configurable scoring criteria
- 🧬 **GraphGen** - Train custom verifier graphs optimized for your specific workflows. Train custom pipelines for other tasks
- 🧰 **Environment Pools** - Managed sandboxes and browser pools for coding and computer-use agents
- 🚀 **No Code Changes** - Wrap existing code in a FastAPI app and optimize via HTTP. Works with any language or framework
- ⚡️ **Local Development** - Run experiments locally with tunneled task apps. No cloud setup required
- 🗂️ **Multi-Experiment Management** - Track and compare prompts/models across runs with built-in experiment queues

## Getting Started

### SDK (Python)

```bash
pip install synth-ai==0.7.15
# or
uv add synth-ai
```

### GEPA Compatibility (Python)

Drop-in usage for `gepa-ai` style workflows:

```python
from synth_ai import gepa

trainset, valset, _ = gepa.examples.aime.init_dataset()
result = gepa.optimize(
    seed_candidate={"system_prompt": "You are a helpful assistant."},
    trainset=trainset,
    valset=valset,
    task_lm="openai/gpt-4.1-mini",
    max_metric_calls=150,
    reflection_lm="openai/gpt-5",
)
print(result.best_candidate["system_prompt"])
```
Requires `SYNTH_API_KEY` and access to the Synth backend.
Full Banking77 runthrough: `demos/gepa_banking77_compat.py`.

### SDK (Rust - Beta)

```bash
cargo add synth-ai
```

### TUI (Homebrew)

```bash
brew install synth-laboratories/tap/synth-ai-tui
synth-ai-tui
```

The TUI provides a visual interface for managing jobs, viewing events, and monitoring optimization runs.

## OpenCode Skills (Synth API)

The Synth-AI TUI integrates with OpenCode and ships a **`synth-api`** skill.

```bash
# List packaged skills shipped with synth-ai
uvx synth-ai skill list
```

```bash
uvx synth-ai skill install synth-api --dir ~/custom/opencode/skill
```

## LocalAPI Deploy (Cloud)

Deploy a LocalAPI with a Dockerfile and get a stable `task_app_url`:

```bash
export SYNTH_API_KEY=sk_live_...
synth localapi deploy \
  --name my-localapi \
  --app my_module:app \
  --dockerfile ./Dockerfile \
  --context . \
  --wait
```

Use the emitted `task_app_url` in training configs. Harbor auth uses `SYNTH_API_KEY`
as the task app API key.

## Tunnels

Synth optimization jobs need HTTPS access to your local task app. Two tunnel backends are available:

### SynthTunnel (Recommended)

Relay-based tunnel — no external binary required, supports 128 concurrent requests:

```python
from synth_ai.core.tunnels import TunneledLocalAPI

tunnel = await TunneledLocalAPI.create(local_port=8001, api_key="sk_live_...")
print(tunnel.url)            # https://st.usesynth.ai/s/rt_...
print(tunnel.worker_token)   # pass to job config
```

Use with optimization jobs:

```python
job = PromptLearningJob.from_dict(
    config,
    task_app_url=tunnel.url,
    task_app_worker_token=tunnel.worker_token,
)
```

### Cloudflare Quick Tunnel

Anonymous tunnel via trycloudflare.com — no API key needed:

```python
from synth_ai.core.tunnels import TunneledLocalAPI, TunnelBackend

tunnel = await TunneledLocalAPI.create(
    local_port=8001,
    backend=TunnelBackend.CloudflareQuickTunnel,
)
```

Requires `cloudflared` installed (`brew install cloudflared`). Use `task_app_api_key` instead of `worker_token` when configuring jobs.

See the [tunnels documentation](https://docs.usesynth.ai/sdk/tunnels) for the full comparison.

### Auth Basics (Don’t Mix These)

There are **three different keys** in the LocalAPI + SynthTunnel flow:

- **Synth API key** (`SYNTH_API_KEY`): Auth for the **backend** (`SYNTH_BACKEND_URL`).
  - Sent as `Authorization: Bearer <SYNTH_API_KEY>`.
  - Used when submitting jobs to `http://127.0.0.1:8080` (local) or `https://api.usesynth.ai` (cloud).
- **Environment API key** (`ENVIRONMENT_API_KEY`): Auth for your **task app**.
  - Sent as `x-api-key: <ENVIRONMENT_API_KEY>` to `/health`, `/info`, `/rollout`, etc.
  - Minted/managed by `ensure_localapi_auth()`.
- **SynthTunnel worker token** (`tunnel.worker_token`): Auth for **tunnel relay → task app**.
  - Passed to jobs as `task_app_worker_token`.
  - **Never** use this as a backend API key.

Common failures:
- `Invalid API key` on `/api/jobs/*` means the backend received the wrong key.
- `SYNTH_TUNNEL_ERROR: Invalid worker token` means the tunnel relay token is wrong.

## Branching and CI

### Branch model (all repos)

```
dev  ──PR──>  staging  ──PR──>  main
                 │
              integration
              tests run
```

| Branch    | Purpose                          |
|-----------|----------------------------------|
| `dev`     | Daily development                |
| `staging` | Pre-release gate with full CI    |
| `main`    | Released / production code       |

### How CI works for this repo

Cross-repo integration tests live in the **testing** repo (`synth-laboratories/testing`).

1. When a PR targets `staging` in `testing`, CI checks out `synth-ai` at the matching branch (e.g. `staging`). Falls back to `main` if the branch doesn't exist.
2. Tests that exercise synth-ai code:
   - `synth_ai_unit_tests` — `pytest tests/unit` (runs on every push)
   - `synth_ai_all_tests` — `pytest tests/` including integration (requires API keys)
   - `testing_unit_tests` — `pytest synth-ai-tests/unit/`

### Standard workflow

1. Work on `dev`.
2. When ready to validate, push `dev` and open a PR in `testing`: `dev -> staging`.
3. CI runs unit + integration tests against the matching `synth-ai` branch.
4. After staging is green, merge `staging -> main` in each repo.

### Running tests locally

From the `testing` repo (sibling checkout):

```bash
cd ../testing
bazel test //:offline_tests             # unit tests only
bazel test //:no_llm_tests              # everything except LLM-dependent tests
bazel test //:all_tests                 # everything
```

Or directly:

```bash
uv run pytest tests/unit -v
```

See `testing/CLAUDE.md` for the full test tier and suite reference.

## Testing

Run the TUI integration tests:

```bash
cd tui/app
bun test
```

Synth is maintained by devs behind the [MIPROv2](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=jauNVA8AAAAJ&citation_for_view=jauNVA8AAAAJ:u5HHmVD_uO8C) prompt optimizer.

## Documentation

**[docs.usesynth.ai](https://docs.usesynth.ai)**

## Community

**[Join our Discord](https://discord.gg/cjfAMcCZef)**

## GEPA Prompt Optimization (SDK)

Run GEPA prompt optimization programmatically:

```python
import asyncio
import os
from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob
from synth_ai.sdk.localapi import LocalAPIConfig, create_local_api

# Create a local task app: app = create_local_api(LocalAPIConfig(app_id="my_app", handler=my_handler))

# Create and submit a GEPA job
pl_job = PromptLearningJob.from_dict({
    "job_type": "prompt_learning",
    "config": {
        "prompt_learning": {
            "gepa": {
                "rollout": {"budget": 100},
                "population_size": 10,
                "generations": 5,
            }
        }
    },
    "task_app_id": "my_task_app",
})

pl_job.submit()
result = pl_job.stream_until_complete(timeout=3600.0)
print(f"Best score: {result.best_score}")
```

See the [Banking77 walkthrough](https://docs.usesynth.ai/cookbooks/banking77-colab) for a complete example with local task apps.

## Online MIPRO (SDK, Ontology Enabled)

Run online MIPRO so rollouts call a proxy URL and rewards stream back to the optimizer. Enable ontology by setting `MIPRO_ONT_ENABLED=1` and `HELIX_URL` on the backend, then follow the [Banking77 online MIPRO notes](simpler_online_mipro.txt).

```python
import os
from synth_ai.sdk.optimization.policy import MiproOnlineSession

# Use the demo config shape from demos/mipro_banking77
mipro_config = {...}

session = MiproOnlineSession.create(
    config_body=mipro_config,
    api_key=os.environ["SYNTH_API_KEY"],
)
urls = session.get_prompt_urls()
proxy_url = urls["online_url"]

# Use proxy_url in your rollout loop, then report rewards
session.update_reward(
    reward_info={"score": 0.9},
    rollout_id="rollout_001",
    candidate_id="candidate_abc",
)
```

## Graph Evolve: Optimize RLM-Based Verifier Graphs

Train a verifier graph with an RLM backbone for long-context evaluation. See the [Image Style Matching walkthrough](https://docs.usesynth.ai/cookbooks/graphs/overview) for a complete Graph Evolve example:

```python
from synth_ai.sdk.api.train.graph_evolve import GraphEvolveJob

# Train an RLM-based verifier graph
verifier_job = GraphEvolveJob.from_dataset(
    dataset="verifier_dataset.json",
    graph_type="rlm",
    policy_models=["gpt-4.1"],
    proposer_effort="medium",  # Use "medium" (gpt-4.1) or "high" (gpt-5.2)
    rollout_budget=200,
)
verifier_job.submit()
result = verifier_job.stream_until_complete(timeout=3600.0)

# Run inference with trained verifier
verification = verifier_job.run_verifier(
    trace=my_trace,
    context={"rubric": my_rubric},
)
print(f"Reward: {verification.reward}, Reasoning: {verification.reasoning}")
```

## Zero-Shot Verifiers (SDK)

Run a built-in verifier graph with rubric criteria passed at runtime. See the [Crafter VLM demo](demos/gepa_crafter_vlm/) for verifier optimization:

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
        trace={"session_id": "s", "session_time_steps": []},
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

You can also call arbitrary graphs directly with the Rust SDK:

```rust
use serde_json::json;
use synth_ai::{GraphCompletionRequest, Synth};

#[tokio::main]
async fn main() -> Result<(), synth_ai::Error> {
    let synth = Synth::from_env()?;

    let request = GraphCompletionRequest {
        job_id: "zero_shot_verifier_rubric_single".to_string(),
        input: json!({
            "trace": {"session_id": "s", "session_time_steps": []},
            "rubric": {"event": [], "outcome": []},
        }),
        model: None,
        prompt_snapshot_id: None,
        stream: None,
    };

    let resp = synth.complete(request).await?;
    println!("Output: {:?}", resp.output);
    Ok(())
}
```
