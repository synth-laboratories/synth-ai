# EngineBench: Unified Prompt + Context Optimization Demo

This demo showcases **unified optimization** - simultaneously optimizing both the system prompt AND context artifacts (documentation, reference code, guides) for an AI coding agent that implements Pokemon TCG cards in Rust.

## The Key Innovation

Traditional prompt optimization only tunes the instruction text. This demo goes further by **co-evolving** four context artifacts:

1. **System Prompt** - Core instructions for the coding agent
2. **Architecture Guide** - Engine patterns and anti-patterns documentation
3. **Reference Snippets** - Code examples demonstrating best practices
4. **Hooks Documentation** - API reference for the runtime system

All four artifacts evolve together via GEPA (Generalized Evolutionary Prompt Algorithm), discovering optimal combinations that improve coding agent performance.

## What is EngineBench?

EngineBench is a SWE-Bench style benchmark where AI coding agents implement Pokemon TCG card effects in Rust:
- **Agent**: OpenCode (or any coding agent)
- **Task**: Implement card mechanics from specifications
- **Validation**: Deterministic cargo tests (no LLM-as-judge)
- **Challenge**: Understanding domain-specific architecture + following established patterns

Repository: https://github.com/JoshuaPurtell/engine-bench

## Prerequisites

1. **EngineBench repo** cloned locally:
   ```bash
   git clone https://github.com/JoshuaPurtell/engine-bench.git ~/Documents/GitHub/engine-bench
   ```

2. **OpenCode CLI** installed:
   ```bash
   bun install -g opencode
   ```

3. **Rust toolchain** installed:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

4. **Synth AI SDK** (already installed if you're in this repo):
   ```bash
   uv pip install -e .
   ```

## Quick Start

### 1. Start the Task App

The task app runs OpenCode agents and evaluates their code:

```bash
# From demos/engine_bench/
uvicorn localapi_engine_bench:app --host 0.0.0.0 --port 8020
```

You should see:
```
[engine_bench] Loaded 212 instances
INFO:     Application startup complete.
```

### 2. Test a Single Rollout (Baseline)

Test the baseline (default context artifacts):

```bash
python -c "
import asyncio
import httpx

async def test():
    response = await httpx.AsyncClient(timeout=600).post(
        'http://localhost:8020/rollout',
        json={
            'trace_correlation_id': 'test-001',
            'env': {'seed': 0},  # df-001-ampharos
            'policy': {
                'config': {
                    'model': 'gpt-4.1-mini',
                    'timeout': 300
                }
            }
        }
    )
    print(response.json())

asyncio.run(test())
"
```

Expected output shows compile status, test results, and score (0.0-1.0).

### 3. Run GEPA Optimization

Now optimize all context artifacts together:

```bash
python run_eval.py --config enginebench_gepa.toml --local
```

This will:
1. Evaluate baseline performance (static artifacts)
2. Run 5 generations of GEPA with 8 candidates each
3. Evolve system_prompt + architecture_guide + reference_snippets + hooks_documentation together
4. Track Pareto-optimal candidates (balancing pass_rate vs compile_success)
5. Save best candidates for inspection

Expected runtime: ~2-4 hours (depends on OpenCode agent speed)

## Understanding the Results

### Metrics

- **pass_rate**: Percentage of eval tests passed (0.0 to 1.0)
  - Primary objective - higher is better
- **compile_success**: Boolean - does generated code compile?
  - Secondary objective - must be True for tests to run
- **score**: Weighted combination (0.3 * compile + 0.7 * test_pass_rate)

### Baseline Performance

Static prompt + static docs typically achieves:
- Compile rate: ~60-70%
- Test pass rate: ~20-30% (of cards that compile)
- Overall score: ~0.30-0.40

### Target After Optimization

With optimized unified context:
- Compile rate: >80%
- Test pass rate: >50% (of cards that compile)
- Overall score: >0.55

## Value Proposition

This demo proves that **context artifact optimization provides significant lift beyond prompt engineering alone**:

- **System prompt** tells the agent *what* to do
- **Context artifacts** teach the agent *how* to do it

By co-evolving both, we discover:
- Which architecture patterns to emphasize
- What level of detail helps vs overwhelms
- Which code examples are most instructive
- How to structure API documentation for comprehension

## Evaluation Dataset

The GEPA config uses 15 diverse cards covering different mechanics:

| Card | Mechanics | Difficulty |
|------|-----------|------------|
| df-001-ampharos | Stage 2, Poke-Body | Medium |
| df-003-heracross | Basic, damage reduction Poke-Body | Easy |
| df-008-ninetales | Poke-Power (deck search) | Medium |
| df-009-pinsir | Armor Poke-Body | Easy |
| df-010-snorlax | Dual Power+Body, sleep mechanics | Hard |
| df-017-jynx | Retreat cost modification | Medium |
| ... | ... | ... |

This balanced mix ensures evolved artifacts generalize across:
- Basic attacks vs complex powers
- Passive Poke-Bodies vs active Poke-Powers
- Damage modifiers vs utility effects
- Single-type vs dual-type Pokemon

## File Structure

```
demos/engine_bench/
├── localapi_engine_bench.py      # Task app with unified context support
├── run_eval.py                   # GEPA job launcher (needs update)
├── enginebench_gepa.toml         # GEPA config for unified optimization
└── README.md                     # This file
```

### Key Code: Unified Context Support

The task app extracts context artifacts from `request.context_override`:

```python
async def run_rollout(request: RolloutRequest, fastapi_request: Request):
    context_override = request.context_override or {}

    # Extract optimizable artifacts (or use defaults)
    system_prompt = context_override.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    architecture_guide = context_override.get("architecture_guide", DEFAULT_ARCHITECTURE_GUIDE)
    reference_snippets = context_override.get("reference_snippets", DEFAULT_REFERENCE_SNIPPETS)
    hooks_documentation = context_override.get("hooks_documentation", DEFAULT_HOOKS_DOCUMENTATION)

    # Build unified prompt
    prompt = build_prompt_with_context(
        instance, system_prompt, architecture_guide,
        reference_snippets, hooks_documentation
    )

    # Run OpenCode agent with combined context
    await run_opencode_agent(prompt, sandbox_dir, model, timeout, inference_url, api_key)
```

GEPA evolves all four artifacts together, discovering optimal combinations.

## Troubleshooting

### "No instances available"

Ensure engine-bench is cloned:
```bash
git clone https://github.com/JoshuaPurtell/engine-bench.git ~/Documents/GitHub/engine-bench
```

Or set custom location:
```bash
export ENGINE_BENCH_DIR=/path/to/engine-bench
```

### "opencode: command not found"

Install OpenCode:
```bash
bun install -g opencode
```

### "cargo: command not found"

Install Rust toolchain:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### Agent timeout errors

Increase timeout in config:
```toml
[policy.config]
timeout = 600  # 10 minutes
```

## Next Steps

After optimization completes:

1. **Inspect evolved artifacts**: Check how GEPA modified the system prompt and context docs
2. **Analyze Pareto frontier**: See trade-offs between compile success and test pass rate
3. **Test generalization**: Evaluate best candidate on held-out cards (seeds 100-150)
4. **Extract insights**: What patterns did GEPA discover? Can we learn from them?

## Related Documentation

- [EngineBench Repository](https://github.com/JoshuaPurtell/engine-bench)
- [GEPA Algorithm Documentation](https://docs.usesynth.ai/algorithms/gepa)
