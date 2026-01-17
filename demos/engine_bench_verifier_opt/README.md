# EngineBench Verifier Optimization Demo (Rust Traces)

This demo trains a verifier graph on EngineBench Rust coding traces. It learns to produce reward estimates that align with deterministic test outcomes from coding agents like OpenCode and Codex.

## Overview

The demo workflow:
1. Load EngineBench eval traces and rewards from `data/engine_bench/...`.
2. Build a GraphGen dataset with trace inputs and gold reward outputs.
3. Optimize a verifier graph that predicts outcome rewards for new traces.

## Prerequisites

- Backend running (local mode uses `http://localhost:8000`)
- `SYNTH_API_KEY` set in your environment or `.env`
- EngineBench traces under `data/engine_bench/{opencode|codex}/...`

## Local Mode (Recommended)

```bash
# Terminal 1: start local backend
cd /Users/joshpurtell/Documents/GitHub/monorepo
./scripts/run_backend_local.sh

# Terminal 2: run the demo
cd /Users/joshpurtell/Documents/GitHub/synth-ai
uv run python demos/engine_bench_verifier_opt/run_demo.py --local
```

## Target a Specific Eval Run

```bash
uv run python demos/engine_bench_verifier_opt/run_demo.py \
  --local \
  --agent opencode \
  --eval-dir /Users/joshpurtell/Documents/GitHub/synth-ai/data/engine_bench/opencode/20260115_124051_eval_e3de6913f3c84e02
```

## Example Progress (Trimmed)

```
INFO:     127.0.0.1:58508 - "POST /rollout HTTP/1.1" 200 OK
  [5m49s] running | Trials: 0/9 (0%) | Best: 0.950
  ✓ Rollout seed_29: reward=0.00 (pred=0.00, gold=1.0) | 10.1s | 9k tok | $0.0257
INFO:     127.0.0.1:58517 - "POST /rollout HTTP/1.1" 200 OK
  ✓ Rollout seed_27: reward=0.00 (pred=0.00, gold=1.0) | 11.7s | 13k tok | $0.0381
INFO:     127.0.0.1:58514 - "POST /rollout HTTP/1.1" 200 OK
  ✓ Rollout seed_35: reward=1.00 (pred=1.00, gold=1.0) | 11.9s | 32k tok | $0.0838
```

## Key Files

- `run_demo.py` — end-to-end verifier optimization for Rust traces
- `data/engine_bench/...` — eval traces and reward summaries

## Follow-on: Optimize an RLM Prompt

If you want to optimize a recursive language model prompt after verifier training, run the RLM demo:

```bash
cd /Users/joshpurtell/Documents/GitHub/monorepo
./scripts/run_backend_local.sh

cd /Users/joshpurtell/Documents/GitHub/synth-ai
uv run python demos/rlm-mit/run_demo.py --local
```

This RLM walkthrough uses the same local backend flow and produces outcome rewards for long-context reasoning tasks.
