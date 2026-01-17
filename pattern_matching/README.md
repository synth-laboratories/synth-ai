# Pattern Matching Test Harness

This folder contains gold patterns and a lightweight test pipeline for validating
pattern discovery against local trace sets.

## Layout

```
pattern_matching/
  gold_patterns/
    hello_world_bench_opencode.toml
    banking77.toml
    crafter_vlm.toml
    ptcg_react.toml
  run_pipeline.py
```

## How to Run

```
uv run python pattern_matching/run_pipeline.py \
  --traces-root pattern_traces \
  --gold-dir pattern_matching/gold_patterns \
  --output pattern_matching/results.json
```

## What It Does

- Loads trace files from `pattern_traces/*/traces/seed_*.json`
- Applies gold patterns to the request messages in each trace
- Reports match_rate + mismatches per use case

This is intentionally fast and local so we can iterate on the pattern discovery
algorithm without running full GEPA jobs.
