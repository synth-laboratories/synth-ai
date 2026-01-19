# RLM OOLONG Benchmark Demos

This directory contains demos for running RLM (Reasoning Language Model) verifiers
on OOLONG-style large context retrieval benchmarks.

## Overview

OOLONG is a benchmark for testing language models on retrieval and reasoning over
massive contexts. These demos test RLM v1 and v2 verifiers' ability to:

1. Handle large documents (100K+ characters)
2. Search effectively using tools (grep, view_lines)
3. Extract specific information buried in filler text
4. Produce accurate answers

## RLM v1 vs v2

| Feature | RLM v1 | RLM v2 |
|---------|--------|--------|
| Tool-based search | Yes | Yes |
| AgentFS integration | No | Yes |
| Multi-agent coordination | No | Yes |
| Message summarization | Basic | Enhanced |
| Parallel subagent queries | No | Yes |

## Running the Demos

### Prerequisites

1. Backend running (local or production)
2. API key set via `SYNTH_API_KEY` or use `--local` mode

### RLM v1 Demo

```bash
# Production mode
uv run python demos/rlm/run_rlm_v1_oolong.py

# Local mode (localhost:8000)
uv run python demos/rlm/run_rlm_v1_oolong.py --local

# Custom options
uv run python demos/rlm/run_rlm_v1_oolong.py --local \
    --context-size 50000 \
    --num-questions 5 \
    --model gpt-4o
```

### RLM v2 Demo

```bash
# Production mode
uv run python demos/rlm/run_rlm_v2_oolong.py

# Local mode with parallel execution
uv run python demos/rlm/run_rlm_v2_oolong.py --local --parallel

# Custom options
uv run python demos/rlm/run_rlm_v2_oolong.py --local \
    --context-size 100000 \
    --num-questions 3 \
    --model gpt-4.1-mini
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--local` | False | Use localhost:8000 backend |
| `--context-size` | 100000 | Size of OOLONG context in characters |
| `--num-questions` | 3 | Number of questions to test |
| `--model` | gpt-4.1-mini | Model for RLM execution |
| `--parallel` | False | (v2 only) Run questions in parallel |

## Test Questions

The demos use financial report questions:

1. "What was the Q3 2024 quarterly revenue?" → "$4.2 billion"
2. "What were the operating expenses in Q3 2024?" → "$2.1 billion"
3. "What was the gross margin percentage?" → "42.3%"
4. "What was the net income and its YoY change?" → "$1.5 billion (15% YoY)"
5. "How many monthly active users?" → "45 million"

## Expected Output

```
======================================================================
RLM V1 OOLONG BENCHMARK
======================================================================
Backend: http://localhost:8000
Model: gpt-4.1-mini
Context size: 100,000 chars
Questions: 3

Creating OOLONG context...
  Context created: 100,000 chars (~25,000 tokens)

Running RLM v1 verifier on 3 questions...
----------------------------------------------------------------------

[1/3] What was the Q3 2024 quarterly revenue?
  Reward: 0.85
  Time: 45000ms
  Status: OK
...

======================================================================
RESULTS SUMMARY
======================================================================
Questions tested: 3
Successful: 3
Errors: 0
Total time: 135000ms
Avg time per question: 45000ms

Average reward: 0.82

======================================================================
BENCHMARK COMPLETE
======================================================================
```

## Integration Tests

Integration tests for these demos are in:
- `monorepo/tests/integration/rlm_verifier/test_rlm_oolong.py`

Run them with:
```bash
pytest tests/integration/rlm_verifier/test_rlm_oolong.py --local -v
```
