# DSPy Scaling Laws Experiments

This directory contains experiments to study how prompt optimization (GEPA) performance scales with the number of LLM calls in DSPy modules.

## Overview

We test DSPy modules with **1, 3, and 5 LLM calls** on two harder benchmarks from the LangProBe suite:
- **HotpotQA**: Multi-hop question answering (baseline ~78-79%, best ~89%)
- **HeartDisease**: Binary medical classification (baseline ~54%, best ~75%)

## Structure

```
dspy_scaling_laws/
├── hotpotqa/
│   └── dspy_hotpotqa_scaling_adapter.py  # Modules with 1, 3, 5 calls
├── heartdisease/
│   └── dspy_heartdisease_scaling_adapter.py  # Modules with 1, 3, 5 calls
├── run_all_scaling_experiments.py  # Run all experiments
└── README.md
```

## Module Designs

### HotpotQA Modules

**1 Call**: Single `ChainOfThought` that directly answers the question.

**3 Calls**: 
- Step 1: Extract key information from context
- Step 2: Connect information across passages
- Step 3: Formulate final answer

**5 Calls**:
- Steps 1-3: Same as above
- Step 4: Refine and verify answer
- Step 5: Final answer with comprehensive support

### HeartDisease Modules

**1 Call**: Single `ChainOfThought` that directly classifies.

**3 Calls**:
- Step 1: Analyze patient features
- Step 2: Identify risk factors
- Step 3: Make classification decision

**5 Calls**:
- Steps 1-3: Same as above
- Step 4: Verify classification
- Step 5: Final classification with comprehensive reasoning

## Running Experiments

### Run Individual Experiment

```bash
# HotpotQA with 1 call
python dspy_scaling_laws/hotpotqa/dspy_hotpotqa_scaling_adapter.py \
    --num-calls 1 \
    --rollout-budget 200

# HotpotQA with 3 calls
python dspy_scaling_laws/hotpotqa/dspy_hotpotqa_scaling_adapter.py \
    --num-calls 3 \
    --rollout-budget 200

# HotpotQA with 5 calls
python dspy_scaling_laws/hotpotqa/dspy_hotpotqa_scaling_adapter.py \
    --num-calls 5 \
    --rollout-budget 200

# HeartDisease with 1 call
python dspy_scaling_laws/heartdisease/dspy_heartdisease_scaling_adapter.py \
    --num-calls 1 \
    --rollout-budget 300

# HeartDisease with 3 calls
python dspy_scaling_laws/heartdisease/dspy_heartdisease_scaling_adapter.py \
    --num-calls 3 \
    --rollout-budget 300

# HeartDisease with 5 calls
python dspy_scaling_laws/heartdisease/dspy_heartdisease_scaling_adapter.py \
    --num-calls 5 \
    --rollout-budget 300
```

### Run All Experiments

```bash
python dspy_scaling_laws/run_all_scaling_experiments.py
```

## Results

Results are saved in:
- `hotpotqa/results/gepa_1calls/`
- `hotpotqa/results/gepa_3calls/`
- `hotpotqa/results/gepa_5calls/`
- `heartdisease/results/gepa_1calls/`
- `heartdisease/results/gepa_3calls/`
- `heartdisease/results/gepa_5calls/`

Each directory contains:
- `dspy_gepa_detailed_results.json`: Detailed optimization results
- `dspy_gepa_*_stats.json`: Summary statistics
- `dspy_gepa.log`: Verbose DSPy logs
- Learning curve CSV/JSON files

## Configuration

### Models
- **Policy Model**: 
  - HotpotQA: `groq/llama-3.3-70b-versatile` (default)
  - HeartDisease: `groq/openai/gpt-oss-20b` (default)
- **Reflection Model**: `groq/llama-3.3-70b-versatile` (for GEPA proposer)

### Rollout Budgets
- HotpotQA: 200 rollouts
- HeartDisease: 300 rollouts

### Dataset Splits
- HotpotQA: 50 train seeds (0-49), 30 val seeds (50-79)
- HeartDisease: 30 train seeds (0-29), 50 val seeds (30-79)

## Expected Outcomes

These experiments will help answer:
1. Does prompt optimization (GEPA) benefit more from multi-step reasoning?
2. How does the number of LLM calls affect optimization efficiency?
3. Are there diminishing returns beyond a certain number of calls?

## Notes

- Each module variant uses the same GEPA optimization settings
- All experiments use the DSPy proposer mode
- Results can be compared across different numbers of LLM calls to study scaling laws

## Gemini Integration Testing

### Test Synth AI Integration

To verify Gemini 2.5 family support works with Synth AI's GEPA and MIPRO optimizers:

```bash
# Terminal 1: Start backend locally
cd /Users/joshpurtell/Documents/GitHub/monorepo
bash scripts/run_backend_local.sh

# Terminal 2: Run integration test
cd /Users/joshpurtell/Documents/GitHub/synth-ai
source .env
uv run python examples/blog_posts/langprobe/dspy_scaling_laws/test_gemini_synth_integration.py
```

This test verifies:
- ✅ GEPA with Gemini policy model (`gemini-2.5-flash-lite`) via `synth_hosted`
- ✅ GEPA with Gemini mutation model (`gemini-2.5-flash`) for prompt proposals
- ✅ MIPRO with Gemini policy model (`gemini-2.5-flash-lite`) via `synth_hosted`
- ✅ MIPRO with Gemini meta model (`gemini-2.5-flash`) for instruction proposals

**Requirements**:
- `GEMINI_API_KEY` must be set in `.env`
- `SYNTH_API_KEY` must be set in `.env`
- Backend must be running locally
- HotpotQA task app must be running (default: `http://127.0.0.1:8110`)

**See**: `gemini_support.txt` for full implementation plan for adding Gemini support to Synth AI.

