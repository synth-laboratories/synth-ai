# Iris GEPA Benchmark - LangProBe Dec 31 2024

Benchmark comparing GEPA prompt optimization across different model sizes on Iris classification.

## Dataset

The Iris dataset has 150 samples with 3 classes (50 each):
- **setosa** (samples 0-49)
- **versicolor** (samples 50-99)
- **virginica** (samples 100-149)

## Experiment Design

**IMPORTANT:** The Iris dataset is ordered by class (0-49: setosa, 50-99: versicolor, 100-149: virginica).
Seeds must be stratified to include all 3 classes in both training and validation!

| Model | Runs | Rollout Budget | Training Seeds | Validation Seeds |
|-------|------|----------------|----------------|------------------|
| gpt-4.1-nano | 3 | 500 | 90 (stratified) | 60 (stratified) |
| gpt-5-nano | 3 | 500 | 90 (stratified) | 60 (stratified) |
| gpt-4o-mini | 3 | 500 | 90 (stratified) | 60 (stratified) |

**Stratified Seed Selection:**
- Training: 30 from each class = 90 total (indices 0-29, 50-79, 100-129)
- Validation: 20 from each class = 60 total (indices 30-49, 80-99, 130-149)

**Total: 9 experiments**

### GEPA Parameters
- Initial population: 10
- Generations: 5
- Children per generation: 4
- Crossover rate: 0.5
- Mutation rate: 0.3
- Pareto set size: 20

## Running the Benchmark

### Prerequisites

```bash
# Ensure synth-ai is installed
cd /path/to/synth-ai
pip install -e .

# Set API key (optional - will mint demo key if not set)
export SYNTH_API_KEY=sk_live_...
```

### Run All Experiments

```bash
cd /path/to/cookbooks/dev/blog_posts/langprobe_dec31/iris
python run_benchmark.py
```

### Run Single Model

```bash
python run_benchmark.py --model gpt-4.1-nano
```

### Run Single Experiment

```bash
python run_benchmark.py --model gpt-4.1-nano --run 1
```

### Evaluate Baseline

```bash
# Evaluate baseline on held-out set
python eval_baseline.py --model gpt-4.1-nano --seeds 100-149
```

### Dry Run

```bash
python run_benchmark.py --dry-run
```

## Results

### Benchmark Results (Dec 31, 2024) - CORRECTED

**Previous results (100% accuracy) were INVALID** due to biased seed selection:
- Training used seeds 0-99 (only setosa + versicolor, NO virginica)
- Validation used seeds 100-149 (ALL virginica)
- This meant the model was never tested on classes it was trained on!

**Corrected results with stratified seeds:**

| Model | Run 1 | Run 2 | Run 3 | Mean | Std |
|-------|-------|-------|-------|------|-----|
| **gpt-4.1-nano** | 95% | 95% | 100% | 96.7% | 2.4% |
| **gpt-4o-mini** | 95% | 100% | 95% | 96.7% | 2.4% |
| **gpt-5-nano** | 55% | 100% | 100% | 85.0% | 21.2% |

**Key Findings:**
- **gpt-4.1-nano** and **gpt-4o-mini** achieved consistent ~97% accuracy with low variance
- **gpt-5-nano** showed high variance (55%-100%), with one run significantly underperforming
- gpt-5-nano is a reasoning model (temp=1.0) that doesn't reliably use function calling API
- Previous "100% accuracy" was due to biased evaluation, not actual model improvement

**Notes:**
- N = 90 training samples with stratified class sampling (30 per class)
- Validation = 60 samples with stratified sampling (20 per class)
- Baseline uses a simple system prompt; GEPA optimizes the prompt through evolutionary search
- gpt-5-nano requires fallback to text extraction when tool calls are not used

### Baseline vs GEPA Comparison

GEPA shows significant improvement over baseline prompts on the same training seeds:

| Model | Baseline (Training) | GEPA Mean | Improvement |
|-------|---------------------|-----------|-------------|
| **gpt-4.1-nano** | 82.2% | 96.7% | +14.5 pp |
| **gpt-4o-mini** | 87.8% | 96.7% | +8.9 pp |
| **gpt-5-nano** | 81.1% | 85.0% | +3.9 pp |

**Key observations:**
- GEPA provides substantial accuracy gains (8-15 percentage points) for deterministic models (gpt-4.1-nano, gpt-4o-mini)
- gpt-5-nano (reasoning model) shows limited improvement with high variance due to unreliable tool calling
- The versicolor class (samples 50-99) is the hardest to distinguish - both baseline and GEPA struggle most here
- gpt-5-nano sometimes returns raw measurements instead of species names, reducing its baseline accuracy

### Output Files

Results are saved to `results/`:
- `iris_{model}_run{N}_result.json` - Individual run results
- `iris_{model}_run{N}_prompt.json` - Optimized prompts
- `benchmark_summary.json` - Full benchmark summary

Each result JSON contains:
```json
{
  "model": "gpt-4.1-nano",
  "run": 1,
  "job_id": "pl_5c357683adc941ac",
  "status": "succeeded",
  "best_score": 1.0,
  "elapsed_seconds": 228.8,
  "timestamp": "2025-12-31T11:26:20.544459",
  "prompt_file": "results/iris_gpt_4.1_nano_run1_prompt.json"
}
```

### Job IDs for Reference

**Corrected runs (stratified seeds):**

| Model | Run | Job ID | Score | Duration |
|-------|-----|--------|-------|----------|
| gpt-4.1-nano | 1 | pl_dcc03889028a4435 | 95% | 363s |
| gpt-4.1-nano | 2 | pl_36ce0d9175d14362 | 95% | ~900s |
| gpt-4.1-nano | 3 | pl_4b0a0e7e699a4997 | 100% | ~350s |
| gpt-4o-mini | 1 | pl_375d288824044d04 | 95% | 406s |
| gpt-4o-mini | 2 | pl_2638df6e6243417a | 100% | 445s |
| gpt-4o-mini | 3 | pl_ca5ea2c10803473b | 95% | 372s |
| gpt-5-nano | 1 | pl_525711619cd849a5 | 55% | 1480s |
| gpt-5-nano | 2 | pl_947f53cbd7c14a7e | 100% | 1871s |
| gpt-5-nano | 3 | pl_52adec271dfc41d4 | 100% | 1788s |

**Previous runs (INVALID - biased seeds):**

| Model | Run | Job ID | Score* | Duration |
|-------|-----|--------|--------|----------|
| gpt-4.1-nano | 1 | pl_5c357683adc941ac | 100%* | 229s |
| gpt-4.1-nano | 2 | pl_de8c6a746c3c410d | 100%* | 282s |
| gpt-4.1-nano | 3 | pl_886e43453fa547c5 | 100%* | 313s |
| gpt-4o-mini | 1 | pl_990fcbeef7de47b8 | 100%* | 330s |
| gpt-4o-mini | 2 | pl_0225ed84292c4fc9 | 100%* | 420s |
| gpt-4o-mini | 3 | pl_4cc52640ef0c404b | 100%* | 460s |

*Scores marked with * are from biased evaluation (training on only 2 classes, validating on 1 class)

## Optimized Prompts

Optimized prompts are saved after each successful GEPA run. Example:
```json
{
  "model": "gpt-4.1-nano",
  "run": 1,
  "job_id": "pl_...",
  "best_score": 0.96,
  "optimized_prompt": {
    "template": {
      "sections": [
        {"role": "system", "content": "...optimized system prompt..."},
        {"role": "user", "content": "...user template..."}
      ]
    }
  }
}
```
