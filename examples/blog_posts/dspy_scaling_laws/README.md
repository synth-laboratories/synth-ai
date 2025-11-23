# DSPy Scaling Laws Experiment

Investigating how pipeline complexity (number of LLM calls) affects performance when using prompt optimization frameworks (GEPA and MIPRO).

## 🎯 Experiment Overview

**Research Question**: Does adding more LLM steps to a pipeline improve performance after optimization?

We test pipelines with **1, 2, 3, and 5 LLM calls** across **3 benchmarks** using **2 optimizers**:

- **Benchmarks**:
  - Banking77: Intent classification (77 classes)
  - HeartDisease: Binary classification
  - HotpotQA: Multi-hop question answering

- **Optimizers**:
  - GEPA: Genetic algorithm-based prompt evolution
  - MIPRO: Meta-prompt based instruction and demo optimization

- **Pipeline Complexities**:
  - 1-step: Single LLM call (baseline)
  - 2-step: Two-stage pipeline (Banking77 only)
  - 3-step: Three-stage pipeline
  - 5-step: Five-stage pipeline

**Total Experiments**: ~18 (3 benchmarks × 3 complexities × 2 optimizers)

## 📁 Directory Structure

```
dspy_scaling_laws/
├── README.md                          # This file
├── plan.txt                           # Detailed experiment plan
├── run_scaling_experiment.py          # Main experiment runner
├── analyze_results.py                 # Results analysis and visualization
├── create_pipeline_configs.py         # TOML config generator (optional)
├── benchmarks/
│   ├── banking77/
│   │   ├── pipeline_3step/
│   │   │   └── banking77_3step_task_app.py
│   │   └── pipeline_5step/
│   │       └── banking77_5step_task_app.py
│   ├── heartdisease/
│   │   └── configs/
│   └── hotpotqa/
│       └── configs/
├── results/                           # Generated after running experiments
│   ├── banking77/
│   ├── heartdisease/
│   ├── hotpotqa/
│   └── aggregate_results.json
└── visualizations/                    # Generated after analysis
    ├── scaling_curves_overall.png
    ├── improvement_heatmap_gepa.png
    └── optimizer_comparison.png
```

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Install dependencies
pip install dspy-ai datasets pandas matplotlib seaborn

# Set API key
export GROQ_API_KEY="your_groq_api_key"
```

### 2. Run Experiments

```bash
cd examples/blog_posts/dspy_scaling_laws

# Run all experiments (~9 hours, ~$30-50 API costs)
python run_scaling_experiment.py

# Or run a single experiment for testing
python -c "
import asyncio
from run_scaling_experiment import run_single_experiment
from pathlib import Path

asyncio.run(run_single_experiment(
    benchmark='banking77',
    num_steps=1,
    optimizer='gepa',
    output_dir=Path('results/test'),
    rollout_budget=50  # Reduced for testing
))
"
```

### 3. Analyze Results

```bash
# Generate visualizations and summary tables
python analyze_results.py

# View results
open visualizations/scaling_curves_overall.png
cat summary_results.md
```

## 📊 Expected Output

### Results Files

Each experiment generates:
- `results.json`: Summary metrics (baseline, final score, improvement)
- `learning_curve.json`: Performance over optimization iterations

### Visualizations

- `scaling_curves_overall.png`: Performance vs pipeline steps (all benchmarks)
- `scaling_curve_{benchmark}.png`: Per-benchmark scaling curves
- `improvement_heatmap_{optimizer}.png`: Heatmap of improvements
- `optimizer_comparison.png`: GEPA vs MIPRO comparison

### Summary Tables

- `summary_results.csv`: Complete results in CSV format
- `summary_results.md`: Markdown summary table

## 🧪 Hypotheses

1. **Scaling Hypothesis**: Performance improves with more pipeline steps
2. **Diminishing Returns**: Improvement rate decreases after 3 steps
3. **Optimizer Preference**: GEPA/MIPRO may favor different complexities
4. **Task Dependency**: Complex reasoning tasks benefit more than simple classification

## 🏗️ Implementation Details

### Approach

Instead of creating separate task apps for every (benchmark × pipeline) combination, we use a **unified DSPy-based runner** with a `MultiStepClassifier` that:

1. Loads benchmark-specific data and signatures
2. Creates modules with 1-5 chained predictors
3. Optimizes with GEPA or MIPRO
4. Tracks and saves results

**Benefits**:
- Reduced code duplication (18 task apps → 1 runner)
- Easier to extend and maintain
- Direct DSPy integration (no backend needed)

**Trade-offs**:
- Current `MultiStepClassifier` is simple (repeats same predictor)
- Production version could add inter-step communication
- Banking77 task apps still useful for Synth backend integration

### Pipeline Architectures

**1-Step** (Baseline):
```
Input → Classifier → Output
```

**3-Step**:
```
Input → Analyzer → Reasoner → Classifier → Output
```

**5-Step**:
```
Input → Parser → Contextualizer → Reasoner → Synthesizer → Classifier → Output
```

## 💰 Resource Requirements

- **Time**: ~9 hours for all experiments
  - GEPA: ~5-10 min per experiment
  - MIPRO: ~10-20 min per experiment

- **Cost**: ~$20-50 (Groq API)
  - Using `llama-3.1-8b-instant` (GEPA)
  - Using `gpt-oss-20b` (MIPRO)

- **Storage**: <100MB for all results

## 🔧 Customization

### Add a New Benchmark

1. Add benchmark config to `BENCHMARK_CONFIG` in `run_scaling_experiment.py`:

```python
BENCHMARK_CONFIG["my_benchmark"] = {
    "signature": MyBenchmarkSignature,
    "loader": load_my_benchmark_data,
    "metric": my_benchmark_metric,
    "train_seeds": list(range(50)),
    "val_seeds": list(range(50, 150)),
}
```

2. Implement the signature, loader, and metric functions

3. Run experiments:

```bash
python run_scaling_experiment.py
```

### Change Hyperparameters

Edit `run_single_experiment()` in `run_scaling_experiment.py`:

```python
# MIPRO
optimizer = MIPROv2(
    metric=config["metric"],
    num_candidates=30,        # Increase candidates
    max_bootstrapped_demos=15, # More demos
    num_trials=20,            # More trials
)

# GEPA
optimizer = GEPA(
    metric=config["metric"],
    max_metric_calls=500,     # Larger budget
    reflection_minibatch_size=5,
)
```

## 📝 Files Reference

| File | Purpose |
|------|---------|
| `plan.txt` | Detailed experiment design and implementation notes |
| `run_scaling_experiment.py` | Main experiment orchestration script |
| `analyze_results.py` | Generate visualizations and summary tables |
| `create_pipeline_configs.py` | Generate TOML configs (optional, for reference) |
| `banking77_3step_task_app.py` | 3-step Banking77 task app (Synth backend) |
| `banking77_5step_task_app.py` | 5-step Banking77 task app (Synth backend) |

## 🐛 Troubleshooting

### Common Issues

**Import Errors**:
```bash
# Make sure langprobe is in the parent directory
ls ../langprobe/integrations/learning_curve_tracker.py
```

**API Key Not Found**:
```bash
# Check .env file or export manually
export GROQ_API_KEY="your_key"
echo $GROQ_API_KEY
```

**Out of Memory**:
```python
# Reduce validation set size in BENCHMARK_CONFIG
"val_seeds": list(range(50, 100))  # Smaller valset
```

**Rate Limiting**:
```python
# Add delays between experiments
import time
time.sleep(30)  # Wait between experiments
```

## 📚 Related Work

- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [MIPRO Paper](https://arxiv.org/abs/2406.11695)
- [GEPA (Genetic Prompt Aggregation)](https://github.com/stanfordnlp/dspy)
- [LangProbe Benchmarks](../langprobe/)

## 📄 License

MIT License - See repository root for details

## 🙋 Questions?

- Review `plan.txt` for detailed experiment design
- Check `run_scaling_experiment.py` for implementation details
- See langprobe examples for more DSPy patterns

---

**Happy experimenting!** 🚀
