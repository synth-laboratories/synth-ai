# GEPA: Genetic Evolution for Prompt Optimization

This directory contains examples and configurations for using GEPA (Genetic Evolution for Prompt Optimization) to optimize prompts for various classification and reasoning tasks.

## Overview

**GEPA** is an evolutionary algorithm that optimizes prompts through genetic operations (mutation, crossover, selection) across multiple generations. It's particularly effective for:
- Intent classification (Banking77)
- Multi-hop QA (HotpotQA)
- Instruction following (IFBench)
- Claim verification (HoVer)
- Privacy-aware delegation (PUPA)

## Supported Tasks

Configuration files live under `configs/`:

| Task | Description | Config Files |
|------|-------------|--------------|
| **Banking77** | Intent classification (77 banking intents) | `banking77_gepa_local.toml`, `banking77_mipro_local.toml` |
| **HotpotQA** | Multi-hop question answering | `hotpotqa_gepa_local.toml`, `hotpotqa_mipro_local.toml` |
| **IFBench** | Instruction following benchmark | `ifbench_gepa_local.toml`, `ifbench_mipro_local.toml` |
| **HoVer** | Claim verification against Wikipedia | `hover_gepa_local.toml`, `hover_mipro_local.toml` |
| **PUPA** | Privacy-aware task delegation | `pupa_gepa_local.toml`, `pupa_mipro_local.toml` |

Each template targets a different default port (8110–8113) so you can run multiple task apps side-by-side.

---

## Quick Start (Banking77 Example)

### Prerequisites

```bash
# 1. Install dependencies
uv pip install -e .

# 2. Set environment variables
export SYNTH_API_KEY="your-backend-api-key"
export GROQ_API_KEY="gsk_your_groq_key"
export ENVIRONMENT_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
```

**Where to get API keys:**
- **GROQ_API_KEY**: Get from https://console.groq.com/keys
- **SYNTH_API_KEY**: Get from your backend admin or `.env.dev` file
- **ENVIRONMENT_API_KEY**: Generate a random secure token (command above)

### Step 1: Start the Backend

```bash
# Make sure your backend is running
curl http://localhost:8000/api/health
# Should return: {"status":"ok"}
```

### Step 2: Deploy Task App

**Option A: Using helper script (recommended)**
```bash
# Terminal 1
./examples/blog_posts/gepa/deploy_banking77_task_app.sh
```

**Option B: Using CLI**
```bash
uvx synth-ai deploy banking77 --runtime uvicorn --port 8102
```

**Option C: Deploy to Modal**
```bash
uvx synth-ai deploy banking77 --runtime modal --name banking77-gepa --env-file .env
```

### Step 3: Run GEPA Optimization

**Option A: Using helper script (recommended)**
```bash
# Terminal 2
./examples/blog_posts/gepa/run_gepa_banking77.sh
```

**Option B: Using CLI directly**
```bash
uvx synth-ai train \
  --config examples/blog_posts/gepa/configs/banking77_gepa_local.toml \
  --backend http://localhost:8000 \
  --poll
```

### Step 4: Monitor Progress

You'll see real-time output like:
```
🧬 Running GEPA on Banking77
=============================
✅ Backend URL: http://localhost:8000
✅ Task app is healthy

🚀 Starting GEPA training...

proposal[0] train_accuracy=0.65 len=120 tool_rate=0.95 N=30
  🔄 TRANSFORMATION:
    [SYSTEM]: Classify customer banking queries into intents...

Generation 1/15: Best reward=0.75 (75% accuracy)
Generation 2/15: Best reward=0.82 (82% accuracy)
...
✅ GEPA training complete!
```

Results are automatically saved to `configs/results/gepa_results_<job_id>_<timestamp>.txt`.

---

## Configuration

### Example: Banking77 GEPA Configuration

```toml
[prompt_learning]
algorithm = "gepa"
task_app_url = "http://127.0.0.1:8102"
task_app_id = "banking77"

# Training seeds (30 seeds from train pool)
evaluation_seeds = [50, 51, 52, ..., 79]

# Validation seeds (50 seeds from validation pool - not in training)
validation_seeds = [0, 1, 2, ..., 49]

[prompt_learning.gepa]
initial_population_size = 20    # Starting population of prompts
num_generations = 15            # Number of evolutionary cycles
mutation_rate = 0.3             # Probability of mutation
crossover_rate = 0.5            # Probability of crossover
rollout_budget = 1000           # Total rollouts across all generations
max_concurrent_rollouts = 20    # Parallel rollout limit
pareto_set_size = 20           # Size of Pareto front
```

### Key Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `initial_population_size` | Starting number of prompt variants | 10-50 |
| `num_generations` | Evolutionary cycles to run | 5-30 |
| `mutation_rate` | Probability of mutating a prompt | 0.1-0.5 |
| `crossover_rate` | Probability of combining two prompts | 0.3-0.7 |
| `rollout_budget` | Total task evaluations allowed | 200-2000 |
| `max_concurrent_rollouts` | Parallel rollout limit | 10-50 |
| `pareto_set_size` | Multi-objective optimization frontier size | 10-30 |

---

## Querying Results

After GEPA completes, you can query job results programmatically:

### Python API

```python
from synth_ai.learning import get_prompts, get_prompt_text, get_scoring_summary

# Get all results
results = get_prompts(
    job_id="pl_abc123",
    base_url="http://localhost:8000",
    api_key="sk_..."
)

# Access best prompt
best_prompt = results["best_prompt"]
best_score = results["best_score"]
print(f"Best Score: {best_score:.3f}")

# Get top-K prompts
for prompt_info in results["top_prompts"]:
    print(f"Rank {prompt_info['rank']}: {prompt_info['train_accuracy']:.3f}")
    print(prompt_info["full_text"])

# Quick access to best prompt text only
best_text = get_prompt_text(
    job_id="pl_abc123",
    base_url="http://localhost:8000",
    api_key="sk_...",
    rank=1  # 1 = best, 2 = second best, etc.
)

# Get scoring statistics
summary = get_scoring_summary(
    job_id="pl_abc123",
    base_url="http://localhost:8000",
    api_key="sk_..."
)
print(f"Best: {summary['best_train_accuracy']:.3f}")
print(f"Mean: {summary['mean_train_accuracy']:.3f}")
print(f"Tried: {summary['num_candidates_tried']}")
```

### Command Line

```bash
# Set environment variables
export BACKEND_BASE_URL="http://localhost:8000"
export SYNTH_API_KEY="sk_..."

# Run the example script
python examples/blog_posts/gepa/query_prompts_example.py pl_abc123
```

### REST API

```bash
# Get job status
curl -H "Authorization: Bearer $SYNTH_API_KEY" \
  http://localhost:8000/api/prompt-learning/online/jobs/JOB_ID

# Stream events
curl -H "Authorization: Bearer $SYNTH_API_KEY" \
  http://localhost:8000/api/prompt-learning/online/jobs/JOB_ID/events/stream

# Get metrics
curl -H "Authorization: Bearer $SYNTH_API_KEY" \
  http://localhost:8000/api/prompt-learning/online/jobs/JOB_ID/metrics
```

---

## Expected Results

GEPA typically improves accuracy over generations:

| Generation | Typical Accuracy | Notes |
|------------|------------------|-------|
| 1 (baseline) | 60-75% | Initial random/baseline prompts |
| 5 | 75-80% | Early optimization gains |
| 10 | 80-85% | Convergence begins |
| 15 (final) | 85-90%+ | Optimized prompts on Pareto front |

The Pareto front contains multiple prompt variants balancing:
- **Accuracy** (primary objective)
- **Token count** (efficiency objective)
- **Tool call rate** (task-specific objective)

---

## Helper Scripts

| Script | Purpose |
|--------|---------|
| `deploy_banking77_task_app.sh` | Start Banking77 task app locally |
| `run_gepa_banking77.sh` | Run GEPA optimization with validation checks |
| `test_gepa_local.sh` | Quick test script for local setup |
| `verify_banking77_setup.sh` | Comprehensive setup verification |
| `query_prompts_example.py` | Example script for querying results |

---

## Troubleshooting

### ❌ "Banking77 task app is not running"

**Solution:** Start the task app first
```bash
./examples/blog_posts/gepa/deploy_banking77_task_app.sh
```

### ❌ "Cannot connect to backend"

**Solution:** Verify backend is running
```bash
curl http://localhost:8000/api/health
```

If not running, start your backend service.

### ❌ "GROQ_API_KEY environment variable is required"

**Solution:** Export your Groq API key
```bash
export GROQ_API_KEY="gsk_your_key_here"
```

### ❌ "Failed to download dataset"

**Solution:** Check internet connection. The task app downloads from Hugging Face.

If you have the dataset locally:
```bash
export BANKING77_DATASET_NAME="/path/to/local/banking77"
```

### ❌ Pattern validation failed

**Solution:** Ensure your config's `initial_prompt.messages` uses the `{query}` wildcard:
```toml
[[prompt_learning.initial_prompt.messages]]
role = "user"
pattern = "Customer Query: {query}\n\nClassify this query."
```

### ⚠️ Metrics not streaming

**Solution:** 
1. Verify backend `/metrics` endpoint exists
2. Check SDK `StreamConfig` enables `StreamType.METRICS`
3. Restart local backend to pick up latest code

---

## Files in This Directory

```
examples/blog_posts/gepa/
├── README.md                         # This file - comprehensive guide
├── configs/                          # Configuration files
│   ├── banking77_gepa_local.toml    # Banking77 GEPA config
│   ├── banking77_mipro_local.toml   # Banking77 MIPRO config
│   ├── hotpotqa_gepa_local.toml     # HotpotQA configs
│   ├── ifbench_gepa_local.toml      # IFBench configs
│   ├── hover_gepa_local.toml        # HoVer configs
│   └── pupa_gepa_local.toml         # PUPA configs
├── deploy_banking77_task_app.sh     # Helper: Start task app
├── run_gepa_banking77.sh            # Helper: Run GEPA
├── test_gepa_local.sh               # Helper: Quick test
├── verify_banking77_setup.sh        # Helper: Verify setup
├── (baseline: examples/baseline/banking77_baseline.py)
├── query_prompts_example.py         # Query results example
└── task_apps.py                     # Task app registry
```

---

## Next Steps

1. **Evaluate optimized prompts**: Test best prompts on held-out validation split
2. **Compare with baseline**: Run `uvx synth-ai baseline banking77` to measure improvement
3. **Experiment with parameters**: Adjust mutation/crossover rates, population size
4. **Try MIPRO**: Compare GEPA with MIPROv2 optimization
5. **Benchmark across tasks**: Test on HotpotQA, IFBench, HoVer, PUPA

---

## Support

For issues or questions:

1. Verify all API keys are set correctly
2. Check task app: `curl -H "X-API-Key: $ENVIRONMENT_API_KEY" http://127.0.0.1:8102/health`
3. Check backend: `curl http://localhost:8000/api/health`
4. Review logs in both terminals for error messages
5. Run verification script: `./verify_banking77_setup.sh`

Happy optimizing! 🧬🚀
