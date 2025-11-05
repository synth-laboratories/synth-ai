# MIPROv2: Multi-Objective Prompt Optimization

This directory contains examples and configurations for using MIPROv2 (Multi-Objective Prompt Optimization) to optimize prompts for various classification and reasoning tasks.

## Overview

**MIPROv2** is a meta-learning algorithm that optimizes prompts using:
- **Bootstrap Phase**: Collects few-shot examples from high-scoring seeds
- **Meta-Model**: LLM that proposes prompt improvements based on demonstrations
- **TPE Optimization**: Tree-structured Parzen Estimator for efficient hyperparameter search
- **Mini-Batch Evaluation**: Efficient online evaluation on small seed pools

MIPROv2 is particularly effective when:
- You want faster convergence with fewer evaluations (~100 vs ~1000 for GEPA)
- You have clear task structure (can bootstrap with examples)
- You need efficient optimization (mini-batch evaluation)
- You want meta-learning benefits (few-shot adaptation)

## Supported Tasks

Configuration files live under `configs/`:

| Task | Description | Config Files |
|------|-------------|--------------|
| **Banking77** | Intent classification (77 banking intents) | `banking77_mipro_local.toml`, `banking77_mipro_test.toml` |

*More task configs coming soon: HotpotQA, IFBench, HoVer, PUPA*

---

## Quick Start (Banking77 Example)

### Prerequisites

```bash
# 1. Install dependencies
uv pip install -e .

# 2. Set environment variables
export SYNTH_API_KEY="your-backend-api-key"
export GROQ_API_KEY="gsk_your_groq_key"
export OPENAI_API_KEY="sk-your-openai-key"  # Required for meta-model
export ENVIRONMENT_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
```

**Where to get API keys:**
- **GROQ_API_KEY**: Get from https://console.groq.com/keys
- **OPENAI_API_KEY**: Get from https://platform.openai.com/api-keys (required for meta-model)
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
./examples/blog_posts/mipro/deploy_banking77_task_app.sh
```

**Option B: Using CLI**
```bash
uvx synth-ai deploy banking77 --runtime uvicorn --port 8102
```

**Option C: Deploy to Modal**
```bash
uvx synth-ai deploy banking77 --runtime modal --name banking77-mipro --env-file .env
```

### Step 3: Run MIPROv2 Optimization

**Option A: Using helper script (recommended)**
```bash
# Terminal 2
./examples/blog_posts/mipro/run_mipro_banking77.sh
```

**Option B: Using CLI directly**
```bash
uvx synth-ai train \
  --config examples/blog_posts/mipro/configs/banking77_mipro_local.toml \
  --backend http://localhost:8000 \
  --poll
```

### Step 4: Monitor Progress

You'll see real-time output like:
```
🔬 Running MIPROv2 on Banking77
=================================
✅ Backend URL: http://localhost:8000
✅ Task app is healthy

🚀 Starting MIPROv2 training...

Bootstrap Phase:
  Evaluating baseline on seeds [0-4]...
  Found 3 high-scoring examples (score >= 0.85)
  Initializing meta-model with few-shot examples...

Iteration 1/16:
  Meta-model proposing 6 prompt variants...
  Evaluating on online pool [5-9]...
  Best score: 0.78

Iteration 2/16:
  ...
  
✅ MIPROv2 training complete!
Best prompt accuracy: 0.87 (87%)
```

Results are automatically saved and can be queried via the Python API or REST endpoints.

---

## Configuration

### Example: Banking77 MIPROv2 Configuration

```toml
[prompt_learning]
algorithm = "mipro"
task_app_url = "http://127.0.0.1:8102"
task_app_id = "banking77"

[prompt_learning.initial_prompt]
messages = [
  { role = "system", content = "You are an expert banking assistant..." },
  { role = "user", pattern = "Customer Query: {query}\n\nClassify..." }
]

[prompt_learning.mipro]
num_iterations = 16                    # Optimization iterations
num_evaluations_per_iteration = 6      # Variants per iteration
batch_size = 6                         # Concurrent evaluations
max_concurrent = 16                    # Max parallel rollouts
meta_model = "gpt-4o-mini"            # Meta-model for proposals
meta_model_provider = "openai"
few_shot_score_threshold = 0.85        # Bootstrap threshold

# Seed pools
bootstrap_train_seeds = [0, 1, 2, 3, 4]    # Bootstrap phase seeds
online_pool = [5, 6, 7, 8, 9]              # Online evaluation seeds
test_pool = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]  # Final test seeds
```

### Key Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `num_iterations` | Optimization iterations | 10-20 |
| `num_evaluations_per_iteration` | Variants per iteration | 4-8 |
| `batch_size` | Concurrent evaluations | 4-10 |
| `few_shot_score_threshold` | Bootstrap threshold | 0.75-0.90 |
| `bootstrap_train_seeds` | Bootstrap phase seeds | 3-10 seeds |
| `online_pool` | Online evaluation seeds | 5-20 seeds |
| `test_pool` | Final test seeds | 5-50 seeds |

---

## How MIPROv2 Works

### Bootstrap Phase

1. **Evaluate Baseline**: Run initial prompt on `bootstrap_train_seeds`
2. **Collect Examples**: Filter seeds with score >= `few_shot_score_threshold`
3. **Generate Demonstrations**: Format high-scoring examples as few-shot demonstrations
4. **Initialize Meta-Model**: Provide demonstrations to meta-model for context
5. **Warm Up TPE**: Initialize Tree-structured Parzen Estimator with initial evaluations

### Optimization Loop

For each iteration (1 to `num_iterations`):

1. **Meta-Model Proposals**: Meta-model proposes `num_evaluations_per_iteration` prompt variants
2. **TPE Selection**: TPE selects hyperparameters (mutation locations, instruction additions)
3. **Mini-Batch Evaluation**: Evaluate variants on `online_pool` seeds (batch_size concurrent)
4. **Update Meta-Model**: Learn from evaluation results
5. **Update TPE**: Refine hyperparameter distribution

### Final Evaluation

- Evaluate best prompts on `test_pool` (held-out seeds)
- Return optimized prompt with test score

---

## Querying Results

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

# Get prompt text
best_text = get_prompt_text(
    job_id="pl_abc123",
    base_url="http://localhost:8000",
    api_key="sk_...",
    rank=1
)
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

MIPROv2 typically achieves similar accuracy to GEPA with fewer evaluations:

| Phase | Typical Accuracy | Notes |
|-------|------------------|-------|
| Baseline | 60-75% | Initial prompt |
| After Bootstrap | 70-80% | Meta-model initialized with examples |
| After 10 iterations | 80-85% | Mid-optimization |
| After 16 iterations | 85-90%+ | Final optimized prompt |

**Total Evaluations**: ~96 rollouts (16 iterations × 6 variants) vs ~1000 for GEPA

---

## GEPA vs MIPROv2

| Aspect | GEPA | MIPROv2 |
|--------|------|---------|
| **Initialization** | Random population | Bootstrap phase (few-shot examples) |
| **Exploration** | Mutation + Crossover | Meta-model + TPE |
| **Evaluation** | Full (30 seeds) | Mini-batch (5 seeds per iteration) |
| **Learning** | Population evolution | Meta-learning |
| **Cost** | ~1000 rollouts | ~96 rollouts |
| **Convergence** | 3-10 generations | 10-20 iterations |
| **Best For** | Diverse solutions | Fast, efficient optimization |

**When to Use GEPA:**
- Need diverse prompt variants (Pareto front)
- Want to explore many approaches
- Have large evaluation budget

**When to Use MIPROv2:**
- Want faster convergence
- Have clear task structure
- Need efficient optimization
- Want meta-learning benefits

---

## Troubleshooting

### ❌ "MIPRO algorithm is not yet implemented"

**Solution:** MIPROv2 support is currently under development. Use GEPA for now:
```bash
uvx synth-ai train --config examples/blog_posts/gepa/configs/banking77_gepa_local.toml
```

### ❌ "OPENAI_API_KEY environment variable is required"

**Solution:** Export your OpenAI API key for the meta-model:
```bash
export OPENAI_API_KEY="sk-your-key-here"
```

### ❌ "Bootstrap phase found no high-scoring examples"

**Solution:** Lower the `few_shot_score_threshold` in your config:
```toml
[prompt_learning.mipro]
few_shot_score_threshold = 0.75  # Lower from 0.85
```

### ❌ "Banking77 task app is not running"

**Solution:** Start the task app first:
```bash
./examples/blog_posts/mipro/deploy_banking77_task_app.sh
```

---

## Files in This Directory

```
examples/blog_posts/mipro/
├── README.md                         # This file - comprehensive guide
├── configs/                          # Configuration files
│   ├── banking77_mipro_local.toml   # Banking77 MIPRO config (local)
│   └── banking77_mipro_test.toml    # Banking77 MIPRO config (test)
├── deploy_banking77_task_app.sh     # Helper: Start task app
└── run_mipro_banking77.sh            # Helper: Run MIPROv2 optimization
```

---

## Next Steps

1. **Test the bootstrap phase**: Verify few-shot example collection works
2. **Run full optimization**: Complete 16 iterations and check results
3. **Compare with GEPA**: Run GEPA on same task and compare accuracy/cost
4. **Experiment with parameters**: Adjust bootstrap threshold, iteration count
5. **Try other tasks**: Adapt configs for HotpotQA, IFBench, etc.

---

## Support

For issues or questions:

1. Verify all API keys are set correctly (SYNTH_API_KEY, GROQ_API_KEY, OPENAI_API_KEY)
2. Check task app: `curl -H "X-API-Key: $ENVIRONMENT_API_KEY" http://127.0.0.1:8102/health`
3. Check backend: `curl http://localhost:8000/api/health`
4. Review logs in both terminals for error messages

Happy optimizing! 🔬🚀

