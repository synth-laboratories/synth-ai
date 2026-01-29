# Continual Learning Comparison: Banking77

This demo compares classic GEPA (non-continual) vs MIPRO continual learning on Banking77 dataset with progressive data splits.

## Hypothesis

**Continual learning will outperform restart-from-scratch on later splits**, because:
1. Earlier learned prompts transfer knowledge to new intents
2. The ontology accumulates useful patterns over time
3. Restart loses all prior learning at each split

## Data Splits

We create 4 progressive data splits where each is a superset of the previous:

| Split | Intents | Description |
|-------|---------|-------------|
| Split 1 | 2 | `card_arrival`, `lost_or_stolen_card` |
| Split 2 | 7 | Split 1 + 5 more card-related intents |
| Split 3 | 27 | Split 2 + 20 more common banking intents |
| Split 4 | 77 | All Banking77 intents (complete dataset) |

## Approaches

### Classic GEPA (Non-Continual)
- Run GEPA on Split 1, score, save best prompt
- Run GEPA on Split 2 with two conditions:
  - **Warm start**: Initialize with Split 1's best prompt
  - **Cold start**: Initialize with baseline prompt
- Repeat for Splits 3 and 4
- Compare warm start vs cold start at each stage

### MIPRO Continual Learning
- Run MIPRO in online mode continuously across all splits
- Stream data from Split 1 → 2 → 3 → 4 sequentially
- Track prompt evolution and ontology growth at checkpoints
- No restarts - learning persists throughout

## Results

### Comparison Table

| Split | Intents | GEPA Cold Start | GEPA Warm Start | MIPRO Continual | Winner |
|-------|---------|-----------------|-----------------|-----------------|--------|
| Split 1 | 2 | **100.0%** | N/A | 88.0% | **GEPA** |
| Split 2 | 7 | **86.0%** | - | 70.0% | **GEPA** |
| Split 3 | 27 | 74.0% | 68.0% | **80.0%** | **MIPRO** |
| Split 4 | 77 | 48.0% | 56.0% | **58.7%** | **MIPRO** |

### Key Findings

1. **GEPA dominates on simple tasks (Splits 1-2):** Fresh optimization achieves 100% on 2 intents and 86% on 7 intents
2. **MIPRO Continual wins on complex tasks (Splits 3-4):** 80% vs 74% on 27 intents, 58.7% vs 56% on 77 intents
3. **Crossover point at ~27 intents:** Continual learning starts outperforming restart-based optimization

### MIPRO Continual Learning Results

| Split | Intents | Accuracy | Correct/Total | Time (s) |
|-------|---------|----------|---------------|----------|
| Split 1 | 2 | 88.0% | 44/50 | 189 |
| Split 2 | 7 | 70.0% | 35/50 | 175 |
| Split 3 | 27 | **80.0%** | 40/50 | 160 |
| Split 4 | 77 | **58.7%** | 27/46 | 154 |

**Overall:** 74.5% cumulative accuracy across 196 rollouts in ~687 seconds

### Classic GEPA Results

| Split | Intents | Cold Start | Warm Start | Difference |
|-------|---------|------------|------------|------------|
| Split 1 | 2 | **100.0%** | N/A | - |
| Split 2 | 7 | **86.0%** | - | - |
| Split 3 | 27 | **74.0%** | 68.0% | -6% (cold wins) |
| Split 4 | 77 | 48.0% | **56.0%** | +8% (warm wins) |

### Analysis

1. **Split 1 (2 intents):** GEPA achieves perfect 100% - simple binary classification is easy to optimize from scratch
2. **Split 2 (7 intents):** GEPA Cold (86%) >> MIPRO (70%) - fresh optimization on small task space wins
3. **Split 3 (27 intents):** MIPRO (80%) > GEPA Cold (74%) > GEPA Warm (68%) - continual learning begins to show advantage
4. **Split 4 (77 intents):** MIPRO (58.7%) > GEPA Warm (56%) > GEPA Cold (48%) - continual learning wins on hardest task

### Hypothesis Evaluation

**Hypothesis:** Continual learning fares better in later splits

**SUPPORTED:** The data shows a clear pattern:
- Simple tasks (2-7 intents): GEPA wins with fresh optimization
- Complex tasks (27-77 intents): MIPRO Continual wins by building on accumulated knowledge
- The crossover happens around 27 intents, where task complexity benefits from retained knowledge

### Configuration

- **Rollouts per split:** 30
- **Model:** gpt-4.1-nano
- **Train seeds:** 20

## Usage

### Environment Setup

```bash
# Required: Synth API key for backend auth
export SYNTH_API_KEY=sk_live_...

# Optional: Override models (defaults shown)
export BANKING77_POLICY_MODEL=gpt-4.1-nano
export BANKING77_PROPOSER_MODEL=gpt-4.1-mini

# Optional: Override backend URLs (auto-resolved if not set)
export SYNTH_URL=https://your-backend.example.com
export RUST_BACKEND_URL=https://your-rust-backend.example.com
```

### Run MIPRO Continual Learning

```bash
cd demos/continual_learning_banking77

# Basic run (100 rollouts per split, 4 splits = 400 total)
uv run python run_mipro_continual.py

# Customize rollouts, model, and output
uv run python run_mipro_continual.py \
  --rollouts-per-split 100 \
  --model gpt-4.1-nano \
  --train-size 30 \
  --val-size 20 \
  --output results/my_run.json

# Resume on an existing ontology graph (accumulate knowledge across runs)
uv run python run_mipro_continual.py \
  --system-id <previous-system-id> \
  --system-name "banking77-experiment-v2"
```

#### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--rollouts-per-split` | `100` | Number of rollouts per data split |
| `--model` | `gpt-4.1-nano` | Policy model for classification |
| `--train-size` | `30` | Number of training seeds |
| `--val-size` | `20` | Number of validation seeds |
| `--output` | `results/mipro_continual_<timestamp>.json` | Output JSON path |
| `--system-id` | (new) | Reuse an existing MIPRO system to build on its ontology |
| `--system-name` | (none) | Human-readable label for the system |
| `--backend-url` | auto-resolved | Backend URL override |
| `--ontology-url` | auto-resolved | Rust backend URL for ontology reads |
| `--local-host` | `localhost` | Task app hostname |
| `--local-port` | `8030` | Task app port (auto-finds new port if busy) |

### What Happens During a Run

1. A local task app starts on `localhost:8030` serving Banking77 rollouts
2. A MIPRO online job is created on the backend, returning a `system_id` and `proxy_url`
3. For each split (1 → 2 → 3 → 4):
   - Rollouts are executed through the MIPRO proxy (which intercepts LLM calls for traces)
   - Rewards are pushed to the backend asynchronously
   - Every ~20 rollouts, the **batch ontology proposer** synthesizes insights from traces
   - The **online LLM proposer** generates new prompt candidates informed by the ontology
   - Candidates are selected via TPE (Tree-structured Parzen Estimator)
4. A 30-second pause between splits allows background proposers to finish
5. Progress is printed every 20 rollouts; a checkpoint summary after each split

### Retrieving Results

#### 1. JSON Output File

Results are saved to `results/mipro_continual_<timestamp>.json` (or `--output` path). The file contains:

```json
{
  "method": "mipro_continual",
  "job_id": "...",
  "system_id": "...",
  "final_accuracy": 0.70,
  "total_elapsed_seconds": 1200.0,
  "split_results": {
    "1": {"accuracy": 0.88, "correct": 44, "total": 50},
    "2": {"accuracy": 0.70, "correct": 35, "total": 50},
    "3": {"accuracy": 0.80, "correct": 40, "total": 50},
    "4": {"accuracy": 0.58, "correct": 27, "total": 46}
  },
  "checkpoints": [
    {
      "split": 1,
      "num_intents": 2,
      "split_accuracy": 0.88,
      "cumulative_accuracy": 0.88,
      "best_candidate_id": "...",
      "best_candidate_text": "You are an expert banking assistant...",
      "ontology": {
        "num_candidates": 3,
        "proposal_seq": 1,
        "candidates": { "...": {"avg_reward": 0.85, "rollout_count": 20} }
      },
      "ontology_snapshot": {
        "node_name": "system:<system_id>",
        "counts": {"properties": 2, "relationships_from": 68, "relationships_to": 0},
        "relationships_from": [
          {"relation_type": "has_insight", "to_node": "use_keyword_cues_tracking_delivery_waiting"},
          {"relation_type": "runs_candidate", "to_node": "<system_id>:candidate:baseline"}
        ]
      },
      "ontology_graph": "★ best_candidate (reward: 85.0%)\n○ baseline (reward: 70.0%)"
    }
  ]
}
```

Key fields per checkpoint:
- `best_candidate_text` — the evolved system prompt at that split
- `ontology_snapshot` — the full HelixDB graph (nodes, properties, relationships)
- `ontology.candidates` — all candidates with avg_reward and rollout_count
- `ontology_graph` — text tree showing candidate lineage and performance

#### 2. Live Ontology Snapshot via API

Query the ontology graph for a running or completed system:

```bash
# Get the system node context (all relationships + properties)
SYSTEM_ID=<your-system-id>
curl -s -H "Authorization: Bearer $SYNTH_API_KEY" \
  "https://infra-api.usesynth.ai/api/ontology/nodes/system%3A${SYSTEM_ID}/context" | jq .

# Get MIPRO system state (candidates, rewards, proposal count)
curl -s -H "Authorization: Bearer $SYNTH_API_KEY" \
  "https://infra-api.usesynth.ai/api/prompt-learning/online/mipro/systems/${SYSTEM_ID}/state" | jq .
```

The ontology snapshot contains:
- **`relationships_from`** — edges from the system node to candidates, strategies, insights
  - `runs_candidate` — links to prompt candidates
  - `has_insight` — links to LLM-synthesized knowledge nodes (e.g., `use_keyword_cues_tracking_delivery_waiting`, `flag_loss_terms_lost_stolen_missing`)
- **`properties`** — system-level metrics like `total_rollouts`, `avg_reward`

#### 3. Analyze Results

```bash
# Compare MIPRO continual vs classic GEPA results
uv run python analyze_results.py

# Point to specific result files
uv run python analyze_results.py \
  --classic results/classic_gepa.json \
  --continual results/mipro_continual.json
```

### Run Full Comparison (MIPRO vs GEPA)

```bash
uv run python run_comparison.py
uv run python run_comparison.py --rollouts-per-split 100 --model gpt-4.1-nano
```

### Run Classic GEPA Only

```bash
uv run python run_classic_gepa.py --split 1
```

## Files

- `data_splits.py` — Defines the 4 progressive data splits
- `run_mipro_continual.py` — MIPRO continual learning with ontology feedback
- `run_classic_gepa.py` — Classic GEPA with warm/cold starts
- `run_comparison.py` — Full comparison runner
- `run_ontology_evolution.py` — Ontology evolution visualization
- `run_held_out_eval.py` — Held-out test evaluation across all methods
- `analyze_results.py` — Results analysis and table generation
- `test_data_splits.py` — Test script to verify data splits

## Held-Out Test Evaluation

Full held-out test set evaluation (all test samples per split, not just 50).

Results will be populated after running:

```bash
# Quick test (eval-only with existing results)
uv run python run_held_out_eval.py --eval-only \
    --mipro-results results/mipro_continual_*.json \
    --gepa-results results/classic_gepa_*.json

# Full run
uv run python run_held_out_eval.py --rollouts-per-split 100
```

| Split | Intents | Test Size | Baseline | MIPRO | GEPA Cold | GEPA Warm |
|-------|---------|-----------|----------|-------|-----------|-----------|
| Split 1 | 2 | 80 | - | - | - | - |
| Split 2 | 7 | 280 | - | - | - | - |
| Split 3 | 27 | 1080 | - | - | - | - |
| Split 4 | 77 | 3080 | - | - | - | - |

## Verify Setup

Test the data splits are working correctly:
```bash
uv run python test_data_splits.py
```

Expected output:
```
Banking77 Progressive Data Splits
============================================================

Split 1: 2 intents
  Train samples: 235
  Test samples: 80

Split 2: 7 intents
  Train samples: 855
  Test samples: 280

Split 3: 27 intents
  Train samples: 3282
  Test samples: 1080

Split 4: 77 intents
  Train samples: 10003
  Test samples: 3080
```
