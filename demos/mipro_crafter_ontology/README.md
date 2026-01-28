# MIPRO Online Crafter with Ontology

This demo runs MIPRO online optimization on Crafter ReAct with an integrated ontology backend that tracks learned concepts.

## What it does

1. **MIPRO Online Optimization**: Runs the MIPRO algorithm in online mode to evolve prompt candidates in real-time
2. **Crafter ReAct Agent**: Uses a VLM-based ReAct agent to play the Crafter survival game
3. **Ontology Learning**: Tracks and learns:
   - Action effectiveness (which actions yield rewards)
   - Achievement discovery (which achievements have been unlocked)
   - Prompt candidate performance
   - Strategy insights from high-performing runs

## Prerequisites

```bash
pip install crafter>=1.8.3 httpx datasets
```

## Usage

```bash
cd /Users/joshpurtell/Documents/GitHub/synth-ai

# Set your API key
export SYNTH_API_KEY=sk_live_your_key_here

# Run with production backend
RUST_BACKEND_URL=https://api.usesynth.ai uv run python demos/mipro_crafter_ontology/run_online_demo.py \
    --rollouts 100 \
    --train-size 20 \
    --val-size 5 \
    --min-proposal-rollouts 20

# Or with a local backend
RUST_BACKEND_URL=http://localhost:8000 uv run python demos/mipro_crafter_ontology/run_online_demo.py \
    --rollouts 50 \
    --train-size 10 \
    --val-size 3
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--rollouts` | 50 | Number of rollouts to run |
| `--train-size` | 20 | Number of training seeds |
| `--val-size` | 5 | Number of validation seeds |
| `--min-proposal-rollouts` | 20 | Min rollouts before generating new proposals |
| `--model` | gpt-4.1-nano | Model for policy inference |
| `--backend-url` | auto | Backend URL (auto-detected from env vars) |
| `--helix-url` | localhost:6969 | HelixDB URL for ontology storage |
| `--output` | auto | Output file path for results JSON |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `SYNTH_API_KEY` | Required. Your Synth API key |
| `RUST_BACKEND_URL` | Backend URL (or `SYNTH_URL`, `SYNTH_BACKEND_URL`) |
| `HELIX_URL` | HelixDB URL for ontology storage (optional, defaults to localhost) |
| `CRAFTER_POLICY_MODEL` | Override policy model |
| `CRAFTER_PROPOSER_MODEL` | Override proposer model |
| `CRAFTER_MAX_STEPS` | Override max steps per episode |
| `CRAFTER_MAX_TURNS` | Override max turns per rollout |

## Output

The script outputs:

1. **Progress updates** every 5 rollouts
2. **MIPRO Results** summary:
   - Total rollouts and average reward
   - Candidate performance breakdown
   - Best candidate prompt text
3. **Learned Ontology**:
   - Action effectiveness rankings with visual bars
   - Discovered achievements
   - Strategy insights
   - Top prompt candidates

Results are saved to `demos/mipro_crafter_ontology/results/` as JSON.

## Example Output

```
======================================================================
LEARNED ONTOLOGY
======================================================================

--- Actions ---
  move_up              [████████████████░░░░] 80.5% (n=45)
  do                   [██████████████░░░░░░] 72.3% (n=38)
  move_right           [████████████░░░░░░░░] 65.1% (n=42)
  place_table          [██████████░░░░░░░░░░] 55.2% (n=12)
  make_wood_pickaxe    [████████░░░░░░░░░░░░] 45.0% (n=8)
  ...

--- Discovered Achievements ---
  - collect_wood: Crafter achievement: collect_wood
  - place_table: Crafter achievement: place_table
  - collect_stone: Crafter achievement: collect_stone

--- Learned Strategies ---
  [85%] Candidate abc123 achieved 85.0% avg reward over 25 rollouts

--- Top Prompt Candidates ---
  candidate_abc123abc1: reward=0.850, rollouts=25
    You are an agent playing Crafter...

======================================================================
Total nodes: 23
Total properties: 67
Total relationships: 15
======================================================================
```

## Architecture

```
                    ┌─────────────────┐
                    │   Synth API     │
                    │  (MIPRO Backend)│
                    └────────┬────────┘
                             │
    ┌────────────────────────┼────────────────────────┐
    │                        │                        │
    ▼                        ▼                        ▼
┌─────────┐          ┌──────────────┐          ┌──────────┐
│ Proxy   │◀────────▶│ Local Task   │◀────────▶│ Crafter  │
│ (MIPRO) │          │ App (FastAPI)│          │ Env      │
└─────────┘          └──────────────┘          └──────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   HelixDB       │
                    │   (Ontology)    │
                    └─────────────────┘
```
