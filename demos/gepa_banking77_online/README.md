# Online GEPA Demo - Banking77 Classification

This demo shows GEPA (Guided Evolutionary Prompt Adaptation) running in **online mode**, learning in real-time as requests come in.

## How Online GEPA Works

Unlike offline GEPA which optimizes prompts in batches, online GEPA:

1. **Proxy Setup**: Creates a proxy URL for your LLM requests
2. **Candidate Selection**: Each request is routed to a prompt candidate (uniform random selection)
3. **Reward Learning**: After each response, you submit a reward signal
4. **Automatic Proposals**: After enough data accumulates (default: 10 rollouts), the system automatically proposes new prompt candidates using a meta-model
5. **Evolutionary Search**: Uses Pareto-style selection to choose parents for mutation, favoring higher-performing candidates

## Quick Start

```bash
# Install dependencies
pip install synth-ai datasets openai httpx

# Run the demo (uses dev backend by default)
python run_online_demo.py

# Run with more queries
python run_online_demo.py --queries 100

# Use a different model
python run_online_demo.py --model gpt-4.1-mini
```

## Configuration Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--queries` | 50 | Number of test queries to run |
| `--model` | gpt-4.1-nano | LLM model to use |
| `--output-dir` | results | Directory for result files |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SYNTH_API_KEY` | (mints demo) | Your Synth API key |
| `SYNTH_API_URL` | https://api-dev.usesynth.ai | Synth backend URL |
| `INFRA_API_URL` | https://infra-api-dev.usesynth.ai | Infrastructure API URL |

## Expected Output

The demo will:
1. Initialize an online GEPA system
2. Run queries through the proxy
3. Submit rewards after each query
4. Show accuracy progress every 10 queries
5. Display the final system state with all candidates

Example output:
```
============================================================
GEPA SYSTEM STATE
============================================================
System ID: banking77_online_e4f3ba10
Rollout count: 50
Reward count: 50
Proposals triggered: 4
Archived candidates: 0

Active Candidates (3):
  - baseline: accuracy=41.18%, rollouts=17, parent=baseline
  - cand_6f1af0ec89c64b01b708558890: accuracy=57.14%, rollouts=7, parent=baseline
  - cand_e39f8a57434240c5a1803ab418: accuracy=50.00%, rollouts=6, parent=cand_6f1a...
```

## How It Learns

1. **Initial State**: Starts with a baseline prompt candidate
2. **Data Collection**: Routes requests uniformly across candidates
3. **Proposal Trigger**: After 10 rollouts, triggers a proposal
4. **Mutation**: Uses GPT-4.1-mini to generate improved prompts based on performance feedback
5. **Selection**: New candidates compete with existing ones
6. **Pruning**: Underperforming candidates (< 30% accuracy after 20 rollouts) are archived

## Files

- `run_online_demo.py` - Main demo script
- `results/` - Output directory for results JSON files
