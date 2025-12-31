# Crafter VLM GEPA Demo

This demo runs GEPA prompt optimization for a Crafter vision-language agent that uses image-only observations.

## Results Preview

**Achievement Frequencies (30 seeds, same seeds for both)**:

| Achievement | Baseline | Optimized | Delta | Notes |
|-------------|----------|-----------|-------|-------|
| **collect_wood** | 8 | 13 | **+5** | Critical - unlocks crafting progression |
| collect_sapling | 24 | 21 | -3 | One-off, doesn't unlock other achievements |
| collect_drink | 3 | 2 | -1 | |
| place_plant | 5 | 0 | -5 | |
| eat_cow | 1 | 0 | -1 | |
| wake_up | 1 | 0 | -1 | |

The optimized prompt significantly improves `collect_wood` (+62%), the critical first achievement that unlocks the entire crafting progression (wood → table → pickaxe → stone → better tools). Other achievements like `collect_sapling` are one-offs that don't contribute to progression.

**Example Optimized Prompt**:

```
You will receive images from the game Crafter alongside text describing your
current state and available actions. Your task is to analyze these images to
understand the agent's surroundings, inventory, health, and nearby resources.
Use this understanding to decide the next 2 to 5 actions to survive and unlock
achievements.

Key premises:
- The 'do' action only works when adjacent to a resource (tree, stone, cow, plant).
- Crafting progression is: wood → table → wood_pickaxe → stone → stone_pickaxe.
- Available actions include movement, interaction (do), crafting, placing, sleeping, noop.

Heuristics:
- Identify resources adjacent to the agent before using 'do'.
- Move towards resources if none are adjacent.
- Craft tools only when required materials are available.
- Avoid redundant or impossible actions (e.g., 'do' when no resource is adjacent).

Constraints:
- Return 2 to 5 valid actions per decision.
- Use only actions listed.
- Never use 'do' unless adjacent to a valid resource.

Output:
Return your chosen actions as a JSON object with the key "actions_list" listing
2 to 5 actions in order, plus a brief "reasoning" explaining your choices.
```

---

## Quick Start

### Option 1: Run via Script

```bash
cd /Users/joshpurtell/Documents/GitHub/synth-ai
export SYNTH_API_KEY=your_api_key_here

uv run python demos/gepa_crafter_vlm/run_notebook.py
```

This executes `demo_prod.ipynb` using papermill and saves:
- Executed notebook to `demo_prod_executed.ipynb`
- Optimized prompt to `results/optimized_prompt.txt`
- Eval results to `results/eval_results.json`

### Option 2: Run Interactively in Jupyter

```bash
cd /Users/joshpurtell/Documents/GitHub/synth-ai
jupyter notebook demos/gepa_crafter_vlm/demo_prod.ipynb
```

### Option 3: Run Legacy Script Directly

```bash
cd /Users/joshpurtell/Documents/GitHub/synth-ai/demos/gepa_crafter_vlm
python demo_crafter_react.py
```

## What the Demo Does

1. Creates a local task app for the Crafter VLM agent
2. Runs GEPA prompt optimization to find the best system prompt (~10-30 minutes)
3. Extracts the optimized prompt from results
4. Runs eval jobs comparing baseline vs optimized prompts
5. Displays comparison results

## Results

After running, `results/` will contain:

```
results/
├── optimized_prompt.txt       # The optimized system prompt
├── comparison_results.json    # Baseline vs optimized comparison (30 seeds)
└── eval_results.json          # Raw eval results
```

See [Results Preview](#results-preview) above for detailed comparison and example optimized prompt.

### Baseline Prompt (for comparison)

```
You are an agent playing Crafter, a survival crafting game. Your goal is to
survive and unlock achievements. Analyze images to understand surroundings,
inventory, health, resources. Use crafter_interact tool. Key: 'do' only works
adjacent to resources (tree, stone, cow, plant). Craft progression: wood ->
table -> wood_pickaxe -> stone -> stone_pickaxe. Actions: [list]. Return 2-5
actions per decision.
```

## Script Options

```bash
uv run python demos/gepa_crafter_vlm/run_notebook.py --help
```

| Option | Default | Description |
|--------|---------|-------------|
| `--backend-url` | `https://api.usesynth.ai` | Backend URL |
| `--api-key` | `SYNTH_API_KEY` env var | Your Synth API key |
| `--policy-model` | `gpt-4.1-nano` | VLM model for the agent |
| `--verifier-model` | `gpt-4.1-nano` | Model for verification |
| `--rollout-budget` | `30` | Total rollout budget |
| `--num-generations` | `2` | Number of GEPA generations |
| `--use-tunnel` | `False` | Use cloudflared tunnels |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `SYNTH_API_KEY` | Your Synth API key (required) |
| `LOCAL_BACKEND` | Set to `true` to use local backend at `http://127.0.0.1:8000` |

## Prerequisites

- Python deps: `crafter`, `numpy`, `Pillow`, `httpx`, `papermill`
- VLM provider keys for policy + verifier (e.g., `OPENAI_API_KEY`)
- Optional: `cloudflared` if you want tunnels

## Model Configuration

- **Policy Model**: `gpt-4.1-nano` (VLM agent)
- **Verifier Model**: `gpt-4.1-nano` (outcome verification)
- **Verifier**: `zero_shot_verifier_crafter_vlm`
