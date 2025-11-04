# Pokémon VL: Vision-Language RL Pipeline

This playbook demonstrates end-to-end vision-language reinforcement learning on Pokémon Red using Synth AI's CLI tools. We follow the eval → collect data → SFT → RL → eval pipeline, but with vision models throughout.

## Overview

**Model**: Qwen3-VL-4B-Instruct (4B parameter vision-language model via Synth API)
**Environment**: Pokémon Red (Game Boy emulation with vision support)
**Benchmark**: Pallet Town progression task (leave bedroom → get starter → win first battle)

## Pipeline Steps

1. **Deploy Task App** - Host the Pokémon Red environment
2. **Collect Vision Rollouts** - Generate high-quality demonstrations using Qwen3-VL
3. **Filter Dataset** - Extract successful trajectories for supervised fine-tuning
4. **Fine-Tune Qwen3-4B VL** - Train vision-language model on filtered data
5. **Vision-Language RL** - Bootstrap RL training from SFT checkpoint
6. **Final Evaluation** - Compare SFT and RL performance

## Prerequisites

```bash
# Install dependencies
uv pip install -e .

# Setup authentication
uvx synth-ai setup

# Copy environment template
cp examples/blog_posts/pokemon_vl/.env.example .env
```

## Quick Start

```bash
# Export trace database path
export POKEMON_VL_TRACE_DB=traces/v3/pokemon_vl_blog.db

# 1. Deploy task app
uvx synth-ai deploy pokemon_red --runtime modal --name pokemon-vl-blog --env-file .env

# 2. Collect vision rollouts with Qwen3-VL
uvx synth-ai eval pokemon_red --config examples/blog_posts/pokemon_vl/configs/eval_qwen3_vl.toml --trace-db "${POKEMON_VL_TRACE_DB}"

# 3. Filter high-reward trajectories
uvx synth-ai filter --config examples/blog_posts/pokemon_vl/configs/filter_high_reward.toml

# 4. Fine-tune Qwen3-4B VL
uvx synth-ai train --type sft --config examples/blog_posts/pokemon_vl/configs/train_sft_qwen4b_vl.toml --env-file .env --poll

# 5. RL from SFT checkpoint (replace JOB_ID)
uvx synth-ai train --type rl --config examples/blog_posts/pokemon_vl/configs/train_rl_from_sft.toml --env-file .env --poll

# 6. Evaluate final RL model
uvx synth-ai eval pokemon_red --config examples/blog_posts/pokemon_vl/configs/eval_rl_final.toml --trace-db "${POKEMON_VL_TRACE_DB}"
```

## Vision Features

- **Full Game Boy Frames**: Base64-encoded PNG screenshots (160x144 resolution)
- **Vision-Only Mode**: Pure image understanding without text state
- **Vision + Text Mode**: Combined visual and structured state information
- **Efficient Action Batching**: `execute_sequence` tool for 5-10 actions per inference call

## Expected Results

| Stage | Model | Mean Reward | Success Rate | Best Achievement |
|-------|-------|-------------|--------------|------------------|
| Initial | Qwen3-VL (vision) | ~150 | 60% | Win first battle |
| SFT | Qwen3-4B VL | ~200 | 75% | Win first battle + explore |
| RL | Qwen3-4B VL + RL | ~350 | 85% | Complete Pallet Town |

## Files

- `configs/` - All TOML configuration files
- `ft_data/` - Filtered datasets for fine-tuning
- `.env.example` - Environment variables template

## Vision Model Configuration

The vision models receive:
- **Input**: Game Boy screenshot + optional structured state (position, HP, party, etc.)
- **Output**: Sequence of button presses via `execute_sequence` tool
- **Action Space**: UP, DOWN, LEFT, RIGHT, A, B, START, SELECT with frame counts

## Reward Function

Dense rewards for Pallet Town progression:
- Leave bedroom (+20)
- Exit house (+30)
- Find Oak's lab (+40)
- Talk to Oak (+50)
- Get starter Pokémon (+100)
- Enter battle (+75)
- Deal damage (+50 per 10HP)
- Win battle (+150)

Total possible: ~700 points
