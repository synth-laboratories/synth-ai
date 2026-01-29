# Continual Learning: EngineBench (Codex + MIPRO)

This demo adapts the Banking77 continual learning workflow to **EngineBench** and the **Codex coding agent**.

## Goal

Measure continual learning performance across a **distribution shift** defined by card difficulty:

- **Split 1 (easy):** lower half of cards by difficulty score  
- **Split 2 (hard):** upper half of cards by difficulty score

The difficulty score is a heuristic computed from tests, attacks, abilities, and stage.

## What Runs

- **Agent:** Codex CLI (default)
- **Optimizer:** MIPRO (online)
- **Task App:** EngineBench local API
- **Shift:** easy → hard cards

## Usage

```bash
cd demos/continual_learning_engine_bench

# Quick run (20 rollouts per split)
uv run python run_mipro_continual.py

# Customize model, agent, and rollouts
uv run python run_mipro_continual.py \
  --rollouts-per-split 30 \
  --model gpt-4o-mini \
  --agent codex \
  --timeout 600 \
  --agents-md ./AGENTS.md \
  --skills-yaml ./.codex/skills.yaml
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--rollouts-per-split` | `20` | Rollouts per difficulty split |
| `--model` | `gpt-4o-mini` | Policy model for the agent |
| `--agent` | `codex` | Agent type (`codex`, `opencode`, `claude_code`) |
| `--timeout` | `600` | Agent timeout in seconds |
| `--agents-md` | (none) | Path to AGENTS.md override |
| `--skills-yaml` | (none) | Path to agent skills.yaml override |
| `--train-size` | `20` | Bootstrap training seeds |
| `--val-size` | `10` | Validation seeds |
| `--output` | auto | Output JSON path |
| `--backend-url` | auto | Backend URL override |
| `--system-id` | (new) | Reuse an existing MIPRO system |
| `--system-name` | (none) | Human-readable system label |

## Notes

- This demo assumes you have the **Codex CLI** installed and in `PATH`.
- LLM calls are routed through the **MIPRO proxy** for optimization.
- Difficulty splits are computed in `data_splits.py` and passed via `env.config.difficulty_split`.
